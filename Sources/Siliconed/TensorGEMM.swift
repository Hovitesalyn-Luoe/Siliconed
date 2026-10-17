import Foundation
import Metal

/// **The product that reads the published weights as they are**: Metal 4's `matmul2d`, fp32
/// activations × bf16 weights → fp32 — a setting (`gemm_tensor`), **off by default**, taken where the
/// capacity map measures it faster than the path it replaces (`CapacityRules`).
///
///     path            the weight before the product           the product
///     default         bf16 → fp32 into the reserve (Widen)    MPS fp32, the AMX on its share of rows
///     gemm_tensor     bf16 copied into the reserve's head     matmul2d  float × bfloat → float
///
/// **Nothing is rounded that the default path does not round**: the weights *are* bf16 values (P19,
/// the widening is a shift), the activations stay fp32 — the header's `float × bfloat → float`
/// combination, `relaxed_precision` false — and the sums are fp32. What changes is the order of the
/// sums, hence the last bits, as the AMX's share does. Measured on the M1 Pro before the code: the
/// same worst error as `float × float` (2.3·10⁻⁷ at 1024×3840×10240), against the fp64 of the
/// unrounded activations.
///
/// What sets it apart — what a port of the MPS path by analogy would get wrong:
///
///   - **the speed is the machine's, never assumed**: on an M5 `matmul2d` runs on the GPU's neural
///     accelerators (6.0 TFLOP/s bf16 × bf16 against 2.9 for MPS fp32, M225); on the M1 Pro, on the
///     shader cores, slower than MPS (2.36 against 3.46). Hence a setting the map decides, not a path
///     chosen by chip;
///   - **the weight is copied, not wrapped**: the reserve keeps its fp32 size, the bf16 weight takes
///     its first half (`Artifact.stage`). `MemoryNeed` does not move, a streamed tensor goes the same
///     way, and the map's pages are wired by no one. The copy reads what the widening read and writes
///     half — the map measures its rate beside the widening's;
///   - **only the `slice` API**: the kernel is compiled at run time by the user's macOS, not ours.
///     `static_slice`, `get_mask` and the header's other examples do not compile on every Metal 4
///     compiler; `slice` checks the edges, so any `m` and `n` are taken whole — no padding tier
///     (MPS's 8-to-32-row fault, `GEMM.poisonedRows`, is MPS's);
///   - **no co-execution**: the AMX reads fp32 weights, which this path no longer makes. The rules
///     compare this path to the better of MPS alone and MPS + AMX, and turn the AMX off when it wins.
package final class TensorGEMM {
    private let queue: MTLCommandQueue
    /// `[k, n]` weights (the forge's layout), then `[n, k]` (a published `nn.Linear`).
    private let byColumns, byRows: MTLComputePipelineState

    static let source = """
    #include <metal_stdlib>
    #include <metal_tensor>
    #include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
    using namespace metal;
    using namespace mpp::tensor_ops;

    // One 64 × 32 tile of C, written once from a cooperative tensor (`run` into a device tensor
    // would accumulate into what C held).
    template <typename Op, typename SA, typename SB, typename SC>
    inline void tile(Op op, SA a, SB b, SC c) {
        auto t = op.template get_destination_cooperative_tensor<SA, SB, float>();
        for (uint16_t i = 0; i < t.get_capacity(); ++i) { if (t.is_valid_element(i)) t[i] = 0; }
        op.run(a, b, t);
        t.store(c);
    }

    // Extents innermost first: A [M, K] row-major is (K, M).
    kernel void linear_kn(device float *a [[buffer(0)]], device bfloat *w [[buffer(1)]], device float *c [[buffer(2)]],
                          constant uint3 &mkn [[buffer(3)]], uint2 g [[threadgroup_position_in_grid]]) {
        const int M = mkn.x, K = mkn.y, N = mkn.z;
        tensor<device float, dextents<int32_t, 2>, tensor_inline> A(a, dextents<int32_t, 2>(K, M));
        tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> B(w, dextents<int32_t, 2>(N, K));
        tensor<device float, dextents<int32_t, 2>, tensor_inline> C(c, dextents<int32_t, 2>(N, M));
        constexpr auto d = matmul2d_descriptor(64, 32, static_cast<int>(dynamic_extent), false, false, false);
        matmul2d<d, execution_simdgroups<4>> op;
        const int x = g.x * 32, y = g.y * 64;
        tile(op, A.slice(0, y), B.slice(x, 0), C.slice(x, y));
    }

    kernel void linear_nk(device float *a [[buffer(0)]], device bfloat *w [[buffer(1)]], device float *c [[buffer(2)]],
                          constant uint3 &mkn [[buffer(3)]], uint2 g [[threadgroup_position_in_grid]]) {
        const int M = mkn.x, K = mkn.y, N = mkn.z;
        tensor<device float, dextents<int32_t, 2>, tensor_inline> A(a, dextents<int32_t, 2>(K, M));
        tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> B(w, dextents<int32_t, 2>(K, N));
        tensor<device float, dextents<int32_t, 2>, tensor_inline> C(c, dextents<int32_t, 2>(N, M));
        constexpr auto d = matmul2d_descriptor(64, 32, static_cast<int>(dynamic_extent), false, true, false);
        matmul2d<d, execution_simdgroups<4>> op;
        const int x = g.x * 32, y = g.y * 64;
        tile(op, A.slice(0, y), B.slice(0, x), C.slice(x, y));
    }
    """

    /// `nil` before macOS 26, on a device without Metal 4, or when this macOS's compiler refuses the
    /// kernel: the caller keeps the default path.
    package init?(device: MTLDevice, queue: MTLCommandQueue) {
        guard #available(macOS 26.0, *), device.supportsFamily(.metal4) else { return nil }
        let options = MTLCompileOptions()
        options.languageVersion = .version4_0
        options.mathMode = .safe
        guard let library = try? device.makeLibrary(source: TensorGEMM.source, options: options),
              let kn = library.makeFunction(name: "linear_kn"), let nk = library.makeFunction(name: "linear_nk"),
              let byColumns = try? device.makeComputePipelineState(function: kn),
              let byRows = try? device.makeComputePipelineState(function: nk) else { return nil }
        self.queue = queue; self.byColumns = byColumns; self.byRows = byRows
    }

    /// `C [m, n] = A [m, k] · W`, `A` and `C` fp32, `W` bf16 — `[k, n]` when `weightIsTransposed`
    /// (the forge's layout), `[n, k]` otherwise; all row-major from offset 0. Returns the GPU seconds.
    package func linear(a: MTLBuffer, w: MTLBuffer, c: MTLBuffer, m: Int, k: Int, n: Int,
                        weightIsTransposed: Bool) -> Double {
        let pipeline = weightIsTransposed ? byColumns : byRows
        let commands = queue.makeCommandBuffer()!
        let encoder = commands.makeComputeCommandEncoder()!
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(a, offset: 0, index: 0)
        encoder.setBuffer(w, offset: 0, index: 1)
        encoder.setBuffer(c, offset: 0, index: 2)
        var mkn = SIMD3<UInt32>(UInt32(m), UInt32(k), UInt32(n))
        encoder.setBytes(&mkn, length: MemoryLayout<SIMD3<UInt32>>.stride, index: 3)
        encoder.dispatchThreadgroups(MTLSize(width: (n + 31) / 32, height: (m + 63) / 64, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: pipeline.threadExecutionWidth * 4,
                                                                    height: 1, depth: 1))
        encoder.endEncoding()
        commands.commit()
        commands.waitUntilCompleted()
        return Attention.verdict(commands, what: "matmul2d float × bfloat \(m)×\(k)×\(n)")
    }
}
