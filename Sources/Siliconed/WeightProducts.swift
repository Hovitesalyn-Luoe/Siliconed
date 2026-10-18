import Metal

/// **A product by a published weight, whatever the model**: `C[m, n] = A[m, k] · W`, plus the weight's
/// LoRA stack — the one place where the engine chooses how a weight is read and multiplied. A model
/// says what it computes (the weight's name, the shapes, its buffers); the settings, which the capacity
/// map decides (`CapacityRules`), say how.
///
///     name ─▶ the weight into the reserve, by the shortest path the settings allow
///              ├─ gemm_tensor, a bf16 weight   copied as it is (`Artifact.stage`)       ─▶ TensorGEMM
///              ├─ widen_gpu                    widened on the GPU (`WidenGPU`)           ─▶ MPS (+ AMX)
///              └─ otherwise                    widened on the CPU (`Artifact.materialize`) ─▶ MPS (+ AMX)
///          ─▶ C = A · W
///          ─▶ + the LoRA stack:  C += (A · down) · up          after the product, or
///                                W += down · up, then C = A · W  merged, where it costs fewer FLOPs
///
/// Every path computes in fp32 from the published values (a bf16 widening is a shift, an 8-bit one a
/// single multiplication): they differ in the order of the sums, hence the last bits, never in what is
/// computed. The default path is the one every model ran before this module, bit for bit.
///
/// What sets it apart — what a model writing its own product would get wrong:
///
///   - **the LoRA stack is added after the product, whichever path made it**: so a bf16-weight product
///     takes a LoRA like any other (the guard that kept Krea 2 under a LoRA widened was such a
///     local exception). The merge (`mergesLoRA`, Qwen-Image-2.1's: a rank-256 stack after the GEMM cost
///     2.0 s per 1024² evaluation, ∝ rows) needs an fp32 weight to add into: under `gemm_tensor` it falls
///     back to after the product;
///   - **the reserve is the model's, sized by its memory need** (`MemoryNeed`): this module wraps it, it
///     never allocates — a bf16 weight takes its first half;
///   - **the timings keep their names** (`staging`, `widening`, `gemm wall`, `gemm GPU`, `lora weights`,
///     `lora wall`, `lora GPU`): the models add them to theirs, so a budget reads the same as before.
package final class WeightProducts {
    private let gemm: GEMM
    private let artifact: Artifact
    private let reserve: UnsafeMutablePointer<Float>
    /// The reserve's floats: a bf16 weight may take twice as many values.
    private let capacity: Int
    private let transposed: Bool
    private let lora: (any LoRAStacks)?
    private let mid: UnsafeMutablePointer<Float>?
    private let midFloats: Int
    private let mergesLoRA: Bool

    /// Per phase, accumulated since the last `resetTimings`.
    package private(set) var timings: [String: Double] = [:]
    package func resetTimings() { timings.removeAll() }

    /// - Parameters:
    ///   - reserve: the model's weight reserve, `capacity` floats (the largest weight it multiplies by).
    ///   - transposed: the weights are laid out `[k, n]` by the forge (`Artifact.linearWeightsTransposed`).
    ///   - lora: the stack per module, `nil` without a LoRA; `mid`, its `[rows, rank]` scratch of `midFloats`.
    ///   - mergesLoRA: the stack may be merged into the widened weight where it costs fewer FLOPs.
    package init(gemm: GEMM, artifact: Artifact, reserve: UnsafeMutablePointer<Float>, capacity: Int,
                 transposed: Bool, lora: (any LoRAStacks)? = nil, mid: UnsafeMutablePointer<Float>? = nil,
                 midFloats: Int = 0, mergesLoRA: Bool = false) {
        self.gemm = gemm; self.artifact = artifact; self.reserve = reserve; self.capacity = capacity
        self.transposed = transposed; self.lora = lora; self.mid = mid; self.midFloats = midFloats
        self.mergesLoRA = mergesLoRA
    }

    private func timed<T>(_ phase: String, _ body: () throws -> T) rethrows -> T {
        let started = DispatchTime.now().uptimeNanoseconds
        let result = try body()
        timings[phase, default: 0] += Double(DispatchTime.now().uptimeNanoseconds - started) * 1e-9
        return result
    }

    /// **`C = A · W` (+ the stack of `target`)**, `W` the map's `name`. `a` and `c` are the model's
    /// buffers, wrapped at their reserved size (`GEMM.wrap` memoizes by address). `target`: the module
    /// the LoRA stack is keyed by — `name` without `.weight` unless said.
    package func linear(_ name: String, a: MTLBuffer, c: MTLBuffer, m: Int, k: Int, n: Int,
                        target: String? = nil) throws {
        let w = try gemm.wrap(UnsafeMutableRawPointer(reserve), bytes: capacity * 4, name: "w")
        let staged = try prepare(name, count: k * n, into: w)
        let target = target ?? String(name.dropLast(".weight".count))
        var stack: (down: MTLBuffer, up: MTLBuffer, rank: Int)?
        if let lora, let module = try timed("lora weights", { try lora.module(target, k: k, n: n) }) {
            stack = (try gemm.wrap(UnsafeMutableRawPointer(module.down), bytes: k * module.rank * 4, name: "loraDown"),
                     try gemm.wrap(UnsafeMutableRawPointer(module.up), bytes: module.rank * n * 4, name: "loraUp"),
                     module.rank)
        }
        let merged = stack != nil && mergesLoRA && !staged && LoRA.merges(rows: m, k: k, n: n)
        if merged, let stack {
            timed("lora wall") {
                timings["lora GPU", default: 0] += gemm.accumulatedLinear(a: stack.down, b: stack.up, c: w,
                                                                         m: k, k: stack.rank, n: n)
            }
        }
        timed("gemm wall") {
            timings["gemm GPU", default: 0] += staged
                ? gemm.linear(a: a, bf16: w, c: c, m: m, k: k, n: n, weightIsTransposed: transposed)
                : gemm.linear(a: a, b: w, c: c, m: m, k: k, n: n, weightIsTransposed: transposed)
        }
        if !merged, let stack, let mid {
            let mb = try gemm.wrap(UnsafeMutableRawPointer(mid), bytes: midFloats * 4, name: "loraMid")
            timed("lora wall") {
                timings["lora GPU", default: 0] += gemm.lora(x: a, down: stack.down, mid: mb, up: stack.up, c: c,
                                                             m: m, k: k, r: stack.rank, n: n)
            }
        }
    }

    /// The weight into the reserve; `true` when it stays bf16 (`TensorGEMM`'s product).
    private func prepare(_ name: String, count: Int, into w: MTLBuffer) throws -> Bool {
        func expect(_ got: Int) throws {
            guard got == count else { throw Artifact.Failure.badHeader("\(name): \(got) values, \(count) expected") }
        }
        if gemm.tensor != nil, let got = try timed("staging", {
            try artifact.stage(name, into: UnsafeMutableRawPointer(reserve), capacity: 2 * capacity)
        }) {
            try expect(got)
            return true
        }
        if let widener = gemm.widener, let source = widener.source(name, in: artifact) {
            guard artifact.tensors[name]?.count == count else {
                throw Artifact.Failure.badHeader("\(name): unexpected shape")
            }
            try timed("widening") { _ = try widener.run(source: source, destination: w, count: count) }
            return false
        }
        // An fp32 tensor in the map, a count that is not a multiple of four, or `widen_gpu` off.
        gemm.widener?.countFallback(count)
        try expect(try timed("widening") { try artifact.materialize(name, into: reserve, capacity: capacity) })
        return false
    }
}

/// **The LoRA stack of one module, expanded**: `down [k, r]` and `up [r, n]`, or `nil` when no LoRA of
/// the stack touches it. `CacheLoRA` keeps them from one evaluation to the next, `StreamedLoRA` streams
/// Qwen-Image-2.1's by layer, `ExpandedLoRA` expands them at each call into two fixed slices.
package protocol LoRAStacks: AnyObject {
    func module(_ target: String, k: Int, n: Int) throws
        -> (down: UnsafeMutablePointer<Float>, up: UnsafeMutablePointer<Float>, rank: Int)?
}

extension CacheLoRA: LoRAStacks {}
extension StreamedLoRA: LoRAStacks {}

/// The stack expanded afresh at every call into `down` and `up` (sized for the largest module), as
/// Anima and Krea 2's text fusion do: no cache to charge, the expansion paid at each product.
package final class ExpandedLoRA: LoRAStacks {
    private let lora: LoRA
    private let down, up: UnsafeMutablePointer<Float>
    package init(lora: LoRA, down: UnsafeMutablePointer<Float>, up: UnsafeMutablePointer<Float>) {
        self.lora = lora; self.down = down; self.up = up
    }
    package func module(_ target: String, k: Int, n: Int) throws
        -> (down: UnsafeMutablePointer<Float>, up: UnsafeMutablePointer<Float>, rank: Int)? {
        guard lora.rank(target) > 0 else { return nil }
        return (down, up, try lora.materialize(target, k: k, n: n, down: down, up: up))
    }
}
