import XCTest
@testable import Siliconed

/// **What is still outside our hand** (`OperatorFootprint`): every term derived from a
/// shape or a named bound — no law fitted on this machine. The measurements stay as what the
/// derivation must stay under, not as a source.
final class OperatorFootprintTests: XCTestCase {

    /// **The SDPA's copy of the queries is the shape's**: `rows·heads·headDim·4`, nothing fitted. The measurement
    /// measured each cached SDPA at ~7 MB (its compiled executable, not charged: open) plus 4.03 B per
    /// element — the copy is within 1 % of every measured slope, and the matrix `H·S·K` is not in it.
    func testTheQueriesCopyIsTheShape() {
        for (heads, rows, headDim) in [(16, 4096, 72), (32, 3400, 128), (32, 2048, 128), (30, 4160, 128)] {
            XCTAssertEqual(OperatorFootprint.sdpaQueries(rows: rows, heads: heads, headDim: headDim), rows * heads * headDim * 4)
        }
        // The measured slice of 32 heads × 2048 rows × 12 588 keys added 41 MB: the copy (33.6 MB) and the
        // ~7.4 MB constant — not the matrix, which would be 3.3 GB.
        let copy = OperatorFootprint.sdpaQueries(rows: 2048, heads: 32, headDim: 128)
        XCTAssertEqual(Double(copy + OperatorFootprint.sdpaExecutable) / 1e6, 41, accuracy: 0.5)
    }

    /// **The causal mask is ours now** (in `Pages`, one fp32 copy), not MPSGraph's three.
    func testTheCausalMaskIsOurs() {
        let terms = MemoryNeed.attention("t", heads: 32, rows: 3400, keys: 3400, headDim: 128, causal: true,
                                         split: false, threads: 6)
        XCTAssertEqual(terms.first { $0.name == "t causal mask" }, .init("t causal mask", Arena.capacity(for: 3400 * 3400 * 4)))
        XCTAssertEqual(terms.filter(\.outside).map(\.bytes), [3400 * 32 * 128 * 4, OperatorFootprint.sdpaExecutable])
    }

    /// **The head split is its host arena and the GPU heads' copy** — `HeadSplitAttention.hostBytes`
    /// is ours, the copy is MPSGraph's for the GPU heads only.
    func testTheHeadSplitIsItsArena() {
        let gpu = HeadSplitAttention.partition(heads: 32, fraction: HeadSplitAttention.cpuFraction).gpu.count
        let terms = MemoryNeed.attention("s", heads: 32, rows: 1229, keys: 18_732, headDim: 128, causal: false,
                                         split: true, threads: 6)
        XCTAssertEqual(terms, [.init("s head split", HeadSplitAttention.hostBytes(heads: 32, sequence: 1229, keys: 18_732,
                                                                                headDim: 128, threads: 6)),
                               .init("s SDPA queries", 1229 * gpu * 128 * 4, outside: true),
                               .init("s SDPA executable", OperatorFootprint.sdpaExecutable, outside: true)])
        // Its arena grows with the keys and the threads (per thread: scores and packed K_h, V_h).
        let few = HeadSplitAttention.hostBytes(heads: 32, sequence: 1229, keys: 8_000, headDim: 128, threads: 6)
        let many = HeadSplitAttention.hostBytes(heads: 32, sequence: 1229, keys: 18_732, headDim: 128, threads: 6)
        XCTAssertLessThan(few, many)
        XCTAssertLessThan(many, HeadSplitAttention.hostBytes(heads: 32, sequence: 1229, keys: 18_732, headDim: 128, threads: 8))
    }

    /// **One GEMM per phase**: MPS's bounded state, and the conductor's workspace from BNNS's own answer
    /// (`Conductor.workspaceBytes`) only when the cut applies.
    func testTheGEMMIsItsStateAndItsWorkspace() {
        let off = MemoryNeed.gemm("g", products: [(4096, 3840, 10240)], amx: false)
        XCTAssertEqual(off, [.init("g MPS GEMM", OperatorFootprint.mpsGEMM, outside: true)])
        let small = MemoryNeed.gemm("g", products: [(Conductor.minimumRows - 1, 3840, 10240)], amx: true)
        XCTAssertEqual(small, off)
        let on = MemoryNeed.gemm("g", products: [(4096, 3840, 10240)], amx: true)
        let slice = 4096 - Conductor.cut(4096, Conductor.fractionRange.lowerBound)
        let expected = [false, true].map { Conductor.workspaceBytes(rows: slice, k: 3840, n: 10240, weightIsTransposed: $0) }.max()!
        XCTAssertGreaterThanOrEqual(on.first { $0.name == "g conductor workspace" }?.bytes ?? 0, Arena.capacity(for: expected))
    }

    /// The process's base is the measured bound; a VAE's `outside` is its SDPA's copies, computed from the
    /// shapes by its plan, charged as its own term, beside its SDPA's compiled executable.
    func testTheBoundsAreNamed() {
        XCTAssertEqual(OperatorFootprint.processBase(app: false), 13_100_000)
        XCTAssertEqual(OperatorFootprint.mpsGEMM, 3_900_000)
        let plan = VAE.plan(latentHeight: 64, latentWidth: 64)
        XCTAssertGreaterThan(plan.outside, 0)
        XCTAssertEqual(MemoryNeed.vae("v", plan).filter(\.outside).map(\.bytes), [plan.outside, OperatorFootprint.sdpaExecutable])
    }
}
