import XCTest
@testable import Siliconed

/// **What is still outside our hand** (`OperatorFootprint`): every term derived from a shape — no law
/// fitted on this machine, and no framework constant folded into the formula. The measurements stay as
/// what the derivation must stay under, not as a source.
final class OperatorFootprintTests: XCTestCase {

    /// **The SDPA's copy of the queries is the shape's**: `rows·heads·headDim·4`, nothing fitted. The
    /// measured slice of 32 heads × 2048 rows × 12 588 keys added 41 MB: the copy (33.6 MB) and MPSGraph's
    /// ~7.4 MB compiled executable — a constant no shape gives, **not charged** (the check reports it) —,
    /// not the matrix, which would be 3.3 GB.
    func testTheQueriesCopyIsTheShape() {
        for (heads, rows, headDim) in [(16, 4096, 72), (32, 3400, 128), (32, 2048, 128), (30, 4160, 128)] {
            XCTAssertEqual(OperatorFootprint.sdpaQueries(rows: rows, heads: heads, headDim: headDim), rows * heads * headDim * 4)
        }
        let copy = OperatorFootprint.sdpaQueries(rows: 2048, heads: 32, headDim: 128)
        XCTAssertEqual(Double(copy) / 1e6, 33.6, accuracy: 0.1)
    }

    /// **The causal mask is ours now** (in `Pages`, one fp32 copy), not MPSGraph's three.
    func testTheCausalMaskIsOurs() {
        let terms = MemoryNeed.attention("t", heads: 32, rows: 3400, keys: 3400, headDim: 128, causal: true,
                                         split: false, threads: 6)
        XCTAssertEqual(terms.first { $0.name == "t causal mask" }, .init("t causal mask", Arena.capacity(for: 3400 * 3400 * 4)))
        XCTAssertEqual(terms.filter(\.outside).map(\.bytes), [3400 * 32 * 128 * 4])
    }

    /// **The head split is its host arena and the GPU heads' copy** — `HeadSplitAttention.hostBytes`
    /// is ours, the copy is MPSGraph's for the GPU heads only.
    func testTheHeadSplitIsItsArena() {
        let gpu = HeadSplitAttention.partition(heads: 32, fraction: HeadSplitAttention.cpuFraction).gpu.count
        let terms = MemoryNeed.attention("s", heads: 32, rows: 1229, keys: 18_732, headDim: 128, causal: false,
                                         split: true, threads: 6)
        XCTAssertEqual(terms, [.init("s head split", HeadSplitAttention.hostBytes(heads: 32, sequence: 1229, keys: 18_732,
                                                                                headDim: 128, threads: 6)),
                               .init("s SDPA queries", 1229 * gpu * 128 * 4, outside: true)])
        // Its arena grows with the keys and the threads (per thread: scores and packed K_h, V_h).
        let few = HeadSplitAttention.hostBytes(heads: 32, sequence: 1229, keys: 8_000, headDim: 128, threads: 6)
        let many = HeadSplitAttention.hostBytes(heads: 32, sequence: 1229, keys: 18_732, headDim: 128, threads: 6)
        XCTAssertLessThan(few, many)
        XCTAssertLessThan(many, HeadSplitAttention.hostBytes(heads: 32, sequence: 1229, keys: 18_732, headDim: 128, threads: 8))
    }

    /// **One GEMM per phase**: its padding tiers (`GEMM.tierBytes`) when a height goes through them,
    /// and the conductor's workspace from BNNS's own answer (`Conductor.workspaceBytes`) only when the
    /// cut applies. MPS's own state is a constant no shape gives: not charged.
    func testTheGEMMIsItsTiersAndItsWorkspace() {
        let off = MemoryNeed.gemm("g", products: [(4096, 3840, 10240)], amx: false)
        XCTAssertEqual(off, [])   // a multiple of 64: nothing through the tiers, no state charged
        let small = MemoryNeed.gemm("g", products: [(Conductor.minimumRows - 1, 3840, 10240)], amx: true)
        XCTAssertEqual(small, [.init("g GEMM tiers", GEMM.tierBytes(products: [(Conductor.minimumRows - 1, 3840, 10240)]))])
        let on = MemoryNeed.gemm("g", products: [(4096, 3840, 10240)], amx: true)
        let slice = 4096 - Conductor.cut(4096, Conductor.fractionRange.lowerBound)
        let expected = [false, true].map { Conductor.workspaceBytes(rows: slice, k: 3840, n: 10240, weightIsTransposed: $0) }.max()!
        XCTAssertGreaterThanOrEqual(on.first { $0.name == "g conductor workspace" }?.bytes ?? 0, Arena.capacity(for: expected))
    }

    /// **The padding tiers are what `GEMM` sends through them**: 8 to 32 rows, or a tail under 64
    /// rows past a body; `a` and `c` at 64 rows of the widest `k` and `n` among those, `mid` at the rank.
    func testTheTiersAreTheRowsSentThroughThem() {
        XCTAssertEqual([1, 7, 8, 32, 33, 63, 64, 128, 4096].map(GEMM.goesThroughTiers),
                       [false, false, true, true, false, false, false, false, false])
        XCTAssertEqual([65, 100, 4097, 4160 - 1].map(GEMM.goesThroughTiers), [true, true, true, true])
        XCTAssertEqual(GEMM.tierBytes(products: [(4096, 3840, 10240), (64, 2560, 3840)]), 0)
        // A 100-token prompt through Z-Image's encoder: k up to 9728, n up to 9728.
        let encoder = [(100, 2560, 4096), (100, 2560, 1024), (100, 4096, 2560), (100, 2560, 9728), (100, 9728, 2560)]
        XCTAssertEqual(GEMM.tierBytes(products: encoder),
                       Arena.capacity(for: 64 * 9728 * 4) + Arena.capacity(for: 64 * 9728 * 4))
        XCTAssertEqual(GEMM.tierBytes(products: encoder, rank: 32) - GEMM.tierBytes(products: encoder),
                       Arena.capacity(for: 64 * 32 * 4))
    }

    /// The process's base is read from the process, not charged as a figure; a VAE's `outside` is its
    /// SDPA's copies, computed from the shapes by its plan, charged as its own term — and nothing else.
    func testTheBaseIsReadAndTheVAEsCopiesComputed() {
        XCTAssertGreaterThan(OperatorFootprint.processBase(), 0)
        let plan = VAE.plan(latentHeight: 64, latentWidth: 64)
        XCTAssertGreaterThan(plan.outside, 0)
        XCTAssertEqual(MemoryNeed.vae("v", plan).filter(\.outside).map(\.bytes), [plan.outside])
    }
}
