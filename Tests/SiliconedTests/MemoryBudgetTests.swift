import Foundation
import XCTest
@testable import Siliconed

/// **B and the plan chosen in it** (`MemoryBudget`, `MemoryPlan`) — without a byte of weights: B is
/// simulated, the needs are Qwen-Image-2.1's (computed without its maps), a map's layers are of
/// Z-Image's order.
final class MemoryBudgetTests: XCTestCase {

    private let gib = 1 << 30

    /// **A generation is not charged its edits' peak**; the need never falls as references are added,
    /// and a count with no tier for the format takes the next one up.
    func testTheNeedFollowsTheReferences() {
        let q = ModelCard.qwenImage21
        // Each reference adds its tokens to the text and the prefix: the edits grow with them. A
        // generation is not an edit (its denoising streams the map's tail, an edit's K/V go to the file).
        let needs = (1...3).map { q.memoryNeed(width: 1024, height: 1024, references: $0) }
        XCTAssertEqual(needs, needs.sorted()); XCTAssertEqual(Set(needs).count, 3)
        // The need is the computed floor: the lean plan's peak, below the default plan's.
        XCTAssertEqual(q.memoryNeed(width: 1024, height: 1024), MemoryNeed.of(q.memoryRequest(width: 1024, height: 1024).lean)!.peak)
        XCTAssertLessThanOrEqual(q.memoryNeed(width: 1024, height: 1024), q.memoryComfort(width: 1024, height: 1024))
        XCTAssertLessThan(q.memoryNeed(width: 512, height: 512), q.memoryNeed(width: 1024, height: 1024))
        XCTAssertLessThan(q.memoryNeed(width: 1024, height: 1024), q.memoryNeed(width: 1024, height: 1536))
        // A family measured without references keeps its one figure (the marked fallback).
        XCTAssertEqual(ModelCard.klein4b.memoryNeed(width: 1024, height: 1024, references: 2),
                       ModelCard.klein4b.memoryNeed(width: 1024, height: 1024))
    }

    /// `SILICONED_MEMORY_BUDGET_GB` replaces B (GiB, comma or point); without it, B is this Mac's RAM's.
    func testTheForcingReplacesB() {
        XCTAssertEqual(MemoryBudget.machineCeiling(environment: [MemoryBudget.forcingVariable: "4"]), 4 << 30)
        XCTAssertEqual(MemoryBudget.machineCeiling(environment: [MemoryBudget.forcingVariable: "4,5"]), 9 << 29)
        XCTAssertEqual(MemoryBudget.machineCeiling(environment: [MemoryBudget.forcingVariable: "22"]), 22 << 30)
        let physical = Int(ProcessInfo.processInfo.physicalMemory)
        XCTAssertEqual(MemoryBudget.machineCeiling(environment: [:]), MemoryBudget.ceiling(physical: physical))
    }

    // ── B ──────────────────────────────────────────────────────────────────────────────────

    /// **B = ¾ · RAM − 2 GiB**: 8 → 4, 16 → 10, 24 → 16, 32 → 22, 64 → 46; never negative below.
    func testBIsThreeQuartersOfTheRamLessTwo() {
        XCTAssertEqual([8, 16, 24, 32, 64].map { MemoryBudget.ceiling(physical: $0 * gib) / gib }, [4, 10, 16, 22, 46])
        XCTAssertEqual(MemoryBudget.ceiling(physical: 6 * gib), 5 * gib / 2)
        XCTAssertEqual(MemoryBudget.ceiling(physical: 2 * gib), 0)
    }

    /// **The inverse**: the smallest RAM, in whole GiB, whose B covers a need — the app's « needs a
    /// Mac with N GB ».
    func testTheSmallestMacInvertsB() {
        XCTAssertEqual(MemoryBudget.physicalNeeded(for: 4 * gib), 8)
        XCTAssertEqual(MemoryBudget.physicalNeeded(for: 4 * gib + 1), 9)
        XCTAssertEqual(MemoryBudget.physicalNeeded(for: 10 * gib), 16)
        XCTAssertEqual(MemoryBudget.physicalNeeded(for: 22 * gib), 32)
        for need in stride(from: 0, to: 40 * gib, by: 97_654_321) {
            let n = MemoryBudget.physicalNeeded(for: need)
            XCTAssertGreaterThanOrEqual(MemoryBudget.ceiling(physical: n * gib), need)
            XCTAssertTrue(n == 1 || MemoryBudget.ceiling(physical: (n - 1) * gib) < need, "\(need)")
        }
    }

    /// **The Mac to buy**: the inverse rounded up to a size sold — 9 GiB raw is a 16 GB Mac, 17 an 18.
    func testTheMacNeededIsASizeSold() {
        XCTAssertEqual(MemoryBudget.macNeeded(for: 4 * gib), 8)
        XCTAssertEqual(MemoryBudget.macNeeded(for: 4 * gib + 1), 16)
        XCTAssertEqual(MemoryBudget.macNeeded(for: MemoryBudget.ceiling(physical: 17 * gib)), 18)
        XCTAssertEqual(MemoryBudget.macNeeded(for: 1000 * gib), MemoryBudget.physicalNeeded(for: 1000 * gib))
        for need in stride(from: 0, to: 300 * gib, by: 977_654_321) {
            let n = MemoryBudget.macNeeded(for: need)
            XCTAssertGreaterThanOrEqual(MemoryBudget.ceiling(physical: n * gib), need)
            if n <= 512 { XCTAssertTrue(MemoryBudget.soldSizes.contains(n), "\(n)") }
        }
    }

    // ── the preflight ───────────────────────────────────────────────────────────────────────

    /// **The floor is judged against B, and against B alone** — the one check the app also asks.
    func testTheFloorIsRefusedAboveB() {
        // Refused one byte short, with both figures.
        let q = ModelCard.qwenImage21
        let floor = q.memoryNeed(width: 1024, height: 1024)
        XCTAssertThrowsError(try q.checkMemory(width: 1024, height: 1024, ceiling: floor - 1)) {
            XCTAssertEqual($0 as? EngineError, .insufficientMemory(needed: floor, budget: floor - 1))
        }
        XCTAssertNoThrow(try q.checkMemory(width: 1024, height: 1024, ceiling: floor))
        XCTAssertNoThrow(try q.checkMemory(width: 512, height: 512, ceiling: floor - 1))
        // A 16 GB Mac takes Qwen's worst case, 1024×1536 with 3 references.
        XCTAssertNoThrow(try q.checkMemory(width: 1024, height: 1536, references: 3,
                                           ceiling: MemoryBudget.ceiling(physical: 16 * gib)))
        // The engine's request: a LoRA and a long prompt raise the need it is judged on.
        let plain = q.memoryRequest(width: 1024, height: 1024, promptTokens: 50)
        var loaded = plain; loaded.loraRanks = [128]
        XCTAssertGreaterThan(q.memoryNeed(loaded), q.memoryNeed(plain))
    }

    // ── the plan, chosen in B ───────────────────────────────────────────────────────────────

    /// **The fastest plan B holds, and nothing above it.** A roomy B takes every buffer; a B between
    /// the floor and the fastest plan keeps what fits; at the floor, the lean plan. The plan's need is
    /// never above B once the floor is.
    func testThePlanIsTheFastestBHolds() {
        let q = ModelCard.qwenImage21
        let request = q.memoryRequest(width: 1024, height: 1024, references: [(1024, 1024)] * 3)
        let floor = MemoryNeed.of(request.lean)!.peak
        let fastest = MemoryNeed.of(request)!.peak
        XCTAssertLessThan(floor, fastest)
        let roomy = MemoryPlan.choose(request, ceiling: 16 << 30)
        XCTAssertEqual(roomy.request.encoderSlots, MemoryPlan.encoderSlots)
        XCTAssertFalse(roomy.request.minimumBands)
        XCTAssertEqual(roomy.need?.peak, fastest)
        XCTAssertEqual(roomy.floor, floor)
        XCTAssertFalse(roomy.lean)
        // At the floor, what raises the peak goes (an edit's text phase holds it: the encoder's
        // buffers); what does not raise it stays — a band in a phase below the peak is free.
        let tight = MemoryPlan.choose(request, ceiling: floor)
        XCTAssertEqual(tight.request.encoderSlots, 0)
        XCTAssertEqual(tight.need?.peak, floor)
        for ceiling in stride(from: floor, through: fastest, by: max(1, (fastest - floor) / 17)) {
            let plan = MemoryPlan.choose(request, ceiling: ceiling)
            XCTAssertLessThanOrEqual(plan.need!.peak, ceiling)
            XCTAssertGreaterThanOrEqual(plan.need!.peak, floor)
        }
        // The map's tail is the lean one in every plan: its buffers are charged, its head is not.
        XCTAssertEqual(roomy.request.tail, .lean)
        XCTAssertLessThanOrEqual(roomy.denoisingCached, roomy.denoisingStreamed)
    }

    /// **The map's head: what B leaves beside the denoising, capped at the model's split.** Thirty
    /// layers of 400 MB and 500 MB of other tensors (a 12.5 GB map, Z-Image's order).
    func testTheHeadIsWhatBLeavesCappedAtTheSplit() {
        let spans = [Int](repeating: 400_000_000, count: 30), map = 12_500_000_000
        func plan(_ ceiling: Int, _ fraction: Double = 0.5) -> (released: Int, slots: Int) {
            MemoryPlan.released(spans: spans, mapBytes: map, ceiling: ceiling, streamed: 3_000_000_000,
                                cached: 2_300_000_000, fraction: fraction)
        }
        // A 16 GB Mac's B: room for more than half, capped at half.
        XCTAssertEqual(plan(MemoryBudget.ceiling(physical: 16 << 30)).released, 15)
        // Less room, fewer layers: 1.5 GB left beside the denoising and the other tensors hold 3 layers.
        XCTAssertEqual(plan(3_000_000_000 + 500_000_000 + 1_500_000_000).released, 3)
        // No room: the whole map streamed, its two buffers.
        XCTAssertEqual(plan(3_000_000_000).released, 0)
        XCTAssertEqual(plan(3_000_000_000).slots, 2)
        // A model whose split is the whole map (Qwen-Image-2.1's generation): no head.
        XCTAssertEqual(plan(MemoryBudget.ceiling(physical: 16 << 30), 1).released, 0)
        // **The whole map fits beside the denoising's need without buffers: nothing streamed**.
        XCTAssertTrue(plan(map + 2_300_000_000) == (30, 0))
        XCTAssertTrue(plan(map + 2_300_000_000, 1) == (30, 0))
        XCTAssertEqual(plan(map + 2_300_000_000 - 1).slots, 2)
        // A `map_tail` of 0: everything through the cache, as before the tail.
        XCTAssertTrue(plan(3_000_000_000, 0) == (30, 0))
        XCTAssertEqual(MemoryPlan.Tail.lean, MemoryPlan.Tail(fraction: 1, slots: 2))
        XCTAssertEqual(Prefetcher.firstReleasedLayer(of: 30, tail: MemoryPlan.Tail.lean.fraction), 0)
    }

    /// **Outside a render, the reference values**; inside, the plan's — the same from the first site
    /// to the last — and the render's state is gone once it ends.
    func testTheRenderStateIsScopedToTheRender() {
        let q = ModelCard.qwenImage21
        let request = q.memoryRequest(width: 1024, height: 1024)
        XCTAssertNil(MemoryPlan.current)
        XCTAssertEqual(MemoryPlan.decoderBand(standard: 1 << 18, minimum: MemoryPlan.qwenMinimumBand, unit: "pixels"), 1 << 18)
        var tight = MemoryPlan.choose(request, ceiling: 16 << 30)
        tight.request.minimumBands = true
        MemoryPlan.during(tight) {
            XCTAssertEqual(MemoryPlan.decoderBand(standard: 1 << 18, minimum: MemoryPlan.qwenMinimumBand, unit: "pixels"),
                           MemoryPlan.qwenMinimumBand)
            // A check that forced a smaller band keeps it.
            XCTAssertEqual(MemoryPlan.decoderBand(standard: 1 << 12, minimum: MemoryPlan.qwenMinimumBand, unit: "pixels"), 1 << 12)
        }
        XCTAssertNil(MemoryPlan.current)
        XCTAssertEqual(MemoryPlan.lastRender?.decisions.count, 3)
        MemoryPlan.during(MemoryPlan.choose(request, ceiling: 16 << 30)) {
            XCTAssertEqual(MemoryPlan.decoderBand(standard: 1 << 18, minimum: MemoryPlan.qwenMinimumBand, unit: "pixels"), 1 << 18)
        }
    }

    /// **The need is the economical plan's floor, computed**, below the fastest plan's. A rotated format needs what its transpose needs (the same shapes).
    func testTheNeedIsTheEconomicalFloor() {
        let q = ModelCard.qwenImage21
        for (w, h) in [(1024, 1024), (832, 1216), (1024, 1536)] {
            XCTAssertLessThanOrEqual(q.memoryNeed(width: w, height: h), q.memoryComfort(width: w, height: h), "\(w)×\(h)")
            // The lean plan drops the encoder's buffers: an edit, whose text phase holds its peak, needs less.
            XCTAssertLessThan(q.memoryNeed(width: w, height: h, references: 3), q.memoryComfort(width: w, height: h, references: 3))
        }
        XCTAssertEqual(q.memoryNeed(width: 832, height: 1216), q.memoryNeed(width: 1216, height: 832))
        // The hidden families' marked fallback: their measured peaks, the larger format's above a smaller one.
        XCTAssertEqual(ModelCard.anima.memoryNeed(width: 1024, height: 1024), 5_300_000_000)
        XCTAssertEqual(ModelCard.klein4b.memoryNeed(width: 1024, height: 1536), 7_800_000_000)
        XCTAssertTrue(MemoryNeed.of(ModelCard.ernie.memoryRequest(width: 1024, height: 1024))!.isFallback)
    }
}

private func * <T>(array: [T], count: Int) -> [T] { Array((0..<count).map { _ in array }.joined()) }
