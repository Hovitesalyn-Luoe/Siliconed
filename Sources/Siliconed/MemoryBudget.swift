import Foundation
import Synchronization

/// **What this Mac offers a render, B — fixed by its RAM, and nothing else.**
///
///     physical RAM ── ¾ · RAM − 2 GiB ── B (`ceiling`) ─┬─ Preflight: the lean plan's need ≤ B, or refused
///                                                       ├─ the app: greyed out ahead, « needs a Mac with N GB »
///                                                       └─ MemoryPlan: the fastest plan whose need ≤ B
///
/// **B is a contract the engine imposes on itself** (12/10/2026): macOS has no reservation,
/// and reading the kernel's state to decide oscillated, read stale figures and arrived after the cliff
/// (measured); a reserve and a compression credit calibrated on six renders of one machine refused
/// renders the kernel would have served. So nothing here reads the kernel: the same render on the same
/// Mac gets the same answer and the same plan, always. What the others hold is theirs; the promise is
/// that Siliconed takes at most B.
public enum MemoryBudget {

    /// **B = ¾ · RAM − 2 GiB** (decided on 12/10/2026): 8 GiB → 4, 16 → 10, 24 → 16, 32 → 22, 64 → 46.
    /// A RAM is counted in GiB (`ProcessInfo.physicalMemory` of a "16 GB" Mac is 16 GiB), and so is B.
    /// **Below 8 GiB nothing is aimed at**: the formula still answers — 6 GiB gives 2.5 (2.68 GB),
    /// which refuses Z-Image (bf16) from 1024² (3.28 GB) and takes its 512² and every Qwen-Image-2.1
    /// request (≤ 2.46 GB, 12/10) — nothing measured there; under 2⅔ GiB, 0: every render refused.
    public static func ceiling(physical: Int) -> Int { max(0, physical / 4 * 3 - 2 * gib) }

    /// **The smallest RAM whose B covers `need`**, in whole GiB — the inverse of `ceiling(physical:)`.
    public static func physicalNeeded(for need: Int) -> Int {
        var n = max(1, ((max(0, need) + 2 * gib) * 4 + 3 * gib - 1) / (3 * gib))
        while n > 1, ceiling(physical: (n - 1) * gib) >= need { n -= 1 }
        while ceiling(physical: n * gib) < need { n += 1 }
        return n
    }

    /// **The memory sizes Macs are sold with**, in GB (Apple silicon, 2020–2026). A table, on purpose: a
    /// « needs a Mac with 9 GB » answers nobody, the question is which Mac. It moves once a chip
    /// generation; a size missing here falls to the next one listed, never to a figure that is not sold.
    public static let soldSizes = [8, 16, 18, 24, 32, 36, 48, 64, 96, 128, 192, 256, 512]

    /// **The N of the app's « Needs a Mac with N GB of memory »**: the smallest size sold whose B covers
    /// `need` (`physicalNeeded` rounded up to `soldSizes`); beyond the largest, the raw inverse.
    public static func macNeeded(for need: Int) -> Int {
        let raw = physicalNeeded(for: need)
        return soldSizes.first { $0 >= raw } ?? raw
    }

    /// **B for this Mac**: the forcing (`SILICONED_MEMORY_BUDGET_GB`) when set, otherwise
    /// `ceiling(physical:)` of its RAM. What the preflight refuses on, the app greys out with, and the
    /// plan is chosen in.
    public static func machineCeiling(environment: [String: String]? = nil) -> Int {
        let environment = environment ?? liveEnvironment([forcingVariable])
        return forcedCeiling(environment) ?? ceiling(physical: Int(ProcessInfo.processInfo.physicalMemory))
    }

    /// Whether `SILICONED_MEMORY_BUDGET_GB` replaces this Mac's B.
    package static var isForced: Bool { forcedCeiling(liveEnvironment([forcingVariable])) != nil }

    private static let gib = 1 << 30

    /// **The forcing** (`SILICONED_MEMORY_BUDGET_GB`, GiB): B itself — "4" simulates an 8 GB Mac, "22"
    /// a 32 GB one, refused and planned as one. A tool of measurement and of the tests, read in the
    /// engine and in the app alike.
    package static let forcingVariable = "SILICONED_MEMORY_BUDGET_GB"

    package static func forcedCeiling(_ environment: [String: String]) -> Int? {
        guard let text = environment[forcingVariable],
              let gib = Double(text.replacingOccurrences(of: ",", with: ".")), gib >= 0 else { return nil }
        return Int(gib * 1_073_741_824)
    }

    /// The process's environment, read live (`getenv`: a test may set it), for these variables only.
    package static func liveEnvironment(_ names: [String]) -> [String: String] {
        var environment: [String: String] = [:]
        for name in names { if let value = getenv(name) { environment[name] = String(cString: value) } }
        return environment
    }

    /// B in one line, for the developer's command line.
    package static func summary(_ ceiling: Int) -> String {
        String(format: "B %.2f GB%@", Double(ceiling) / 1e9, (isForced ? " (forced: \(forcingVariable))" : "") as NSString)
    }
}

// ── the plan, chosen in B ────────────────────────────────────────────────────────────────────

/// **The execution settings of one render, chosen in B before its first computation** —
/// beside `EngineSettings.effective`, which is read once per process. Nothing is read from the machine,
/// and nothing changes once the render runs: the same request on the same Mac takes the same plan.
///
///     request ── the lean plan: whole map streamed (two buffers),   ── need ≤ B, or refused (Preflight)
///                no encoder buffer, the decoders' smallest bands
///                  + the text encoder's four buffers                ── kept if the need stays ≤ B
///                  + the decoders' standard bands                   ── kept if the need stays ≤ B
///                  └─ the map's head left to the page cache: what B leaves beside the denoising (`released`)
///
/// What only buys time is chosen here. What changes a bit — `mlpBudget`, the prefill's groups,
/// `maxMatrix`, `vaeRequestBlock`, `vaeTile`, the decoder's band alignment — is not, and stays fixed.
///
/// What a reading by analogy would get wrong:
///
///   - **Each knob lives in its own phase** — the encoder's buffers in `text`, the bands in `decoding`,
///     the map's buffers in `denoising` — and the need is the largest phase: each is kept or dropped
///     alone, and adding one never raises another phase.
///   - **The map's head is never in the need.** It is clean, file-backed page cache: the kernel drops
///     it without writing, never swaps it. It is the soft part of B — what B leaves beside the
///     denoising's anonymous need — and **capped at the model's own split** (0.5; Qwen-Image-2.1's
///     generation 1, no head): on a 16 GB Mac a head larger than half was read back more, not less
///     (measured: heads of 0.43–0.73 read 6.6–8.6 GB an evaluation, 5.4 at 0.5) — the page cache
///     holds less than any figure says. Only a map that fits whole in that room leaves the stream (no
///     buffer at all: a 32 GB Mac, an 8-bit map).
///   - **Never one buffer**: measured, one serializes the reads behind the computation (Z-Image 512²
///     14.4 s a step against 4.1 with two; Qwen 512 20 s against 6), the same bits. The lean plan streams
///     the whole map with its two: a budget too short for a head is a cache too short for it, and the
///     map through the cache is what swapped (234.7 GB through the compressor in a swapping render,
///     13.1 with the whole map streamed).
///   - **The check is against the plan's own need** (`Render.need`): a render on a smaller plan is
///     judged on that plan, not on the fastest one it did not take. A map left wholly to the cache
///     peaks below it (its two buffers are charged, not allocated).
///
/// Outside a render (a check, a bench) there is no plan: each site keeps the reference values it had
/// before — the measurement tools stay what they were. `SILICONED_MAP_TAIL` set in the environment is a
/// forcing of measurement: the DiT then streams that fraction with two buffers, whatever B.
package enum MemoryPlan {

    /// The DiT's plan for its map, as the need charges it: the share of its layers streamed around the
    /// page cache, and how many staging buffers of its largest block.
    package struct Tail: Sendable, Equatable {
        package var fraction: Double
        package var slots: Int
        package static let none = Tail(fraction: 0, slots: 0)
        /// **The lean plan's map: all of it beside the cache, with its two buffers**.
        package static let lean = Tail(fraction: 1, slots: 2)
        /// The first of `layers` streamed (`layers`: none).
        package func released(layers: Int) -> Int {
            slots == 0 ? layers : Prefetcher.firstReleasedLayer(of: layers, tail: fraction)
        }
    }

    /// The text encoder's reference: four buffers (three layers ahead). Nothing between four and none:
    /// one buffer was measured worse than none on the DiT, two and three never.
    package static let encoderSlots = 4

    /// FLUX's decoder (Z-Image, klein, ERNIE): the smallest band proved and timed (7.3 s at 1024² against 4.4).
    package static let fluxMinimumBand = 1 << 20
    /// Qwen-Image-2.1's decoder: a quarter of the measured band (bits proved down to 2 rows; time not measured).
    package static let qwenMinimumBand = 1 << 16

    /// **One render's plan**, chosen by `choose` before its first computation.
    package struct Render: Sendable {
        /// B, the plan's ceiling.
        package var ceiling: Int
        /// The request at the plan chosen: its encoder buffers, its bands (the map's tail: `.lean`'s
        /// buffers, the head decided by `released` where the map is open).
        package var request: MemoryNeed.Request
        /// The chosen plan's need, phase by phase — what the render's peak is checked against.
        package var need: MemoryNeed?
        /// The lean plan's peak: the floor the preflight judged.
        package var floor: Int
        /// The denoising's need with the map streamed (its two buffers), and with it wholly cached.
        package var denoisingStreamed: Int
        package var denoisingCached: Int
        /// What each site decided, in order, for the report (the developer's command line prints it).
        package var decisions: [String] = []
        /// The lean plan, all of it: no encoder buffer and the smallest bands.
        package var lean: Bool { request.encoderSlots == 0 && request.minimumBands }
    }

    /// **The plan, from the request and B**: the lean plan, then each buffer that buys time while the
    /// need stays ≤ B. Pure: the tests drive it with simulated ceilings.
    package static func choose(_ request: MemoryNeed.Request, ceiling: Int) -> Render {
        var plan = request.lean
        let floor = MemoryNeed.of(plan)?.peak ?? 0
        for faster in [{ (r: inout MemoryNeed.Request) in r.encoderSlots = encoderSlots },
                       { (r: inout MemoryNeed.Request) in r.minimumBands = false }] {
            var candidate = plan
            faster(&candidate)
            if (MemoryNeed.of(candidate)?.peak ?? 0) <= ceiling { plan = candidate }
        }
        let need = MemoryNeed.of(plan)
        var cached = plan
        cached.tail = .none
        var render = Render(ceiling: ceiling, request: plan, need: need, floor: floor,
                            denoisingStreamed: need?.need(of: .denoising) ?? 0,
                            denoisingCached: MemoryNeed.of(cached)?.need(of: .denoising) ?? 0)
        render.decisions.append(String(format: "plan: need %.2f GB in B %.2f (floor %.2f)%@", Double(need?.peak ?? 0) / 1e9,
                                       Double(ceiling) / 1e9, Double(floor) / 1e9,
                                       (render.lean ? " — lean plan" : "") as NSString))
        return render
    }

    /// **The map's head, from B** — pure: the first layer streamed and the staging buffers.
    /// `spans`: each layer's bytes in the map; `mapBytes`: the whole map; `streamed` / `cached`: the
    /// denoising's need with the two buffers and without; `fraction`: the model's own split (its cap).
    ///
    /// The map fits whole beside the denoising's need: nothing streamed, no buffer. Otherwise the
    /// layers from the front whose bytes fit in what B leaves (the map's non-layer tensors, cached
    /// too, deducted first), never more than the model's split keeps.
    package static func released(spans: [Int], mapBytes: Int, ceiling: Int, streamed: Int, cached: Int,
                                 fraction: Double) -> (released: Int, slots: Int) {
        let layers = spans.count
        guard fraction > 0, layers > 0, mapBytes > ceiling - cached else { return (layers, 0) }
        let room = ceiling - streamed - (mapBytes - spans.reduce(0, +))
        var head = 0, held = 0
        while head < layers, held + spans[head] <= room { held += spans[head]; head += 1 }
        let released = min(head, Prefetcher.firstReleasedLayer(of: layers, tail: fraction))
        return released < layers ? (released, 2) : (layers, 0)
    }

    // ── the render in progress ─────────────────────────────────────────────────────────────

    private static let state = Mutex<Render?>(nil)
    private static let last = Mutex<Render?>(nil)

    /// The render in progress, or `nil` outside one.
    package static var current: Render? { state.withLock { $0 } }
    /// The last render's plan, once it is over (or while it runs).
    package static var lastRender: Render? { last.withLock { $0 } ?? current }

    /// Wires a render's plan in for the duration of `body` — the engine, around a render (renders are
    /// serialized: the engine queue and `RenderLock.machine`).
    package static func during<T>(_ render: Render, _ body: () throws -> T) rethrows -> T {
        state.withLock { $0 = render }
        defer {
            let ended = state.withLock { s -> Render? in let r = s; s = nil; return r }
            last.withLock { $0 = ended }
        }
        return try body()
    }

    package static func note(_ decision: String) {
        state.withLock { $0?.decisions.append(decision) }
    }

    /// `note`, unless it is already the last decision — a site asked at every evaluation.
    package static func noteOnce(_ decision: String) {
        state.withLock { s in if s?.decisions.last != decision { s?.decisions.append(decision) } }
    }

    // ── what the sites ask ─────────────────────────────────────────────────────────────────

    /// **The DiT's tail**, where it opens its map (`DiT.forward`, `QwenImage21DiT.pass`): the first
    /// layer streamed and the staging buffers (`released`). The same at every evaluation of a render:
    /// nothing it depends on moves. Outside a render, the model's split with two buffers; under
    /// `SILICONED_MAP_TAIL` (a forcing of measurement), that fraction with two buffers, journaled.
    ///
    /// `fraction`: the model's own split, when it has measured one — the setting's (0.5) otherwise. A
    /// `map_tail` set in the profile or the environment still wins.
    package static func ditTail(artifact: Artifact, layerBlocks: [[String]],
                                fraction preferred: Double? = nil) -> (released: Int, slots: Int) {
        let layers = layerBlocks.count
        let spans = layerBlocks.map { artifact.span(of: $0) }
        let block = spans.max() ?? 0
        func decide(_ plan: (released: Int, slots: Int), _ why: String) -> (released: Int, slots: Int) {
            let streamed = plan.released < layers ? plan.slots : 0
            noteOnce(String(format: "map tail %d of %d layers, %d buffer(s) of %.0f MB%@", layers - plan.released,
                            layers, streamed, Double(block) / 1e6, why as NSString))
            return plan
        }
        let settings = EngineSettings.effective
        if settings.provenance["map_tail"] == .environment {
            return decide((Tail(fraction: settings.mapTail, slots: 2).released(layers: layers), 2),
                          " (forced: SILICONED_MAP_TAIL)")
        }
        let fraction = settings.provenance["map_tail"] == nil ? preferred ?? settings.mapTail : settings.mapTail
        guard let render = current else {
            return decide(fraction > 0 ? (Tail(fraction: fraction, slots: 2).released(layers: layers), 2) : (layers, 0), "")
        }
        let mapBytes = artifact.tensors.values.reduce(0) { $0 + $1.bytes }
        let plan = released(spans: spans, mapBytes: mapBytes, ceiling: render.ceiling,
                            streamed: render.denoisingStreamed, cached: render.denoisingCached, fraction: fraction)
        let split = Prefetcher.firstReleasedLayer(of: layers, tail: fraction)
        let why = plan.slots == 0 && fraction > 0 ? " (the whole map in B)"
            : plan.released < split ? " (head: what B leaves beside the denoising)" : ""
        return decide(plan, why)
    }

    /// **The text encoder's buffers**, as it starts: the plan's. Outside a render, the reference (4).
    package static func encoderSlots(artifact: Artifact, layerBlocks: [[String]]) -> Int {
        guard let render = current else { return encoderSlots }
        let block = layerBlocks.map { artifact.span(of: $0) }.max() ?? 0
        note(String(format: "text encoder %d buffer(s) of %.0f MB", render.request.encoderSlots, Double(block) / 1e6))
        return render.request.encoderSlots
    }

    /// **A decoder's band**, as it starts decoding: the plan's — the measured size (FLUX's 2²³ elements,
    /// Qwen's 2¹⁸ pixels), or the smallest proved one on a plan that could not hold it. Never larger:
    /// 2²⁴ costs 140 MB more for the same time. Outside a render, `standard` — the static a check may
    /// have set.
    package static func decoderBand(standard: Int, minimum: Int, unit: String) -> Int {
        guard let render = current else { return standard }
        let band = render.request.minimumBands ? min(standard, minimum) : standard
        note("decoder bands of 2^\(band.trailingZeroBitCount) \(unit)")
        return band
    }
}

extension Artifact {
    /// The bytes a run of tensors spans in the file — a streamed block's staging size.
    package func span(of names: [String]) -> Int {
        let ranges = names.compactMap { tensors[$0] }.map { ($0.offset, $0.offset + $0.bytes) }
        guard let first = ranges.map(\.0).min(), let end = ranges.map(\.1).max() else { return 0 }
        return end - first
    }
}
