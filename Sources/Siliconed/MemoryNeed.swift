import Foundation
import os
import Synchronization

/// **The memory a render needs, computed before its first computation from its inputs** — never read
/// from the machine, never fitted on it, never looked up in a table of measured renders.
///
///     Request ── family's phases ──┬─ text        ┐
///      (family, W×H, references,   ├─ vision      │  peak = max over phases of
///       prompt tokens, LoRA ranks, ├─ references  │        (what is alive at the phase's peak)
///       card, plan)                ├─ denoising   │        + what crosses the phases
///                                  └─ decoding    ┘        + the process's base
///
/// **Every term is a shape function its allocator calls** — `DiT.arenaBytes`, `TextEncoder.arenaBytes`,
/// `HeadSplitAttention.hostBytes`, `Attention.maskBytes`, `Conductor.workspaceBytes`, `VAE.plan`, … —
/// so the formula equals the allocation by construction: an arena, a `Pages`, a VAE heap is mapped at
/// the size the formula charges. What no constructor of ours sizes is in `OperatorFootprint`, named,
/// bounded, and short (MPSGraph's copy of the queries, MPS's GEMM state, the process's base).
///
/// A phase charges **only what is alive at its peak**, following the pipeline's real order: a
/// tokenizer dropped before its encoder maps its arena, a half-size attention dropped when the full
/// steps begin, a VAE's heaps released before the next phase are not charged beside what follows them.
///
/// **The verification is the product's** (`MemoryCheck`): every render reads its own peak
/// (`ri_interval_max_phys_footprint`) and compares it with this; a peak above the need is a defect of
/// the code — an allocation outside our hand found —, never a figure to fold back in here.
package struct MemoryNeed: Sendable, Equatable {

    /// **What a render's need depends on** — every input that makes it larger.
    package struct Request: Sendable {
        package var card: ModelCard
        package var width: Int
        package var height: Int
        /// Each reference image at its own size (named subjects' included; Z-Image's img2img start),
        /// in pixels as handed to the engine.
        package var references: [(width: Int, height: Int)] = []
        /// The prompt's tokens (template included, as the encoder sees them) — or an upper bound of
        /// them (`promptTokens(_:family:references:)`).
        package var promptTokens: Int
        /// The rank of each LoRA applied.
        package var loraRanks: [Int] = []
        /// The DiT's map plan (tail fraction and staging buffers).
        package var tail: MemoryPlan.Tail = .init(fraction: 0.5, slots: 2)
        /// The text encoder's staging buffers (`MemoryPlan.encoderSlots`; 0 on the lean plan).
        package var encoderSlots: Int = MemoryPlan.encoderSlots
        /// The decoders' smallest bands (`MemoryPlan.fluxMinimumBand`, `qwenMinimumBand`), as the lean
        /// plan takes them; otherwise the standard ones.
        package var minimumBands = false

        package init(card: ModelCard, width: Int, height: Int, references: [(width: Int, height: Int)] = [],
                     promptTokens: Int, loraRanks: [Int] = [], tail: MemoryPlan.Tail = .init(fraction: 0.5, slots: 2)) {
            self.card = card; self.width = width; self.height = height; self.references = references
            self.promptTokens = promptTokens; self.loraRanks = loraRanks; self.tail = tail
        }

        /// **The most economical plan** (`MemoryPlan`'s lean one): the whole map streamed through its two
        /// buffers, no encoder buffer, the decoders' smallest bands — the floor a render can go down to.
        package var lean: Request {
            var r = self
            r.tail = .lean; r.encoderSlots = 0; r.minimumBands = true
            return r
        }

        /// **An upper bound of the encoder's tokens for `prompt`**, without the tokenizer: a BPE token is
        /// at least one UTF-8 byte, so the bytes of the template and the prompt bound the tokens, cut where
        /// the encoder cuts (512). Z-Image: the chat template around the prompt; Qwen-Image-2.1: the
        /// system turn, the user header, each reference's tags and the closing turn (`QwenImage21Need`).
        package static func promptTokens(_ prompt: String, family: Family, references: Int = 0) -> Int {
            promptTokens(text: prompt.utf8.count, family: family, references: references)
        }

        /// The same for `text` tokens of prompt (or bytes, their bound): the template around them added.
        package static func promptTokens(text: Int, family: Family, references: Int = 0) -> Int {
            switch family {
            case .qwenImage21:
                return QwenImage21Need.dropCount + 3 + references * (QwenImage21Need.tokensBefore + QwenImage21Need.tokensAfter)
                    + 5 + min(text, Qwen3VLPrompt.maxPromptTokens)
            default:
                return min(Tokenizer.chatPrefix.utf8.count + text + Tokenizer.chatSuffix.utf8.count, 512)
            }
        }

        /// The prompt the app and the diagnostic assume before a prompt is known: 128 tokens of text.
        /// A figure for a display, never the render's own (the engine passes `promptTokens(_:…)`).
        package static let typicalPromptTokens = 128
    }

    /// **One allocation alive at a phase's peak**: its name (the function it comes from), its bytes,
    /// and whether it is ours (sized by our shape function) or outside our hand (`OperatorFootprint`).
    package struct Term: Sendable, Equatable {
        package var name: String
        package var bytes: Int
        package var outside: Bool
        package init(_ name: String, _ bytes: Int, outside: Bool = false) {
            self.name = name; self.bytes = bytes; self.outside = outside
        }
    }

    /// One phase of a render: what is alive at its peak.
    package struct Phase: Sendable, Equatable {
        package enum Kind: String, Sendable, CaseIterable, Codable { case text, vision, references, denoising, decoding }
        package var kind: Kind
        package var terms: [Term]
        package init(kind: Kind, terms: [Term]) { self.kind = kind; self.terms = terms.filter { $0.bytes != 0 } }
        /// Ours: arenas, `Pages`, VAE heaps, host copies — each from its shape function.
        package var own: Int { terms.filter { !$0.outside }.reduce(0) { $0 + $1.bytes } }
        /// Outside our hand: `OperatorFootprint`'s named terms.
        package var outside: Int { terms.filter(\.outside).reduce(0) { $0 + $1.bytes } }
        package var bytes: Int { own + outside }
        /// The bytes of the terms whose name starts with `prefix`.
        package func bytes(_ prefix: String) -> Int { terms.filter { $0.name.hasPrefix(prefix) }.reduce(0) { $0 + $1.bytes } }
    }

    package var phases: [Phase]
    /// What crosses the phases (latents, encoded text, images handed in), held through all of them.
    package var crossing: Int
    /// The process's own base (`OperatorFootprint.processBase`).
    package var base: Int
    /// **No formula for this family yet**: the need is the fallback of measured peaks
    /// (`MemoryNeed.unmodelled`), not a computation.
    package var isFallback = false

    /// **The peak**: the largest phase plus what crosses them, plus the base.
    package var peak: Int { (phases.map(\.bytes).max() ?? 0) + crossing + base }

    /// **A phase's own need**: its terms, what crosses, the base — what that phase's measured peak
    /// is compared with.
    package func need(of kind: Phase.Kind) -> Int? {
        phases.filter { $0.kind == kind }.map(\.bytes).max().map { $0 + crossing + base }
    }

    /// **The need of a request.** Z-Image and Qwen-Image-2.1 are computed; the four hidden families
    /// (Anima, Krea 2, klein, ERNIE) have no shape functions for their DiTs yet and take the marked
    /// fallback (`unmodelled`). `nil` only when Z-Image's maps are not installed (the render is
    /// refused before, by the preflight's first check).
    package static func of(_ request: Request, app: Bool = false) -> MemoryNeed? {
        switch request.card.family {
        case .zImage: zImage(request, app: app)
        case .qwenImage21: qwenImage21(request, app: app)
        default: unmodelled(request)
        }
    }

    // ── the pieces every formula takes, from the constructors' functions ─────────────────────

    /// **One cached attention**, as `Block`, `TextEncoder`, `QwenImage21DiT` and the vision tower build
    /// it: cut by heads (`HeadSplitAttention`: its host arena, ours, and MPSGraph's copy of the GPU
    /// heads' queries) or in one MPSGraph SDPA (its queries' copy, and its causal mask in our pages).
    package static func attention(_ name: String, heads: Int, rows: Int, keys: Int, headDim: Int, causal: Bool,
                                  split: Bool, threads: Int) -> [Term] {
        if split {
            let gpu = HeadSplitAttention.partition(heads: heads, fraction: HeadSplitAttention.cpuFraction).gpu.count
            return [Term("\(name) head split", HeadSplitAttention.hostBytes(heads: heads, sequence: rows, keys: keys,
                                                                           headDim: headDim, threads: threads)),
                    Term("\(name) SDPA queries", OperatorFootprint.sdpaQueries(rows: rows, heads: gpu, headDim: headDim),
                         outside: true),
                    Term("\(name) SDPA executable", OperatorFootprint.sdpaExecutable, outside: true)]
        }
        return [Term("\(name) SDPA queries", OperatorFootprint.sdpaQueries(rows: rows, heads: heads, headDim: headDim),
                     outside: true),
                Term("\(name) SDPA executable", OperatorFootprint.sdpaExecutable, outside: true)]
            + (causal ? [Term("\(name) causal mask", Attention.maskBytes(sequence: rows, keys: keys))] : [])
    }

    /// **One model's `GEMM` over a phase**: MPS's state, and the conductor's BNNS workspace — the
    /// largest it can grow to over the products the phase runs (`Conductor.workspaceBytes` of the AMX
    /// slice, either weight layout), when the cut applies. The slice is at most `m − cut(m, T)` with the
    /// servo's `T` held in `Conductor.fractionRange`: the bound is its lower end (and the frozen cut's).
    package static func gemm(_ name: String, products: [(m: Int, k: Int, n: Int)],
                             amx: Bool = EngineSettings.effective.amx) -> [Term] {
        var workspace = 0
        if amx {
            for p in products where p.m >= Conductor.minimumRows {
                let slices = Set([p.m - Conductor.cut(p.m, Conductor.fractionRange.lowerBound),
                                  p.m - Conductor.cut(p.m, EngineSettings.effective.amxFraction)])
                for rows in slices where rows > 0 {
                    for transposed in [false, true] {
                        workspace = max(workspace, Conductor.workspaceBytes(rows: rows, k: p.k, n: p.n,
                                                                            weightIsTransposed: transposed))
                    }
                }
            }
        }
        return [Term("\(name) MPS GEMM", OperatorFootprint.mpsGEMM, outside: true)]
            + (workspace > 0 ? [Term("\(name) conductor workspace", Arena.capacity(for: workspace))] : [])
    }

    /// **A VAE's two heaps and its host copies** (`VAELayout.bytes`), and what its graphs still
    /// allocate beside them: MPSGraph's SDPA, the one pass left whole (`VAELayout.outside`).
    package static func vae(_ name: String, _ layout: VAELayout) -> [Term] {
        [Term("\(name) registers", layout.registers), Term("\(name) workspace", layout.workspace),
         Term("\(name) host", layout.host), Term("\(name) SDPA outside", layout.outside, outside: true)]
            + (layout.outside > 0 ? [Term("\(name) SDPA executable", OperatorFootprint.sdpaExecutable, outside: true)] : [])
    }

    // ── the families without a formula ────────────────────────────────────────────────────────

    /// **⚠ FALLBACK — not a computation.** Anima, Krea 2, FLUX.2 [klein] and ERNIE have no shape
    /// functions for their DiTs and encoders yet (their constructors size their buffers inline): their
    /// need is still the whole-render peak measured on the M1 Pro, the table `measuredPeaks`
    /// held for them. A format falls in the smallest measured area that holds it and takes the largest
    /// figure up to it; beyond the family's measurements, the largest figure any of these families
    /// measured at or beyond it (klein and ERNIE at 1024×1536 take Krea 2's 7.8 GB). The references
    /// are not modelled (these families were measured without). Hidden families: to be replaced by
    /// their formulas when their constructors expose their sizes.
    package static func unmodelled(_ request: Request) -> MemoryNeed? {
        let area = request.width * request.height
        func table(_ family: Family) -> [(area: Int, bytes: Int, source: String)] {
            func p(_ w: Int, _ h: Int, _ gb: Double, _ s: String) -> (Int, Int, String) { (w * h, Int((gb * 1e9).rounded()), s) }
            switch family {
            case .anima: return [p(832, 1216, 5.3, "M1 Pro"), p(1024, 1024, 5.2, "M1 Pro"), p(1024, 1536, 7.6, "M1 Pro")]
            case .krea2: return [p(512, 512, 0.984, "M1 Pro"), p(832, 1216, 5.2, "M1 Pro"), p(1024, 1024, 5.1, "M1 Pro"),
                                 p(1024, 1536, 7.8, "M1 Pro")]
            case .klein4b: return [p(1024, 1024, 5.1, "M1 Pro")]
            case .ernie: return [p(1024, 1024, 4.9, "M1 Pro")]
            default: return []
            }
        }
        let own = table(request.card.family)
        guard !own.isEmpty else { return nil }
        let figure: (bytes: Int, source: String)
        if let tier = own.map(\.area).filter({ $0 >= area }).min() {
            let best = own.filter { $0.area <= tier }.max { $0.bytes < $1.bytes }!
            figure = (best.bytes, best.source)
        } else {
            let beyond = [Family.anima, .krea2, .klein4b, .ernie].flatMap(table).filter { $0.area >= area }
            let best = (beyond + own).max { $0.bytes < $1.bytes }!
            figure = (best.bytes, best.source)
        }
        return MemoryNeed(phases: [Phase(kind: .denoising,
                                         terms: [Term("measured peak (fallback, \(figure.source))", figure.bytes, outside: true)])],
                          crossing: 0, base: 0, isFallback: true)
    }
}

// ── the product's own verification ───────────────────────────────────────────────────────────

/// **How a render's peaks were read** (`FootprintInterval`).
public enum PeakMethod: String, Codable, Sendable, Equatable {
    /// The kernel's `ri_interval_max_phys_footprint`, reset at each boundary: exact.
    case kernelInterval
    /// `phys_footprint` sampled every 10 ms, the kernel's reset missing or failing: a lower bound.
    case sampled10ms
}

/// **What a render needed, and what it took** — computed at its start (`MemoryNeed.of`), read at its
/// end (`ri_interval_max_phys_footprint`, phase by phase, or the 10 ms sampler: `peakMethod`). A phase
/// whose measured peak passes its need is a **defect of the code** (an allocation outside our hand
/// found), logged under the `memory` category and carried by the render's result; never a figure to
/// fold back into the formula.
public struct MemoryCheck: Codable, Sendable, Equatable {
    /// One phase of the render, as `PhaseBoundary` marked it.
    public struct Phase: Codable, Sendable, Equatable {
        /// The boundary's name (`text`, `vision`, `reference 1`, `denoising`, `decoding`…).
        public var name: String
        /// The formula's need for this phase, crossing and base included; `nil` for a boundary the
        /// formula has no phase for (`start`, `denoiser released`: charged to their neighbours).
        public var needBytes: Int?
        /// `phys_footprint`'s maximum over the phase (the kernel's interval maximum).
        public var measuredBytes: Int
        /// Measured minus need: positive is a defect.
        public var gapBytes: Int?
        public init(name: String, needBytes: Int?, measuredBytes: Int) {
            self.name = name; self.needBytes = needBytes; self.measuredBytes = measuredBytes
            gapBytes = needBytes.map { measuredBytes - $0 }
        }
        private enum CodingKeys: String, CodingKey { case name, needBytes, measuredBytes, gapBytes }
        /// Explicit `null`s, as the diagnostic's JSON writes them: "no need for this boundary" is said.
        public func encode(to encoder: any Encoder) throws {
            var c = encoder.container(keyedBy: CodingKeys.self)
            try c.encode(name, forKey: .name); try c.encode(needBytes, forKey: .needBytes)
            try c.encode(measuredBytes, forKey: .measuredBytes); try c.encode(gapBytes, forKey: .gapBytes)
        }
    }
    /// The formula's peak (`MemoryNeed.peak`).
    public var needBytes: Int
    /// The render's own peak: the largest phase's.
    public var measuredBytes: Int
    public var phases: [Phase]
    /// The need came from the hidden families' measured fallback, not from a formula.
    public var fallback: Bool
    /// Measured minus need, over the whole render.
    public var gapBytes: Int
    /// How the phases' peaks were read (`FootprintInterval`): exact, or sampled every 10 ms.
    public var peakMethod: PeakMethod
    public init(needBytes: Int, measuredBytes: Int, phases: [Phase], fallback: Bool,
                peakMethod: PeakMethod = .kernelInterval) {
        self.needBytes = needBytes; self.measuredBytes = measuredBytes; self.phases = phases; self.fallback = fallback
        self.peakMethod = peakMethod
        gapBytes = measuredBytes - needBytes
    }
    /// **The phases above their need** — each one a defect.
    public var exceeded: [Phase] {
        phases.filter { ($0.gapBytes ?? 0) > 0 } + (gapBytes > 0 && !phases.contains { ($0.gapBytes ?? 0) > 0 }
            ? [Phase(name: "render", needBytes: needBytes, measuredBytes: measuredBytes)] : [])
    }
    public var isDefect: Bool { !exceeded.isEmpty }

    /// The engine's memory log: `log stream --predicate 'subsystem == "dev.gwenn-ha.siliconed" && category == "memory"'`.
    package static let log = Logger(subsystem: "dev.gwenn-ha.siliconed", category: "memory")

    /// The `PhaseBoundary` names, mapped to the formula's phases.
    package static func kind(of boundary: String) -> MemoryNeed.Phase.Kind? {
        switch boundary {
        case "vision": .vision
        case "text": .text
        case "image": .references
        case "denoising", "denoiser released": .denoising
        case "decoding": .decoding
        default: boundary.hasPrefix("reference") ? .references : nil
        }
    }

    /// **The check of one render**: each marked phase against its own need.
    package init(need: MemoryNeed, marks: [(name: String, peak: Int)],
                 peakMethod: PeakMethod = FootprintInterval.method) {
        let phases = marks.map { mark in
            Phase(name: mark.name, needBytes: MemoryCheck.kind(of: mark.name).flatMap { need.need(of: $0) },
                  measuredBytes: mark.peak)
        }
        self.init(needBytes: need.peak, measuredBytes: marks.map(\.peak).max() ?? 0, phases: phases,
                  fallback: need.isFallback, peakMethod: peakMethod)
    }

    /// One line per defect, for the log and the CLI.
    public var defects: [String] {
        exceeded.map {
            String(format: "memory defect: %@ peaked at %.0f MB, its need is %.0f MB (+%.0f MB)%@", $0.name as NSString,
                   Double($0.measuredBytes) / 1e6, Double($0.needBytes ?? needBytes) / 1e6,
                   Double($0.gapBytes ?? gapBytes) / 1e6, fallback ? " — need from the measured fallback" : "")
        }
    }
}

// ── the shape functions' memory ──────────────────────────────────────────────────────────────

/// **A shape function's results, kept by its inputs.** The need is asked at every preflight and by the
/// app for each format it greys out, and a VAE's plan walks every pass and places every slot (a Qwen
/// decoder 16 ms, three references' sequence 12 ms, in a debug build): the same inputs give the same
/// bytes, so each is walked once per process. A pure function only — its key holds every input it
/// reads; the value is what the function returned, never a copy rebuilt. Bounded: past `capacity`
/// keys, everything is dropped and walked again (a format grid, not a growing log).
package final class ShapeMemo<Key: Hashable & Sendable, Value: Sendable>: Sendable {
    private let table = Mutex<[Key: Value]>([:])
    private let capacity: Int

    package init(capacity: Int = 64) { self.capacity = capacity }

    /// The value of `key`, computed by `compute` the first time (outside the lock: two threads may both
    /// walk it, and keep the same bytes).
    package func callAsFunction(_ key: Key, _ compute: () -> Value) -> Value {
        if let value = table.withLock({ $0[key] }) { return value }
        let value = compute()
        table.withLock { table in
            if table.count >= capacity { table.removeAll(keepingCapacity: true) }
            table[key] = value
        }
        return value
    }
}

/// The key of a VAE's plan: its config and the integers of its shape, in the function's order.
package struct VAEPlanKey<Config: Hashable & Sendable>: Hashable, Sendable {
    package var config: Config
    package var shape: [Int]
    package init(_ config: Config, _ shape: [Int]) { self.config = config; self.shape = shape }
}
