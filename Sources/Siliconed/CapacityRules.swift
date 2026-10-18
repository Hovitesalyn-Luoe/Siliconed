import Foundation

/// **The rules: what the capacity map decides about the engine's compute branches** — each one a rule on
/// measured capacities (a throughput, a correctness check), never on a chip's name. One place: the app
/// reads a map, these rules turn it into settings (`EngineSettings.load(from:map:)`), and the report says
/// what they chose (`Report.decisions`), so a shared JSON tells what that Mac uses.
///
///     a map (Diagnostic.Report)
///        │  margin = max(5 %, the drift witness's worst |1 − after / before|, GPU and AMX)
///        │
///        ├─ products, m = 1024 and 4096        g_m = Σ t(GPU alone) / Σ t(GPU + AMX)   ─▶ amx
///        │                                     share = R_gpu / (R_gpu + R_amx)          ─▶ amx_fraction
///        │  cuts, k and n axes at both m         ρ = (1 − T*) / T*, T* balanced side by side ─▶ amx_shares, T(k, n, m)
///        │                                     g_m = Σ [t(fp32 path) + t(widen k·n)]
///        │                                         / Σ [t(bf16 weight) + t(copy k·n)]    ─▶ gemm_tensor
///        │                                     g_m = Σ [t(fp32 path) + t(widen convrot k·n)]
///        │                                         / Σ [t(int8 weight) + t(copy k·n)]    ─▶ gemm_int8
///        ├─ attention, S = 1024 and 4096       g_S = Σ t(MPSGraph) / Σ t(flash)          ─▶ flash
///        │                                     g_S = Σ t(MPSGraph) / Σ t(head split)     ─▶ head_split
///        └─ a branch is on iff   every engine it would use passed its check (none missing)
///                           and  gain = geometric mean of the g over the two sizes  > 1 + margin
///                           and  no size loses beyond the noise: min g > 1 − margin
///
/// `t = flops / rate` for each item; a `g` sums the items of one size — both models' blocks, up and
/// down, so a size counts as the products one of its DiT blocks runs. The two sizes are a 512² and a
/// 1024² image's tokens, and **each weighs the same** in the geometric mean: the app renders both, and
/// the 4096 items, four times the work, would otherwise decide for the 512² render too. The `min`
/// clause keeps a gain at 1024² from buying a loss at 512².
///
/// What sets these rules apart — what rules written by analogy with the defaults would get wrong:
///
///   - **the margin is the map's own noise, not a constant.** A map taken on a hot or battery-throttled
///     machine drifts (this M1 Pro on battery, 14/10/2026: the GPU ×0.84 over 16 s); 5 % is only the
///     floor. Within the margin, a branch stays on the side the GPU alone stands — MPSGraph, no cut:
///     the path with fewer moving parts. Flash on this M1 Pro is such a tie, and stays off;
///   - **`amx_fraction` is the share that balances the whole set of products**: `R` is each engine's
///     total work over its total time (all eight items, each weighted by its flops), the GPU taking
///     `R_gpu / (R_gpu + R_amx)` of the rows — **only without `cuts`**: with them, every product's share is
///     computed from its `k`, `n` and `m` (`Conductor.Shares`; one frozen share left the GPU waiting 2.4 s of a
///     1024² Krea 2 evaluation), at quantum 64. Both move the frozen cut, hence the last bits of a
///     render — like `amx` itself; neither re-associates a sum (one engine per output row);
///   - **`gemm_tensor` is judged against the path it replaces, preparation included**: the fp32 path
///     is the GPU alone, or GPU + AMX when `amx` went on, each weight widened first; the bf16 path
///     copies the weight instead (`Artifact.stage`) — both at the map's own CPU rates (`widening`
///     "bf16" and "bf16 copy"). It has no AMX: when it goes on, `amx` goes off in the settings (the
///     decision still says what co-execution measured);
///   - **`gemm_int8` likewise, against what an int8 map's default path does**: the convrot widening
///     ("int8 convrot") then the fp32 path, against the bytes copied ("int8 copy") then the int8 product —
///     beside the AMX on its columns when the AMX stays on (`amx` on, `gemm_tensor` off), alone otherwise.
///     It keeps the AMX: it reads its own widened columns. A map measures it on synthetic convrot weights
///     whatever is installed; on a bf16 map the setting does nothing;
///   - **flash and the head split are not exclusive**: Z-Image's blocks take flash before the split
///     (`Block`), Qwen-Image-2.1's target only has the split. So flash goes on only if it also beats the
///     split where the split goes on — otherwise Z-Image would trade a faster path for a slower one;
///   - **what the map does not decide stays at its default**, by rule, not by omission:
///       · `widen_gpu`: its isolated throughput does not decide it — in a render the widening queues
///         behind the GEMM, which is the bottleneck, and the isolated measure flips between two
///         runs of the same machine;
///       · the head split's `cpuFraction` (0.4): the map measures the split only at 0.4;
///       · fp16 / bf16 *activations*: P20 (never under the published weights nor under bf16) and the
///         `sdpa_fp16` decision stand — the half-precision figures answer a fleet question, they choose
///         nothing (`gemm_tensor` rounds nothing: its activations stay fp32);
///       · `amx_quantum` (64 under the grid, MPS's row tile), `amx_min`, `parallel`: nothing measures them.
extension Diagnostic {
    /// **One branch, decided**: on or off, and the figures behind it.
    public struct Decision: Codable, Sendable, Equatable {
        /// The branch is taken.
        public var on: Bool
        /// Every engine the branch would use ran and computed right (within its threshold).
        public var passed: Bool
        /// The geometric mean of `gains`: what is compared to `1 + margin`.
        public var gain: Double
        /// Per size (`Diagnostic.heights`, 1024 then 4096): the reference's summed time over the branch's.
        public var gains: [Double]

        public init(on: Bool, passed: Bool, gain: Double, gains: [Double]) {
            self.on = on; self.passed = passed
            self.gain = Diagnostic.significant(gain, 3); self.gains = gains.map { Diagnostic.significant($0, 3) }
        }
    }

    /// **What a map makes the engine use** — the rules above applied to one report.
    public struct Decisions: Codable, Sendable, Equatable {
        /// The noise a gain must clear: `max(minimumMargin, the drift witness's worst deviation)`.
        public var margin: Double
        /// GPU + AMX co-execution of the DiT products (`amx`).
        public var amx: Decision
        /// The GPU's share of the rows (`amx_fraction`) when `amx` is on; `nil` otherwise.
        public var amxFraction: Double?
        /// The GPU's share per weight shape and height (`amx_shares`, `Conductor.Shares`' text) when `amx` is
        /// on and the report measured `cuts`; `nil` otherwise — the shapes then take `amxFraction`.
        public var amxShares: String?
        /// The in-house flash kernel instead of MPSGraph (`flash`).
        public var flash: Decision
        /// The SDPA's heads cut between the GPU and the CPU from S = 1024 on (`head_split`).
        public var headSplit: Decision
        /// The DiT's products on the bf16 weights (`gemm_tensor`, `TensorGEMM`); `nil` in a decision
        /// stored by a report of schema 5.
        public var tensor: Decision?
        /// The products on the published int8 weights (`gemm_int8`); `nil` before schema 8.
        public var int8: Decision?

        public init(margin: Double, amx: Decision, amxFraction: Double?, flash: Decision, headSplit: Decision,
                    tensor: Decision? = nil, amxShares: String? = nil, int8: Decision? = nil) {
            self.margin = Diagnostic.significant(margin, 3); self.amx = amx
            self.amxFraction = amxFraction.map { Diagnostic.significant($0, 3) }
            self.amxShares = amxShares
            self.flash = flash; self.headSplit = headSplit; self.tensor = tensor; self.int8 = int8
        }

        private enum CodingKeys: String, CodingKey { case margin, amx, amxFraction, amxShares, flash, headSplit, tensor, int8 }
        public func encode(to encoder: any Encoder) throws {
            var c = encoder.container(keyedBy: CodingKeys.self)
            try c.encode(margin, forKey: .margin); try c.encode(amx, forKey: .amx)
            try c.encode(amxFraction, forKey: .amxFraction)
            try c.encode(amxShares, forKey: .amxShares)
            try c.encode(flash, forKey: .flash); try c.encode(headSplit, forKey: .headSplit)
            try c.encode(tensor, forKey: .tensor); try c.encode(int8, forKey: .int8)
        }

        /// The settings they write, under `EngineSettings`' keys (those of `store/profil.json`).
        package var settings: [String: Any] {
            let tensorOn = tensor?.on == true
            var values: [String: Any] = ["amx": amx.on && !tensorOn, "flash": flash.on, "head_split": headSplit.on,
                                         "gemm_tensor": tensorOn, "gemm_int8": int8?.on == true]
            if amx.on, !tensorOn, let amxFraction { values["amx_fraction"] = amxFraction }
            if amx.on, !tensorOn, let amxShares { values["amx_shares"] = amxShares; values["amx_quantum"] = 64 }
            return values
        }
    }

    /// The margin's floor: below it, two runs of one machine already disagree (invariant 6: 4 to 13 %
    /// over a few renders; the map's own witness ×0.990–0.997 on a quiet M1 Pro).
    public static let minimumMargin = 0.05

    /// **The rules.** Pure: the same report always gives the same decisions.
    public static func decide(products: [Product], attention: [AttentionCapacity], widening: [Widening],
                              witness: Witness, cuts: [Cut] = []) -> Decisions {
        var drift = abs(1 - witness.gpuAfter / witness.gpuBefore)
        if let before = witness.amxBefore, let after = witness.amxAfter, before > 0 { drift = max(drift, abs(1 - after / before)) }
        let margin = max(minimumMargin, drift.isFinite ? drift : 1)

        /// `on`, from the per-size gains and the checks.
        func decision(passed: Bool, gains: [Double]) -> Decision {
            let usable = passed && !gains.isEmpty && gains.allSatisfy { $0.isFinite && $0 > 0 }
            let gain = usable ? exp(gains.map { log($0) }.reduce(0, +) / Double(gains.count)) : 0
            let on = usable && gain > 1 + margin && gains.min()! > 1 - margin
            return Decision(on: on, passed: passed, gain: gain, gains: usable ? gains : [])
        }
        /// Per size: Σ flops / rate of the reference over that of the candidate.
        func gains<T>(_ items: [T], size: (T) -> Int, flops: (T) -> Double, reference: (T) -> Measure?,
                      candidate: (T) -> Measure?) -> [Double] {
            heights.compactMap { h in
                let these = items.filter { size($0) == h }
                guard !these.isEmpty else { return nil }
                var (tReference, tCandidate) = (0.0, 0.0)
                for item in these {
                    guard let r = reference(item), let c = candidate(item), r.rate > 0, c.rate > 0 else { return .nan }
                    tReference += flops(item) / r.rate; tCandidate += flops(item) / c.rate
                }
                return tReference / tCandidate
            }
        }

        // ── amx: the co-execution against the GPU alone ──
        let productFlops = { (p: Product) in 2.0 * Double(p.m) * Double(p.k) * Double(p.n) }
        let amxPassed = !products.isEmpty && products.allSatisfy { p in
            p.amx?.pass == true && p.coexecution?.pass == true
        }
        let amx = decision(passed: amxPassed, gains: gains(products, size: \.m, flops: productFlops,
                                                           reference: { $0.gpu }, candidate: \.coexecution))
        var fraction: Double?
        if amx.on {
            // Each engine's total work over its total time: the share that ends both together.
            var (work, tGPU, tAMX) = (0.0, 0.0, 0.0)
            for p in products {
                let f = productFlops(p)
                work += f; tGPU += f / p.gpu.rate; tAMX += f / p.amx!.rate
            }
            let (gpu, cpu) = (work / tGPU, work / tAMX)
            fraction = min(Conductor.fractionRange.upperBound, max(Conductor.fractionRange.lowerBound, gpu / (gpu + cpu)))
        }

        // ── gemm_tensor: the bf16-weight product against the fp32 path it replaces, preparations included ──
        // Seconds, not ratios of rates: a product's flops at TFLOP/s and a weight's values at G values/s
        // add up in one sum. The fp32 path is the one `amx` chose.
        let widen = widening.first { $0.kind == "bf16" }?.cpu, copy = widening.first { $0.kind == "bf16 copy" }?.cpu
        let tensorPassed = !products.isEmpty && widen?.pass == true && copy?.pass == true
            && products.allSatisfy { $0.tensor?.pass == true && (!amx.on || $0.coexecution?.pass == true) }
        let tensorGains: [Double] = heights.compactMap { h in
            let these = products.filter { $0.m == h }
            guard !these.isEmpty else { return nil }
            var (tReference, tCandidate) = (0.0, 0.0)
            for p in these {
                let reference = amx.on ? p.coexecution : p.gpu
                guard let r = reference?.rate, let t = p.tensor?.rate, let w = widen?.rate, let c = copy?.rate,
                      r > 0, t > 0, w > 0, c > 0 else { return .nan }
                let (flops, values) = (productFlops(p) * 1e-12, Double(p.k * p.n) * 1e-9)
                tReference += flops / r + values / w
                tCandidate += flops / t + values / c
            }
            return tReference / tCandidate
        }
        let tensor = decision(passed: tensorPassed, gains: tensorGains)

        // ── gemm_int8: the int8 product against the convrot widening + the fp32 path, preparations included ──
        // The AMX runs beside it when it stays on in the settings (`amx` on, `gemm_tensor` off).
        let amxRuns = amx.on && !tensor.on
        let rotated = widening.first { $0.kind == "int8 convrot" }?.cpu, copy8 = widening.first { $0.kind == "int8 copy" }?.cpu
        let int8Candidate = { (p: Product) in amxRuns ? p.int8Coexecution : p.int8 }
        let int8Passed = !products.isEmpty && rotated?.pass == true && copy8?.pass == true
            && products.allSatisfy { p in p.int8?.pass == true && int8Candidate(p)?.pass == true
                                          && (!amxRuns || p.coexecution?.pass == true) }
        let int8Gains: [Double] = heights.compactMap { h in
            let these = products.filter { $0.m == h }
            guard !these.isEmpty else { return nil }
            var (tReference, tCandidate) = (0.0, 0.0)
            for p in these {
                let reference = amxRuns ? p.coexecution : p.gpu
                guard let r = reference?.rate, let t = int8Candidate(p)?.rate, let w = rotated?.rate, let c = copy8?.rate,
                      r > 0, t > 0, w > 0, c > 0 else { return .nan }
                let (flops, values) = (productFlops(p) * 1e-12, Double(p.k * p.n) * 1e-9)
                tReference += flops / r + values / w
                tCandidate += flops / t + values / c
            }
            return tReference / tCandidate
        }
        let int8 = decision(passed: int8Passed, gains: int8Gains)

        // ── the attention: flash and the head split against MPSGraph ──
        let attentionFlops = { (a: AttentionCapacity) in
            4.0 * Double(a.sequence) * Double(a.sequence) * Double(a.heads) * Double(a.headDim)
        }
        let graphPassed = attention.allSatisfy { $0.graph?.pass == true }
        let flashPassed = !attention.isEmpty && graphPassed && attention.allSatisfy { $0.flash?.pass == true }
        let splitPassed = !attention.isEmpty && graphPassed && attention.allSatisfy { $0.headSplit?.pass == true }
        let split = decision(passed: splitPassed, gains: gains(attention, size: \.sequence, flops: attentionFlops,
                                                               reference: \.graph, candidate: \.headSplit))
        var flash = decision(passed: flashPassed, gains: gains(attention, size: \.sequence, flops: attentionFlops,
                                                               reference: \.graph, candidate: \.flash))
        // Z-Image takes flash before the split: flash only where it is also the faster of the two.
        if flash.on && split.on && flash.gain <= split.gain { flash.on = false }

        // ── amx_shares: ρ = (1 − T) / T at each measured point; `Conductor.Shares` computes every product's T ──
        let shares = amx.on && !cuts.isEmpty
            ? Conductor.Shares(cuts.map { .init(k: $0.k, n: $0.n, m: $0.m, ratio: (1 - $0.share) / $0.share) }).description : nil
        return Decisions(margin: margin, amx: amx, amxFraction: fraction, flash: flash, headSplit: split, tensor: tensor,
                         amxShares: shares, int8: int8)
    }

    /// The rules applied to `report`'s measures (never to its stored `decisions`: the rules live in the code).
    public static func decide(_ report: Report) -> Decisions {
        decide(products: report.products, attention: report.attention, widening: report.widening, witness: report.witness,
               cuts: report.cuts ?? [])
    }
}
