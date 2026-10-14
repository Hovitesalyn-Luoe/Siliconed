import Foundation

/// **More detail** — ComfyUI's Detail Daemon (Jonseed, `detail_daemon_node.py` at `3394e44`, MIT,
/// itself after muerrilla's `sd-webui-detail-daemon`), reduced to one named choice.
///
///     the sampler's σᵢ ──────────────────────────────────────────▶ x += (σᵢ₊₁ − σᵢ)·v     unchanged
///            └─ × (1 − 0.1·mᵢ) ─▶ σ'ᵢ ─▶ DiT(x, σ'ᵢ) ─▶ v
///
///     mᵢ: a bell over the steps,  0 ─ rises (start → mid) ─ amount ─ falls (mid → end) ─ 0
///
/// **The model is told less noise remains than really does**, in the middle of the trajectory only:
/// it removes less, and what it leaves is fine texture. The Euler step keeps the true σ, so the
/// trajectory still lands at σ = 0. Nothing else changes: same evaluations, same cost — the only
/// work added is `steps` multiplications on the CPU.
///
/// What the node does that a port by reading its README would miss:
///
///   - the bell is laid over **the steps the sampler receives** (`len(sigmas) − 1`) — on Z-Image
///     since 0.8.2 every one is an evaluation —, and in img2img over the
///     tail that runs, not the full schedule (ComfyUI hands the sampler the truncated σs);
///   - the indices are `int(round(x·(steps − 1)))` with **Python's `round`, half to even**: at 6
///     steps `mid·5 = 2.5` rounds to 2, and the bell peaks at the third step, not the fourth —
///     on a turbo's 8 steps the node's bell touches 4 evaluations (Z-Image's and the late bell, 2);
///   - `start_values.any()` guards the scaling: a rising edge of a single point stays at **0**, not
///     at `start_offset`;
///   - the multiplier is ×0.1 and ×CFG (`1 − 0.1·m·cfg`, floored at 10⁻⁶). The turbos here run
///     **without CFG**, so `cfg = 1` — the node would read 1 from them too;
///   - the schedule is stored in **float32** (`torch.tensor(…, dtype=float32)`), and σ' is the
///     float32 σ times the factor rounded to float32;
///   - a σ ≤ 0 is passed unchanged (`get_dd_schedule` returns 0 there);
///   - **the bell is in step indices, not in σ**: on a schedule that lingers near σ = 1
///     (Qwen-Image-2.1) the node's default lands where the composition is decided, and changes the picture instead of
///     its texture. Hence one bell per family (`Bell`), hidden like the rest.
///
/// Every denoiser honours it — Z-Image through `Sampler`, Anima and Krea 2
/// through `euler` (their modulation tables tabulated at σ'), Qwen-Image-2.1 through its own
/// sampler (late bell). The preview, the spectral schedule and img2img's start read the true σ.
///
/// **`normal` is the identity, to the bit**: `modelSigmas` returns its argument, and every sampler
/// passes the same σ it passed before. Judged on the products' md5.
public enum Detail: String, Sendable, Hashable, CaseIterable, Codable {
    case normal, more, most

    /// The node's knobs. None of them reaches the app: `more` and `most` fix them (`schedule`).
    package struct Schedule: Equatable, Sendable {
        package var amount: Double
        package var start = 0.2, end = 0.8, bias = 0.5, exponent = 1.0
        package var startOffset = 0.0, endOffset = 0.0, fade = 0.0
        package var smooth = true
        package init(amount: Double, start: Double = 0.2, end: Double = 0.8, bias: Double = 0.5,
                     exponent: Double = 1, startOffset: Double = 0, endOffset: Double = 0,
                     fade: Double = 0, smooth: Bool = true) {
            self.amount = amount; self.start = start; self.end = end; self.bias = bias
            self.exponent = exponent; self.startOffset = startOffset; self.endOffset = endOffset
            self.fade = fade; self.smooth = smooth
        }
    }

    /// **Where a denoiser's bell sits.** The node lays it over step *indices*, so where it falls in
    /// σ depends on the schedule — and one shape fits no two families:
    ///
    ///   - `standard`, the node's defaults (start 0.2, end 0.8, bias 0.5), for the two hidden
    ///     families, unjudged. It was Z-Image's while Z-Image ran 7 evaluations (σ 1, 0.947, 0.882,
    ///     0.8, 0.692, 0.545, 0.333): peak at σ 0.69, clean up to amount 2. On the 8 of the
    ///     official recipe (0.8.2) the same indices fall on σ 0.9 ×¼, 0.833 ×¾, 0.75 ×1, 0.643 ×½,
    ///     and More already moved the pose; at 1.5, the framing.
    ///   - `zImage` (start 0.4, end 0.9), Z-Image's since 0.8.2: σ 0.75 ×½ and **0.643 ×1, nothing
    ///     else**. Between two walls: a lie at σ ≥ 0.75 moves the pose and the framing, a lie at
    ///     σ 0.5 sows specks for little sharpness (start 0.5, end 1, all on 0.643 and 0.5: +10 % at
    ///     amount 3, +10 % at 4, the flat areas +50 %). The node's later bell (start 0.4, end 1,
    ///     σ 0.5 ×½ too) sharpens more per amount but specks at 2.5 already.
    ///   - `late` (start 0.6, end 1.0), for Qwen-Image-2.1. ⚠️ Calibrated before the official Turbo,
    ///     on the 6 steps of the third-party turbo LoRA the family carried until 10/10/2026 (512²: σ = 1,
    ///     0.96, 0.92, 0.84, 0.63, 0.36): the standard bell landed on σ 0.92 and 0.84, where the
    ///     composition is still being decided — amount 0.5 already changed the woman and her clothes,
    ///     1 grew a bird on branches; the late bell touched only the 5th step (σ 0.63) — same picture,
    ///     finer texture; up to 1.3 clean, at 1.6 white crackle lines. On the Turbo's 8 fixed steps
    ///     (1, 0.978, 0.954, 0.927, 0.895, 0.845, 0.705, 0.415) the same bell weighs ½ on the 6th
    ///     (σ 0.845) and 1 on the 7th (σ 0.705): not judged by eye yet.
    package enum Bell: Sendable { case standard, zImage, late }

    /// **The calibrated settings**, by eye at 512², the library prompt, on the two visible families
    /// at their default steps. `more` ≈ +10 % of high-frequency energy (mean |Laplacian| of the
    /// luminance) and the same picture; `most` the strongest that stays clean, a step short of where
    /// artifacts begin. The two hidden families take the standard bell, unjudged.
    ///
    ///   - Z-Image (8 evaluations, seeds 42, 7, 0): More 2 → +9 / +8 / +11 %, same pose;
    ///     Most 2.5 → +11 / +10 / +15 %, clean (seed 42's hand goes to the hip, as Most's did on
    ///     7 evaluations); 3 streaks seed 0's hair blue. Less headroom than on 7 evaluations, where
    ///     Most gave +20 %: the σ that sharpened then sat between two of today's.
    ///   - Qwen-Image-2.1 (seeds 42, 7): judged on the third-party turbo's 6 steps, not yet
    ///     on the Turbo's 8.
    package func schedule(_ bell: Bell = .standard) -> Schedule? {
        switch (self, bell) {
        case (.normal, _): return nil
        case (.more, .standard): return Schedule(amount: 1)
        case (.most, .standard): return Schedule(amount: 2)
        case (.more, .zImage): return Schedule(amount: 2, start: 0.4, end: 0.9)
        case (.most, .zImage): return Schedule(amount: 2.5, start: 0.4, end: 0.9)
        case (.more, .late): return Schedule(amount: 1, start: 0.6, end: 1)
        case (.most, .late): return Schedule(amount: 1.3, start: 0.6, end: 1)
        }
    }

    /// **The amount follows the image's side**: calibrated at 512², it is multiplied by
    /// `512 / √(width·height)` — the same lie paints four times the tokens at 1024².
    ///
    ///   - Z-Image: on 7 evaluations and the node's bell, at 1024² More's amount 1 gave Most's
    ///     +20 % and Most's 2 confetti on skin and walls; scaled, More 0.5 and Most 1 clean. Its own
    ///     bell on the 8 (seeds 42, 7, 0, against Normal without the spectral schedule, as
    ///     Detail runs): More 1 → +7 / +8 / +6 %, Most 1.25 → +9 / +10 / +8 %, same picture,
    ///     clean; 1.5 already flecks seed 42's hair. The rule holds.
    ///   - Qwen-Image-2.1 (late bell, judged on the third-party turbo's 6 steps): at 1024², More's 1
    ///     already strews white flecks, Most's 1.3 everywhere; 0.65 is the last clean amount — exactly
    ///     Most scaled. More is 0.5.
    ///
    /// Never more than at 512².
    package static func scale(width: Int, height: Int) -> Double {
        min(1, 512 / Double(width * height).squareRoot())
    }

    /// **The σ each evaluation hands the model**, for a full schedule (`steps + 1` values, ending in
    /// 0) of which steps `start…` run. `normal`: `sigmas` itself. `scale`: `Detail.scale` of the
    /// render's size (1 at 512²).
    package func modelSigmas(_ sigmas: [Float], start: Int = 0, bell: Bell = .standard, scale: Double = 1) -> [Float] {
        guard var schedule = schedule(bell), sigmas.count - start >= 2 else { return sigmas }
        schedule.amount *= scale
        return Detail.adjusted(sigmas, start: start, schedule)
    }

    /// `make_detail_daemon_schedule`, line for line, in double.
    package static func multipliers(steps: Int, _ s: Schedule) -> [Double] {
        guard steps > 0 else { return [] }
        let start = min(s.start, s.end)
        let mid = start + s.bias * (s.end - start)
        func index(_ x: Double) -> Int { Int((x * Double(steps - 1)).rounded(.toNearestOrEven)) }
        let (startIndex, midIndex, endIndex) = (index(start), index(mid), index(s.end))
        var multipliers = [Double](repeating: 0, count: steps)

        func edge(from a: Double, to b: Double, count: Int, offset: Double) -> [Double] {
            var values = linspace(a, b, count)
            if s.smooth { values = values.map { 0.5 * (1 - cos($0 * Double.pi)) } }
            values = values.map { pow($0, s.exponent) }
            if values.contains(where: { $0 != 0 }) {
                values = values.map { $0 * (s.amount - offset) + offset }
            }
            return values
        }
        let rising = edge(from: 0, to: 1, count: midIndex - startIndex + 1, offset: s.startOffset)
        let falling = edge(from: 1, to: 0, count: endIndex - midIndex + 1, offset: s.endOffset)
        // NumPy's slice assignments, in the node's order (the falling edge overwrites `mid`).
        func assign(_ values: [Double], at lower: Int) {
            for (k, v) in values.enumerated() where lower + k >= 0 && lower + k < steps { multipliers[lower + k] = v }
        }
        assign(rising, at: startIndex)
        assign(falling, at: midIndex)
        for i in 0..<min(max(startIndex, 0), steps) { multipliers[i] = s.startOffset }
        if endIndex + 1 < steps { for i in (endIndex + 1)..<steps { multipliers[i] = s.endOffset } }
        return multipliers.map { $0 * (1 - s.fade) }
    }

    /// `np.linspace(a, b, n)`: `i·step + a`, the last point set to `b`.
    private static func linspace(_ a: Double, _ b: Double, _ n: Int) -> [Double] {
        guard n > 1 else { return n == 1 ? [a] : [] }
        let step = (b - a) / Double(n - 1)
        var y = (0..<n).map { Double($0) * step + a }
        y[n - 1] = b
        return y
    }

    /// The sampler wrapper: the schedule over the steps that run, in float32, then
    /// `σ·max(10⁻⁶, 1 − 0.1·m·cfg)` for each σ > 0. Steps before `start` are left as they are.
    package static func adjusted(_ sigmas: [Float], start: Int, _ s: Schedule, cfgScale: Double = 1) -> [Float] {
        let m = multipliers(steps: sigmas.count - 1 - start, s).map { Float($0) }
        var out = sigmas
        for (k, multiplier) in m.enumerated() where sigmas[start + k] > 0 {
            let factor = Float(max(1e-6, 1 - Double(multiplier) * 0.1 * cfgScale))
            out[start + k] = sigmas[start + k] * factor
        }
        return out
    }
}
