import Accelerate
import Foundation

/// The flow-matching schedule of Z-Image Turbo: **`N` steps are `N` evaluations**.
///
/// diffusers 0.41 (`get_default_z_image_sigmas`, PR #13730) and the official recipe — Tongyi's
/// model card asks the older pipeline for 9 steps, « this actually results in 8 DiT forwards »:
///
///     σᵢ = shift(linspace(1, 1/N, N))ᵢ,  shift(s) = 3s / (1 + 2s),  then a terminal 0
///     N = 8:  1, 0.954545, 0.9, 0.833333, 0.75, 0.642857, 0.5, 0.3, 0
///
/// What a port by analogy would miss:
///
///   - **The 0.36 pipeline was not this.** It forced `scheduler.sigma_min = 0` and walked
///     `linspace(1, 0, N)`: its last two σ were `0, 0`, a final step that moves nothing, and `N` steps
///     were `N − 1` evaluations. The engine carried that until 0.8.2 — 7 evaluations where the
///     model was distilled for 8. Every step now moves the latent; no step is skipped.
///   - **The ramp is torch's `linspace` in float32**, not NumPy's in double: `step = (end − 1)/(N − 1)`
///     in float, `1 + step·i` on the first half and `end − step·(N − 1 − i)` on the second, **each a
///     single rounding** (the CPU kernel fuses them), `end = Float(1/N)` with `1/N` a Python double.
///     Then the shift in float32 (`set_timesteps` casts the list to `np.float32` first).
///   - **Time is inverted**: the DiT receives `(1000 − t)/1000 = 1 − σ`.
///   - **The output is negated**: `noise_pred = −v`.
package struct FlowMatchSchedule {
    package let sigmas: [Float]          // `steps + 1` values, strictly decreasing, ending in zero

    package init(steps: Int, shift: Float = 3.0) {
        let end = Float(1.0 / Double(steps))
        let step = steps > 1 ? (end - 1) / Float(steps - 1) : 0
        var values: [Float] = (0..<steps).map { i in
            let raw = i < steps / 2 ? fmaf(step, Float(i), 1) : fmaf(-step, Float(steps - 1 - i), end)
            return shift * raw / (1 + (shift - 1) * raw)
        }
        values.append(0)
        sigmas = values
    }

    /// What the DiT receives: the inverted time.
    package func modelTime(at step: Int) -> Float { 1 - sigmas[step] }
}

package final class Sampler {
    package let dit: DiT
    package let schedule: FlowMatchSchedule

    /// What each step produced, to compare against the golden trajectory.
    package private(set) var velocities: [[Float]] = []
    package private(set) var latents: [[Float]] = []
    package var recordTrajectory = false

    package init(dit: DiT, steps: Int) {
        self.dit = dit
        self.schedule = FlowMatchSchedule(steps: steps)
    }

    /// **The spectral schedule**: the first `spectralSteps` evaluations run at half-size.
    ///
    ///     x_{k+1} = x_k + Δσ · 𝓘 v_θ(𝓟 x_k, σ_k)
    ///
    /// In rectified flow, the SNR at frequency `f` reaches 1 at `f* = (1−σ)/σ`. On the eight-step
    /// schedule `f*` is [0; 0.048; 0.111; 0.2; 0.333; 0.556; 1.0; 2.333]: **the first four evaluations
    /// carry less than 9 % of the last one's band**, and yet we compute at 1024² to
    /// produce what 512² would carry just as well.
    ///
    /// **The state never goes down** — only the evaluation does. A scheme that downsampled the state
    /// then upsampled it back would pay the interpolation cumulatively; here every step restarts from the
    /// full-resolution latent, and the resampling leaves no trace in what is
    /// kept.
    ///
    /// Measured on the spectral oracle (09-21, on the old seven-evaluation schedule): `k=0` reproduces the reference **to the
    /// bit** — without this reference nothing that follows could be attributed to the trick rather than to the
    /// harness; `k=3` removes **32.1 % of the tokens** and the image stays sharp; `k=4` covers it
    /// with speckles. **The verdict is in the image**: the three splits render 10.83 / 10.91 /
    /// 10.72 dB, a saturated PSNR that no longer separates anything.
    package var spectralSteps = 0

    /// **The size divisor, step by step** — the generalization of `spectralSteps`.
    ///
    /// `spectralSteps = k` says "the first `k` steps at half-size". It is a special case of
    /// `divisors = [2, 2, 1, 1, …]`, and the special case costs a step: **at the first step,
    /// `σ = 1` and `f*(σ) = (1 − σ)/σ` is ZERO** — no frequency reaches an SNR of 1 there, not
    /// even the lowest. Evaluating that step at half-size is still computing four times too much.
    ///
    /// The product's calibrated rule says: half-size as long as `f* ≤ 0.10`. A grid divided by
    /// `d` carries `d` times less band, so the same argument gives **`f* ≤ 0.10/d`** — and on the
    /// eight-step schedule, `f* = [0; 0.048; 0.111; …]` would give `[4, 4, 1, 1, 1, 1, 1, 1]`. A quarter
    /// of a 1024² is 256², below the 512² floor (`Spectral.minimumSide`): `run` refuses it.
    ///
    /// ⚠️ **The quarter has no oracle**, and that must be said: the spectral oracle only measured
    /// factor-of-two descents. The ÷4 path **composes the verified operators twice**
    /// (𝓟 then 𝓟, 𝓘 then 𝓘) — that is not the same as a ÷4 in one pass, and it is a
    /// convention we choose rather than one we read off. Its verdict is therefore in the image, like
    /// that of `k`.
    package var divisors: [Int] = []

    /// **More detail** (`Detail`): the σ the DiT is told, lowered mid-trajectory; the Euler step
    /// keeps the true σ. `normal`: the schedule's own, to the bit.
    package var detail: Detail = .normal
    /// `Detail.scale` of the render's size.
    package var detailScale: Double = 1

    /// **The update rule** (`Solver`): Euler, the distillation's own, unless the user picked another.
    /// The schedule, the steps and what the DiT is told do not change.
    package var solver: Solver = .euler
    /// The ancestral rule's noise: one stream from this seed, step after step — the render's seed,
    /// so the same seed gives the same image.
    package var solverSeed: UInt64 = 0

    /// **After each evaluation** (every step is one): `(absolute step, seconds, evaluated
    /// grid h, w, x, derivative dx/dσ = negated, σₓ)`, the estimate x̂₀ being `x − σₓ·negated`. This
    /// is how the product's module numbers its steps, computes the preview and checks cancellation; it
    /// may throw, and the loop stops. `onStep`, for its part, remains the instrument of the checks.
    ///
    /// Under Euler, `x` is the latent after the update and `σₓ` the next σ — as it always was, so a
    /// sketch keeps its bits. Under another solver the latent after the update is not on the line
    /// `x̂₀ + σ'·m` (the ancestral rule has fresh noise in it, the multistep rules a correction), so
    /// `x` is the walk's own x̂₀ (`SolverWalk.estimate`) and `σₓ` is 0.
    package var afterEvaluation: ((Int, Double, Int, Int, UnsafePointer<Float>, UnsafePointer<Float>, Float) throws -> Void)?

    /// **Can the descent run these divisors on this grid?** Each a power of two, and the largest
    /// dividing both sides — then every level halves exactly. All ones: nothing to divide.
    package static func divides(_ divisors: [Int], height: Int, width: Int) -> Bool {
        let largest = divisors.max() ?? 1
        guard largest > 1 else { return true }
        return divisors.allSatisfy { $0 >= 1 && $0 & ($0 - 1) == 0 }
            && height > 0 && width > 0 && height % largest == 0 && width % largest == 0
    }

    /// The effective divisor of a step: the one that was set, otherwise the one `spectralSteps` implies.
    func divisor(_ step: Int) -> Int {
        if divisors.indices.contains(step) { return max(1, divisors[step]) }
        return step < spectralSteps ? 2 : 1
    }

    /// `latent` as `[C, H, W]`, modified in place. Returns the number of evaluations actually done.
    ///
    /// - Parameter latentHeight, latentWidth: `H` and `W`. Required as soon as `spectralSteps` is not
    ///   zero — without them, we cannot decimate.
    /// - Parameter start: the first step executed of the full calendar — img2img's `t_start`
    ///   (`Strength.startStep`), 0 in txt2img. The calendar does not change; only steps
    ///   `start … N − 1` run, under their absolute number (`modelTime(at:)` included).
    /// - Parameter onStep: `(step, σ, seconds, height and width of the evaluated latent)`.
    @discardableResult
    package func run(latent: UnsafeMutablePointer<Float>, count: Int,
                    caps: UnsafePointer<Float>, latentHeight: Int = 0, latentWidth: Int = 0,
                    start: Int = 0, onStep: ((Int, Float, Double, Int, Int) -> Void)? = nil) throws -> Int {
        let (height, width) = (latentHeight, latentWidth)
        let maxDivisor = (0..<(schedule.sigmas.count - 1)).map { divisor($0) }.max() ?? 1
        guard maxDivisor == 1 || (height > 0 && width > 0 && count % (height * width) == 0) else {
            throw Failure.missingSide
        }
        // **512 px per side is the floor, and it is imposed here, not recommended.** Below a latent
        // of 64, the model is out of its domain: what is evaluated there is neither an image nor a
        // measurement. Per side: the smaller of the two decides.
        guard maxDivisor == 1 || min(height, width) / maxDivisor >= Spectral.minimumSide else {
            throw Failure.belowFloor(side: min(height, width) / maxDivisor)
        }
        // **Every divisor a power of two, dividing both sides.** The descent is `𝓟` applied
        // `log₂ d` times, and its buffers are sized by halving `height` down to `height / maxDivisor`:
        // a divisor of 3 (or 2 on an odd side), possible under `SILICONED_DIVISORS`, would decimate
        // into a level that does not exist, or upsample past the end of `latent`.
        let all = (0..<(schedule.sigmas.count - 1)).map { divisor($0) }
        guard Self.divides(all, height: height, width: width) else {
            throw Failure.indivisible(height: height, width: width, divisor: maxDivisor)
        }
        let channels = height > 0 && width > 0 ? count / (height * width) : 0
        // **One buffer per descent level**, reserved once for all the low
        // evaluations. At 1024² they weigh 4 MB for the half and 1 MB for the quarter — we do not
        // count them, but reallocating them at every step would be eight times the work of the first.
        // **Stable allocations, not Swift arrays.** An `UnsafePointer(array)` taken
        // outside its `withUnsafeBufferPointer` is only valid for the duration of the call — and the descent
        // needs to hold a pointer from one level to the next. Two simultaneous exclusive accesses to
        // two elements of the same array would be one more fault.
        // In `Pages`: unmapped when the descent ends, not left to `free`.
        var levelPages: [Pages] = []
        var levelGrids: [(height: Int, width: Int)] = []
        var grid = (height: height, width: width)
        while grid.height > height / maxDivisor {
            grid = (grid.height / 2, grid.width / 2)
            levelPages.append(Pages(bytes: channels * grid.height * grid.width * 4))
            levelGrids.append(grid)
        }
        let levels = levelPages.map { $0.floats() }
        defer { withExtendedLifetime(levelPages) {} }

        // What the DiT is told: `1 − σ'` (`FlowMatchSchedule.modelTime`) with σ' the detail's σ.
        let told = detail.modelSigmas(schedule.sigmas, start: start, bell: .zImage, scale: detailScale)
        // The update, step by step, on the true σ (`Solver`). Its history lives in `Pages`.
        var stream = Noise(seed: solverSeed ^ 0x5EED_A9CE_57A1_0000)
        let walk = SolverWalk(solver: solver, schedule: schedule.sigmas, start: start, count: count,
                              noise: solver == .eulerAncestral ? { _, e in stream.fill(e, count: count) } : nil)
        var evaluations = 0
        for step in start..<(schedule.sigmas.count - 1) {
            // Between two steps, not in the middle of one: see `Cancellation`.
            try dit.cancellation.check()
            let sigma = schedule.sigmas[step], next = schedule.sigmas[step + 1]
            let started = Date()
            let d = divisor(step)
            let spectral = d > 1
            // The half-size steps are over: their attentions never run again in this render.
            if !spectral, evaluations > 0, divisor(step - 1) > 1 {
                dit.forgetReducedAttentions()
            }
            let (evaluatedHeight, evaluatedWidth) = (height / d, width / d)
            // `noise_pred = −v`, then the solver — Euler: `x ← x + Δσ · noise_pred`, hence `x − Δσ · v`.
            // The negation precedes the upsampling, as in the reference — 𝓘 is linear, so
            // the order has no effect on the bits, but reading it in the same order removes the
            // question.
            var negated: [Float]
            if spectral {
                // **The descent, one factor of two at a time.** `𝓟` only exists as a factor of two —
                // that is the convention read off `torch` and verified to the bit (the resampling
                // check). A ÷4 is therefore `𝓟 ∘ 𝓟`, and the upsampling `𝓘 ∘ 𝓘`: two correct
                // operators composed, and not a third one that would need verifying.
                let depth = Int(log2(Double(d)).rounded())
                var source = UnsafePointer<Float>(latent)
                for level in 0..<depth {
                    Resample.decimate(source, into: levels[level], channels: channels,
                                      height: levelGrids[level].height,
                                      width: levelGrids[level].width)
                    source = UnsafePointer(levels[level])
                }
                // One autorelease pool per evaluation: what MPS autoreleases (descriptors, NDArrays and
                // their buffers) dies with the evaluation, not with the render's queue item.
                var speed = try autoreleasepool {
                    try dit.forward(latent: UnsafePointer(levels[depth - 1]),
                                    caps: caps, sigma: 1 - told[step],
                                    latentHeight: evaluatedHeight, latentWidth: evaluatedWidth)
                }
                var minusOne: Float = -1
                speed.withUnsafeMutableBufferPointer {
                    vDSP_vsmul($0.baseAddress!, 1, &minusOne, $0.baseAddress!, 1,
                               vDSP_Length($0.count))
                }
                // The upsampling, symmetric: from the lowest level up to the full size, one factor
                // of two at a time. `speed` always carries the current level.
                for level in stride(from: depth - 1, through: 0, by: -1) {
                    let source = levelGrids[level]
                    var largest = [Float](repeating: 0,
                                            count: channels * (source.height * 2) * (source.width * 2))
                    speed.withUnsafeBufferPointer { down in
                        largest.withUnsafeMutableBufferPointer { up in
                            Resample.bilinearDouble(down.baseAddress!, into: up.baseAddress!,
                                                    channels: channels, height: source.height,
                                                    width: source.width)
                        }
                    }
                    speed = largest
                }
                negated = speed
                negated.withUnsafeBufferPointer { walk.step(latent: latent, derivative: $0.baseAddress!) }
            } else {
                negated = try autoreleasepool { try dit.forward(latent: latent, caps: caps, sigma: 1 - told[step]) }
                negated.withUnsafeMutableBufferPointer { buffer in
                    var minusOne: Float = -1
                    vDSP_vsmul(buffer.baseAddress!, 1, &minusOne, buffer.baseAddress!, 1, vDSP_Length(count))
                }
                negated.withUnsafeBufferPointer { walk.step(latent: latent, derivative: $0.baseAddress!) }
            }
            evaluations += 1
            // **Where does the trajectory go wrong?** An entirely `NaN` image does not say at which step
            // it became so, and there are eight. The check costs 262,144 comparisons per
            // step — nothing next to an evaluation — and it turns "the render is broken"
            // into "the velocity of step 3 contains 412 non-finite values". It announces itself (a warning,
            // hence an event during a render) and does not stop: the rest of the trajectory says whether it propagates or
            // resolves, and the two do not have the same cause.
            if let bad = Sampler.countNotFinite(negated, count) {
                Warnings.emit("✗ step \(step): the VELOCITY carries \(bad.count) non-finite values "
                    + "(first at index \(bad.first), evaluation at \(evaluatedWidth)×\(evaluatedHeight))")
            }
            if let bad = Sampler.countNotFinite(UnsafeBufferPointer(start: latent, count: count), count) {
                Warnings.emit("✗ step \(step): the LATENT carries \(bad.count) non-finite values "
                    + "(first at index \(bad.first))")
            }
            if recordTrajectory {
                velocities.append(negated)
                latents.append(Array(UnsafeBufferPointer(start: latent, count: count)))
            }
            let seconds = Date().timeIntervalSince(started)
            if let afterEvaluation {
                try negated.withUnsafeBufferPointer {
                    if let estimate = walk.estimate {
                        try afterEvaluation(step, seconds, evaluatedHeight, evaluatedWidth, estimate, $0.baseAddress!, 0)
                    } else {
                        try afterEvaluation(step, seconds, evaluatedHeight, evaluatedWidth, UnsafePointer(latent),
                                            $0.baseAddress!, next)
                    }
                }
            }
            onStep?(step, sigma, seconds, evaluatedHeight, evaluatedWidth)
        }
        return evaluations
    }

    /// The non-finite values of a buffer, or `nil` if there are none — the common case, and the one
    /// that must cost the least. A single pass, bailing out as soon as we know there are some.
    private static func countNotFinite(_ values: some Collection<Float>, _ count: Int)
            -> (count: Int, first: Int)? {
        var bad = 0, first = -1
        for (i, value) in values.enumerated() where !value.isFinite {
            bad += 1
            if first < 0 { first = i }
        }
        return bad > 0 ? (bad, first) : nil
    }

    package enum Failure: Error, CustomStringConvertible {
        case missingSide
        case belowFloor(side: Int)
        /// A divisor that is not a power of two, or a grid it does not divide (`SILICONED_DIVISORS`):
        /// the descent halves exactly, level by level, or reads past its buffers.
        case indivisible(height: Int, width: Int, divisor: Int)
        package var description: String {
            switch self {
            case let .indivisible(height, width, divisor):
                return "spectral divisor \(divisor) on a latent of \(width)×\(height): a power of two dividing both sides is needed"
            case .missingSide:
                return "the spectral schedule requires the latent grid: `run(..., latentHeight:, latentWidth:)`"
            case .belowFloor(let side):
                return "an evaluation would go down to \(side * 8) px on a side: the floor is "
                    + "\(Spectral.minimumSide * 8) px per side, the model is out of its domain below"
            }
        }
    }
}
