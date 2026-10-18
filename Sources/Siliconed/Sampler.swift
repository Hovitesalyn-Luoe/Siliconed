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
///   - **Time is inverted**: the DiT receives `(1000 − t)/1000` with `t = σ·1000`, **in float32,
///     operation by operation** (`modelTime`) — not `1 − σ`, which is exact where the pipeline rounds
///     twice: at σ₁ = 0.954545 the two are 8 ulps apart, at σ₂ 3 and σ₅ 1 (until 18/10 the engine took
///     `1 − σ`).
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
    package func modelTime(at step: Int) -> Float { Self.modelTime(sigma: sigmas[step]) }

    /// `timestep = (1000 − t) / 1000`, `t = σ · 1000` (`scheduler.timesteps`), each operation a float32
    /// rounding as torch does them — `ZImagePipeline.__call__`, and its img2img's.
    package static func modelTime(sigma: Float) -> Float {
        let t = sigma * 1000
        return (1000 - t) / 1000
    }
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

    /// `latent` as `[C, H, W]`, modified in place. Returns the number of evaluations actually done.
    ///
    /// - Parameter latentHeight, latentWidth: `H` and `W`, the grid every step evaluates — handed to
    ///   `afterEvaluation` (the preview reads it); 0 when nobody reads it.
    /// - Parameter start: the first step executed of the full schedule — img2img's `t_start`
    ///   (`Strength.startStep`), 0 in txt2img. The schedule does not change; only steps
    ///   `start … N − 1` run, under their absolute number (`modelTime(at:)` included).
    /// - Parameter onStep: `(step, σ, seconds)`.
    @discardableResult
    package func run(latent: UnsafeMutablePointer<Float>, count: Int,
                    caps: UnsafePointer<Float>, latentHeight: Int = 0, latentWidth: Int = 0,
                    start: Int = 0, onStep: ((Int, Float, Double) -> Void)? = nil) throws -> Int {
        let (height, width) = (latentHeight, latentWidth)
        // What the DiT is told: `(1000 − 1000σ')/1000` (`FlowMatchSchedule.modelTime`) with σ' the detail's σ.
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
            // `noise_pred = −v`, then the solver — Euler: `x ← x + Δσ · noise_pred`, hence `x − Δσ · v`.
            // One autorelease pool per evaluation: what MPS autoreleases (descriptors, NDArrays and
            // their buffers) dies with the evaluation, not with the render's queue item.
            // Under DPM++ and UniPC, `(1000 − k)/1000` with diffusers' integer timestep (`Solver.integerTimestep`).
            let time = solver.integerTimestep(told[step], first: step == start).map { Float(1000 - $0) / 1000 }
                ?? FlowMatchSchedule.modelTime(sigma: told[step])
            var negated = try autoreleasepool { try dit.forward(latent: latent, caps: caps, sigma: time) }
            negated.withUnsafeMutableBufferPointer { buffer in
                var minusOne: Float = -1
                vDSP_vsmul(buffer.baseAddress!, 1, &minusOne, buffer.baseAddress!, 1, vDSP_Length(count))
            }
            negated.withUnsafeBufferPointer { walk.step(latent: latent, derivative: $0.baseAddress!) }
            evaluations += 1
            // **Where does the trajectory go wrong?** An entirely `NaN` image does not say at which step
            // it became so, and there are eight. The check costs 262,144 comparisons per
            // step — nothing next to an evaluation — and it turns "the render is broken"
            // into "the velocity of step 3 contains 412 non-finite values". It announces itself (a
            // warning, hence an event during a render) and does not stop: the rest of the trajectory
            // says whether it propagates or resolves, and the two do not have the same cause.
            if let bad = Sampler.countNotFinite(negated, count) {
                Warnings.emit("✗ step \(step): the VELOCITY carries \(bad.count) non-finite values "
                    + "(first at index \(bad.first), evaluation at \(width)×\(height))")
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
                        try afterEvaluation(step, seconds, height, width, estimate, $0.baseAddress!, 0)
                    } else {
                        try afterEvaluation(step, seconds, height, width, UnsafePointer(latent),
                                            $0.baseAddress!, next)
                    }
                }
            }
            onStep?(step, sigma, seconds)
        }
        return evaluations
    }

    /// The non-finite values of a buffer (how many, and the first one's index), or `nil` if there are
    /// none — the common case. A single pass.
    private static func countNotFinite(_ values: some Collection<Float>, _ count: Int)
            -> (count: Int, first: Int)? {
        var bad = 0, first = -1
        for (i, value) in values.enumerated() where !value.isFinite {
            bad += 1
            if first < 0 { first = i }
        }
        return bad > 0 ? (bad, first) : nil
    }
}
