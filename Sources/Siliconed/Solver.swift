import Accelerate
import Foundation

/// **The update rule of a step**: what the sampler does with the DiT's derivative once it has it.
/// The σ schedule is the model's own and does not change — only how a step walks it.
///
///     euler            x' = x + (σ' − σ)·m                                    (the distillation's own)
///     euler-ancestral  x' = (1 − σ')·x̂₀ + σ'·ε,         x̂₀ = x − σ·m       (fresh noise every step)
///     dpmpp-2m         λ = log(1 − σ) − log σ,  x' = (σ'/σ)·x − (1 − σ')(e^−h − 1)·(D₀ + ½D₁)
///     unipc            UniP-bh2 predictor + UniC corrector, order 2
///
/// `m = dx/dσ` is the engine's `negated` (`−v`), diffusers' `model_output`. Each rule is diffusers'
/// scheduler read line by line (`FlowMatchEulerDiscreteScheduler`, its `stochastic_sampling`,
/// `DPMSolverMultistepScheduler` and `UniPCMultistepScheduler`, both in `flow_prediction` with
/// `use_flow_sigmas`), checked against them on a synthetic model (`SolverTests`).
///
/// What a port by analogy would miss:
///
///   - **The solver walks only the steps that move.** Z-Image's schedule ends `…, 0, 0`; the last
///     step is never evaluated (`Sampler`), and a multistep rule there would take `log 0 − log 0`.
///     The orders, the warm-up and the « last step » are counted on the moving steps.
///   - **σ = 1 makes `λ = −∞`.** DPM++ lives with it: `h₀ = +∞`, `r₀ = +∞`, `D₁ = 0`, so its first
///     second-order step is a first-order one — IEEE arithmetic, not a special case. UniPC does not
///     (its corrector solves a system with a `−∞` in it): diffusers moves σ = 1 to `1 − 10⁻⁶` in
///     `set_timesteps`, **for the solver's coefficients only** — the DiT is still told σ.
///   - **UniPC corrects after the evaluation, not before**: the DiT sees the predictor's sample,
///     `x̂₀` is computed on that same sample, and only then is the sample corrected — the corrected
///     one feeds the next predictor, never the DiT.
///   - **Both multistep rules drop to first order at the last step** (`lower_order_final`,
///     `final_sigmas_type = "zero"`): σ' = 0 makes `λ' = +∞`, and the step lands on `x̂₀`, like Euler.
///   - **The ancestral rule renoises entirely** (diffusers' `stochastic_sampling`, not k-diffusion's
///     `eta`): the whole `σ'` part of the next sample is fresh noise.
public enum Solver: String, CaseIterable, Sendable, Codable {
    case euler
    case eulerAncestral = "euler-ancestral"
    case dpmpp2m = "dpmpp-2m"
    case unipc

    /// How many latents a render under this rule keeps besides its own: the previous `x̂₀` for both
    /// multistep rules; for UniPC also the predictor's sample, its `D₁` (the corrector's) and the
    /// current `x̂₀`; one noise draw and its `x̂₀` for the ancestral one.
    package var history: Int {
        switch self {
        case .euler: return 0
        case .dpmpp2m: return 1
        case .eulerAncestral: return 2
        case .unipc: return 4
        }
    }

    /// The bytes of `history` for a latent of `count` floats — the sampler's term of `MemoryNeed`.
    package func historyBytes(count: Int) -> Int { history * Arena.capacity(for: count * 4) }
}

/// One render's walk under a `Solver`: the σ it walks and the latents it keeps between two steps.
package final class SolverWalk {
    package let solver: Solver
    /// The moving σ from the first step run, ending in 0 — `effective(_:start:)`.
    package let sigmas: [Float]
    let count: Int
    /// `ε` for the ancestral rule: `(moving step, buffer of count)`.
    let noise: ((Int, UnsafeMutablePointer<Float>) -> Void)?

    // UniPC's σ, σ = 1 moved off (see `Solver`). The other rules read `sigmas`.
    private let solverSigmas: [Float]
    private var pages: [Pages] = []
    /// The previous step's `x̂₀` (multistep), or the noise draw (ancestral).
    private var previous: UnsafeMutablePointer<Float>?
    /// The ancestral rule's `x̂₀`.
    private var landing: UnsafeMutablePointer<Float>?

    /// **The last step's estimate x̂₀ = x − σ·m**, on the sample the DiT saw — what a preview and a
    /// sketch show. `nil` under Euler, whose `x' − σ'·m` is the same estimate (`Sampler.afterEvaluation`).
    package var estimate: UnsafePointer<Float>? {
        switch solver {
        case .euler: return nil
        case .eulerAncestral: return landing.map { UnsafePointer($0) }
        case .dpmpp2m, .unipc: return previous.map { UnsafePointer($0) }
        }
    }
    /// UniPC: the sample the last predictor started from, its `D₁`, the current `x̂₀`.
    private var lastSample: UnsafeMutablePointer<Float>?
    private var lastD1: UnsafeMutablePointer<Float>?
    private var current: UnsafeMutablePointer<Float>?
    private var lastOrder = 0
    private var done = 0

    /// - Parameter schedule: the full schedule, `steps + 1` values (`FlowMatchSchedule.sigmas`).
    /// - Parameter start: the first step run (img2img's `t_start`).
    package init(solver: Solver, schedule: [Float], start: Int, count: Int,
                 noise: ((Int, UnsafeMutablePointer<Float>) -> Void)? = nil) {
        self.solver = solver
        self.count = count
        self.noise = noise
        sigmas = Self.effective(schedule, start: start)
        var nudged = sigmas
        if solver == .unipc, let first = nudged.first, abs(first - 1) < 1e-6 { nudged[0] = first - 1e-6 }
        solverSigmas = nudged
        if solver.history > 0 {
            pages = (0..<solver.history).map { _ in Pages(bytes: count * 4) }
            previous = pages[0].floats()
            if solver == .eulerAncestral { landing = pages[1].floats() }
            if solver == .unipc { (lastSample, lastD1, current) = (pages[1].floats(), pages[2].floats(), pages[3].floats()) }
        }
    }

    /// The σ a walk visits: `schedule[start]`, then every next value that differs from the one before.
    package static func effective(_ schedule: [Float], start: Int) -> [Float] {
        var out = [schedule[start]]
        for k in start..<(schedule.count - 1) where schedule[k + 1] != schedule[k] { out.append(schedule[k + 1]) }
        return out
    }

    /// `λ = log α − log σ`, `α = 1 − σ`, in float like the reference's 0-d tensors.
    private static func lambda(_ sigma: Float) -> Float { logf(1 - sigma) - logf(sigma) }

    /// One moving step: `latent` (the sample the DiT just saw) becomes the next sample, from the
    /// derivative `m = dx/dσ` the DiT returned on it.
    package func step(latent x: UnsafeMutablePointer<Float>, derivative m: UnsafePointer<Float>) {
        let k = done
        precondition(k < sigmas.count - 1, "the walk has no step left")
        defer { done += 1 }
        let (s, t) = (sigmas[k], sigmas[k + 1])
        let n = count
        switch solver {
        case .euler:
            var delta = t - s
            vDSP_vsma(m, 1, &delta, x, 1, x, 1, vDSP_Length(n))

        case .eulerAncestral:
            // `x0 = sample − σ·m; prev = (1 − σ')·x0 + σ'·noise`, the noise drawn every step, even the
            // last (where σ' = 0 throws it away), as `randn_tensor` is.
            let e = previous!, x0 = landing!
            noise?(k, e)
            for i in 0..<n {
                x0[i] = x[i] - s * m[i]
                x[i] = (1 - t) * x0[i] + t * e[i]
            }

        case .dpmpp2m:
            // `convert_model_output`: x̂₀ = x − σ·m. First and last steps first order.
            let prior = previous!
            let (alphaT, lambdaT, lambdaS) = (1 - t, Self.lambda(t), Self.lambda(s))
            let h = lambdaT - lambdaS
            let ratio = t / s, c = alphaT * (expf(-h) - 1)
            if k == 0 || k == sigmas.count - 2 {
                for i in 0..<n {
                    let d0 = x[i] - s * m[i]
                    x[i] = ratio * x[i] - c * d0
                    prior[i] = d0
                }
            } else {
                // `r₀ = h₀/h`, `D₁ = (1/r₀)(m₀ − m₁)`, midpoint: `− c·D₀ − ½c·D₁`.
                let h0 = lambdaS - Self.lambda(sigmas[k - 1])
                let inverse = 1 / (h0 / h), half = 0.5 * c
                for i in 0..<n {
                    let d0 = x[i] - s * m[i]
                    let d1 = inverse * (d0 - prior[i])
                    x[i] = ratio * x[i] - c * d0 - half * d1
                    prior[i] = d0
                }
            }

        case .unipc:
            stepUniPC(k, x, m)
        }
    }

    /// `UniPCMultistepScheduler.step`, `predict_x0`, `bh2`, `solver_order = 2`, `lower_order_final`.
    private func stepUniPC(_ k: Int, _ x: UnsafeMutablePointer<Float>, _ m: UnsafePointer<Float>) {
        let sig = solverSigmas, n = count
        let prior = previous!, last = lastSample!, older = lastD1!, x0 = current!
        // `x̂₀` on the sample the DiT saw, before any correction.
        for i in 0..<n { x0[i] = x[i] - sig[k] * m[i] }

        // ── UniC: correct the sample the last predictor produced, now that its x̂₀ is known ──
        if k > 0 {
            let (sT, sS0) = (sig[k], sig[k - 1])
            let h = Self.lambda(sT) - Self.lambda(sS0)
            let hh = -h, phi1 = expm1f(hh), bh = expm1f(hh)
            let b1 = (phi1 / hh - 1) / bh
            var rho: (r: Float, t: Float)
            if lastOrder == 1 {
                rho = (0, 0.5)
            } else {
                let rk = (Self.lambda(sig[k - 2]) - Self.lambda(sS0)) / h
                let b2 = ((phi1 / hh - 1) / hh - 0.5) * 2 / bh
                // `linalg.solve([[1, 1], [rk, 1]], [b1, b2])`.
                let first = (b1 - b2) / (1 - rk)
                rho = (first, b1 - first)
            }
            let (ratio, c) = (sT / sS0, (1 - sT) * phi1), cb = (1 - sT) * bh
            for i in 0..<n {
                // `m0` = the previous x̂₀, `D1_t = model_t − m0`; order 2 adds `(m₋₂ − m0)/rk` — the
                // previous predictor's own `D₁`: same σ, same `h`, same `rk`.
                var correction = rho.t * (x0[i] - prior[i])
                if lastOrder == 2 { correction += rho.r * older[i] }
                x[i] = ratio * last[i] - c * prior[i] - cb * correction
            }
        }

        // ── UniP: predict the next sample from the (corrected) one ─────────────────────────────
        let order = min(2, sig.count - 1 - k, done + 1)
        lastOrder = order
        for i in 0..<n { last[i] = x[i] }
        let (sT, sS0) = (sig[k + 1], sig[k])
        let h = Self.lambda(sT) - Self.lambda(sS0)
        let hh = -h, phi1 = expm1f(hh), bh = expm1f(hh)
        let (ratio, c, cb) = (sT / sS0, (1 - sT) * phi1, (1 - sT) * bh)
        if order == 1 {
            for i in 0..<n { x[i] = ratio * x[i] - c * x0[i] }
        } else {
            let rk = (Self.lambda(sig[k - 1]) - Self.lambda(sS0)) / h
            for i in 0..<n {
                let d1 = (prior[i] - x0[i]) / rk
                older[i] = d1
                x[i] = ratio * x[i] - c * x0[i] - cb * (0.5 * d1)
            }
        }
        for i in 0..<n { prior[i] = x0[i] }
    }
}
