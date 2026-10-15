import XCTest
@testable import Siliconed

/// **The solvers against diffusers' schedulers**, on Z-Image's, Qwen-Image-2.1's and Anima's schedules and a synthetic model
/// `m = −0.75·x + b_k`. The final samples below were printed by diffusers 0.41.0 (`oracle_solvers.py`)
/// (torch fp32) on that model: `FlowMatchEulerDiscreteScheduler` (and its `stochastic_sampling`),
/// `DPMSolverMultistepScheduler` and `UniPCMultistepScheduler` in `flow_prediction`.
final class SolverTests: XCTestCase {

    /// `(solver, steps, start, the sample after the last step)`.
    private let oracle: [(String, Int, Int, [Float])] = [
        ("euler", 6, 0, [-1.1033508777618408, 1.7132935523986816, -1.9699612855911255, 1.333426594734192, -1.970532774925232, 0.22864893078804016, -2.4484200477600098, -0.02536393702030182]),
        ("euler", 8, 0, [-1.1106786727905273, 1.6444131135940552, -1.6489256620407104, 0.8760226368904114, -1.6430608034133911, 0.23633979260921478, -2.5730843544006348, 0.05320030078291893]),
        ("euler", 9, 0, [-0.9208986759185791, 1.56882643699646, -1.37153160572052, 0.9550409317016602, -1.942929744720459, 0.4314452111721039, -2.6684513092041016, 0.2718547582626343]),
        ("euler", 8, 3, [-0.6616952419281006, 1.4369295835494995, -1.6500006914138794, 0.9568566083908081, -1.7024173736572266, 0.5493326187133789, -2.275459051132202, 0.1985270380973816]),
        ("euler-ancestral", 6, 0, [0.9222689867019653, 0.6808022260665894, -2.111231803894043, 0.11638568341732025, 1.2595845460891724, 0.006280690431594849, -0.7522937655448914, -0.23731088638305664]),
        ("euler-ancestral", 8, 0, [-0.15103070437908173, 1.3915375471115112, -0.4829846918582916, -0.7958823442459106, 0.09408301115036011, 0.18237313628196716, 1.070834994316101, -1.0317296981811523]),
        ("euler-ancestral", 9, 0, [-0.6805782914161682, 0.7346886992454529, 0.5830985307693481, -0.09964106231927872, -0.8737106919288635, -0.05708365887403488, 0.9349198937416077, -0.05678915977478027]),
        ("euler-ancestral", 8, 3, [-0.1439712643623352, -0.3592674732208252, -0.7892913818359375, 0.9594094157218933, 1.3405905961990356, -1.1495860815048218, -1.1880152225494385, 0.7585704922676086]),
        ("dpmpp-2m", 6, 0, [-0.9123567342758179, 1.6385133266448975, -2.179819107055664, 1.3789942264556885, -2.1721620559692383, 0.43884313106536865, -2.594515323638916, 0.1888832449913025]),
        ("dpmpp-2m", 8, 0, [-1.1924399137496948, 1.8086758852005005, -1.643510103225708, 0.7150586843490601, -1.6277917623519897, 0.17756974697113037, -2.4942166805267334, 0.0006021708250045776]),
        ("dpmpp-2m", 9, 0, [-0.8817951679229736, 1.4450653791427612, -1.277989387512207, 0.8626824617385864, -1.839207649230957, 0.4943021535873413, -2.880439281463623, 0.34153133630752563]),
        ("dpmpp-2m", 8, 3, [-0.6871606111526489, 1.3288705348968506, -1.465782880783081, 0.8607569932937622, -1.5104215145111084, 0.542015790939331, -2.4509267807006836, 0.4008936285972595]),
        ("unipc", 6, 0, [-0.955305814743042, 1.652481198310852, -2.206585645675659, 1.4174067974090576, -2.1964921951293945, 0.40157854557037354, -2.6016600131988525, 0.26472049951553345]),
        ("unipc", 8, 0, [-1.2816945314407349, 1.8757015466690063, -1.5837924480438232, 0.6629835963249207, -1.5663479566574097, 0.09234323352575302, -2.4421517848968506, 0.0032588765025138855]),
        ("unipc", 9, 0, [-0.8977227210998535, 1.4219862222671509, -1.219353437423706, 0.829765260219574, -1.7791211605072021, 0.4817572236061096, -2.916081666946411, 0.4080297350883484]),
        ("unipc", 8, 3, [-0.657537579536438, 1.2877357006072998, -1.440201997756958, 0.8403950333595276, -1.48361074924469, 0.5745089650154114, -2.5027225017547607, 0.42934462428092957]),
        // Qwen-Image-2.1 Turbo's published schedule (`steps` 0).
        ("euler", 0, 0, [-1.2069569826126099, 1.617618441581726, -1.4886418581008911, 0.6948248147964478, -1.4939619302749634, 0.11396308243274689, -2.502941131591797, 0.019483402371406555]),
        ("euler-ancestral", 0, 0, [-0.7163186073303223, 1.3111557960510254, 0.5710384249687195, -0.9948921203613281, -0.3675674498081207, 0.04091916233301163, 1.7205355167388916, -1.1122753620147705]),
        ("dpmpp-2m", 0, 0, [-1.5139200687408447, 1.938751459121704, -1.3454623222351074, 0.4451591372489929, -1.3419554233551025, -0.17240361869335175, -2.2583086490631104, -0.2596471309661865]),
        ("unipc", 0, 0, [-1.49125075340271, 1.9566712379455566, -1.3006551265716553, 0.35510727763175964, -1.2963987588882446, -0.14798621833324432, -2.2468819618225098, -0.1913665384054184]),
    ]

    private let n = 8
    /// Qwen-Image-2.1 Turbo's `sample_sigmas`, then 0.
    private let qwen: [Float] = [1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568, 0]
    private func xInit() -> [Float] { (0..<n).map { Float((($0 * 5 + 3) % 11) - 5) / 4 } }
    private func bias(_ k: Int) -> [Float] { (0..<n).map { Float((($0 * 7 + k * 13 + 1) % 17) - 8) / 8 } }
    private func eps(_ k: Int) -> [Float] { (0..<n).map { Float((($0 * 3 + k * 11 + 2) % 13) - 6) / 4 } }

    /// The walk as the sampler drives it: the model sees the sample, then the solver steps.
    private func walk(_ solver: Solver, steps: Int, start: Int, schedule: [Float]? = nil) -> [Float] {
        let walk = SolverWalk(solver: solver, schedule: schedule ?? (steps == 0 ? qwen : FlowMatchSchedule(steps: steps).sigmas), start: start,
                              count: n, noise: { k, e in for (i, v) in self.eps(k).enumerated() { e[i] = v } })
        var x = xInit()
        for k in 0..<(walk.sigmas.count - 1) {
            let b = bias(k)
            let m = (0..<n).map { -0.75 * x[$0] + b[$0] }
            x.withUnsafeMutableBufferPointer { xs in
                m.withUnsafeBufferPointer { walk.step(latent: xs.baseAddress!, derivative: $0.baseAddress!) }
            }
        }
        return x
    }

    func testTheSolversAreDiffusers() {
        for (name, steps, start, want) in oracle {
            let solver = Solver(rawValue: name)!
            let got = walk(solver, steps: steps, start: start)
            for (i, (g, w)) in zip(got, want).enumerated() {
                XCTAssertEqual(g, w, accuracy: 2e-6 * max(1, abs(w)), "\(name), \(steps) steps from \(start), value \(i)")
            }
        }
    }

    /// **Anima Turbo's schedule**: its pipeline hands `np.linspace(1, 1/N, N)` (float64) to the
    /// published scheduler, where Z-Image's hands a torch float32 ramp — and the engine computes Anima's
    /// in double (`AnimaDiT.sigmas`). At 8 steps diffusers' four schedulers land on the same samples as
    /// on Z-Image's σ (`oracle_solvers.py --anima` prints the rows above for 8 steps from 0 and from 3,
    /// to the last digit): the walk on Anima's own σ must too.
    func testTheSolversAreDiffusersOnAnimasSchedule() {
        for (name, steps, start, want) in oracle where steps == 8 {
            let got = walk(Solver(rawValue: name)!, steps: steps, start: start, schedule: AnimaDiT.sigmas(steps: steps))
            for (i, (g, w)) in zip(got, want).enumerated() {
                XCTAssertEqual(g, w, accuracy: 2e-6 * max(1, abs(w)), "Anima, \(name), from \(start), value \(i)")
            }
        }
    }

    /// The walk is the schedule's tail: every Z-Image step moves (no final `0 → 0` since 0.8.2), and
    /// img2img starts inside the schedule.
    func testTheWalkIsTheScheduleTail() {
        let σ = FlowMatchSchedule(steps: 8).sigmas
        XCTAssertEqual(σ.count, 9)
        XCTAssertEqual(σ.last, 0)
        XCTAssertNotEqual(σ[7], 0)
        XCTAssertEqual(SolverWalk(solver: .euler, schedule: σ, start: 0, count: n).sigmas, σ)
        XCTAssertEqual(SolverWalk(solver: .euler, schedule: σ, start: 3, count: n).sigmas, Array(σ[3...]))
    }

    /// Euler's walk is the sampler's own update, `x + Δσ·m`, to the bit.
    func testEulerIsTheSampler() {
        let σ = FlowMatchSchedule(steps: 8).sigmas
        var x = xInit(), y = xInit()
        let walk = SolverWalk(solver: .euler, schedule: σ, start: 0, count: n)
        for k in 0..<8 {
            let m = bias(k)
            x.withUnsafeMutableBufferPointer { xs in
                m.withUnsafeBufferPointer { walk.step(latent: xs.baseAddress!, derivative: $0.baseAddress!) }
            }
            for i in 0..<n { y[i] = fmaf(σ[k + 1] - σ[k], m[i], y[i]) }   // `vDSP_vsma` fuses
        }
        XCTAssertEqual(x, y)
    }

    /// The estimate a preview and a sketch show: `x − σ·m` on the sample the DiT saw, for every rule
    /// but Euler (whose `x' − σ'·m` is it already).
    func testTheEstimateIsTheLastLanding() {
        let σ = FlowMatchSchedule(steps: 8).sigmas
        for solver in Solver.allCases {
            let walk = SolverWalk(solver: solver, schedule: σ, start: 0, count: n, noise: { k, e in
                for (i, v) in self.eps(k).enumerated() { e[i] = v } })
            XCTAssertEqual(walk.estimate == nil, solver == .euler)
            var x = xInit()
            for k in 0..<3 {
                let seen = x, m = bias(k)
                x.withUnsafeMutableBufferPointer { xs in
                    m.withUnsafeBufferPointer { walk.step(latent: xs.baseAddress!, derivative: $0.baseAddress!) }
                }
                guard let estimate = walk.estimate else { continue }
                let s = solver == .unipc && k == 0 ? σ[0] - 1e-6 : σ[k]
                for i in 0..<n { XCTAssertEqual(estimate[i], seen[i] - s * m[i], accuracy: 1e-6, "\(solver), step \(k)") }
            }
        }
    }
}
