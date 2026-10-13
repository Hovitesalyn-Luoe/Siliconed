import XCTest
@testable import Siliconed

/// **The solvers against diffusers' schedulers**, on Z-Image's and Qwen-Image-2.1's schedules and a synthetic model
/// `m = −0.75·x + b_k`. The final samples below were printed by diffusers 0.41
/// (torch fp32) on that model: `FlowMatchEulerDiscreteScheduler` (and its `stochastic_sampling`),
/// `DPMSolverMultistepScheduler` and `UniPCMultistepScheduler` in `flow_prediction`.
final class SolverTests: XCTestCase {

    /// `(solver, steps, start, the sample after the last moving step)`.
    private let oracle: [(String, Int, Int, [Float])] = [
        ("euler", 6, 0, [-0.658808708190918, 1.560785174369812, -1.8806416988372803, 1.054038166999817, -1.8859535455703735, 0.6621302366256714, -2.559845447540283, 0.35101139545440674]),
        ("euler", 8, 0, [-1.1968371868133545, 1.9685002565383911, -1.8624361753463745, 0.9589888453483582, -1.8594027757644653, 0.14357388019561768, -2.2244551181793213, -0.06988832354545593]),
        ("euler", 9, 0, [-1.1106786727905273, 1.6444131135940552, -1.6489256620407104, 0.8760226368904114, -1.6430608034133911, 0.23633979260921478, -2.5730843544006348, 0.05320030078291893]),
        ("euler", 8, 3, [-0.8601443767547607, 1.4669686555862427, -1.2936415672302246, 0.8381101489067078, -1.3590532541275024, 0.3205619752407074, -2.132797956466675, -0.11293517053127289]),
        ("euler-ancestral", 6, 0, [0.7412613034248352, -0.5007343292236328, -1.0025780200958252, 0.4944396913051605, 1.9272149801254272, -0.8671252131462097, -1.5649948120117188, 0.7538313269615173]),
        ("euler-ancestral", 8, 0, [0.4509100019931793, 1.5684382915496826, -1.6201778650283813, -0.8887099027633667, 0.6046029329299927, 0.27450278401374817, 0.5987105369567871, -1.3275967836380005]),
        ("euler-ancestral", 9, 0, [-0.15103070437908173, 1.3915375471115112, -0.4829846918582916, -0.7958823442459106, 0.09408301115036011, 0.18237313628196716, 1.070834994316101, -1.0317296981811523]),
        ("euler-ancestral", 8, 3, [-1.3336827754974365, -0.4111764430999756, -0.275115430355072, 1.4598032236099243, 0.7372839450836182, -1.7487987279891968, -0.5699385404586792, 0.6639351844787598]),
        ("dpmpp-2m", 6, 0, [-0.5269708037376404, 1.360835313796997, -1.7462971210479736, 0.9547661542892456, -1.7450432777404785, 0.8092877864837646, -2.8166966438293457, 0.5006758570671082]),
        ("dpmpp-2m", 8, 0, [-1.3460988998413086, 2.0367484092712402, -1.9855856895446777, 1.101831316947937, -1.9733024835586548, 0.015894606709480286, -2.2363698482513428, -0.19229890406131744]),
        ("dpmpp-2m", 9, 0, [-1.1924399137496948, 1.8086758852005005, -1.643510103225708, 0.7150586843490601, -1.6277917623519897, 0.17756974697113037, -2.4942166805267334, 0.0006021708250045776]),
        ("dpmpp-2m", 8, 3, [-1.0346959829330444, 1.6660857200622559, -1.1634387969970703, 0.6513292789459229, -1.2223894596099854, 0.16108527779579163, -1.9896736145019531, 0.032342709600925446]),
        ("unipc", 6, 0, [-0.5923221111297607, 1.3537884950637817, -1.6736040115356445, 0.9365320801734924, -1.669541835784912, 0.7504900693893433, -2.848085403442383, 0.5799223184585571]),
        ("unipc", 8, 0, [-1.4677730798721313, 2.1079211235046387, -1.9897242784500122, 1.1405292749404907, -1.975389003753662, -0.100990891456604, -2.1829845905303955, -0.21050290763378143]),
        ("unipc", 9, 0, [-1.2816945314407349, 1.8757015466690063, -1.5837924480438232, 0.6629835963249207, -1.5663479566574097, 0.09234323352575302, -2.4421517848968506, 0.0032588765025138855]),
        ("unipc", 8, 3, [-1.0704576969146729, 1.7093820571899414, -1.135180950164795, 0.6085901856422424, -1.1927217245101929, 0.12861335277557373, -1.9585962295532227, 0.0638900101184845]),
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
    private func walk(_ solver: Solver, steps: Int, start: Int) -> [Float] {
        let walk = SolverWalk(solver: solver, schedule: steps == 0 ? qwen : FlowMatchSchedule(steps: steps).sigmas, start: start,
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

    /// The walk skips Z-Image's final `0 → 0` step, and img2img starts inside the schedule.
    func testTheWalkIsTheMovingSteps() {
        let σ = FlowMatchSchedule(steps: 8).sigmas
        XCTAssertEqual(σ.suffix(2), [0, 0])
        XCTAssertEqual(SolverWalk.effective(σ, start: 0), Array(σ.dropLast()))
        XCTAssertEqual(SolverWalk.effective(σ, start: 3), Array(σ[3..<8]))
    }

    /// Euler's walk is the sampler's own update, `x + Δσ·m`, to the bit.
    func testEulerIsTheSampler() {
        let σ = FlowMatchSchedule(steps: 8).sigmas
        var x = xInit(), y = xInit()
        let walk = SolverWalk(solver: .euler, schedule: σ, start: 0, count: n)
        for k in 0..<7 {
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
