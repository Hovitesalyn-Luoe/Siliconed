import Foundation
import XCTest
@testable import Siliconed

/// **The rules that turn a capacity map into the engine's branches** (`CapacityRules.swift`), on synthetic
/// maps: what they choose, against which margin, and when a map is stale. No GPU, no weights.
final class CapacityRulesTests: XCTestCase {
    /// The rates of one map: per size (1024, 4096) and per model (Z-Image, Qwen-Image-2.1), up then down.
    private struct Rates {
        var gpu: [[Double]]                  // [size][product]: z up, z down, qwen up, qwen down
        var amx: [[Double]]
        var both: [[Double]]
        var graph: [[Double]]                // [size][model]
        var flash: [[Double]]
        var split: [[Double]]
        var drift = (gpu: 1.0, amx: 1.0)
        var amxError = 8e-7
        /// `TensorGEMM` on the bf16 weights, [size][product]; `nil`: no Metal 4.
        var tensor: [[Double]]? = nil
        var tensorError = 6e-7
        /// The CPU's bf16 widening and bf16 copy, G values/s (the M1 Pro's: 21.3 and ~40).
        var widen = 21.3, copy = 40.0
        /// The int8 products, [size][product]: alone, beside the AMX by columns; `nil`: no Metal 4.
        var int8: [[Double]]? = nil, int8Both: [[Double]]? = nil
        /// The CPU's convrot widening and int8 copy, G values/s (the M1 Pro's: 11.3 and 77).
        var rotated = 11.3, copy8 = 77.0
        /// The Q8_0 products, likewise; and the CPU's Q8_0 widening and copy.
        var q8: [[Double]]? = nil, q8Both: [[Double]]? = nil
        var q8Widen = 23.0, q8Copy = 70.0
        /// GPU + ANE, [size][product]; `nil`: a map before schema 10. Its error, threshold (bf16's + 25 %) and share.
        var ane: [[Double]]? = nil
        var aneError = 4e-4, aneThreshold = 4e-3, aneShare = 0.55
    }

    private static let machine = Diagnostic.Machine(chip: "Apple M1 Pro", hardwareModel: "MacBookPro18,3", performanceCores: 6,
                                                    efficiencyCores: 2, gpuCores: 14, memoryBytes: 17_179_869_184,
                                                    macOS: "27.0.1 (26A434)", version: "0.9 (90)", build: "0123456789ab")

    private func report(_ r: Rates, machine: Diagnostic.Machine = CapacityRulesTests.machine) -> Diagnostic.Report {
        func ok(_ rate: Double, _ error: Double = 1e-6) -> Diagnostic.Measure { .init(rate: rate, error: error, threshold: 1e-5) }
        var products: [Diagnostic.Product] = []
        for (b, block) in Diagnostic.blocks.enumerated() {
            for (h, m) in Diagnostic.heights.enumerated() {
                for (p, name) in ["up", "down"].enumerated() {
                    let i = 2 * b + p
                    let (k, n) = name == "up" ? (block.dim, block.hidden) : (block.hidden, block.dim)
                    let (g, a) = (r.gpu[h][i], r.amx[h][i])
                    products.append(.init(model: block.model, product: name, m: m, k: k, n: n, gpu: ok(g),
                                          amx: ok(a, r.amxError), gpuShare: g / (g + a), coexecution: ok(r.both[h][i]),
                                          coexecutionGPURows: nil, tensor: r.tensor.map { ok($0[h][i], r.tensorError) },
                                          int8: r.int8.map { ok($0[h][i]) }, int8Coexecution: r.int8Both.map { ok($0[h][i]) },
                                          q8: r.q8.map { ok($0[h][i]) }, q8Coexecution: r.q8Both.map { ok($0[h][i]) },
                                          ane: r.ane.map { .init(rate: $0[h][i], error: r.aneError, threshold: r.aneThreshold) },
                                          aneGPUShare: r.ane.map { _ in r.aneShare }))
                }
            }
        }
        var attention: [Diagnostic.AttentionCapacity] = []
        for (b, block) in Diagnostic.blocks.enumerated() {
            for (h, s) in Diagnostic.heights.enumerated() {
                attention.append(.init(model: block.model, heads: block.heads, headDim: block.headDim, sequence: s,
                                       graph: ok(r.graph[h][b]), flash: ok(r.flash[h][b]), headSplit: ok(r.split[h][b])))
            }
        }
        let conditions = Diagnostic.Conditions(thermal: "nominal", lowPowerMode: false, onBattery: false, swapUsedBytes: 0)
        let first = products[0]
        return Diagnostic.Report(date: "2026-10-14T20:11:08Z", seconds: 16.1, machine: machine, cpuFeatures: [:],
                                 before: conditions, after: conditions, products: products, lowPrecision: [],
                                 attention: attention,
                                 widening: [.init(kind: "bf16", values: 39_321_600, cpu: .init(rate: r.widen, error: 0, threshold: 0)),
                                            .init(kind: "bf16 copy", values: 39_321_600, cpu: .init(rate: r.copy, error: 0, threshold: 0)),
                                            .init(kind: "int8 convrot", values: 39_321_600, cpu: .init(rate: r.rotated, error: 0, threshold: 0)),
                                            .init(kind: "int8 copy", values: 39_321_600, cpu: .init(rate: r.copy8, error: 0, threshold: 0)),
                                            .init(kind: "q8_0", values: 39_321_600, cpu: .init(rate: r.q8Widen, error: 0, threshold: 0)),
                                            .init(kind: "q8_0 copy", values: 39_321_600, cpu: .init(rate: r.q8Copy, error: 0, threshold: 0))],
                                 witness: .init(gpuBefore: first.gpu.rate, gpuAfter: first.gpu.rate * r.drift.gpu,
                                                amxBefore: first.amx?.rate, amxAfter: first.amx.map { $0.rate * r.drift.amx }))
    }

    /// **This M1 Pro's map of 14/10/2026** (on battery, the GPU ×0.84 over the run): co-execution and the
    /// head split on, flash off — today's defaults, the GPU's share 0.614 where the frozen one is 0.631.
    private var m1Pro: Rates {
        Rates(gpu: [[3.471, 2.84, 2.863, 3.518], [2.838, 2.887, 2.898, 2.937]],
              amx: [[2.222, 1.625, 2.192, 1.42], [2.139, 1.723, 2.2, 1.572]],
              both: [[3.516, 3.751, 3.883, 3.652], [4.542, 3.793, 4.581, 3.593]],
              graph: [[1.217, 1.569], [1.234, 1.333]], flash: [[1.274, 1.594], [1.3, 1.335]],
              split: [[1.71, 1.862], [2.014, 2.138]], drift: (2.923 / 3.471, 2.02 / 2.222))
    }

    func testM1ProMapGivesTodaysBranches() {
        let d = Diagnostic.decide(report(m1Pro))
        XCTAssertEqual(d.margin, 0.158, accuracy: 0.001)   // the witness's drift, above the 5 % floor
        XCTAssertTrue(d.amx.on)
        XCTAssertTrue(d.amx.passed)
        XCTAssertEqual(d.amx.gains.count, 2)
        XCTAssertEqual(d.amx.gains[0], 1.18, accuracy: 0.01)
        XCTAssertEqual(d.amx.gains[1], 1.41, accuracy: 0.01)
        XCTAssertEqual(d.amxFraction!, 0.614, accuracy: 0.001)
        XCTAssertFalse(d.flash.on, "a tie with MPSGraph stays on MPSGraph")
        XCTAssertTrue(d.headSplit.on)
        XCTAssertEqual(d.headSplit.gain, 1.45, accuracy: 0.01)
        let settings = d.settings
        XCTAssertEqual(settings["amx"] as? Bool, true)
        XCTAssertEqual(settings["flash"] as? Bool, false)
        XCTAssertEqual(settings["head_split"] as? Bool, true)
        XCTAssertEqual(settings["amx_fraction"] as? Double, d.amxFraction)
        XCTAssertEqual(settings["gemm_tensor"] as? Bool, false)
        XCTAssertNotNil(d.tensor)
        XCTAssertFalse(d.tensor!.passed, "no Metal 4 measure: the bf16 path is not a candidate")
    }

    /// **The M1 Pro with its `matmul2d`** (2.36 TFLOP/s on the bf16 weights, measured 17/10/2026): slower
    /// than MPS + AMX even without the widening — the default path stays.
    func testM1ProKeepsTheWidenedPath() {
        var r = m1Pro
        r.tensor = [[2.36, 2.3, 2.36, 2.3], [2.4, 2.35, 2.4, 2.35]]
        let d = Diagnostic.decide(report(r))
        XCTAssertTrue(d.tensor!.passed)
        XCTAssertFalse(d.tensor!.on)
        XCTAssertLessThan(d.tensor!.gain, 1)
        XCTAssertEqual(d.settings["amx"] as? Bool, true)
        XCTAssertEqual(d.settings["gemm_tensor"] as? Bool, false)
    }

    /// **The M5's map of 17/10/2026**, its `matmul2d` at the bf16 × bf16 rate it measured (6.0):
    /// the bf16 path beats MPS + AMX, takes the products, and turns the AMX off in the settings.
    func testM5TakesTheBf16Path() {
        var r = m1Pro
        r.gpu = [[2.919, 2.831, 2.881, 2.847], [2.951, 2.872, 2.949, 2.835]]
        r.amx = [[1.805, 1.75, 1.812, 1.768], [1.838, 1.829, 1.828, 1.842]]
        r.both = [[3.35, 3.071, 3.353, 3.16], [4.569, 4.259, 4.579, 4.313]]
        r.drift = (2.924 / 2.919, 1.814 / 1.805)
        r.tensor = [[6.0, 5.9, 6.0, 5.9], [6.0, 5.9, 6.0, 5.9]]
        r.widen = 20.07
        r.copy = 35
        let d = Diagnostic.decide(report(r))
        XCTAssertTrue(d.amx.on, "co-execution still measured a gain")
        XCTAssertTrue(d.tensor!.on)
        XCTAssertGreaterThan(d.tensor!.gains.min()!, 1.3)
        let settings = d.settings
        XCTAssertEqual(settings["gemm_tensor"] as? Bool, true)
        XCTAssertEqual(settings["amx"] as? Bool, false, "the bf16 path has no AMX")
        XCTAssertNil(settings["amx_fraction"])
    }

    /// **The preparation counts**: at the GPU's own rate, the bf16 path wins by the widening it skips —
    /// and loses it again when the copy is no faster than the widening.
    func testThePreparationIsInTheComparison() {
        var r = m1Pro
        r.drift = (1.0, 1.0)
        r.amx = r.gpu.map { $0.map { $0 * 0.1 } }              // an AMX that does not pay: `amx` off
        r.both = r.gpu
        r.tensor = r.gpu
        r.widen = 2; r.copy = 20                               // a slow widening: ×1.1–1.4 on the preparations alone
        var d = Diagnostic.decide(report(r))
        XCTAssertFalse(d.amx.on)
        XCTAssertTrue(d.tensor!.on)
        r.copy = r.widen
        d = Diagnostic.decide(report(r))
        XCTAssertEqual(d.tensor!.gain, 1, accuracy: 1e-9)
        XCTAssertFalse(d.tensor!.on)
    }

    /// **This M1 Pro's map of 18/10/2026**: the int8 GPU alone loses to MPS + AMX (2.4 against
    /// ~4.5 TFLOP/s), beside the AMX by columns it wins at 1024 rows (×1.19) and ties at 4096 — on, the
    /// AMX kept; and the same int8 product alone would stay off.
    func testM1ProTakesTheInt8PathBesideTheAMX() {
        var r = m1Pro
        r.drift = (1.0, 1.0)
        r.gpu = [[3.464, 3.509, 3.457, 3.507], [3.507, 3.603, 3.490, 3.599]]
        r.amx = [[2.232, 1.932, 2.242, 1.794], [2.393, 2.009, 2.404, 1.853]]
        r.both = [[3.985, 4.482, 3.990, 4.518], [5.481, 4.926, 5.080, 5.056]]
        r.tensor = [[2.309, 2.316, 2.319, 2.313], [2.334, 2.338, 2.344, 2.330]]
        r.int8 = [[2.4, 2.4, 2.4, 2.4], [2.45, 2.45, 2.45, 2.45]]
        r.int8Both = [[4.734, 4.122, 4.700, 4.018], [5.276, 4.547, 5.378, 4.609]]
        var d = Diagnostic.decide(report(r))
        XCTAssertTrue(d.amx.on)
        XCTAssertFalse(d.tensor!.on)
        XCTAssertTrue(d.int8!.on)
        XCTAssertGreaterThan(d.int8!.gains[0], 1.1)
        XCTAssertEqual(d.settings["gemm_int8"] as? Bool, true)
        XCTAssertEqual(d.settings["amx"] as? Bool, true, "the int8 path keeps the AMX on its columns")
        r.int8Both = nil
        d = Diagnostic.decide(report(r))
        XCTAssertFalse(d.int8!.passed, "the AMX stays on: the int8 path is judged beside it, unmeasured here")
        XCTAssertEqual(d.settings["gemm_int8"] as? Bool, false)
    }

    /// **The Q8_0 product is decided apart from the int8 one** (this M1 Pro, 18/10/2026): beside the AMX it runs
    /// ×0.45–0.67 of the widened path where the int8 one wins — `gemm_q8` off, `gemm_int8` on, from one map.
    func testM1ProKeepsTheQ8PathOffWhereTheInt8OneWins() {
        var r = m1Pro
        r.drift = (1.0, 1.0)
        r.gpu = [[3.464, 3.509, 3.457, 3.507], [3.507, 3.603, 3.490, 3.599]]
        r.amx = [[2.232, 1.932, 2.242, 1.794], [2.393, 2.009, 2.404, 1.853]]
        r.both = [[3.985, 4.482, 3.990, 4.518], [5.481, 4.926, 5.080, 5.056]]
        r.tensor = [[2.309, 2.316, 2.319, 2.313], [2.334, 2.338, 2.344, 2.330]]
        r.int8 = [[2.4, 2.4, 2.4, 2.4], [2.45, 2.45, 2.45, 2.45]]
        r.int8Both = [[4.734, 4.122, 4.700, 4.018], [5.276, 4.547, 5.378, 4.609]]
        r.q8 = [[1.2, 1.2, 1.2, 1.2], [1.25, 1.25, 1.25, 1.25]]
        r.q8Both = [[2.1, 2.0, 2.1, 2.0], [2.4, 2.3, 2.4, 2.3]]
        var d = Diagnostic.decide(report(r))
        XCTAssertTrue(d.int8!.on)
        XCTAssertTrue(d.q8!.passed)
        XCTAssertFalse(d.q8!.on)
        XCTAssertLessThan(d.q8!.gains.max()!, 1)
        XCTAssertEqual(d.settings["gemm_q8"] as? Bool, false)
        XCTAssertEqual(d.settings["gemm_int8"] as? Bool, true)
        // A chip where it wins beside the AMX takes it.
        r.q8Both = r.both.map { $0.map { $0 * 1.4 } }
        d = Diagnostic.decide(report(r))
        XCTAssertTrue(d.q8!.on)
        XCTAssertEqual(d.settings["gemm_q8"] as? Bool, true)
        // Without its widening's rate, never.
        r.q8Widen = 0
        XCTAssertFalse(Diagnostic.decide(report(r)).q8!.on)
    }

    /// **The Neural Engine is a branch the map decides, exclusive with the AMX**: when both clear the margin,
    /// `gemm_ane` on with its share and `amx` off down to 90 % of the AMX's gain (thermal efficiency); below
    /// it the AMX; a product over its threshold, a gain within the margin, or a map without it, keeps it off.
    func testTheNeuralEngineIsDecidedAgainstTheAMX() {
        var r = m1Pro
        r.drift = (1.0, 1.0)
        r.ane = r.gpu.map { $0.map { $0 * 1.7 } }
        var d = Diagnostic.decide(report(r))
        XCTAssertTrue(d.ane!.on)
        XCTAssertTrue(d.amx.passed)
        XCTAssertFalse(d.amx.on, "the larger gain keeps its branch")
        XCTAssertEqual(d.settings["gemm_ane"] as? Bool, true)
        XCTAssertEqual(d.settings["amx"] as? Bool, false)
        XCTAssertEqual(d.settings["ane_fraction"] as? Double, 0.55)
        XCTAssertNil(d.settings["amx_fraction"])
        // The AMX faster, the ANE at 93 % of its rates: the ANE all the same.
        r.ane = r.both.map { $0.map { $0 * 0.93 } }
        d = Diagnostic.decide(report(r))
        XCTAssertLessThan(d.ane!.gain, d.amx.gain)
        XCTAssertTrue(d.ane!.on)
        XCTAssertFalse(d.amx.on)
        XCTAssertEqual(d.settings["amx"] as? Bool, false)
        // At 85 %: the AMX, clearly faster.
        r.ane = r.both.map { $0.map { $0 * 0.85 } }
        d = Diagnostic.decide(report(r))
        XCTAssertTrue(d.ane!.passed)
        XCTAssertFalse(d.ane!.on)
        XCTAssertTrue(d.amx.on)
        // Within the margin of the GPU alone: off, and the AMX keeps the rows.
        r.ane = r.gpu.map { $0.map { $0 * 1.02 } }
        d = Diagnostic.decide(report(r))
        XCTAssertTrue(d.ane!.passed)
        XCTAssertFalse(d.ane!.on)
        XCTAssertTrue(d.amx.on)
        XCTAssertEqual(d.settings["gemm_ane"] as? Bool, false)
        XCTAssertNil(d.settings["ane_fraction"])
        // Faster, but one product over bf16's error + 25 %: off.
        r.ane = r.gpu.map { $0.map { $0 * 1.7 } }
        r.aneError = 5e-3
        d = Diagnostic.decide(report(r))
        XCTAssertFalse(d.ane!.passed)
        XCTAssertFalse(d.ane!.on)
        XCTAssertTrue(d.amx.on)
        // A map before schema 10: no decision, the branch off.
        r.ane = nil
        d = Diagnostic.decide(report(r))
        XCTAssertNil(d.ane)
        XCTAssertEqual(d.settings["gemm_ane"] as? Bool, false)
    }

    /// **Under `gemm_tensor` the AMX is off**: the int8 product is judged alone, against the GPU alone.
    func testInt8IsJudgedAloneWhenTheBf16PathTurnsTheAMXOff() {
        var r = m1Pro
        r.tensor = r.gpu.map { $0.map { $0 * 3 } }
        r.int8 = r.gpu.map { $0.map { $0 * 3 } }
        r.int8Both = r.gpu.map { $0.map { $0 * 0.1 } }        // ignored: no AMX beside it
        let d = Diagnostic.decide(report(r))
        XCTAssertTrue(d.tensor!.on)
        XCTAssertTrue(d.int8!.on)
        XCTAssertEqual(d.settings["amx"] as? Bool, false)
    }

    /// A bf16 path that computed wrong, or a map without the copy's rate, is never taken.
    func testAWrongOrUnmeasuredBf16PathStaysOff() {
        var r = m1Pro
        r.tensor = r.gpu.map { $0.map { $0 * 3 } }
        XCTAssertTrue(Diagnostic.decide(report(r)).tensor!.on)
        r.tensorError = 2e-3
        XCTAssertFalse(Diagnostic.decide(report(r)).tensor!.on)
        r.tensorError = 6e-7
        var noCopy = report(r)
        noCopy.widening.removeAll { $0.kind == "bf16 copy" }
        XCTAssertFalse(Diagnostic.decide(noCopy).tensor!.passed)
        XCTAssertEqual(Diagnostic.decide(noCopy).settings["gemm_tensor"] as? Bool, false)
    }

    /// **An M5-like chip**: the AMX nearly as fast as the GPU (an M5, an M4). The co-execution
    /// nearly doubles the products; the GPU's share falls near one half.
    func testAMXCloseToTheGPUTurnsCoexecutionOn() {
        var r = m1Pro
        r.gpu = [[2.0, 2.0, 2.0, 2.0], [2.1, 2.1, 2.1, 2.1]]
        r.amx = [[1.8, 1.8, 1.8, 1.8], [1.85, 1.85, 1.85, 1.85]]
        r.both = [[3.5, 3.5, 3.5, 3.5], [3.8, 3.8, 3.8, 3.8]]
        r.drift = (0.99, 1.0)
        let d = Diagnostic.decide(report(r))
        XCTAssertEqual(d.margin, Diagnostic.minimumMargin)
        XCTAssertTrue(d.amx.on)
        XCTAssertEqual(d.amxFraction!, 2.05 / (2.05 + 1.825), accuracy: 0.01)
        XCTAssertLessThan(d.amxFraction!, 0.6)
    }

    /// **An engine that computed wrong is never chosen**, however fast.
    func testAFailedAMXCheckKeepsCoexecutionOff() {
        var r = m1Pro
        r.amxError = 3e-3
        let d = Diagnostic.decide(report(r))
        XCTAssertFalse(d.amx.passed)
        XCTAssertFalse(d.amx.on)
        XCTAssertNil(d.amxFraction)
        XCTAssertNil(d.settings["amx_fraction"])
        XCTAssertEqual(d.settings["amx"] as? Bool, false)
        // The same for a wrong co-execution, the AMX alone right.
        var wrong = report(m1Pro)
        wrong.products[3].coexecution = .init(rate: 9, error: 1e-2, threshold: 1e-5)
        XCTAssertFalse(Diagnostic.decide(wrong).amx.on)
        // And a missing engine: BNNS refused one shape.
        var missing = report(m1Pro)
        missing.products[5].amx = nil
        XCTAssertFalse(Diagnostic.decide(missing).amx.on)
        var noSplit = report(m1Pro)
        noSplit.attention[2].headSplit = nil
        XCTAssertFalse(Diagnostic.decide(noSplit).headSplit.on)
    }

    /// **A gain within the margin does not count**, and the margin is the map's own drift when larger.
    func testATieWithinTheMarginStaysOnTheGPUAlone() {
        var r = m1Pro
        r.drift = (1.0, 1.0)
        r.both = r.gpu.map { $0.map { $0 * 1.04 } }        // +4 %: under the 5 % floor
        r.flash = r.graph.map { $0.map { $0 * 1.06 } }     // +6 %: over it
        var d = Diagnostic.decide(report(r))
        XCTAssertEqual(d.margin, 0.05)
        XCTAssertFalse(d.amx.on)
        XCTAssertNil(d.amxFraction)
        XCTAssertTrue(d.amx.passed)
        XCTAssertTrue(d.flash.on == false, "the split (×1.45) is faster: Z-Image keeps it")
        r.split = r.graph                                    // no split gain: flash's +6 % now decides
        d = Diagnostic.decide(report(r))
        XCTAssertTrue(d.flash.on)
        XCTAssertFalse(d.headSplit.on)
        r.drift = (0.92, 1.0)                                // a noisy map: 8 % of drift swallows the 6 %
        d = Diagnostic.decide(report(r))
        XCTAssertEqual(d.margin, 0.08, accuracy: 0.001)
        XCTAssertFalse(d.flash.on)
    }

    /// **A gain at 1024² does not buy a loss at 512²**: one size losing beyond the noise keeps it off.
    func testOneSizeLosingKeepsTheBranchOff() {
        var r = m1Pro
        r.drift = (1.0, 1.0)
        r.both = [r.gpu[0].map { $0 * 0.9 }, r.gpu[1].map { $0 * 1.8 }]   // geometric mean ×1.27, but 512² −10 %
        let d = Diagnostic.decide(report(r))
        XCTAssertGreaterThan(d.amx.gain, 1.2)
        XCTAssertFalse(d.amx.on)
    }

    /// Both attention paths winning: the split stays for Qwen-Image-2.1, flash only where it is the faster.
    func testFlashAndTheSplitBothWinning() {
        var r = m1Pro
        r.drift = (1.0, 1.0)
        r.flash = r.graph.map { $0.map { $0 * 2 } }
        let d = Diagnostic.decide(report(r))
        XCTAssertTrue(d.flash.on)
        XCTAssertTrue(d.headSplit.on)
    }

    /// The report carries its decisions — what a shared JSON says the Mac chose — and reads back.
    func testTheReportCarriesItsDecisions() throws {
        let original = report(m1Pro)
        XCTAssertEqual(original.decisions, Diagnostic.decide(original))
        let text = String(decoding: try original.json(pretty: false), as: UTF8.self)
        XCTAssertTrue(text.contains("\"decisions\":{\"amx\":{"), text)
        XCTAssertEqual(try JSONDecoder().decode(Diagnostic.Report.self, from: try original.json()), original)
        // A report written before the decisions still reads, and the rules still decide on it.
        var older = try JSONSerialization.jsonObject(with: try original.json()) as! [String: Any]
        older.removeValue(forKey: "decisions")
        let read = try JSONDecoder().decode(Diagnostic.Report.self, from: try JSONSerialization.data(withJSONObject: older))
        XCTAssertNil(read.decisions)
        XCTAssertEqual(Diagnostic.decide(read), original.decisions)
    }

    // ── the fingerprint ─────────────────────────────────────────────────────────────────────

    /// **A map is this process's only if nothing of the fingerprint moved**: hardware, chip, macOS and its
    /// build, the binary's version and its stamp.
    func testAStaleFingerprintIsNotCurrent() {
        let now = Diagnostic.Fingerprint.current
        var machine = Self.machine
        (machine.hardwareModel, machine.chip, machine.macOS, machine.version, machine.build)
            = (now.hardwareModel, now.chip, now.macOS, now.version, now.build)
        let fresh = report(m1Pro, machine: machine)
        XCTAssertEqual(fresh.fingerprint, now)
        XCTAssertTrue(fresh.isCurrent)
        let changes: [(String, (inout Diagnostic.Machine) -> Void)] = [
            ("hardware", { $0.hardwareModel += "x" }), ("chip", { $0.chip = "Apple M5" }),
            ("macOS build", { $0.macOS += "a" }), ("version", { $0.version += ".1" }), ("build", { $0.build = "000000000000" }),
        ]
        for (what, change) in changes {
            var other = machine
            change(&other)
            XCTAssertFalse(report(m1Pro, machine: other).isCurrent, what)
        }
        // Not the memory, nor the GPU's figures: they do not change what a branch costs on one chip.
        var sameChip = machine
        sameChip.memoryBytes *= 2
        XCTAssertTrue(report(m1Pro, machine: sameChip).isCurrent)
        var otherSchema = fresh
        otherSchema.schema = Diagnostic.schema + 1
        XCTAssertFalse(otherSchema.isCurrent)
    }

    // ── the GPU's share per shape (`cuts` → `amx_shares` → `Conductor.Shares`) ───────────────────────

    /// **A map with its grid gives each product its own share**: `ρ = (1 − T*) / T*` at every point,
    /// handed to the engine as text with MPS's row tile as quantum; the share read back at a measured
    /// point is the point's.
    func testCutsGiveEachShapeItsShare() throws {
        var r = report(m1Pro)
        r.cuts = Diagnostic.cutDepths.flatMap { k in Diagnostic.heights.map { m in
            Diagnostic.Cut(k: k, n: Diagnostic.cutWidth, m: m, share: k < 8192 ? 0.62 : 0.69) } }
        let d = Diagnostic.decide(r)
        let text = try XCTUnwrap(d.amxShares)
        XCTAssertEqual(d.settings["amx_shares"] as? String, text)
        XCTAssertEqual(d.settings["amx_quantum"] as? Int, 64)
        let shares = try XCTUnwrap(Conductor.Shares(text: text))
        XCTAssertEqual(shares.points.count, Diagnostic.cutDepths.count * Diagnostic.heights.count)
        XCTAssertEqual(shares.share(k: 4096, n: 6144, m: 1024)!, 0.62, accuracy: 0.001)
        XCTAssertEqual(shares.share(k: 16384, n: 2048, m: 4096)!, 0.69, accuracy: 0.001)
        XCTAssertNil(Diagnostic.decide(report(m1Pro)).amxShares, "no grid: the scalar share serves")
    }

    /// **The co-execution off, no grid in the settings**: the shares are the conductor's, and there is none.
    func testCutsWithoutCoexecutionGiveNoShares() {
        var rates = m1Pro
        rates.amxError = 1e-3
        var off = report(rates)
        off.cuts = [Diagnostic.Cut(k: 2048, n: 6144, m: 1024, share: 0.6)]
        let d = Diagnostic.decide(off)
        XCTAssertFalse(d.amx.on)
        XCTAssertNil(d.amxShares)
        XCTAssertNil(d.settings["amx_shares"])
    }

    /// **The formula, pure**: two axes through the square point, `ρ(k, n) = ρ(k, w) · ρ(w, n) / ρ(w, w)`, each
    /// read linearly in `log k` or `log n` (the geometric middle gives the mean ratio), then in `log m`; the
    /// nearest point outside the grid; clamped to `Conductor.fractionRange`; a grid without its square point
    /// reads `k` only; the text reads back; an empty grid gives no share.
    func testSharesComposeTheTwoAxes() throws {
        func point(_ k: Int, _ n: Int, _ m: Int, _ ratio: Double) -> Conductor.Shares.Point { .init(k: k, n: n, m: m, ratio: ratio) }
        let shares = Conductor.Shares([point(2048, 4096, 1024, 0.6), point(4096, 4096, 1024, 0.5), point(8192, 4096, 1024, 0.4),
                                       point(4096, 1024, 1024, 0.25), point(4096, 16384, 1024, 1.0),
                                       point(2048, 4096, 4096, 0.8), point(4096, 4096, 4096, 0.7), point(8192, 4096, 4096, 0.6),
                                       point(4096, 1024, 4096, 0.35), point(4096, 16384, 4096, 1.4)])
        XCTAssertEqual(shares.share(k: 4096, n: 4096, m: 1024)!, 1 / 1.5, accuracy: 1e-12)
        XCTAssertEqual(shares.share(k: 8192, n: 16384, m: 1024)!, 1 / 1.8, accuracy: 1e-12)
        XCTAssertEqual(shares.share(k: 2048, n: 2048, m: 1024)!, 1 / 1.45, accuracy: 1e-12)
        XCTAssertEqual(shares.share(k: 4096, n: 4096, m: 2048)!, 1 / 1.6, accuracy: 1e-12)
        XCTAssertEqual(shares.share(k: 1024, n: 4096, m: 512)!, 1 / 1.6, accuracy: 1e-12, "the nearest point outside")
        XCTAssertEqual(shares.share(k: 4096, n: 65536, m: 8192)!, 1 / 2.4, accuracy: 1e-12)
        let depthOnly = Conductor.Shares([point(2048, 6144, 1024, 0.6), point(8192, 6144, 1024, 0.4)])
        XCTAssertEqual(depthOnly.share(k: 4096, n: 1, m: 1024)!, 1 / 1.5, accuracy: 1e-12)
        XCTAssertEqual(depthOnly.share(k: 4096, n: 1, m: 1024), depthOnly.share(k: 4096, n: 65536, m: 1024))
        let skewed = Conductor.Shares([point(2048, 6144, 1024, 10), point(8192, 6144, 1024, 0)])
        XCTAssertEqual(skewed.share(k: 2048, n: 6144, m: 1024), Conductor.fractionRange.lowerBound)
        XCTAssertEqual(skewed.share(k: 8192, n: 6144, m: 1024), Conductor.fractionRange.upperBound)
        XCTAssertEqual(Conductor.Shares(text: shares.description), shares)
        XCTAssertTrue(shares.description.hasPrefix("2048×4096@1024=0.600,"))
        XCTAssertEqual(Conductor.Shares(text: ""), Conductor.Shares.none)
        XCTAssertNil(Conductor.Shares(text: "2048@1024=0.5"), "a grid without its n does not read")
        XCTAssertNil(Conductor.Shares.none.share(k: 4096, n: 6144, m: 1024))
    }
}
