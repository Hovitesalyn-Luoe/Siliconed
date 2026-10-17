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
                                          coexecutionGPURows: nil, tensor: r.tensor.map { ok($0[h][i], r.tensorError) }))
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
                                 widening: [.init(kind: "bf16", values: 39_321_600, cpu: .init(rate: r.widen, error: 0, threshold: 0), gpu: nil),
                                            .init(kind: "bf16 copy", values: 39_321_600, cpu: .init(rate: r.copy, error: 0, threshold: 0), gpu: nil)],
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

    /// **The M5's map of 17/10/2026** (M225), its `matmul2d` at the bf16 × bf16 rate it measured (6.0):
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
}
