import Foundation
import XCTest
@testable import Siliconed

/// **The diagnostic without a byte of weights**: the derived render time, the JSON a report becomes
/// (stable, explicit `null`s, read back whole), the issue URL, and the embedded golden as the library
/// finds it in its resource bundle.
final class DiagnosticTests: XCTestCase {
    private func sample(golden: Bool) -> Diagnostic.Report {
        let machine = Diagnostic.Machine(chip: "Apple M1 Pro", hardwareModel: "MacBookPro18,3", performanceCores: 6,
                                         efficiencyCores: 2, gpuCores: 14, memoryBytes: 17_179_869_184,
                                         macOS: "26.0.1 (25A362)", version: "git 0123456789-dirty", build: "x+y",
                                         gpu: "Apple M1 Pro", gpuFamily: "apple7", gpuWorkingSetBytes: 11_453_251_584)
        var model = Diagnostic.ModelReport(
            model: "qwen-image-2.1", name: "Qwen-Image-2.1 Turbo", family: "qwen-image-2.1", steps: 6, evaluations: 6,
            seconds: .init(encoder: 3, firstEvaluation: 6, steadyEvaluation: 4, decoding: 1, total: 15),
            estimatedRenderSeconds: 30, peakFootprintBytes: 4_000_000_000, swapouts: 0,
            golden: golden ? .init(file: "diagnostic/qwen-image-2.1.safetensors", source: "s", sourceSHA256: "h",
                                   generated: "2026-10-04") : nil,
            deviation: golden ? .init(worst: 9.3e-6, channel: 2, median: 5.5e-6, threshold: 5e-4, pass: true,
                                      firstEvaluationWorst: 3.7e-6) : nil,
            error: nil)
        model.diskReadBytes = 5_400_000_000
        model.variant = "standard"
        model.thermalAfter = "nominal"
        if golden {
            model.memory = .init(floorBytes: 3_000_000_000, lean: true,
                                 decisions: ["map tail 60 of 60 layers, 2 buffer(s) of 362 MB"],
                                 budgetBytes: MemoryBudget.ceiling(physical: 16 << 30), needBytes: 3_900_000_000,
                                 needByPhase: ["text": 2_100_000_000, "denoising": 2_900_000_000, "decoding": 3_900_000_000],
                                 check: MemoryCheck(needBytes: 3_900_000_000, measuredBytes: 4_000_000_000, phases: [
                                    .init(name: "start", needBytes: nil, measuredBytes: 30_000_000),
                                    .init(name: "text", needBytes: 2_100_000_000, measuredBytes: 1_900_000_000),
                                    .init(name: "denoising", needBytes: 2_900_000_000, measuredBytes: 2_700_000_000),
                                    .init(name: "decoding", needBytes: 3_900_000_000, measuredBytes: 4_000_000_000),
                                 ], fallback: false, peakMethod: .sampled10ms),
                                 peakMethod: .sampled10ms)
        }
        let conditions = Diagnostic.Conditions(thermal: "nominal", lowPowerMode: false, onBattery: nil, swapUsedBytes: 0)
        return Diagnostic.Report(date: "2026-10-04T10:00:00Z", machine: machine, models: [model],
                                 totalSeconds: 30, peakFootprintBytes: 4_000_000_000, swapouts: 0,
                                 before: conditions, after: conditions,
                                 witnessBefore: .init(gpuTFLOPS: 3.36, amxTFLOPS: 1.99, gpuWorstRowError: 0),
                                 witnessAfter: nil, settings: ["amx": "true (default)"], profile: "none",
                                 libraryOnInternalDisk: true)
    }

    // ── the estimate ────────────────────────────────────────────────────────────────────────

    func testEstimatedRenderIsEncoderFirstStepsAndDecoder() {
        // 6 evaluations: the first (with its construction), then 5 like the second.
        XCTAssertEqual(Diagnostic.estimatedRenderSeconds(encoder: 3, first: 6, steady: 4, decoding: 1, evaluations: 6), 30)
        // 7 evaluations.
        XCTAssertEqual(Diagnostic.estimatedRenderSeconds(encoder: 2, first: 5, steady: 3, decoding: 1, evaluations: 7), 26)
        XCTAssertEqual(Diagnostic.estimatedRenderSeconds(encoder: 2, first: 5, steady: 3, decoding: 1, evaluations: 1), 8)
        XCTAssertEqual(Diagnostic.estimatedRenderSeconds(encoder: 2, first: 5, steady: 3, decoding: 1, evaluations: 0), 8)
    }

    // ── the JSON ────────────────────────────────────────────────────────────────────────────

    func testJSONIsStableAndReadsBack() throws {
        let report = sample(golden: true)
        let a = try report.json(), b = try report.json()
        XCTAssertEqual(a, b)
        XCTAssertEqual(try JSONDecoder().decode(Diagnostic.Report.self, from: a), report)
        let text = String(decoding: a, as: UTF8.self)
        // Sorted keys: `date` before `machine` before `models` before `prompt`.
        let order = ["\"date\"", "\"machine\"", "\"models\"", "\"prompt\"", "\"schema\""].map { text.range(of: $0)!.lowerBound }
        XCTAssertEqual(order, order.sorted())
        XCTAssertTrue(text.contains("\"estimatedRenderSeconds\" : 30"))
        XCTAssertTrue(text.contains("\"schema\" : \(Diagnostic.schema)"))
    }

    /// **Schema 3, kept by 4**: B, the need per phase, the measured peak of each phase and the gap — a phase above
    /// its need is a defect, said by name; a boundary the formula has no phase for writes `null`.
    func testSchemaThreeCarriesTheNeedAgainstThePeak() throws {
        XCTAssertEqual(Diagnostic.schema, 4)
        let report = sample(golden: true)
        let text = String(decoding: try report.json(), as: UTF8.self)
        // Schema 4: the kernel's reading is gone.
        for gone in ["reclaimableBytes", "availableBytes", "reserveBytes", "comfortableBytes", "compressorHeadroomBytes"] {
            XCTAssertFalse(text.contains(gone), gone)
        }
        for key in ["budgetBytes", "needBytes", "needByPhase", "measuredBytes", "gapBytes", "check", "peakMethod"] {
            XCTAssertTrue(text.contains("\"\(key)\""), key)
        }
        // The method that read the peaks, in the check and beside it (said even when the check is `null`).
        XCTAssertTrue(text.contains("\"peakMethod\" : \"sampled10ms\""))
        XCTAssertEqual(report.models[0].memory?.peakMethod, .sampled10ms)
        XCTAssertTrue(text.contains("\"needBytes\" : null"))   // the `start` boundary
        let check = try XCTUnwrap(report.models[0].memory?.check)
        XCTAssertEqual(check.gapBytes, 100_000_000)
        XCTAssertEqual(check.exceeded.map(\.name), ["decoding"])
        XCTAssertTrue(check.isDefect)
        XCTAssertEqual(check.defects.count, 1)
        XCTAssertTrue(check.defects[0].contains("decoding"))
        // A check built from marks: each boundary mapped to the formula's phase, the render's peak the largest.
        let need = MemoryNeed(phases: [.init(kind: .text, terms: [.init("a", 100)]),
                                       .init(kind: .denoising, terms: [.init("b", 300)])], crossing: 10, base: 5)
        let built = MemoryCheck(need: need, marks: [("start", 20), ("text", 90), ("denoising", 400), ("denoiser released", 30)])
        XCTAssertEqual(built.peakMethod, FootprintInterval.method)
        XCTAssertEqual(built.needBytes, 315)
        XCTAssertEqual(built.phases.map(\.needBytes), [nil, 115, 315, 315])
        XCTAssertEqual(built.measuredBytes, 400)
        XCTAssertEqual(built.exceeded.map(\.name), ["denoising"])
    }

    func testMissingGoldenIsAnExplicitNull() throws {
        let report = sample(golden: false)
        let text = String(decoding: try report.json(), as: UTF8.self)
        XCTAssertTrue(text.contains("\"golden\" : null"), text)
        XCTAssertTrue(text.contains("\"deviation\" : null"))
        XCTAssertTrue(text.contains("\"error\" : null"))
        // A model refused before its budget was read says so, it does not drop the field.
        XCTAssertTrue(text.contains("\"memory\" : null"))
        XCTAssertEqual(try JSONDecoder().decode(Diagnostic.Report.self, from: try report.json()), report)
        let compact = String(decoding: try report.json(pretty: false), as: UTF8.self)
        XCTAssertFalse(compact.contains("\n"))
        XCTAssertTrue(compact.contains("\"golden\":null"))
    }

    /// The machine's state reads without weights, and `amx` is among the settings a report always names.
    func testConditionsAndSettingsRows() {
        let c = Diagnostic.conditions()
        XCTAssertTrue(["nominal", "fair", "serious", "critical"].contains(c.thermal))
        XCTAssertGreaterThanOrEqual(c.swapUsedBytes, 0)
        let keys = EngineSettings.effective.rows.map(\.key)
        XCTAssertTrue(keys.contains("amx"))
        XCTAssertEqual(Set(keys).count, keys.count)
    }

    // ── the issue URL ───────────────────────────────────────────────────────────────────────

    func testIssueURLCarriesTheFieldsAndTheWholeJSON() throws {
        let report = sample(golden: true)
        let url = try report.issueURL()
        XCTAssertTrue(url.absoluteString.hasPrefix("https://github.com/gwenn-ha-dev/Siliconed/issues/new?template=diagnostic.yml&"))
        let items = URLComponents(url: url, resolvingAgainstBaseURL: false)!.queryItems!
        func field(_ name: String) -> String? { items.first { $0.name == name }?.value }
        XCTAssertEqual(field("chip"), "Apple M1 Pro (14-core GPU)")
        XCTAssertEqual(field("memory"), "16 GB")
        XCTAssertEqual(field("macos"), "26.0.1 (25A362)")
        XCTAssertEqual(field("version"), "git 0123456789-dirty")
        XCTAssertEqual(field("labels"), "diagnostic")
        let json = try XCTUnwrap(field("report"))
        XCTAssertEqual(try JSONDecoder().decode(Diagnostic.Report.self, from: Data(json.utf8)), report)
        // `+` is escaped: GitHub reads a bare one back as a space.
        XCTAssertFalse(url.absoluteString.contains("+"))
        XCTAssertTrue(url.absoluteString.contains("x%2By"))
    }

    /// Too long for a link: the same form, every field but the report, which the user pastes.
    func testIssueURLWithoutTheReport() throws {
        let report = sample(golden: true)
        let items = URLComponents(url: try report.issueURL(includingReport: false), resolvingAgainstBaseURL: false)!.queryItems!
        XCTAssertNil(items.first { $0.name == "report" })
        XCTAssertEqual(items.first { $0.name == "chip" }?.value, "Apple M1 Pro (14-core GPU)")
        XCTAssertEqual(items.first { $0.name == "template" }?.value, "diagnostic.yml")
    }

    /// No account name in what the app shows or a report publishes.
    func testTheHomeFolderIsWrittenTilde() {
        let home = FileManager.default.homeDirectoryForCurrentUser.path
        XCTAssertEqual(Library.withoutHome("file missing: \(home)/Library/x.silicon"), "file missing: ~/Library/x.silicon")
        XCTAssertEqual(Library(root: URL(fileURLWithPath: home + "/Library/Application Support/Siliconed")).displayPath,
                       "~/Library/Application Support/Siliconed")
        XCTAssertEqual(Library.withoutHome("/opt/x"), "/opt/x")
    }

    // ── the embedded golden ─────────────────────────────────────────────────────────────────

    func testEmbeddedQwenGoldenReads() throws {
        let golden = try XCTUnwrap(try Diagnostic.EmbeddedGolden.load("qwen-image-2.1"),
                                   "Resources/diagnostic/qwen-image-2.1.safetensors not found in the resource bundle")
        XCTAssertEqual([golden.channels, golden.height, golden.width], [64, 32, 32])
        XCTAssertEqual(golden.inputs.map(\.count), [65_536, 65_536])
        XCTAssertEqual(golden.outputs.map(\.count), [65_536, 65_536])
        XCTAssertEqual(golden.metadata["source"], "goldens-qwen21-trajectory-512.safetensors")
        // `x_t.0` is the oracle's seed-0 `torch.randn((1, 1, 64, 32, 32))` (checked in Python, to the bit).
        // `Latent.noise` draws it — but under `swift test` (a debug build) 765 of its 65,536 values differ
        // by 1–2 ulp: not claimed to the bit here (the release check `qwen21-trajectory` compares it).
        let noise = Latent.noise(.qwenImage21, height: 32, width: 32, seed: 0).values
        let ulps = zip(noise, golden.inputs[0]).map { abs(Int($0.bitPattern) - Int($1.bitPattern)) }.max() ?? .max
        XCTAssertLessThanOrEqual(ulps, 2)
        // The golden against itself is 0; a shifted channel is seen on its own.
        XCTAssertEqual(Diagnostic.channelError(planar: golden.outputs[1], golden.outputs[1], channels: 64).worst, 0)
        var moved = golden.outputs[1]
        for i in (5 * 1024)..<(6 * 1024) { moved[i] *= 1.001 }
        let e = Diagnostic.channelError(planar: moved, golden.outputs[1], channels: 64)
        XCTAssertEqual(e.channel, 5)
        XCTAssertEqual(e.worst, 1e-3, accuracy: 1e-6)
        // A golden from before the Turbo (the base model under a third-party turbo LoRA, written
        // 2026-10-04, σ₁ = 0.96255648) would stop here and say why. The embedded one is the Turbo's
        // (written 2026-10-10 from the oracle's 8-step `trajectory 512`), so the σ below are checked.
        try XCTSkipIf(golden.info.generated == "2026-10-04",
                      "PENDING: Resources/diagnostic/qwen-image-2.1.safetensors predates Qwen-Image-2.1-Turbo — regenerate it")
        // The Turbo's first two σ, the same at every size: `sample_sigmas[0:2]` made float32.
        XCTAssertEqual(golden.sigmas[0], 1)
        XCTAssertEqual(golden.sigmas[1].bitPattern, 0x3f7a7be5)
    }

    func testNoGoldenIsNil() throws {
        XCTAssertNil(try Diagnostic.EmbeddedGolden.load("anima"))
    }

    func testChannelsLastTransposes() {
        // [2, 1, 3] planar → [3, 2].
        XCTAssertEqual(Diagnostic.channelsLast([1, 2, 3, 10, 20, 30], channels: 2), [1, 10, 2, 20, 3, 30])
    }
}
