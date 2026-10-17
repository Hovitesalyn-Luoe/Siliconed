import XCTest
@testable import Siliconed

/// **Where a Krea 2 LoRA lands, without a byte of weights.** Every module the forge accepts is one the
/// engine applies — the timestep path in double inside the tabulated modulation, `img_in` in the DiT's
/// cache —, the one it cannot reach is refused on import, and the cache serves what the need counts.
/// Maps written by hand, values exact in bf16.
final class Krea2LoRATests: XCTestCase {
    private var root: URL!

    override func setUpWithError() throws {
        root = FileManager.default.temporaryDirectory.appendingPathComponent("siliconed-krea2-lora-\(UUID())")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws { try? FileManager.default.removeItem(at: root) }

    /// A `SILICON` v3 map, each tensor bf16 on a 16-byte page: its `values` (exact in bf16), zeros without.
    private func map(_ header: [String: Any], tensors: [(name: String, shape: [Int], values: [Float]?)]) throws -> String {
        let page = 16, dataStart = 1 << 16
        var table: [String: Any] = [:], offset = dataStart, data: [(Int, [Float])] = []
        for t in tensors {
            let count = t.shape.reduce(1, *)
            table[t.name] = ["offset": offset, "bytes": count * 2, "shape": t.shape, "dtype": "bfloat16"]
            if let v = t.values { precondition(v.count == count); data.append((offset, v)) }
            offset += (count * 2 + page - 1) / page * page
        }
        var full = header
        full["page"] = page; full["tensors"] = table; full["order"] = tensors.map(\.name)
        let json = try JSONSerialization.data(withJSONObject: full)
        precondition(16 + json.count <= dataStart, "header longer than the test's layout")
        var bytes = Array("SILICON".utf8) + [3]
        for i in 0..<8 { bytes.append(UInt8((UInt64(json.count) >> (8 * UInt64(i))) & 0xff)) }
        bytes += json
        bytes += [UInt8](repeating: 0, count: max(offset, dataStart + 16) - bytes.count)
        for (start, values) in data {
            for (i, v) in values.enumerated() {
                let bits = v.bitPattern >> 16
                precondition(Float(bitPattern: bits << 16) == v, "\(v) is not exact in bf16")
                bytes[start + 2 * i] = UInt8(bits & 0xff); bytes[start + 2 * i + 1] = UInt8(bits >> 8)
            }
        }
        let path = root.appendingPathComponent("m-\(UUID()).silicon").path
        try Data(bytes).write(to: URL(fileURLWithPath: path))
        return path
    }

    /// Values exact in bf16, a few per eighth: `((i·a + b) mod 13 − 6) / 32`.
    private func values(_ count: Int, _ a: Int, _ b: Int) -> [Float] {
        (0..<count).map { Float(($0 * a + b) % 13 - 6) / 32 }
    }

    private let (d, timeDim, channels) = (16, 8, 16)

    /// A one-block DiT, `d = 16`, a sinusoid of 8: the timestep's three weights carry values.
    private func dit() throws -> Artifact {
        let header: [String: Any] = ["kind": "krea2-turbo-dit", "rope_layout": "interleaved", "rmsnorm_plus_one_folded": true,
                                     "linear_weights_transposed": true,
                                     "config": ["num_attention_heads": 2, "num_key_value_heads": 1, "attention_head_dim": 8,
                                                "intermediate_size": 32, "num_layers": 1, "in_channels": channels,
                                                "timestep_embed_dim": timeDim]]
        return try Artifact(path: try map(header, tensors: [
            ("time_embed.linear_1.weight", [timeDim, d], values(timeDim * d, 5, 1)),
            ("time_embed.linear_2.weight", [d, d], values(d * d, 7, 2)),
            ("time_mod_proj.weight", [d, 6 * d], values(6 * d * d, 3, 4)),
        ]))
    }

    private func lora(_ targets: [(String, Int, Int)], rank: Int = 2, strength: Float = 0.5) throws -> LoRA {
        let tensors = targets.flatMap { t -> [(name: String, shape: [Int], values: [Float]?)] in
            [("\(t.0).down", [t.1, rank], values(t.1 * rank, 4, 3)), ("\(t.0).up", [rank, t.2], values(rank * t.2, 6, 5))]
        }
        let path = try map(["kind": "lora"], tensors: tensors)
        return try LoRA(layers: [LoRA.Layer(artifact: try Artifact(path: path), strength: strength, name: path)])
    }

    // ── what the forge accepts is what the engine applies ─────────────────────────────────────

    /// diffusers names `tmlp.*`, `tproj.1`, `first`, `last.linear` and PEFT adapts them: the forge
    /// accepts them; the fusion's projector, a weighted sum the engine computes outside a GEMM, is refused.
    func testTheForgeKeepsWhatTheEngineAppliesAndRefusesTheProjector() throws {
        for (published, target) in [("tmlp.0", "time_embed.linear_1"), ("tmlp.2", "time_embed.linear_2"),
                                    ("tproj.1", "time_mod_proj"), ("first", "img_in"), ("last.linear", "final_layer.linear"),
                                    ("txtmlp.1", "txt_in.linear_1"), ("blocks.2.mlp.down", "transformer_blocks.2.ff.down"),
                                    ("txtfusion.refiner_blocks.1.attn.gate", "text_fusion.refiner_blocks.1.attn.to_gate")] {
            XCTAssertEqual(try ForgeLoRA.targets(published, family: .krea2).map(\.module), [target], published)
        }
        XCTAssertThrowsError(try ForgeLoRA.targets("txtfusion.projector", family: .krea2)) {
            XCTAssertTrue("\($0)".contains("projector"), "\($0)")
        }
        XCTAssertThrowsError(try ForgeLoRA.targets("transformer_blocks.0.norm1", family: .krea2))
        XCTAssertThrowsError(try ForgeLoRA.targets("transformer_blocks.x.attn.to_q", family: .krea2))
    }

    /// The DiT refuses, before any computation, a map forged before the forge refused: nothing left out silently.
    func testTheDiTRefusesAModuleItWouldIgnore() throws {
        let artifact = try dit()
        let stack = try lora([("text_fusion.projector", 12, 1)])
        XCTAssertThrowsError(try Krea2DiT(artifact: artifact, latentHeight: 8, latentWidth: 8, maxText: 4, lora: stack)) {
            XCTAssertTrue("\($0)".contains("text_fusion.projector"), "\($0)")
        }
    }

    // ── the cache serves what the need counts ─────────────────────────────────────────────────

    /// `img_in` was asked of the cache by `linearGPU`, but neither retained by it nor counted by the
    /// need: one list now, read by both.
    func testTheCacheServesWhatTheNeedCounts() throws {
        let artifact = try dit()
        let config = Krea2DiT.Config(header: artifact.header)
        let targets = Krea2DiT.loraTargets(config: config)
        XCTAssertEqual(targets.count, 8 * config.layers + 2)
        XCTAssertTrue(targets.allSatisfy { Krea2DiT.appliesLoRA($0.name) })
        XCTAssertTrue(targets.contains { $0.name == "img_in" && $0.k == channels && $0.n == d })
        XCTAssertEqual(Krea2DiT.loraModules(config: config).map(\.k), targets.map(\.k))
        XCTAssertEqual(Krea2DiT.loraModules(config: config).map(\.n), targets.map(\.n))
        let rank = 2
        let stack = try lora([("img_in", channels, d)], rank: rank)
        let dit = try Krea2DiT(artifact: artifact, latentHeight: 8, latentWidth: 8, maxText: 4, lora: stack)
        let served = CacheLoRA.arenaBytes(CacheLoRA.bytes(modules: [(k: channels, n: d, rank: rank)]))
        XCTAssertEqual(dit.loraCacheCapacity, served)
        XCTAssertGreaterThan(served, CacheLoRA.arenaBytes(CacheLoRA.bytes(modules: [])))
    }

    // ── the timestep path: in the tabulation, in double ───────────────────────────────────────

    /// The reference in double, transcribed from diffusers + PEFT: `temb = L2(gelu(L1(sinusoid)))`,
    /// `temb_mod = L3(gelu(temb))`, each `Lᵢ(x) = Wᵢx + s·Bᵢ(Aᵢx)`.
    private func reference(_ artifact: Artifact, sigma: Float, lora: LoRA?) throws -> (temb: [Double], mod: [Double]) {
        func weight(_ name: String) throws -> (w: [Float], k: Int, n: Int) {
            let shape = artifact.tensors[name + ".weight"]!.shape
            var w = [Float](repeating: 0, count: shape[0] * shape[1])
            _ = try w.withUnsafeMutableBufferPointer { try artifact.materialize(name + ".weight", into: $0) }
            return (w, shape[0], shape[1])
        }
        func linear(_ name: String, _ x: [Double]) throws -> [Double] {
            let (w, k, n) = try weight(name)
            var y = (0..<n).map { j in (0..<k).reduce(0.0) { $0 + x[$1] * Double(w[$1 * n + j]) } }
            if let layer = lora?.layers.first, let a = layer.artifact.tensors[name + ".down"] {
                let r = a.shape[1]
                var down = [Float](repeating: 0, count: k * r), up = [Float](repeating: 0, count: r * n)
                _ = try down.withUnsafeMutableBufferPointer { try layer.artifact.materialize(name + ".down", into: $0) }
                _ = try up.withUnsafeMutableBufferPointer { try layer.artifact.materialize(name + ".up", into: $0) }
                let mid = (0..<r).map { c in (0..<k).reduce(0.0) { $0 + x[$1] * Double(down[$1 * r + c]) } }
                for j in 0..<n { y[j] += Double(layer.strength) * (0..<r).reduce(0.0) { $0 + mid[$1] * Double(up[$1 * n + j]) } }
            }
            return y
        }
        var hidden = try linear("time_embed.linear_1", Krea2Exact.sinusoid(sigma: sigma, dim: timeDim).map(Double.init))
        Krea2Exact.geluTanh(&hidden)
        let temb = try linear("time_embed.linear_2", hidden)
        var enabled = temb
        Krea2Exact.geluTanh(&enabled)
        return (temb, try linear("time_mod_proj", enabled))
    }

    private func worst(_ got: [Float], _ exact: [Double]) -> Double {
        let scale = exact.map(abs).max() ?? 1
        return zip(got, exact).map { abs(Double($0) - $1) / scale }.max() ?? 0
    }

    /// A LoRA on `tmlp.*` / `tproj.1` enters the table (diffusers applies it: PEFT wraps those `Linear`s
    /// like any other), to fp32's rounding of the double reference; a stack that does not touch them
    /// leaves the table at the bits it has without a LoRA.
    func testATimestepLoRAEntersTheTabulatedModulation() throws {
        let artifact = try dit()
        let sigma: Float = 0.75
        func table(_ stack: LoRA?) throws -> (temb: [Float], mod: [Float]) {
            let dit = try Krea2DiT(artifact: artifact, latentHeight: 8, latentWidth: 8, maxText: 4, lora: stack)
            try dit.tabulateModulation(sigmas: [sigma])
            return try XCTUnwrap(dit.modulations[sigma.bitPattern])
        }
        let bare = try table(nil)
        let elsewhere = try table(try lora([("transformer_blocks.0.attn.to_q", d, d)]))
        XCTAssertEqual(elsewhere.temb.map(\.bitPattern), bare.temb.map(\.bitPattern))
        XCTAssertEqual(elsewhere.mod.map(\.bitPattern), bare.mod.map(\.bitPattern))

        let stack = try lora([("time_embed.linear_1", timeDim, d), ("time_embed.linear_2", d, d), ("time_mod_proj", d, 6 * d)])
        let adapted = try table(stack)
        let exact = try reference(artifact, sigma: sigma, lora: stack), plain = try reference(artifact, sigma: sigma, lora: nil)
        XCTAssertLessThan(worst(adapted.temb, exact.temb), 1e-7)
        XCTAssertLessThan(worst(adapted.mod, exact.mod), 1e-7)
        XCTAssertLessThan(worst(bare.mod, plain.mod), 1e-7)
        // the LoRA's effect, far above that rounding: a table left without it would fail here
        XCTAssertGreaterThan(worst(bare.mod, exact.mod), 1e-3)
    }
}
