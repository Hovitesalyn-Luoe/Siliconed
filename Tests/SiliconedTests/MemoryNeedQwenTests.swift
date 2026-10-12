import XCTest
@testable import Siliconed

/// **Qwen-Image-2.1's memory need, without a byte of weights**: the formula
/// charges what the constructors allocate — the same static functions —, and it grows with every
/// input that makes a render larger.
final class MemoryNeedQwenTests: XCTestCase {
    typealias Grid = QwenImage21Sequence.Grid

    /// The library prompt's template: 27 tokens, 6 more per reference (its tags).
    static func request(_ width: Int, _ height: Int, references: [(width: Int, height: Int)] = [],
                        tokens: Int? = nil, ranks: [Int] = []) -> MemoryNeed.Request {
        MemoryNeed.Request(card: .qwenImage21, width: width, height: height, references: references,
                           promptTokens: tokens ?? 27 + 6 * references.count, loraRanks: ranks)
    }

    static func need(_ request: MemoryNeed.Request) -> MemoryNeed { MemoryNeed.of(request)! }

    static func phase(_ need: MemoryNeed, _ kind: MemoryNeed.Phase.Kind) -> MemoryNeed.Phase? {
        need.phases.first { $0.kind == kind }
    }

    // ── the formula is the constructors' ─────────────────────────────────────────────────────

    /// The prefill's groups on the oracle's edit-2 layout: one group for the whole prefix (2 072 rows
    /// under the budget of max(1 024, 1 024) + 60 text rows? no — 2 072 > 1 084: two groups), and the
    /// scratch is the largest group.
    func testThePrefillGroupsAreTheDiTs() throws {
        let slots = [Bool](repeating: false, count: 8) + [Bool](repeating: true, count: 247)
            + [Bool](repeating: false, count: 6) + [Bool](repeating: true, count: 256)
            + [Bool](repeating: false, count: 46) + [Bool](repeating: true, count: 256)
        let s = try QwenImage21Sequence(slots: slots, images: [Grid(height: 26, width: 38), Grid(height: 32, width: 32),
                                                               Grid(height: 32, width: 32)])
        let layout = QwenImage21DiT.layout(s, fullPass: false)
        // Budget: max(S 1 024, largest image 1 024) + 60 text rows = 1 084.
        XCTAssertEqual(layout.groups.map { [$0.start, $0.end] }, [[0, 1002], [1002, 2072]])
        XCTAssertEqual(layout.scratch, 1070)
        XCTAssertEqual(QwenImage21DiT.layout(s, fullPass: true).scratch, 3096)
    }

    /// The arena of a 1024² generation, written out float by float — what `QwenImage21DiT.init` asks.
    func testTheDiTArenaIsItsSlots() {
        let need = QwenImage21Need(Self.request(1024, 1024))
        let seq = need.sequence
        XCTAssertEqual(seq.target, 4096); XCTAssertEqual(seq.prefix, 13)   // 27 − 14 text rows
        let (d, h, all, n) = (4096, 12_288, seq.count, 4096)
        let mlpRows = QwenImage21DiT.mlpRows(scratch: n, hidden: h)
        XCTAssertEqual(mlpRows, 4096)   // 403 MB: one chunk
        // Every slice a whole number of pages (a row of `d` is one page) but the RoPE table's: no margin.
        let floats = all * d + 2 * n * d + 2 * all * d + mlpRows * h + mlpRows * h + d * h
            + 4096 * 64 + 2 * 5 * d
        XCTAssertEqual(QwenImage21DiT.arenaBytes(config: need.dit, rows: all, prefix: seq.prefix, scratch: n,
                                                 largestImage: 4096, rank: 0, fullPass: false),
                       floats * 4 + Arena.capacity(for: all * 128 * 4))
        XCTAssertEqual(QwenImage21DiT.cacheArenaBytes(config: need.dit, prefix: 13),
                       QwenImage21CachePolicy.bytes(prefix: 13))
    }

    /// The denoising phase is the sum of the statics the DiT, its cache and its LoRA call, plus the
    /// staging blocks of a generation, plus the sampler's copies.
    func testTheDenoisingIsTheConstructorsSum() {
        let request = Self.request(1024, 1024, ranks: [32])
        let need = QwenImage21Need(request)
        let seq = need.sequence
        XCTAssertEqual(need.storage, .memory)
        let scratch = QwenImage21DiT.layout(seq, fullPass: false).scratch
        let arena = QwenImage21DiT.arenaBytes(config: need.dit, rows: seq.count, prefix: seq.prefix, scratch: scratch,
                                              largestImage: seq.target, rank: 32, fullPass: false)
        let cache = QwenImage21DiT.cacheArenaBytes(config: need.dit, prefix: seq.prefix)
        let block = (4 * 4096 * 4096 + 3 * 4096 * 12_288 + 256) * 2
        let staging = 2 * ((block + Arena.alignment - 1) / Arena.alignment * Arena.alignment)
        let module = { (k: Int, n: Int) in (k, n, 32) }
        let lora = 2 * StreamedLoRA.layout([
            "attn.to_q": module(4096, 4096), "attn.to_k": module(4096, 4096), "attn.to_v": module(4096, 4096),
            "attn.to_out.0": module(4096, 4096), "img_mlp.gate_layer": module(4096, 12_288),
            "img_mlp.proj": module(4096, 12_288), "img_mlp.out": module(12_288, 4096)]).bytes
        let sampler = 64 * 4096 * 4 + 8 * 5 * 4096 * 4
        let phase = need.denoisingPhase
        XCTAssertEqual(phase.bytes("DiT arena") + phase.bytes("K/V arena") + phase.bytes("map staging")
                        + phase.bytes("LoRA slots") + phase.bytes("sampler"), arena + cache + staging + lora + sampler)
        // The rest: the attentions the DiT keeps, its GEMM, and the modulation's widening when it is
        // the larger moment — nothing unnamed.
        let rest = phase.terms.filter { !["DiT arena", "K/V arena", "map staging", "LoRA slots", "sampler"].contains($0.name) }
        for term in rest {
            XCTAssertTrue(term.name.hasPrefix("DiT ") || term.name.hasPrefix("modulation widening"), term.name)
        }
        // The widened `modulation.1` (`d × 4d`) is alive before the attentions, never beside them.
        let running = rest.filter { !$0.name.hasPrefix("modulation") }.reduce(0) { $0 + $1.bytes }
        XCTAssertEqual(running + phase.bytes("modulation widening"),
                       max(running, Krea2Exact.widenedBytes(inputs: 4096, outputs: 4 * 4096)
                            + Arena.capacity(for: 4096 * 32 * 4) + Arena.capacity(for: 32 * 4 * 4096 * 4)))
    }

    /// `StreamedLoRA`'s slot at rank 256 on the 7 Linears of a block: the "2 × 84 MB" of its header.
    func testTheLoRASlotIsItsHeaders() {
        let r = 256
        let slot = StreamedLoRA.layout([
            "attn.to_q": (4096, 4096, r), "attn.to_k": (4096, 4096, r), "attn.to_v": (4096, 4096, r),
            "attn.to_out.0": (4096, 4096, r), "img_mlp.gate_layer": (4096, 12_288, r),
            "img_mlp.proj": (4096, 12_288, r), "img_mlp.out": (12_288, 4096, r)])
        XCTAssertEqual(slot.bytes, 81_920 * r * 4)   // every piece already a whole page
        XCTAssertEqual(Double(slot.bytes) / 1e6, 84, accuracy: 0.5)
        XCTAssertEqual(slot.layout["attn.to_k"]!.down, 0)   // sorted by suffix
    }

    /// The decoder's tail at 1024²: a 256-wide input, bands of 64 kept rows + 2 × 6 of margin; its
    /// input is `x` 576 and `skip` 1152 channels at ×4 — what `decode` checks it meets.
    func testTheDecoderBandIsTheTails() {
        let config = QwenImage21VAE.Config()
        let input = QwenImage21VAEDecoder.tailInput(config: config)
        XCTAssertEqual([input.x, input.skip, input.scale], [576, 1152, 4])
        let band = QwenImage21VAEDecoder.band(config: config, tailHeight: 256, tailWidth: 256, bandPixels: 1 << 18)
        XCTAssertEqual([band.margin, band.core, band.window], [6, 64, 76])
        // A tail shorter than a band reads it whole.
        XCTAssertEqual(QwenImage21VAEDecoder.band(config: config, tailHeight: 40, tailWidth: 40, bandPixels: 1 << 18).window, 40)
        // Before the tail: the second up block's shortcut — 1152 in at ×4, its skip at ×2, 1152 out.
        XCTAssertEqual(QwenImage21VAEDecoder.stageRegisterFloats(config: config, latentHeight: 64, latentWidth: 64),
                       (2 * 1152 * 16 + 1152 * 4) * 64 * 64)
        // The encoder: block 1 — the RGBA input still held, block 0's 96 channels at ½, its 192 at ¼.
        XCTAssertEqual(QwenImage21VAEEncoder.registerFloats(config: config, height: 1024, width: 1024),
                       (4 + 96 / 4 + 192 / 16) * 1024 * 1024)
    }

    /// The vision tower's arena is its constructor's formula; one reference of 1024² is 4 096 patches.
    func testTheVisionArenaIsTheTowers() {
        let config = Qwen3VLVision.Config.published
        XCTAssertEqual([config.hidden, config.intermediate, config.patchValues, config.merged], [1152, 4304, 1536, 4608])
        let n = 4096, d = 1152
        let largest = 4608 * 4608, widest = 4608
        XCTAssertEqual(Qwen3VLVision.arenaBytes(config: config, patches: n),
                       (n * 1536 + 10 * n * d + n * 4304 + n / 4 * 4096 + n * 72 + largest) * 4
                        + 3 * Arena.capacity(for: widest * 4))   // whole pages but the bias rows: no margin
        let need = QwenImage21Need(Self.request(1024, 1024, references: [(1024, 1024)]))
        XCTAssertEqual(need.references[0].patches, 4096)
        XCTAssertEqual(need.encoderTokens, 33 + 1024)
        XCTAssertEqual(need.sequence.prefix, 19 + 4096)
    }

    /// The K/V's place follows the prefix: in memory to 256 rows, on disk beyond — a long prompt
    /// moves a generation's K/V to the file, and its staging blocks with it.
    func testALongPromptTakesTheFile() {
        XCTAssertEqual(QwenImage21Need(Self.request(1024, 1024, tokens: 270)).storage, .memory)   // P = 256
        let long = QwenImage21Need(Self.request(1024, 1024, tokens: 271))
        XCTAssertEqual(long.storage, .file)
        XCTAssertLessThan(long.denoisingPhase.own, QwenImage21Need(Self.request(1024, 1024, tokens: 270)).denoisingPhase.own)
        XCTAssertEqual(QwenImage21Need(Self.request(1024, 1024, references: [(1024, 1024)])).storage, .file)
    }

    // ── what makes a render larger makes its need larger ─────────────────────────────────────

    func testTheNeedGrowsWithTheArea() {
        let peaks = [(512, 512), (832, 1216), (1024, 1024), (1024, 1536)].map { Self.need(Self.request($0.0, $0.1)).peak }
        XCTAssertEqual(peaks, peaks.sorted()); XCTAssertEqual(Set(peaks).count, peaks.count)
        let edits = [(512, 512), (1024, 1024), (1024, 1536)].map {
            Self.need(Self.request($0.0, $0.1, references: [(1024, 1024)])).peak
        }
        XCTAssertEqual(edits, edits.sorted()); XCTAssertEqual(Set(edits).count, edits.count)
    }

    func testTheNeedGrowsWithTheReferences() {
        let refs: [(width: Int, height: Int)] = [(1024, 1024), (1024, 1024), (1024, 1024)]
        let peaks = (0...3).map { Self.need(Self.request(1024, 1024, references: Array(refs.prefix($0)))).peak }
        XCTAssertEqual(peaks, peaks.sorted()); XCTAssertEqual(Set(peaks).count, 4)
        // Each phase too, the decoding aside (it never sees the references).
        for kind in [MemoryNeed.Phase.Kind.text, .denoising] {
            let own = (0...3).map { Self.phase(Self.need(Self.request(1024, 1024, references: Array(refs.prefix($0)))), kind)!.own }
            // From one reference on: a generation's denoising holds staging blocks an edit does not.
            XCTAssertEqual(Array(own.dropFirst()), own.dropFirst().sorted(), "\(kind)")
        }
    }

    /// A reference's size: every original is prepared at ~R² (`conditionSize`), so what grows with it
    /// is the original held through the render — never less for a larger one.
    func testTheNeedGrowsWithAReferencesSize() {
        let sizes = [(512, 512), (1024, 1024), (2048, 2048), (4096, 4096)]
        let peaks = sizes.map { Self.need(Self.request(1024, 1024, references: [$0])).peak }
        XCTAssertEqual(peaks, peaks.sorted())
        XCTAssertLessThan(peaks.first!, peaks.last!)
    }

    func testTheNeedGrowsWithThePrompt() {
        for references in [[], [(width: 1024, height: 1024)]] as [[(width: Int, height: Int)]] {
            // Within one storage: the generation's K/V moves to the file past 256 text rows (above).
            let tokens = references.isEmpty ? [27, 100, 200, 270] : [33, 100, 400, 540]
            let peaks = tokens.map { Self.need(Self.request(1024, 1024, references: references, tokens: $0)).peak }
            XCTAssertEqual(peaks, peaks.sorted()); XCTAssertEqual(Set(peaks).count, peaks.count)
        }
    }

    func testTheNeedGrowsWithTheRank() {
        // The peak never falls with the rank; it may not rise either: the decoder (3.3 GB at 1024²,
        // measured) holds a generation's peak, and a LoRA lives in the denoising.
        let ranks: [[Int]] = [[], [16], [64], [128, 128]]
        let peaks = ranks.map { Self.need(Self.request(1024, 1024, ranks: $0)).peak }
        XCTAssertEqual(peaks, peaks.sorted())
        let generation = ranks.map { Self.phase(Self.need(Self.request(1024, 1024, ranks: $0)), .denoising)!.bytes }
        XCTAssertEqual(generation, generation.sorted()); XCTAssertEqual(Set(generation).count, generation.count)
        let denoising = [[], [64], [256]].map {
            Self.phase(Self.need(Self.request(1024, 1024, references: [(1024, 1024)], ranks: $0)), .denoising)!.own
        }
        XCTAssertEqual(denoising, denoising.sorted()); XCTAssertEqual(Set(denoising).count, 3)
    }

    func testAGenerationCostsLessThanAnEdit() {
        for (w, h) in [(512, 512), (1024, 1024), (1248, 832), (1024, 1536)] {
            XCTAssertLessThan(Self.need(Self.request(w, h)).peak,
                              Self.need(Self.request(w, h, references: [(w, h)])).peak, "\(w)×\(h)")
        }
    }

    /// Three references of mixed sizes: each prepared at its own ratio, the tower sized for the
    /// largest, the prefill in groups, every phase present.
    func testThreeReferencesOfMixedSizes() {
        let request = Self.request(1024, 1536, references: [(3000, 2000), (800, 1200), (640, 640)])
        let need = QwenImage21Need(request)
        XCTAssertEqual(need.references.map { [$0.prepared.width, $0.prepared.height] }, [[1248, 832], [832, 1248], [1024, 1024]])
        XCTAssertEqual(need.sequence.images.count, 4)
        XCTAssertEqual(need.sequence.prefix, need.textRows + 78 * 52 * 2 + 64 * 64)
        XCTAssertEqual(need.storage, .file)
        let full = Self.need(request)
        XCTAssertEqual(Set(full.phases.map(\.kind)), Set(MemoryNeed.Phase.Kind.allCases))
        XCTAssertEqual(full.peak, full.phases.map(\.bytes).max()! + full.crossing + full.base)
        // The vision tower is sized for the largest image (4 096 patches each here: 78·52 = 4 056).
        XCTAssertEqual(Self.phase(full, .vision)!.bytes("vision arena"),
                       Qwen3VLVision.arenaBytes(config: need.vision, patches: 4096))
        XCTAssertEqual(Self.phase(full, .vision)!.bytes("vision SDPA queries"), 4096 * 16 * 72 * 4)
        // More than the same with three 1024² references? The prefix is shorter (4 056 + 4 056 + 4 096).
        let squares = Self.need(Self.request(1024, 1536, references: [(1024, 1024), (1024, 1024), (1024, 1024)]))
        XCTAssertLessThan(Self.phase(full, .denoising)!.own, Self.phase(squares, .denoising)!.own)
        // The denoising: the target's SDPA cut by heads, one GEMM.
        let terms = Self.phase(full, .denoising)!.terms
        XCTAssertTrue(terms.contains { $0.name.hasSuffix("head split") })
        XCTAssertEqual(terms.filter { $0.name.hasSuffix("MPS GEMM") }.count, 1)
    }

    // ── the VAEs, the attentions: their shape functions ──────────────────────────────────────

    /// **The VAEs are their plans**: the encoder's two heaps and host copies at the largest
    /// reference with the RGBA the module hands it; the decoder's, then the RGB copy beyond its heaps.
    func testTheVAEsAreTheirPlans() {
        for (w, h) in [(512, 512), (1024, 1024), (1024, 1536)] {
            let need = QwenImage21Need(Self.request(w, h, references: [(w, h), (640, 640)]))
            let decoder = QwenImage21VAEDecoder.plan(config: need.vae, latentHeight: h / 16, latentWidth: w / 16)
            XCTAssertEqual(need.decodingPhase.bytes,
                           max(decoder.bytes, decoder.host + 3 * w * h * 4) + decoder.outside + OperatorFootprint.sdpaExecutable, "\(w)×\(h)")
            let largest = need.references.map(\.prepared).max { $0.width * $0.height < $1.width * $1.height }!
            XCTAssertEqual(need.referencesPhase!.bytes,
                           4 * largest.width * largest.height * 4
                            + { $0.bytes + $0.outside }(QwenImage21VAEEncoder.plan(config: need.vae, height: largest.height, width: largest.width))
                            + OperatorFootprint.sdpaExecutable)
        }
        // The lean plan's decoder takes the smallest band.
        let lean = QwenImage21Need(Self.request(1024, 1024).lean)
        XCTAssertEqual(lean.decodingPhase.bytes("Qwen decoder registers"),
                       QwenImage21VAEDecoder.plan(config: lean.vae, latentHeight: 64, latentWidth: 64,
                                                  bandPixels: min(QwenImage21VAEDecoder.bandPixels, MemoryPlan.qwenMinimumBand)).registers)
    }

    /// **Nothing of the encoder reaches the later phases** (`finish` hands its heaps back before it
    /// returns): an edit's denoising and decoding carry no encoder term.
    func testTheEncoderDoesNotOutliveItself() {
        let edit = QwenImage21Need(Self.request(1024, 1024, references: [(1024, 1024)]))
        for phase in [edit.denoisingPhase, edit.decodingPhase] {
            XCTAssertFalse(phase.terms.contains { $0.name.hasPrefix("Qwen encoder") })
        }
    }

    /// Every attention the DiT keeps, by its cache key: the causal text segments with their masks, the
    /// reference segments plain, the target cut by heads.
    func testEveryCachedAttentionIsCharged() {
        let need = QwenImage21Need(Self.request(1024, 1024, references: [(1024, 1024)]))
        let terms = need.denoisingPhase.terms
        let texts = need.sequence.segments.filter(\.isText).count
        let masks = terms.filter { $0.name.hasSuffix("causal mask") }
        XCTAssertGreaterThanOrEqual(masks.count, 1)
        XCTAssertLessThanOrEqual(masks.count, texts)
        XCTAssertGreaterThanOrEqual(terms.filter { $0.name.hasSuffix("head split") }.count, 1)
        let attention = terms.filter { $0.name.hasSuffix("causal mask") || $0.name.hasSuffix("head split")
            || $0.name.hasSuffix("SDPA queries") }.reduce(0) { $0 + $1.bytes }
        XCTAssertLessThan(attention, 600_000_000)
        // The encoder's causal SDPA: its mask is the square of its sequence, in our pages.
        XCTAssertEqual(need.textPhase.bytes("encoder causal mask"),
                       Attention.maskBytes(sequence: need.encoderTokens, keys: need.encoderTokens))
    }

    /// **No fitted term remains**: the outside terms are `OperatorFootprint`'s named ones only.
    func testNoFittedTermRemains() {
        for request in [Self.request(1024, 1024), Self.request(1024, 1536, references: [(1024, 1024), (640, 640)], ranks: [64])] {
            let need = Self.need(request)
            for term in need.phases.flatMap(\.terms) where term.outside {
                XCTAssertTrue(term.name.hasSuffix("SDPA queries") || term.name.hasSuffix("MPS GEMM")
                              || term.name.hasSuffix(" outside") || term.name.hasSuffix("SDPA executable"), term.name)
            }
        }
    }

    /// **The figures of the report**: generation 1024², edit 1024² with 3 references of
    /// 1024², per phase, default and lean plans; the library prompt (27 tokens, +6 per reference).
    func testPrintTheNeeds() {
        let points: [(String, MemoryNeed.Request)] = [
            ("generation 1024²", Self.request(1024, 1024)),
            ("edit 1024², 3 refs", Self.request(1024, 1024, references: [(1024, 1024), (1024, 1024), (1024, 1024)])),
            ("generation 1024×1536", Self.request(1024, 1536)),
            ("edit 1024×1536, 3 refs", Self.request(1024, 1536, references: [(1024, 1024), (1024, 1024), (1024, 1024)])),
        ]
        for (name, request) in points {
            for lean in [false, true] {
                let need = Self.need(lean ? request.lean : request)
                let phases = need.phases.map {
                    String(format: "%@ %.0f (ours %.0f, outside %.0f)", $0.kind.rawValue as NSString, Double($0.bytes) / 1e6,
                           Double($0.own) / 1e6, Double($0.outside) / 1e6)
                }
                print(String(format: "Q %@%@: peak %.0f MB (crossing %.1f, base %.1f) — %@", name as NSString,
                             (lean ? " lean" : "") as NSString, Double(need.peak) / 1e6, Double(need.crossing) / 1e6,
                             Double(need.base) / 1e6, phases.joined(separator: ", ") as NSString))
            }
        }
    }
}
