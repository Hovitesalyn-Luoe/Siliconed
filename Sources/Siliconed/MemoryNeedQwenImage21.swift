import Foundation

/// **Qwen-Image-2.1's need** — the render of `Chain.render` for this family,
/// phase by phase, each charged with what its constructors allocate: the same static functions they
/// call, and `OperatorFootprint`'s few named terms outside our hand.
///
///     references (originals) ─ Lanczos ~R² ─ prepared [3, Hᵢ, Wᵢ] ──────────────────────────────── crossing
///          │                                                                                       │
///          ├─ vision  : Qwen3VLVision arena (largest image's patches), every pixel_values, the    ┐
///          │            features done  + the tower's SDPA (queries' copy) · its GEMM             │
///          ├─ text    : TextEncoder arena (S = prompt + image tokens), its staging layers,        │ one at a
///          │            pixel_values, features + causal SDPA (mask S², queries' copy) · its GEMM │ time:
///          ├─ refs    : the RGBA handed in + the encoder's two heaps (largest reference)          ┤ peak =
///          ├─ denoise : DiT arena (P + S rows, scratch, MLP chunk), K/V arena or file, staging     │ max +
///          │            blocks (generation only), LoRA slots, reference copies                    │ crossing
///          │            + every SDPA the DiT keeps (masks, head splits, queries' copies) · GEMM   │
///          └─ decode  : the decoder's two heaps and its RGBA, then the RGB copy                    ┘
///
/// **What crosses** (alive from its birth to the end of the render, charged to every phase — the
/// earlier phases are overcharged by a few tens of MB, never undercharged): the references as handed
/// and as prepared (`Chain.render` keeps `prepared` to its end), the encoded text (`text`, read by the
/// result), the reference latents, the target latent. **The K/V do not cross here**: the prefill runs
/// inside the first step (`QwenImage21DiT.prefillPending`) and the cache dies with the DiT, before the
/// decoding (`Context.keptDenoiser = nil`); in memory they are the denoising's, on disk nobody's.
///
/// What the formula has to know of the code that an estimate by analogy would miss:
///
///   - **the K/V's place follows the prefix, not the operation** (`QwenImage21CachePolicy`): in memory
///     up to 256 MB (P ≤ 256 rows), on disk beyond. A long prompt (P > 256 text rows) moves a
///     *generation's* K/V to the file — and with it its staging blocks go (below);
///   - **the staging blocks of the map's tail exist only with the K/V in memory**
///     (`QwenImage21DiT.body`): an edit — or a long generation — streams nothing, whatever the plan's
///     tail says; a generation takes the plan's slots (`Request.tail`), each a whole layer (~436 MB bf16);
///   - **the scratch is not the prefix**: the prefill runs by groups of segments
///     (`QwenImage21DiT.layout`), so `normed`, `o`, the MLP chunk scale with the largest group; `x`,
///     `keys`, `values` and the RoPE table scale with the whole sequence `P + S`;
///   - **the text rows are the encoder's rows minus the 14 of the system turn** (`_drop_idx`), each
///     image slot is 4 rows of the DiT (one 2×2 group of latent tokens);
///   - **the DiT keeps every SDPA it built** (`attentions`, `headSplits`, by `rows×visible` key): each
///     causal text segment with its mask, each reference segment, each slice of the target cut by heads;
///   - **the modulation is tabulated before the first attention exists** (`tabulateModulation`):
///     `modulation.1` widened to fp32 (`Krea2Exact.widenedBytes`, `d × 4d`) is alive beside the arenas,
///     not beside the attentions — the larger of the two moments is charged;
///   - **the tokenizer lives through the whole text phase** (`Qwen3VL` holds it) — its two tables
///     (`Tokenizer.tableBytes`, ~20 MB) are sized from the file's counts, which no shape function
///     gives yet: **not charged** (open);
///   - **a generation's half-size DiT** (the spectral schedule, `QwenImage21Sampler`) is built and
///     released before the full one: never both, the full one is the larger;
///   - **the VAEs hand everything back before they return**: nothing of the encoder is carried
///     into the denoising any more, and the decoder's RGBA outlives its heaps only by the RGB copy.
extension MemoryNeed {
    static func qwenImage21(_ request: Request, app: Bool) -> MemoryNeed? {
        QwenImage21Need(request).need(app: app)
    }
}

/// **The shapes of one Qwen-Image-2.1 render**, derived from a `MemoryNeed.Request` as the engine
/// derives them (`Qwen3VLImages.conditionSize`, the template, the joint sequence), then charged.
package struct QwenImage21Need {
    typealias Term = MemoryNeed.Term

    /// One reference: as handed, as prepared (`QwenImage21Pipeline.reference`), and what it becomes.
    package struct Reference {
        package let original: (width: Int, height: Int)
        package let prepared: (width: Int, height: Int)
        /// The latent grid (`/16`) — also the vision tower's patch grid (patch 16).
        package var latent: (height: Int, width: Int) { (prepared.height / 16, prepared.width / 16) }
        package var patches: Int { latent.height * latent.width }
        /// `<|image_pad|>` tokens: one per 2×2 group of patches.
        package var slots: Int { patches / 4 }
    }

    /// `_drop_idx`: the system turn's tokens (`Qwen3VLPrompt.systemTurn` tokenized), not passed to the DiT.
    package static let dropCount = 14
    /// The template's text tokens around an image: `<imageN><|vision_start|>` before (the separating
    /// space merged in), `<|vision_end|>` after — the oracle's edit-2 layout has 8 text rows before
    /// image 1 (3 of `<|im_start|>user\n`) and 6 between. What `promptTokens` includes per reference.
    package static let tokensBefore = 5, tokensAfter = 1
    /// The published map is bf16 (`Family.installedSize(.standard)`): a staging block is a layer of it.
    /// The card does not say its type yet (open: an 8-bit card's blocks are about half).
    package static let weightBytes = 2

    package let request: MemoryNeed.Request
    package let references: [Reference]
    /// The encoder's sequence: the prompt with its template, plus every image token.
    package let encoderTokens: Int
    /// The DiT's text rows: the encoder's rows past the system turn, image slots excluded.
    package let textRows: Int
    package let target: (height: Int, width: Int)
    /// The DiT's joint sequence: text and references in the template's order, the target last.
    package let sequence: QwenImage21Sequence
    package let dit = QwenImage21DiT.Config(header: [:])
    package let vae = QwenImage21VAE.Config()
    package let vision = Qwen3VLVision.Config.published
    /// The P cores the head split runs on.
    package var threads = EngineSettings.effective.machine.performanceCores

    /// - `request.promptTokens`: the encoder's tokens **without the image slots** — the template
    ///   (system turn included), the prompt, each image's tags; at least what the template needs.
    package init(_ request: MemoryNeed.Request) {
        self.request = request
        references = request.references.map { r in
            Reference(original: (r.width, r.height),
                      prepared: Qwen3VLImages.conditionSize(width: r.width, height: r.height,
                                                            resolution: QwenImage21Pipeline.outputResolution))
        }
        let n = references.count
        // `<|im_start|>user\n` (3), each image's tags, and after the last image at least the closing
        // `<|im_end|>\n<|im_start|>assistant\n` (5): what the template has with an empty prompt.
        let fixed = 3 + n * (QwenImage21Need.tokensBefore + QwenImage21Need.tokensAfter) + 5
        textRows = max(request.promptTokens - QwenImage21Need.dropCount, fixed)
        encoderTokens = textRows + QwenImage21Need.dropCount + references.reduce(0) { $0 + $1.slots }
        target = (request.height / 16, request.width / 16)
        let (references, textRows, target) = (references, textRows, target)
        // Its inputs: the text rows, each reference's latent grid, the target's (`sequences`).
        let key = [textRows, target.height, target.width] + references.flatMap { [$0.latent.height, $0.latent.width] }
        sequence = QwenImage21Need.sequences(key) {
            // The joint sequence's layout: the user header, each image between its tags, the rest of the
            // text after the last image, then the target's slots.
            var slots: [Bool] = []
            func text(_ count: Int) { slots += [Bool](repeating: false, count: count) }
            text(3)
            for (i, reference) in references.enumerated() {
                text(QwenImage21Need.tokensBefore)
                slots += [Bool](repeating: true, count: reference.slots)
                if i < n - 1 { text(QwenImage21Need.tokensAfter) }
            }
            text(textRows - slots.filter { !$0 }.count)
            slots += [Bool](repeating: true, count: target.height * target.width / 4)
            return try! QwenImage21Sequence(
                slots: slots, images: references.map { .init(height: $0.latent.height, width: $0.latent.width) }
                    + [.init(height: target.height, width: target.width)])
        }
    }

    /// The joint sequences built, by their layout's inputs: `QwenImage21Sequence` lays out every row
    /// and its RoPE position (a three-reference edit at 1024×1536, 12 ms in a debug build), and the
    /// need reads only its segments and counts.
    private static let sequences = ShapeMemo<[Int], QwenImage21Sequence>()

    // ── the phases ───────────────────────────────────────────────────────────────────────────

    /// The vision tower, one image at a time in an arena sized for the largest (its `Attention` built
    /// per image and dropped with it); every image's `pixel_values` and the features of those done are
    /// held (they feed the language model).
    package var visionPhase: MemoryNeed.Phase? {
        guard let largest = references.map(\.patches).max() else { return nil }
        let (d, i, m, o) = (vision.hidden, vision.intermediate, vision.merged, vision.output)
        return .init(kind: .vision, terms: [
            Term("vision arena", Qwen3VLVision.arenaBytes(config: vision, patches: largest)),
            Term("pixel_values", pixelValues), Term("vision features", features),
        ] + MemoryNeed.attention("vision", heads: vision.heads, rows: largest, keys: largest, headDim: vision.headDim,
                                 causal: false, split: false, threads: threads)
          + MemoryNeed.gemm("vision", products: [(largest, vision.patchValues, d), (largest, d, 3 * d), (largest, d, d),
                                                 (largest, d, i), (largest, i, d), (largest / 4, m, m), (largest / 4, m, o)]))
    }

    /// The language model on the whole sequence, with its staging layers (`Request.encoderSlots`).
    package var textPhase: MemoryNeed.Phase {
        let s = encoderTokens
        let (h, kv, i) = (QwenImage21Need.encoderHidden, QwenImage21Need.encoderKV, QwenImage21Need.encoderIntermediate)
        let a = QwenImage21Need.encoderHeads * QwenImage21Need.encoderHeadDim
        return .init(kind: .text, terms: [
            Term("encoder arena", TextEncoder.arenaBytes(config: QwenImage21Need.encoderConfig, sequence: s)),
            Term("encoder staging", TextEncoder.stagingSlots(granted: request.encoderSlots) * QwenImage21Need.encoderLayerBytes),
            Term("pixel_values", pixelValues), Term("vision features", features),
        ] + MemoryNeed.attention("encoder", heads: QwenImage21Need.encoderHeads, rows: s, keys: s,
                                 headDim: QwenImage21Need.encoderHeadDim, causal: true, split: false, threads: threads)
          + MemoryNeed.gemm("encoder", products: [(s, h, a), (s, h, kv), (s, a, h), (s, h, i), (s, i, h)]))
    }

    /// The pixels of the largest prepared reference: what the encoder is planned for.
    private var largestReference: (width: Int, height: Int)? {
        references.map(\.prepared).max { $0.width * $0.height < $1.width * $1.height }
    }

    /// The VAE encoder, one reference at a time (`QwenImage21EncodingModule`, which dies on return):
    /// the RGBA array the module builds and hands in (`rgba`, `[4, H, W]`), and the encoder's two heaps
    /// and host copies (`QwenImage21VAEEncoder.plan`).
    package var referencesPhase: MemoryNeed.Phase? {
        guard let largest = largestReference else { return nil }
        return .init(kind: .references, terms: [Term("rgba", vae.inChannels * largest.width * largest.height * 4)]
            + MemoryNeed.vae("Qwen encoder", QwenImage21VAEEncoder.plan(config: vae, height: largest.height,
                                                                         width: largest.width)))
    }

    /// Where the conditions' K/V live (`QwenImage21CachePolicy`, the pipeline's choice).
    package var storage: QwenImage21DiT.CacheStorage { QwenImage21CachePolicy.storage(prefix: sequence.prefix) }

    /// The full-size DiT (the half-size one, a generation's first steps, is gone before it is built).
    package var denoisingPhase: MemoryNeed.Phase {
        let p = sequence.prefix, s = sequence.target, all = sequence.count
        let (_, scratch) = QwenImage21DiT.layout(sequence, fullPass: storage == .recompute)
        let rank = request.loraRanks.reduce(0, +)
        let largestImage = sequence.images.map { $0.height * $0.width }.max() ?? 0
        let (d, h) = (dit.dim, dit.hidden)
        var terms = [
            Term("DiT arena", QwenImage21DiT.arenaBytes(config: dit, rows: all, prefix: p, scratch: scratch,
                                                        largestImage: largestImage, rank: rank,
                                                        fullPass: storage == .recompute)),
            Term("K/V arena", p > 0 && storage == .memory ? QwenImage21DiT.cacheArenaBytes(config: dit, prefix: p) : 0),
            // The map's staging blocks: only with the K/V in memory (`QwenImage21DiT.body`).
            Term("map staging", storage == .memory && request.tail.fraction > 0 ? request.tail.slots * ditLayerBytes : 0),
        ]
        // Two slots of the user's stack, on the 7 Linears of a block (`StreamedLoRA`), and, with more
        // than one LoRA, the `down` staging of the module a slot fills (`LoRA.downBytes`).
        if rank > 0 {
            let modules: [String: (k: Int, n: Int, rank: Int)] = [
                "attn.to_q": (d, d, rank), "attn.to_k": (d, d, rank), "attn.to_v": (d, d, rank),
                "attn.to_out.0": (d, d, rank), "img_mlp.gate_layer": (d, h, rank), "img_mlp.proj": (d, h, rank),
                "img_mlp.out": (h, d, rank)]
            terms.append(Term("LoRA slots", 2 * StreamedLoRA.layout(modules).bytes))
            if request.loraRanks.count > 1 {
                terms.append(Term("LoRA down staging", LoRA.downBytes(k: h, rank: request.loraRanks.max() ?? 0)))
            }
        }
        // The sampler's copies of the reference latents (`Pages`), the velocity, the modulation table.
        terms.append(Term("sampler", references.reduce(0) { $0 + Arena.capacity(for: vae.zDim * $1.patches * 4) }
            + vae.zDim * s * 4 + QwenImage21Pipeline.defaultSteps * dit.modulation * 4))
        // The two moments past the arenas: the modulation tabulated (`modulation.1` widened, and under a
        // LoRA its `down`/`up` in `Pages`), then every attention and the GEMM. The larger is charged.
        let attentions = ditAttentions(scratch: scratch)
        let mlpRows = QwenImage21DiT.mlpRows(scratch: scratch, hidden: h)
        let running = attentions + MemoryNeed.gemm("DiT", products: [(scratch, d, d), (mlpRows, d, h), (mlpRows, h, d),
                                                                     (textRows, dit.context, d),
                                                                     (largestImage, dit.channels, d)])
        let widened = Krea2Exact.widenedBytes(inputs: d, outputs: 4 * d)
            + (rank > 0 ? Arena.capacity(for: d * rank * 4) + Arena.capacity(for: rank * 4 * d * 4) : 0)
        let runningBytes = running.reduce(0) { $0 + $1.bytes }
        terms += running
        terms.append(Term("modulation widening beyond the attentions", max(0, widened - runningBytes)))
        return .init(kind: .denoising, terms: terms)
    }

    /// The decoder's two heaps and the RGBA it returns (`QwenImage21VAEDecoder.plan`); then, its heaps
    /// handed back, the RGB copy `QwenImage21DecodingModule` takes of the RGBA (`Array(rgba.prefix(3·H·W))`)
    /// — the larger of the two moments.
    package var decodingPhase: MemoryNeed.Phase {
        let (H, W) = (vae.factor * target.height, vae.factor * target.width)
        let band = request.minimumBands ? min(QwenImage21VAEDecoder.bandPixels, MemoryPlan.qwenMinimumBand)
                                        : QwenImage21VAEDecoder.bandPixels
        let plan = QwenImage21VAEDecoder.plan(config: vae, latentHeight: target.height, latentWidth: target.width,
                                              bandPixels: band)
        let rgb = 3 * H * W * 4
        return .init(kind: .decoding, terms: MemoryNeed.vae("Qwen decoder", plan)
            + [Term("RGB copy beyond the heaps", max(0, rgb - plan.registers - plan.workspace))])
    }

    /// The references as handed and as prepared, the encoded text, the reference latents, the target latent.
    package var crossing: Int {
        let images = references.reduce(0) {
            $0 + 3 * ($1.original.width * $1.original.height + $1.prepared.width * $1.prepared.height)
        }
        let rows = encoderTokens - QwenImage21Need.dropCount
        let latents = references.reduce(0) { $0 + vae.zDim * $1.patches } + vae.zDim * target.height * target.width
        return (images + rows * QwenImage21Need.encoderHidden + latents) * 4 + rows
    }

    package func need(app: Bool) -> MemoryNeed {
        MemoryNeed(phases: [visionPhase, textPhase, referencesPhase, denoisingPhase, decodingPhase].compactMap { $0 },
                   crossing: crossing, base: OperatorFootprint.processBase(app: app))
    }

    // ── the pieces ───────────────────────────────────────────────────────────────────────────

    /// `pixel_values` of every reference (`Qwen3VLImages.Pixels`, in `Pages`).
    private var pixelValues: Int {
        references.reduce(0) { $0 + Qwen3VLImages.Pixels.valueBytes(gridHeight: $1.latent.height, gridWidth: $1.latent.width) }
    }
    /// The tower's output per image (`Qwen3VLVision.Features`, in `Pages`): the merger's and the
    /// deepstack mergers' `[slots, output]`.
    private var features: Int {
        references.reduce(0) { $0 + Qwen3VLVision.featureBytes(tokens: $1.slots, output: vision.output,
                                                               levels: vision.deepstack.count) }
    }

    /// One layer of the DiT's map, as `Artifact.span` measures a staging block: 4 attention
    /// projections `d²`, the 3 MLP weights `d·h`, the two QK-Norm weights, rounded to a page.
    private var ditLayerBytes: Int {
        let (d, h) = (dit.dim, dit.hidden)
        return Arena.capacity(for: (4 * d * d + 3 * d * h + 2 * dit.headDim) * QwenImage21Need.weightBytes)
    }

    /// **Every SDPA the DiT keeps** (`attentions`, `headSplits`: one per cache key, alive to the end of
    /// the denoising), enumerated as `QwenImage21DiT.attend` builds them — each slice's rows and visible
    /// keys, the causal text segments with their mask, the target cut by heads.
    private func ditAttentions(scratch: Int) -> [Term] {
        let heads = dit.heads, headDim = dit.headDim, p = sequence.prefix, all = sequence.count
        var terms: [Term] = []
        var built = Set<String>()
        func attend(count: Int, keys: Int, causal: Bool, target: Bool) {
            let slices = QwenImage21DiT.attentionSlices(heads: heads, queries: count, keys: keys)
            for piece in 0..<slices {
                let end = (piece + 1) * count / slices, rows = end - piece * count / slices
                let visible = causal ? keys - count + end : keys
                let split = target && !Attention.useHalfPrecision && rows >= HeadSplitAttention.minimumSequence
                // `attentions` and `headSplits` are two dictionaries: the same key may live in both.
                let key = "\(rows)×\(visible)\(causal ? "c" : "")"
                guard built.insert((split ? "s" : "a") + key).inserted else { continue }
                terms += MemoryNeed.attention("DiT \(key)", heads: heads, rows: rows, keys: visible, headDim: headDim,
                                              causal: causal, split: split, threads: threads)
            }
        }
        for segment in sequence.segments {
            attend(count: segment.end - segment.start, keys: segment.end, causal: segment.isText, target: false)
        }
        attend(count: all - p, keys: all, causal: false, target: true)
        return terms
    }

    // ── Qwen3-VL-8B's language model, as `TextEncoder` sizes it ─────────────────────────────

    /// Qwen3-VL-8B's text config (`text_config` of the encoder's map): hidden = attention width 4096,
    /// 32 heads of 128, 8 K/V heads, MLP 12 288.
    private static let encoderHidden = 4096, encoderHeads = 32, encoderHeadDim = 128, encoderKV = 1024,
                       encoderIntermediate = 12_288

    /// Qwen3-VL-8B's text config as published, through the encoder's own parser: the formula does not
    /// open the card, and the arena then comes from the very function `TextEncoder.init` calls.
    private static let encoderConfig = try! TextEncoder.Config(
        ["config": ["hidden_size": encoderHidden, "num_attention_heads": encoderHeads,
                    "num_key_value_heads": encoderKV / encoderHeadDim, "head_dim": encoderHeadDim,
                    "intermediate_size": encoderIntermediate]], lastLayer: 35)

    /// One layer of the encoder's map (bf16), a staging buffer of `TextEncoder.encode`: q, o `h·a`,
    /// k, v `h·kv`, the 3 MLP weights `h·i`, two RMSNorms and the QK-Norms.
    private static var encoderLayerBytes: Int {
        let (h, kv, i) = (encoderHidden, encoderKV, encoderIntermediate)
        return Arena.capacity(for: (2 * h * h + 2 * h * kv + 3 * h * i + 2 * h + 2 * encoderHeadDim) * weightBytes)
    }
}
