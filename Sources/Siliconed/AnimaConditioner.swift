import Accelerate
import Foundation

/// Anima's **conditioner** (`llm_adapter`): learned T5 queries that go and read Qwen.
///
///     T5 ids ─ embed [32128, 1024] ─┬─ ×6 : x += SelfAttn(RMS(x))              RoPE on both
///                                   │       x += CrossAttn(RMS(x), Qwen)        RoPE on both, own positions
///     last_hidden_state Qwen ───────┘       x += W₂ GELU(W₁ RMS(x) + b₁) + b₂
///                                   ─ out_proj (+b) ─ RMSNorm ─ zero-padded to 512
///
/// What a port by analogy would miss:
///
///   - the **cross**-attention **carries a RoPE** — the queries rotate at their T5 position, the
///     keys at their Qwen position: it is not the DiT's cross-attention, which has none;
///   - the RoPE is **split** (`rotate_half`, pairs `(j, j+32)`), `θ = 10,000`, heads of 64;
///     the weights do not go through a forge, so it is applied as-is here;
///   - per-head QK-Norm, eps 1e-6; the MLPs and `out_proj` **have a bias**, the attentions do not.
///
/// **Everything is computed in double**, and it is free: forty tokens, six blocks, once per
/// render. This is the rule — more precision, never less — with nothing to trade off.
///
/// **Its working memory is one arena** (`slices`, `arenaBytes`), mapped for a call of `condition` and
/// unmapped on its return: the activations in double, one widened weight (`mlp.0`, `4096 × 1024`,
/// 32 MB of doubles, and its fp32 widening beside), the scores. They were Swift arrays, a fresh one
/// per weight and per activation — up to ~80 MB of `malloc` at 512 tokens, outside every shape
/// function, left in the footprint by `free` (`Pages`). The same values in the same order: only where
/// they live moved.
///
/// The weights are read from the published Turbo file, without a forge: 0.13 G parameters read once,
/// of which only the `embed` table rows of the prompt tokens are touched.
package final class AnimaConditioner {
    /// The adapter's published shapes — what its arena is sized on (`slices`).
    package enum Shape {
        package static let dim = 1024, heads = 16, headDim = 64, blocks = 6, hidden = 4096
        /// The T5 tokens are at most the prompt's bound and `</s>` (`AnimaTextModule` refuses past it),
        /// and the output is padded with zeros to 512: the DiT's text is at least 512 rows — a
        /// minimum of the model, where diffusers' cut at 512 made it look like a fixed length.
        package static let maximumTokens = Request.maximumPromptTokens + 1, minimumLength = 512
    }
    package let dim = Shape.dim, heads = Shape.heads, headDim = Shape.headDim, blocks = Shape.blocks, hidden = Shape.hidden
    package let minimumLength = Shape.minimumLength
    private let weights: Safetensors
    private let prefix: String
    package var recordBoundaries = false
    package private(set) var boundaries: [String: [Float]] = [:]

    /// - Parameter path: the ComfyUI Turbo file (`anima-turbo-v1.1.safetensors`), which mixes the
    ///   DiT and the adapter under `model.diffusion_model.llm_adapter.`.
    package init(path: String) throws {
        weights = try Safetensors(path: path)
        prefix = "model.diffusion_model.llm_adapter."
        guard weights.entries[prefix + "embed.weight"]?.shape == [32128, 1024] else {
            throw Safetensors.Failure.badHeader("\(path): no Anima adapter")
        }
    }

    // ── sizing: what `condition` maps, and the memory need charges ──────────────────────────────

    /// **The arena's slices, in bytes, in the order `condition` reserves them**, for `t5` query rows
    /// reading `qwen` rows: in double, `source [qwen, d]`, `x`, `normed` (the three RMS inputs, one at a
    /// time), `q`, `attended`, `projected` (an attention's or the MLP's output, before it is added) at
    /// `[t5, d]`; `k`, `v` at the larger of the two lengths (self: `t5`, cross: `qwen`); the MLP's
    /// `[t5, 4096]`; one weight `[4096, 1024]` (the largest: `mlp.0`, `mlp.2`), its bias (4096), a norm
    /// (1024), one row of scores; in fp32, the weight's widening from the file and the embedding row.
    package static func slices(t5 t: Int, qwen q: Int) -> [(name: String, bytes: Int)] {
        let d = Shape.dim, h = Shape.hidden, c = max(t, q)
        let w = MemoryLayout<Double>.size, f = MemoryLayout<Float>.size
        return [("source", q * d * w), ("x", t * d * w), ("normed", t * d * w), ("q", t * d * w), ("k", c * d * w),
                ("v", c * d * w), ("attended", t * d * w), ("projected", t * d * w), ("hidden", t * h * w),
                ("weight", h * d * w), ("bias", h * w), ("norm", d * w), ("scores", c * w),
                ("widened", h * d * f), ("row", d * f)]
    }

    /// **The arena a call of `condition` maps** for `t5` query rows and `qwen` rows: its slices, each
    /// on its page.
    package static func arenaBytes(t5: Int, qwen: Int) -> Int {
        Arena.capacity(slices: slices(t5: t5, qwen: qwen).map(\.bytes))
    }

    /// The arena's slices, as `condition` uses them.
    private struct Work {
        let source, x, normed, q, k, v, attended, projected, hidden: UnsafeMutablePointer<Double>
        let weight, bias, norm, scores: UnsafeMutablePointer<Double>
        let widened, row: UnsafeMutablePointer<Float>
        /// What `weight` and `widened` hold, in values; `bias` and `norm` theirs.
        let weightCapacity, biasCapacity, normCapacity: Int

        init(arena: Arena, t5: Int, qwen: Int) throws {
            var named: [String: UnsafeMutableRawPointer] = [:]
            for slice in AnimaConditioner.slices(t5: t5, qwen: qwen) {
                named[slice.name] = try arena.reserve(slice.name, bytes: slice.bytes)
            }
            func wide(_ n: String) -> UnsafeMutablePointer<Double> { named[n]!.assumingMemoryBound(to: Double.self) }
            source = wide("source"); x = wide("x"); normed = wide("normed"); q = wide("q"); k = wide("k"); v = wide("v")
            attended = wide("attended"); projected = wide("projected"); hidden = wide("hidden")
            weight = wide("weight"); bias = wide("bias"); norm = wide("norm"); scores = wide("scores")
            widened = named["widened"]!.assumingMemoryBound(to: Float.self)
            row = named["row"]!.assumingMemoryBound(to: Float.self)
            weightCapacity = Shape.hidden * Shape.dim; biasCapacity = Shape.hidden; normCapacity = Shape.dim
        }
    }

    // ── the computation ─────────────────────────────────────────────────────────────────────

    /// A tensor of the adapter, widened from the file into `widened` then to double into `into`.
    private func load(_ name: String, into destination: UnsafeMutablePointer<Double>, capacity: Int,
                      work: Work) throws {
        guard let count = weights.materialize(prefix + name, into: work.widened, capacity: min(capacity, work.weightCapacity)) else {
            throw Safetensors.Failure.badHeader("adapter: \(name) missing, or larger than the published shapes (`Shape`)")
        }
        vDSP_vspdp(work.widened, 1, destination, 1, vDSP_Length(count))
    }

    /// `y[r, o] = Σ x[r, i] · W[o, i] (+ b[o])` — `W` in `nn.Linear` layout, `[output, input]`. `y` is
    /// cleared (or filled with the bias) first, as a fresh array was.
    private func linear(_ x: UnsafePointer<Double>, rows: Int, _ name: String, inputs: Int, outputs: Int,
                        bias: Bool = false, into y: UnsafeMutablePointer<Double>, work: Work) throws {
        try load(name + ".weight", into: work.weight, capacity: work.weightCapacity, work: work)
        vDSP_vclrD(y, 1, vDSP_Length(rows * outputs))
        if bias {
            try load(name + ".bias", into: work.bias, capacity: work.biasCapacity, work: work)
            for r in 0..<rows { for o in 0..<outputs { y[r * outputs + o] = work.bias[o] } }
        }
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, Int32(rows), Int32(outputs), Int32(inputs),
                    1, x, Int32(inputs), work.weight, Int32(inputs), bias ? 1 : 0, y, Int32(outputs))
    }

    /// `nn.RMSNorm`: `x · rsqrt(mean(x²) + eps) · w`, per slice of `width`, from `x` into `y` (the same
    /// buffer for the QK-Norms: each value is read before it is written).
    private func rmsNorm(_ x: UnsafePointer<Double>, count: Int, width: Int, weight: UnsafePointer<Double>,
                         into y: UnsafeMutablePointer<Double>, eps: Double = 1e-6) {
        for start in stride(from: 0, to: count, by: width) {
            var sum = 0.0
            for j in 0..<width { sum += x[start + j] * x[start + j] }
            let scale = 1 / (sum / Double(width) + eps).squareRoot()
            for j in 0..<width { y[start + j] = x[start + j] * scale * weight[j] }
        }
    }

    /// The split RoPE of `AnimaRotaryEmbedding`: the angle `p · θ^(−2j/64)` rotates `(j, j+32)`.
    private func rope(_ x: UnsafeMutablePointer<Double>, rows: Int) {
        let half = headDim / 2
        let table = Self.ropeTable(rows: rows, headDim: headDim)
        for r in 0..<rows {
            for h in 0..<heads {
                let base = (r * heads + h) * headDim
                for j in 0..<half {
                    let (c, s) = (Double(table[(r * half + j) * 2]), Double(table[(r * half + j) * 2 + 1]))
                    let (a, b) = (x[base + j], x[base + j + half])
                    x[base + j] = a * c - b * s
                    x[base + j + half] = b * c + a * s
                }
            }
        }
    }

    /// **`AnimaRotaryEmbedding`'s table**, `[rows, headDim/2, (cos, sin)]` in float32, as the reference
    /// computes it before the rotation (which runs in double here): `inv_freq = 1/θ^(2j/64)` in float32,
    /// `inv_freq @ position.float()` — a matmul of inner size 1, one float32 product —, `.cos()` /
    /// `.sin()` (SLEEF in torch: a one-bit residue, `Ops.polar`). Until 18/10 the angle and its cosine
    /// were in double: the witness's table, not the reference's.
    package static func ropeTable(rows: Int, headDim: Int = Shape.headDim, theta: Float = 10000) -> [Float] {
        let half = headDim / 2
        let inverse = (0..<half).map { 1 / powf(theta, Float(2 * $0) / Float(headDim)) }
        var table = [Float](repeating: 0, count: rows * half * 2)
        for r in 0..<rows {
            for j in 0..<half {
                let angle = Float(r) * inverse[j]
                table[(r * half + j) * 2] = cosf(angle); table[(r * half + j) * 2 + 1] = sinf(angle)
            }
        }
        return table
    }

    /// One attention of `rows` queries over `contextRows` keys, its `o_proj` written into `output`.
    private func attention(_ name: String, query x: UnsafePointer<Double>, rows: Int, context: UnsafePointer<Double>,
                           contextRows: Int, into output: UnsafeMutablePointer<Double>, work: Work) throws {
        let (q, k, v) = (work.q, work.k, work.v)
        try linear(x, rows: rows, name + ".q_proj", inputs: dim, outputs: dim, into: q, work: work)
        try linear(context, rows: contextRows, name + ".k_proj", inputs: dim, outputs: dim, into: k, work: work)
        try linear(context, rows: contextRows, name + ".v_proj", inputs: dim, outputs: dim, into: v, work: work)
        try load(name + ".q_norm.weight", into: work.norm, capacity: work.normCapacity, work: work)
        rmsNorm(q, count: rows * dim, width: headDim, weight: work.norm, into: q)
        try load(name + ".k_norm.weight", into: work.norm, capacity: work.normCapacity, work: work)
        rmsNorm(k, count: contextRows * dim, width: headDim, weight: work.norm, into: k)
        rope(q, rows: rows)
        rope(k, rows: contextRows)
        let scale = 1 / Double(headDim).squareRoot()
        let out = work.attended, scores = work.scores
        vDSP_vclrD(out, 1, vDSP_Length(rows * dim))
        for h in 0..<heads {
            for r in 0..<rows {
                var peak = -Double.infinity
                for c in 0..<contextRows {
                    var dot = 0.0
                    for j in 0..<headDim { dot += q[r * dim + h * headDim + j] * k[c * dim + h * headDim + j] }
                    scores[c] = dot * scale
                    peak = max(peak, scores[c])
                }
                var total = 0.0
                for c in 0..<contextRows { scores[c] = exp(scores[c] - peak); total += scores[c] }
                for c in 0..<contextRows {
                    let p = scores[c] / total
                    for j in 0..<headDim { out[r * dim + h * headDim + j] += p * v[c * dim + h * headDim + j] }
                }
            }
        }
        try linear(out, rows: rows, name + ".o_proj", inputs: dim, outputs: dim, into: output, work: work)
    }

    private func record(_ name: String, _ x: UnsafePointer<Double>, count: Int) {
        guard recordBoundaries else { return }
        boundaries[name] = (0..<count).map { Float(x[$0]) }
    }

    /// - Parameters:
    ///   - t5: the T5 identifiers, `</s>` included.
    ///   - qwen: Qwen3-0.6B's `last_hidden_state`, `[qwenRows, 1024]`.
    /// - Returns: `[max(512, t5.count), 1024]`, the rows beyond the T5 tokens zero — they
    ///   **count** in the softmax of the DiT's cross-attention, which does not mask them.
    package func condition(t5: [Int], qwen: UnsafeBufferPointer<Float>, qwenRows: Int) throws -> [Float] {
        let rows = t5.count
        // The arena is sized for at most `Shape.maximumTokens` queries, what the need charges.
        guard rows <= Shape.maximumTokens else {
            throw Safetensors.Failure.badHeader("T5: \(rows) tokens, at most \(Shape.maximumTokens) (`T5Tokenizer.encode`)")
        }
        guard qwen.count == qwenRows * dim else {
            throw Safetensors.Failure.badHeader("Qwen: \(qwen.count) values for \(qwenRows) × \(dim)")
        }
        // The working memory of this call, unmapped on its return (`slices`).
        let arena = try Arena(capacity: AnimaConditioner.arenaBytes(t5: rows, qwen: qwenRows))
        let work = try Work(arena: arena, t5: rows, qwen: qwenRows)
        let (source, x) = (work.source, work.x)
        vDSP_vspdp(qwen.baseAddress!, 1, source, 1, vDSP_Length(qwen.count))

        // The table has 32,128 rows: we only expand those of the prompt.
        // The adapter is written by the forge in its published type (bf16, fp16, fp32), or fp32 when
        // it came 8-bit (`ForgeDiT.writeAdapter` stores a dequantized tensor in fp32, never rounded).
        guard let table = weights.pointer(prefix + "embed.weight"),
              let tableType = weights.entries[prefix + "embed.weight"]?.dtype, ["BF16", "F16", "F32"].contains(tableType) else {
            throw Safetensors.Failure.badHeader("embed: bf16, fp16 or fp32 expected")
        }
        let row = work.row
        for (r, id) in t5.enumerated() {
            guard id >= 0, id < 32128 else { throw Safetensors.Failure.badHeader("T5 token \(id) outside the vocabulary") }
            if tableType == "F32" {
                UnsafeMutableRawPointer(row).copyMemory(from: table.advanced(by: id * dim * 4), byteCount: dim * 4)
            } else if tableType == "F16" {
                Widen.float16ToFloat32(table.advanced(by: id * dim * 2), count: dim, into: row)
            } else {
                Widen.bfloat16ToFloat32(source: table.advanced(by: id * dim * 2), destination: UnsafeMutableRawPointer(row),
                                        count: dim)
            }
            for j in 0..<dim { x[r * dim + j] = Double(row[j]) }
        }

        let n = rows * dim
        let (normed, projected, hiddenRows) = (work.normed, work.projected, work.hidden)
        for b in 0..<blocks {
            let p = "blocks.\(b)."
            try load(p + "norm_self_attn.weight", into: work.norm, capacity: work.normCapacity, work: work)
            rmsNorm(x, count: n, width: dim, weight: work.norm, into: normed)
            try attention(p + "self_attn", query: normed, rows: rows, context: normed, contextRows: rows,
                          into: projected, work: work)
            for i in 0..<n { x[i] += projected[i] }
            try load(p + "norm_cross_attn.weight", into: work.norm, capacity: work.normCapacity, work: work)
            rmsNorm(x, count: n, width: dim, weight: work.norm, into: normed)
            try attention(p + "cross_attn", query: normed, rows: rows, context: source, contextRows: qwenRows,
                          into: projected, work: work)
            for i in 0..<n { x[i] += projected[i] }
            try load(p + "norm_mlp.weight", into: work.norm, capacity: work.normCapacity, work: work)
            rmsNorm(x, count: n, width: dim, weight: work.norm, into: normed)
            try linear(normed, rows: rows, p + "mlp.0", inputs: dim, outputs: hidden, bias: true, into: hiddenRows, work: work)
            for i in 0..<(rows * hidden) { hiddenRows[i] = 0.5 * hiddenRows[i] * (1 + erf(hiddenRows[i] / 2.0.squareRoot())) }
            try linear(hiddenRows, rows: rows, p + "mlp.2", inputs: hidden, outputs: dim, bias: true, into: projected, work: work)
            for i in 0..<n { x[i] += projected[i] }
            record("cond_block\(b)_out", x, count: n)
        }
        try linear(x, rows: rows, "out_proj", inputs: dim, outputs: dim, bias: true, into: projected, work: work)
        try load("norm.weight", into: work.norm, capacity: work.normCapacity, work: work)
        rmsNorm(projected, count: n, width: dim, weight: work.norm, into: normed)

        var out = [Float](repeating: 0, count: max(minimumLength, rows) * dim)
        for i in 0..<n { out[i] = Float(normed[i]) }
        withExtendedLifetime(arena) {}
        return out
    }
}
