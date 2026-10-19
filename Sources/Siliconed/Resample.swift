import Accelerate
import Foundation

/// The two operators of the former spectral schedule, 𝓟 and 𝓘 — no render uses them since the half-size
/// start (𝓑 and 𝓛 below) replaced them; the resampling check still holds them to `torch`.
///
///     x_{k+1} = x_k + Δσ · 𝓘 v_θ(𝓟 x_k, σ_k)
///
/// The state stays at full resolution from the first step to the last; only the **evaluation** goes down.
/// A scheme that downsampled the state then upsampled it back would pay the interpolation at every step,
/// cumulatively — here it never touches what is kept.
///
/// **The two operators are not symmetric, and that is not an inconsistency**: 𝓟 sees a
/// state dominated by noise, 𝓘 a velocity dominated by signal.
///
/// - 𝓟 **decimates**, it does not filter. At σ = 1 the latent *is* white noise: an operator that
///   averages preserves the mean while destroying the variance (bilinear + antialias gives a
///   standard deviation of 0.3166 against 1.0022 expected, i.e. ×3.2 too small — and the model then receives an
///   out-of-distribution input). Decimation keeps the marginal law **exactly**: it is a
///   subset of i.i.d. samples. It folds the top of the spectrum onto the bottom, which is
///   precisely what the spectral argument says does not matter in the early steps, where the noise
///   still covers the high band.
/// - 𝓘 **interpolates**, because we are spreading a low-frequency velocity without removing anything. A nearest-neighbor
///   upsampling would leave 2×2 steps in the velocity field, which the following
///   high-resolution steps would have to erase.
///
/// **Both conventions are read off `torch`, never deduced** (the resampling oracle's golden
/// `goldens-resample.safetensors`, checked by the resampling check): being off by half a pixel
/// gives a latent just as white and an image just as plausible — a defect no norm
/// reports.
package enum Resample {
    /// 𝓟 — `F.interpolate(mode: "nearest-exact")` at exactly a factor of two.
    ///
    /// **Parity is the trap.** `nearest-exact` places the sample at the center of the pixel:
    /// `src = ⌊(j + ½)·2⌋ = 2j + 1`. It therefore takes the **odd** indices, `x[1::2, 1::2]` — where
    /// `nearest`, which computes `⌊2j⌋`, would take the even ones. Half a pixel off, every other sample,
    /// and nothing in the final image to say so.
    ///
    /// - Parameters:
    ///   - source: `[C, 2h, 2w]`, laid out by rows
    ///   - destination: `[C, h, w]`, owned by the caller
    ///   - height, width: the **destination** grid
    package static func decimate(_ source: UnsafePointer<Float>,
                                into destination: UnsafeMutablePointer<Float>,
                                channels: Int, height: Int, width: Int) {
        let fullHeight = height * 2, fullWidth = width * 2
        for c in 0..<channels {
            let plane = source + c * fullHeight * fullWidth
            let out = destination + c * height * width
            for y in 0..<height {
                let row = plane + (2 * y + 1) * fullWidth
                let target = out + y * width
                // `2x + 1`, so every other read starting from the second: `vDSP_vgathr` would be
                // of no use, the constant stride is enough.
                for x in 0..<width { target[x] = row[2 * x + 1] }
            }
        }
    }

    /// 𝓘 — `F.interpolate(mode: "bilinear", align_corners: false)` at exactly a factor of two.
    ///
    /// `src = ½·j − ¼`, **clamped to zero if negative** (the edge clause of
    /// `area_pixel_compute_source_index`, which is not in the documentation), then
    /// `i₁ = min(i₀ + 1, n − 1)`. The weights therefore alternate ¾/¼ in the interior, and the two edges
    /// copy: `output[0] = input[0]`, `output[2n−1] = input[n−1]`.
    ///
    /// The order of operations is the one `Interp<n>::eval` describes — `h₀·(w₀·i₀₀ + w₁·i₀₁) +
    /// h₁·(w₀·i₁₀ + w₁·i₁₁)`, horizontal innermost.
    ///
    /// **There remains 1 ulp of deviation from `torch`, on 23 % of the elements, and we stop there.** Four
    /// orders were tried against the golden — this one, FMA accumulation, everything in
    /// double with a single rounding, and the two `a + (b − a)·w` forms — and this one is the
    /// closest; none drops to zero. The difference is 4.8·10⁻⁷ in absolute terms on values
    /// of order 1, i.e. **a hundred times below the bench noise floor** (`model_out` at 1.05·10⁻⁵) and
    /// ten thousand times below the guided pass's threshold (5·10⁻⁴). It therefore cannot overturn any
    /// verdict, and chasing it further would mean paying to learn how `torch` associates a sum
    /// of four terms.
    ///
    /// The resampling check therefore judges 𝓘 **to within one ulp** and 𝓟 **to the bit** — two different
    /// criteria because two different operators: 𝓟 copies samples, and a
    /// copy has no ulp.
    ///
    /// - Parameters:
    ///   - source: `[C, h, w]`
    ///   - destination: `[C, 2h, 2w]`
    ///   - height, width: the **source** grid
    package static func bilinearDouble(_ source: UnsafePointer<Float>,
                                      into destination: UnsafeMutablePointer<Float>,
                                      channels: Int, height: Int, width: Int) {
        // One table per axis: the factor is the same, the length is not. When square, the two
        // tables are identical and so is the computation, to the bit.
        func table(_ n: Int) -> (index0: [Int], index1: [Int], weight1: [Float]) {
            var index0 = [Int](repeating: 0, count: 2 * n)
            var index1 = [Int](repeating: 0, count: 2 * n)
            var weight1 = [Float](repeating: 0, count: 2 * n)
            for j in 0..<(2 * n) {
                let src = max(0.5 * Double(j) - 0.25, 0)
                let floored = Int(src)
                index0[j] = floored
                index1[j] = min(floored + 1, n - 1)
                weight1[j] = Float(src - Double(floored))
            }
            return (index0, index1, weight1)
        }
        let rows = table(height), columns = table(width)
        let fullHeight = height * 2, fullWidth = width * 2
        for c in 0..<channels {
            let plane = source + c * height * width
            let out = destination + c * fullHeight * fullWidth
            for y in 0..<fullHeight {
                let row0 = plane + rows.index0[y] * width
                let row1 = plane + rows.index1[y] * width
                let h1 = rows.weight1[y], h0 = 1 - h1
                let target = out + y * fullWidth
                for x in 0..<fullWidth {
                    let a = columns.index0[x], b = columns.index1[x]
                    let w1 = columns.weight1[x], w0 = 1 - w1
                    target[x] = h0 * (w0 * row0[a] + w1 * row0[b])
                             + h1 * (w0 * row1[a] + w1 * row1[b])
                }
            }
        }
    }

    // ── the half-size start (Z-Image's `Sampler`, Qwen-Image-2.1's `QwenImage21Sampler`, Anima's `denoise`) ──

    /// 𝓑 — the 2×2 mean, `[C, 2h, 2w]` → `[C, h, w]`, of `a + scale·b` (`b` may be `a`'s noise).
    ///
    /// Not 𝓟: the half-size start feeds its small grid `𝓑x + σ𝓑ε`, whose noise `2𝓑ε` is the full
    /// latent's own low band (white, variance σ²) where `𝓟x` hands the model an alias of the high band.
    package static func average(_ a: UnsafePointer<Float>, plus scale: Float, times b: UnsafePointer<Float>,
                               into destination: UnsafeMutablePointer<Float>,
                               channels: Int, height: Int, width: Int) {
        let fullWidth = width * 2
        for c in 0..<channels {
            let offset = c * height * 2 * fullWidth
            for y in 0..<height {
                let top = offset + 2 * y * fullWidth, bottom = top + fullWidth
                let target = destination + c * height * width + y * width
                for x in 0..<width {
                    let i = top + 2 * x, j = bottom + 2 * x
                    let sum = (a[i] + scale * b[i]) + (a[i + 1] + scale * b[i + 1])
                        + (a[j] + scale * b[j]) + (a[j + 1] + scale * b[j + 1])
                    target[x] = 0.25 * sum
                }
            }
        }
    }

    /// 𝓛 — ×2 upsampling by Lanczos-3 (a = 3, six taps), on bilinear's grid (`src = ½·j − ¼`), the
    /// edges replicated, the taps normalized to a sum of one. What it upsamples is an estimate x̂₀:
    /// bilinear lets the spectral images of its upper band through, the full steps would read them
    /// as texture.
    package static func lanczosDouble(_ source: UnsafePointer<Float>,
                                     into destination: UnsafeMutablePointer<Float>,
                                     channels: Int, height: Int, width: Int) {
        func kernel(_ x: Double) -> Double {
            if x == 0 { return 1 }
            if abs(x) >= 3 { return 0 }
            let p = Double.pi * x
            return 3 * sin(p) * sin(p / 3) / (p * p)
        }
        func table(_ n: Int) -> [[(index: Int, weight: Float)]] {
            (0..<(2 * n)).map { j in
                let src = 0.5 * Double(j) - 0.25
                let base = Int(src.rounded(.down))
                let taps = (base - 2...base + 3).map { i in (min(max(i, 0), n - 1), kernel(src - Double(i))) }
                let total = taps.reduce(0) { $0 + $1.1 }
                return taps.map { (index: $0.0, weight: Float($0.1 / total)) }
            }
        }
        let rows = table(height), columns = table(width)
        let fullWidth = width * 2
        var horizontal = [Float](repeating: 0, count: height * fullWidth)
        for c in 0..<channels {
            let plane = source + c * height * width
            for y in 0..<height {
                for x in 0..<fullWidth {
                    var sum: Float = 0
                    for tap in columns[x] { sum += tap.weight * plane[y * width + tap.index] }
                    horizontal[y * fullWidth + x] = sum
                }
            }
            let out = destination + c * height * 2 * fullWidth
            for y in 0..<(2 * height) {
                let target = out + y * fullWidth
                for x in 0..<fullWidth { target[x] = 0 }
                for tap in rows[y] {
                    for x in 0..<fullWidth { target[x] += tap.weight * horizontal[tap.index * fullWidth + x] }
                }
            }
        }
    }
}
