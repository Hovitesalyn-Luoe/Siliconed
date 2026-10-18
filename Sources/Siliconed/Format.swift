import Foundation

// **The formats a render accepts** (`Format`): the per-side floor, the multiple of 16, the area
// ceiling, and what a family adds — a coarser multiple, a shorter longest side. The request that
// carries them is in `Request.swift`.

/// **The formats a render accepts — three rules, each with its reason.**
///
/// 1. **Each side ≥ 512 px** (latent ≥ 64). This is the repository's floor
///    (`Spectral.minimumSide`), read **per side** and not per area: 832×1216 has the area of
///    1024², but a 416×608 would have that of 512² with one side outside the models' domain. The
///    floor applies to *every* evaluation — hence the spectral schedule's rule, which goes down
///    only if both half-sizes stay at the floor (`Spectral.divisible`).
/// 2. **Each side a multiple of 16**: the VAE divides by 8, the DiTs' patch by 2.
/// 3. **Area at most `maxSurface`** — measured, not deduced. See its note.
///
/// **A family may ask more** (`check(width:height:family:)`, what the preflight applies), each read
/// in its reference:
///
/// - **Qwen-Image-2.1: each side a multiple of 32** — its VAE divides by 16, the patch by 2.
///   `pipeline_qwenimage21.py` floors a side to 32 without a word; the engine refuses it instead,
///   before any computation: 848×1216 would otherwise die after the encoder.
/// - **Anima: each side at most 2048 px** — the reference's RoPE table holds 128 positions per side
///   (`AnimaDiT.Config.ropePositions`); past it the reference fails, and the engine would extrapolate.
package enum Format {
    /// The smallest side, in pixels: `Spectral.minimumSide` latent cells of 8 px.
    package static let minimumSide = Spectral.minimumSide * 8
    package static let multiple = 16
    /// **The area ceiling, in pixels: that of 1024×1536, measured.**
    ///
    /// The 16 GB criterion had only been measured at 1024², and an app lets users type 2048×2048.
    /// A measurement rendered the three models at 832×1216 and 1216×832 (the area of 1024² within 3.5 %):
    /// peak of 4.8 GB (Z-Image), 5.3 GB (Anima), 5.2 GB (Krea 2). Then at 1024×1536, ×1.5:
    /// Z-Image and Anima passed, but **Krea 2 died** in its first SDPA — 48 heads over 6,156
    /// tokens, a 7.3 GB matrix that `MPSGraph` split itself by committing into our buffer. Since
    /// then Krea 2's SDPA splits its own queries beyond `Attention.maxMatrix`, and the three
    /// models fit at 1024×1536 — measured peaks (`/usr/bin/time -l`, footprint): **Krea 2
    /// 7.8 GB** (439 s), **Z-Image 7.5 GB** (with PopArt, 174 s), **Anima 7.6 GB** (91 s), all at
    /// decoding. Beyond that, nothing is measured: the ceiling is the last format that was.
    package static let maxSurface = 1024 * 1536

    /// The longest side the area ceiling leaves above the floor: 1536·1024 / 512 = 3072 px.
    package static let maximumSide = maxSurface / minimumSide

    /// **The multiple each side of `family` must be**: 32 for Qwen-Image-2.1 (VAE ×16, patch ×2), 16
    /// otherwise.
    package static func multiple(for family: Family) -> Int {
        family == .qwenImage21 ? 32 : multiple
    }

    /// **The longest side `family` renders**, in pixels: Anima's RoPE table (128 positions × patch 2 ×
    /// VAE 8 = 2048), `maximumSide` otherwise.
    package static func maximumSide(for family: Family) -> Int {
        switch family {
        case .anima:
            return min(maximumSide, AnimaDiT.Config.publishedRopePositions * 2 * LatentSpace.qwenImage.factor)
        case .zImage, .krea2, .qwenImage21: return maximumSide
        }
    }

    /// Throws if the format is not renderable by any model. The order of the rules is that of the
    /// most useful error.
    package static func check(width: Int, height: Int) throws(EngineError) {
        try check(width: width, height: height, multiple: multiple, maximumSide: maximumSide)
    }

    /// **Throws if `family` does not render the format** — the general rules, then the family's own:
    /// what the preflight checks, what the app's size field judges.
    package static func check(width: Int, height: Int, family: Family) throws(EngineError) {
        try check(width: width, height: height, multiple: multiple(for: family), maximumSide: maximumSide(for: family))
    }

    private static func check(width: Int, height: Int, multiple: Int, maximumSide side: Int) throws(EngineError) {
        guard width >= minimumSide, height >= minimumSide else {
            throw .imageTooSmall(side: min(width, height), minimum: minimumSide)
        }
        guard width % multiple == 0, height % multiple == 0 else {
            throw .formatRefused(width: width, height: height, reason: .notMultiple, bound: multiple)
        }
        // Each side bounded BEFORE the product: `99999999984 × 99999999984` would overflow.
        guard width <= maximumSide, height <= maximumSide, width * height <= maxSurface else {
            throw .formatRefused(width: width, height: height, reason: .tooLarge, bound: maxSurface)
        }
        guard width <= side, height <= side else {
            throw .formatRefused(width: width, height: height, reason: .sideTooLong, bound: side)
        }
    }

    /// **`LxH`: width × height**, the order of Draw Things and of screens — `832x1216` is a
    /// portrait. `×` counts as `x`; a lone integer is a square. Checks nothing but the syntax.
    package static func parse(_ text: String) -> (width: Int, height: Int)? {
        let pieces = text.lowercased().replacingOccurrences(of: "×", with: "x")
            .split(separator: "x", omittingEmptySubsequences: false)
        switch pieces.count {
        case 1: return Int(pieces[0]).map { ($0, $0) }
        case 2:
            guard let l = Int(pieces[0]), let h = Int(pieces[1]) else { return nil }
            return (l, h)
        default: return nil
        }
    }

    /// `512` for a square, `832x1216` otherwise — what a file name carries. Square names do not
    /// change: the close-out and the journal find them again.
    package static func fileLabel(width: Int, height: Int) -> String {
        width == height ? "\(width)" : "\(width)x\(height)"
    }

    /// `512²` or `832×1216` — what is displayed.
    package static func label(width: Int, height: Int) -> String {
        width == height ? "\(width)²" : "\(width)×\(height)"
    }
}
