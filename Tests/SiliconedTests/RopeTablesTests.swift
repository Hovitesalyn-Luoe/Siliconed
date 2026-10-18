import XCTest
@testable import Siliconed

/// **The RoPE tables follow each reference's arithmetic** (18/10): a few values printed by the reference
/// classes themselves (torch 2.14 on the CPU, diffusers 0.41.0), as bit patterns, at the far end of
/// an axis — where an angle computed another way (float32 for float64, double for float32, `θ^(−x)` for
/// `1/θ^x`) moves the most. The full tables are judged by the repository's `rope` check.
final class RopeTablesTests: XCTestCase {

    private func bits(_ c: Float, _ s: Float) -> [UInt32] { [c.bitPattern, s.bitPattern] }

    /// Z-Image's DiT: float64 angles rounded once, then `torch.polar` (`__sincosf_stret`). The former
    /// float32 angle `p · θ^(−2j/d)` was 1.2·10⁻⁴ off at the end of the segment axis.
    func testZImageTableIsTheReferences() {
        let tables = RopeTables(axesDims: [32, 48, 48], axesLens: [1536, 512, 512], theta: 256)
        var row = [Float](repeating: 0, count: 128 * 2)
        let cases: [([Int], Int, [UInt32])] = [
            ([1535, 0, 0], 1, [0xbc3990cc, 0xbf7ffbcc]),
            ([1000, 0, 0], 5, [0x3f296ee9, 0x3f3fe83f]),
            ([0, 511, 0], 16 + 2, [0x3dd3709e, 0x3f7ea1cb]),
        ]
        for (ids, pair, want) in cases {
            row.withUnsafeMutableBufferPointer { tables.write(ids: ids, into: $0.baseAddress!) }
            XCTAssertEqual(bits(row[2 * pair], row[2 * pair + 1]), want, "ids \(ids), pair \(pair)")
        }
    }

    /// Qwen-Image-2.1's DiT: float32 frequencies, a float32 product, `torch.polar` — negative positions too.
    func testQwenImage21TableIsTheReferences() {
        var row = [Float](repeating: 0, count: 64 * 2)
        for (position, want) in [(999, [UInt32(0x3f7c146b), 0x3e3284a3]), (-300, [0xbf7f81db, 0x3d7e0445])] {
            row.withUnsafeMutableBufferPointer {
                QwenImage21Rope.write(into: $0.baseAddress!, positions: [[position, 0, 0]], axes: [16, 56, 56], theta: 10000)
            }
            XCTAssertEqual(bits(row[6], row[7]), want, "position \(position)")
        }
    }

    /// Krea 2's DiT: float64 throughout, `.float()` at the end — it already was.
    func testKrea2TableIsTheReferences() {
        var table = [Float](repeating: 0, count: (1 + 64 * 64) * 128)
        table.withUnsafeMutableBufferPointer {
            Krea2Rope.write(into: $0.baseAddress!, textRows: 1, tilesHigh: 64, tilesWide: 64, axes: [32, 48, 48], theta: 1000)
        }
        let token = 1 + 63 * 64 + 63
        XCTAssertEqual(bits(table[token * 128 + 80], table[token * 128 + 81]), [0x3f7c63b8, 0x3e2b5f4b])
    }

    /// `torch.polar` on macOS is `__sincosf_stret`, not `cosf`/`sinf`: at least one angle tells them apart.
    func testPolarIsNotCosfSinf() {
        let apart = (0..<20_000).contains { i in
            let a = Float(i) * 0.0537
            let (c, s) = Ops.polar(a)
            return c != cosf(a) || s != sinf(a)
        }
        XCTAssertTrue(apart)
    }

    /// The text encoders' 1D table is the mRoPE's with `t = h = w = index`, and its frequency is
    /// `1/θ^(2j/d)` (transformers), not `θ^(−2j/d)`.
    func testTextTableIsTheMRoPEOnTheDiagonal() {
        let n = 300, d = 128
        var one = [Float](repeating: 0, count: n * d), three = one
        one.withUnsafeMutableBufferPointer {
            TextEncoder.writeRope(into: $0.baseAddress!, positions: [Array(0..<n)], count: n, theta: 5_000_000, headDim: d, sections: nil)
        }
        three.withUnsafeMutableBufferPointer {
            TextEncoder.writeRope(into: $0.baseAddress!, positions: Array(repeating: Array(0..<n), count: 3), count: n,
                                  theta: 5_000_000, headDim: d, sections: [24, 20, 20])
        }
        XCTAssertEqual(one, three)
        XCTAssertEqual(TextEncoder.inverseFrequencies(theta: 1e6, dim: d)[5], 1 / powf(1e6, Float(10) / Float(d)))
    }
}
