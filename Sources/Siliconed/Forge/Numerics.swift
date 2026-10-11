import Accelerate
import Foundation

/// **The forge's conversions** — everything that computes goes through fp32, where every published
/// format fits exactly; a 16-bit or 8-bit weight that is only moved (transposed, rows permuted) is
/// moved as its published bits, which the fp32 round trip would give back unchanged.
///
/// bf16, fp16 and fp8 E4M3 are all subsets of fp32: widening them is exact, and a
/// transformation (transpose, permute, fold `1 +`, multiply by an fp8's scale) is done on values
/// that are the file's. The way back to bf16 rounds **to nearest even**, like
/// `torch.Tensor.to(torch.bfloat16)` — this is what makes a map forged here identical, byte for
/// byte, to the one Python forged.
package enum Numerics {
    package struct Failure: Error, CustomStringConvertible {
        package let description: String
    }

    /// Widens `count` values of a safetensors dtype to fp32.
    package static func toFloat32(_ source: UnsafeRawPointer, dtype: String, count: Int,
                                    to destination: UnsafeMutablePointer<Float>) throws {
        switch dtype {
        case "F32":
            destination.update(from: source.assumingMemoryBound(to: Float.self), count: count)
        case "BF16":
            Widen.bfloat16ToFloat32(source: source, destination: UnsafeMutableRawPointer(destination), count: count)
        case "F16":
            var src = vImage_Buffer(data: UnsafeMutableRawPointer(mutating: source), height: 1,
                                    width: vImagePixelCount(count), rowBytes: count * 2)
            var dst = vImage_Buffer(data: UnsafeMutableRawPointer(destination), height: 1,
                                    width: vImagePixelCount(count), rowBytes: count * 4)
            vImageConvert_Planar16FtoPlanarF(&src, &dst, 0)
        case "F8_E4M3":
            // E5M2 never gets here: `QuantizedLayouts.recognize` refuses the file before.
            let bytes = source.assumingMemoryBound(to: UInt8.self)
            e4m3.withUnsafeBufferPointer { t in
                for i in 0..<count { destination[i] = t[Int(bytes[i])] }
            }
        case "F64":
            let d = source.assumingMemoryBound(to: Double.self)
            for i in 0..<count { destination[i] = Float(d[i]) }
        default:
            throw Failure(description: "dtype \(dtype): the forge cannot read it here (8-bit weights, GGUF Q8_0 "
                        + "included, are read with their scale, GGUF K-quants by super-block; other 4-bit formats "
                        + "and mxfp8 are refused)")
        }
    }

    /// fp8 E4M3 "fn": bias 7, no infinity, `0x7F`/`0xFF` are NaN.
    static let e4m3: [Float] = (0..<256).map { b in
        let sign: Float = b & 0x80 != 0 ? -1 : 1
        let e = (b >> 3) & 0xF, m = b & 0x7
        if e == 0xF && m == 0x7 { return .nan }
        if e == 0 { return sign * Float(m) / 8 * pow(2, -6) }
        return sign * (1 + Float(m) / 8) * pow(2, Float(e) - 7)
    }

    /// fp32 → bf16, rounded to nearest even (`c10::BFloat16`). Returns the number of values that do
    /// not read back identically — zero when the published file was already bf16.
    @discardableResult
    package static func toBFloat16(_ source: UnsafePointer<Float>, count: Int,
                                     to destination: UnsafeMutablePointer<UInt16>) -> Int {
        let bits = UnsafeRawPointer(source).assumingMemoryBound(to: UInt32.self)
        var inexact = 0
        for i in 0..<count {
            let u = bits[i]
            let h: UInt16
            if (u & 0x7FFF_FFFF) > 0x7F80_0000 {
                h = 0x7FC0                                                // c10's NaN
            } else {
                h = UInt16(truncatingIfNeeded: (u &+ 0x7FFF &+ ((u >> 16) & 1)) >> 16)
            }
            destination[i] = h
            if UInt32(h) << 16 != u { inexact += 1 }
        }
        return inexact
    }

    /// fp32 → fp16, rounded to nearest even. Returns the number of values that do not read back
    /// identically — zero for values that came from an fp16 file.
    @discardableResult
    package static func toFloat16(_ source: UnsafePointer<Float>, count: Int,
                                  to destination: UnsafeMutablePointer<UInt16>) -> Int {
        var inexact = 0
        for i in 0..<count {
            let h = Float16(source[i])
            destination[i] = h.bitPattern
            if Float(h).bitPattern != source[i].bitPattern && !(h.isNaN && source[i].isNaN) { inexact += 1 }
        }
        return inexact
    }

    /// Do these fp32 values all read back identically from fp16?
    package static func exactInFloat16(_ source: UnsafePointer<Float>, count: Int) -> Bool {
        for i in 0..<count {
            let v = source[i]
            if Float(Float16(v)).bitPattern != v.bitPattern && !v.isNaN { return false }
        }
        return true
    }

    /// Do these fp32 values all read back identically from bf16? (No rounding — the forge's rule.)
    package static func exactInBFloat16(_ source: UnsafePointer<Float>, count: Int) -> Bool {
        let bits = UnsafeRawPointer(source).assumingMemoryBound(to: UInt32.self)
        var low: UInt32 = 0
        for i in 0..<count { low |= bits[i] & 0xFFFF }
        return low == 0
    }

    /// `[rows, columns]` → `[columns, rows]`, in fp32 (`vDSP_mtrans`).
    package static func transpose(_ source: UnsafePointer<Float>, rows: Int, columns: Int,
                                   to destination: UnsafeMutablePointer<Float>) {
        vDSP_mtrans(source, 1, destination, 1, vDSP_Length(columns), vDSP_Length(rows))
    }

    /// `[rows, columns]` → `[columns, rows]` for any element — the 8- and 16-bit bytes the forge
    /// moves without converting them. By 32 × 32 tiles, over the cores: a job writes whole rows of
    /// the result, so no two jobs share an output line but at their two ends.
    package static func transpose<T: BitwiseCopyable>(_ source: UnsafePointer<T>, rows: Int, columns: Int,
                                                       to destination: UnsafeMutablePointer<T>) {
        let tile = 32
        nonisolated(unsafe) let (s, d) = (source, destination)
        let bands = (columns + tile - 1) / tile
        let band: @Sendable (Int) -> Void = { b in
            let c0 = b * tile, c1 = min(c0 + tile, columns)
            for r0 in stride(from: 0, to: rows, by: tile) {
                let r1 = min(r0 + tile, rows)
                for c in c0..<c1 {
                    let out = d + c * rows
                    for r in r0..<r1 { out[r] = s[r * columns + c] }
                }
            }
        }
        if bands * rows < 1 << 16 { for b in 0..<bands { band(b) } } else { DispatchQueue.concurrentPerform(iterations: bands, execute: band) }
    }

    /// The largest |x| of bf16 (`half` false) or fp16 bits, read on the bits: magnitude orders as
    /// `bits & 0x7FFF` does, so it is `absMax` of the widened values exactly. `nil`: a NaN.
    package static func absMax16(_ bits: UnsafePointer<UInt16>, count: Int, half: Bool) -> Float? {
        var m: UInt16 = 0
        for i in 0..<count { m = max(m, bits[i] & 0x7FFF) }
        if m > (half ? 0x7C00 : 0x7F80) { return nil }
        return half ? Float(Float16(bitPattern: m)) : Float(bitPattern: UInt32(m) << 16)
    }

    /// The largest |x| — `float(t.abs().max())`.
    package static func absMax(_ source: UnsafePointer<Float>, count: Int) -> Float {
        guard count > 0 else { return 0 }
        var m: Float = 0
        vDSP_maxmgv(source, 1, &m, vDSP_Length(count))
        return m
    }
}
