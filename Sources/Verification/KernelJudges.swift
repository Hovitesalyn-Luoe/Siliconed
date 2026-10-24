import Foundation
import Metal
import Siliconed

// The kernel stages whose judgment is long enough to stand apart from the registry's entries
// (`KernelStages.swift`): the GELU over a sweep of bit patterns, the RoPE tables and the resampling
// operators against torch's.

// MARK: - GELU

enum GELUJudge {
    /// The GPU's GELU against double precision, and against the CPU path it replaced, on the same sweep:
    /// one bit pattern in 64 over [−12, 12] (≈ 33 M values, every scale), plus ±0. Judged in ulp of the
    /// **operand** — the scale of what the next GEMM multiplies; an output that cancels has an ulp that
    /// says nothing of the arithmetic.
    static func run(into report: Report) throws {
        var values: [Float] = [0, -0]
        let top = Float(12).bitPattern
        var bits: UInt32 = 0
        while bits <= top {
            let v = Float(bitPattern: bits)
            values.append(v); values.append(-v)
            bits &+= 64
        }
        let n = values.count
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue(),
              let buffer = device.makeBuffer(bytes: values, length: n * 4, options: .storageModeShared) else {
            throw GEMM.Failure.noDevice
        }
        _ = try AnimaGELU(device: device, queue: queue).run(buffer, count: n)
        let gpu = Array(UnsafeBufferPointer(start: buffer.contents().assumingMemoryBound(to: Float.self), count: n))
        var cpu = values
        cpu.withUnsafeMutableBufferPointer { AnimaOps.gelu($0.baseAddress!, count: n) }

        func worst(_ y: [Float]) -> (operand: Double, notFinite: Int) {
            var operand = 0.0, notFinite = 0
            for i in 0..<n {
                let x = Double(values[i])
                let exact = 0.5 * x * erfc(-x / 2.0.squareRoot())
                guard y[i].isFinite else { notFinite += 1; continue }
                // A subnormal output: the GPU flushes it to zero, as MPS's GEMMs do; outside the verdict.
                if abs(exact) < Double(Float.leastNormalMagnitude) { continue }
                let ulpX = Double(max(abs(values[i]), .leastNormalMagnitude).ulp)
                operand = max(operand, abs(Double(y[i]) - exact) / ulpX)
            }
            return (operand, notFinite)
        }
        let g = worst(gpu), c = worst(cpu)
        report.add(Judge.bound("GELU, \(n) values", g.notFinite == 0 ? g.operand : .infinity, c.operand,
                               unit: "ulp of |x|", note: String(format: "CPU erff %.3f", c.operand)))
    }
}

// MARK: - RoPE

enum RopeJudge {
    /// The residue each table is allowed, in units of 2⁻²⁴ (one bit of a value near 1). torch's float32
    /// `pow`, `cos` and `sin` are SLEEF's vectorized ones, within one ulp but not correctly rounded: a
    /// table built without them (`torch.polar`, or float64 throughout) is bit for bit, the others carry
    /// that residue — `pow` differs from libm's on one frequency of 64 at θ = 10⁶, hence the 64.
    static let bound: [String: Int] = [
        "zimage-dit.0": 0, "zimage-dit.1": 0, "zimage-dit.2": 0, "qwen21-dit.0": 0, "krea2-dit": 0,
        "qwen3-text-4b": 1, "qwen3-text-0.6b": 1, "qwen3vl-vision": 1, "anima-dit": 1, "anima-adapter": 1,  // SLEEF's cos/sin
        "qwen3vl-text": 8,                                  // + SLEEF's pow on one frequency, positions up to 342
        "qwen21-dit.1": 64, "qwen21-dit.2": 64,             // SLEEF's pow on one frequency, positions up to 1 200
    ]

    static func run(goldens: Safetensors, into report: Report) throws {
        func floats(_ name: String) throws -> UnsafeBufferPointer<Float> {
            guard let v = goldens.float32(name) else { throw Skip("\(goldens.path): no \(name)") }
            return v
        }
        func positions(_ name: String) throws -> (rows: [[Int]], axes: Int) {
            guard let v = goldens.int32(name + ".positions"), let shape = goldens.entries[name + ".positions"]?.shape else {
                throw Skip("\(goldens.path): no \(name).positions")
            }
            let axes = shape.count > 1 ? shape[1] : 1
            return ((0..<shape[0]).map { r in (0..<axes).map { Int(v[r * axes + $0]) } }, axes)
        }
        func meta(_ name: String) -> [String: Any] {
            (goldens.metadata[name].flatMap { try? JSONSerialization.jsonObject(with: Data($0.utf8)) } as? [String: Any]) ?? [:]
        }
        /// `table` is `[rows, pairs, (cos, sin)]`.
        func judge(_ name: String, _ table: [Float]) throws {
            let cos = try floats(name + ".cos"), sin = try floats(name + ".sin")
            guard table.count == 2 * cos.count else {
                report.add(Judge.identity(name, differing: -1, of: table.count / 2, against: cos.count)); return
            }
            var apart = 0, worst: Float = 0
            for i in 0..<cos.count {
                for (got, want) in [(table[2 * i], cos[i]), (table[2 * i + 1], sin[i])] where got != want {
                    apart += 1
                    worst = max(worst, abs(got - want))
                }
            }
            let units = Double((Double(worst) * 16_777_216).rounded(.up))
            report.add(Judge.bound(name, units, Double(bound[name] ?? 0), unit: "· 2⁻²⁴",
                                   note: "\(apart) of \(2 * cos.count) values apart"))
        }

        // Z-Image's DiT: each axis its own table (`RopeTables` writes the axes side by side).
        for axis in 0..<3 {
            let name = "zimage-dit.\(axis)", m = meta(name)
            let dims = m["dims"] as! Int, theta = Float(m["theta"] as! Double)
            let (rows, _) = try positions(name)
            let tables = RopeTables(axesDims: [dims], axesLens: [rows.count], theta: theta)
            var table = [Float](repeating: 0, count: rows.count * dims)
            for (r, p) in rows.enumerated() { table.withUnsafeMutableBufferPointer { tables.write(ids: [p[0]], into: $0.baseAddress! + r * dims) } }
            try judge(name, table)
        }
        // Qwen3's text encoders (one axis) and Qwen3-VL's (three, interleaved sections).
        for name in ["qwen3-text-4b", "qwen3-text-0.6b", "qwen3vl-text"] {
            let m = meta(name)
            let dims = m["dims"] as! Int, theta = Float(m["theta"] as! Double), sections = m["sections"] as? [Int]
            let (rows, axes) = try positions(name)
            var table = [Float](repeating: 0, count: rows.count * dims)
            let byAxis = (0..<axes).map { a in rows.map { $0[a] } }
            table.withUnsafeMutableBufferPointer {
                TextEncoder.writeRope(into: $0.baseAddress!, positions: byAxis, count: rows.count, theta: theta,
                                      headDim: dims, sections: sections)
            }
            try judge(name, table)
        }
        do {
            let name = "qwen3vl-vision", m = meta(name)
            let dims = m["dims"] as! Int, theta = Float(m["theta"] as! Double)
            let (rows, _) = try positions(name)
            var table = [Float](repeating: 0, count: rows.count * dims)
            table.withUnsafeMutableBufferPointer {
                Qwen3VLVision.writeRope(into: $0.baseAddress!, positions: rows.map { ($0[0], $0[1]) }, headDim: dims, theta: theta)
            }
            try judge(name, table)
        }
        for axis in 0..<3 {
            let name = "qwen21-dit.\(axis)", m = meta(name)
            let dims = m["dims"] as! Int, theta = Double(m["theta"] as! Int)
            let (rows, _) = try positions(name)
            var table = [Float](repeating: 0, count: rows.count * dims)
            table.withUnsafeMutableBufferPointer {
                QwenImage21Rope.write(into: $0.baseAddress!, positions: rows, axes: [dims], theta: theta)
            }
            try judge(name, table)
        }
        do {
            let name = "anima-dit", m = meta(name)
            let dims = m["dims"] as! Int, scale = m["scale"] as! [Double]
            let (rows, _) = try positions(name)
            let high = rows.map { $0[1] }.max()! + 1, wide = rows.map { $0[2] }.max()! + 1
            var table = [Float](repeating: 0, count: rows.count * dims)
            table.withUnsafeMutableBufferPointer {
                AnimaRope.write(into: $0.baseAddress!, tilesHigh: high, tilesWide: wide, headDim: dims, scale: scale)
            }
            try judge(name, table)
        }
        do {
            let name = "anima-adapter", m = meta(name)
            let (rows, _) = try positions(name)
            try judge(name, AnimaConditioner.ropeTable(rows: rows.count, headDim: m["dims"] as! Int,
                                                       theta: Float(m["theta"] as! Double)))
        }
        do {
            let name = "krea2-dit", m = meta(name)
            let axes = m["axes"] as! [Int], theta = m["theta"] as! Double
            let (rows, _) = try positions(name)
            let text = m["text"] as! Int
            let image = rows.dropFirst(text)
            let high = image.map { $0[1] }.max()! + 1, wide = image.map { $0[2] }.max()! + 1
            var table = [Float](repeating: 0, count: rows.count * axes.reduce(0, +))
            table.withUnsafeMutableBufferPointer {
                Krea2Rope.write(into: $0.baseAddress!, textRows: text, tilesHigh: high, tilesWide: wide, axes: axes, theta: theta)
            }
            try judge(name, table)
        }
    }
}

// MARK: - Resampling

enum ResampleJudge {
    /// The two operators of the spectral schedule against torch's. `nearest-exact` and `nearest` differ
    /// by half a pixel — odd indices or even ones at factor two — and both give a latent of the same
    /// distribution and a plausible image: no norm tells the wrong one. Hence a pseudo-random test set
    /// and the counter-test, the even decimation, which must differ.
    ///
    /// Tolerance in ulp of the **input's** scale, not of each value: a weighted mean of values of order 4
    /// that nearly cancel is of order 10⁻⁵, and its own ulp measures the cancellation, not the operator.
    /// Zero for the decimation, which copies samples; one for the bilinear, whose accumulation order in
    /// torch was not found and whose last bit decides nothing.
    static func run(goldens: Safetensors, into report: Report) throws {
        guard let stateEntry = goldens.entries["state"], let state = goldens.float32("state"),
              let down = goldens.float32("down"), let velocity = goldens.float32("velocity"),
              let up = goldens.float32("up") else {
            throw Skip("\(goldens.path): state / down / velocity / up missing")
        }
        // `[1, C, H, W]`: the rectangular set catches an axis inversion the square hides.
        let channels = stateEntry.shape[1]
        let (fullH, fullW) = (stateEntry.shape[2], stateEntry.shape[3]), (lowH, lowW) = (fullH / 2, fullW / 2)
        let shape = "\(fullW)×\(fullH)"

        func beyond(_ got: [Float], _ expected: UnsafeBufferPointer<Float>, scale: UnsafeBufferPointer<Float>) -> Double {
            var magnitude: Float = 0
            for value in scale { magnitude = max(magnitude, abs(value)) }
            let unit = Double(magnitude.ulp)
            var worst = 0.0
            for i in 0..<got.count where got[i].bitPattern != expected[i].bitPattern {
                worst = max(worst, abs(Double(got[i]) - Double(expected[i])) / unit)
            }
            return worst
        }
        var decimated = [Float](repeating: .nan, count: channels * lowH * lowW)
        decimated.withUnsafeMutableBufferPointer {
            Resample.decimate(state.baseAddress!, into: $0.baseAddress!, channels: channels, height: lowH, width: lowW)
        }
        report.add(Judge.bound("\(shape) nearest-exact", beyond(decimated, down, scale: state), 0,
                               unit: "ulp of the scale"))
        var even = [Float](repeating: .nan, count: channels * lowH * lowW)
        for c in 0..<channels {
            for y in 0..<lowH {
                for x in 0..<lowW { even[(c * lowH + y) * lowW + x] = state[(c * fullH + 2 * y) * fullW + 2 * x] }
            }
        }
        let evenDiffering = zip(even, down).filter { $0.bitPattern != $1.bitPattern }.count
        report.add(Judge.bound("\(shape) counter-test: even", Double(evenDiffering), 0, unit: "values differ", above: true,
                               note: "must differ, or the check is blind to the parity"))
        var raised = [Float](repeating: .nan, count: channels * fullH * fullW)
        raised.withUnsafeMutableBufferPointer {
            Resample.bilinearDouble(velocity.baseAddress!, into: $0.baseAddress!, channels: channels, height: lowH, width: lowW)
        }
        report.add(Judge.bound("\(shape) bilinear", beyond(raised, up, scale: velocity), 1, unit: "ulp of the scale"))
    }
}
