import Accelerate
import CryptoKit
import XCTest
@testable import Siliconed

/// **The fast forge writes the slow forge's bytes**: 16-bit weights moved without going
/// through fp32, an 8-bit weight's largest |w| read without dequantizing it, tensors produced
/// several at a time and hashed in order, a download's sha256 computed behind it — each against
/// the plain computation it replaces, on synthetic data.
final class ForgePipelineTests: XCTestCase {
    private var root: URL!
    private var generator = SystemRandomNumberGenerator()

    override func setUpWithError() throws {
        root = FileManager.default.temporaryDirectory.appendingPathComponent("siliconed-pipeline-\(UUID())")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws { try? FileManager.default.removeItem(at: root) }

    private func safetensors(_ pieces: [(name: String, dtype: String, shape: [Int], bytes: [UInt8])]) throws -> String {
        let path = root.appendingPathComponent("model.safetensors").path
        try ForgeDiT.writeSafetensors(pieces.map { ($0.name, $0.dtype, $0.shape, Data($0.bytes)) }, to: path)
        return path
    }

    /// Random 16-bit patterns, finite (no NaN) unless asked: subnormals, zeros of both signs and
    /// infinities included.
    private func bits16(_ n: Int, half: Bool, nan: Bool = false) -> [UInt16] {
        var b = (0..<n).map { _ -> UInt16 in
            var x = UInt16.random(in: 0...UInt16.max, using: &generator)
            let infinity: UInt16 = half ? 0x7C00 : 0x7F80
            if x & 0x7FFF > infinity { x &= half ? 0xBBFF : 0xBF7F }        // a NaN made finite
            return x
        }
        if n > 4 { b[1] = 0x8000; b[2] = half ? 0xFC00 : 0xFF80; b[3] = 1 }
        if nan { b[n / 2] = half ? 0x7E01 : 0x7FA1 }
        return b
    }

    private func bytes(_ v: [UInt16]) -> [UInt8] { v.withUnsafeBytes { Array($0) } }

    private func random(_ n: Int) -> Data {
        var d = Data(count: n)
        d.withUnsafeMutableBytes { if n > 0 { arc4random_buf($0.baseAddress!, n) } }
        return d
    }

    func testTransposeMovesEveryElement() {
        for (rows, columns) in [(1, 1), (3, 5), (33, 70), (64, 129), (300, 257)] {
            let a8 = (0..<(rows * columns)).map { UInt8(truncatingIfNeeded: $0 &* 131) }
            let a16 = (0..<(rows * columns)).map { UInt16(truncatingIfNeeded: $0 &* 40_503) }
            var t8 = [UInt8](repeating: 0, count: a8.count), t16 = [UInt16](repeating: 0, count: a16.count)
            Numerics.transpose(a8, rows: rows, columns: columns, to: &t8)
            Numerics.transpose(a16, rows: rows, columns: columns, to: &t16)
            for r in 0..<rows {
                for c in 0..<columns {
                    XCTAssertEqual(t8[c * rows + r], a8[r * columns + c])
                    XCTAssertEqual(t16[c * rows + r], a16[r * columns + c])
                }
            }
        }
    }

    func testTheLargest16BitValueIsReadOnTheBits() {
        for half in [false, true] {
            let b = bits16(10_000, half: half)
            let widened = b.map { half ? Float(Float16(bitPattern: $0)) : Float(bitPattern: UInt32($0) << 16) }
            XCTAssertEqual(Numerics.absMax16(b, count: b.count, half: half)?.bitPattern,
                           Numerics.absMax(widened, count: widened.count).bitPattern)
            let finite = b.map { $0 & 0x7FFF >= (half ? 0x7C00 : 0x7F80) ? 0x3C00 : $0 }
            let w = finite.map { half ? Float(Float16(bitPattern: $0)) : Float(bitPattern: UInt32($0) << 16) }
            XCTAssertEqual(Numerics.absMax16(finite, count: finite.count, half: half)?.bitPattern,
                           Numerics.absMax(w, count: w.count).bitPattern)
            XCTAssertNil(Numerics.absMax16(bits16(100, half: half, nan: true), count: 100, half: half))
            XCTAssertEqual(Numerics.absMax16([0x8000, 0], count: 2, half: half)?.bitPattern, 0)
        }
    }

    /// `absMax()` against the dequantized tensor's maximum, bit for bit, on every layout it
    /// shortcuts — int8 −128 included, scales of the three types, per row, per tensor, by block,
    /// fp8 with and without a scale — and on the ones it does not (a NaN, an infinite scale).
    func testTheLargestEightBitWeightWithoutDequantizing() {
        func scaleBytes(_ n: Int, _ type: Artifact.DType) -> [UInt8] {
            (0..<n).flatMap { _ -> [UInt8] in
                let s = Float.random(in: -3e-2...3e-2, using: &generator)
                switch type {
                case .bfloat16: return withUnsafeBytes(of: UInt16(s.bitPattern >> 16)) { Array($0) }
                case .float16: return withUnsafeBytes(of: Float16(s).bitPattern) { Array($0) }
                default: return withUnsafeBytes(of: s.bitPattern) { Array($0) }
                }
            }
        }
        func check(_ t: QuantizedTensor, _ label: String) {
            let whole = t.dequantized().withUnsafeBufferPointer { Numerics.absMax($0.baseAddress!, count: $0.count) }
            XCTAssertEqual(t.absMax().bitPattern, whole.bitPattern, label)
        }
        let (rows, columns) = (24, 96)
        for kind in [Artifact.DType.int8, .float8_e4m3] {
            for type in [Artifact.DType.bfloat16, .float16, .float32] {
                var values = (0..<(rows * columns)).map { _ -> UInt8 in
                    var b = UInt8.random(in: 0...255, using: &generator)
                    if kind == .float8_e4m3 && b & 0x7F == 0x7F { b ^= 1 }
                    return b
                }
                values[5] = 0x80                                              // int8 −128
                for perRow in [true, false] {
                    let t = QuantizedTensor(kind: kind, rows: rows, columns: columns, values: values,
                                            scale: scaleBytes(perRow ? rows : 1, type), scaleType: type, perRow: perRow)
                    check(t, "\(kind) \(type) perRow \(perRow)")
                }
            }
        }
        let fp8 = (0..<(rows * columns)).map { _ in UInt8.random(in: 0...0x7E, using: &generator) | (Bool.random() ? 0x80 : 0) }
        check(QuantizedTensor(kind: .float8_e4m3, rows: rows, columns: columns, values: fp8, scale: [], scaleType: .float32,
                              perRow: false), "fp8 cast")
        var nan = fp8; nan[7] = 0x7F
        check(QuantizedTensor(kind: .float8_e4m3, rows: rows, columns: columns, values: nan, scale: scaleBytes(rows, .float32),
                              scaleType: .float32, perRow: true), "fp8 NaN")
        let int8 = (0..<(rows * columns)).map { _ in UInt8.random(in: 0...255, using: &generator) }
        check(QuantizedTensor(kind: .int8, rows: rows, columns: columns, values: int8, scale: scaleBytes(rows * columns / 32, .float16),
                              scaleType: .float16, perRow: false, block: 32), "Q8_0 blocks")
        var infinite = scaleBytes(rows, .float32)
        withUnsafeBytes(of: Float.infinity.bitPattern) { for (i, b) in $0.enumerated() { infinite[4 + i] = b } }
        check(QuantizedTensor(kind: .int8, rows: rows, columns: columns, values: int8, scale: infinite, scaleType: .float32,
                              perRow: true), "an infinite scale")
    }

    /// The 16-bit path's bits are the fp32 path's — widened, permuted, transposed, rounded back —
    /// for bf16 and fp16, transposed or not, rows permuted or not; a NaN falls back to the fp32 path.
    func testSixteenBitPathGivesTheFp32PathsBytes() throws {
        let path = try safetensors([
            ("a", "BF16", [48, 40], bytes(bits16(48 * 40, half: false))),
            ("b", "F16", [40, 24], bytes(bits16(40 * 24, half: true))),
            ("c", "BF16", [96], bytes(bits16(96, half: false))),
            ("n", "BF16", [8, 16], bytes(bits16(8 * 16, half: false, nan: true))),
        ])
        let source = try TensorSource(paths: [path])
        let order48 = ForgeDiT.interleaveOrder(rows: 48, head: 8)
        for (name, transposed, order) in [("a", true, nil), ("a", true, order48), ("a", false, order48), ("b", true, nil),
                                          ("b", false, nil), ("c", false, nil), ("c", false, ForgeDiT.interleaveOrder(rows: 96, head: 16))] {
            let p = Recipes.asPublished(source, name)
            let (t, outcome) = try ForgeDiT.tensor(name, p, .init(transposed: transposed, named: .bfloat16, rowOrder: order))
            XCTAssertEqual(outcome, name == "b" ? .kept16bit : .asNamed)
            let bits = try XCTUnwrap(try t.produceBits?() ?? nil)
            let v = try t.produce()
            var expected = [UInt16](repeating: 0, count: v.count)
            if t.dtype == .float16 { Numerics.toFloat16(v, count: v.count, to: &expected) }
            else { Numerics.toBFloat16(v, count: v.count, to: &expected) }
            XCTAssertEqual(bits.values, expected, "\(name) transposed \(transposed) permuted \(order != nil)")
            XCTAssertEqual(bits.absMax.bitPattern, Numerics.absMax(v, count: v.count).bitPattern)
        }
        let (t, _) = try ForgeDiT.tensor("n", Recipes.asPublished(source, "n"), .init(transposed: true, named: .bfloat16))
        XCTAssertNil(try t.produceBits?() ?? nil, "a NaN takes the fp32 path")
    }

    /// Tensors of every path, produced over the cores: the file holds each one's bytes at its
    /// offset, and the fingerprint is the one SHA-256 over names and bytes in file order.
    func testTheMapIsTheSameWhateverItsProducers() throws {
        var pieces: [(name: String, dtype: String, shape: [Int], bytes: [UInt8])] = []
        for k in 0..<40 {
            let rows = 8 + k % 5 * 8
            pieces.append(("w\(k)", k % 3 == 0 ? "F16" : "BF16", [rows, 24], bytes(bits16(rows * 24, half: k % 3 == 0, nan: k == 7))))
        }
        pieces.append(("f", "F32", [16], (0..<16).flatMap { i in withUnsafeBytes(of: Float(i) * 0.1) { Array($0) } }))
        let source = try TensorSource(paths: [try safetensors(pieces)])
        var tensors = try pieces.map { piece in
            try ForgeDiT.tensor(piece.name, Recipes.asPublished(source, piece.name),
                                .init(transposed: piece.shape.count == 2, named: piece.name == "f" ? .float32 : .bfloat16)).0
        }
        tensors.append(MapWriter.Tensor(name: "q", shape: [32, 8], dtype: .int8, scale: .init(dtype: .float32, shape: [8]),
                                        transposed: true) {
            .init(values: (0..<256).map { (i: Int) -> UInt8 in UInt8(i) }, scale: [UInt8](repeating: 0x3F, count: 32), absMax: 2)
        })
        let path = root.appendingPathComponent("map.silicon").path
        let tally = try MapWriter.write(to: path, tensors: tensors, header: { _ in [.init("format", 1), .init("page", .integer(MapWriter.page))] })
        var sha = SHA256()
        let map = try Artifact(path: path)
        for t in tensors {
            let made = try MapWriter.produce(t)
            sha.update(data: Data(t.name.utf8))
            for (part, at) in made.parts {
                part.withBytes { b in
                    sha.update(bufferPointer: b)
                    XCTAssertEqual(memcmp(map.pointer(t.name)! + at, b.baseAddress!, b.count), 0, t.name)
                }
            }
        }
        XCTAssertEqual(tally.sha256, sha.finalize().map { String(format: "%02x", $0) }.joined())
        XCTAssertEqual(tally.tensors, tensors.count)
    }

    func testAFilesSha256AcrossItsBlocks() throws {
        for size in [0, 1, 8 << 20, (8 << 20) + 1, (17 << 20) + 12_345] {
            let data = random(size)
            let path = root.appendingPathComponent("f\(size)").path
            try data.write(to: URL(fileURLWithPath: path))
            XCTAssertEqual(try Downloader.sha256(path), SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined())
        }
    }

    /// The running hash follows the chunks in file order whatever order they arrive in — resumed
    /// ones whole from the start — and gives the whole file's sha256.
    func testTheRunningHashWaitsForEachChunkInOrder() throws {
        let size = (3 << 20) + 777
        let data = random(size)
        let path = root.appendingPathComponent("download.partiel").path, ledgerPath = path + ".morceaux"
        try data.write(to: URL(fileURLWithPath: path))
        FileManager.default.createFile(atPath: ledgerPath, contents: nil)
        let ledger = try XCTUnwrap(FileHandle(forWritingAtPath: ledgerPath))
        defer { try? ledger.close() }
        let chunks = Downloader.chunks(size: size, chunk: 256 << 10)
        let run = Downloader.ChunkRun(fd: -1, ledger: ledger, size: size, label: "test", whole: [0, 3], already: 0,
                                      cancellation: nil, journal: { _ in })
        let hash = Downloader.RunningHash(path: path, chunks: chunks, run: run)
        for k in chunks.indices.reversed() where k != 0 && k != 3 { run.written(k) }
        XCTAssertEqual(try hash.result(), SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined())
        // Stopped before the end: no digest.
        let stopping = Cancellation()
        let cut = Downloader.ChunkRun(fd: -1, ledger: ledger, size: size, label: "test", whole: [0], already: 0,
                                      cancellation: stopping, journal: { _ in })
        let partial = Downloader.RunningHash(path: path, chunks: chunks, run: cut)
        stopping.cancel()
        XCTAssertNil(try partial.result())
    }

    /// The DiT's download beside the encoder's forge has a token of its own: raised by a Stop (the
    /// installation's token) or by the forge failing (its own), never the other way.
    func testALinkedTokenObeysItsParent() {
        let parent = Cancellation(), child = Cancellation(following: parent)
        XCTAssertFalse(child.isCancelled)
        child.cancel()
        XCTAssertTrue(child.isCancelled)
        XCTAssertFalse(parent.isCancelled)
        let other = Cancellation(following: parent)
        parent.cancel()
        XCTAssertTrue(other.isCancelled)
        XCTAssertThrowsError(try other.check())
    }
}
