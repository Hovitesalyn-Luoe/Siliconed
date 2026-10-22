import XCTest
@testable import Siliconed

/// **The ANE's programs, written without coremltools**: `coremldata.bin` byte for byte what coremltools 9
/// compiled for the same `matmul(x, w)` (fp16 in and out), at three shapes whose dimensions take
/// different varints; and the cut's rows.
final class NeuralEngineTests: XCTestCase {
    private func bytes(_ hex: String) -> Data {
        var data = Data()
        var index = hex.startIndex
        while index < hex.endIndex {
            let next = hex.index(index, offsetBy: 2)
            data.append(UInt8(hex[index..<next], radix: 16)!)
            index = next
        }
        return data
    }

    private static let header = "f6010000090000000000000000000000000000000000000000000000070000000000000067656e6572696309"
        + "000000000000000000000000000000000000000000000000000000000000003c00000000000000"
    private static let trailer = "f6010000000000000000000000000000"

    func testDescriptionMatchesCoremltools() {
        let cases: [(rows: Int, k: Int, n: Int, payload: String)] = [
            (2048, 2048, 2048, "0a110a01781a0c2a0a0a0480108010109080040a110a01771a0c2a0a0a04801080101090800452110a01791a0c2a0a0a048010801010908004a20600"),
            (1536, 2048, 8192, "0a110a01781a0c2a0a0a04800c8010109080040a110a01771a0c2a0a0a04801080401090800452110a01791a0c2a0a0a04800c804010908004a20600"),
            (3072, 8192, 2048, "0a110a01781a0c2a0a0a0480188040109080040a110a01771a0c2a0a0a04804080101090800452110a01791a0c2a0a0a048018801010908004a20600"),
        ]
        for c in cases {
            XCTAssertEqual(NeuralEngine.description(rows: c.rows, k: c.k, n: c.n),
                           bytes(Self.header + c.payload + Self.trailer), "\(c.rows)×\(c.k)→\(c.n)")
        }
    }

    /// A one-byte dimension shortens the description: its length in the header follows.
    func testDescriptionLengthFollowsThePayload() {
        let data = NeuralEngine.description(rows: 64, k: 2048, n: 2048)
        let length = data[0x4b..<0x53].enumerated().reduce(0) { $0 | Int($1.element) << (8 * $1.offset) }
        XCTAssertEqual(length, data.count - 0x53 - 16)
        XCTAssertEqual(length, 58)
    }

    /// K cut within the ANE's limits: K ≤ 8 192 and a part's weight ≤ 3 · 2²⁴ values.
    func testParts() {
        XCTAssertEqual(NeuralEngine.parts(k: 2048, n: 8192), 1)
        XCTAssertEqual(NeuralEngine.parts(k: 8192, n: 2048), 1)
        XCTAssertEqual(NeuralEngine.parts(k: 4096, n: 12288), 1, "100 MB: within")
        XCTAssertEqual(NeuralEngine.parts(k: 10240, n: 3840), 2)
        XCTAssertEqual(NeuralEngine.parts(k: 12288, n: 4096), 2)
        XCTAssertEqual(NeuralEngine.parts(k: 8192, n: 8192), 2, "134 MB: two parts of 67")
        XCTAssertEqual(NeuralEngine.parts(k: 16384, n: 2048), 2)
        XCTAssertEqual(NeuralEngine.parts(k: 8193, n: 64), 2)
    }

    func testRows() {
        let ane = NeuralEngine(gpuShare: 0.5)
        XCTAssertEqual(ane.rows(4096), 2048)
        XCTAssertEqual(ane.rows(1024), 512)
        XCTAssertEqual(ane.rows(512), 0, "under minimumRows: the GPU alone")
        XCTAssertEqual(NeuralEngine(gpuShare: 0.375).rows(4096), 2560)
        XCTAssertEqual(NeuralEngine(gpuShare: 0.99).rows(4096), 0, "under one unit: the GPU alone")
        XCTAssertEqual(NeuralEngine(gpuShare: 0.01).rows(4096), 0, "the whole product: the GPU keeps a share")
    }
}
