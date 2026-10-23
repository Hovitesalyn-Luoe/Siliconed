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

    /// K cut within the ANE's limits: K ≤ 4 096 and a part's weight ≤ 1.5 · 2²⁴ values, in equal parts of a
    /// multiple of 64 when a few more give them.
    func testParts() {
        XCTAssertEqual(NeuralEngine.parts(k: 2048, n: 8192), 1)
        XCTAssertEqual(NeuralEngine.parts(k: 3840, n: 3840), 1)
        XCTAssertEqual(NeuralEngine.parts(k: 3840, n: 10240), 2, "39 M values: two of 1 920")
        XCTAssertEqual(NeuralEngine.parts(k: 6144, n: 6144), 2, "two parts of 3 072")
        XCTAssertEqual(NeuralEngine.parts(k: 8192, n: 2048), 2, "K ≤ 4 096")
        XCTAssertEqual(NeuralEngine.parts(k: 10240, n: 3840), 4, "three would be uneven: four of 2 560")
        XCTAssertEqual(NeuralEngine.parts(k: 12288, n: 4096), 3)
        XCTAssertEqual(NeuralEngine.parts(k: 16384, n: 6144), 4)
        XCTAssertEqual(NeuralEngine.parts(k: 4097, n: 64), 2, "no even split within reach: the fewest parts")
    }

    func testRows() {
        let ane = NeuralEngine(gpuShare: 0.5)
        XCTAssertEqual(ane.rows(4096, k: 2048, n: 2048), 2048)
        XCTAssertEqual(ane.rows(1024, k: 2048, n: 2048), 512)
        XCTAssertEqual(ane.rows(512, k: 2048, n: 2048), 0, "under minimumRows: the GPU alone")
        XCTAssertEqual(NeuralEngine(gpuShare: 0.375).rows(4096, k: 2048, n: 2048), 2560)
        XCTAssertEqual(NeuralEngine(gpuShare: 0.99).rows(4096, k: 2048, n: 2048), 0, "under one unit: the GPU alone")
        XCTAssertEqual(NeuralEngine(gpuShare: 0.01).rows(4096, k: 2048, n: 2048), 0, "the whole product: the GPU keeps a share")
        // A grid replaces the scalar share: T = 1 / (1 + ρ), here ρ = 1 at 4096 rows (½), 1/3 at 1024 (¾).
        let grid = NeuralEngine(gpuShare: 0.9, shares: Conductor.Shares([.init(k: 2048, n: 2048, m: 4096, ratio: 1),
                                                                          .init(k: 2048, n: 2048, m: 1024, ratio: 1.0 / 3)]))
        XCTAssertEqual(grid.rows(4096, k: 2048, n: 2048), 2048)
        XCTAssertEqual(grid.rows(1024, k: 2048, n: 2048), 256)
    }

    /// The Standard renders in fp32 unless asked, the 8- and 4-bit versions take the Neural Engine; the PNG
    /// says it only when the denoiser could take it, and a choice the map left without effect keys nothing.
    func testTheStandardStaysFp32AndThePNGSaysWhenTheEngineWasTaken() throws {
        XCTAssertFalse(Variant.standard.neuralEngineByDefault)
        XCTAssertTrue(Variant.compact.neuralEngineByDefault)
        XCTAssertTrue(Variant.light.neuralEngineByDefault)
        var render = Engine.Render(model: "krea2", prompt: "woman posing in a library", seed: 42, steps: 8, loras: [],
                                   loraSummary: nil, image: ImageRGB(pixels: [Float](repeating: 0, count: 3 * 4), height: 2, width: 2),
                                   reproducible: true, evaluations: 8, sketch: nil, spectral: 0, strength: nil,
                                   startStep: 0, startSigma: 1, timings: .init(), footprints: .init(), tokens: 1)
        XCTAssertNil(render.metadata["neural-engine"])
        XCTAssertEqual(Engine.Render.Recipe(metadata: render.metadata)?.neuralEngine, false)
        render.neuralEngine = true
        let recipe = try XCTUnwrap(Engine.Render.Recipe(metadata: PNG.text(try render.png())))
        XCTAssertTrue(recipe.neuralEngine)
        XCTAssertEqual(recipe.unreadable, [])
        XCTAssertFalse(EngineSettings.effective.gemmANE)
        XCTAssertEqual(RenderCache.numerics(reproducible: true, neuralEngine: true), RenderCache.numerics(reproducible: true))
    }
}
