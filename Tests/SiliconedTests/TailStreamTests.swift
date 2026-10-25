import XCTest
@testable import Siliconed

/// The streamed tail by units: the engine's tensors, in its order, arrive in a few buffers of the
/// largest tensor's size, whatever the file's order, and the stream reads on by itself.
final class TailStreamTests: XCTestCase {
    private var root: URL!
    override func setUpWithError() throws {
        root = FileManager.default.temporaryDirectory.appendingPathComponent("tail-\(UUID())")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    }
    override func tearDownWithError() throws { try? FileManager.default.removeItem(at: root) }

    private static func byte(_ i: Int) -> UInt8 { UInt8(truncatingIfNeeded: i &* 7 &+ 3) }

    /// A map of `layers` layers whose tensors (names `L.<l>.<name>`, file order = alphabetical) have `sizes`
    /// bytes, patterned by their file offset; page 16.
    private func map(layers: Int, sizes: [String: Int]) throws -> String {
        let page = 16, dataStart = 1 << 16
        var table: [String: Any] = [:], order: [String] = [], offset = dataStart
        for l in 0..<layers {
            for name in sizes.keys.sorted() {
                let key = "L.\(l).\(name)"
                table[key] = ["offset": offset, "bytes": sizes[name]!, "shape": [sizes[name]!], "dtype": "int8"]
                order.append(key)
                offset += (sizes[name]! + page - 1) / page * page
            }
        }
        let json = try JSONSerialization.data(withJSONObject: ["page": page, "tensors": table, "order": order] as [String: Any])
        var bytes = Array("SILICON".utf8) + [3]
        for i in 0..<8 { bytes.append(UInt8((UInt64(json.count) >> (8 * UInt64(i))) & 0xff)) }
        bytes += json
        bytes += [UInt8](repeating: 0, count: dataStart - bytes.count)
        bytes += (dataStart..<offset).map(Self.byte)
        let path = root.appendingPathComponent("m.silicon").path
        try Data(bytes).write(to: URL(fileURLWithPath: path))
        return path
    }

    func testVisitsAreCutAtTheLargestTensorAndKeepRepeats() throws {
        let artifact = try Artifact(path: try map(layers: 1, sizes: ["a": 100, "b": 2000, "c": 100, "d": 2000]))
        let (a, b, c, d) = ("L.0.a", "L.0.b", "L.0.c", "L.0.d")
        // Lengths are page-rounded (16): 112, 2000, 112, 2000.
        XCTAssertEqual(artifact.stagingUnits(layers: [[a, b, c, d]], cap: 2112), [[a, b], [c, d]])
        XCTAssertEqual(artifact.stagingUnits(layers: [[a, b, c, d]], cap: 2000), [[a], [b], [c], [d]])
        // A repeat inside a visit costs nothing and stays in the list; across visits it is read again.
        XCTAssertEqual(artifact.stagingUnits(layers: [[c, c, d, c, d]], cap: 2112), [[c, c, d, c, d]])
        XCTAssertEqual(artifact.stagingUnits(layers: [[c, c, d, c, d]], cap: 2000), [[c, c], [d], [c], [d]])
        XCTAssertEqual(artifact.stagingUnits(layers: [[a, "missing"], []], cap: 2000), [[a]])
    }

    /// The MLP's weights used for each of two chunks, after the attention's: the stream reads them twice, in
    /// the engine's order, and every use gets its own bytes.
    func testATensorUsedTwiceInARowIsReadTwice() throws {
        let artifact = try Artifact(path: try map(layers: 2, sizes: ["a": 3000, "b": 3000, "c": 3000, "n": 20]))
        let layers = (0..<2).map { l in ["a", "n", "b", "c", "b", "c"].map { "L.\(l).\($0)" } }
        try artifact.streamUnits(ordered: layers)
        let stream = try XCTUnwrap(artifact.stream)
        XCTAssertEqual(stream.slotBytes, 3008)
        for _ in 0..<2 {
            for name in layers.joined() {
                let t = try XCTUnwrap(artifact.tensors[name])
                let p = try XCTUnwrap(artifact.pointer(name))
                XCTAssertEqual(p.load(fromByteOffset: 7, as: UInt8.self), Self.byte(t.offset + 7), name)
                XCTAssertEqual(p.load(fromByteOffset: t.bytes - 1, as: UInt8.self), Self.byte(t.offset + t.bytes - 1), name)
                artifact.dropFromCache([name])
            }
        }
        XCTAssertEqual(stream.fallbacks, 0)
    }

    func testTheStreamServesTheEnginesOrderFromSmallBuffersAndReadsOnByItself() throws {
        let sizes = ["a": 4000, "b": 100, "c": 2500, "d": 20, "e": 4000, "f": 700]
        let artifact = try Artifact(path: try map(layers: 4, sizes: sizes))
        // The engine's order, not the file's.
        let order = ["e", "b", "d", "c", "a", "f"]
        let layers = (1..<4).map { l in order.map { "L.\(l).\($0)" } }
        let cap = layers.joined().map { artifact.stagingLength(of: $0) }.max()!
        XCTAssertEqual(cap, 4000)
        XCTAssertEqual(cap, Artifact.largestRun(header: artifact.header, prefixes: ["L."], fileSize: artifact.size))
        let units = artifact.stagingUnits(layers: layers, cap: cap)
        XCTAssertEqual(units.joined().count, 18)
        try artifact.streamTail(blocks: units, slots: 3, sequential: true, buffer: cap)
        let stream = try XCTUnwrap(artifact.stream)
        XCTAssertEqual(stream.slotBytes, cap)
        XCTAssertEqual(stream.slotCount, 3)
        // Three evaluations: the stream wraps from the last unit to the first.
        for _ in 0..<3 {
            for name in layers.joined() {
                let t = try XCTUnwrap(artifact.tensors[name])
                let p = try XCTUnwrap(artifact.pointer(name))
                XCTAssertTrue(UnsafeBufferPointer(start: p.assumingMemoryBound(to: UInt8.self), count: t.bytes)
                    .enumerated().allSatisfy { $0.element == Self.byte(t.offset + $0.offset) }, name)
                artifact.dropFromCache([name])
            }
            // The layer's end repeats the calls: nothing happens, nothing is dropped from the map.
            for layer in layers { artifact.dropFromCache(layer) }
        }
        XCTAssertEqual(stream.fallbacks, 0)
    }

    func testATensorAskedOutOfOrderIsStillRight() throws {
        let artifact = try Artifact(path: try map(layers: 2, sizes: ["a": 3000, "b": 3000, "c": 3000]))
        let names = artifact.order
        let units = artifact.stagingUnits(layers: [names], cap: 3008)
        try artifact.streamTail(blocks: units, slots: 2, sequential: true, buffer: 3008)
        for name in names.reversed() {
            let t = try XCTUnwrap(artifact.tensors[name])
            let p = try XCTUnwrap(artifact.pointer(name))
            XCTAssertEqual(p.load(fromByteOffset: 0, as: UInt8.self), Self.byte(t.offset))
            XCTAssertEqual(p.load(fromByteOffset: t.bytes - 1, as: UInt8.self), Self.byte(t.offset + t.bytes - 1))
            artifact.dropFromCache([name])
        }
    }
}
