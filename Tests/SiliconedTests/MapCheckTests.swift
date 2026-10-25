import Foundation
import Testing
@testable import Siliconed
@testable import Verification

/// `check` names the map it judges and skips one that is not the golden's variant.
struct MapCheckTests {
    private func summary(_ dtypes: [String: Int]) -> MapSummary { MapSummary(path: "/tmp/x.silicon", bytes: 3_900_000_000, dtypes: dtypes) }

    @Test func variantFollowsTheTensorTypes() {
        #expect(summary(["bfloat16": 300, "float32": 4]).variant == .standard)
        #expect(summary(["bfloat16": 10, "float16": 2]).variant == .standard)
        #expect(summary(["int8": 200, "bfloat16": 20]).variant == .eightBit)
        #expect(summary(["float8_e4m3": 5, "bfloat16": 1]).variant == .eightBit)
        #expect(summary(["q4_k": 100, "q6_k": 3, "int8": 1]).variant == .packed)
    }

    @Test func quantizedMapIsRefusedAgainstTheStandardGolden() throws {
        let why = try #require(summary(["int8": 200, "bfloat16": 20]).mismatch(golden: .standard))
        #expect(why.contains("8-bit") && why.contains("drop --map"))
        #expect(summary(["bfloat16": 300]).mismatch(golden: .standard) == nil)
        #expect(summary(["q4_0": 9]).mismatch(golden: .standard) != nil)
    }

    @Test func aGoldenOfAnotherVariantAcceptsItsOwn() {
        #expect(summary(["int8": 200]).mismatch(golden: .eightBit) == nil)
        #expect(summary(["bfloat16": 300]).mismatch(golden: .eightBit) != nil)
    }

    @Test func lineNamesPathSizeAndTypes() {
        let line = summary(["bfloat16": 300, "float32": 4]).line
        #expect(line == "map /tmp/x.silicon · 3.9 GB · bfloat16 ×300, float32 ×4 (standard weights)")
        #expect(summary(["bfloat16": 1]).line.contains("· bfloat16 (standard"))
    }

    @Test func headerOfARealFileIsRead() throws {
        let path = NSTemporaryDirectory() + "mapcheck-\(UUID().uuidString).silicon"
        defer { try? FileManager.default.removeItem(atPath: path) }
        let json = #"{"page":16384,"tensors":{"a":{"offset":16384,"bytes":4,"shape":[2],"dtype":"int8"},"b":{"offset":32768,"bytes":4,"shape":[2],"dtype":"bfloat16"}}}"#
        var data = Data(Array("SILICON".utf8) + [2])
        withUnsafeBytes(of: UInt64(json.utf8.count)) { data.append(contentsOf: $0) }
        data.append(Data(json.utf8)); data.append(Data(count: 40000))
        try data.write(to: URL(fileURLWithPath: path))
        let s = try MapSummary(path: path)
        #expect(s.dtypes == ["int8": 1, "bfloat16": 1] && s.variant == .eightBit)
        #expect(throws: Skip.self) { try MapCheck.use(path) }
    }
}
