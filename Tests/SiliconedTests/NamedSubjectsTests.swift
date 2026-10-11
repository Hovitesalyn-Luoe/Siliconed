import XCTest
@testable import Siliconed

/// **Named subjects** (`NamedSubjects`, `NamedSubjectStore`): `@alice` in a prompt → the model's prompt
/// and its references, and the folder they live in. Pure, plus a few files in a temporary folder.
final class NamedSubjectsTests: XCTestCase {
    private typealias N = NamedSubjects

    private func subject(_ name: String, _ kind: N.Kind, pictures: Int = 1) -> N.Subject {
        N.Subject(name: name, kind: kind, pictures: (1...max(pictures, 1)).prefix(pictures).map { N.Picture(file: "\($0).jpg") })
    }

    private lazy var library = [subject("alice", .person), subject("red-coat", .garment),
                                subject("the-beach", .place), subject("Bob", .person, pictures: 2),
                                subject("lamp", .object), subject("ink", .style)]

    private func names(_ p: String) -> [String] { N.mentions(in: p).map(\.name) }

    // ── mentions ──

    func testMentionsAreReadInOrderWithTheirRanges() {
        let p = "@alice in @red-coat on @the-beach."
        let m = N.mentions(in: p)
        XCTAssertEqual(m.map(\.name), ["alice", "red-coat", "the-beach"])
        XCTAssertEqual(m.map { String(p[$0.range]) }, ["@alice", "@red-coat", "@the-beach"])
    }

    func testWhatIsNotAMention() {
        XCTAssertEqual(names("mail me@home.org"), [])            // after a letter
        XCTAssertEqual(names("a lone @ sign, @, @-"), [])         // nothing a name has
        XCTAssertEqual(names("@@alice"), [])                       // an @ after an @
        XCTAssertEqual(names("(@alice)"), ["alice"])
        XCTAssertEqual(names("@alice_ and @bob-, @élodie"), ["alice", "bob", "élodie"])  // trailing joiners left out
        XCTAssertEqual(names("@a-b_c2"), ["a-b_c2"])
        XCTAssertEqual(names("line\n@alice"), ["alice"])
    }

    // ── names ──

    func testNamesAreCheckedForMentions() {
        XCTAssertNil(N.check(name: "red-coat"))
        XCTAssertNil(N.check(name: "élodie_2"))
        XCTAssertEqual(N.check(name: ""), .empty)
        XCTAssertEqual(N.check(name: String(repeating: "a", count: 33)), .tooLong)
        XCTAssertEqual(N.check(name: "red coat"), .characters)
        XCTAssertEqual(N.check(name: "-alice"), .characters)
        XCTAssertEqual(N.check(name: "alice-"), .characters)
        XCTAssertEqual(N.check(name: "Alice", taken: ["alice"]), .taken)
        // A valid name is cited whole.
        for n in ["red-coat", "a_b-c", "x"] { XCTAssertEqual(names("@\(n)!"), [n]) }
    }

    func testSuggestedNamesComeFromAFileName() {
        XCTAssertEqual(N.suggestedName("IMG 2041 (copy).jpeg"), "IMG-2041-copy-jpeg")
        XCTAssertEqual(N.suggestedName("  red coat  "), "red-coat")
        XCTAssertEqual(N.suggestedName("..."), "")
        XCTAssertNil(N.check(name: N.suggestedName("Portrait de Marie, été 2026.heic")))
    }

    // ── the rewrite ──

    func testAPromptWithoutMentionIsUnchanged() throws {
        for p in ["woman posing in a library", "", "me@home", "  spaces  "] {
            let r = try N.rewrite(p, subjects: library, capacity: 3)
            XCTAssertEqual(r.prompt, p)
            XCTAssertEqual(r.placed, [])
        }
        // Even for a model that reads no image.
        XCTAssertEqual(try N.rewrite("woman posing in a library", subjects: [], capacity: 0).prompt, "woman posing in a library")
    }

    func testTheNameStaysAndTheLegendNumbersByFirstAppearance() throws {
        let r = try N.rewrite("@alice in @red-coat on @the-beach", subjects: library, capacity: 3)
        XCTAssertEqual(r.placed.map(\.number), [1, 2, 3])
        XCTAssertEqual(r.placed.map(\.subject.name), ["alice", "red-coat", "the-beach"])
        XCTAssertEqual(r.cited.map(\.name), ["alice", "red-coat", "the-beach"])
        let lines = r.prompt.components(separatedBy: "\n")
        XCTAssertEqual(lines[0], "alice in red coat on the beach")
        XCTAssertEqual(lines[1], "")
        XCTAssertTrue(lines[2].hasPrefix("alice is the person in \(N.Phrasing.reference(1))."), lines[2])
        XCTAssertTrue(lines[3].hasPrefix("red coat is the garment in \(N.Phrasing.reference(2))."), lines[3])
        XCTAssertTrue(lines[4].hasPrefix("the beach is the place in \(N.Phrasing.reference(3))."), lines[4])
        XCTAssertEqual(lines.count, 5)
        XCTAssertFalse(r.prompt.contains("@"))
    }

    func testASubjectCitedTwiceIsPlacedOnce() throws {
        let r = try N.rewrite("@lamp on a desk, light from @LAMP", subjects: library, capacity: 3)
        XCTAssertEqual(r.placed.count, 1)
        // The mention's case is the user's; the name spoken is the library's.
        XCTAssertTrue(r.prompt.hasPrefix("lamp on a desk, light from lamp\n\n"), r.prompt)
    }

    func testAccentsCaseAndAddressesInARewrite() throws {
        let lib = [subject("Élodie", .person), subject("café-noir", .place)]
        // Any case, precomposed or not: the library's spelling is the one spoken.
        let r = try N.rewrite("@élodie at @CAFÉ-NOIR, mail elodie@cafe.fr", subjects: lib, capacity: 3)
        XCTAssertEqual(r.placed.map(\.subject.name), ["Élodie", "café-noir"])
        XCTAssertTrue(r.prompt.hasPrefix("Élodie at café noir, mail elodie@cafe.fr\n\n"), r.prompt)
        let decomposed = "@E\u{301}lodie"
        XCTAssertEqual(try N.rewrite(decomposed, subjects: lib, capacity: 3).placed.map(\.subject.name), ["Élodie"])
    }

    func testTwoPicturesOfOneSubjectTakeTwoPlaces() throws {
        let r = try N.rewrite("@Bob with @lamp", subjects: library, capacity: 3)
        XCTAssertEqual(r.placed.map(\.number), [1, 2, 3])
        XCTAssertEqual(r.placed.map(\.picture.file), ["1.jpg", "2.jpg", "1.jpg"])
        let both = N.Phrasing.references([1, 2])
        XCTAssertTrue(r.prompt.contains("Bob is the same person in \(both)."), r.prompt)
        XCTAssertTrue(r.prompt.contains("lamp is the object in \(N.Phrasing.reference(3))."), r.prompt)
    }

    func testAnEditsImagesComeFirst() throws {
        let r = try N.rewrite("put @alice in image 1", subjects: library, capacity: 3, first: 2)
        XCTAssertEqual(r.placed.map(\.number), [2])
        XCTAssertTrue(r.prompt.contains("in \(N.Phrasing.reference(2))."))
        XCTAssertThrowsError(try N.rewrite("@alice @lamp", subjects: library, capacity: 3, first: 3)) {
            XCTAssertEqual($0 as? N.Failure, .tooManyImages(count: 4, max: 3))
        }
    }

    func testTheRefusals() {
        func refusal(_ p: String, capacity: Int = 3) -> N.Failure? {
            do { _ = try N.rewrite(p, subjects: library, capacity: capacity); return nil } catch { return error }
        }
        XCTAssertEqual(refusal("@alice and @carol"), .unknownSubject(name: "carol"))
        // Three subjects, four pictures; the cap is ours even when the model reads four.
        XCTAssertEqual(refusal("@Bob @alice @lamp", capacity: 4), .tooManyImages(count: 4, max: 3))
        XCTAssertEqual(refusal("@alice @lamp", capacity: 1), .tooManyImages(count: 2, max: 1))
        XCTAssertEqual(refusal("@alice", capacity: 0), .modelReadsNoImages)
        // An unknown name is said before the model's limits: it is the user's typo to fix first.
        XCTAssertEqual(refusal("@nobody", capacity: 0), .unknownSubject(name: "nobody"))
        let bare = N.Subject(name: "ghost", kind: .object, pictures: [])
        XCTAssertThrowsError(try N.rewrite("@ghost", subjects: [bare], capacity: 3)) {
            XCTAssertEqual($0 as? N.Failure, .noPicture(name: "ghost"))
        }
    }

    func testEveryKindHasItsSentence() throws {
        for kind in N.Kind.allCases {
            let s = N.Phrasing.sentence(name: "x", kind: kind, numbers: [1])
            XCTAssertTrue(s.hasPrefix("x is the \(N.Phrasing.noun(kind)) "), s)
            XCTAssertTrue(s.hasSuffix("."), s)
        }
        XCTAssertTrue(try N.rewrite("in the manner of @ink", subjects: library, capacity: 3).prompt
            .contains("ink is the style of \(N.Phrasing.reference(1))."))
    }

    // ── the image's metadata ──

    /// The PNG keeps the prompt typed — what a drop or « Reuse » restores —, the model's beside it;
    /// nobody cited, the engine's metadata untouched.
    func testTheMetadataKeepsThePromptTyped() throws {
        let engine = ["prompt": "alice in a library\n\nalice is the person in <image1>.", "seed": "42",
                      "model": "qwen-image-2.1", "steps": "8", "format": "1024x1024", "reproducible": "true",
                      "Software": "Siliconed"]
        XCTAssertEqual(N.metadata(engine, typed: "@alice in a library", modelPrompt: nil), engine)
        let m = N.metadata(engine, typed: "@alice in a library", modelPrompt: engine["prompt"])
        XCTAssertEqual(m["prompt"], "@alice in a library")
        XCTAssertEqual(m[N.modelPromptKey], engine["prompt"])
        XCTAssertEqual(Set(m.keys), Set(engine.keys).union([N.modelPromptKey]))
        let recipe = try XCTUnwrap(Engine.Render.Recipe(metadata: m))
        XCTAssertEqual(recipe.prompt, "@alice in a library")
        XCTAssertEqual(recipe.unreadable, [])
        // Through the PNG's text chunks and back.
        let image = ImageRGB(pixels: [Float](repeating: 0, count: 3 * 2 * 2), height: 2, width: 2)
        let png = try image.png(metadata: m)
        XCTAssertEqual(PNG.text(png)["prompt"], "@alice in a library")
        XCTAssertEqual(PNG.text(png)[N.modelPromptKey], engine["prompt"])
    }

    // ── frames ──

    func testFramesClampAndRoundByTheirEnds() {
        let c = N.Frame(x: -0.2, y: 0.5, width: 2, height: 0.8).clamped()
        XCTAssertEqual([c.x, c.width, c.height], [0, 1, 0.8])
        XCTAssertEqual(c.y, 0.2, accuracy: 1e-12)
        XCTAssertEqual(N.Frame(x: .nan, y: 0, width: 0, height: 1).clamped().width, 0.02)
        let p = N.Frame(x: 0.25, y: 0.1, width: 0.5, height: 0.3).pixels(width: 1001, height: 333)
        XCTAssertEqual(p.x, 250); XCTAssertEqual(p.width, 751 - 250)   // ends 250.25 → 250, 750.75 → 751
        XCTAssertEqual(p.y, 33); XCTAssertEqual(p.height, 133 - 33)
        let whole = N.Frame.whole.pixels(width: 640, height: 480)
        XCTAssertEqual([whole.x, whole.y, whole.width, whole.height], [0, 0, 640, 480])
        let tiny = N.Frame(x: 0.999, y: 0.999, width: 0.001, height: 0.001).pixels(width: 10, height: 10)
        XCTAssertGreaterThanOrEqual(tiny.width, 1)
        XCTAssertLessThanOrEqual(tiny.x + tiny.width, 10)
    }

    // ── the folder ──

    private func temporaryStore() throws -> (NamedSubjectStore, URL) {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("subjects-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: root) }
        return (NamedSubjectStore(folder: root.appendingPathComponent("subjects")), root)
    }

    func testTheStoreCreatesUpdatesAndDeletesWholeFolders() throws {
        let (store, root) = try temporaryStore()
        let photo = root.appendingPathComponent("Photo.JPG")
        try Data([1, 2, 3]).write(to: photo)
        let frame = N.Frame(x: 0.1, y: 0.1, width: 0.5, height: 0.5)
        let alice = try store.create(name: "alice", kind: .person, pictures: [(.file(photo), frame)])
        XCTAssertEqual(alice.pictures, [N.Picture(file: "1.jpg", frame: frame)])
        XCTAssertEqual(store.list(), [alice])
        XCTAssertEqual(try Data(contentsOf: store.url(of: alice.pictures[0], in: alice)), Data([1, 2, 3]))
        // The name is taken, whatever its case.
        XCTAssertThrowsError(try store.create(name: "ALICE", kind: .object, pictures: [(.png(Data([9])), .whole)])) {
            XCTAssertEqual($0 as? NamedSubjectStore.Failure, .badName(.taken))
        }
        // Renamed, a second picture added: the folder follows.
        var renamed = alice
        renamed.name = "Alice-B"
        renamed.kind = .garment
        let b = try store.update("alice", to: renamed, adding: [(.png(Data([7])), .whole)])
        XCTAssertEqual(b.pictures.map(\.file), ["1.jpg", "2.png"])
        XCTAssertEqual(store.list().map(\.name), ["Alice-B"])
        XCTAssertEqual(store.list().first?.kind, .garment)
        // Its first picture removed: the file goes with it.
        var one = b
        one.pictures.removeFirst()
        try store.update("Alice-B", to: one)
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: store.folder(of: "Alice-B").path).sorted(),
                       ["2.png", "subject.json"])
        XCTAssertThrowsError(try store.update("Alice-B", to: N.Subject(name: "Alice-B", kind: .person, pictures: [])))
        // Deleted: nothing left on disk.
        try store.delete("Alice-B")
        XCTAssertEqual(store.list(), [])
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: store.folder.path), [])
    }

    /// **Visible files only**: what no subject shows is removed at the next reading.
    func testTheStoreRemovesWhatNoSubjectShows() throws {
        let (store, _) = try temporaryStore()
        let s = try store.create(name: "lamp", kind: .object, pictures: [(.png(Data([1])), .whole)])
        let fm = FileManager.default
        try fm.createDirectory(at: store.folder.appendingPathComponent(".partial-x"), withIntermediateDirectories: true)
        try fm.createDirectory(at: store.folder.appendingPathComponent("orphan"), withIntermediateDirectories: true)
        try Data([0]).write(to: store.folder(of: "lamp").appendingPathComponent("stray.png"))
        // Young, they may be another instance's work in progress: left alone.
        XCTAssertEqual(store.list(), [s])
        XCTAssertEqual(try fm.contentsOfDirectory(atPath: store.folder.path).sorted(), [".partial-x", "lamp", "orphan"])
        XCTAssertEqual(try fm.contentsOfDirectory(atPath: store.folder(of: "lamp").path).sorted(), ["1.png", "stray.png", "subject.json"])
        // Past the grace, removed.
        let later = Date().addingTimeInterval(NamedSubjectStore.grace + 60)
        XCTAssertEqual(store.list(now: later), [s])
        XCTAssertEqual(try fm.contentsOfDirectory(atPath: store.folder.path), ["lamp"])
        XCTAssertEqual(try fm.contentsOfDirectory(atPath: store.folder(of: "lamp").path).sorted(), ["1.png", "subject.json"])
    }

    /// A staging folder that holds a whole subject (a crash before its rename) is put in place.
    func testAWholeSubjectLeftAsideIsPutBack() throws {
        let (store, root) = try temporaryStore()
        let s = try store.create(name: "Lamp", kind: .object, pictures: [(.png(Data([1])), .whole)])
        let fm = FileManager.default
        try fm.moveItem(at: store.folder(of: "Lamp"), to: store.folder.appendingPathComponent(".partial-crash"))
        XCTAssertEqual(store.list(), [])
        XCTAssertEqual(store.list(now: Date().addingTimeInterval(NamedSubjectStore.grace + 60)), [s])
        XCTAssertEqual(try fm.contentsOfDirectory(atPath: store.folder.path), ["Lamp"])
        _ = root
    }

    /// Names and files read from disk never lead out of their folder.
    func testTheStoreStaysInItsFolder() throws {
        let (store, root) = try temporaryStore()
        let s = try store.create(name: "lamp", kind: .object, pictures: [(.png(Data([1])), .whole)])
        let outside = root.appendingPathComponent("outside.png")
        try Data([2]).write(to: outside)
        for name in ["", ".", "..", "../subjects", "lamp/1.png"] {
            XCTAssertThrowsError(try store.delete(name), name)
            XCTAssertThrowsError(try store.update(name, to: s), name)
        }
        XCTAssertEqual(store.list(), [s])
        // A hand-edited JSON pointing out of the folder: the picture is not read.
        let json = store.folder(of: "lamp").appendingPathComponent("subject.json")
        let edited = N.Subject(name: "lamp", kind: .object, pictures: [N.Picture(file: "../../outside.png"), N.Picture(file: "1.png")])
        try JSONEncoder().encode(edited).write(to: json)
        XCTAssertEqual(store.list().first?.pictures.map(\.file), ["1.png"])
        XCTAssertTrue(fm(outside))
    }

    private func fm(_ url: URL) -> Bool { FileManager.default.fileExists(atPath: url.path) }
}
