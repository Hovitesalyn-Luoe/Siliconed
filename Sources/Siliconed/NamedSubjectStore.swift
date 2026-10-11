import Foundation

/// **The named subjects on disk: one readable folder per subject.**
///
///     <root>/subjects/
///         alice/              the name as typed (names are case-insensitive: one folder per subject)
///             subject.json    {"name": "alice", "kind": "person", "pictures": [{"file": "1.jpg", "frame": {…}}]}
///             1.jpg           the picture as it was given — bytes copied, never re-encoded
///             2.png           a second picture of the same subject (two at most)
///
/// **Visible files only** (the app shows every subject this folder holds, and nothing else lives
/// there): deleting a subject deletes its folder, pictures included; a picture its JSON no longer
/// names is deleted at the next reading, and so is a folder without `subject.json` (a creation cut
/// short). A creation writes into a hidden `.partial-…` folder and renames it whole: a crash never
/// leaves a half subject. **The sweep spares what is young** (`grace`): another instance of the app
/// may be writing it at this very moment — its staging folder, the picture an edit adds before its
/// JSON names it. A staging folder that holds a whole subject (a crash between its JSON and its
/// rename, or in a change of case's step aside) is put in place, not deleted: « Create » was clicked.
///
/// The frame is kept, not a cropped copy (`NamedSubjects.Frame`): the picture is the user's own,
/// whole, and can be reframed. Writing here is the user's gesture (create, edit, delete), like
/// « Save »: nothing is written behind their back.
public struct NamedSubjectStore: Sendable {
    public let folder: URL
    static let descriptor = "subject.json"
    private static let partial = ".partial-"
    /// How old a file must be before the sweep takes it for an abandoned one (its status change
    /// time: a copy or a rename renews it, unlike the modification date a copy keeps).
    static let grace: TimeInterval = 600

    public init(library: Library) { folder = library.root.appendingPathComponent("subjects", isDirectory: true) }
    public init(folder: URL) { self.folder = folder }

    /// What a picture is made from: a file (copied, its extension kept), or PNG bytes.
    public enum Source: Sendable {
        case file(URL)
        case png(Data)
    }

    public enum Failure: Error, Equatable, Sendable, CustomStringConvertible {
        case badName(NamedSubjects.NameProblem)
        case noSuchSubject(String)
        case pictures(Int)
        case unreadable(String)
        case io(String)

        public var description: String {
            switch self {
            case .badName(let p): "name refused: \(p)"
            case .noSuchSubject(let n): "no subject @\(n)"
            case .pictures(let n): "\(n) pictures: a subject has 1 or \(NamedSubjects.maxPictures)"
            case .unreadable(let f): "unreadable: \(f)"
            case .io(let m): m
            }
        }
    }

    /// A subject's folder.
    public func folder(of name: String) -> URL { folder.appendingPathComponent(name, isDirectory: true) }

    /// **A name or a file that stays one entry of its folder**: not empty, no `/`, not hidden (`.`,
    /// `..`, `.partial-…`) — what `delete` and `update` require of a name they did not check, and the
    /// JSON of a picture's file (a hand edit could otherwise point at `../../x`).
    static func isEntry(_ name: String) -> Bool {
        !name.isEmpty && !name.contains("/") && !name.contains(":") && !name.hasPrefix(".")
    }

    /// A picture's file.
    public func url(of picture: NamedSubjects.Picture, in subject: NamedSubjects.Subject) -> URL {
        folder(of: subject.name).appendingPathComponent(picture.file)
    }

    // ── reading ──

    /// **The subjects, by name** (ignoring case). Removes on the way what no subject shows and is
    /// older than `grace`: an interrupted creation, a folder without its JSON, a picture its JSON does
    /// not name. `now`: the tests' clock.
    public func list(now: Date = Date()) -> [NamedSubjects.Subject] {
        let fm = FileManager.default
        guard let names = try? fm.contentsOfDirectory(atPath: folder.path) else { return [] }
        func abandoned(_ url: URL) -> Bool {
            let changed = (try? url.resourceValues(forKeys: [.attributeModificationDateKey]))?.attributeModificationDate
            return changed.map { now.timeIntervalSince($0) > Self.grace } ?? false
        }
        var subjects: [NamedSubjects.Subject] = []
        var queue = names.filter { $0 != ".DS_Store" }
        while !queue.isEmpty {
            let name = queue.removeFirst()
            let dir = folder.appendingPathComponent(name, isDirectory: true)
            var isFolder: ObjCBool = false
            guard fm.fileExists(atPath: dir.path, isDirectory: &isFolder), isFolder.boolValue else { continue }
            if name.hasPrefix(Self.partial) {
                guard abandoned(dir) else { continue }
                // A whole subject left aside is put in place when its name is free; anything else goes.
                if let s = Self.readDescriptor(in: dir), Self.isEntry(s.name), NamedSubjects.check(name: s.name) == nil,
                   !names.contains(where: { NamedSubjects.key($0) == NamedSubjects.key(s.name) }),
                   (try? fm.moveItem(at: dir, to: folder(of: s.name))) != nil {
                    queue.append(s.name)
                } else {
                    try? fm.removeItem(at: dir)
                }
                continue
            }
            let json = dir.appendingPathComponent(Self.descriptor)
            guard fm.fileExists(atPath: json.path) else {
                if abandoned(dir) { try? fm.removeItem(at: dir) }
                continue
            }
            // A JSON that does not read is left alone: it may be a later version's, or a hand edit
            // to fix — never deleted on a guess.
            guard var s = Self.readDescriptor(in: dir), NamedSubjects.key(s.name) == NamedSubjects.key(name) else { continue }
            s.name = name
            s.pictures = s.pictures
                .filter { Self.isEntry($0.file) && $0.file != Self.descriptor
                    && fm.fileExists(atPath: dir.appendingPathComponent($0.file).path) }
                .prefix(NamedSubjects.maxPictures).map { NamedSubjects.Picture(file: $0.file, frame: $0.frame.clamped()) }
            let kept = Set(s.pictures.map(\.file) + [Self.descriptor, ".DS_Store"])
            for file in (try? fm.contentsOfDirectory(atPath: dir.path)) ?? [] where !kept.contains(file) {
                let url = dir.appendingPathComponent(file)
                if abandoned(url) { try? fm.removeItem(at: url) }
            }
            subjects.append(s)
        }
        return subjects.sorted { $0.name.localizedStandardCompare($1.name) == .orderedAscending }
    }

    private static func readDescriptor(in dir: URL) -> NamedSubjects.Subject? {
        guard let data = try? Data(contentsOf: dir.appendingPathComponent(descriptor)) else { return nil }
        return try? JSONDecoder().decode(NamedSubjects.Subject.self, from: data)
    }

    // ── writing ──

    /// **Creates a subject**: its pictures copied, its JSON written last, the folder renamed into
    /// place whole.
    @discardableResult
    public func create(name: String, kind: NamedSubjects.Kind,
                        pictures: [(source: Source, frame: NamedSubjects.Frame)]) throws(Failure) -> NamedSubjects.Subject {
        if let p = NamedSubjects.check(name: name, taken: list().map(\.name)) { throw .badName(p) }
        guard (1...NamedSubjects.maxPictures).contains(pictures.count) else { throw .pictures(pictures.count) }
        let fm = FileManager.default
        let staging = folder.appendingPathComponent(Self.partial + UUID().uuidString, isDirectory: true)
        do {
            try fm.createDirectory(at: staging, withIntermediateDirectories: true)
            var placed: [NamedSubjects.Picture] = []
            for (i, p) in pictures.enumerated() {
                let file = try Self.write(p.source, as: "\(i + 1)", into: staging)
                placed.append(NamedSubjects.Picture(file: file, frame: p.frame.clamped()))
            }
            let subject = NamedSubjects.Subject(name: name, kind: kind, pictures: placed)
            try Self.writeDescriptor(subject, into: staging)
            try fm.moveItem(at: staging, to: folder(of: name))
            return subject
        } catch let f as Failure {
            try? fm.removeItem(at: staging)
            throw f
        } catch {
            try? fm.removeItem(at: staging)
            throw .io(Library.withoutHome(error.localizedDescription))
        }
    }

    /// **Changes a subject**: its name (the folder follows), its kind, its frames, its pictures —
    /// `added` appended (two at most), the pictures `subject` no longer lists deleted.
    @discardableResult
    public func update(_ old: String, to subject: NamedSubjects.Subject,
                        adding added: [(source: Source, frame: NamedSubjects.Frame)] = []) throws(Failure) -> NamedSubjects.Subject {
        guard Self.isEntry(old) else { throw .noSuchSubject(old) }
        let others = list().map(\.name).filter { NamedSubjects.key($0) != NamedSubjects.key(old) }
        if let p = NamedSubjects.check(name: subject.name, taken: others) { throw .badName(p) }
        let count = subject.pictures.count + added.count
        guard (1...NamedSubjects.maxPictures).contains(count) else { throw .pictures(count) }
        let fm = FileManager.default
        var dir = folder(of: old)
        guard fm.fileExists(atPath: dir.appendingPathComponent(Self.descriptor).path) else { throw .noSuchSubject(old) }
        do {
            if subject.name != old {
                if NamedSubjects.key(subject.name) == NamedSubjects.key(old) {
                    // A change of case alone needs a step aside on a case-insensitive disk (a crash
                    // in between is put back in place by `list`).
                    let aside = folder.appendingPathComponent(Self.partial + UUID().uuidString, isDirectory: true)
                    try fm.moveItem(at: dir, to: aside)
                    try fm.moveItem(at: aside, to: folder(of: subject.name))
                } else {
                    try fm.moveItem(at: dir, to: folder(of: subject.name))
                }
                dir = folder(of: subject.name)
            }
            var s = subject
            s.pictures = s.pictures.map { NamedSubjects.Picture(file: $0.file, frame: $0.frame.clamped()) }
            // The pictures no longer listed go first: a new picture may take a freed number.
            let listed = Set(s.pictures.map(\.file) + [Self.descriptor])
            for file in (try? fm.contentsOfDirectory(atPath: dir.path)) ?? [] where !listed.contains(file) {
                try? fm.removeItem(at: dir.appendingPathComponent(file))
            }
            for p in added {
                // The first free number: `1.jpg` deleted and a new one added gives `1.png`, not `3.png`.
                let stems = Set(s.pictures.map { ($0.file as NSString).deletingPathExtension })
                let stem = (1...9).map(String.init).first { !stems.contains($0) } ?? UUID().uuidString
                let file = try Self.write(p.source, as: stem, into: dir)
                s.pictures.append(NamedSubjects.Picture(file: file, frame: p.frame.clamped()))
            }
            try Self.writeDescriptor(s, into: dir)
            return s
        } catch let f as Failure {
            throw f
        } catch {
            throw .io(Library.withoutHome(error.localizedDescription))
        }
    }

    /// **Deletes a subject**: its folder, pictures included.
    public func delete(_ name: String) throws(Failure) {
        // Never the library's folder itself (`""`), nor a way out of it (`..`).
        guard Self.isEntry(name) else { throw .noSuchSubject(name) }
        let dir = folder(of: name)
        guard FileManager.default.fileExists(atPath: dir.path) else { throw .noSuchSubject(name) }
        do { try FileManager.default.removeItem(at: dir) } catch {
            throw .io(Library.withoutHome(error.localizedDescription))
        }
    }

    private static func write(_ source: Source, as stem: String, into dir: URL) throws -> String {
        switch source {
        case .file(let url):
            let ext = url.pathExtension.isEmpty ? "img" : url.pathExtension.lowercased()
            let file = "\(stem).\(ext)"
            guard FileManager.default.isReadableFile(atPath: url.path) else {
                throw Failure.unreadable(Library.withoutHome(url.path))
            }
            try FileManager.default.copyItem(at: url, to: dir.appendingPathComponent(file))
            return file
        case .png(let data):
            let file = "\(stem).png"
            try data.write(to: dir.appendingPathComponent(file), options: .atomic)
            return file
        }
    }

    private static func writeDescriptor(_ s: NamedSubjects.Subject, into dir: URL) throws {
        let e = JSONEncoder()
        e.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        try e.encode(s).write(to: dir.appendingPathComponent(descriptor), options: .atomic)
    }
}
