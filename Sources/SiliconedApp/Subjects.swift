// **Named subjects in the app — experimental, off by default**.
//
//     the subjects shelf    the library: one chip per subject (thumbnail, name, its kind's color);
//        │                  a click cites it, a drop or « + » names a new one (`SubjectSheet.swift`)
//        ▼
//     the prompt            `@alice in @red-coat` — each cited subject drawn as a chip in the text
//        │                  (`PromptField.swift`), completed at `@`, an unknown one underlined
//        ▼
//     the click             each expanded prompt rewritten (`NamedSubjects.rewrite`): the model's prompt,
//                           hidden but shown on demand, and the subjects' framed pictures appended to
//                           the edit's images as `Request.references` — the engine is unchanged
//
// **One switch turns it on** (`SubjectLibrary.enabledKey`, the app menu's « Experimental »); off, the
// app is exactly the app without it: the prompt is a `TextEditor`, nothing is rewritten, no box. Only
// Qwen-Image-2.1 shows it — the one visible family whose encoder reads images.
//
// **Visible files** (`NamedSubjectStore`): `<library>/subjects/<name>/`, a picture and a small JSON;
// deleting a subject deletes its folder. Creating, editing and deleting are the user's gestures:
// incognito holds (nothing written behind their back).
//
// The rewrite and the store (`NamedSubjects`, `NamedSubjectStore`) are public — the app is a pure
// client, and `tools/translations.sh` compiles its files without the package's name — but not yet
// in `docs/API.md`: the feature is experimental; it will be documented once its wording is settled.

import AppKit
import ImageIO
import Siliconed
import SwiftUI

/// **The subjects, as the app holds them**: the list read from disk, their thumbnails, the switch.
@MainActor @Observable
final class SubjectLibrary {
    /// **The switch** — a flag in the product, against the rule « one version lives » (`CLAUDE.md`):
    /// an exception wanted for an experimental feature, so that it ships hidden until its words have
    /// been judged by eye. One key; off by default.
    static let enabledKey = "namedSubjects"
    var enabled = UserDefaults.standard.bool(forKey: SubjectLibrary.enabledKey) {
        didSet {
            guard enabled != oldValue else { return }
            UserDefaults.standard.set(enabled, forKey: Self.enabledKey)
            if enabled { reload() }
        }
    }

    let store: NamedSubjectStore
    private(set) var all: [NamedSubjects.Subject] = []
    /// Bumped at each reading: the prompt field redraws its chips (a thumbnail, a name changed).
    private(set) var version = 0
    /// The thumbnails, by subject, picture and frame — decoded once, from a 512-px copy.
    @ObservationIgnored private var thumbnails: [String: NSImage] = [:]

    init(library: Library) {
        store = NamedSubjectStore(library: library)
        if enabled { reload() }
    }

    func reload() {
        all = store.list()
        thumbnails = [:]
        version += 1
    }

    func subject(named name: String) -> NamedSubjects.Subject? {
        all.first { $0.id == NamedSubjects.key(name) }
    }

    func url(_ s: NamedSubjects.Subject, _ p: NamedSubjects.Picture) -> URL { store.url(of: p, in: s) }

    /// **A picture's thumbnail, framed** — what the chips, the prompt and the sheet show.
    func thumbnail(_ s: NamedSubjects.Subject, picture: Int = 0) -> NSImage? {
        guard s.pictures.indices.contains(picture) else { return nil }
        let p = s.pictures[picture]
        let key = "\(s.id)/\(p.file)/\(p.frame.x),\(p.frame.y),\(p.frame.width),\(p.frame.height)"
        if let known = thumbnails[key] { return known }
        guard let upright = SubjectPictures.upright(url(s, p), maxSide: 512),
              let cropped = SubjectPictures.crop(upright, p.frame) else { return nil }
        let image = NSImage(cgImage: FinishedImage.reduce(cropped, side: 160) ?? cropped, size: .zero)
        thumbnails[key] = image
        return image
    }

    /// **The references a rewrite places**, in its order: each picture read at render time from its
    /// file, cropped to its frame (`ReferenceImage.Source.framed`).
    func references(_ r: NamedSubjects.Rewritten) -> [ReferenceImage] {
        // Never dropped, even unreadable: the prompt's numbers count every picture. A file gone since
        // fails the render with `image_unreadable`, naming it.
        r.placed.map { placed in
            let url = url(placed.subject, placed.picture)
            let size = ReferenceImage.size(of: url) ?? (width: 1024, height: 1024)
            let px = placed.picture.frame.pixels(width: size.width, height: size.height)
            let picture = placed.subject.pictures.firstIndex(of: placed.picture) ?? 0
            return ReferenceImage(source: .framed(url, placed.picture.frame),
                                  vignette: thumbnail(placed.subject, picture: picture) ?? NSImage(),
                                  name: "@" + placed.subject.name, width: px.width, height: px.height)
        }
    }

    // ── the user's gestures ──

    func create(name: String, kind: NamedSubjects.Kind,
                pictures: [(source: NamedSubjectStore.Source, frame: NamedSubjects.Frame)]) -> Problem? {
        defer { reload() }
        do { try store.create(name: name, kind: kind, pictures: pictures) } catch { return Problem(error) }
        return nil
    }

    func update(_ old: String, to s: NamedSubjects.Subject,
                adding: [(source: NamedSubjectStore.Source, frame: NamedSubjects.Frame)]) -> Problem? {
        defer { reload() }
        do { try store.update(old, to: s, adding: adding) } catch { return Problem(error) }
        return nil
    }

    func delete(_ s: NamedSubjects.Subject) -> Problem? {
        defer { reload() }
        do { try store.delete(s.name) } catch { return Problem(error) }
        return nil
    }
}

/// **Reading a picture upright**, whatever its EXIF orientation — the frame is in that space.
enum SubjectPictures {
    /// Decoded upright, its long side at most `maxSide` (`nil`: full size).
    static func upright(_ url: URL, maxSide: Int? = nil) -> CGImage? {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil) else { return nil }
        return upright(source, maxSide: maxSide, url: url)
    }

    static func upright(_ data: Data, maxSide: Int? = nil) -> CGImage? {
        guard let source = CGImageSourceCreateWithData(data as CFData, nil) else { return nil }
        return upright(source, maxSide: maxSide, url: nil)
    }

    private static func upright(_ source: CGImageSource, maxSide: Int?, url: URL?) -> CGImage? {
        var side = maxSide
        if side == nil, let url, let s = ReferenceImage.size(of: url) { side = max(s.width, s.height) }
        if side == nil, let p = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
           let w = p[kCGImagePropertyPixelWidth] as? Int, let h = p[kCGImagePropertyPixelHeight] as? Int {
            side = max(w, h)
        }
        let options: [CFString: Any] = [kCGImageSourceCreateThumbnailFromImageAlways: true,
                                        kCGImageSourceThumbnailMaxPixelSize: side ?? 4096,
                                        kCGImageSourceCreateThumbnailWithTransform: true]
        return CGImageSourceCreateThumbnailAtIndex(source, 0, options as CFDictionary)
    }

    /// The frame's pixels of an upright image (`Frame.pixels`: its ends rounded).
    static func crop(_ image: CGImage, _ frame: NamedSubjects.Frame) -> CGImage? {
        let p = frame.pixels(width: image.width, height: image.height)
        return image.cropping(to: CGRect(x: p.x, y: p.y, width: p.width, height: p.height))
    }
}

// MARK: - The kinds: their words and their colors

extension NamedSubjects.Kind {
    /// The kind's name, in the user's language.
    var label: String {
        switch self {
        case .person: String(localized: "Person", comment: "Kind of named subject")
        case .garment: String(localized: "Garment", comment: "Kind of named subject")
        case .object: String(localized: "Object", comment: "Kind of named subject")
        case .place: String(localized: "Place", comment: "Kind of named subject")
        case .style: String(localized: "Style", comment: "Kind of named subject")
        }
    }

    /// What the model takes from the picture — the instruction `NamedSubjects.Phrasing` writes, said
    /// to the user.
    var taken: String {
        switch self {
        case .person: String(localized: "Their face, hair and build — not the clothes or the background.")
        case .garment: String(localized: "The garment alone — not who wears it.")
        case .object: String(localized: "The object alone, its exact design.")
        case .place: String(localized: "The setting and its light — not its people.")
        case .style: String(localized: "The look: medium, palette, lighting — not the content.")
        }
    }

    var symbol: String {
        switch self {
        case .person: "person.fill"
        case .garment: "tshirt.fill"
        case .object: "cube.fill"
        case .place: "mountain.2.fill"
        case .style: "paintpalette.fill"
        }
    }

    /// **Five hues at the wires' lightness and chroma** (OKLCH 0.80 / 0.105 on a dark ground, 0.58 /
    /// 0.12 on a light one — `Palette`): rose, blue, olive, green, orchid, none louder than the others.
    var nsColor: NSColor {
        switch self {
        case .person: Self.dynamic(dark: 0xFAA2A2, light: 0xB65A5C)
        case .garment: Self.dynamic(dark: 0x7CC7FB, light: 0x2382BA)
        case .object: Self.dynamic(dark: 0xC8C26F, light: 0x857D0B)
        case .place: Self.dynamic(dark: 0x92D094, light: 0x478D4B)
        case .style: Self.dynamic(dark: 0xE0A6E3, light: 0x9E60A1)
        }
    }
    var color: Color { Color(nsColor: nsColor) }

    private static func dynamic(dark: Int, light: Int) -> NSColor {
        func color(_ v: Int) -> NSColor {
            NSColor(srgbRed: CGFloat(v >> 16 & 0xFF) / 255, green: CGFloat(v >> 8 & 0xFF) / 255, blue: CGFloat(v & 0xFF) / 255, alpha: 1)
        }
        return NSColor(name: nil) { a in a.bestMatch(from: [.darkAqua, .aqua]) == .darkAqua ? color(dark) : color(light) }
    }
}

// MARK: - The refusals, in the user's language

extension Problem {
    init(_ f: NamedSubjects.Failure) {
        switch f {
        case .unknownSubject(let name):
            self.init(String(localized: "No subject is called @\(name)."),
                      suggestion: String(localized: "Name it first: drop an image under Subjects, in the Images box."))
        case let .tooManyImages(count, max):
            self.init(String(localized: "This prompt takes \(String(count)) images: a render takes \(String(max)) at most."),
                      suggestion: String(localized: "A subject with two pictures counts twice, and so do the edit's images."))
        case .modelReadsNoImages:
            self.init(String(localized: "This model doesn't read images: its prompt can't cite subjects."))
        case .noPicture(let name):
            self.init(String(localized: "The subject @\(name) has no picture."))
        }
    }

    init(_ f: NamedSubjectStore.Failure) {
        switch f {
        case .badName(let p): self.init(p.sentence)
        case .noSuchSubject(let name): self.init(String(localized: "No subject is called @\(name)."))
        case .pictures: self.init(String(localized: "A subject has one or two pictures."))
        case .unreadable(let file): self.init(String(localized: "This image can't be read: \(file)"))
        case .io(let detail):
            self.init(String(localized: "The subject couldn't be written."))
            self.detail = detail
        }
    }
}

extension NamedSubjects.NameProblem {
    var sentence: String {
        switch self {
        case .empty: String(localized: "Give it a name.")
        case .tooLong: String(localized: "A name has \(NamedSubjects.maxNameLength) characters at most.")
        case .characters: String(localized: "Letters, digits, - and _ only, without spaces.")
        case .taken: String(localized: "A subject already has this name.")
        }
    }
}

// MARK: - The rack's and the render's side

extension AppState {
    /// **The subjects shelf shows, and `@names` are read**: the switch is on and the chosen model's
    /// encoder reads images (Qwen-Image-2.1).
    var subjectsActive: Bool { subjects.enabled && promptNamesImages && maxReferences > 0 }

    /// **One expanded prompt, rewritten** — `nil` when the feature is off or the prompt cites nobody.
    /// For the rack's model and edit images by default; the remote control passes its request's.
    func rewrite(_ expanded: String, model: Model? = nil,
                 editImages: Int? = nil) -> Result<NamedSubjects.Rewritten, NamedSubjects.Failure>? {
        guard subjects.enabled, !NamedSubjects.mentions(in: expanded).isEmpty else { return nil }
        let capacity = model.map { $0.chain.text.output.readsImages ? $0.chain.denoising.maxReferences : 0 }
            ?? (promptNamesImages ? maxReferences : 0)
        let first = (editImages ?? references.count) + 1
        return Result { () throws(NamedSubjects.Failure) in
            try NamedSubjects.rewrite(expanded, subjects: subjects.all, capacity: capacity, first: first)
        }
    }

    /// **The rewrite the rack shows**: the first expanded prompt's (a series cites the same subjects
    /// unless a group holds one — each job is rewritten on its own at the click).
    var shownRewrite: Result<NamedSubjects.Rewritten, NamedSubjects.Failure>? {
        rewrite((try? alternatives.get().prompts.first) ?? prompt)
    }

    /// What the subjects refuse, said like the rest of the form — the first refusal of any expansion.
    var subjectsBlocker: String? {
        guard subjects.enabled, let prompts = try? alternatives.get().prompts else { return nil }
        for p in prompts {
            if case .failure(let f)? = rewrite(p) { return Problem(f).title }
        }
        return nil
    }

    /// The images the subjects add to the next render (the first expansion's): what the cost, the
    /// plan and the learned speed count with the edit's own.
    var subjectImageCount: Int {
        if case .success(let r)? = shownRewrite { return r.placed.count }
        return 0
    }

    /// **The job's prompt and references for one expanded prompt**: rewritten when it cites subjects,
    /// the subjects' framed pictures after the edit's images.
    func namedJob(_ expanded: String, settings: inout RenderSettings, references edit: [ReferenceImage],
                  model: Model? = nil) -> [ReferenceImage] {
        guard case .success(let r)? = rewrite(expanded, model: model, editImages: edit.count), !r.placed.isEmpty else { return edit }
        let placed = subjects.references(r)
        settings.modelPrompt = r.prompt
        settings.subjectImages = placed.count
        settings.references = edit.count + placed.count
        return edit + placed
    }

    /// **`@name` inserted where the prompt's caret is** (a click on a chip) — appended when the field
    /// is not on screen. A space before it when it would glue to a word.
    func cite(_ s: NamedSubjects.Subject) {
        if let field = PromptTextView.current, field.window != nil {
            field.insertMention(s.name)
            return
        }
        let glue = prompt.last.map { !$0.isWhitespace } ?? false
        prompt += (glue ? " " : "") + "@\(s.name) "
    }

    // ── the sheet ──

    /// The open panel, then the sheet: an image to name.
    func chooseSubjectImage() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.image]
        panel.prompt = String(localized: "Choose")
        panel.message = String(localized: "An image to name: a person, a garment, an object, a place or a style.")
        guard panel.runModal() == .OK, let url = panel.url else { return }
        nameSubject(.file(url))
    }

    /// A file or PNG bytes dropped on the subjects shelf: the sheet opens on it.
    func nameSubject(_ source: NamedSubjectStore.Source) {
        guard let draft = SubjectDraft(new: source, taken: subjects.all.map(\.name)) else {
            flash(String(localized: "This image can't be read"))
            return
        }
        subjectDraft = draft
    }

    func editSubject(_ s: NamedSubjects.Subject) {
        subjectDraft = SubjectDraft(editing: s, library: subjects)
    }

    /// Deleting a subject deletes its folder — its pictures with it.
    func deleteSubject(_ s: NamedSubjects.Subject) {
        if let p = subjects.delete(s) { problem = p } else {
            flash(String(localized: "@\(s.name) deleted"))
        }
    }

    /// Asked first: the picture goes with it, and prompts that cite it are refused afterwards.
    func confirmDeletion(_ s: NamedSubjects.Subject) {
        let alert = NSAlert()
        alert.messageText = String(localized: "Delete @\(s.name)?")
        alert.informativeText = String(localized: "Its folder and its pictures are deleted from the library. Prompts that cite it will be refused.")
        alert.addButton(withTitle: String(localized: "Delete"))
        alert.addButton(withTitle: String(localized: "Cancel"))
        alert.buttons[0].hasDestructiveAction = true
        if alert.runModal() == .alertFirstButtonReturn { deleteSubject(s) }
    }

    /// A subject renamed: the prompt's mentions of it follow.
    func renameMentions(_ old: String, to new: String) {
        var text = prompt
        for m in NamedSubjects.mentions(in: prompt).reversed() where NamedSubjects.key(m.name) == NamedSubjects.key(old) {
            if let r = Range(NSRange(m.range, in: prompt), in: text) { text.replaceSubrange(r, with: "@" + new) }
        }
        prompt = text
    }

    func revealSubject(_ s: NamedSubjects.Subject) {
        NSWorkspace.shared.activateFileViewerSelecting([subjects.store.folder(of: s.name)])
    }
}
