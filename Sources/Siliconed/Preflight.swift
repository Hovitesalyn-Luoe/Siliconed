import Foundation
import Synchronization

// **The preflights: what a render checks before its first computation, in this order.**
//
//     1. the model is installed          its maps and published files are all there (`ModelCard.missing`)
//     2. its license is accepted         `<racine>/accepted-licenses.json`, written by the app
//     3. the format                      side ≥ 512, multiple of 16, area ≤ 1024×1536 (`Format.check`)
//     4. the memory                      the request's computed floor ≤ B, this Mac's budget (`MemoryBudget`)
//     5. no other render                 `RenderLock.machine`, one for the machine, held for the whole render
//
// The first that fails throws its `EngineError`. 1, 2 and 5 need the model's library: a
// hand-composed chain (`Model(card:chain:)`, no library) has neither registry nor lock, and its
// files are checked by its own constructors. Then the request's own refusals follow (empty prompt,
// steps, references, strength) in `Engine.execute`, still before any computation.
//
// **The memory is B, fixed by this Mac's RAM and nothing else** (`MemoryBudget.ceiling(physical:)`,
// ¾ · RAM − 2 GiB): no reading of the kernel's state decides — the same render on the same Mac gets
// the same answer, always. Against it, the **floor**: the need of the most economical plan,
// computed from the request's shapes (`MemoryNeed`, `ModelCard.memoryNeed`) — what remains above it
// buys time (`MemoryPlan`), never another image. The app asks the same function ahead, to grey out
// what this Mac cannot do (`ModelCard.checkMemory`).
//
// The plan is then chosen in the same B (`MemoryPlan.choose`): the fastest whose need ≤ B.

package enum Preflight {

    /// **The checks 1 to 4, pure** — what the test target drives with an injected machine and catalog.
    /// `memory`: the request's own inputs (prompt, references' sizes); without it, the references
    /// at the output's size and a typical prompt.
    package static func check(card: ModelCard, width: Int, height: Int, references: Int = 0, installed: Bool,
                              licenseAccepted: Bool, ceiling: Int,
                              memory: MemoryNeed.Request? = nil) throws(EngineError) {
        guard installed else { throw .modelNotInstalled(model: card.id) }
        guard licenseAccepted else { throw .licenseNotAccepted(model: card.id) }
        try Format.check(width: width, height: height)
        if let memory {
            try card.checkMemory(memory, ceiling: ceiling)
        } else {
            try card.checkMemory(width: width, height: height, references: references, ceiling: ceiling)
        }
    }

    /// **All five, for a model about to render**: the lock comes back held, released when dropped.
    package static func run(_ model: Model, width: Int, height: Int, references: Int = 0,
                            ceiling: Int = MemoryBudget.machineCeiling(),
                            memory: MemoryNeed.Request? = nil) throws(EngineError) -> RenderLock? {
        let library = model.library
        try check(card: model.card, width: width, height: height, references: references,
                  installed: library.map { model.card.missing(in: $0) == nil } ?? true,
                  licenseAccepted: library.map { Licenses.implicit || $0.isLicenseAccepted(model.card) } ?? true,
                  ceiling: ceiling, memory: memory)
        return library == nil ? nil : try RenderLock.acquire(RenderLock.machine)
    }
}

// ── 4 · the memory a render needs ────────────────────────────────────────────────────────

extension ModelCard {
    /// **The request the need is computed on** (`MemoryNeed.Request`): the format, each reference at
    /// its size, the prompt's tokens (or `MemoryNeed.Request.typicalPromptTokens` when no prompt is known
    /// yet — the app's display), the LoRA ranks.
    package func memoryRequest(width: Int, height: Int, references: [(width: Int, height: Int)] = [],
                               promptTokens: Int? = nil, loraRanks: [Int] = []) -> MemoryNeed.Request {
        let tokens = promptTokens ?? MemoryNeed.Request.promptTokens(text: MemoryNeed.Request.typicalPromptTokens,
                                                                     family: family, references: references.count)
        return MemoryNeed.Request(card: self, width: width, height: height, references: references,
                                  promptTokens: tokens, loraRanks: loraRanks)
    }

    /// **Refuses a render whose floor exceeds B** — the one memory check, the engine's preflight and the
    /// app's greying out alike. `ceiling`: B, this Mac's by default (`MemoryBudget.machineCeiling()`,
    /// `SILICONED_MEMORY_BUDGET_GB` included — no reading of the kernel). `references`: the edit's
    /// images, named subjects' included (0: a generation), taken at the output's size.
    public func checkMemory(width: Int, height: Int, references: Int = 0,
                            ceiling: Int = MemoryBudget.machineCeiling()) throws(EngineError) {
        try checkMemory(memoryRequest(width: width, height: height,
                                      references: Array(repeating: (width, height), count: references)), ceiling: ceiling)
    }

    /// **The same check on a whole request** — what the engine asks, with the prompt, the references'
    /// own sizes and the LoRA ranks: the lean plan's need against B, nothing else.
    package func checkMemory(_ request: MemoryNeed.Request, ceiling: Int = MemoryBudget.machineCeiling()) throws(EngineError) {
        let needed = memoryNeed(request)
        guard needed <= ceiling else { throw .insufficientMemory(needed: needed, budget: ceiling) }
    }

    /// **The memory a render of this model needs at `width × height`**, in bytes: its floor — the peak
    /// of the most economical plan (`MemoryNeed.Request.lean`), computed from the shapes
    /// (`MemoryNeed.of`). Above the floor the plan buys time (`MemoryPlan.choose`); at the floor the render is
    /// slower, never another image. The four hidden families take the marked fallback of their
    /// measured peaks (`MemoryNeed.unmodelled`). 0 when the model's maps are not installed (the
    /// preflight's first check refuses it before).
    public func memoryNeed(width: Int, height: Int, references: Int = 0) -> Int {
        memoryNeed(memoryRequest(width: width, height: height, references: Array(repeating: (width, height), count: references)))
    }

    package func memoryNeed(_ request: MemoryNeed.Request) -> Int { MemoryNeed.of(request.lean)?.peak ?? 0 }

    /// **The fastest plan's peak at `width × height`**: every buffer that buys time taken — what
    /// `MemoryPlan.choose` takes whole when B holds it.
    package func memoryComfort(width: Int, height: Int, references: Int = 0) -> Int {
        MemoryNeed.of(memoryRequest(width: width, height: height, references: Array(repeating: (width, height), count: references)))?.peak ?? 0
    }
}

// ── 2 · the accepted licenses ────────────────────────────────────────────────────────────

/// The developer CLI accepts the licenses implicitly (it says so when it renders); an app never does.
package enum Licenses {
    private static let implicitFlag = Atomic<Bool>(false)
    package static var implicit: Bool {
        get { implicitFlag.load(ordering: .relaxed) }
        set { implicitFlag.store(newValue, ordering: .relaxed) }
    }
}

extension Library {
    /// `<racine>/accepted-licenses.json` — `{ model identifier: license text accepted }`. The text is
    /// kept so that a license that changes asks again.
    package var acceptedLicensesFile: String { root.appendingPathComponent("accepted-licenses.json").path }

    /// The models whose license was accepted, with the text accepted. Empty if the file is absent
    /// or unreadable — nothing is accepted by default.
    public func acceptedLicenses() -> [String: String] {
        guard let data = FileManager.default.contents(atPath: acceptedLicensesFile),
              let object = try? JSONSerialization.jsonObject(with: data) as? [String: String] else { return [:] }
        return object
    }

    /// Accepted, and for the license the card states today.
    public func isLicenseAccepted(_ card: ModelCard) -> Bool {
        acceptedLicenses()[card.id] == card.license.text
    }

    /// **Records that the user accepted `card`'s license** — what the app writes after showing it.
    public func acceptLicense(_ card: ModelCard) throws(EngineError) {
        var accepted = acceptedLicenses()
        accepted[card.id] = card.license.text
        try writeAcceptedLicenses(accepted)
    }

    /// Withdraws an acceptance: the next render of `card` is refused until it is accepted again.
    package func revokeLicense(_ card: ModelCard) throws(EngineError) {
        var accepted = acceptedLicenses()
        guard accepted.removeValue(forKey: card.id) != nil else { return }
        try writeAcceptedLicenses(accepted)
    }

    private func writeAcceptedLicenses(_ accepted: [String: String]) throws(EngineError) {
        try EngineError.boundary {
            try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
            let data = try JSONSerialization.data(withJSONObject: accepted, options: [.prettyPrinted, .sortedKeys])
            try data.write(to: URL(fileURLWithPath: acceptedLicensesFile), options: .atomic)
        }
    }
}

// ── 5 · one render on the machine ────────────────────────────────────────────────────────

/// **"Two engines do not fit in 16 GB", enforced across processes.** Within a process the engine
/// queue already serializes renders (`Engine.file`); between processes — the app and the CLI, two
/// libraries, two users — an advisory `flock` on `RenderLock.machine`, taken without waiting: the
/// second one is refused (`renderAlreadyRunning`) instead of swapping both. The CLI also asks the
/// app whether its queue is busy (`appBusy`); this lock is what the engine itself holds.
///
/// The kernel releases it when the process dies: a crash never leaves the machine locked.
package final class RenderLock {
    private let descriptor: Int32

    private init(_ descriptor: Int32) { self.descriptor = descriptor }

    /// **`/private/tmp/siliconed-render.lock` — one lock for the machine, every user.**
    ///
    /// It used to be `<root>/render.lock`, one per library: two processes on two libraries (the app
    /// and a CLI given `--library`, or two users) rendered side by side, while what the lock protects
    /// — the memory — is the machine's, not the library's nor the user's (fast user switching keeps
    /// both sessions' processes running). `/private/tmp` is the one directory every user can create in
    /// and nobody else can delete from (sticky bit), and it is emptied at boot, where no lock can
    /// outlive its holder anyway. Another user's file is opened read-only: `flock` does not need to write.
    package static let machine = "/private/tmp/siliconed-render.lock"

    /// The lock, held; `nil` if the file can neither be created nor opened (a machine without a
    /// writable `/tmp` renders unguarded, as before the lock existed).
    package static func acquire(_ path: String) throws(EngineError) -> RenderLock? {
        // 0o644 and read-only: the file another user created stays theirs, and still locks for us.
        let fd = open(path, O_RDONLY | O_CREAT, 0o644)
        guard fd >= 0 else { return nil }
        guard flock(fd, LOCK_EX | LOCK_NB) == 0 else {
            let busy = errno == EWOULDBLOCK
            close(fd)
            if busy { throw .renderAlreadyRunning }
            return nil
        }
        return RenderLock(fd)
    }

    deinit {
        flock(descriptor, LOCK_UN)
        close(descriptor)
    }
}
