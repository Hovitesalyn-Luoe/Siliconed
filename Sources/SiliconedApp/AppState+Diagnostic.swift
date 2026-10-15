// **`AppState`: the diagnostic** (« Report My Configuration »), and `unmount` when the window closes.
// The state itself is in `AppState.swift`.

import AppKit
import ImageIO
import Siliconed
import SwiftUI
import UniformTypeIdentifiers

extension AppState {
    // MARK: Diagnostic

    /// Why the diagnostic cannot start now, or `nil`. It occupies the GPU and the CPU like a render —
    /// measured beside one, it would measure both.
    var diagnosticBlocker: String? {
        if diagnosticRunning { return String(localized: "A diagnostic is already running.") }
        if busy { return String(localized: "A render is running: the diagnostic can start once the queue is empty.") }
        if forgeInProgress != nil { return String(localized: "An installation is running: the diagnostic can start once it is done.") }
        return nil
    }

    func openDiagnostic() {
        managementOpen = false
        switch diagnostic {
        case .finished, .running: break
        case .ready, .failed: diagnostic = .ready
        }
        diagnosticOpen = true
    }

    /// **Runs `Diagnostic.run`** off the main thread (~30 s), the queue held meanwhile.
    func startDiagnostic() {
        guard diagnosticBlocker == nil else { return }
        let cancellation = Cancellation()
        diagnosticCancellation = cancellation
        let total = Diagnostic.Stage.allCases.count
        diagnostic = .running(done: 0, total: total, line: String(localized: "Preparing…"))
        Task.detached(priority: .userInitiated) { [weak self] in
            let outcome = Result { () throws(EngineError) -> Diagnostic.Report in
                try Diagnostic.run(cancellation: cancellation) { stage in
                    Task { @MainActor in self?.diagnosticProgress(stage) }
                }
            }
            await MainActor.run {
                guard let self else { return }
                self.diagnosticCancellation = nil
                switch outcome {
                case _ where cancellation.isCancelled: self.diagnostic = .ready
                case .success(let report):
                    self.diagnostic = .finished(report)
                    self.keep(report)   // what this Mac uses from the next launch
                case .failure(let error): self.diagnostic = .failed(Problem(error))
                }
                self.next()
            }
        }
    }

    func cancelDiagnostic() { diagnosticCancellation?.cancel() }

    /// The library names the stage (`Diagnostic.Stage`): the app says it in the user's language.
    private func diagnosticProgress(_ stage: Diagnostic.Stage) {
        guard case .running = diagnostic, let i = Diagnostic.Stage.allCases.firstIndex(of: stage) else { return }
        let line = switch stage {
        case .products: String(localized: "matrix products in fp32")
        case .lowPrecision: String(localized: "matrix products in fp16 and bf16")
        case .attention: String(localized: "attention")
        case .widening: String(localized: "weight widening")
        case .witness: String(localized: "the first measurement again")
        }
        diagnostic = .running(done: i, total: Diagnostic.Stage.allCases.count, line: line)
    }

    /// **« Open the issue »**: the GitHub form, prefilled by the report. A URL beyond ~8 000 characters is
    /// cut by browsers and GitHub: then the JSON goes to the clipboard, and the form opens with every
    /// field but the report's.
    /// Returns what to tell the user, or `nil` when the link carried everything.
    func openIssue(_ report: Diagnostic.Report) -> String? {
        do {
            let url = try report.issueURL()
            if url.absoluteString.count <= 8000 {
                NSWorkspace.shared.open(url)
                return nil
            }
            copyJSON(report)
            NSWorkspace.shared.open(try report.issueURL(includingReport: false))
            return String(localized: "The report is too long for a link: it is in the clipboard. Paste it into the form's “Report” field.")
        } catch {
            return Problem(error).text
        }
    }

    func copyJSON(_ report: Diagnostic.Report) {
        guard let data = try? report.json() else { return }
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(String(decoding: data, as: UTF8.self), forType: .string)
        flash(String(localized: "Report copied (JSON)"))
    }

    /// The window closes: nothing must keep occupying the GPU with no view to show it.
    func unmount() {
        toastTask?.cancel()
        clearQueue()
        task?.cancel()
        diagnosticCancellation?.cancel()
    }
}
