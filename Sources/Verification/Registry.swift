import Foundation

// **The registry: every stage that is verified, in one list.** `siliconed check list` prints it, the
// help is read from it, the verification guide's table is its Markdown, and the maintainer's closing run
// walks it. A stage that is not here is not verified.
//
//     siliconed check list [--markdown]
//     siliconed check kernels [NAME]                 the stages without weights, all or one
//     siliconed check <model> <stage> [options]      one stage (Invocation.swift: the options)

package enum Registry {
    /// In the order a reader meets them: the kernels, then each model from its text to its image.
    package static let stages: [Stage] = KernelStages.all + ZImageStages.all + QwenImageStages.all + AnimaStages.all + Krea2Stages.all

    package static func stage(_ model: String, _ name: String) -> Stage? {
        stages.first { $0.model == model && $0.name == name }
    }
}

/// `siliconed check …`: the dispatch. Returns the process's exit code. `stages`: the registry, or the
/// registry and a maintainer's own stages beside it.
package enum CheckCommand {
    package static func run(_ arguments: [String], stages all: [Stage] = Registry.stages) -> Int32 {
        guard let first = arguments.first else { print(usage); return 64 }
        switch first {
        case "list":
            print(arguments.dropFirst().first == "--markdown" ? markdown(all) : table(all))
            return 0
        case "kernels":
            let kernels = all.filter { $0.model == "kernels" }
            if let name = arguments.dropFirst().first, !name.hasPrefix("--") {
                guard let stage = kernels.first(where: { $0.name == name }) else {
                    return unknown("kernel '\(name)'", among: kernels.map(\.name))
                }
                return stage.execute(Array(arguments.dropFirst(2))).exitCode
            }
            var codes: Set<Int32> = []
            for stage in kernels {
                print("\n── \(stage.name)")
                codes.insert(stage.execute(Array(arguments.dropFirst())).exitCode)
            }
            // A failure outranks a skip that comes after it: 64, then 1, then 2, then 0.
            return [64, 1, 2].first(where: codes.contains) ?? 0
        default:
            let stages = all.filter { $0.model == first }
            guard !stages.isEmpty else {
                return unknown("model '\(first)'", among: all.reduce(into: []) { if !$0.contains($1.model) { $0.append($1.model) } })
            }
            guard let name = arguments.dropFirst().first else {
                print("siliconed check \(first) <stage>: " + stages.map(\.name).joined(separator: " · "))
                return 64
            }
            guard let stage = stages.first(where: { $0.name == name }) else {
                return unknown("stage '\(name)' of \(first)", among: stages.map(\.name))
            }
            return stage.execute(Array(arguments.dropFirst(2))).exitCode
        }
    }

    static let usage = """
        siliconed check list [--markdown]              every verified stage: what it proves, its rule, its oracle
        siliconed check kernels [NAME]                 the kernels against their definition, no weights
        siliconed check <model> <stage> [--size 512|1024|WxH] [--goldens DIR] …
        """

    private static func unknown(_ what: String, among: [String]) -> Int32 {
        FileHandle.standardError.write("✗ unknown \(what): \(among.joined(separator: " · "))\n".data(using: .utf8)!)
        return 64
    }

    /// The registry for a terminal: one stage per block.
    static func table(_ stages: [Stage]) -> String {
        stages.map { s in
            """
            \(s.model) \(s.name)\(s.sizes.count > 1 || s.sizes != ["512"] ? "  [" + s.sizes.joined(separator: " · ") + "]" : "")
                \(s.proves)
                rule: \(s.rule)
                oracle: \(s.oracle ?? "none — judged against its definition")
                weights: \(s.weights.isEmpty ? "none" : s.weights.joined(separator: ", "))
            """
        }.joined(separator: "\n")
    }

    /// The registry as the verification guide's table.
    static func markdown(_ stages: [Stage]) -> String {
        var lines = ["| Stage | Proves | Rule | Oracle | Weights |", "|---|---|---|---|---|"]
        for s in stages {
            // A `|` inside a cell (`max |Δ|`) would end it.
            let cell = { (text: String) in text.replacingOccurrences(of: "|", with: "\\|") }
            lines.append("| `\(s.model) \(s.name)` | \(cell(s.proves)) | \(cell(s.rule)) | "
                         + (s.oracle.map { "`\($0)`" } ?? "—") + " | "
                         + (s.weights.isEmpty ? "—" : s.weights.joined(separator: ", ")) + " |")
        }
        return lines.joined(separator: "\n")
    }
}
