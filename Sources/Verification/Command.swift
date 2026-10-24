import Foundation

/// **`siliconed`'s verbs.** The executable is this dispatch and nothing else, so that the maintainer's
/// tool can carry the same verbs beside its own.
package enum Command {
    package static func run(_ arguments: [String], stages: [Stage] = Registry.stages) -> Int32 {
        switch arguments.first {
        case "check": return CheckCommand.run(Array(arguments.dropFirst()), stages: stages)
        case nil, "help", "--help", "-h": print(help); return 0
        case let verb?:
            FileHandle.standardError.write("✗ unknown command '\(verb)', see `siliconed help`\n".data(using: .utf8)!)
            return 64
        }
    }

    static var help: String {
        """
        siliconed — image generation on Apple silicon, checked against the reference implementations

        \(CheckCommand.usage)

        Exit codes: 0 passed · 1 failed · 2 skipped (an input is missing, said) · 64 usage.
        """
    }
}
