import Foundation
import Verification

// `siliconed` — every stage of every model checked against the reference implementation's goldens
// (`siliconed check list`, `docs/VERIFY.md`). The models are the ones the app installed.
setvbuf(stdout, nil, _IONBF, 0)
exit(Command.run(Array(CommandLine.arguments.dropFirst())))
