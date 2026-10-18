// swift-tools-version:6.0
import Foundation
import PackageDescription

let package = Package(
    name: "Siliconed",
    platforms: [.macOS(.v15)],   // MPSGraph.scaledDotProductAttention : macOS 15+
    products: [
        .library(name: "Siliconed", targets: ["Siliconed"]),
        .executable(name: "SiliconedApp", targets: ["SiliconedApp"]),
        .executable(name: "silicontrol", targets: ["Silicontrol"]),
    ],
    targets: [
        // The current CBLAS headers of Accelerate (the old ones are deprecated since macOS 13.3).
        // Same routines, same 32-bit integers: ILP64 is not requested.
        .target(name: "Siliconed", dependencies: ["AMXLegacy"], cSettings: [.define("ACCELERATE_NEW_LAPACK")]),
        // `BNNSMatMul`, deprecated and kept: isolated in C, where the deprecation is acknowledged once.
        .target(name: "AMXLegacy", linkerSettings: [.linkedFramework("Accelerate")]),
        // **The app, `Siliconed.app` — the product.** A client of the library as any app would be
        // (no `@testable`): what it calls is `public`, the rest is `package`. The target is `SiliconedApp`, not
        // `Siliconed`: the library holds that name, and the file system does not tell it from
        // `siliconed`. `tools/app.sh` copies its binary to `Siliconed.app/Contents/MacOS/Siliconed`.
        // Its phrases (`Localizable.xcstrings`, English source, fr de es it translations) are not a resource of it:
        // `tools/app.sh` compiles them into `Siliconed.app/Contents/Resources`, where `Text("…")`
        // looks for them (the main bundle). Under `swift run`, the app speaks the source language.
        .executableTarget(name: "SiliconedApp", dependencies: ["Siliconed", "SilicontrolHelp"],
                          exclude: ["Localizable.xcstrings"]),
        // **`silicontrol`, the app's remote control**, shipped as
        // `Siliconed.app/Contents/Helpers/silicontrol`: a pipe to the running app's socket,
        // Foundation only — it does not link the engine.
        .executableTarget(name: "Silicontrol", dependencies: ["SilicontrolHelp"]),
        // **The remote control's help**, Foundation only: one text for both sides — `silicontrol`
        // answers `help` without launching the app, the app answers it on its socket.
        .target(name: "SilicontrolHelp"),
        // **The fast target.** It touches neither a model's map nor a golden tensor: it exercises
        // pure functions and small synthetic files, in a few seconds and without a byte of
        // weights. A few tests do touch Metal — the GPU dequantization of synthetic 8-bit maps on
        // the Metal device (`GGUFTests`, `QuantizedMapTests`), an `MPSGraph` built to fail on
        // purpose (`TokenizerAndVAETests`) — on kilobytes, in milliseconds.
        .testTarget(name: "SiliconedTests", dependencies: ["Siliconed", "SilicontrolHelp"],
                    exclude: ["Fixtures"]),
    ]
)

// **A maintainer's command line, when present.** The checks against the reference implementations
// (diffusers and transformers in fp32) live outside this repository; when their sources sit in
// `Sources/CLI`, they build with the package as `siliconed-dev`, reaching its `package` symbols.
// A clone without them builds the library, the app and `silicontrol` alone.
if FileManager.default.fileExists(atPath: Context.packageDirectory + "/Sources/CLI") {
    package.products.append(.executable(name: "siliconed-dev", targets: ["CLI"]))
    package.targets.append(.executableTarget(name: "CLI", dependencies: ["Siliconed", "AMXLegacy"],
                                             cSettings: [.define("ACCELERATE_NEW_LAPACK")]))
}
