import CryptoKit
import Foundation
import Synchronization

/// **Rebuilding a family from scratch**: download from the vendor what it publishes, forge, throw
/// away the sources. Siliconed ships no weights; it knows how to fetch them, and every file is
/// taken at a **pinned revision** (the commit that today's maps were forged from). Every weight
/// file (`.safetensors`, `.gguf`) is verified by its sha256 — the one pinned in the recipe, else the
/// one the Hub announces at that revision, and refused when there is neither; the small text files
/// (configs, tokenizers) are not LFS objects and have none. A rebuild yields the same maps.
///
///     store/composants/<famille>/   tokenizers, VAE, configs, DiT names: the small files the
///                                   engine reads as is (a few hundred MB)
///     store/<encoder>.silicon      the forged text encoder
///     store/<dit>.silicon           the forged vendor DiT (optional: a user who only brings their
///                                   own checkpoint does not need it)
///
/// Downloads live under `<root>/telechargements/` for the duration of a forge, then go away.
package struct Installer {
    package let library: Library
    /// Keep the downloaded sources (to reforge without redownloading).
    package var keepSources = false
    package var journal: @Sendable (String) -> Void = { print($0) }
    /// Once raised, it stops the installation between two files or in the middle of a download
    /// (`EngineError.cancelled`). The files already downloaded in full stay in `telechargements/` and
    /// are reused by the next attempt, sha256 checked; a file cut midway starts over from its first
    /// byte. A forge already begun runs to completion.
    package var cancellation: Cancellation?

    package init(library: Library) { self.library = library }

    // MARK: - What each family goes to fetch

    struct Repository { let name: String; let revision: String }
    /// `sha256`: the LFS object pinned in the recipe (a Compact's third-party files): the Hub must
    /// announce it, and the bytes must hash to it. Without it, the sha256 the Hub announces at the
    /// pinned revision is checked.
    struct File { let repository: Repository; let path: String; let destination: String?; var sha256: String? = nil }

    static let zImage = Repository(name: "Tongyi-MAI/Z-Image-Turbo", revision: "f332072aa78be7aecdf3ee76d5c247082da564a6")
    static let animaBase = Repository(name: "circlestone-labs/Anima-Base-v1.0-Diffusers", revision: "073c3a9db359c31ad0e8aa268d15775473c2176c")
    static let anima = Repository(name: "circlestone-labs/Anima", revision: "f973fc41ec7545364ac9776c2440285f43ff2a30")
    static let krea2 = Repository(name: "krea/Krea-2-Turbo", revision: "98e0fe118d17c9e3547fbb2e25acdbae2cadf7c7")
    /// The official Turbo, the Standard's every file: DiT, encoder, VAE, `processor/`, configs, schedule —
    /// and the Compact's and the Light's but their DiT and their encoder's weights. No recipe reads
    /// the base model (`Qwen/Qwen-Image-2.1`) any more.
    static let qwenImage21Turbo = Repository(name: "Qwen/Qwen-Image-2.1-Turbo", revision: "d65dbc9a7e8f6b5479e33dee6030eaab2a906509")
    // The Compacts' third parties: 8-bit weights of the same models, read by `Range` and
    // judged before being recipes.
    static let zImageGGUF = Repository(name: "unsloth/Z-Image-Turbo-GGUF", revision: "6c80814333b7b6a70a2e5b469a7c6437ce65de0f")
    static let zImageSDNQ = Repository(name: "Disty0/Z-Image-Turbo-SDNQ-int8", revision: "43d0947f91c14b93da75704be27565ee30694f21")
    // Qwen: the Compact's DiT is Comfy-Org's int8 convrot of the Turbo (the repository holds the base's
    // too); the 8-bit encoder is unsloth's of the base model's, which is the Turbo's to the bit (750
    // tensors) — no 8-bit encoder of the Turbo is published, and none is needed.
    static let qwenImage21Comfy = Repository(name: "Comfy-Org/Qwen-Image-2.1", revision: "df94239739eef7205973a145ffa6f441f7b64e84")
    static let qwenImage21Unsloth = Repository(name: "unsloth/Qwen-Image-2.1-FP8", revision: "bb21a8ba7f0371c19ad1f72c82a5e8fc10f39939")
    // The Light's DiT: unsloth's GGUF Q4_K_M, judged whole at the bit.
    // Z-Image's is in the Compact's repository (`zImageGGUF`), at the same revision.
    static let qwenImage21GGUF = Repository(name: "unsloth/Qwen-Image-2.1-Turbo-GGUF", revision: "5d1f9f759ea877ae556cda88dbec958b62ad3809")
    // Anima's Compact and Light DiTs: vanes430's GGUFs of Turbo v1.1 (ComfyUI-GGUF, `cosmos`), the
    // Compact's Q8_0 and the Light's Q4_K_M in one repository; its encoder is the Standard's.
    static let animaGGUF = Repository(name: "vanes430/Anima-Turbo-V1.1-GGUF", revision: "efdf070de980a52fc03027d2f6ed543cc1d87a8f")

    /// The small files read at render time, and their place under `store/composants/<famille>/`.
    static func components(_ f: Family) -> [File] {
        switch f {
        case .zImage:
            return [File(repository: zImage, path: "tokenizer/tokenizer.json", destination: "tokenizer/tokenizer.json"),
                    File(repository: zImage, path: "vae/diffusion_pytorch_model.safetensors", destination: "vae.safetensors"),
                    File(repository: zImage, path: "transformer/config.json", destination: "transformer.json")]
        case .anima:
            return [File(repository: animaBase, path: "tokenizer/tokenizer.json", destination: "tokenizer/tokenizer.json"),
                    File(repository: animaBase, path: "t5_tokenizer/tokenizer.json", destination: "t5_tokenizer/tokenizer.json"),
                    File(repository: animaBase, path: "vae/diffusion_pytorch_model.safetensors", destination: "vae.safetensors"),
                    File(repository: animaBase, path: "transformer/config.json", destination: "transformer.json")]
        case .krea2:
            return [File(repository: krea2, path: "tokenizer/tokenizer.json", destination: "tokenizer/tokenizer.json"),
                    File(repository: krea2, path: "model_index.json", destination: "model_index.json"),
                    File(repository: krea2, path: "vae/diffusion_pytorch_model.safetensors", destination: "vae.safetensors"),
                    File(repository: krea2, path: "transformer/config.json", destination: "transformer.json")]
        case .qwenImage21:
            // `processor/` as published (transformers 5's serialization: `tokenizer.json` holds the
            // vocabulary, `processor_config.json` the image processor's values). The engine reads
            // `tokenizer.json` alone — the same tokens as the base's — and has the image values in code
            // (`Qwen3VLImages`). The VAE is published in bf16 (the base's fp32 rounded to nearest):
            // widened to fp32 at load, computed in fp32; its config carries `latents_mean`/`latents_std`.
            // The schedule: `sample_sigmas` in `model_index.json`, checked against the scheduler config.
            return ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "processor_config.json"]
                .map { File(repository: qwenImage21Turbo, path: "processor/" + $0, destination: "processor/" + $0) }
                + [File(repository: qwenImage21Turbo, path: "vae/diffusion_pytorch_model.safetensors", destination: "vae.safetensors"),
                   File(repository: qwenImage21Turbo, path: "vae/config.json", destination: "vae.json"),
                   File(repository: qwenImage21Turbo, path: "transformer/config.json", destination: "transformer.json"),
                   File(repository: qwenImage21Turbo, path: "scheduler/scheduler_config.json", destination: "scheduler.json"),
                   File(repository: qwenImage21Turbo, path: "model_index.json", destination: "model_index.json")]
        }
    }

    /// **The license files a user reads before the first byte**, at the pinned revisions: the model's
    /// repository first, then each repository the installation also downloads from under a license
    /// file of its own. Checked by hand against the Hub (`/api/models/<repo>/revision/<sha>`):
    /// - Z-Image publishes no license file; Apache 2.0 is declared in the model card → `README.md`.
    /// - Anima's DiT comes from `circlestone-labs/Anima` (CircleStone Non-Commercial v1.2), its encoder,
    ///   VAE and tokenizers from the Diffusers repository, which carries its own file (v1.0).
    /// - Krea 2 is gated: `LICENSE.pdf` answers 401 until the Hub's gate is accepted (the card's
    ///   `license_link` points to the same PDF outside the Hub).
    /// - Qwen-Image-2.1-Turbo carries the base model's Qwen Research text (same git blob): one file.
    /// - **A Compact also downloads from third parties**, which publish no license file: their model
    ///   card (`README.md`) declares it — Apache 2.0 for unsloth's and Disty0's Z-Image, Qwen
    ///   Research (`license_link` to Qwen's file) for Comfy-Org's and unsloth's Qwen-Image-2.1 (the
    ///   Turbo's DiT; the encoder, the base's, the same tensors). They
    ///   come after the Standard's: the license is the publisher's, the cards say whose bytes.
    /// - **A Light** links the card of its DiT's repository (unsloth's GGUF: Apache 2.0 for Z-Image,
    ///   Qwen Research for Qwen-Image-2.1), then the card of its encoder's, the Compact's.
    /// - **Anima's Compact and Light** link vanes430's card after the publisher's two files. The card
    ///   says `apache-2.0` and `base_model: circlestone-labs/Anima`: the bytes are CircleStone's DiT
    ///   requantized, under CircleStone's license, which the publisher's files say — as for the others,
    ///   the license is the publisher's, the card says whose bytes. No encoder card: the Standard's.
    static func licenseFiles(_ f: Family, _ v: Variant = .standard) -> [File] {
        switch (f, v) {
        case (.zImage, .compact), (.zImage, .light):
            return licenseFiles(f) + [File(repository: zImageGGUF, path: "README.md", destination: nil),
                                      File(repository: zImageSDNQ, path: "README.md", destination: nil)]
        case (.qwenImage21, .compact):
            return licenseFiles(f) + [File(repository: qwenImage21Comfy, path: "README.md", destination: nil),
                                      File(repository: qwenImage21Unsloth, path: "README.md", destination: nil)]
        case (.qwenImage21, .light):
            return licenseFiles(f) + [File(repository: qwenImage21GGUF, path: "README.md", destination: nil),
                                      File(repository: qwenImage21Unsloth, path: "README.md", destination: nil)]
        case (.anima, .compact), (.anima, .light):
            return licenseFiles(f) + [File(repository: animaGGUF, path: "README.md", destination: nil)]
        default: break
        }
        switch f {
        case .zImage: return [File(repository: zImage, path: "README.md", destination: nil)]
        case .anima: return [File(repository: anima, path: "LICENSE.md", destination: nil),
                             File(repository: animaBase, path: "LICENSE.md", destination: nil)]
        case .krea2: return [File(repository: krea2, path: "LICENSE.pdf", destination: nil)]
        case .qwenImage21: return [File(repository: qwenImage21Turbo, path: "LICENSE", destination: nil)]
        }
    }

    /// `https://huggingface.co/<repo>/blob/<sha>/<file>` — the Hub's page for the file, at the revision
    /// the installation takes (`License.urls`).
    package static func licenseURLs(_ f: Family, _ v: Variant = .standard) -> [URL] {
        licenseFiles(f, v).map { Downloader.page($0.repository.name, revision: $0.repository.revision, path: $0.path) }
    }

    /// The published text encoder (a `text_encoder/` folder) — **its `config.json` first**. A
    /// Compact's comes from two repositories: the publisher's `config.json` (the third party's, when
    /// it publishes one, carries a `quantization_config` the forge has no use for), and the third
    /// party's 8-bit weights, which `ForgeText` keeps 8-bit. The Light's is the Compact's. Anima's
    /// three versions read the Standard's (`Family.encoderMap(_:)`).
    static func encoder(_ f: Family, _ v: Variant = .standard) -> [File] {
        switch (f, v) {
        case (.zImage, .compact), (.zImage, .light):
            return [File(repository: zImage, path: "text_encoder/config.json", destination: nil),
                    File(repository: zImageSDNQ, path: "text_encoder/model.safetensors", destination: nil,
                         sha256: "21a5e09e4d05ddb2900dc5c51db5e2bc7bc8b3fa08ba10b372798fba3c594677")]
        case (.qwenImage21, .compact), (.qwenImage21, .light):
            // The Turbo's `config.json` (the base's but `pad_token_id: null`), the base's 8-bit weights.
            return [File(repository: qwenImage21Turbo, path: "text_encoder/config.json", destination: nil),
                    File(repository: qwenImage21Unsloth, path: "Qwen-Image-2.1-text_encoder-INT8-ConvRot.safetensors",
                         destination: nil, sha256: "0cb5ed0653e2ec8065f25a6cfb2ab6a89b6c8f143a500acea19f0e36214814b7")]
        default: break
        }
        switch f {
        case .zImage:
            return ["config.json", "model.safetensors.index.json", "model-00001-of-00003.safetensors",
                    "model-00002-of-00003.safetensors", "model-00003-of-00003.safetensors"]
                .map { File(repository: zImage, path: "text_encoder/" + $0, destination: nil) }
        case .anima:
            return ["config.json", "model.safetensors"].map { File(repository: animaBase, path: "text_encoder/" + $0, destination: nil) }
        case .krea2:
            return ["config.json", "model.safetensors"].map { File(repository: krea2, path: "text_encoder/" + $0, destination: nil) }
        case .qwenImage21:
            // One file, the base's four shards to the bit (750 tensors): the map's tensors are the same.
            return ["config.json", "model.safetensors"].map { File(repository: qwenImage21Turbo, path: "text_encoder/" + $0, destination: nil) }
        }
    }

    /// The vendor's DiT — or a Compact's single 8-bit file: unsloth's GGUF Q8_0 of Z-Image,
    /// Comfy-Org's int8 convrot of Qwen-Image-2.1-Turbo, vanes430's GGUF Q8_0 of Anima, all kept
    /// 8-bit by `ForgeDiT` — or a Light's single GGUF Q4_K_M (unsloth's; vanes430's for Anima), its
    /// blocks kept whole. No Q4_K_M carries a type under 4 bits: Z-Image's Q4_K, Q5_K, Q6_K;
    /// Qwen-Turbo's Q4_K, Q5_K, Q6_K and Q8_0 (the base's types and offsets, two metadata keys more);
    /// Anima's below.
    static func dit(_ f: Family, _ v: Variant = .standard) -> [File] {
        switch (f, v) {
        case (.zImage, .light):
            return [File(repository: zImageGGUF, path: "z-image-turbo-Q4_K_M.gguf", destination: nil,
                         sha256: "e6494f87de6abaf6a561924f50317a5f271fc34bb4222aabbd801197df8f7daa")]
        case (.qwenImage21, .light):
            return [File(repository: qwenImage21GGUF, path: "qwen-image-2.1-turbo-Q4_K_M.gguf", destination: nil,
                         sha256: "8940e8b522af886e0c0f5afddf720e82dd223cb9a43580b5f61c4d740452ac20")]
        case (.zImage, .compact):
            return [File(repository: zImageGGUF, path: "z-image-turbo-Q8_0.gguf", destination: nil,
                         sha256: "f163d60b0eb427469510b8226243d196574a18139a2e40c017409cfbda95ecfe")]
        case (.anima, .light):
            // Q4_K ×386, Q6_K ×124, F32 ×169, BF16 ×6: nothing under 4 bits. 118 of its tensors are the
            // text adapter (Q4_K ×50, Q6_K ×12, F32 ×56), written dequantized in fp32 beside the map.
            return [File(repository: animaGGUF, path: "anima-turbo-v1.1-Q4_K_M.gguf", destination: nil,
                         sha256: "709fceeec6b4f0eb155330c83fd2539406e86c01e431407276a1f31095484782")]
        case (.anima, .compact):
            // Q8_0 ×510, F32 ×169, BF16 ×6 (the adapter: Q8_0 ×62, F32 ×56).
            return [File(repository: animaGGUF, path: "anima-turbo-v1.1-Q8_0.gguf", destination: nil,
                         sha256: "74d6c9251f34bc4152fda460dd7befaa9cec79946c270a378fee3d33048153bd")]
        case (.qwenImage21, .compact):
            // Its 31 `comfy_attention.config` (ComfyUI's attention preference) are ignored by the forge.
            return [File(repository: qwenImage21Comfy, path: "diffusion_models/qwen_image_2.1_turbo_int8_convrot.safetensors",
                         destination: nil, sha256: "8144a0182964aa8403ad6dbc011db837304167d235161b9aabaa8532e967314c")]
        default: break
        }
        switch f {
        case .zImage:
            return (["diffusion_pytorch_model.safetensors.index.json"] + (1...3).map { "diffusion_pytorch_model-0000\($0)-of-00003.safetensors" })
                .map { File(repository: zImage, path: "transformer/" + $0, destination: nil) }
        case .krea2:
            return (["diffusion_pytorch_model.safetensors.index.json"] + (1...3).map { "diffusion_pytorch_model-0000\($0)-of-00003.safetensors" })
                .map { File(repository: krea2, path: "transformer/" + $0, destination: nil) }
        case .anima:
            return [File(repository: anima, path: "split_files/diffusion_models/anima-turbo-v1.1.safetensors", destination: nil)]
        case .qwenImage21:
            // The Turbo's two shards, pinned: the base's names, shapes, dtypes and sizes, other values.
            let shards = ["6cccd922767f01694461bdcf1f34ea7b771f3442e4d5e338ea9aa55277431cc0",
                          "69f53ebb063d2f606bdaff7e22e0e2144f28d978d73dc75e6e4293fc5982eab5"]
            return [File(repository: qwenImage21Turbo, path: "transformer/diffusion_pytorch_model.safetensors.index.json", destination: nil)]
                + shards.enumerated().map {
                    File(repository: qwenImage21Turbo, path: "transformer/diffusion_pytorch_model-0000\($0.offset + 1)-of-00002.safetensors",
                         destination: nil, sha256: $0.element)
                }
        }
    }

    // MARK: - Where things live

    package func componentsFolder(_ f: Family) -> String {
        library.root.appendingPathComponent("store/composants/\(f.rawValue)").path
    }
    // on-disk layout, kept for existing libraries (telechargements/, store/composants/, *.partiel)
    var downloads: String { library.root.appendingPathComponent("telechargements").path }
    func source(_ file: File) -> String {
        downloads + "/" + file.repository.name.replacingOccurrences(of: "/", with: "--") + "/" + file.path
    }

    /// Are a family's components there (without its DiT or its encoder)? A superseded installation's are not.
    package func presentComponents(_ f: Family) -> Bool {
        let fm = FileManager.default
        return Self.components(f).allSatisfy { fm.fileExists(atPath: componentsFolder(f) + "/" + $0.destination!) }
            && fm.fileExists(atPath: componentsFolder(f) + "/dit.json")
            && supersededFiles(f).isEmpty
    }

    /// **What a superseded installation of the family left**: Qwen-Image-2.1 installed from the base
    /// model (Standard, or a third party's 8-bit or GGUF DiT of it in Compact or Light), with a third
    /// party's turbo LoRA and its schedule — its DiT map, and its whole components folder, whose files
    /// have the names of the Turbo's (`vae.safetensors`, `scheduler.json`) and not their content.
    /// Marked by one of its DiT maps, the LoRA, its schedule or the base's `preprocessor_config.json`.
    /// Removed by the next installation before its first byte — that version no longer renders
    /// (`Family.supersededQwenImage21Kind`) — and by an uninstallation; nothing is kept. The encoder
    /// maps stay: the Turbo's encoder is the base's to the bit, and the Compact's 8-bit encoder is
    /// still the one the Compact and the Light read.
    package func supersededFiles(_ f: Family) -> [String] {
        guard f == .qwenImage21 else { return [] }
        let folder = componentsFolder(f), maps = Family.supersededQwenImage21Maps.map(library.map)
        let marks = maps + [folder + "/viggle-turbo-v0.3.lora.silicon", folder + "/scheduler-turbo.json",
                            folder + "/processor/preprocessor_config.json"]
        guard marks.contains(where: Library.exists) else { return [] }
        return (maps + [folder]).filter(Library.exists)
    }

    /// The version a superseded installation of the family was in, read from its DiT map — what its
    /// reinstallation keeps when none is asked (`install`): the `variant` file goes with the folder,
    /// and the Compact's encoder alone would make a Light look like a Compact.
    func supersededVariant(_ f: Family) -> Variant? {
        guard f == .qwenImage21 else { return nil }
        let names = Family.supersededQwenImage21Maps
        if Library.exists(library.map(names[2])) { return .light }
        if Library.exists(library.map(names[1])) { return .compact }
        if Library.exists(library.map(names[0])) { return .standard }
        return nil
    }

    /// What is missing for a family to be able to render with **a** DiT (its own or an imported one):
    /// the encoder of the version installed.
    package func isReady(_ f: Family) -> Bool {
        presentComponents(f) && FileManager.default.fileExists(atPath: library.map(library.encoderMap(f)))
    }

    // MARK: - Installer

    /// **Installs a family** in `variant` (`nil`: the version already there, Standard if none):
    /// components, encoder, and the DiT if `baseDiT`. What is already there is neither redownloaded
    /// nor reforged. `baseDiT` false never changes the version installed (`Library.resolvedVariant`).
    ///
    /// **One version at a time, and never none**: the other version's DiT and encoder (the encoder
    /// unless another family reads it) are removed **once the new version is complete** — until
    /// then the render reads the old one (`Library.variant(of:)`), and a full disk, a cut, a Stop or
    /// a refused file leave it whole. Only when the disk cannot hold both at once does the old one
    /// go before the first byte (`Library.InstallPlan.removesFirst`, said before the user accepts),
    /// after a check that counts the space it gives back.
    ///
    /// **What overlaps**: each large file's sha256 runs behind its download; the DiT downloads
    /// while the encoder is forged, when the disk holds both at once (else one after the other, the
    /// plan's peak). Not the DiT's forge with its download, shard by shard: the map is one file in
    /// execution order behind one SHA-256 in that order, and its forge now takes ~10 s against
    /// minutes of download — what it would hide is not worth a second order of things.
    package func install(_ f: Family, variant: Variant? = nil, baseDiT: Bool = true) throws {
        let fm = FileManager.default
        let superseded = supersededFiles(f)
        let variant = variant ?? (superseded.isEmpty ? nil : supersededVariant(f) ?? library.installedVariant(of: f))
        if !superseded.isEmpty {
            let bytes = superseded.reduce(0) { $0 + Library.size($1) }
            journal(String(format: "  the superseded installation removed first (base model and third-party turbo LoRA): %.2f GB freed",
                           Double(bytes) / 1e9))
            for path in superseded { try fm.removeItem(atPath: path) }
        }
        let plan = try library.installPlan(f, variant: variant, baseDiT: baseDiT, free: library.freeSpace())
        let v = plan.variant
        journal("── \(f.name)" + (f.variants.count > 1 ? " (\(v.rawValue))" : ""))
        try cancellation.check()
        // **Before the first byte**: the whole installation against the volume, from the measured
        // sizes of the forged maps. Each part is checked again, exactly, from the Hub's headers.
        if plan.removesFirst, let old = plan.replacing {
            journal(String(format: "  not enough room for both versions: the %@ version is removed first (%.2f GB)",
                           old.rawValue, Double(plan.freed) / 1e9))
            try library.release(f, keeping: v)
        }
        try library.checkSpace(plan.peak)
        try fm.createDirectory(atPath: componentsFolder(f), withIntermediateDirectories: true)
        // The version aimed at, for `Library.variant(of:)` to break a tie (both versions complete for
        // an instant, or neither after a stop that followed `removesFirst`).
        if f.variants.count > 1 {
            try Data(v.rawValue.utf8).write(to: URL(fileURLWithPath: library.intendedVariantPath(f)), options: .atomic)
        }
        for c in Self.components(f) {
            let target = componentsFolder(f) + "/" + c.destination!
            if fm.fileExists(atPath: target) { continue }
            try Downloader.download(c.repository.name, revision: c.repository.revision, path: c.path, to: target,
                                         cancellation: cancellation, journal: journal)
        }
        let reference = componentsFolder(f) + "/dit.json"
        if !fm.fileExists(atPath: reference) {
            try writeReference(f, to: reference)
        }
        let config = try OrderedJSON.read(componentsFolder(f) + "/transformer.json")

        let encoderName = f.encoderMap(v), ditName = f.ditMap(v)
        let encoderMap = library.map(encoderName), ditMap = library.map(ditName)
        let ditPresent = fm.fileExists(atPath: ditMap)
            && (f != .anima || fm.fileExists(atPath: adapterPath(fromMap: ditMap)))
        let ditFiles = Self.dit(f, v)
        /// The DiT's sources downloaded and verified while the encoder was forged.
        var verified: Set<String> = []
        if !fm.fileExists(atPath: encoderMap) {
            let files = Self.encoder(f, v)
            try checkSpace(files)
            for x in files { try download(x) }
            // **The DiT downloads while the encoder is forged** — when the disk holds, at once, the
            // encoder's sources, its map, and the DiT's sources and map (`checkSpace(_:alsoWriting:)`);
            // otherwise one after the other, as the plan the user accepted counts it. The forge
            // then says no percentages: the app shows the download, and that a map is being prepared.
            journal("forging the encoder → \(encoderName)")
            let begin = Date()
            let beside = baseDiT && !ditPresent
                ? startDownloads(ditFiles, alsoWriting: files.reduce(0) { $0 + Library.size(source($1)) }) : nil
            let forged = Result { () throws -> MapWriter.Tally in
                if f.encoderMap(v) == f.encoderMap(.standard) {
                    return try ForgeText.forge(folder: (source(files[0]) as NSString).deletingLastPathComponent, family: f,
                                               to: encoderMap, progressHandler: beside == nil ? progressHandler : nil)
                }
                // Two repositories: the publisher's config, the third party's weights.
                let weights = files.dropFirst()
                return try ForgeText.forge(source: TensorSource(paths: weights.map(source)), published: OrderedJSON.read(source(files[0])),
                                           family: f, to: encoderMap,
                                           directory: weights[weights.startIndex].repository.name + "@" + weights[weights.startIndex].repository.revision,
                                           progressHandler: beside == nil ? progressHandler : nil)
            }
            if case .failure = forged, let beside { beside.stop(); _ = try? beside.wait() }
            let b = try forged.get()
            journal(String(format: "  %.2f GB, %d tensors, %d values rounded", Double(b.bytes) / 1e9, b.tensors, b.inexact))
            journal(String(format: "  ✓ forged %@ in %.0f s", encoderName, Date().timeIntervalSince(begin)))
            if !keepSources { for x in files { try? fm.removeItem(atPath: source(x)) } }
            if let beside { verified = try beside.wait() }
        }

        if baseDiT && !ditPresent {
            let files = ditFiles
            if verified.isEmpty { try checkSpace(files) }
            for x in files where !verified.contains(source(x)) { try download(x) }
            journal("forging the DiT → \(ditName)")
            let begin = Date()
            let sourceDiT: TensorSource
            let description: OrderedJSON
            if v != .standard {
                // A single third-party file, kept as published (`QuantizedLayouts`): GGUF Q8_0, int8
                // convrot (Compact), GGUF Q4_K_M blocks (Light).
                sourceDiT = try TensorSource(paths: [source(files[0])])
                description = .object([.init("file", .string((files[0].path as NSString).lastPathComponent)),
                                       .init("repository", .string(files[0].repository.name)),
                                       .init("revision", .string(files[0].repository.revision))])
            } else if f == .anima {
                sourceDiT = try TensorSource(paths: [source(files[0])])
                description = .object([.init("file", .string((files[0].path as NSString).lastPathComponent))])
            } else {
                let folder = (source(files[0]) as NSString).deletingLastPathComponent
                sourceDiT = try TensorSource(folder: folder)
                description = .object([.init("directory", .string(files[0].repository.revision)),
                                      .init("shards", .list(files.dropFirst().map { .string(($0.path as NSString).lastPathComponent) }))])
            }
            let r = try ForgeDiT.forge(source: sourceDiT, family: f, config: config, to: ditMap,
                                        reference: try readReference(f), descriptionSource: description, progressHandler: progressHandler)
            r.rows.forEach { journal("  " + $0) }
            journal(String(format: "  ✓ forged %@ in %.0f s", ditName, Date().timeIntervalSince(begin)))
            if !keepSources { for x in files { try? fm.removeItem(atPath: source(x)) } }
        }
        // The new version is complete: only now does the other one go.
        let freed = try library.release(f, keeping: v)
        if freed > 0 { journal(String(format: "  the other version removed: %.2f GB freed", Double(freed) / 1e9)) }
        if !keepSources { cleanDownloads() }
        journal("✓ \(f.name) installed")
    }

    /// **What an installation will occupy at its peak**, offline: the forged parts of `variant`
    /// still missing (`Family.installedSize`, measured), plus the sources of the largest of them,
    /// which live on disk while it is forged and are thrown away after. **A map is as large as its
    /// sources**: never wider, never narrower than published (bf16 stays bf16, 8-bit stays 8-bit)
    /// — so the sources weigh about their map (a little more when the forge drops tensors:
    /// `lm_head`, the layers past the one read). The exact figure is checked again before each
    /// download, from the Hub's headers (`checkSpace(_:)`). The DiT downloaded while the encoder is
    /// forged — both sources and both maps on disk at once — is not part of it: that happens only when
    /// the disk holds it then (`startDownloads`), never as a condition of the installation.
    package func spaceNeeded(_ f: Family, variant v: Variant = .standard, baseDiT: Bool) -> Int {
        let fm = FileManager.default
        let size = f.installedSize(v)
        var missing: [Int] = []
        if !presentComponents(f) { missing.append(size.components) }
        if !fm.fileExists(atPath: library.map(f.encoderMap(v))) { missing.append(size.encoder) }
        if baseDiT && !fm.fileExists(atPath: library.map(f.ditMap(v))) { missing.append(size.dit) }
        return missing.reduce(0, +) + (missing.max() ?? 0)
    }

    func download(_ x: File) throws {
        try cancellation.check()
        try Downloader.download(x.repository.name, revision: x.repository.revision, path: x.path, to: source(x),
                                     pinned: x.sha256, cancellation: cancellation, journal: journal)
    }

    /// **The space needed before downloading**: the sources (minus what is already there) and the
    /// forged map, no larger than its sources (never wider than published), plus its page alignment
    /// (16 KB per tensor: under 1 %, 2 % kept). Read from the Hub's headers. `alsoWriting`: another
    /// map being written meanwhile, by its sources' size (the encoder's, beside the DiT's download).
    func checkSpace(_ files: [File], alsoWriting: Int = 0) throws {
        var toDownload = 0, sources = 0
        for x in files where x.path.hasSuffix(".safetensors") || x.path.hasSuffix(".gguf") {
            let size = try Downloader.announcedFootprint(Downloader.url(x.repository.name, revision: x.repository.revision, path: x.path)).size ?? 0
            sources += size
            if Library.size(source(x)) != size { toDownload += size }
        }
        let other = alsoWriting + alsoWriting / 50
        try library.checkSpace(toDownload + sources + sources / 50 + other)
    }

    /// Downloads running on a thread of their own, beside a forge.
    final class Downloads: @unchecked Sendable {
        let stopping: Cancellation
        private let finished = DispatchSemaphore(value: 0)
        private var outcome: Result<Set<String>, Error> = .success([])
        init(stopping: Cancellation) { self.stopping = stopping }
        func run(_ body: @escaping @Sendable () throws -> Set<String>) {
            DispatchQueue.global(qos: .userInitiated).async { [self] in
                outcome = Result { try body() }
                finished.signal()
            }
        }
        /// Stops them (the forge beside them failed); `wait` still has to be called.
        func stop() { stopping.cancel() }
        /// The sources downloaded and verified, once all are; the first failure otherwise.
        func wait() throws -> Set<String> { finished.wait(); return try outcome.get() }
    }

    /// **Starts downloading `files` beside the forge about to run**, when the disk holds them with
    /// the map that forge writes (`alsoWriting`); `nil` — and nothing started — when it does not.
    /// A Stop reaches them through the installation's token; a failed forge, through their own.
    func startDownloads(_ files: [File], alsoWriting: Int) -> Downloads? {
        guard (try? checkSpace(files, alsoWriting: alsoWriting)) != nil else { return nil }
        let downloads = Downloads(stopping: Cancellation(following: cancellation))
        var copy = self
        copy.cancellation = downloads.stopping
        let beside = copy
        downloads.run {
            for x in files { try beside.download(x) }
            return Set(files.map(beside.source))
        }
        return downloads
    }

    func cleanDownloads() {
        let fm = FileManager.default
        // Only empty folders: a kept download (`keepSources`) stays.
        guard let e = fm.enumerator(atPath: downloads) else { return }
        var folders: [String] = []
        while let x = e.nextObject() as? String {
            var d: ObjCBool = false
            if fm.fileExists(atPath: downloads + "/" + x, isDirectory: &d), d.boolValue { folders.append(x) }
        }
        for d in folders.sorted(by: { $0.count > $1.count }) {
            let p = downloads + "/" + d
            if (try? fm.contentsOfDirectory(atPath: p))?.isEmpty == true { try? fm.removeItem(atPath: p) }
        }
        if (try? fm.contentsOfDirectory(atPath: downloads))?.isEmpty == true { try? fm.removeItem(atPath: downloads) }
    }

    var progressHandler: (Int, Int, String) -> Void {
        let j = journal
        return { k, n, _ in if k % max(1, n / 10) == 0 && k > 0 { j("  … \(k * 100 / n) %") } }
    }

    // MARK: - The reference: the names and shapes of the vendor's DiT

    /// `dit.json`: `{map name: published shape}` of the vendor's DiT, read from the remote
    /// **headers** (a few hundred KB, not a weight). An imported checkpoint must match it exactly —
    /// this is what proves it really belongs to this family.
    func writeReference(_ f: Family, to path: String) throws {
        let catalog = try RemoteCatalog(files: Self.dit(f).filter { $0.path.hasSuffix(".safetensors") })
        let n = try Recipes.normalize(catalog, family: f)
        let pairs = n.dit.keys.sorted().map { OrderedJSON.Pair($0, .list(n.dit[$0]!.shape.map { .integer($0) })) }
        try Data(OrderedJSON.object(pairs).jsonText.utf8).write(to: URL(fileURLWithPath: path), options: .atomic)
        journal("  reference DiT names: \(pairs.count) tensors")
    }

    package func readReference(_ f: Family) throws -> [String: [Int]] {
        let path = componentsFolder(f) + "/dit.json"
        guard FileManager.default.fileExists(atPath: path) else { throw MissingFile(path, .published) }
        var r: [String: [Int]] = [:]
        for p in try OrderedJSON.read(path).pairs ?? [] { r[p.key] = p.value.elements?.compactMap(\.integer) }
        return r
    }
}

extension Family {
    /// **The license files an installation of `variant` links**, at the revisions it downloads: the
    /// publisher's (`ModelCard.license.urls`), then — for a Compact or a Light — each third party's model card.
    public func licenseURLs(_ variant: Variant) -> [URL] { Installer.licenseURLs(self, variant) }
}

/// Safetensors headers read remotely (`Range` requests): names, shapes, dtypes — not a weight.
final class RemoteCatalog: TensorCatalog {
    private(set) var names: [String] = []
    private var entries: [String: (dtype: String, shape: [Int])] = [:]

    init(files: [Installer.File]) throws {
        for f in files {
            let url = Downloader.url(f.repository.name, revision: f.repository.revision, path: f.path)
            let length = try Downloader.range(url, 0, 7)
            // The same bounds as `Safetensors`: a length that is not an `Int`, zero, or beyond the
            // specification's 100 MB is refused before a byte of it is requested.
            let declared = length.withUnsafeBytes { $0.loadUnaligned(as: UInt64.self) }
            guard let n = Int(exactly: declared), n > 0, n <= Safetensors.maximumHeader else {
                throw Numerics.Failure(description: "\(f.path) : header length \(declared)")
            }
            let json = try Downloader.range(url, 8, 8 + n - 1)
            guard let object = try JSONSerialization.jsonObject(with: json) as? [String: Any] else {
                throw Numerics.Failure(description: "\(f.path) : unreadable header")
            }
            let sortedEntries = object.compactMap { (k, v) -> (String, Int, String, [Int])? in
                guard let e = v as? [String: Any], let o = e["data_offsets"] as? [Int], o.count == 2 else { return nil }
                return (k, o[0], e["dtype"] as? String ?? "", e["shape"] as? [Int] ?? [])
            }.sorted { $0.1 < $1.1 }
            for (k, _, d, s) in sortedEntries { names.append(k); entries[k] = (d, s) }
        }
    }

    func shape(_ name: String) -> [Int]? { entries[name]?.shape }
    func dtype(_ name: String) -> String? { entries[name]?.dtype }
    func read(_ name: String) throws -> [Float] { throw Numerics.Failure(description: "remote catalog: no values") }
}
