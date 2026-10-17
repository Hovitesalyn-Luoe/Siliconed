import AppKit
import CoreImage
import Siliconed
import VideoToolbox

// **« Enlarge »: a history image ×4, by macOS's own super-resolution** — VideoToolbox's
// `VTSuperResolutionScaler` (macOS 26), the kind of network Pixelmator or Photoshop's Super Resolution
// use: it sees the image and nothing else. No prompt, no denoising, so nothing the prompt could paint
// into a corner of the picture; what it adds is local — edges, strands, grain —, never a new object.
//
//     Entry.png ─ CGImage (sRGB, 8 bits) ─ CIContext → CVPixelBuffer RGhA (half float, sRGB-encoded)
//       ─ VTFrameProcessor: super-resolution, image mode, ×4 ─ CVPixelBuffer (4w × 4h)
//       ─ CIContext → CGImage (sRGB, 8 bits) ─ PNG: the source's metadata + `enlarged`
//
// What a port by analogy would miss, read in the SDK's headers and measured on the M1 Pro (17/10):
//
//   - **the model is Apple's, not the library's**: macOS downloads it once, on request
//     (`downloadConfigurationModel`), and keeps it among its own assets — nothing in our folder,
//     nothing to verify against an oracle. It is a system tool, like ImageIO; the render it starts
//     from stays in the history, untouched;
//   - **×4 only** (`supportedScaleFactors` = [4] on the M1 Pro), and an input of **1920 at most per
//     side** in image mode: an enlarged image cannot be enlarged again;
//   - **one pixel format**, 64-bit RGBA half float, both ways; Core Image fills and reads it with
//     sRGB-encoded values, the encoding the network gets from a video frame;
//   - **the same bits twice**: two enlargements of one 1024² image, one md5. About 1 to 2.5 s for
//     1024² → 4096² once the model is warm; a 4096² PNG weighs some 20 MB. The colors stay: brought
//     back to 1024² (box filter), the mean shift is under 0.6/255 on every channel (two fixtures).
//
// It does not go through the queue: a render running, the button waits (`canEnlarge`) — two GPU
// jobs side by side would each slow the other, and the render's memory plan (B) counts none of this.

/// macOS's super-resolution, wrapped: availability, the model's state, its download, the enlargement.
enum SuperResolution {
    static let factor = 4
    /// The largest side the image mode takes (`VTSuperResolutionScalerConfiguration`'s header).
    static let maximumSide = 1920

    enum State: Equatable { case unavailable, downloadRequired, downloading, ready }

    /// Whether this Mac has it at all: macOS 26, and a ×4 its hardware offers.
    static var available: Bool {
        guard #available(macOS 26.0, *) else { return false }
        return VTSuperResolutionScalerConfiguration.isSupported
            && VTSuperResolutionScalerConfiguration.supportedScaleFactors.contains(factor)
    }

    static func state(width: Int, height: Int) -> State {
        guard available, #available(macOS 26.0, *), let c = configuration(width: width, height: height) else { return .unavailable }
        switch c.configurationModelStatus {
        case .ready: return .ready
        case .downloading: return .downloading
        default: return .downloadRequired
        }
    }

    /// Downloads Apple's model; `progress` receives 0…1 while it comes.
    static func download(width: Int, height: Int, progress: @escaping @MainActor (Double) -> Void) async throws {
        guard #available(macOS 26.0, *), let c = configuration(width: width, height: height) else { throw Failure.unavailable }
        let watch = Task { @MainActor in
            while !Task.isCancelled {
                progress(Double(c.configurationModelPercentageAvailable))
                try? await Task.sleep(for: .milliseconds(300))
            }
        }
        defer { watch.cancel() }
        try await c.downloadConfigurationModel()
    }

    /// `png` enlarged ×4, as an sRGB `CGImage` — off the main thread.
    static func enlarge(_ png: Data) async throws -> Decoded {
        try await Task.detached(priority: .userInitiated) {
            guard #available(macOS 26.0, *) else { throw Failure.unavailable }
            guard let source = CGImageSourceCreateWithData(png as CFData, nil),
                  let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else { throw Failure.unreadable }
            return Decoded(try run(image))
        }.value
    }

    enum Failure: Error { case unavailable, unreadable, buffer, failed(String) }

    @available(macOS 26.0, *)
    private static func configuration(width: Int, height: Int) -> VTSuperResolutionScalerConfiguration? {
        VTSuperResolutionScalerConfiguration(frameWidth: width, frameHeight: height, scaleFactor: factor,
                                             inputType: .image, usePrecomputedFlow: false,
                                             qualityPrioritization: .normal,
                                             revision: VTSuperResolutionScalerConfiguration.defaultRevision)
    }

    @available(macOS 26.0, *)
    private static func run(_ image: CGImage) throws -> CGImage {
        let w = image.width, h = image.height
        guard let configuration = configuration(width: w, height: h) else { throw Failure.unavailable }
        let srgb = CGColorSpace(name: CGColorSpace.sRGB)!
        let context = CIContext(options: [.workingColorSpace: srgb, .outputColorSpace: srgb])
        func buffer(_ attributes: [String: Any], _ width: Int, _ height: Int) throws -> CVPixelBuffer {
            var b: CVPixelBuffer?
            guard CVPixelBufferCreate(nil, width, height, kCVPixelFormatType_64RGBAHalf,
                                      attributes as CFDictionary, &b) == kCVReturnSuccess, let b else { throw Failure.buffer }
            return b
        }
        let input = try buffer(configuration.sourcePixelBufferAttributes, w, h)
        let output = try buffer(configuration.destinationPixelBufferAttributes, w * factor, h * factor)
        context.render(CIImage(cgImage: image), to: input, bounds: CGRect(x: 0, y: 0, width: w, height: h), colorSpace: srgb)

        let processor = VTFrameProcessor()
        do { try processor.startSession(configuration: configuration) }
        catch { throw Failure.failed(error.localizedDescription) }
        defer { processor.endSession() }
        guard let source = VTFrameProcessorFrame(buffer: input, presentationTimeStamp: .zero),
              let destination = VTFrameProcessorFrame(buffer: output, presentationTimeStamp: .zero),
              let parameters = VTSuperResolutionScalerParameters(
                sourceFrame: source, previousFrame: nil, previousOutputFrame: nil, opticalFlow: nil,
                submissionMode: .random, destinationFrame: destination) else { throw Failure.buffer }
        // The synchronous call is unavailable in Swift; one frame, so the completion is simply awaited.
        let semaphore = DispatchSemaphore(value: 0)
        nonisolated(unsafe) var failure: Error?
        processor.process(parameters: parameters) { _, error in
            failure = error
            semaphore.signal()
        }
        semaphore.wait()
        if let failure { throw Failure.failed(failure.localizedDescription) }

        let result = CIImage(cvPixelBuffer: output, options: [.colorSpace: srgb])
        guard let enlarged = context.createCGImage(result, from: result.extent, format: .RGBA8, colorSpace: srgb) else {
            throw Failure.buffer
        }
        return enlarged
    }
}

extension AppState {
    /// Why « Enlarge » cannot take this image now, in a sentence — `nil` when it can.
    func enlargeRefusal(_ entry: Entry) -> String? {
        if !SuperResolution.available { return String(localized: "Enlarging needs macOS 26 or later") }
        if entry.grid != nil || entry.sketch != nil { return String(localized: "A grid is not enlarged") }
        if entry.enlarged != nil || max(entry.width, entry.height) > SuperResolution.maximumSide {
            return String(localized: "This image is already enlarged")
        }
        if enlarging != nil { return String(localized: "An image is being enlarged") }
        if busy { return String(localized: "Enlarging waits for the queue to finish") }
        return nil
    }

    /// The button's sentence: what comes out, and how long it takes once learned this session.
    func enlargeHelp(_ entry: Entry) -> String {
        if let refusal = enlargeRefusal(entry) { return refusal }
        let f = SuperResolution.factor
        let size = "\(entry.width)×\(entry.height) → \(entry.width * f)×\(entry.height * f)"
        let megapixels = Double(entry.width * entry.height) / 1_000_000
        if let speed = enlargeSpeed {
            return String(localized: "Enlarge ×4 with macOS's super-resolution: \(size), about \(Figures.seconds(speed * megapixels)). The original stays.")
        }
        return String(localized: "Enlarge ×4 with macOS's super-resolution: \(size). The original stays.")
    }

    /// **« Enlarge »**: the model there, the image is enlarged at once; missing, an alert offers to
    /// download it — Apple's, kept by macOS —, then enlarges.
    func enlarge(_ entry: Entry) {
        guard enlargeRefusal(entry) == nil else { return }
        switch SuperResolution.state(width: entry.width, height: entry.height) {
        case .unavailable:
            flash(String(localized: "Enlarging needs macOS 26 or later"))
        case .ready:
            runEnlargement(entry)
        case .downloading:
            flash(String(localized: "macOS is downloading the enlargement model"))
        case .downloadRequired:
            let alert = NSAlert()
            alert.messageText = String(localized: "Enlarge with macOS's super-resolution?")
            alert.informativeText = String(localized: "Siliconed enlarges ×4 with a model made by Apple: it sees only the image — no prompt — and sharpens edges and textures without adding anything. macOS downloads it once from Apple and keeps it; it is not in your models folder.")
            alert.addButton(withTitle: String(localized: "Download and Enlarge"))
            alert.addButton(withTitle: String(localized: "Cancel"))
            guard alert.runModal() == .alertFirstButtonReturn else { return }
            enlarging = entry.id
            enlargeDownload = 0
            Task {
                do {
                    try await SuperResolution.download(width: entry.width, height: entry.height) { p in
                        self.enlargeDownload = p
                    }
                    enlargeDownload = nil
                    enlarging = nil
                    runEnlargement(entry)
                } catch {
                    enlargeDownload = nil
                    enlarging = nil
                    flash(String(localized: "The enlargement model could not be downloaded"))
                }
            }
        }
    }

    private func runEnlargement(_ entry: Entry) {
        enlarging = entry.id
        let start = Date()
        Task {
            defer { enlarging = nil }
            do {
                guard let image = try await SuperResolution.enlarge(entry.png).image else { throw SuperResolution.Failure.buffer }
                let seconds = Date().timeIntervalSince(start)
                var metadata = entry.metadata
                metadata["enlarged"] = "\(SuperResolution.factor)x macOS super-resolution, from \(entry.width)x\(entry.height)"
                let finished = try await Task.detached(priority: .userInitiated) { [metadata] in
                    let png = try PNG.data(image, metadata: metadata)
                    return (png, Decoded(FinishedImage.reduce(image, side: 240) ?? image))
                }.value
                enlargeSpeed = seconds / (Double(entry.width * entry.height) / 1_000_000)
                var timings = Engine.Timings()
                timings.total = seconds
                lastImage += 1
                let enlarged = Entry(ordinal: lastImage, job: entry.job, png: finished.0, vignette: finished.1.image ?? image,
                                     width: image.width, height: image.height, seed: entry.seed, timings: timings,
                                     footprints: Engine.Footprints(), evaluations: 0, settings: entry.settings,
                                     metadata: metadata, enlarged: SuperResolution.factor)
                history.insert(enlarged, at: 0)
                selectedImage = (enlarged.id, image)
                selection = enlarged.id
                marked = []; anchor = enlarged.id
                sessionSeconds += seconds
            } catch {
                flash(String(localized: "This image could not be enlarged"))
            }
        }
    }
}
