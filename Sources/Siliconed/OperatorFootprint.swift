import Foundation

/// **What is still outside our hand** — the few bytes of a render no constructor of ours sizes, each
/// named, with its reason and its bound.
///
/// Everything else of a render's memory is ours and exact by construction: arenas, `Pages`, the VAEs'
/// two heaps, each sized by the shape function its constructor calls (`MemoryNeed`). The per-operator laws
/// once fitted on this M1 Pro — `a + b·elements` per operator — are gone: every operator whose cost they
/// carried now runs on buffers we size, and its bytes are its shape function's
/// (`HeadSplitAttention.hostBytes`, `Conductor.workspaceBytes`, `Attention.maskBytes`, `VAE.plan`, …).
/// **No term here is fitted.** What remains:
///
///     sdpaQueries   MPSGraph's SDPA transposes the queries into [H, S, Dh] before its kernel: one fp32
///                   copy, rows·heads·headDim·4 — derived from the shape (measured: 4.03 B per element
///                   on six shapes: the copy, nothing else grows with the shape). The matrix
///                   H·S·K is never materialized on macOS 27 (measured: 41 MB where it would be 3.3 GB)
///     mpsGEMM       `MPSMatrixMultiplication`'s own state: constant, whatever the shape (measured: 3.8 MB on
///                   four shapes from 11.8 M to 64 M results). Bounded, not derivable: MPS's
///     sdpaExecutable  MPSGraph's compiled executable of one SDPA graph, alive with its graph: the
///                   constant of the measured per-operator law (7.4 MB on six shapes, whatever the
///                   shape), left uncharged until the product's check settled it — Qwen-Image-2.1 512²'s text phase peaked
///                   10 MB above a need without it. Bounded, not derivable: MPSGraph's
///     processBase   the process before any model — Metal device and queue, compiled kernels, a first
///                   SDPA and a first GEMM (measured: 12.0–13.1 MB over three CLI processes). Bounded, not
///                   derivable: the code and the frameworks' own
///
/// **Not charged, and why** (open, the product check reports it): what a model phase leaves to the
/// next — measured on Qwen-Image-2.1 512²: 32 MB at the decoder's start against the 14 charged: `malloc`'s
/// small regions kept dirty by MPSGraph's compilation churn (~12 MB of freed pages, 4.4 MB live;
/// `malloc_zone_pressure_relief` returns none of it) and ~11 MB of graphics Metal frees late. Not a
/// shape's, and it goes only with MPSGraph; what MPS keeps in its heap cache between two graphs of a
/// phase.
package enum OperatorFootprint {

    /// **The queries' fp32 copy MPSGraph's SDPA makes** (the transposition to heads-first), for one
    /// cached `Attention` or the GPU part of a `HeadSplitAttention`: `rows · heads · headDim · 4` bytes.
    package static func sdpaQueries(rows: Int, heads: Int, headDim: Int) -> Int { rows * heads * headDim * 4 }

    /// **MPSGraph's compiled SDPA**, per SDPA graph alive in a phase — a bound measured, not a shape's
    /// (measured: the 7.4 MB constant of the law, on six shapes).
    package static let sdpaExecutable = 7_400_000

    /// **`MPSMatrixMultiplication`'s state**, per `GEMM` alive in a phase — a bound measured, not a
    /// shape's (measured: 3.8 MB on four shapes). One per phase: one model's `GEMM` at a time.
    package static let mpsGEMM = 3_900_000

    /// **The process's footprint before any model**: the CLI's, 12.0–13.1 MB. ⚠ The app's is
    /// **not measured**: until it is, the app is charged the CLI's, which is an underestimate — the
    /// product's check (`MemoryCheck`) reports the difference on every render.
    package static func processBase(app: Bool) -> Int { 13_100_000 }
}
