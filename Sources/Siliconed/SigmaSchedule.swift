import Foundation

/// **Whose σ a render walks**: the model's official recipe, or what ComfyUI walks for it — a
/// comparison, offered in Developer Mode on Qwen-Image-2.1 only. The steps, the solver and what the
/// DiT is told per σ do not change; only the σ list does.
///
/// ComfyUI reads no diffusers config: it recognizes a DiT by its tensors and applies one setting per
/// architecture (`supported_models.py`). For Z-Image and Anima that setting **is** the Turbo's recipe
/// (`shift 3` on `linspace(1, 1/N, N)`, the same σ to 8 steps): nothing to compare, so they offer
/// `official` alone. For Qwen-Image-2.1 it is the base model's, applied to the Turbo too:
///
///     official   model_index.json's sample_sigmas    1, .978, .954, .927, .895, .845, .705, .415, 0
///     comfyUI    flux_time_shift(μ = 0.69), simple   1, .933, .857, .769, .666, .545, .399, .222, 0
///
/// The Turbo spends 6 of its 8 steps above σ 0.84 and ends in two long jumps; ComfyUI's spreads them
/// and keeps three under 0.55. Checked to the bit against ComfyUI's own code
/// (ComfyUI's own code run once, `QwenImage21PipelineTests`).
package enum SigmaSchedule: String, CaseIterable, Sendable, Codable {
    case official
    case comfyUI = "comfyui"
}
