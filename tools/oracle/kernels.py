#!/usr/bin/env python3
"""Goldens for the engine's numerical kernels, which need no weights and take seconds.

    kernels.py rope        every RoPE table of the engine           -> goldens-rope
    kernels.py resample    the spectral schedule's two operators    -> goldens-resample, goldens-resample-rect
    kernels.py all

The folder is `--goldens DIR`, else `SILICONED_GOLDENS`, else `./goldens`.
"""
import argparse
import sys
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

sys.dont_write_bytecode = True      # the repository carries no caches
import common  # noqa: E402


# ======================================================================================== rope

def rope(goldens: Path) -> None:
    """Every RoPE table of the engine, as each reference computes it.

    The tables are computed once per render, so they cost nothing, but they carry a lot: an angle
    near 1000 rad rounded once in float32 is 3e-5 rad off, in double 1e-13. Each reference has its own
    recipe, and a port by analogy takes one for another:

        zimage-dit       RopeEmbedder.precompute_freqs_cis   fp64 freqs and outer, `.float()`, torch.polar
        qwen3-text       Qwen3RotaryEmbedding                 fp32 inv_freq = 1/theta^(2j/d), fp32 product
        qwen3vl-text     Qwen3VLTextRotaryEmbedding           the same, three axes, interleaved recomposition
        qwen3vl-vision   Qwen3VLVisionRotaryEmbedding         fp32, theta^(2j/(d/2)), rows then columns
        qwen21-dit       QwenImage21Rope                      fp32 `1/torch.pow(theta, j/d)`, fp32 outer, torch.polar
        anima-dit        CosmosRotaryPosEmbed                 NTK theta (a Python double cast to fp32), fp32 outer
        anima-adapter    AnimaRotaryEmbedding                 fp32 inv_freq, fp32 matmul (K = 1)
        krea2-dit        Krea2RotaryPosEmbed                  fp64 freqs and outer, cos/sin in fp64, `.float()`

    Each table is written as `<name>.cos` and `<name>.sin`, `[rows, pairs]` float32, column j being
    the reference's frequency j before any duplication (`cat(f, f)`, `repeat_interleave`, the complex
    number's real and imaginary parts). The positions the rows stand for go in `<name>.positions`
    (int32, `[rows, axes]`). The parameters are the published configurations'; the engine reads its
    own from its maps' headers and the check says where they differ."""
    from diffusers.models.condition_embedders.condition_embedder_anima import AnimaRotaryEmbedding
    from diffusers.models.transformers.transformer_cosmos import CosmosRotaryPosEmbed
    from diffusers.models.transformers.transformer_krea2 import Krea2RotaryPosEmbed
    from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21Rope
    from diffusers.models.transformers.transformer_z_image import RopeEmbedder
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig, Qwen3VLVisionConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (Qwen3VLTextRotaryEmbedding,
                                                                Qwen3VLVisionRotaryEmbedding)
    out, meta = {}, {}

    def put(name, cos, sin, positions, **params):
        out[name + ".cos"] = cos.contiguous().float()
        out[name + ".sin"] = sin.contiguous().float()
        out[name + ".positions"] = torch.as_tensor(positions, dtype=torch.int32).reshape(cos.shape[0], -1).clone().contiguous()
        meta[name] = json.dumps(params)

    # Z-Image's DiT: three axes, each its own table indexed by position.
    dims, lens, theta = [32, 48, 48], [1536, 512, 512], 256.0
    for axis, table in enumerate(RopeEmbedder.precompute_freqs_cis(dims, lens, theta)):
        put(f"zimage-dit.{axis}", table.real, table.imag, torch.arange(lens[axis]), dims=dims[axis], theta=theta)

    # Qwen3 (Z-Image's 4B encoder, Anima's 0.6B): one axis, the token index.
    for tag, hidden, heads in (("4b", 2560, 32), ("0.6b", 1024, 16)):
        config = Qwen3Config(hidden_size=hidden, num_attention_heads=heads, head_dim=128,
                             rope_parameters={"rope_type": "default", "rope_theta": 1_000_000.0})
        cos, sin = Qwen3RotaryEmbedding(config)(torch.zeros(1, dtype=torch.float32), torch.arange(1024)[None])
        put(f"qwen3-text-{tag}", cos[0, :, :64], sin[0, :, :64], torch.arange(1024), dims=128, theta=1_000_000.0)

    # Qwen3-VL's text (Qwen-Image-2.1's 8B, Krea 2's 4B: same theta and sections): three axes. A prompt
    # as `get_rope_index` lays it out: 12 text tokens, a 24x30 image block (t frozen, h and w counting from
    # the block's start), then text from start + max(h, w).
    t, h, w = list(range(12)), list(range(12)), list(range(12))
    for r in range(24):
        for c in range(30):
            t.append(12); h.append(12 + r); w.append(12 + c)
    t += list(range(42, 342)); h += list(range(42, 342)); w += list(range(42, 342))
    positions = torch.tensor([t, h, w])
    config = Qwen3VLTextConfig(head_dim=128, hidden_size=4096, num_attention_heads=32,
                               rope_parameters={"rope_type": "default", "rope_theta": 5_000_000.0,
                                                "mrope_section": [24, 20, 20], "mrope_interleaved": True})
    cos, sin = Qwen3VLTextRotaryEmbedding(config)(torch.zeros(1, dtype=torch.float32), positions[:, None, :])
    put("qwen3vl-text", cos[0, :, :64], sin[0, :, :64], positions.T, dims=128, theta=5_000_000.0, sections=[24, 20, 20])

    # Qwen3-VL's vision tower (hidden 1152, 16 heads, head 72): rows, then columns.
    pos = torch.tensor([[r, c] for r in range(40) for c in range(56)])
    vision = Qwen3VLVisionConfig(hidden_size=1152, num_heads=16)
    vision.rope_parameters = {"rope_type": "axial", "rope_theta": 10_000.0}
    cos, sin = Qwen3VLVisionRotaryEmbedding(vision)(torch.zeros(1, dtype=torch.float32), pos)
    put("qwen3vl-vision", cos[:, :36], sin[:, :36], pos, dims=72, theta=10_000.0)

    # Qwen-Image-2.1's DiT: axes [16, 56, 56], positive and negative indices.
    qwen = QwenImage21Rope(theta=10000, axes_dim=[16, 56, 56])
    index = torch.arange(-300, 1200)
    for axis, table in enumerate(qwen.freqs):
        picked = table[index]          # negative indices read the table's tail, as `forward` does
        put(f"qwen21-dit.{axis}", picked.real, picked.imag, index, dims=[16, 56, 56][axis], theta=10000)

    # Anima's DiT: CosmosRotaryPosEmbed, head 128, NTK scales [1, 4, 4], one frame of 64x48 patches.
    cosmos = CosmosRotaryPosEmbed(hidden_size=128, max_size=(128, 240, 240), patch_size=(1, 2, 2),
                                  rope_scale=(1.0, 4.0, 4.0))
    cos, sin = cosmos(torch.zeros(1, 16, 1, 128, 96))
    grid = torch.tensor([[0, r, c] for r in range(64) for c in range(48)])
    put("anima-dit", cos[:, :64], sin[:, :64], grid, dims=128, scale=[1.0, 4.0, 4.0])

    # Anima's adapter: head 64, theta 10 000, the token index.
    cos, sin = AnimaRotaryEmbedding(head_dim=64, rope_theta=10000.0)(torch.zeros(1, dtype=torch.float32),
                                                                      torch.arange(512)[None])
    put("anima-adapter", cos[0, :, :32], sin[0, :, :32], torch.arange(512), dims=64, theta=10000.0)

    # Krea 2's DiT: Flux's recipe, theta 1000, axes [32, 48, 48], repeat_interleave; text at (0, 0, 0).
    ids = torch.tensor([[0, 0, 0]] * 4 + [[0, r, c] for r in range(64) for c in range(64)])
    cos, sin = Krea2RotaryPosEmbed(theta=1000, axes_dim=[32, 48, 48])(ids)
    put("krea2-dit", cos[:, ::2], sin[:, ::2], ids, axes=[32, 48, 48], theta=1000.0, text=4)

    path = goldens / "goldens-rope.safetensors"
    save_file(out, str(path), metadata=meta)
    print(f"-> {path}: {len(out) // 3} tables")
    for name in meta:
        print(f"  {name}: {tuple(out[name + '.cos'].shape)}  {meta[name]}")


# ==================================================================================== resample

def resample(goldens: Path) -> None:
    """The spectral schedule's two operators, frozen without the model.

    Neither can be guessed. Does `nearest-exact` at factor 2 keep the EVEN or the ODD indices? It
    differs from `nearest` by half a pixel, hence by one sample in two; the wrong one gives a latent
    just as white and an image slightly elsewhere, which no norm reveals. And `bilinear` with
    `align_corners=False` has an edge rule (a negative source coordinate clamps to zero) the
    documentation does not spell out.

    The test set is pseudo-random, never periodic: on a ramp or a checkerboard even and odd decimation
    return equal or mirrored outputs and the check could not fail. A second set is a RECTANGLE
    (128x192 to 64x96, height by width) with its own seed, since a square does not reveal an axis swap."""
    for rect in (False, True):
        channels, full, low = 16, 128, 64
        full_w, low_w = (192, 96) if rect else (full, low)
        suffix = "-rect" if rect else ""
        torch.manual_seed(20260923 if rect else 20260921)
        state = torch.randn(1, channels, full, full_w, dtype=torch.float32)
        velocity = torch.randn(1, channels, low, low_w, dtype=torch.float32)
        down = F.interpolate(state, size=(low, low_w), mode="nearest-exact")
        up = F.interpolate(velocity, size=(full, full_w), mode="bilinear", align_corners=False)

        even = bool(torch.equal(down, state[:, :, ::2, ::2]))
        odd = bool(torch.equal(down, state[:, :, 1::2, 1::2]))
        print(f"nearest-exact x1/2{suffix}: even decimation {even}, odd decimation {odd}")
        tensors = {"state": state, "down": down, "velocity": velocity, "up": up}
        common.write_golden(goldens, f"goldens-resample{suffix}", tensors, {
            "channels": channels, "full": full, "low": low, "full_w": full_w, "low_w": low_w,
            "down_mode": "nearest-exact", "up_mode": "bilinear/align_corners=False",
            "down_is_even_decimation": even, "down_is_odd_decimation": odd})


def main() -> None:
    common.unbuffered()
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("stage", choices=["rope", "resample", "all"])
    p.add_argument("--goldens", help="output folder (default: $SILICONED_GOLDENS, else ./goldens)")
    args = p.parse_args()
    goldens = common.goldens_directory(args.goldens)
    if args.stage in ("rope", "all"):
        rope(goldens)
    if args.stage in ("resample", "all"):
        resample(goldens)


if __name__ == "__main__":
    main()
