English · [Français](VERIFY.fr.md)

# Verifying Siliconed

Siliconed claims that each stage of each model it ships computes what the reference implementation computes. **This page shows how to check that claim yourself**, on your Mac, without trusting our figures.

The reference is the publisher's own: [diffusers](https://github.com/huggingface/diffusers) and [transformers](https://github.com/huggingface/transformers), in float32 on the CPU, with the published weights. Python scripts run it and freeze its answers into *goldens*; `siliconed check` runs the engine on the same inputs and judges it against them, stage by stage.

## Five minutes, no weights

The numerical kernels (RoPE tables, resampling, the GPU's element-wise operations, attention, the split of a matrix product between GPU and AMX) are checked without any model.

```sh
swift build -c release --product siliconed
python3.12 -m venv oracle-env && oracle-env/bin/pip install -r tools/oracle/requirements.txt
oracle-env/bin/python tools/oracle/kernels.py all          # the RoPE and resampling goldens, seconds
.build/release/siliconed check kernels
```

Each stage prints its measures, then one verdict line, `✓`, `✗` or `—` (skipped, and why). The exit code says the same: 0 passed, 1 failed, 2 skipped, 64 a usage error.

## One model stage

Install a model in the app first (Z-Image is the smallest download): `siliconed check` reads the weights the app installed, never the ones the oracle downloads. Then, for the text encoder:

```sh
oracle-env/bin/python tools/oracle/z-image.py text         # downloads the reference weights once
.build/release/siliconed check z-image text
```

`siliconed check list` lists every stage, what it proves, how it is judged and which command writes its goldens; the table below is its output. Goldens go to `./goldens` unless `--goldens DIR` or `SILICONED_GOLDENS` says otherwise, for the scripts and the checks alike. To write every golden of a model at once, `oracle-env/bin/python tools/oracle/z-image.py all` (likewise `qwen-image-2.1.py`, `anima.py`, `krea2.py`); what it costs is below.

## Are your goldens ours?

We do not publish the goldens (several gigabytes, and weights whose licences forbid it). We publish their fingerprints: shape, type and statistics of every tensor.

```sh
oracle-env/bin/python tools/oracle/fingerprints.py compare
```

Integers (token identifiers, masks) must match exactly. Floats are compared within a tolerance (1e-3 on norms), not bit for bit: PyTorch on the CPU does not promise the same float32 bits from one machine to another, and two honest runs of the reference differ by rounding. That rounding is far below what the checks judge.

## The rule

A stage's figure is the **relative error per channel**: for each channel (the last axis of a tensor, or each plane of a latent), the L2 norm of the difference over the L2 norm of the reference. The worst channel is reported, never an average: a fault confined to one channel would vanish in a global figure. The same rule judges every boundary of every model:

    without a witness:   error(engine, reference)  ≤ 5e-4
    with a witness:      error(engine, witness)    ≤ max(floor, 1.25 · error(reference, witness))

The *witness* is the same computation in float64. Where it exists, it decides: float32 has its own rounding error, and on an ill-conditioned boundary the reference itself can be 4e-3 away from the exact value; the engine may then be no further from the exact value than the reference is, plus 25 %. Each stage's line in `check list` gives its own rule: an identity (tokens, schedules) is judged bit for bit, a few stages have a tighter floor, and a rectangle is checked against its transpose, which must fail, so that a swapped height and width cannot pass.

Krea 2's denoiser has a few boundaries so ill-conditioned that a change of one unit in the last place of the input moves the output by more than the rule allows, the reference included. There, a measure that fails at the witness is judged by *clouds*: the reference and the engine are both run on inputs moved by one ulp, and the engine's median error must stay within 1.25 times the reference's worst. These are the stages whose rule says "clouds when the oracle has draws".

Two Krea 2 measures pass with little to spare on our M1 Pro: the velocity `v_2` of the 512² trajectory (5.08e-4 for an allowance of 5.14e-4) and `block0_attn_out` of the 1024² denoiser (3.53e-3 for 3.59e-3). The allowance is computed from your reference's own error, so it moves with your CPU's rounding, and these two may fail by a hair on another Mac. The rule is not loosened for them; if they fail on yours, an issue with the output of `siliconed check` helps us find which layer carries the gap.

What is **not** claimed: the same bits as the reference, or as another Mac. The engine computes in another order on another device; it is judged on how close it comes to the exact value, not on its bits. LoRAs are not covered here: they are files you choose, not part of what Siliconed ships.

## What it costs

On a MacBook Pro M1 Pro, 16 GB. Download: the reference weights the scripts fetch, beside the app's own. Oracle: the scripts' `all`, wall time, then its heaviest stage with its peak memory. Check: `siliconed check` over every stage of the model, wall time.

| Model | Download | Oracle, every stage | Its heaviest stage | Check |
|---|---|---|---|---|
| kernels | — | under 5 s | — | a few seconds |
| Z-Image | 31 GB | 39 min | the 1024² trajectory: 14 min, 13.3 GB | 7 min |
| Qwen-Image-2.1 | 30 GB | 27 min | the fp64 decoder witness: 43 s, 15.5 GB | 10 min |
| Anima | 5 GB | 7 min | the 1024² decoder: 1 min, 20.8 GB | 3 min |
| Krea 2 | 33 GB | 1 h 45 min | the eight fp64 step witnesses at 1024²: 44 min, 11 GB | 16 min |

The oracle streams the denoiser block by block in float32 (Krea 2's is 48 GB at that precision), so a 16 GB Mac runs every script; a stage whose peak passes the machine's memory (the 1024² decoders) runs in swap and takes longer.

## The stages

The output of `siliconed check list --markdown`. `{size}` is 512, 1024 or a rectangle such as 768x512; the stages that take no size are checked at the reference's own.

| Stage | Proves | Rule | Oracle | Weights |
|---|---|---|---|---|
| `kernels widen` | bf16 → fp32 widening and its inverse are exact, on the values a naive shift gets wrong | bit for bit | — | — |
| `kernels arena` | the arena hands out distinct page-aligned slices, refuses duplicates and growth | each property holds | — | — |
| `kernels elementwise` | the GPU's SwiGLU, gate and gated residual are as close to double precision as the CPU's; the residual rounds twice, as torch | GPU worst ulp ≤ max(CPU's, 1), mean ≤ CPU's + 5 %; residual bit for bit | — | — |
| `kernels gelu` | the GPU's GELU (erf) is as close to double precision as the CPU's, over one bit pattern in 64 of [−12, 12] | GPU worst deviation ≤ the CPU's, in ulp of the operand | — | — |
| `kernels sdpa` | the attention cut by heads (GPU + CPU), by query blocks, past maxMatrix and the matrix kernel return the whole graph's result | against the whole graph: query blocks ≤ 1e-6 per channel; head cut ≤ 1e-5 with the GPU heads bit for bit; matrix kernel ≤ 1e-5 | — | — |
| `kernels gemm-split` | the token-wise split of a GEMM between the GPU and the AMX stitches back cblas_sgemm's product | against cblas_sgemm, per row: the GPU's rows bit for bit, the AMX's ≤ 1e-5 (another summation order over k = 3 840) | — | — |
| `kernels rope` | every RoPE table of the engine is the reference's, value by value (13 tables, 5 bit for bit) | max \|Δ\| per table within its bound, in units of 2⁻²⁴: 0 where the reference is bit-reproducible, 1 to 64 where torch uses SLEEF's cos, sin, pow | `tools/oracle/kernels.py rope` | — |
| `kernels resample` | the spectral schedule's nearest-exact decimation and bilinear upsampling are torch's, on a square and a rectangle | decimation bit for bit (and the even decimation, its counter-test, must differ); bilinear within one ulp of the input's scale | `tools/oracle/kernels.py resample` | — |
| `z-image tokenizer` | the chat template and the tokenizer give transformers' 512 identifiers and mask, padding included | bit for bit | `tools/oracle/z-image.py text` | tokenizer |
| `z-image text` | the Qwen3-4B encoder gives the reference's hidden states, layer by layer, and its cap_feats | the rule, floor 2e-4 | `tools/oracle/z-image.py text` | text encoder |
| `z-image block` | block 0 matches the reference at its 19 internal boundaries: norms, projections, attention, SwiGLU | the rule, floor 2e-4 | `tools/oracle/z-image.py dit --size {size}` | DiT |
| `z-image dit` | one evaluation of the DiT, latent to velocity: embedders, refiners, 30 blocks, head, unpatchify | the rule, floor 5e-4; on a rectangle, the transposed golden must deviate | `tools/oracle/z-image.py dit --size {size}; witness --size {size}` | DiT |
| `z-image trajectory` | the 8-step sampler: each step's velocity and latent, from the reference's own latent at that step | the rule on the guided pass, floor 5e-4; the free pass is shown with its gain, not judged | `tools/oracle/z-image.py trajectory --size {size}` | DiT |
| `z-image vae` | Flux's decoder on a rectangular latent (the centered crop of the final 1024² latent): what a square cannot see | PSNR ≥ 80 dB against the reference's image, and the transposed image must deviate | `tools/oracle/z-image.py vae --size {size}` | VAE |
| `qwen-image-2.1 text` | Qwen3-VL-8B reads the prompt (and with --edit N, the N pictures): template, identifiers, positions, 36 layers, prompt_embeds | identifiers and masks bit for bit; the rule, floor 5e-4 (with images, the deep layers judged on their text rows) | `tools/oracle/qwen-image-2.1.py text [--edit N]` | text encoder |
| `qwen-image-2.1 vision` | the processor (Lanczos, patches, grids), the 3D positions of both edit templates, and the vision tower with its deepstack, image by image | pixels, grids, identifiers and positions bit for bit; the rule, floor 5e-4 | `tools/oracle/qwen-image-2.1.py vision` | text encoder |
| `qwen-image-2.1 references` | a reference with transparency, prepared as the pipeline does: the VAE's RGBA input and the vision copy over white (at 512, Pillow premultiplies around its Lanczos; at 443 it copies) | bit for bit; the vision copy on its bytes (the Pil and Fast processors differ by 6e-8 in float) | `tools/oracle/qwen-image-2.1.py references [--side 443]` | — |
| `qwen-image-2.1 dit` | one evaluation three ways — prefill of the conditions, the step from their K/V cache, the full pass under the block-causal mask — plus the layout and the modulation | layout bit for bit; the rule, floor 5e-4, at the fp64 witness where one exists; cache and full pass ≤ 1e-5 apart; a second step the same bits | `tools/oracle/qwen-image-2.1.py dit [--edit N]; witness` | DiT |
| `qwen-image-2.1 trajectory` | the 8 steps: σ, timesteps and noise to the bit, then each step's velocity from the oracle's latent and encoder states; the product's free run from the real inputs is shown | schedule and noise bit for bit; the rule on each guided velocity, floor 5e-4; the free run's final latent and image are shown, not judged | `tools/oracle/qwen-image-2.1.py trajectory [--edit N]` | DiT, text encoder, VAE |
| `qwen-image-2.1 vae` | the decoder on the trajectory's final latent (with --edit N, the edit's): unpacking and de-normalization to the bit, then the RGBA image plane by plane | vae_in bit for bit; the rule per plane, floor 5e-4, at the fp64 witness where one exists; on a rectangle, the transposed image must deviate | `tools/oracle/qwen-image-2.1.py vae [--edit N]; vae-witness` | VAE |
| `qwen-image-2.1 vae-encode` | the encoder on both edit pictures (RGBA): the raw mean and the normalized latent, plane by plane; the product's module (RGB in, alpha added) gives the same bits | the rule per plane, floor 5e-4, at the fp64 witness where one exists; the module bit for bit | `tools/oracle/qwen-image-2.1.py vae-encode; vae-witness` | VAE |
| `anima tokenizer` | the T5 tokenizer against `tokenizers` on a corpus of edge cases: normalization, whitespace, special tokens, truncation | every case bit for bit | `tools/oracle/anima.py tokenizer` | tokenizer |
| `anima text` | the prompt through both tokenizers, Qwen3-0.6B (final norm included) and the conditioner, to the DiT's [512, 1024] context | identifiers bit for bit; the rule, floor 5e-4 | `tools/oracle/anima.py text` | text encoder, DiT (the conditioner) |
| `anima dit` | one evaluation of the DiT at σ = 1, boundary by boundary, and its output per latent plane against the fp64 witness | the rule, floor 5e-4; model_out per latent plane at the fp64 witness; on a rectangle, the transposed golden must deviate | `tools/oracle/anima.py dit --size {size}; witness --size {size} --step 0` | DiT |
| `anima trajectory` | the schedule, then the 8 Euler steps: each step's velocity from the oracle's latent at that step | σ within 1e-6; the rule on each guided velocity, floor 5e-4; the free pass is shown with its gain, not judged | `tools/oracle/anima.py trajectory --size {size}` | DiT |
| `anima vae` | the Qwen-Image decoder on the final latent of the oracle's trajectory, plane by plane | the rule per plane, floor 5e-4, at the fp64 witness (no floor there); on a rectangle, the transposed image must deviate | `tools/oracle/anima.py vae --size {size}; vae-witness --size {size}` | VAE |
| `krea2 text` | the prompt's identifiers, the 12 Qwen3-VL taps (padding removed, judged against states computed with it), then the fusion and txt_in on the oracle's taps | identifiers bit for bit; the rule, floor 5e-4 | `tools/oracle/krea2.py text` | text encoder, DiT (the fusion) |
| `krea2 dit` | one evaluation of the DiT at σ = 1, boundary by boundary, judged at the fp64 witness where it exists | the rule, floor 5e-4, at the fp64 witness where it exists, clouds when the oracle has draws; model_out per latent plane at the witness; on a rectangle, the transposed golden must deviate | `tools/oracle/krea2.py dit --size {size}; witness --size {size}; witness-step --size {size} --step 0; draw --size {size} --step 0 --jitter 1 2 3` | DiT |
| `krea2 trajectory` | the schedule, then the 8 Euler steps: each step's velocity from the oracle's latent at that step, at the step's fp64 witness where it exists | σ within 1e-6; the rule on each guided velocity, floor 5e-4, at the step's fp64 witness where it exists, clouds when the oracle has draws; the free pass is shown, not judged | `tools/oracle/krea2.py trajectory --size {size}; witness-step --size {size} --step 0 1 2 3 4 5 6 7` | DiT |
| `krea2 vae` | the Qwen-Image decoder with Krea's weights on the final latent of the oracle's trajectory, plane by plane | the rule per plane, floor 5e-4, at the fp64 witness (no floor there); on a rectangle, the transposed image must deviate | `tools/oracle/krea2.py vae --size {size}; vae-witness --size {size}` | VAE |
