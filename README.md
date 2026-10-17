English · [Français](README.fr.md)

# Siliconed

*Siliconed — enhanced by silicone.*

**Image generation for Apple silicon that does the full computation, proves it, and never takes more memory than its budget.**

Siliconed is a free, open-source macOS app for **local text-to-image generation and AI image
editing** on Apple silicon Macs (M1 and later). It runs recent diffusion models —
[Z-Image Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo),
[Qwen-Image-2.1 Turbo](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo),
[Anima Turbo](https://huggingface.co/circlestone-labs/Anima) and
[Krea 2 Turbo](https://huggingface.co/krea/Krea-2-Turbo) — natively in Swift and Metal: no
Python, no server, no cloud, nothing leaves your Mac. It works offline once a model is installed.

![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue)
![macOS 15+](https://img.shields.io/badge/macOS-15%2B-black)
![Apple silicon](https://img.shields.io/badge/Apple%20silicon-M1%20and%20later-black)

![A summer photo made with Z-Image Turbo — a red-haired Irish woman smiling on a beach in the west of Ireland, her surfboard under her arm, in a shorty wetsuit — and its cel-shaded comic-book edit by Qwen-Image-2.1, under a sliding curtain](docs/images/curtain.gif)

*A photo from Z-Image Turbo, edited by Qwen-Image-2.1 with one sentence (“Convert this photo to a western
comic book style with cel shading”), compared under the curtain.*

## Why Siliconed

- **Checked against the original models.** Every stage (text encoder, denoiser, VAE) is compared,
  channel by channel, with the reference implementation (diffusers / transformers in fp32 on CPU).
  Deviation at 1024²: 1.2·10⁻⁵ for Z-Image. What cannot be checked against the reference does not
  ship.
- **Full fp32, still fast.** Z-Image Turbo at 1024² in 113–119 s on an M1 Pro, with the 8
  evaluations its recipe calls for ([measured next to Draw Things](#next-to-draw-things), the
  reference on the Mac). Until 0.8.1 it ran only 7 of them, and the times published then (82–85 s)
  were for 7; fixed in 0.8.2.
- **34 GB of model on a 16 GB Mac, within a fixed budget.** Weights are read from disk as they are
  needed. Siliconed never takes more than ¾ of the Mac's memory minus 2 GB (10 GB on a 16 GB Mac):
  each render computes what it needs before it starts, takes the fastest plan that fits, and checks
  its own peak against that figure. What the other applications hold is theirs.
- **Built to be driven by an AI.** `silicontrol` drives the running app from a terminal, with JSON
  output; `silicontrol help` is all an agent needs to read.
- **Your models, your way.** Import LoRAs and fine-tunes from Civitai or Hugging Face, install
  full-size, in 8 bits or in 4 to 6 bits, and see each license and the disk space before the first
  byte is downloaded.
- **Explore before you commit.** A grid of quick sketches across seeds, steps, LoRA strengths,
  models or prompt variants; the cell you pick becomes the finished image.
- **Reproducible to the bit.** Same prompt, seed, settings and version, same image, byte for byte:
  the edit above, redone in another session, came out identical. A version may change the pixels:
  0.8.2 did for Z-Image (its eighth evaluation), so its images from 0.8.1 and earlier do not come
  out again to the bit.

## Models

Only recent models whose text encoder is a language model — no Stable Diffusion, SDXL or Flux.1.

| Model | Publisher | What it does | License |
|---|---|---|---|
| [Z-Image Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) | Tongyi-MAI | text-to-image, image-to-image, LoRA | Apache 2.0 |
| [Qwen-Image-2.1 Turbo](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo) | Qwen | text-to-image, editing by instruction with 1 to 3 images, LoRA | Qwen Research (non-commercial) |
| [Anima Turbo v1.1](https://huggingface.co/circlestone-labs/Anima) | CircleStone Labs × Comfy Org | text-to-image (illustration, painting, anime; no photorealism, by its publisher's choice), image-to-image, LoRA | CircleStone Labs Non-Commercial; its images may be used commercially |
| [Krea 2 Turbo](https://huggingface.co/krea/Krea-2-Turbo) | Krea | text-to-image (aesthetic images, from photography to painting), LoRA | Krea 2 Community: commercial use under $1M annual revenue, a content filter required when deployed |

The Compact versions use 8-bit weights published by third parties:
[unsloth/Z-Image-Turbo-GGUF](https://huggingface.co/unsloth/Z-Image-Turbo-GGUF),
[Disty0/Z-Image-Turbo-SDNQ-int8](https://huggingface.co/Disty0/Z-Image-Turbo-SDNQ-int8),
[unsloth/Qwen-Image-2.1-FP8](https://huggingface.co/unsloth/Qwen-Image-2.1-FP8),
[Comfy-Org/Qwen-Image-2.1](https://huggingface.co/Comfy-Org/Qwen-Image-2.1); the Light versions, the
GGUF Q4_K_M of [unsloth/Z-Image-Turbo-GGUF](https://huggingface.co/unsloth/Z-Image-Turbo-GGUF) and
[unsloth/Qwen-Image-2.1-Turbo-GGUF](https://huggingface.co/unsloth/Qwen-Image-2.1-Turbo-GGUF) with the Compact's
text encoder. Anima's Compact and Light take the DiT in GGUF Q8_0 and Q4_K_M from
[vanes430/Anima-Turbo-V1.1-GGUF](https://huggingface.co/vanes430/Anima-Turbo-V1.1-GGUF) and keep
the publisher's text encoder in bf16 (no 8-bit version of it is published).

Krea 2 Turbo downloads nothing from [krea/Krea-2-Turbo](https://huggingface.co/krea/Krea-2-Turbo):
that repository asks for a Hugging Face account, and the app has no sign-in. Each of its files
comes from an open copy proven identical to Krea's: the DiT and the Qwen3-VL-4B text encoder in bf16
from [Comfy-Org/Krea-2](https://huggingface.co/Comfy-Org/Krea-2) (the DiT the very same file, the
encoder equal tensor for tensor), the VAE from [Qwen/Qwen-Image](https://huggingface.co/Qwen/Qwen-Image)
(Krea's fp32 VAE is this bf16 one widened, value for value), the tokenizer from
[Qwen/Qwen-Image-2.1-Turbo](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo) (the same file); Krea's
own configuration files are embedded as published. Its Compact takes the DiT and the text encoder in
int8 convrot from [Comfy-Org/Krea-2](https://huggingface.co/Comfy-Org/Krea-2); its Light, the GGUF
Q4_K_M of [vantagewithai/Krea-2-Turbo-GGUF](https://huggingface.co/vantagewithai/Krea-2-Turbo-GGUF)
with the Compact's text encoder. Every file is pinned to a revision and checked by sha256
(`Sources/Siliconed/Forge/Installation.swift`).

![The window with an Anima Turbo image: in the rack, Anima Turbo with the inCase LoRA at 0.80; in the canvas, a painted illustration of a young red-haired Irish woman reading a book on a wooden chair by a tall window in an old library, autumn light through the panes, shelves of books beside her](docs/images/anima.jpg)

*Anima Turbo with the inCase LoRA at 0.8: “An oil painting of a young Irish woman with long red hair and
freckles, sitting by a tall window in an old library and reading a book. Warm afternoon light falls
across the wooden shelves; visible brushstrokes, muted earthy colors, painterly realism.” 1024², seed
44, 8 steps, 49 s on an M1 Pro (Standard).*

<!-- KR7: a Krea 2 Turbo render goes here (docs/images/krea2.jpg), made in the app; its alt text,
then its caption: prompt, format, seed, steps, time on an M1 Pro (Standard). -->

## The app

`Siliconed.app` generates images from a prompt, edits them by instruction, and keeps a queue and a
history. It follows the system language (English, French, German, Spanish or Italian).

### Create

- **Four models.** Z-Image Turbo (Apache 2.0), Qwen-Image-2.1 Turbo (Qwen Research license,
  non-commercial), Anima Turbo (CircleStone Labs license, non-commercial; its images may be used
  commercially) and Krea 2 Turbo (Krea 2 Community license: commercial under $1M annual revenue,
  with a content filter when deployed). All four read the prompt with a language model.
- **The chain, visible.** The rack shows the pipeline top to bottom (prompt, encoders, denoiser,
  decoder), with typed wires and the active stage lit; each stage holds its own settings.
- **More detail** with Z-Image and Qwen-Image-2.1: « Detail: Normal · More · Most » adds fine
  texture, at no extra time.
- **Variations**: cousins of an image, subtle or strong, with the same settings.
- **Enlarge ×4** with macOS's own super-resolution (macOS 26): no prompt, nothing added, edges and
  texture sharpened — 1024² becomes 4096² in a second or two.
- **LoRAs**, and **image-to-image** with Z-Image and Anima.

![The window: the rack on the left (prompt, text encoder, denoiser, decoder, each with its settings, time and memory), a Z-Image Turbo image at 1248×832 on the right — a red-haired Irish woman on a beach, her surfboard under her arm —, the history below](docs/images/window.jpg)

### Edit

- **Editing by instruction** with Qwen-Image-2.1: give it 1 to 3 images and a sentence (“replace the
  cloudy sky with a blue sky”); its encoder sees the images. In the app an edit is capped at a
  surface of 1024².
- **Curtain**: drag a line across two images, or across an edit or an enlargement and its
  original, to see what changed; zoom in toward the pointer to look at a detail.
- **A history like Photos**: arrows, ⌫ and ⌘Z, multiple selection, side-by-side comparison, export,
  drag and drop, Share.

![An edit by Qwen-Image-2.1: the surfer's photo as image 1 in the Images box, the instruction “Convert this photo to a western comic book style with cel shading”, the cel-shaded result in the canvas](docs/images/edit.jpg)

![An edit by Qwen-Image-2.1 from two images, under the sliding curtain: on the left the Z-Image Turbo photo of the red-haired surfer in a shorty on the beach; on the right the same woman, same beach and light, in the cream blouse and long green skirt of the woman reading in the Anima painting](docs/images/outfit.gif)

*An edit that draws on a second image: the surfer as image 1, Anima's painted reader as image 2, and
“Dress the woman of image 1 in the clothes of the woman in image 2”. 1248×832, 297 s on an M1 Pro.*

### Explore

The exploration panel builds a grid in the canvas. Its axes: seeds, steps, the strength of a LoRA,
LoRAs to try, models, the alternatives written in braces in the prompt, each image of an edit, or,
in Developer Mode, the solvers.
Each cell is a sketch, stopped after a few steps, at 512 on the short side (with Qwen-Image-2.1, a
finished image); once a model has rendered at that size, the panel shows what the grid will cost on
this Mac before you start. « Finish This Image » renders the
chosen cell to the end with the same seed: the image its sketch announced.

![An exploration with Anima Turbo and the inCase LoRA in the canvas: four hair colors (red, blonde, chestnut brown, jet black) by two seeds, a woman reading by a library window, sketches after 4 evaluations, the axes as labels, « Finish This Image » under the chosen cell](docs/images/grid.jpg)

### Model management

- **Nothing is preinstalled.** You pick a model; before the first byte is downloaded the app shows
  its license and the disk space it takes, and asks you to accept. Every file is pinned to a
  revision and checked by sha256.
- **Standard, Compact or Light.** Each model installs with the publisher's weights (Standard, the
  default), with 8-bit weights published by third parties (Compact: 11.7 GB instead of 20.4 for
  Z-Image, 17.3 GB instead of 31.2 for Qwen-Image-2.1, 4.10 GB instead of 5.65 for Anima, 19.5 GB
  instead of 34.4 for Krea 2; an image slightly different, about as fast for the first three), or with the transformer in 4 to 6 bits
  as a third party publishes it (GGUF Q4_K_M; Light: 9.5 GB, 14.2 GB, 3.31 GB and 12.8 GB, the least
  read from disk at each step; another set of weights, an image that may compose differently). For
  Z-Image, Qwen-Image-2.1 and Krea 2 the Compact's text encoder is 8-bit too, and the Light uses it;
  Anima's text encoder stays the publisher's bf16 in all three, since no 8-bit version of it is
  published. The Light is
  preselected on a Mac with 8 GB of memory, the Standard elsewhere; you choose before installing. One version at a time:
  switching replaces the other, which keeps rendering until the new one is complete, then frees its
  space — if the disk cannot hold both, the app says so before you accept (`docs/API.md` §3.11).
- **Civitai and Hugging Face imports.** LoRAs (Civitai, Comfy, kohya, diffusers layouts) and a
  fine-tune's DiT (`.safetensors`, or `.gguf` from Q4 to Q8_0), by drag and drop. A quantized file
  stays as published on disk, never re-quantized (Z-Image: 6.2 GB in 8 bits, about 5 GB in GGUF
  Q4_K_M, instead of 12.3 GB; each step a little slower, the computation stays fp32); below 4 bits,
  the file is refused (`docs/API.md` §3.11).
- **Statistics and diagnostic.** The statistics palette (⌥⌘I) is a toolbar button; « Report My
  Configuration » (what the chip computes, measured, exported as JSON — and the paths the engine
  takes on this Mac) opens from « Chain » at the head of the rack, orange until this Mac is measured.

Models live in `~/Library/Application Support/Siliconed` (override with `SILICONED_ROOT`).

![The Models window: Standard, Compact or Light, the license and the disk space shown before anything is downloaded](docs/images/models.jpg)

### Private by construction

Rendering happens on your Mac. The only network traffic is the model download you accepted; the
diagnostic copies its report and opens a GitHub issue form that you choose to submit or not. Before each render the
engine checks the model, its license, the format and the memory it needs, and refuses up front a
render beyond this Mac's budget.

## Download

The app ships as a `.dmg` on the [Releases](https://github.com/gwenn-ha-dev/Siliconed/releases) page.
Requirements: macOS 15 or later, Apple silicon. The timings below were measured with 16 GB of memory.

## Numbers

Reference machine: MacBook Pro M1 Pro (8 cores, 14 GPU cores), 16 GB. 1024², seed 42.

| | Z-Image Turbo | Qwen-Image-2.1 Turbo | Anima Turbo | Krea 2 Turbo |
|---|---|---|---|---|
| 1024², Standard | 113–119 s (8 steps; 1024×1536 190 s) | 134 s (8 steps); 1024×1536 232 s; an edit with 1 image 211 s, with 2 images 297 s | 43.7 s (8 steps); 512² 12.1 s; 1024×1536 75.6 s | —<!-- KR: speed, Standard --> |
| 1024², Light (4 to 6 bits) | 118–120 s | not remeasured since 0.10 (every step at full size) | not measured | —<!-- KR: speed, Light --> |
| peak memory, Standard / Light | 2.6 GB / 2.0 GB | 1.8–2.7 GB / 1.5 GB | 1.4 GB (the engine alone) / not measured | 3.07 GB (1024×1536: 3.51 GB) / 1.35 GB, the engine alone |
| deviation from the fp32 reference (`model_out`) | 1.2·10⁻⁵ | 3.0·10⁻⁶ from the fp64 exact (512²) | 1.4·10⁻⁵ from the fp64 exact (1024²), the fp32 reference at 1.5·10⁻⁵ | —<!-- KR: deviation --> |

Each time is a render queued in the app by `silicontrol`, from the queue to the PNG: Qwen-Image-2.1
measured on 15/10/2026 with 0.10, Z-Image on 14/10/2026 with 0.8.2. Peak memory is the app's
`phys_footprint`, sampled every 100 ms; 0 swap-out over the series. Anima Turbo's times and peak are
those of the engine alone, without the app, render cache off (14–15/10/2026); Krea 2 Turbo's peaks
too (17/10/2026). How the deviation is measured: [Verification](#verification).

Until 0.8.1, Z-Image Turbo ran 7 of the 8 DiT evaluations its recipe calls for (“8 steps” followed
an older diffusers schedule whose last step moved nothing), the first two at half size at 1024².
Fixed in 0.8.2: 8 full-size evaluations. The Z-Image times published before (82–85 s at 1024²)
were for the shorter schedule.

### Next to Draw Things

[Draw Things](https://drawthings.ai) is the reference on the Mac; here, its community engine,
[draw-things-community](https://github.com/drawthingsai/draw-things-community) at `bc3970c`, with
the model's defaults (UniPC trailing, CFG 1). The same machine, freshly restarted, one engine loaded
at a time; Z-Image Turbo, 1024², 8 steps (8 evaluations on each side), seed 42, three renders on each
side over three variants of the same prompt. Draw Things was measured on 13/10/2026 (a 70-minute
session) and nothing changed on its side since; Siliconed on 14/10/2026, after the fix.

| | weights | time | peak memory | read from disk per render |
|---|---|---|---|---|
| Siliconed, Standard | the publisher's bf16, computed in fp32 | 113–119 s | 2.6 GB | 51–58 GB |
| Siliconed, Light | GGUF Q4_K_M | 118–120 s | 2.0 GB | 3.5–8.7 GB |
| Draw Things, Exact | f16 | 167–179 s | 5.3 GB | 99 GB |
| Draw Things, default | 8 bits (`q8p`) | 142–164 s | 3.5–4.3 GB | 11 GB |
| Draw Things, 6-bit | 6 bits (`q6p`) | 136–138 s | 3.5 GB | 6–14 GB |

The comparison is about wall-clock time, not identical arithmetic: the samplers and the weights
differ, so do the images. Peak memory does not count the weights mapped from disk, on either side.
Siliconed runs the machine hotter, which follows from computing in fp32.
Until 0.8.1 this table set Siliconed's 7 evaluations against Draw Things' 8; it is now 8 against 8.

**Within the budget, measured.** Each model's peak `phys_footprint` and its swap-outs (`vm_stat` delta)
are measured at the product's worst case, on a 16 GB Mac with nothing else running: Qwen-Image-2.1 at
1024×1536 with 3 references, 0 swap-out; Z-Image at 1024×1536, 3.8 GB peak, 0 swap-out (its VAE
decodes in bands, to the same bits).

Each side must be at least 512 and a multiple of 16, with a surface of at most 1024×1536. Below
512² the models are outside their domain.

## How it compares

- **[Draw Things](https://drawthings.ai)**, the reference native app on the Mac: its engine is
  [measured above](#next-to-draw-things) on the same machine. Siliconed computes in fp32 and is
  checked stage by stage against the reference implementation.
- **[ComfyUI](https://github.com/comfyanonymous/ComfyUI) or [diffusers](https://github.com/huggingface/diffusers)
  in Python**: the same models, without a Python environment or a node graph to wire. Siliconed
  implements them natively and checks itself against diffusers and
  [transformers](https://github.com/huggingface/transformers); `silicontrol` covers scripting and
  automation.

## Community diagnostic

One machine is not a benchmark. The app's **Report My Configuration** button maps what your Mac's
chip computes, in under 30 seconds and without any model: the matrix products of the models' blocks
on the GPU, on the CPU's matrix unit and on both at once, fp16 and bf16 products, attention through
the engine's three kernels, weight widening on the CPU and on the GPU — each result checked against
the exact one computed in double precision. It then copies the JSON report and opens a GitHub issue form to paste it into (or
you send it another way): chip, CPU and GPU cores, memory, macOS, the CPU's feature flags, each throughput and whether
it computed right, the thermal state and power source before and after. Nothing is sent unless you
submit the issue.

The reports fill this table, one line per chip (TFLOP/s, fp32 unless noted):

| Chip | CPU cores | GPU cores | Memory | GPU | CPU matrix unit | Both | Attention | GPU fp16 | Source |
|---|---|---|---|---|---|---|---|---|---|
| M1 Pro | 8 | 14 | 16 GB | 3.5 | 2.2 | 5.1 | 2.5 | 3.4 | reference machine |
| *your line here* | | | | | | | | | [report](https://github.com/gwenn-ha-dev/Siliconed/issues/new?template=diagnostic.yml) |

Render times on the reference machine are under [Numbers](#numbers).

## For AI agents and scripts: silicontrol

Drives the running app from a terminal. It ships in
`Siliconed.app/Contents/Helpers/silicontrol`; Settings → Command Line links it into `/usr/local/bin`. What you
add appears in the app's window, and there is one queue, never two renders at once.

```
silicontrol help                 the full reference (always start here)
silicontrol models
silicontrol add --wait --out out "a woman posing in a library"
silicontrol status
```

Output is JSON lines; the last one carries `"ok"`. How it works inside: `docs/REMOTE-CONTROL.md`.

## For developers

**The `Siliconed` library.** A Swift API for writing your own app on top of the engine:
`docs/API.md`. The app is its reference example, compiled with the package.

### Build from source

macOS 15 or later, Apple silicon, Xcode 26 or later (the app uses macOS 26 APIs where the system has them).

```
swift build                 the library, the app and silicontrol
swift test                  the pure unit tests, no weights needed
tools/app.sh                builds Siliconed.app at the repository root and opens it
tools/app.sh --no-open      builds only
tools/dmg.sh                packages the built app as .build/Siliconed-<version>.dmg
```

`swift test` needs nothing else: no weights, no Python.

By default the app is signed ad hoc: it runs on the Mac that built it. To distribute it, sign it
with your Developer ID:

```
tools/app.sh --sign "Developer ID Application: <name> (<team>)"
```

If `SILICONED_NOTARY_PROFILE` names a `notarytool` keychain profile, the app is also notarized and
stapled; `tools/dmg.sh` then signs and notarizes the disk image with the same identity.

### Verification

Each stage (text encoder, denoiser, VAE) is compared with the reference implementation, diffusers
and transformers in fp32 on CPU, on the same input. The deviation is measured channel by channel,
each channel against its own norm, so that a fault confined to one channel cannot hide in a global
figure. Where the fp32 reference carries its own rounding error, both are judged against an fp64
computation of the same stage, and the engine may be no further from it than the reference plus
25 %. Peak memory is read from `phys_footprint`, sampled during the render, swap from `vm_stat`, and
each timing is framed by a witness measurement before and after, since the machine drifts by several
percent from one render to the next. The oracle harness and the measurement journal are not part
of this repository; what ships is the pure unit tests, and the diagnostic, which checks a machine
against exact results computed on its own CPU.

## Documentation

- `docs/API.md`: the `Siliconed` library API, for writing an app on top of it.
- `docs/REMOTE-CONTROL.md`: `silicontrol`, protocol and internals.
- `RELEASE-NOTES.md`: what changed in each version.
- Issues: [diagnostic report, bug, idea](https://github.com/gwenn-ha-dev/Siliconed/issues/new/choose).

## Credits

The models belong to their publishers (table above), each under its own license. The reference
implementations are [diffusers](https://github.com/huggingface/diffusers) and
[transformers](https://github.com/huggingface/transformers) by Hugging Face; the image resampling
reproduces [Pillow](https://github.com/python-pillow/Pillow)'s Lanczos to the bit. LoRAs and
fine-tunes come from the community, on [Civitai](https://civitai.com) and
[Hugging Face](https://huggingface.co).

## License

The code is under the [Apache License 2.0](LICENSE). The model weights are not part of this
repository: each is downloaded by you, under its own license, which the app shows before installing
it.
