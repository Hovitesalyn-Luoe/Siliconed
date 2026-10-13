English · [Français](RELEASE-NOTES.fr.md)

# Siliconed — release notes

## 0.8.1

### Fixed

- **The rack's last rows take the click again.** Just above the column's footer, the LoRA row,
  **+ LoRA** and **Import…** were drawn but did not respond until the column was scrolled. The
  footer now sits under the rack instead of over it.

### PNG metadata Civitai reads

- **Upload an image to Civitai and its settings show up.** Beside its own keys, each PNG now carries
  the text Civitai reads: the prompt with a `<lora:name:weight>` tag per LoRA, then the steps, the
  solver, the seed, the size and the model. Z-Image Turbo and Qwen-Image-2.1 renders credit their
  Civitai page with no manual entry.
- **LoRAs are credited too**, by the sha256 of the file as it was published, kept when you import
  it. A LoRA imported with an earlier version is named but not credited: import it again to credit it.

## 0.8

### Check for Updates

- **In the app menu, on demand only.** Siliconed asks GitHub for its latest release and compares it
  with the version running; if a newer one exists, it opens its page. Nothing is checked in the
  background, nothing is installed for you.

### Solvers (Developer Mode)

- **Euler ancestral, DPM++ 2M and UniPC**, besides Euler, on Z-Image Turbo and Qwen-Image-2.1: the
  same steps, the same σ, another walk between them. Each one is checked against diffusers'
  scheduler of the same name. Euler, the default, gives the same images as before, bit for bit.
- **Another image, the same time.** UniPC may recompose a Qwen-Image-2.1 picture; the ancestral
  solver may flatten a background. Edits keep image 1 under all four. Previews and sketches show
  each solver's own estimate. In the rack, the history caption, the PNG and `silicontrol add|grid
  --solver`.

### The numbers

- **Remeasured through the app**, every render queued by `silicontrol`: Z-Image Turbo 1024² in 82 to
  85 s, Qwen-Image-2.1 in 118 s, an edit with one image in 212 s, with two in 285 s; 0 swap-out.
  The README sets them next to the Draw Things engine on the same Mac.

Timings and memory are measured on a MacBook Pro M1 Pro with 16 GB.

## 0.7.1

0.7 was never released: 0.7.1 carries it, with the memory computation made exact in the cases it
missed.

### Memory: a fixed budget

- **Siliconed never takes more than its budget**: three quarters of the Mac's memory minus 2 GB
  (10 GB on a 16 GB Mac, 4 on an 8 GB one). It no longer reads how busy the Mac is: the same render
  on the same Mac always gets the same answer, whatever else is open. What the other apps hold is
  theirs.
- **Known before the first step.** Each render computes the memory it needs from its format, its
  prompt, its images and its LoRAs, takes the fastest plan that fits in the budget, and checks its
  own peak against that figure. Nothing switches plans halfway through any more. The figure is an
  upper bound: a long prompt on Qwen-Image-2.1, a batch of several images and the app's own memory
  are counted, and a render whose need cannot be computed is refused rather than started blind.
- **Greyed out ahead.** A format or a model beyond this Mac's budget is greyed out before you click,
  with the Mac it would need: “needs a Mac with 24 GB”.
- **Batches of Qwen-Image-2.1** no longer keep two copies of the model at once.
- Same images, bit for bit. Measured on the M1 Pro: Z-Image Turbo 1024² in 81 s, Qwen-Image-2.1
  1024² in 116 s; the largest edit (1024×1536, three images) peaks at 3.2 GB.
- The diagnostic report (schema 4) gives the budget, the need of each stage and the peak measured
  against it.

### Named Subjects (Experimental)

- **Nothing kept.** Subjects now live in memory only, like the session's images: nothing is written
  to disk, they go when the app quits. A subjects folder written by 0.6 is deleted at launch.
- **Images or Subjects.** The box above the prompt holds either the images of an edit (1 to 3) or
  your subjects (as many as you like, the prompt citing 3 at most) — one or the other, never both.
- **The name never reaches the model.** “@alice” becomes “the person in image 1”: a name read as
  content gave Alice in Wonderland.

Timings and memory are measured on a MacBook Pro M1 Pro with 16 GB.

## 0.6

### One Images box, above the prompt

- **The images come first.** On Qwen-Image-2.1, the box that holds the images to edit is now called
  Images and sits above the prompt: you lay the picture down, then say what to make of it. Its wires
  run past the prompt to the encoders that read them.
- **One budget, counted once.** With Named Subjects on (Experimental), the subjects live in the same
  box, under the edit's images: to the model, a subject you cite is one more image. The box's title
  counts both against the model's 3 images, and turns red past them.

### Memory

- **An idle Mac no longer refuses Qwen-Image-2.1 at 1024².** A plain generation was charged the
  memory of an edit with 3 images, and the safety margin ignored what macOS frees by compressing the
  other apps' memory. Now a render is charged the peak measured for its own number of images, and
  the margin counts that compression: on the Mac of the report, from 2.9 to 4.5 GB available for
  the same desktop. The no-swap rule is unchanged: everything is still measured at the worst case.

### Named Subjects (Experimental)

- **Compare with the subject.** “@ana as a western comic” keeps the picture's framing, as an edit
  does: the curtain under the image now compares the result with its image 1, whether it came from
  the Images box or from a subject (its frame, as it was sent).

### Under the hood

- Superseded kernels and settings removed (a scalar attention kernel, a half-precision mode): one
  version of each computation lives, and it is the one that is checked. Nothing changes in the
  images.

Timings and memory are measured on a MacBook Pro M1 Pro with 16 GB.

## 0.5

### Qwen-Image-2.1 is now Qwen's official Turbo

- **One model, from its publisher.** Qwen-Image-2.1 now runs Qwen's own distilled model,
  [Qwen-Image-2.1 Turbo](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo). Until 0.4, the app ran
  the base model under a third party's turbo LoRA (Viggle's); that LoRA and the base model are
  gone. Text-to-image, editing by instruction with 1 to 3 images and your own LoRAs work as before.
- **8 steps, on the schedule Qwen publishes.** The step count is fixed at 8: the 9-step mode is
  gone, and any other count is refused.
- **Compact and Light follow.** The Compact takes Comfy-Org's 8-bit (int8 convrot) Turbo, the
  Light unsloth's GGUF Q4_K_M Turbo; the Compact's 8-bit text encoder is unchanged. On disk:
  31.2 GB in Standard, 17.3 GB in Compact, 14.2 GB in Light (33.2, 19.4 and 16.2 GB before: the
  turbo LoRA is no longer stored).
- **An existing installation is recognized.** The Models window asks you to “Finish
  Installing…”: the old transformer and small files are removed, the Turbo's are downloaded in
  the same version (Standard, Compact or Light). The text encoder is kept: the Turbo's is the base
  model's, bit for bit, so it is not downloaded again. Fine-tunes of Qwen-Image-2.1 imported for
  the base model can no longer render and are removed.
- **Same license**: Qwen Research, non-commercial — the Turbo carries the base model's text.
- **Checked like the rest.** Every stage is compared with Qwen's reference implementation; the
  engine stays within 3.0·10⁻⁶ of the exact (fp64) computation.

### Faster than 0.4 for an image, slower for an edit

The Turbo takes 8 steps where the third-party LoRA took 6. We chose it anyway: it is the quality Qwen
publishes and checks, with no third-party piece in between that could change or disappear. Then we
made its steps cheaper, without lowering the precision: everything is still computed in fp32 and
checked against the exact computation.

- **The first two steps at half size.** At 1024² and above, a generation computes its first two
  steps at 512², then enlarges, as Z-Image already does. The image changes (the same seed gives
  another composition) but stays clean. Not under a LoRA or Detail, nor in an edit.
- **Attention shared between the GPU and the CPU.** In every layer, 13 of the 32 attention heads
  go to the CPU's matrix unit while the GPU computes the others.
- **Smaller savings**, with the same bits: the model is read in one stream, and normalizations are
  fused.

On the reference Mac:

- 1024² generation: ~120 s (~134 s in 0.4, ~160 s for the Turbo without these savings);
- an edit with 1 reference: ~210 s (~190 s), with 2 references ~285 s;
- the worst case, 1024×1536 with 3 references: ~515 s, 4.7 GB of memory at the peak, without any
  swap;
- the Light takes the same time as the Standard.

### Installation

- **Faster, and lighter on memory.** Preparing a downloaded model for the engine now takes seconds
  instead of minutes, and the next file downloads while the previous one is prepared. On a fast
  connection (85 MB/s), Qwen-Image-2.1 installs in ~6½ min in Standard (~8 min in 0.4), ~3½ min in
  Compact (~5½) and ~3 min in Light (~4); the network is now what sets the time. The installed
  model is the same, byte for byte.
- **No swap while installing.** The installation's memory peak goes from 4.6 GB to 2.2–2.5 GB, and
  freshly downloaded files no longer crowd the rest of your Mac out of memory: an installation
  that could swap on a busy Mac no longer does.

### Batches

- **Batches no longer swap.** Each image of a batch used to rebuild the model, and on a busy Mac a
  batch could end up swapping. Now every image is computed with the same model, then every image
  is decoded: the same images, with no swap, on Z-Image and Qwen-Image-2.1.
- **Including with your own LoRA.** On Qwen-Image-2.1, a batch with a LoRA of yours keeps the
  model, LoRA applied, from one image to the next.

Timings are measured on a MacBook Pro M1 Pro with 16 GB.

## 0.4

### Installation

- **Standard, Compact or Light.** Z-Image Turbo and Qwen-Image-2.1 install with the publisher's
  weights (Standard), with 8-bit weights published by third parties for the transformer and the text
  encoder (Compact: 11.7 GB instead of 20.4 for Z-Image, 19.4 GB instead of 33.2 for
  Qwen-Image-2.1), or with the transformer in 4 to 6 bits (GGUF Q4_K_M) and the Compact's text
  encoder (Light: 9.5 GB and 16.2 GB). The Light is preselected on a Mac with 8 GB of memory. The
  installed version keeps rendering until the new one is complete.
- **Downloads that resume.** Large files come in chunks, several at a time; a stalled chunk is asked
  again, and a cut download resumes where it stopped. Every file is still checked by sha256.
- **Signed and notarized.** The app and the disk image are signed with a Developer ID and notarized
  by Apple: they open with a double click.

### Editing

- **Edit from anywhere.** Right-click an image in the history and choose « Edit with
  Qwen-Image-2.1 », even when the rack is set to a model that does not edit: the rack switches to
  the editor and the image becomes image 1. A thumbnail dragged from the history onto the Edit box
  becomes a reference.
- **Original / edited curtain.** An edited image has a button that lays it over its original, with a
  line to drag across to see exactly what changed. Escape leaves it.

### Images

- **Detail at 1024².** « Detail: Normal · More · Most » now gives clean results at 1024² too, on
  Z-Image and Qwen-Image-2.1: its strength follows the size of the image, and it no longer leaves
  a grainy texture at high resolution.
- **Variations follow « Images ».** Subtle or strong variations of an image now make as many images
  as the rack's « Images » field (1 by default), instead of always four.

### Exploration

- **Pick the model in the panel.** The exploration panel has its own « Model » menu.
- **One status line.** The panel says where it is in a single line: « Sketch at n steps of N », or
  « Finished images, N steps » for Qwen-Image-2.1, which now explores with finished images (its
  early sketches were blurry).

### Imported models

- **8-bit stays 8-bit.** A fine-tune imported in 8 bits — fp8, int8, GGUF Q8_0 or int8 convrot — is
  kept in 8 bits on disk, about half the size of the 16-bit file (Z-Image: 6.2 GB instead of 12.3
  GB). The computation is still fp32, and every weight is checked bit for bit against the
  publisher's own conversion. The app imports `.gguf` files as well as `.safetensors`.
- **GGUF down to 4 bits.** A `.gguf` fine-tune in Q4_0 to Q6_K is kept as published, its blocks
  copied whole (Z-Image: about 5 GB in Q4_K_M instead of 12.3 GB). Files below 4 bits, and 4-bit
  files outside GGUF, are refused, with the reason.

### Speed and memory

- **Z-Image is faster at 1024²**: about 82 s instead of about 107 s, and 153 s instead of 168 s at
  1024×1536, still without any swap.
- **A memory check on the machine's real state**: before each render, the app reads the memory it can
  really have at that moment (not just the Mac's total) and refuses, with the figures, a render that
  would not fit — instead of letting it swap. If memory gets short during a render, the engine
  switches to a leaner plan: the same image, a little slower, rather than swapping.

### Languages

- **German, Spanish and Italian**, besides English and French: the app follows the system language.

Timings are measured on a MacBook Pro M1 Pro with 16 GB.
