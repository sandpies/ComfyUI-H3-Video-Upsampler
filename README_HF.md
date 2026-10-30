---
license: other
license_name: see-third-party-notices
base_model: Comfy-Org/MiniMax-H3
library_name: comfyui
tags:
- minimax-h3
- comfyui
- lora
- video-upscale
---

# H3 Video Upsampler

> Code: [GitHub](https://github.com/sandpies/ComfyUI-H3-Video-Upsampler). The Regen2K LoRA is in [`lora/`](./lora).

Upscale a **finished** MiniMax H3 clip (or any clip) in ComfyUI: the H3 latent
upscaler enlarges it, then a 3-step Open-H3-Regenerate-2K re-draw in tiles restores
detail -- the same upsampler Hand-Tie-Clips runs over a whole chain. Mouths, motion and the soundtrack come from your clip. **H3 Save Video
(long-form)** writes the result without holding the film in RAM.

## Install

1. Clone into `ComfyUI/custom_nodes` and restart ComfyUI.
2. Put the latent upscaler `minimax_h3_latent_upscaler_3d_bf16.safetensors` in
   `ComfyUI/models/latent_upscale_models`
   ([LBH-123-AI](https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler))
   and the Regen2K LoRA [`open_h3_regenerate_2k_r256pp_bf16.safetensors`](https://huggingface.co/sandpies/ComfyUI-H3-Video-Upsampler/blob/main/lora/open_h3_regenerate_2k_r256pp_bf16.safetensors)
   (3.77 GB, from this pack's Hugging Face `lora/`) in `ComfyUI/models/loras`.
3. Load `example_workflows/H3_Video_Upsampler_File.json` (recommended: any length, flat RAM) or `example_workflows/H3_Video_Upsampler.json` (IMAGE in, IMAGE out), pick a clip, queue.

Needs a ComfyUI with MiniMax H3 support and the H3 models (the official
`minimax_h3_fl2va_pruned_int8_convrot` DiT, video and audio VAE).

On a ComfyUI from 28 October 2026 on, the 3D upscale runs through core's own
H3 latent upscaler (Kijai): the whole clip in one pass with clip-wide statistics, so
every chunk starts the re-draw from one consistently toned film, and faster. Older
ComfyUI uses the bundled copy, chunk by chunk, with the same checkpoint.

## The node: H3 Video Upsampler

| widget | what it does |
|---|---|
| `megapixels` | Output size, shape kept (2 = about 1920x1080) |
| `denoise` | How strongly the Regen2K LoRA re-draws detail. **0.8** = normal; 0 = no re-draw, only enlarge |
| `lora` | The Regen2K LoRA from `models/loras` (`open_h3_regenerate_2k_r256pp_bf16.safetensors`) |
| `tile_tokens` | How much work goes to the GPU at once. **0 = automatic (recommended):** it learns what your card can take and remembers it for the next run (in `ComfyUI/user/htc_upsampler_memory.json`). A number fixes it: lower uses less VRAM by cutting the clip into shorter pieces (never under 68 frames) and then tiles |
| `sampler_name` | Keep `euler`: Regen2K is tuned for it |
| `redraw_seed` | Noise seed for the re-draw (42). Same seed, same result |
| `attention` | `dense` (default) or `sparse`: ComfyUI's sparse attention, much faster on long clips but approximate, so compare faces. Runs dense on a ComfyUI without it |
| `chunk_join` | How the overlapping pieces of a long clip are joined. **`stepped` (default):** all pieces, and all tiles when a frame has to be tiled, are re-drawn together and merged after every step, so faces don't change at a join and VRAM stays at one piece's worth. `crossfade`: each piece on its own, overlaps blended (the old way; a face can shift at a join) |
| `chunking` / `tiling` | On (default) = the work may be split when it's too much for the GPU; neither forces a split. Chunking cuts the clip into overlapping pieces of about 4 s; tiling also cuts frames into tiles, only when even a short piece doesn't fit. **Both off = the whole clip in one pass**, for short clips on big cards. With tiling off, running out of VRAM stops with a message saying what to change |
| `redraw_start` | How much the re-draw may change. **0.6 (default)** keeps faces close to your clip. 0.8 = the original Regen2K recipe: sharper, but faces drift more. Lower than 0.6 changes less again. The rest of the schedule follows automatically (0.6 → 0.48 → 0.3 → 0 at 3 steps) |
| `redraw_steps` | **3 (recipe)** or **2**: a third faster, and looked the same in side-by-side tests |
| `fps` (optional input) | The clip's frame rate (from Video Frames). Needed for lip sync when the clip isn't 24 fps: what the re-draw hears is retimed to match. The output soundtrack is always the original |
| `conditioning` (optional input) | A text prompt (CLIPTextEncode, H3 text encoder) describing the clip. Unwired works well for most clips. See *Guiding the re-draw* |

The recipe: 102-frame chunks sharing 17, 70000-token tiles with 128 px overlap,
fresh noise per chunk, the soundtrack pinned clean.

- Wire a **plain** diffusion model loader as `model` (no LoRAs): LoRAs under the
  re-draw add their own texture.
- `audio_vae` wired: the re-draw hears the soundtrack (best at 24 fps). Unwired:
  it hears silence. Either way you get your original soundtrack back.
- **Low VRAM / RAM:** with [ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes)
  installed, the re-draw model gets its MiniMax low-VRAM attention and chunked
  feed-forward (same maths, less memory); without it they are skipped. Other
  models are unloaded before each stage, the latent upscale falls back to CPU and
  the re-draw to smaller tiles if the card runs out, and the result is decoded
  in slices so a long clip never sits in RAM as one tensor.
- Any frame size and length: frames are fitted to 32 px and padded to a valid H3
  length, then the padding is trimmed off again.

## The node: H3 Video Upsampler (file to file)

The same upsampler, from a **Load Video** straight to an MP4. Use it for
anything longer than a short clip, or on a machine with less RAM.

The IMAGE node is limited by ComfyUI's IMAGE type. Get Video Components
decodes the whole clip into one float32 tensor (briefly twice while it stacks
the frames), the node makes a fitted copy, and it returns the whole result as
another tensor. At 2 MP, one minute at 24 fps is about 16 GB in and 36 GB
out; at 48 fps it's about 32 GB in and 72 GB out. The re-draw in between only
needs the latent, which is about 100 MB for that minute.

The file node never holds the frames:

1. It reads the video 102 frames at a time and encodes each piece as it
   arrives. The H3 VAE encodes every 17 frames on their own, so the latent is
   exactly what the whole clip encodes to.
2. It runs the same 3D upscale and Regen2K re-draw as the IMAGE node.
3. It decodes slice by slice straight into H3 Save Video's streaming encoder
   (NVENC, keyframes on cuts, `frame_gen`).

RAM stays at a few GB however long the film is. The frame rate and
soundtrack come from the file, so lip sync always uses the real rate.
Widgets are the IMAGE node's, plus H3 Save Video's `frame_gen`, `codec`,
`quality` and `filename_prefix`.

Two more widgets:

- **`frame_rate`**: `source` keeps every frame of the file. `24 (H3 native)`
  re-times the clip on read: each output frame is the source frame nearest
  its time on a 24 fps clock, so frames are dropped above 24 and repeated
  below. The real duration is kept, so the soundtrack lines up exactly with
  no retiming. For a 48 fps clip interpolated from a 24 fps render, this
  gets back the original frames and halves the re-draw; `frame_gen` x2
  saves 48 fps again.
- **`reuse_encoded`** (on): the encoded clip (about 100 MB a minute) is kept
  in ComfyUI's temp folder, keyed to the file's content, trim, crop,
  `frame_rate` and VAE. The next run on the same clip skips reading and
  encoding, which helps A/B runs of settings or seeds. Up to 4 clips
  are kept, and the temp folder is cleared when ComfyUI starts.

A VIDEO trimmed or cropped upstream is read with core's own rules (trim
window, rotation, even-pixel crop), still a chunk at a time.

## The node: H3 Load Video (path)

Core's Load Video reads from `ComfyUI/input`, and ComfyUI refuses browser
uploads over 100 MB unless it's started with `--max-upload-size`. This node
reads the file **where it is**: nothing is uploaded or copied, so there's no
size limit and no duplicate of a multi-GB file.

- **`path`**: paste a full path (Windows' *Copy as path* quotes are fine),
  or a name relative to `ComfyUI/input`.
- **Outputs:** a normal `VIDEO` for the file-to-file node, Video Frames or any
  core video node, plus `fps`, `frame_count` and an `info` line (resolution,
  fps, frame count, duration, codec, bit depth, audio, file size). Wire `info`
  to Preview Any to see it. It's also printed to the console.
- A **variable frame rate** (common in phone and screen recordings) is flagged.
  Frames are timed by count, so convert such files to a constant rate before
  long runs.
- It re-runs only when the file changes (size or modified time). A wrong path
  or a file that isn't a video is reported when you queue.

It can open any video the ComfyUI process can read, so on a ComfyUI shared
with other people, keep that in mind.

## The node: H3 Video Frames (lean)

A drop-in for **Get Video Components** after Load Video: the same `images`,
`audio` and `fps`, plus `frame_count` and the VIDEO passed through. Core
decodes every frame to float32 into a list and then stacks it, so it briefly
holds the clip twice. This node applies its settings while decoding, so
skipped frames are never held, and fills the IMAGE once from 8-bit frames. On
a 600-frame 640×360 test clip, peak RAM was **1.08×** the output, against
**2.01×** for core-style decoding.

| widget | what it does |
|---|---|
| `start_seconds` | skip this much first |
| `duration_seconds` | how much to take, in seconds (0 = to the end) |
| `snap_to_h3` | on (default): round to the nearest length H3 uses natively (17n + 5 frames: 30 s at 24 fps = 719), so nothing is padded; the log says what you got. Off = exactly seconds x fps |
| `frame_rate` | `source`, or `24 (H3 native)`: nearest-frame re-time on read, duration and sync kept |
| `resize` / `resize_value` | shrink while decoding to a megapixel area or a long edge in px; never enlarges |

The audio is cut to the frames kept. The IMAGE workflow uses this node now;
for whole films, the file-to-file node avoids the IMAGE altogether.

## Guiding the re-draw

The re-draw starts from mostly noise and, with no prompt, only has the
3D-upscaled frames to go on. Realistic footage mostly holds up, but faces can
drift slightly, and animation can get extra lines and shading. Two controls
help:

- **`redraw_start`** sets how much the re-draw may change. Side-by-side tests
  traced face drift to the re-draw itself (not the upscaler, seed, LoRA
  strength or prompt), and starting at **0.6** instead of 0.8 kept faces
  closest to the source, with a little less sharpening, so 0.6 is the default.
- **`conditioning`** says what the clip is. Wire `CLIPTextEncode` with the H3
  text encoder and either type a caption or feed it a VLM's (Qwen-VL, Florence,
  etc.) description of a few frames. Every spatial tile hears the whole prompt,
  so when a frame splits into tiles a prompt naming objects ("a woman holding a
  cup") can make each tile draw them. Describe the **look** and keep objects
  general: *"2D anime, flat cel shading, clean black line art, no film grain"*,
  or *"live-action, natural skin texture, shallow depth of field"*.

## Clips that aren't 24 fps

H3 places every frame 1/24 s after the last and lays the soundtrack under them
on that clock. A 60 s clip at 48 fps is 2,880 frames, which H3 reads as 120 s.
Without `fps` wired, the real soundtrack covered only the first half, at the
wrong speed, and the second half heard silence. The re-draw then pulled mouths
toward the wrong syllables. Wire `fps` and what the re-draw hears is
stretched to match.

If the 48 fps file was interpolated from a 24 fps render, it's better to
upscale the original 24 fps frames (drop every other frame) and turn on
`frame_gen` x2 in H3 Save Video. That's half the re-draw work and keeps sync
exact.

## The node: H3 Save Video (long-form)

Frames in, an MP4 out, in one streaming pass: frames go to the GPU in small
chunks, optional RTX frame generation runs per pair (`frame_gen` x2 / x3 / x4,
needs an RTX card and the `nvidia-vfx` package), the frames are converted to
BT.709 NV12 on the GPU and encoded with NVENC at constant quality (`quality`
high / max / archive), with a keyframe on every hard cut and every 2 s. Without
CUDA it encodes on the CPU (x264 / x265 / SVT-AV1). The soundtrack is trimmed or
padded to the picture. The workflow wires the clip's own fps into it.

## The h3upscale x2 LoRA (removed)

Versions up to 0.2.2 guided the re-draw with an h3upscale x2 LoRA. 0.4.0 replaces
it with the Regen2K LoRA and the old file is gone from `lora/`.

## Credits

Built in [ComfyUI-Hand-Tie-Clips](https://github.com/sandpies/ComfyUI-Hand-Tie-Clips), where the same upsampler runs over a
whole chain. Latent upscaler network: LBH-123-AI. Upscale recipe: bbaudio-2025's
MMH3 Ultimate Upscale. Open-H3-Regenerate-2K refiner and 3-step schedule: NVlabs
(`Efficient-Large-Model/Open-H3-Regenerate-2K`) and Kijai. Core's H3 latent upscaler, used when present: Kijai. Guide layout and
dataset of the retired LoRA: Alissonerdx. See `THIRD_PARTY_NOTICES.md`.
