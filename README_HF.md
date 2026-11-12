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

Make a finished MiniMax H3 video (or any video) bigger and sharper in ComfyUI.
Faces, motion, mouths and the soundtrack stay your clip's own.

## Quick start

1. Clone this repo into `ComfyUI/custom_nodes` and restart ComfyUI.
2. Download the files below into the folders shown.
3. Load `example_workflows/H3_Video_Upsampler_File.json`, pick your video, press Queue.
   The MP4 lands in `ComfyUI/output/video/`.

Needs **ComfyUI v0.38.0 or newer**.

## Files to download

| File | Put it in | Size |
|---|---|---|
| [`minimax_h3_fl2va_pruned_int8_convrot`](https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors) | `models/diffusion_models` | 21 GB |
| [`minimax_h3_video_vae_int8_convrot`](https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/vae/minimax_h3_video_vae_int8_convrot.safetensors) | `models/vae` | 2.8 GB |
| [`minimax_h3_audio_vae_fp32`](https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/vae/minimax_h3_audio_vae_fp32.safetensors) | `models/vae` | 0.6 GB |
| [`minimax_h3_latent_upscaler_3d_conv_v1_bf16`](https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler/blob/main/minimax_h3_latent_upscaler_3d_conv_v1/minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors) | `models/latent_upscale_models` | small |
| [`open_h3_regenerate_2k_r256pp_bf16.safetensors`](https://huggingface.co/sandpies/ComfyUI-H3-Video-Upsampler/blob/main/lora/open_h3_regenerate_2k_r256pp_bf16.safetensors) | `models/loras` | 3.8 GB |

The latent upscaler can instead be LBH-123-AI's 2D one,
[`minimax_h3_latent_upscaler_2d_conv_v1_bf16`](https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler/blob/main/minimax_h3_latent_upscaler_2d_conv_v1/minimax_h3_latent_upscaler_2d_conv_v1_bf16.safetensors)
(same folder, 0.36 GB): pick it in `upscale_model` and set `sharpness`. The old 3D file name,
`minimax_h3_latent_upscaler_3d_bf16.safetensors`, still loads.

That's everything for the normal mode. Optional: [`minimax_h3_sol_vae_fp16`](https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/vae/minimax_h3_sol_vae_fp16.safetensors)
in place of the video VAE makes a run about 20% faster (needs a ComfyUI from late October 2026).

## Two ways to re-draw: `method`

- **`regen2k`** (default): fast, and good for most clips.
- **`ic`**: keeps the original frames in view while it re-draws, so faces, writing
  on signs and mouths stay closest to your clip. About 1.7x slower.

`ic` needs these extra files:

| File | Put it in | Size |
|---|---|---|
| [`minimax_h3_ref2va_int8_convrot`](https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/diffusion_models/minimax_h3_ref2va_int8_convrot.safetensors) (use this as the model for `ic`, with no LoRAs) | `models/diffusion_models` | 34 GB |
| [`minimax_h3_open_regenerate_2k_ic_lora_avg_rank_21_bf16`](https://huggingface.co/Kijai/MiniMax-H3-experimental/blob/main/loras/minimax_h3_open_regenerate_2k_ic_lora_avg_rank_21_bf16.safetensors) | `models/loras` | 0.2 GB |
| [`minimax_h3_hyperflow_8step_v1.0`](https://huggingface.co/videorebirth/hyperflow/blob/main/minimax_h3_hyperflow_8step_v1.0.safetensors) | `models/loras` | 2.8 GB |
| *optional:* [`minimax_h3_lms_v1.0_r64`](https://huggingface.co/Alissonerdx/Minimax-H3-ComfyUI/blob/main/loras/minimax_h3_lms_v1.0_r64.safetensors) (a little extra sharpness) | `models/loras` | 1.2 GB |

To switch: set `method` to `ic` and load `minimax_h3_ref2va_int8_convrot` in the model loader.
If something is missing, the node stops straight away and tells you which file.
The `clip` input can stay empty: wiring the text encoder (27 GB) looked the same in our tests.

## Settings worth knowing

Hover any setting in ComfyUI for a short explanation. The ones you might change:

| Setting | What to do |
|---|---|
| `megapixels` | Output size. **2** is about 1080p. It must be bigger than your clip. |
| `redraw_steps` | **3** is the normal recipe. **2** is a third faster and looked the same in our tests. |
| `redraw_start` | How much the re-draw may change (`regen2k` only). **0.6** keeps faces close; 0.8 is sharper but faces can change. |
| `denoise` | **0** only enlarges, with no re-draw. |
| `sharpness` | Only with the 2D latent upscaler: **1** stays closest to your clip, **6** is sharpest. The 3D upscaler ignores it. |
| `conditioning` | Optional text describing the look, e.g. *"2D anime, flat cel shading"*. Helps animation. |

Leave the rest at their defaults: they adjust themselves to your graphics card.

## The nodes

- **H3 Video Upsampler (file to file)**: video file in, MP4 out. Best for anything
  longer than a few seconds: memory use stays low however long the video is.
- **H3 Video Upsampler**: frames in, frames out, for building into bigger workflows.
- **H3 Load Video (path)**: opens a video where it is on your disk, so there's no
  upload size limit.
- **H3 Video Frames**: a lighter Get Video Components that can trim and shrink while reading.
- **H3 Save Video**: saves long videos without filling memory; can double the
  frame rate on RTX cards (`frame_gen`).

## Tips and problems

- **48 fps video?** Set `frame_rate` to `24 (H3 native)` and `frame_gen` to x2:
  half the work, and the lip sync stays exact.
- **Running out of memory?** Leave `tile_tokens` at 0 and `chunking` / `tiling` on.
  [ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes) installed lowers memory
  further, automatically.
- **Small faces in the distance** get their own close-up re-draw automatically
  (`regen2k`; `ic` keeps faces from your clip anyway). It needs OpenCV, which most
  ComfyUI installs have.
- **Anime getting extra lines?** Add a `conditioning` prompt describing the look.
- **Log says `dense (ComfyUI-KJNodes is too old for sparse attention: update it)`?**
  The run still works, just slower. Update ComfyUI-KJNodes to get sparse attention back.

## Credits

Built in [ComfyUI-Hand-Tie-Clips](https://github.com/sandpies/ComfyUI-Hand-Tie-Clips), where the same upsampler runs over a
whole chain. Latent upscaler network: LBH-123-AI. Upscale recipe: bbaudio-2025's
MMH3 Ultimate Upscale. Open-H3-Regenerate-2K and Regenerate-2K-IC: NVlabs
(`Efficient-Large-Model`); IC LoRA resize and core's H3 latent upscaler: Kijai.
HyperFlow: videorebirth. LMS Sharp and the guide layout: Alissonerdx. Face detector:
YuNet (Shiqi Yu, opencv_zoo). See `THIRD_PARTY_NOTICES.md`.
