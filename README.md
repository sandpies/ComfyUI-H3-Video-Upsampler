# ComfyUI H3 Video Upsampler

> The LoRA and a mirror of this code: [Hugging Face](https://huggingface.co/sandpies/ComfyUI-H3-Video-Upsampler)

Upscale a **finished** MiniMax H3 clip (or any clip) in ComfyUI: the H3 latent
upscaler enlarges it, then a light two-step re-draw in tiles restores detail. With the
**h3upscale x2 LoRA** those steps also get the clip itself, at half the output size,
as a guide -- so it restores the clip's own detail instead of inventing texture.
Mouths, motion and the soundtrack come from your clip.

## Install

1. Clone into `ComfyUI/custom_nodes` and restart ComfyUI.
2. Put the latent upscaler `minimax_h3_latent_upscaler_3d_bf16.safetensors` in
   `ComfyUI/models/latent_upscale_models`
   ([LBH-123-AI](https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler)).
3. Put `h3upscale_v2_x2.safetensors` in `ComfyUI/models/loras` (from [Hugging Face](https://huggingface.co/sandpies/ComfyUI-H3-Video-Upsampler/tree/main/lora)).
4. Load `example_workflows/H3_Video_Upsampler.json`, pick a clip, queue.

Needs a ComfyUI with MiniMax H3 support and the H3 models (DiT, text encoder,
video and audio VAE).

## The node: H3 Video Upsampler

| widget | what it does |
|---|---|
| `megapixels` | output size, aspect kept (2 = about 1920x1080) |
| `denoise` | fraction of the schedule the two re-draw steps cover; **0.1** is the setting that held, 0 only enlarges |
| `lora` | `h3upscale_v2_x2.safetensors` = guided re-draw (detected from its header). Any other LoRA just sits on the re-draw |
| `tile_tokens` | **VRAM.** Largest sequence per tile; lower = more, smaller tiles, slower, less memory. 70000 is the default; try 40000 or 25000 on smaller cards |
| `prompt` | defaults to the caption the LoRA was trained on; keep it with that LoRA |

- Wire the **plain** diffusion model loader as `model` when using the LoRA: it was
  trained on the bare DiT, and extra LoRAs under it re-add their own texture.
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

## What the LoRA does and does not do

Tested by eye on H3 renders: guided at denoise 0.1, the 2 MP result beat the native
render in a blind comparison (0.1.0, one step). 0.2.0 runs the re-draw exactly as
Hand-Tie-Clips' chain upsampler does: two steps on the denoise schedule, 192 px
tile overlap. It is an **upscaler**: generating a video from noise
with it flickers every 17 frames (it was trained on 5-frame clips); refining an
existing clip, as this node does, does not.

## Credits

Built in [ComfyUI-Hand-Tie-Clips](https://github.com/sandpies/ComfyUI-Hand-Tie-Clips), where the same upsampler runs over a
whole chain. Latent upscaler network: LBH-123-AI. Upscale recipe: bbaudio-2025's
MMH3 Ultimate Upscale. Guide layout and dataset: Alissonerdx. See
`THIRD_PARTY_NOTICES.md`.
