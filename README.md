# ComfyUI H3 Video Upsampler

> A mirror of this code: [Hugging Face](https://huggingface.co/sandpies/ComfyUI-H3-Video-Upsampler)

Upscale a **finished** MiniMax H3 clip (or any clip) in ComfyUI: the H3 latent
upscaler enlarges it, then one light re-draw step in tiles restores detail -- the
original MMH3 Ultimate Upscale recipe, the same upsampler Hand-Tie-Clips runs over a
whole chain. Mouths, motion and the soundtrack come from your clip. **H3 Save Video
(long-form)** writes the result without holding the film in RAM.

## Install

1. Clone into `ComfyUI/custom_nodes` and restart ComfyUI.
2. Put the latent upscaler `minimax_h3_latent_upscaler_3d_bf16.safetensors` in
   `ComfyUI/models/latent_upscale_models`
   ([LBH-123-AI](https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler)).
3. Load `example_workflows/H3_Video_Upsampler.json`, pick a clip, queue.

Needs a ComfyUI with MiniMax H3 support and the H3 models (the official
`minimax_h3_ref2va_int8_convrot` DiT, text encoder, video and audio VAE). No LoRA.

## The node: H3 Video Upsampler

| widget | what it does |
|---|---|
| `megapixels` | output size, aspect kept (2 = about 1920x1080) |
| `denoise` | **0.2** is the original recipe (one re-draw step from sigma 0.75); 0 only enlarges |
| `lora` | retired and ignored (kept so saved graphs load): the h3upscale guide cost fine detail and drew tile grids against the plain recipe |
| `tile_tokens` | **VRAM.** Largest sequence per tile; lower = more, smaller tiles, slower, less memory. 70000 is the recipe; try 40000 or 25000 on smaller cards |
| `prompt` | what the re-draw follows; the default is a plain restore-detail caption |

The recipe: 102-frame chunks sharing 17, anchor 0.999, 70000-token tiles with
128 px overlap, seed 444.

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

## The node: H3 Save Video (long-form)

Frames in, an MP4 out, in one streaming pass: frames go to the GPU in small
chunks, optional RTX frame generation runs per pair (`frame_gen` x2 / x3 / x4,
needs an RTX card and the `nvidia-vfx` package), the frames are converted to
BT.709 NV12 on the GPU and encoded with NVENC at constant quality (`quality`
high / max / archive), with a keyframe on every hard cut and every 2 s. Without
CUDA it encodes on the CPU (x264 / x265 / SVT-AV1). The soundtrack is trimmed or
padded to the picture. The workflow wires the clip's own fps into it.

## The h3upscale x2 LoRA (retired)

Earlier versions guided the re-draw with an h3upscale x2 LoRA. Against the plain
recipe it cost fine detail and drew tile grids at 2 MP, so 0.3.0 runs the recipe
with no LoRA. The file stays on Hugging Face under `lora/` for reference.

## Credits

Built in [ComfyUI-Hand-Tie-Clips](https://github.com/sandpies/ComfyUI-Hand-Tie-Clips), where the same upsampler runs over a
whole chain. Latent upscaler network: LBH-123-AI. Upscale recipe: bbaudio-2025's
MMH3 Ultimate Upscale. Guide layout and dataset of the retired LoRA: Alissonerdx.
See `THIRD_PARTY_NOTICES.md`.
