# Third-party notices

The code here is generated from [ComfyUI-Hand-Tie-Clips](https://github.com/sandpies/ComfyUI-Hand-Tie-Clips) (MIT).

## `tiled_upsampler.py` — the H3 latent upscaler's layout and statistics

**Original author: [LBH-123-AI](https://github.com/LBH-123-AI).**
Code: [`Comfyui_Minimax_h3_latent_Upscaler`](https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler),
MIT. Weights: [`Minimax_h3_latent_Upscaler`](https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler),
Apache-2.0. No weights ship with this pack; the user downloads them.

A network only loads into the module layout it was trained in, so two things
in this file are theirs: the layout (a conv-in, residual blocks with a scale
embedding, depthwise temporal convs, a trilinear resize, a conv-out, under
their key names) and the 24 means and 24 standard deviations the weights were
trained against. `tools/check_tiled_upsampler.py --weights` runs the real
checkpoint through this module and through theirs and requires identical
output. Everything else in the file is this pack's code.

The node's recipe -- chunk and tile sizes, the joint-step tiling, one step at
denoise 0.2 -- is the one the user proved in bbaudio-2025's
[Comfyui-MMH3-UltimateUpscale](https://github.com/bbaudio-2025/Comfyui-MMH3-UltimateUpscale) graph (MIT).
No code was copied from it; the checker compares the two end to end when that
pack is installed and requires identical latents.

## `upscale_guide.py` -- the h3upscale guide layout

The guide geometry (`target_crop_v2` / `target_grid_stride_v1`: the source
reduced by the LoRA's factor, packed on every factor-th patch of the target's
grid) is the layout of [Alissonerdx](https://huggingface.co/Alissonerdx)'s
ai-toolkit fork, branch `minimax-h3-latent-guides` (ai-toolkit: MIT). No code
was copied; this pack's checker compared the two layouts row for row.

## `lora/h3upscale_v2_x2.safetensors` (retired, kept on Hugging Face)

Trained by Sandpies with that fork on Alissonerdx's
[h3upscale-4k](https://huggingface.co/datasets/Alissonerdx/h3upscale-4k)
dataset, on the MiniMax H3 Ref2VA base. Read both of their terms before
redistributing or using it commercially.
