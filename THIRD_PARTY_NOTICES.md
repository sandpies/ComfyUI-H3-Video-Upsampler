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

## Open-H3-Regenerate-2K (`redraw=regen2k`)

The 3-step Regenerate-2K schedule (`[0.8, 24/35, 0.48, 0]`; release `[0.909375, 6/7, 2/3, 0]`) and refiner LoRA are
from NVlabs' [Open-H3-Regenerate-2K](https://huggingface.co/Efficient-Large-Model/Open-H3-Regenerate-2K)
and [Kijai/MiniMax-H3-experimental](https://huggingface.co/Kijai/MiniMax-H3-experimental).

## `upscale_guide.py` -- the h3upscale guide layout

The guide geometry (`target_crop_v2` / `target_grid_stride_v1`: the source
reduced by the LoRA's factor, packed on every factor-th patch of the target's
grid) is the layout of [Alissonerdx](https://huggingface.co/Alissonerdx)'s
ai-toolkit fork, branch `minimax-h3-latent-guides` (ai-toolkit: MIT). No code
was copied; this pack's checker compared the two layouts row for row.

## `lora/open_h3_regenerate_2k_r256pp_bf16.safetensors` (Hugging Face only)

A rank-256 compression of the
[Open-H3-Regenerate-2K](https://huggingface.co/Efficient-Large-Model/Open-H3-Regenerate-2K)
LoRA, redistributed under its license, the
[MiniMax H3 community license](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/6818f6c32d12b210915e44ad56a4228c2608f160/LICENSE).
Read it before using the LoRA commercially.

## Ideas from ComfyUI-UtilsCollection (no code copied)

The file-to-file and Video Frames nodes' streamed reading, the latent cache
and the 24 fps re-timing follow ideas from **[silveroxides](https://github.com/silveroxides)'**
[`ComfyUI-UtilsCollection`](https://github.com/silveroxides/ComfyUI-UtilsCollection)
(AGPL-3.0), from its H3 reference-video path:

- decoding a VIDEO in batches with core's own trim, rotation and crop rules,
  instead of assembling the whole clip as one tensor;
- keying decoded media to the file's content, so a rerun skips the decode;
- resampling to H3's 24 fps by nearest frame.

The code in `video_upsampler.py` was written fresh for this pack and shares
none of that repository's code. silveroxides agreed to this on the condition
that the ideas are credited, which is what this section does.
