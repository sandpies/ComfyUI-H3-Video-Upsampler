# Generated from ComfyUI-Hand-Tie-Clips (https://github.com/sandpies/ComfyUI-Hand-Tie-Clips) by tools/build_upsampler_pack.py.
# Edit it there, not here: the next build overwrites this file.
"""H3 Video Upsampler: a finished MiniMax H3 clip, upscaled with the H3 latent
upscaler and a light two-step re-draw in tiles. An h3upscale guide LoRA gets its
guide inside those steps."""
from .tiled_upsampler import register_folder as _register_folder
from .video_upsampler import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

_register_folder()

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
