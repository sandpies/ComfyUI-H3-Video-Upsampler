# Generated from ComfyUI-Hand-Tie-Clips (https://github.com/sandpies/ComfyUI-Hand-Tie-Clips) by tools/build_upsampler_pack.py.
# Edit it there, not here: the next build overwrites this file.
"""H3 Video Upsampler -- a finished clip in, a bigger one out, in one node.

The chain's upsampler (tiled_upsampler.upsample) needs an H3 AV latent and the
conditioning that made it, which only exist inside a chain run. This node
starts from what a finished clip actually is -- frames and a soundtrack -- and
does the rest itself:

1. fits the frames onto the H3 grid (both edges to 32 px) and pads them, by
   repeating the last frame, to the next valid H3 length (17g + 5);
2. encodes them with the video VAE, and the soundtrack with the audio VAE when
   both are wired (otherwise a silent stream: the audio is frozen in the
   re-draw either way, it only conditions it);
3. encodes the prompt as Ref2VA conditioning with no references;
4. runs the same upsampler the chain does -- latent upscale per chunk, one
   light re-draw in tiles. A LoRA whose header makes it an h3upscale guide
   LoRA gets its guide inside that same step, exactly as on the chain node;
5. decodes, trims the padding off, and hands back the frames with the
   original soundtrack untouched.

`tile_tokens` is the one thing the chain's recipe fixes that a smaller card
needs to move: the largest sequence one tile forward may carry. Lower means
more, smaller tiles -- less VRAM, more forwards, the same feathered merge.
70,000 is the recipe's value.
"""
from __future__ import annotations

import math

import torch

from . import tiled_upsampler as _tu
from . import upscale_guide as _ug

TAG = "H3VideoUpsampler"
GRID_PX = 32                # H3 canvases are multiples of 32 on both edges
H3_FPS = 24                 # what H3's audio grid assumes per video frame


def valid_length(n):
    """The shortest H3 clip length (17g + 5) that holds n frames."""
    n = max(5, int(n))
    return 17 * math.ceil((n - 5) / 17) + 5


def grid_size(w, h, grid=GRID_PX):
    """(w, h) rounded to the nearest multiple of `grid`, at least one cell."""
    return (max(grid, int(round(w / grid)) * grid),
            max(grid, int(round(h / grid)) * grid))


def fit_frames(images, w, h):
    """[N, H, W, 3] -> [N, h, w, 3], lanczos, only when the size changes."""
    if int(images.shape[2]) == w and int(images.shape[1]) == h:
        return images
    import comfy.utils
    out = torch.empty((int(images.shape[0]), h, w, 3), dtype=images.dtype)
    for s in range(0, int(images.shape[0]), 16):
        x = images[s:s + 16, ..., :3].movedim(-1, 1)
        out[s:s + 16] = comfy.utils.common_upscale(x, w, h, "lanczos", "disabled").movedim(1, -1)
    return out


def pad_frames(images, length):
    """Repeat the last frame up to `length` (the VAE pads a short clip the same way)."""
    n = int(images.shape[0])
    if n >= length:
        return images[:length]
    return torch.cat([images, images[-1:].expand(length - n, -1, -1, -1)], dim=0)


def fit_audio_latent(z, audio_t):
    """An encoded soundtrack trimmed or zero-padded to the clip's audio steps."""
    have = int(z.shape[-1])
    if have >= audio_t:
        return z[..., :audio_t]
    pad = torch.zeros(z.shape[:-1] + (audio_t - have,), dtype=z.dtype, device=z.device)
    return torch.cat([z, pad], dim=-1)


def silent_audio(n_frames, fps, sample_rate=44100):
    return {"waveform": torch.zeros((1, 2, int(round(n_frames / float(fps) * sample_rate)))),
            "sample_rate": sample_rate}


def decode_streamed(decode, vid, n, slice_t=30, warm_t=5, tail_t=2):
    """The first `n` frames of `vid`, decoded as upsample_chain decodes: 30-token
    slices, each opened on a 5-token (17-frame) warm-up it then drops and read
    2 tokens past its end, so the full clip is never one tensor in RAM."""
    tv = int(vid.shape[2])
    if tv <= slice_t + warm_t:
        img = decode(vid).float().cpu()
        return img.reshape(-1, img.shape[-3], img.shape[-2], img.shape[-1])[:n]
    out, at = None, 0
    for k0 in range(0, tv, slice_t):
        k1 = min(tv, k0 + slice_t)
        a = max(0, k0 - warm_t)
        b = min(tv, k1 + tail_t)
        part = decode(vid[:, :, a:b].contiguous()).float().cpu()
        part = part.reshape(-1, part.shape[-3], part.shape[-2], part.shape[-1])
        head = _tu.frames_in(k0) - _tu.frames_in(a)
        keep = min(_tu.frames_in(k1) - _tu.frames_in(k0), n - at)
        if keep <= 0:
            break
        if out is None:
            out = torch.empty((n, int(part.shape[1]), int(part.shape[2]), 3), dtype=torch.float32)
        out[at:at + keep] = part[head:head + keep]
        at += keep
        del part
    return out[:at]


class HTCVideoUpsampler:
    """A finished H3 (or any) clip upsampled with the chain's upsampler."""

    @classmethod
    def INPUT_TYPES(cls):
        import comfy.samplers
        import folder_paths
        return {
            "required": {
                "model": ("MODEL", {"tooltip": (
                    "The H3 model for the re-draw. With an h3upscale LoRA, wire the plain "
                    "UNETLoader output: the LoRA was trained on the bare DiT.")}),
                "clip": ("CLIP", {"tooltip": "The H3 text encoder, for the prompt."}),
                "vae": ("VAE", {"tooltip": "The H3 video VAE."}),
                "images": ("IMAGE", {"tooltip": (
                    "The clip's frames (Get Video Components). Any size; fitted to a "
                    "multiple of 32 px.")}),
                "prompt": ("STRING", {
                    "default": _ug.CAPTION, "multiline": True,
                    "tooltip": ("What the re-draw follows. The default is the caption the "
                                "h3upscale LoRA was trained on; keep it with that LoRA.")}),
                "upscale_model": (_tu.choices(), {
                    "tooltip": f"From models/{_tu.FOLDER}. The recipe uses {_tu.UPSCALER_FILE}."}),
                "megapixels": ("FLOAT", {
                    "default": _tu.MEGAPIXELS, "min": 0.1, "max": 16.0, "step": 0.05,
                    "tooltip": "Output size as area, aspect kept (1 MP = 1024 x 1024)."}),
                "denoise": ("FLOAT", {
                    "default": 0.1, "min": 0.0, "max": 1.0, "step": 0.005,
                    "tooltip": ("Fraction of the schedule the two re-draw steps cover, as the "
                                "chain's upsample_denoise. 0.1 is the setting that held; 0 only "
                                "enlarges.")}),
                "lora": (["None"] + folder_paths.get_filename_list("loras"), {
                    "default": "None",
                    "tooltip": ("A LoRA on the re-draw. An h3upscale LoRA (read off its header) "
                                "also gets its guide: the clip at half the output size, so the "
                                "re-draw restores the clip's own detail instead of inventing it.")}),
                "lora_strength": ("FLOAT", {"default": 1.0, "min": -4.0, "max": 4.0, "step": 0.05}),
                "tile_tokens": ("INT", {
                    "default": _tu.TOKEN_BUDGET, "min": 8192, "max": 1048576, "step": 1024,
                    "tooltip": ("Largest sequence one tile forward may carry. Lower = more, "
                                "smaller tiles: less VRAM, slower. "
                                f"{_tu.TOKEN_BUDGET} is the recipe's value.")}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "euler"}),
            },
            "optional": {
                "audio": ("AUDIO", {"tooltip": "The clip's soundtrack. Passed through untouched."}),
                "audio_vae": ("VAE", {"tooltip": (
                    "The H3 audio VAE. With it, the re-draw is conditioned on the real "
                    "soundtrack (best at 24 fps); without it, on silence.")}),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "STRING")
    RETURN_NAMES = ("images", "audio", "report")
    FUNCTION = "run"
    CATEGORY = "video/upscale"
    DESCRIPTION = (
        "Upsamples a finished clip with the chain's upsampler: H3 latent upscale, then a "
        "light two-step re-draw in tiles. An h3upscale LoRA gets its guide in those steps. "
        "tile_tokens lets a smaller card cut more tiles.")

    def run(self, model, clip, vae, images, prompt, upscale_model, megapixels, denoise,
            lora, lora_strength, tile_tokens, sampler_name, audio=None, audio_vae=None):
        import comfy.nested_tensor as nt
        import comfy.samplers
        import folder_paths
        import nodes
        from comfy_extras.nodes_minimax_h3 import MiniMaxH3ReferenceToVideo, temporal_shape

        _tu.register_folder()
        n = int(images.shape[0])
        if n < 1:
            raise ValueError(f"{TAG}: no frames")
        w, h = grid_size(int(images.shape[2]), int(images.shape[1]))
        length = valid_length(n)
        _, latent_t, audio_t = temporal_shape(length)

        frames = pad_frames(fit_frames(images, w, h), length)
        video = vae.encode(frames[..., :3])
        if int(video.shape[2]) != latent_t:
            raise RuntimeError(f"{TAG}: {length} frames encoded to {int(video.shape[2])} "
                               f"latent steps, expected {latent_t}")
        del frames
        if audio is not None and audio_vae is not None:
            import comfy.audio
            wav = audio["waveform"]
            vsr = getattr(audio_vae, "audio_sample_rate", 44100)
            if int(audio["sample_rate"]) != vsr:
                wav = comfy.audio.resample(wav, int(audio["sample_rate"]), vsr)
            a_lat = fit_audio_latent(audio_vae.encode(wav[:1].movedim(1, -1)), audio_t)
            audio_note = "conditioned on the soundtrack"
        else:
            a_lat = torch.zeros((1, 32, 2, audio_t), dtype=video.dtype)
            audio_note = "conditioned on silence (no audio_vae + audio)"
        latent = {"samples": nt.NestedTensor((video, a_lat.to(video.device, video.dtype)))}

        got = MiniMaxH3ReferenceToVideo.execute(
            clip=clip, prompt=str(prompt), width=w, height=h, length=length,
            ref_image_size="match")
        cond = (got.args if hasattr(got, "args") else tuple(got))[0]
        try:
            import comfy.model_management as _mm
            if hasattr(clip, "patcher") and hasattr(clip.patcher, "model"):
                clip.patcher.model.to(_mm.text_encoder_offload_device())
                _mm.soft_empty_cache()
        except Exception:
            pass

        factor, guide = 0, None
        if str(lora) != "None" and float(lora_strength) != 0.0:
            path = folder_paths.get_full_path_or_raise("loras", str(lora))
            factor = _ug.lora_factor(path)
            model = nodes.LoraLoaderModelOnly().load_lora_model_only(
                model, str(lora), float(lora_strength))[0]
        # As upsample_chain: port dense LoRA AdaLN patches (pruned DiTs).
        fix = getattr(nodes, "NODE_CLASS_MAPPINGS", {}).get("H3AdaLNLoRAFix")
        if fix is not None and hasattr(model, "patches"):
            try:
                res = fix.execute(model=model, mode="port")
                model = (res.args if hasattr(res, "args") else tuple(res))[0]
            except Exception:
                pass
        if factor and float(denoise) > 0.0:
            _ug.install()

            def decode(v):
                img = vae.decode(v)
                return img.reshape(-1, img.shape[-3], img.shape[-2], img.shape[-1])
            guide = _tu.Guide(factor=factor, cond=cond, decode=decode,
                              encode=vae.encode, pixels=_ug.guide_pixels)

        sampler = comfy.samplers.sampler_object(str(sampler_name))
        print(f"[{TAG}] {n} frames at {int(images.shape[2])}x{int(images.shape[1])} -> H3 "
              f"canvas {w}x{h}, {length} frames ({latent_t} latent steps), {audio_note}; "
              f"tile budget {int(tile_tokens)} tokens"
              + (f"; guided by {lora} (factor {factor})" if guide is not None else
                 f"; re-draw LoRA {lora}" if str(lora) != "None" else ""), flush=True)
        # The chain's re-draw, exactly as upsample_chain calls it.
        steps = 2 if float(denoise) > 0.0 else _tu.STEPS
        out = _tu.upsample(model, cond, latent, sampler, str(upscale_model),
                           float(megapixels), float(denoise), guide=guide,
                           budget=int(tile_tokens), steps=steps, start_sigma=0.0,
                           overlap_px=192, detail_sigma=0.0)
        _tu._free(model)
        if _tu._device().type == "cuda":
            _tu._free_all_gpu()
        img = decode_streamed(vae.decode, out["samples"].tensors[0], n)
        del out
        report = (f"{n} frames {int(images.shape[2])}x{int(images.shape[1])} -> "
                  f"{int(img.shape[2])}x{int(img.shape[1])}"
                  + (f", guided ({lora})" if guide is not None else "")
                  + f", {steps} step(s) at denoise {float(denoise):g}"
                  + f", tile budget {int(tile_tokens)}, overlap 192px")
        print(f"[{TAG}] {report}", flush=True)
        return (img.float().cpu(), audio if audio is not None else silent_audio(n, H3_FPS),
                report)


NODE_CLASS_MAPPINGS = {"H3VideoUpsampler": HTCVideoUpsampler}
NODE_DISPLAY_NAME_MAPPINGS = {"H3VideoUpsampler": "H3 Video Upsampler"}
