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
3. builds zeroed-out conditioning (no prompt, no references), or takes the
   `conditioning` wired in -- a text encode of what the clip shows, typed or
   written by a VLM -- with its references and keyframes stripped, as the
   chain's regen2k_cond=hop_text does;
4. runs the same Open-H3-Regenerate-2K upsampler the chain does -- 3D latent
   upscale per chunk, then 3 Euler steps in tiles (noise="per_chunk", seed 42);
5. decodes, trims the padding off, and hands back the frames with the
   original soundtrack untouched.

`structure_lock` holds the re-draw's large shapes to the source's at every
step (tiled_upsampler.structure_lock_fn): 0 is the recipe as it was.
`lock_sigma` is what counts as large, in latent cells (16 px each).

H3 puts every frame 1/24 s after the last, and the soundtrack under them on
that clock. A 48 fps clip of 60 s is 2880 frames, which H3 reads as 120 s:
the real 60 s of audio would sit under the first half, twice too fast, and
the second half would hear silence -- the re-draw then pulls the mouths
toward the wrong syllables. With `fps` wired and not 24, `off_24_audio`
decides what the re-draw hears instead (the soundtrack that comes OUT is
always the original, untouched): `retime` slows or speeds the soundtrack by
fps / 24 so each syllable sits under its own frame (pitch moves with it),
`silence` conditions on silence and leaves the mouths to the frames (and to
structure_lock), `as_is` is the old behaviour.

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
REGEN2K_DEFAULT_LORA = "open_h3_regenerate_2k_r256pp_bf16.safetensors"
REGEN2K_DEFAULT_STRENGTH = 0.8
REGEN2K_SEED = 42
REGEN2K_TEXT_DIM = 5120
OFF_24_AUDIO = ("retime", "silence", "as_is")


def zero_out_cond():
    """Zeroed-out 1-token conditioning (equivalent to ConditioningZeroOut on an
    empty text encode), with no prompt and no references."""
    return [[torch.zeros((1, 1, REGEN2K_TEXT_DIM), dtype=torch.float32),
             {"pooled_output": None,
              "minimax_token_tags": torch.ones(1, dtype=torch.long)}]]


def text_only_cond(cond):
    """A wired conditioning with only its text kept: references, keyframes and
    their noise-aug settings dropped (the chain's regen2k_cond=hop_text)."""
    out = []
    for emb, d in cond:
        d = dict(d)
        for k in ("minimax_refs", "minimax_keyframes",
                  "minimax_visual_cond_noise_aug", "minimax_audio_cond_noise_aug"):
            d.pop(k, None)
        out.append([emb, d])
    return out


def retime_audio(wav, sample_rate, out_rate, fps):
    """`wav` at `sample_rate` -> `out_rate`, played fps / 24 times as long, so
    second t of the clip lands where H3 puts frame t * fps (at t * fps / 24 s)."""
    import comfy.audio
    stretched = int(round(float(out_rate) * float(fps) / H3_FPS))
    return comfy.audio.resample(wav, int(sample_rate), max(1, stretched))


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
        loras = [c for c in folder_paths.get_filename_list("loras") if c != "None"]
        if REGEN2K_DEFAULT_LORA not in loras:
            loras = [REGEN2K_DEFAULT_LORA] + loras
        loras.append("None")
        return {
            "required": {
                "model": ("MODEL", {"tooltip": (
                    "The H3 model for the re-draw: wire UNETLoader with "
                    "minimax_h3_fl2va_pruned_int8_convrot.safetensors.")}),
                "vae": ("VAE", {"tooltip": "The H3 video VAE."}),
                "images": ("IMAGE", {"tooltip": (
                    "The clip's frames (Get Video Components). Any size; fitted to a "
                    "multiple of 32 px.")}),
                "upscale_model": (_tu.choices(), {
                    "tooltip": f"From models/{_tu.FOLDER}. Uses {_tu.UPSCALER_FILE}."}),
                "megapixels": ("FLOAT", {
                    "default": _tu.MEGAPIXELS, "min": 0.1, "max": 16.0, "step": 0.05,
                    "tooltip": "Output size as area, aspect kept (1 MP = 1024 x 1024)."}),
                "denoise": ("FLOAT", {
                    "default": REGEN2K_DEFAULT_STRENGTH, "min": 0.0, "max": 1.0, "step": 0.005,
                    "tooltip": ("Regen2K LoRA strength for the 3-step Euler re-draw "
                                "(0.8 is the default; 0 skips the re-draw and only enlarges).")}),
                "lora": (loras, {
                    "default": REGEN2K_DEFAULT_LORA,
                    "tooltip": ("Open-H3-Regenerate-2K LoRA from models/loras "
                                f"(default: {REGEN2K_DEFAULT_LORA}).")}),
                "tile_tokens": ("INT", {
                    "default": _tu.TOKEN_BUDGET, "min": 8192, "max": 1048576, "step": 1024,
                    "tooltip": ("Largest sequence one tile forward may carry. Lower = shorter "
                                "chunks (down to 68 f) then more, smaller tiles: less VRAM, "
                                f"slower. {_tu.TOKEN_BUDGET} is the default.")}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "euler"}),
                # Not named seed: ComfyUI hangs a randomize control on any
                # widget called seed, and the default 42 should stay fixed.
                "redraw_seed": ("INT", {
                    "default": REGEN2K_SEED, "min": 0, "max": 0x7FFFFFFF,
                    "tooltip": "Noise seed for the Regen2K re-draw (default: 42)."}),
                "structure_lock": ("FLOAT", {
                    "default": _tu.STRUCTURE_LOCK, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": ("0 = off. Holds the re-draw's large shapes (faces, hands, "
                                "line work) to the source clip at every step, so chunks and "
                                "tiles stop inventing and agree where they overlap; the "
                                "re-draw still adds the fine detail. Try 0.5-1, higher for "
                                "animation.")}),
                "lock_sigma": ("FLOAT", {
                    "default": _tu.LOCK_SIGMA, "min": 0.5, "max": 16.0, "step": 0.25,
                    "tooltip": ("What structure_lock counts as 'large', in latent cells (16 px "
                                "each; 2 = 32 px). Lower (1-1.5) also holds eyes and mouths on "
                                "medium shots, but leaves the re-draw less to sharpen.")}),
                "off_24_audio": (list(OFF_24_AUDIO), {
                    "default": OFF_24_AUDIO[0],
                    "tooltip": ("Only with fps wired and not 24: what the re-draw hears. retime = "
                                "the soundtrack slowed/sped by fps/24 so it lines up with the "
                                "frames H3 sees at 24 fps; silence = no sound, the mouths come "
                                "from the frames; as_is = the old behaviour (out of sync). The "
                                "output soundtrack is always the original.")}),
            },
            "optional": {
                "conditioning": ("CONDITIONING", {"tooltip": (
                    "What the clip shows, from the H3 text encoder (CLIPTextEncode), typed "
                    "or written by a VLM. Unwired: zeroed-out conditioning. References and "
                    "keyframes in it are dropped. Every tile hears the whole prompt, so "
                    "describe the look (\"2D anime, flat cel shading, clean line art\") "
                    "more than the objects.")}),
                "audio": ("AUDIO", {"tooltip": "The clip's soundtrack. Passed through untouched."}),
                "fps": ("FLOAT", {"forceInput": True, "tooltip": (
                    "The clip's frame rate (Get Video Components fps). Unwired: taken as 24. "
                    "H3 works on a 24 fps clock; see off_24_audio. For a 48 fps clip made by "
                    "interpolating a 24 fps render, upscaling the 24 fps frames and using H3 "
                    "Save Video's frame_gen x2 is faster and keeps sync exactly.")}),
                "audio_vae": ("VAE", {"tooltip": (
                    "The H3 audio VAE. With it, the re-draw pins the real "
                    "soundtrack (best at 24 fps); without it, silence.")}),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "STRING")
    RETURN_NAMES = ("images", "audio", "report")
    FUNCTION = "run"
    CATEGORY = "video/upscale"
    DESCRIPTION = (
        "Upsamples a finished clip with Open-H3-Regenerate-2K: H3 3D latent upscale, then "
        "3 Euler steps on [0.8, 24/35, 0.48, 0] with zeroed-out conditioning (or the wired "
        "conditioning's text) and the soundtrack pinned clean (noise=per_chunk, seed 42). "
        "structure_lock holds the re-draw to the source's large shapes; tile_tokens lets a "
        "smaller card shorten chunks or cut more tiles.")

    def run(self, model, vae, images, upscale_model, megapixels, denoise,
            lora, tile_tokens, sampler_name, redraw_seed=REGEN2K_SEED,
            structure_lock=0.0, lock_sigma=_tu.LOCK_SIGMA, off_24_audio=OFF_24_AUDIO[0],
            audio=None, audio_vae=None, conditioning=None, fps=None):
        import comfy.nested_tensor as nt
        import comfy.samplers
        import folder_paths
        import nodes
        from comfy_extras.nodes_minimax_h3 import temporal_shape

        _tu.register_folder()
        _fsm = getattr(vae, "first_stage_model", None)
        if (_fsm is not None and type(_fsm).__name__ == "MiniMaxH3VideoVAE"
                and not hasattr(_fsm, "decoder")):
            raise RuntimeError(
                f"{TAG}: the wired Video VAE has no decoder (encoder-only, 343 MB — "
                "e.g. minimax_h3_sol_vae_fp16.safetensors uses decoder.blocks.* keys "
                "which ComfyUI's stock VAELoader skips). Pick "
                "minimax_h3_lynnreal_light_vae_int8_convrot.safetensors or "
                "minimax_h3_video_vae_int8_convrot.safetensors in VAELoader.")
        r2k_sigmas = _tu.REGEN2K_SIGMAS
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
        clip_fps = float(fps) if fps else float(H3_FPS)
        off_24 = abs(clip_fps - H3_FPS) > 1e-3
        mode = str(off_24_audio or OFF_24_AUDIO[0]) if off_24 else "as_is"
        if off_24:
            print(f"[{TAG}] WARNING: {clip_fps:g} fps clip; H3 reads every frame as 1/{H3_FPS} s, "
                  f"so {n} frames span {n / H3_FPS:.1f} s on its clock vs {n / clip_fps:.1f} s "
                  f"of soundtrack. off_24_audio={mode}.", flush=True)
        if audio is not None and audio_vae is not None and mode != "silence":
            import comfy.audio
            wav = audio["waveform"]
            vsr = getattr(audio_vae, "audio_sample_rate", 44100)
            if mode == "retime":
                wav = retime_audio(wav, int(audio["sample_rate"]), vsr, clip_fps)
            elif int(audio["sample_rate"]) != vsr:
                wav = comfy.audio.resample(wav, int(audio["sample_rate"]), vsr)
            a_lat = fit_audio_latent(audio_vae.encode(wav[:1].movedim(1, -1)), audio_t)
            audio_note = "conditioned on the soundtrack" + (
                f" retimed x{clip_fps / H3_FPS:g} ({clip_fps:g} fps)" if mode == "retime"
                else f" out of sync ({clip_fps:g} fps)" if off_24 else "")
        elif off_24 and mode == "silence" and audio is not None and audio_vae is not None:
            a_lat = torch.zeros((1, 32, 2, audio_t), dtype=video.dtype)
            audio_note = f"conditioned on silence (off_24_audio=silence, {clip_fps:g} fps)"
        else:
            a_lat = torch.zeros((1, 32, 2, audio_t), dtype=video.dtype)
            audio_note = "conditioned on silence (no audio_vae + audio)"
        latent = {"samples": nt.NestedTensor((video, a_lat.to(video.device, video.dtype)))}

        if conditioning:
            cond = text_only_cond(conditioning)
            cond_note = "wired conditioning (text only)"
        else:
            cond = zero_out_cond()
            cond_note = "zeroed-out conditioning"
        lock = max(0.0, min(1.0, float(structure_lock or 0.0)))
        lock_sig = max(0.5, float(lock_sigma or _tu.LOCK_SIGMA))
        model = model.clone() if hasattr(model, "clone") else model
        eff_lora = str(lora or "").strip()
        if not eff_lora or eff_lora == "None":
            eff_lora = None if getattr(model, "patches", None) else REGEN2K_DEFAULT_LORA
        eff_strength = float(denoise)
        if eff_strength <= 0.0:
            lora_tag = "denoise 0 (3D upscale only)"
        elif eff_lora is None:
            lora_tag = f"pre-patched ({len(model.patches)} targets)"
        else:
            folder_paths.get_full_path_or_raise("loras", eff_lora)
            model = nodes.LoraLoaderModelOnly().load_lora_model_only(
                model, eff_lora, eff_strength)[0]
            lora_tag = f"{eff_lora} @ {eff_strength:g}"
        # As upsample_chain: port dense LoRA AdaLN patches (pruned DiTs).
        fix = getattr(nodes, "NODE_CLASS_MAPPINGS", {}).get("H3AdaLNLoRAFix")
        if fix is not None and hasattr(model, "patches"):
            try:
                res = fix.execute(model=model, mode="port")
                model = (res.args if hasattr(res, "args") else tuple(res))[0]
            except Exception:
                pass

        sampler = comfy.samplers.sampler_object(str(sampler_name))
        sig_list = [round(float(s), 4) for s in r2k_sigmas]
        print(f"[{TAG}] {n} frames at {int(images.shape[2])}x{int(images.shape[1])} -> H3 "
              f"canvas {w}x{h}, {length} frames ({latent_t} latent steps), {audio_note}, "
              f"{cond_note}, structure lock {lock:g} at {lock_sig:g} cells; "
              f"tile budget {int(tile_tokens)} tokens; regen2k ({lora_tag}, "
              f"sigmas {sig_list})", flush=True)
        _cd = _tu.CHAIN_DEFAULTS
        overlap_px = int(_cd["tile_overlap_px"])
        if eff_strength <= 0.0:
            steps = 0
            out = _tu.upsample(model, cond, latent, sampler, str(upscale_model),
                               float(megapixels), 0.0, guide=None,
                               budget=int(tile_tokens), steps=1, start_sigma=0.0,
                               overlap_px=overlap_px, detail_sigma=0.0)
        else:
            steps = len(r2k_sigmas) - 1
            out = _tu.upsample(model, cond, latent, sampler, str(upscale_model),
                               float(megapixels), 1.0, guide=None,
                               budget=int(tile_tokens), steps=steps, start_sigma=0.0,
                               overlap_px=overlap_px, detail_sigma=0.0,
                               seed=int(redraw_seed), noise="per_chunk",
                               sigmas=r2k_sigmas, structure_lock=lock, lock_sigma=lock_sig)
        _tu._free(model)
        if _tu._device().type == "cuda":
            _tu._free_all_gpu()
        img = decode_streamed(vae.decode, out["samples"].tensors[0], n)
        del out
        report = (f"{n} frames {int(images.shape[2])}x{int(images.shape[1])} -> "
                  f"{int(img.shape[2])}x{int(img.shape[1])}"
                  f", regen2k ({lora_tag}, {steps} steps, {cond_note}"
                  f", structure lock {lock:g} at {lock_sig:g} cells)"
                  f", {audio_note}"
                  f", tile budget {int(tile_tokens)}, overlap {overlap_px}px")
        print(f"[{TAG}] {report}", flush=True)
        return (img.float().cpu(), audio if audio is not None else silent_audio(n, H3_FPS),
                report)


NODE_CLASS_MAPPINGS = {"H3VideoUpsampler": HTCVideoUpsampler}
NODE_DISPLAY_NAME_MAPPINGS = {"H3VideoUpsampler": "H3 Video Upsampler"}
