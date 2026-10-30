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
import os

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
    for part in decode_slices(decode, vid, n, slice_t, warm_t, tail_t):
        if out is None:
            out = torch.empty((n, int(part.shape[1]), int(part.shape[2]), 3), dtype=torch.float32)
        out[at:at + int(part.shape[0])] = part
        at += int(part.shape[0])
    return out[:at]


def decode_slices(decode, vid, n, slice_t=30, warm_t=5, tail_t=2):
    """`decode_streamed`'s slices, yielded as they are decoded ([k, H, W, 3]
    float32 on the CPU, `n` frames in all) for a consumer that writes them out
    and lets them go."""
    tv = int(vid.shape[2])
    if tv <= slice_t + warm_t:
        img = decode(vid).float().cpu()
        yield img.reshape(-1, img.shape[-3], img.shape[-2], img.shape[-1])[:n]
        return
    at = 0
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
        yield part[head:head + keep]
        at += keep
        del part


# --- the file in, the file out ----------------------------------------------
#
# The IMAGE node above is bounded by ComfyUI's IMAGE type: Get Video Components
# hands it the whole clip as one float32 tensor (stacked from a list, so twice
# for a moment), it fits and pads a copy, and it hands back the whole result as
# another -- a minute at 48 fps is ~32 GB in and ~72 GB out at 2 MP. The redraw
# in between only ever needs the LATENT, which is tiny (a minute is ~100 MB).
# So the file node reads the video a chunk at a time and encodes as it goes
# (the H3 VAE encodes every 17 frames on their own, so 102-frame pieces give
# the very latent the whole clip does), redraws exactly as the IMAGE node does,
# and decodes slice by slice straight into H3 Save Video's encoder.
#
# Three ideas here come from silveroxides' ComfyUI-UtilsCollection (its H3
# reference-video path), written fresh for this pack -- no code is shared:
# decoding a VIDEO in batches with core's own semantics instead of assembling
# the clip, keying decoded media to the file's content so a rerun skips the
# decode, and resampling to H3's 24 fps by nearest frame on read.

READ_FRAMES = 102           # frames read and encoded at a time: 6 VAE clips of 17
FRAME_RATES = ("source", "24 (H3 native)")
LATENT_CACHE_DIR = "htc_upsampler_latents"   # under ComfyUI's temp dir (cleared at start)
LATENT_CACHE_KEEP = 4       # encoded clips kept on disk, newest first


def _video_file(video):
    """(source, start, duration, crop) when frames can be read from the VIDEO's
    file directly, else None (components or a list: read whole instead). Core
    has no public accessor for a crop, so it is read off the instance."""
    if type(video).__name__ != "VideoFromFile" or not hasattr(video, "get_stream_source"):
        return None
    start, duration = video.get_active_trim_window()
    crop = getattr(video, "_VideoFromFile__crop", None)
    return video.get_stream_source(), float(start), float(duration), crop


def crop_rect(crop, width, height):
    """A VIDEO crop (x, y, w, h) fitted to a width x height frame the way core
    fits it: inside the frame, on even pixels, None when it is the whole frame
    or empty."""
    x, y, w, h = (int(v) for v in crop)
    if w <= 0 or h <= 0:
        return None
    x = max(0, min(x, width - 1)) // 2 * 2
    y = max(0, min(y, height - 1)) // 2 * 2
    w, h = min(w, width - x), min(h, height - y)
    if (x, y, w, h) == (0, 0, width, height):
        return None
    w, h = w // 2 * 2, h // 2 * 2
    return (x, y, w, h) if w > 0 and h > 0 else None


def nearest_frames(frames, src_fps, out_fps):
    """`frames` (any per-frame iterable at `src_fps`) re-timed to `out_fps`:
    output frame j shows the source frame nearest to its time j / out_fps, so
    frames are dropped above the rate and repeated below it and the clip keeps
    its real duration (and its soundtrack's sync)."""
    ratio = float(src_fps) / float(out_fps)
    j = 0
    for i, frame in enumerate(frames):
        while math.floor(j * ratio + 0.5) == i:
            yield frame
            j += 1


def decode_frames(source, start, duration, crop=None):
    """The file's frames one at a time as HxWx3 arrays (uint8, or float32 0..1
    for sources deeper than 8 bits), with the trim window, rotation and crop
    applied as ComfyUI's own loader applies them."""
    import av
    import numpy as np
    try:
        from comfy_api.latest._input_impl.video_types import _rotation_quadrant
    except Exception:  # noqa: BLE001
        _rotation_quadrant = None
    with av.open(source, mode="r") as container:
        vs = container.streams.video[0]
        vs.thread_type = "AUTO"
        start_pts = int(start / vs.time_base)
        end_pts = int((start + duration) / vs.time_base)
        if start_pts:
            container.seek(start_pts, stream=vs)
        fmt, rect = None, False
        for frame in container.decode(vs):
            if frame.pts is not None and frame.pts < start_pts:
                continue
            if duration and frame.pts is not None and frame.pts >= end_pts:
                break
            if fmt is None:
                deep = any(int(c.bits) > 8 for c in frame.format.components)
                fmt = "rgb48le" if deep else "rgb24"
            img = frame.to_ndarray(format=fmt)
            img = img.astype(np.float32) / 65535.0 if fmt == "rgb48le" else img
            q = _rotation_quadrant(frame) if _rotation_quadrant is not None else 0
            if q:
                img = np.rot90(img, k=q, axes=(0, 1))
            if rect is False:
                rect = crop_rect(crop, img.shape[1], img.shape[0]) if crop is not None else None
            if rect is not None:
                x, y, w, h = rect
                img = img[y:y + h, x:x + w]
            yield np.ascontiguousarray(img)


def read_frames(source, start, duration, k, crop=None, src_fps=None, out_fps=None):
    """[<=k, H, W, 3] float32 batches, 0..1, decoded from the file in order;
    re-timed to `out_fps` by nearest frame when both rates are given."""
    frames = decode_frames(source, start, duration, crop)
    if src_fps and out_fps and abs(float(src_fps) - float(out_fps)) > 1e-3:
        frames = nearest_frames(frames, src_fps, out_fps)
    buf = []
    for img in frames:
        buf.append(torch.from_numpy(img))
        if len(buf) == k:
            yield _to_float(torch.stack(buf))
            buf = []
    if buf:
        yield _to_float(torch.stack(buf))


def _to_float(x):
    return x.float().div_(255.0) if x.dtype == torch.uint8 else x


def decodable_audio(container):
    """The last audio stream FFmpeg can decode, or None. A stream without a
    codec context (an iPhone's spatial-audio APAC track) crashes the process
    when decoded, so it is never picked."""
    return next((a for a in reversed(container.streams.audio)
                 if a.codec_context is not None), None)


def read_audio(source, start, duration):
    """The file's soundtrack as an AUDIO dict (trimmed like the frames), or None."""
    import av
    import numpy as np
    try:
        with av.open(source, mode="r") as container:
            ast = decodable_audio(container)
            if ast is None:
                return None
            sr = int(ast.sample_rate or 0)
            if sr <= 0:
                return None
            resampler = av.audio.resampler.AudioResampler(format="fltp")
            chunks, t0 = [], None
            for frame in container.decode(ast):
                for rf in resampler.resample(frame):
                    if t0 is None:
                        t0 = float(rf.pts * rf.time_base) if rf.pts is not None else 0.0
                    chunks.append(rf.to_ndarray())
            if not chunks:
                return None
            data = np.concatenate(chunks, axis=1)
            s0 = max(0, int(round((float(start) - (t0 or 0.0)) * sr)))
            s1 = s0 + int(round(float(duration) * sr)) if duration else data.shape[1]
            data = np.ascontiguousarray(data[:, s0:s1])
            return {"waveform": torch.from_numpy(data).unsqueeze(0), "sample_rate": sr}
    except Exception as exc:  # noqa: BLE001
        print(f"[{TAG}] soundtrack not read ({exc!r}); conditioning on silence", flush=True)
        return None


def file_fingerprint(source):
    """What identifies the file's content, cheaply: path, size, mtime and the
    first and last MiB (a path), or every byte (an in-memory upload)."""
    import hashlib
    import os
    h = hashlib.sha256()
    if isinstance(source, (str, os.PathLike)):
        st = os.stat(source)
        h.update(f"{os.path.abspath(source)}|{st.st_size}|{st.st_mtime_ns}".encode())
        with open(source, "rb") as f:
            h.update(f.read(1 << 20))
            if st.st_size > 2 << 20:
                f.seek(-(1 << 20), os.SEEK_END)
                h.update(f.read(1 << 20))
    else:
        source.seek(0)
        h.update(source.read())
        source.seek(0)
    return h.hexdigest()


def vae_fingerprint(vae):
    """Which VAE weights encoded a latent: class, parameter count and the sums
    of the first and last parameter (a different file differs in all three).
    None when the weights cannot be read (the cache is then skipped)."""
    fsm = getattr(vae, "first_stage_model", vae)
    try:
        params = list(fsm.parameters()) if hasattr(fsm, "parameters") else []
        if not params:
            return type(fsm).__name__
        total = sum(int(p.numel()) for p in params)
        ends = [float(p.detach().float().sum()) for p in (params[0], params[-1])]
    except Exception:  # noqa: BLE001 -- quantized or offloaded weights
        return None
    return f"{type(fsm).__name__}|{total}|{ends[0]:.6e}|{ends[1]:.6e}"


def _cache_dir():
    import os
    import folder_paths
    return os.path.join(folder_paths.get_temp_directory(), LATENT_CACHE_DIR)


def cache_load(key):
    """The encoded clip stored under `key`, or None."""
    import os
    path = os.path.join(_cache_dir(), key + ".pt")
    if not os.path.isfile(path):
        return None
    try:
        got = torch.load(path, map_location="cpu", weights_only=True)
        os.utime(path)          # newest-first pruning keeps what is in use
        return got
    except Exception as exc:  # noqa: BLE001 -- a broken entry is just a miss
        print(f"[{TAG}] latent cache entry unreadable ({exc!r}); re-encoding", flush=True)
        return None


def cache_store(key, entry):
    """Store an encoded clip, keeping the newest LATENT_CACHE_KEEP."""
    import os
    d = _cache_dir()
    try:
        os.makedirs(d, exist_ok=True)
        tmp = os.path.join(d, key + ".tmp")
        torch.save(entry, tmp)
        os.replace(tmp, os.path.join(d, key + ".pt"))
        files = sorted((os.path.join(d, f) for f in os.listdir(d) if f.endswith(".pt")),
                       key=os.path.getmtime, reverse=True)
        for old in files[LATENT_CACHE_KEEP:]:
            os.remove(old)
    except OSError as exc:
        print(f"[{TAG}] latent cache not written ({exc!r})", flush=True)


def encode_video_streamed(vae, batches, seg=READ_FRAMES):
    """Encode frame batches (`seg` frames each, the last one shorter) as they
    arrive. -> (video latent of valid_length(n) frames, n, (src_w, src_h)).

    Every piece but the last is a whole number of 17-frame VAE clips, so
    `encode_steps` gives exactly its share of the whole clip's latent; the last
    is padded with its final frame to the clip's 17g + 5 length, as the IMAGE
    node pads the whole clip."""
    import comfy.model_management as mm
    from comfy_extras.nodes_minimax_h3 import temporal_shape
    if seg % 17:
        raise ValueError(f"{TAG}: read size {seg} is not a whole number of 17-frame clips")
    it = iter(batches)
    cur = next(it, None)
    if cur is None or int(cur.shape[0]) == 0:
        raise ValueError(f"{TAG}: no frames")
    src_w, src_h = int(cur.shape[2]), int(cur.shape[1])
    w, h = grid_size(src_w, src_h)
    lats, n, steps_done = [], 0, 0
    while cur is not None:
        mm.throw_exception_if_processing_interrupted()
        nxt = next(it, None)
        frames = fit_frames(cur[..., :3], w, h)
        before, n = n, n + int(cur.shape[0])
        if nxt is None:
            length = valid_length(n)
            _, latent_t, _ = temporal_shape(length)
            frames = pad_frames(frames, length - before)
            lat = _tu.encode_steps(vae.encode, frames, latent_t - steps_done)
        else:
            if int(cur.shape[0]) != seg:
                raise RuntimeError(f"{TAG}: a {int(cur.shape[0])}-frame read before the end")
            lat = _tu.encode_steps(vae.encode, frames, seg // 17 * 5)
        lats.append(lat.cpu())
        steps_done += int(lat.shape[2])
        del frames, lat
        cur = nxt
    return torch.cat(lats, dim=2), n, (src_w, src_h)


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
                    "default": 0, "min": 0, "max": 1048576, "step": 1024,
                    "tooltip": ("VRAM. 0 = auto: starts at what this GPU learned last time "
                                f"({_tu.TOKEN_BUDGET} the first time), measures every chunk, and "
                                "stores where the next run should start -- grown when there was "
                                "room, lowered for good after running out. A number = that "
                                "fixed budget: the largest sequence one forward may carry; lower "
                                "means shorter chunks (down to 68 f) then tiles.")}),
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
                "lock_anchor": (list(_tu.LOCK_ANCHORS), {
                    "default": "source", "tooltip": _tu.LOCK_ANCHOR_TIP}),
                "off_24_audio": (list(OFF_24_AUDIO), {
                    "default": OFF_24_AUDIO[0],
                    "tooltip": ("Only with fps wired and not 24: what the re-draw hears. retime = "
                                "the soundtrack slowed/sped by fps/24 so it lines up with the "
                                "frames H3 sees at 24 fps; silence = no sound, the mouths come "
                                "from the frames; as_is = the old behaviour (out of sync). The "
                                "output soundtrack is always the original.")}),
                "attention": (list(_tu.ATTENTION_MODES), {
                    "default": "dense", "tooltip": _tu.ATTENTION_TIP}),
                "chunk_join": (list(_tu.JOIN_MODES), {
                    "default": "crossfade", "tooltip": _tu.JOIN_TIP}),
                "chunking": ("BOOLEAN", {"default": True, "tooltip": _tu.CHUNKING_TIP}),
                "tiling": ("BOOLEAN", {"default": True, "tooltip": _tu.TILING_TIP}),
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
            attention="dense", chunk_join="crossfade", chunking=True, tiling=True,
            lock_anchor="source", audio=None, audio_vae=None, conditioning=None, fps=None):
        check_vae(vae)
        n = int(images.shape[0])
        if n < 1:
            raise ValueError(f"{TAG}: no frames")
        src_w, src_h = int(images.shape[2]), int(images.shape[1])
        w, h = grid_size(src_w, src_h)
        length = valid_length(n)
        frames = pad_frames(fit_frames(images, w, h), length)
        video = vae.encode(frames[..., :3])
        del frames
        vid, report = redraw(
            model, video, n, (src_w, src_h), upscale_model, megapixels, denoise, lora,
            tile_tokens, sampler_name, redraw_seed, structure_lock, lock_sigma, off_24_audio,
            audio, audio_vae, conditioning, fps, attention, chunk_join, chunking, tiling,
            lock_anchor)
        img = decode_streamed(vae.decode, vid, n)
        del vid
        return (img.float().cpu(), audio if audio is not None else silent_audio(n, H3_FPS),
                report)


def check_vae(vae):
    """Fail before any work when the wired Video VAE cannot decode."""
    _fsm = getattr(vae, "first_stage_model", None)
    if (_fsm is not None and type(_fsm).__name__ == "MiniMaxH3VideoVAE"
            and not hasattr(_fsm, "decoder")):
        raise RuntimeError(
            f"{TAG}: the wired Video VAE has no decoder (encoder-only, 343 MB — "
            "e.g. minimax_h3_sol_vae_fp16.safetensors uses decoder.blocks.* keys "
            "which ComfyUI's stock VAELoader skips). Pick "
            "minimax_h3_lynnreal_light_vae_int8_convrot.safetensors or "
            "minimax_h3_video_vae_int8_convrot.safetensors in VAELoader.")


def redraw(model, video, n, src_size, upscale_model, megapixels, denoise, lora,
           tile_tokens, sampler_name, redraw_seed, structure_lock, lock_sigma, off_24_audio,
           audio, audio_vae, conditioning, fps, attention="dense", chunk_join="crossfade",
           chunking=True, tiling=True, lock_anchor="source"):
    """Steps 2-4 for a clip already encoded: `video` is the H3 video latent of
    `n` frames padded to `valid_length(n)`. -> (upsampled video latent, report)."""
    import comfy.nested_tensor as nt
    import comfy.samplers
    import folder_paths
    import nodes
    from comfy_extras.nodes_minimax_h3 import temporal_shape

    _tu.register_folder()
    r2k_sigmas = _tu.REGEN2K_SIGMAS
    w, h = int(video.shape[4]) * _tu.SPATIAL, int(video.shape[3]) * _tu.SPATIAL
    length = valid_length(n)
    _, latent_t, audio_t = temporal_shape(length)
    if int(video.shape[2]) != latent_t:
        raise RuntimeError(f"{TAG}: {length} frames encoded to {int(video.shape[2])} "
                           f"latent steps, expected {latent_t}")
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
    print(f"[{TAG}] {n} frames at {src_size[0]}x{src_size[1]} -> H3 "
          f"canvas {w}x{h}, {length} frames ({latent_t} latent steps), {audio_note}, "
          f"{cond_note}, structure lock {lock:g} at {lock_sig:g} cells; "
          f"tile budget {int(tile_tokens) or 'auto'}; regen2k ({lora_tag}, "
          f"sigmas {sig_list})", flush=True)
    _cd = _tu.CHAIN_DEFAULTS
    overlap_px = int(_cd["tile_overlap_px"])
    if eff_strength <= 0.0:
        steps = 0
        out = _tu.upsample(model, cond, latent, sampler, str(upscale_model),
                           float(megapixels), 0.0, guide=None,
                           budget=int(tile_tokens), steps=1, start_sigma=0.0,
                           overlap_px=overlap_px, detail_sigma=0.0, chunking=chunking,
                           tiling=tiling)
    else:
        steps = len(r2k_sigmas) - 1
        out = _tu.upsample(model, cond, latent, sampler, str(upscale_model),
                           float(megapixels), 1.0, guide=None,
                           budget=int(tile_tokens), steps=steps, start_sigma=0.0,
                           overlap_px=overlap_px, detail_sigma=0.0,
                           seed=int(redraw_seed), noise="per_chunk",
                           sigmas=r2k_sigmas, structure_lock=lock, lock_sigma=lock_sig,
                           attention=attention, chunk_join=chunk_join, chunking=chunking,
                           tiling=tiling, lock_anchor=lock_anchor)
    _tu._free(model)
    if _tu._device().type == "cuda":
        _tu._free_all_gpu()
    vid = out["samples"].tensors[0]
    del out
    report = (f"{n} frames {src_size[0]}x{src_size[1]} -> "
              f"{int(vid.shape[4]) * _tu.SPATIAL}x{int(vid.shape[3]) * _tu.SPATIAL}"
              f", regen2k ({lora_tag}, {steps} steps, {cond_note}"
              f", structure lock {lock:g} at {lock_sig:g} cells)"
              f", {audio_note}, attention {attention}, chunk_join {chunk_join}"
              f", chunking {'on' if chunking else 'off'}, tiling {'on' if tiling else 'off'}"
              f", lock anchor {lock_anchor}"
              f", tile budget {int(tile_tokens) or 'auto'}, overlap {overlap_px}px")
    print(f"[{TAG}] {report}", flush=True)
    return vid, report



class HTCVideoUpsamplerToFile:
    """The upsampler from a video file to an MP4, never holding the film's frames."""

    @classmethod
    def INPUT_TYPES(cls):
        base = HTCVideoUpsampler.INPUT_TYPES()
        req = {"video": ("VIDEO", {"tooltip": (
            "The clip (Load Video). Read and encoded a chunk at a time; its frame rate and "
            "soundtrack come from the file.")})}
        for k, v in base["required"].items():
            if k != "images":
                req[k] = v
        try:
            from .save_video import CPU_CODECS, FRAME_GEN, QUALITY
            req["frame_gen"] = (list(FRAME_GEN), {"default": "off", "tooltip": (
                "RTX frame generation while saving: x2 turns 24 fps into 48 (RTX card + "
                "nvidia-vfx).")})
            req["codec"] = (["auto"] + list(CPU_CODECS), {"default": "auto"})
            req["quality"] = (list(QUALITY), {"default": "max"})
        except Exception:  # noqa: BLE001 -- PyAV missing: the node fails at run time
            pass
        req["filename_prefix"] = ("STRING", {"default": "video/H3_upsampled"})
        req["frame_rate"] = (list(FRAME_RATES), {"default": FRAME_RATES[0], "tooltip": (
            "source = every frame of the file. 24 = re-timed to H3's own rate on read "
            "(nearest frame: dropped above 24, repeated below), so the soundtrack lines "
            "up exactly and a 48 fps clip costs half the re-draw. Set frame_gen x2 to "
            "save 48 again.")})
        req["reuse_encoded"] = ("BOOLEAN", {"default": True, "tooltip": (
            "Keep the encoded clip (about 100 MB a minute) in ComfyUI's temp folder, "
            "keyed to the file's content, trim, crop, frame_rate and VAE, so the next "
            "run on the same clip skips reading and encoding. The temp folder is "
            "cleared when ComfyUI starts.")})
        opt = {k: v for k, v in base["optional"].items()
               if k in ("audio_vae", "conditioning")}
        return {"required": req, "optional": opt,
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"}}

    RETURN_TYPES = ("VIDEO", "STRING", "STRING")
    RETURN_NAMES = ("video", "filepath", "report")
    FUNCTION = "run"
    CATEGORY = "video/upscale"
    OUTPUT_NODE = True
    DESCRIPTION = (
        "H3 Video Upsampler from a video file straight to an MP4: the clip is read, "
        "encoded, re-drawn, decoded and saved a chunk at a time, so RAM stays flat however "
        "long the film is. Same re-draw as the IMAGE node; the frame rate and soundtrack "
        "come from the file.")

    def run(self, video, model, vae, upscale_model, megapixels, denoise, lora, tile_tokens,
            sampler_name, redraw_seed=REGEN2K_SEED, structure_lock=0.0,
            lock_sigma=_tu.LOCK_SIGMA, lock_anchor="source", off_24_audio=OFF_24_AUDIO[0],
            attention="dense", chunk_join="crossfade", chunking=True, tiling=True,
            frame_gen="off",
            codec="auto", quality="max", filename_prefix="video/H3_upsampled",
            frame_rate=FRAME_RATES[0], reuse_encoded=True,
            audio_vae=None, conditioning=None, prompt=None, extra_pnginfo=None):
        import hashlib
        import json
        import time
        from . import save_video as _sv

        check_vae(vae)
        t0 = time.perf_counter()
        to_24 = str(frame_rate) != FRAME_RATES[0]
        src = _video_file(video)
        key = None
        if src is not None:
            src_fps = float(video.get_frame_rate())
            source, start, duration, crop = src
            fps = float(H3_FPS) if to_24 else src_fps
            audio = read_audio(source, start, duration)
            vae_id = vae_fingerprint(vae) if reuse_encoded else None
            if vae_id:
                key = hashlib.sha256(json.dumps([
                    1, file_fingerprint(video.get_stream_source()), start, duration,
                    list(crop) if crop is not None else None, fps, READ_FRAMES,
                    vae_id]).encode()).hexdigest()[:40]
            hit = cache_load(key) if key else None
            if hit is not None:
                lat, n, src_size = hit["latent"], int(hit["n"]), tuple(hit["src_size"])
                how = "encoded latent reused (same file, trim, crop, rate and VAE)"
            else:
                batches = read_frames(video.get_stream_source(), start, duration, READ_FRAMES,
                                      crop, src_fps, fps if to_24 else None)
                how = ("read from the file a chunk at a time"
                       + (f", re-timed {src_fps:g} -> {fps:g} fps" if to_24 and abs(src_fps - fps) > 1e-3
                          else ""))
        else:
            comp = video.get_components()
            audio, imgs = comp.audio, comp.images
            src_fps = float(comp.frame_rate)
            fps = float(H3_FPS) if to_24 else src_fps
            if to_24 and abs(src_fps - fps) > 1e-3:
                pick = list(nearest_frames(range(int(imgs.shape[0])), src_fps, fps))
                imgs = imgs[pick]
            batches = (imgs[s:s + READ_FRAMES] for s in range(0, int(imgs.shape[0]), READ_FRAMES))
            how = "not a plain file (components or list): decoded whole"
            hit = None
        if hit is None:
            lat, n, src_size = encode_video_streamed(vae, batches)
            if key:
                cache_store(key, {"latent": lat, "n": n, "src_size": list(src_size)})
        t1 = time.perf_counter()
        print(f"[{TAG}] {n} frames at {fps:g} fps, {how}, in {t1 - t0:.1f} s "
              f"({int(lat.shape[2])} latent steps, {lat.numel() * lat.element_size() / 1048576.0:.0f} MB)",
              flush=True)
        vid, report = redraw(
            model, lat, n, src_size, upscale_model, megapixels, denoise, lora,
            tile_tokens, sampler_name, redraw_seed, structure_lock, lock_sigma, off_24_audio,
            audio, audio_vae, conditioning, fps, attention, chunk_join, chunking, tiling,
            lock_anchor)
        del lat
        out_w, out_h = int(vid.shape[4]) * _tu.SPATIAL, int(vid.shape[3]) * _tu.SPATIAL
        res = _sv.encode_stream(decode_slices(vae.decode, vid, n), n, out_w, out_h, fps,
                                frame_gen, codec, quality, filename_prefix, audio,
                                prompt, extra_pnginfo)
        _video, path, save_report = res["result"]
        report = f"{report}; {save_report}; total {time.perf_counter() - t0:.0f} s"
        print(f"[{TAG}] {report}", flush=True)
        return {"ui": res["ui"], "result": (_video, path, report)}


# --- frames for IMAGE graphs ------------------------------------------------
#
# Core's Load Video is already cheap -- a VIDEO is a reference to the file --
# but Get Video Components decodes every frame to float32 into a list and then
# stacks it (twice the clip for a moment), with no way to take less. This node
# replaces that step: the window, the frame rate and the size are applied while
# decoding, so a dropped frame never reaches RAM; frames are held as 8 bits
# (16-bit floats for deeper sources) and filled into the float32 IMAGE once,
# so the peak is about 1.25x the output instead of 2x or more.

RESIZE_MODES = ("none", "megapixels", "long edge")
FRAME_BATCH = 32            # frames decoded, resized and stored per step


def resize_target(w, h, mode, value, even=2):
    """(w, h) after `resize`: area to `value` megapixels or the long edge to
    `value` px, aspect kept, never enlarged, both edges on a multiple of 2."""
    if mode == "none" or float(value) <= 0.0:
        return w, h
    if mode == "megapixels":
        scale = math.sqrt(float(value) * 1048576.0 / float(w * h))
    else:
        scale = float(value) / float(max(w, h))
    scale = min(1.0, scale)
    return (max(even, int(round(w * scale / even)) * even),
            max(even, int(round(h * scale / even)) * even))


def _store(batch, size):
    """A [k, H, W, 3] uint8 or float32 batch, resized to `size` (w, h) when it
    differs, as compact storage: uint8 stays uint8, deeper frames as float16."""
    w, h = size
    if (int(batch.shape[2]), int(batch.shape[1])) != (w, h):
        x = _to_float(batch).movedim(-1, 1)
        x = torch.nn.functional.interpolate(x, size=(h, w), mode="bicubic",
                                            align_corners=False, antialias=True).clamp_(0, 1)
        x = x.movedim(1, -1)
        return (x * 255.0).round_().to(torch.uint8) if batch.dtype == torch.uint8 else x.half()
    return batch if batch.dtype == torch.uint8 else batch.half()


def frames_to_image(parts):
    """Stored batches -> one float32 IMAGE, each batch freed as it is copied."""
    n = sum(int(p.shape[0]) for p in parts)
    first = parts[0]
    out = torch.empty((n, int(first.shape[1]), int(first.shape[2]), 3), dtype=torch.float32)
    at = 0
    while parts:
        p = parts.pop(0)
        k = int(p.shape[0])
        out[at:at + k] = p.float().div_(255.0) if p.dtype == torch.uint8 else p.float()
        at += k
        del p
    return out


class HTCVideoFrames:
    """Get Video Components, lean: a window, a frame rate and a size, applied
    while decoding."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO", {"tooltip": "From Load Video (or any VIDEO)."}),
                "start_seconds": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 86400.0, "step": 0.01,
                    "tooltip": "Skip this much of the video first."}),
                "max_frames": ("INT", {
                    "default": 0, "min": 0, "max": 1000000,
                    "tooltip": "Stop after this many frames (counted after frame_rate). 0 = to the end."}),
                "frame_rate": (list(FRAME_RATES), {
                    "default": FRAME_RATES[0],
                    "tooltip": ("source = every frame. 24 = re-timed to H3's own rate while "
                                "decoding (nearest frame: dropped above 24, repeated below); "
                                "the duration and the soundtrack's sync are kept.")}),
                "resize": (list(RESIZE_MODES), {
                    "default": "none",
                    "tooltip": ("Shrink while decoding, aspect kept, never enlarging: to "
                                "resize_value megapixels, or the long edge to resize_value px.")}),
                "resize_value": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 16384.0, "step": 0.01,
                    "tooltip": "Megapixels (resize = megapixels) or pixels (resize = long edge)."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "FLOAT", "INT", "VIDEO")
    RETURN_NAMES = ("images", "audio", "fps", "frame_count", "video")
    FUNCTION = "run"
    CATEGORY = "video/upscale"
    DESCRIPTION = (
        "Get Video Components without the RAM bill: start, max_frames, a 24 fps re-time and a "
        "resize are applied while decoding, so skipped frames are never held, and the frames "
        "are filled into the IMAGE once (peak about 1.25x the output).")

    def run(self, video, start_seconds=0.0, max_frames=0, frame_rate=FRAME_RATES[0],
            resize="none", resize_value=1.0):
        import itertools
        import numpy as np
        import comfy.model_management as mm

        to_24 = str(frame_rate) != FRAME_RATES[0]
        start_s = max(0.0, float(start_seconds))
        limit = int(max_frames) if int(max_frames) > 0 else None
        src = _video_file(video)
        if src is not None:
            source, start, duration, crop = src
            src_fps = float(video.get_frame_rate())
            fps = float(H3_FPS) if to_24 else src_fps
            if duration and start_s >= duration:
                raise ValueError(f"{TAG}: start_seconds {start_s:g} is past the video's "
                                 f"{duration:g} s")
            w_start = start + start_s
            w_dur = max(0.0, duration - start_s) if duration else 0.0
            frames = decode_frames(video.get_stream_source(), w_start, w_dur, crop)
            if to_24 and abs(src_fps - fps) > 1e-3:
                frames = nearest_frames(frames, src_fps, fps)
            if limit is not None:
                frames = itertools.islice(frames, limit)
            parts, size, buf = [], None, []

            def flush():
                nonlocal size
                batch = torch.from_numpy(np.stack(buf))
                if size is None:
                    size = resize_target(int(batch.shape[2]), int(batch.shape[1]),
                                         str(resize), resize_value)
                parts.append(_store(batch, size))
                buf.clear()

            for img in frames:
                buf.append(img)
                if len(buf) == FRAME_BATCH:
                    mm.throw_exception_if_processing_interrupted()
                    flush()
            if buf:
                flush()
            if not parts:
                raise ValueError(f"{TAG}: no frames in the chosen window")
            images = frames_to_image(parts)
            n = int(images.shape[0])
            audio = read_audio(source, w_start, n / fps)
            how = "decoded in batches"
        else:
            comp = video.get_components()
            imgs, audio = comp.images, comp.audio
            src_fps = float(comp.frame_rate)
            fps = float(H3_FPS) if to_24 else src_fps
            first = int(round(start_s * src_fps))
            idx = range(first, int(imgs.shape[0]))
            if to_24 and abs(src_fps - fps) > 1e-3:
                idx = [first + i for i in nearest_frames(range(len(idx)), src_fps, fps)]
            idx = list(idx)[:limit] if limit is not None else list(idx)
            if not idx:
                raise ValueError(f"{TAG}: no frames in the chosen window")
            size = resize_target(int(imgs.shape[2]), int(imgs.shape[1]), str(resize), resize_value)
            images = imgs[idx]
            if (int(images.shape[2]), int(images.shape[1])) != size:
                images = _store(images.float(), size).float()
            n = len(idx)
            if audio is not None:
                sr = int(audio["sample_rate"])
                a0 = int(round(start_s * sr))
                audio = {"waveform": audio["waveform"][..., a0:a0 + int(round(n / fps * sr))],
                         "sample_rate": sr}
            how = "not a plain file (components or list): decoded whole, then selected"
        print(f"[{TAG}] video frames: {n} at {int(images.shape[2])}x{int(images.shape[1])}, "
              f"{fps:g} fps (source {src_fps:g}), from {start_s:g} s, {how}; IMAGE "
              f"{images.numel() * 4 / 1048576.0:.0f} MB", flush=True)
        return (images, audio, fps, n, video)


# --- a video from a path ----------------------------------------------------
#
# Core's Load Video reads from ComfyUI's input folder, and a browser upload
# above 100 MB is refused unless ComfyUI starts with --max-upload-size. This
# loader reads the file where it is: nothing is uploaded or copied. The idea
# (a VIDEO from a typed path, re-run on the file's size and mtime) follows
# silveroxides' UC_LoadVideoPath; the code is this pack's own.

def resolve_video_path(path):
    """A typed path -> an absolute file path, or "" when empty. Surrounding
    quotes (Windows' Copy as path) are dropped, and a relative path is looked
    up in ComfyUI's input folder."""
    import os
    p = str(path or "").strip().strip('"').strip("'").strip()
    if not p:
        return ""
    p = os.path.expanduser(os.path.expandvars(p))
    if not os.path.isabs(p):
        try:
            import folder_paths
            p = os.path.join(folder_paths.get_input_directory(), p)
        except Exception:  # noqa: BLE001
            p = os.path.abspath(p)
    return os.path.normpath(p)


def probe_video(path):
    """What the file is, from its headers only: a dict, or raises ValueError
    when there is no readable video stream."""
    import av
    try:
        with av.open(path, mode="r") as c:
            if not c.streams.video:
                raise ValueError("no video stream")
            v = c.streams.video[0]
            ctx = v.codec_context
            avg = float(v.average_rate) if v.average_rate else 0.0
            base = float(v.base_rate) if getattr(v, "base_rate", None) else avg
            dur = (float(v.duration * v.time_base) if v.duration else
                   float(c.duration) / 1e6 if c.duration else 0.0)
            frames = int(v.frames or 0)
            fmt = ctx.format if ctx is not None else None
            bits = max((int(x.bits) for x in fmt.components), default=8) if fmt is not None else 8
            a = decodable_audio(c)
            return {
                "width": int(ctx.width) if ctx is not None else 0,
                "height": int(ctx.height) if ctx is not None else 0,
                "fps": avg, "variable_rate": bool(avg and base and abs(avg - base) / base > 0.01),
                "frames": frames or int(round(dur * avg)), "frames_exact": bool(frames),
                "duration": dur, "codec": v.codec_context.name if ctx is not None else "?",
                "bits": bits,
                "audio": (f"{a.codec_context.name} {a.sample_rate} Hz {a.channels} ch"
                          if a is not None else ("undecodable track ignored"
                                                 if len(c.streams.audio) else "none")),
                "size_mb": os.path.getsize(path) / 1e6,
            }
    except ValueError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"not a readable video ({exc})") from None


def describe_video(info):
    rate = f"{info['fps']:.3f}".rstrip("0").rstrip(".")
    return (f"{info['width']}x{info['height']}, {rate} fps"
            + (" (VARIABLE frame rate: frames are timed by count, so picture and sound may "
               "drift; convert to a constant rate first for long clips)" if info["variable_rate"] else "")
            + f", {'' if info['frames_exact'] else '~'}{info['frames']} frames, "
            f"{info['duration']:.2f} s, {info['codec']} {info['bits']}-bit, "
            f"audio: {info['audio']}, {info['size_mb']:.0f} MB")


class HTCLoadVideoPath:
    """A VIDEO from a file path: no upload, no copy, no size limit."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"path": ("STRING", {
            "default": "", "multiline": False,
            "placeholder": "D:/videos/clip.mp4 (or a name in ComfyUI/input)",
            "tooltip": ("The video file. Paste a full path (quotes from Copy as path are fine), "
                        "or a name relative to ComfyUI's input folder. Read in place: nothing is "
                        "uploaded or copied, so there is no upload size limit.")})}}

    RETURN_TYPES = ("VIDEO", "FLOAT", "INT", "STRING")
    RETURN_NAMES = ("video", "fps", "frame_count", "info")
    FUNCTION = "run"
    CATEGORY = "video/upscale"
    DESCRIPTION = ("Load Video without the upload: reads the file where it is (no 100 MB upload "
                   "limit, no copy into input/), and reports its size, frame rate, length, codec "
                   "and audio.")

    @classmethod
    def IS_CHANGED(cls, path):
        import os
        p = resolve_video_path(path)
        try:
            st = os.stat(p)
            return f"{p}|{st.st_size}|{st.st_mtime_ns}"
        except OSError:
            return p

    @classmethod
    def VALIDATE_INPUTS(cls, path):
        import os
        p = resolve_video_path(path)
        if not p:
            return "No video path given."
        if not os.path.isfile(p):
            return f"No file at {p}"
        try:
            probe_video(p)
        except ValueError as exc:
            return f"{p}: {exc}"
        return True

    def run(self, path):
        from comfy_api.latest import InputImpl
        p = resolve_video_path(path)
        info = probe_video(p)
        text = f"{os.path.basename(p)}: {describe_video(info)}"
        print(f"[{TAG}] {text}", flush=True)
        return {"ui": {"text": [text]},
                "result": (InputImpl.VideoFromFile(p), float(info["fps"]), int(info["frames"]), text)}

NODE_CLASS_MAPPINGS = {"H3VideoUpsampler": HTCVideoUpsampler,
                       "H3VideoUpsamplerToFile": HTCVideoUpsamplerToFile,
                       "H3VideoFrames": HTCVideoFrames,
                       "H3LoadVideoPath": HTCLoadVideoPath}
NODE_DISPLAY_NAME_MAPPINGS = {"H3VideoUpsampler": "H3 Video Upsampler",
                              "H3VideoUpsamplerToFile": "H3 Video Upsampler (file to file)",
                              "H3VideoFrames": "H3 Video Frames (lean)",
                              "H3LoadVideoPath": "H3 Load Video (path)"}
