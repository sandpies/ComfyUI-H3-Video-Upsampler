# Generated from ComfyUI-Hand-Tie-Clips (https://github.com/sandpies/ComfyUI-Hand-Tie-Clips) by tools/build_upsampler_pack.py.
# Edit it there, not here: the next build overwrites this file.
"""HTC Tiled Upsampler -- a finished H3 clip made bigger and sharper, in one node.

WHAT IT DOES, in order, once per time chunk of the clip:

1. **Upscale in latent space.** LBH-123-AI's learned H3 latent upscaler takes
   the chunk's video latent to the size the megapixels widget asks for. Audio
   is never touched.
2. **Re-draw it lightly, in tiles.** One sampling step at denoise 0.2 on the
   `simple` schedule sharpens what the upscale smeared. The frame is cut into
   the fewest equal tiles whose forward fits a fixed token budget, and the
   tiles are stepped TOGETHER: one forward per tile, merged into one latent
   under a raised-cosine feather, before anything moves on. Every tile slices
   one noise field drawn at the chunk's full size, so two tiles covering the
   same spot start from the same noise. Together that is why there are no tile
   seams.
3. **Join the chunks.** Chunks overlap by one 17-frame grid step. Each chunk
   after the first has its frame-0 keyframe pinned to the previous chunk's
   re-drawn frame, and the overlap is crossfaded linearly.

Then the video is decoded with the VAE and the audio with the audio VAE.

THE RECIPE IS FIXED. Every number below is the one from the user's
MMH3 Ultimate Upscale graph (bbaudio-2025's Comfyui-MMH3-UltimateUpscale,
MIT), which is where this recipe was proven: 102-frame chunks with a 17-frame
overlap, anchor 0.999, auto tiling at 70,000 tokens per forward with 128 px
overlaps and a 256 px floor, joint steps on, one `simple` step at denoise 0.2,
seed 444, upscaler in fp16. Only the size is a widget; it defaults to 2 MP
(the proving run used 4, a stress test). This file is HTC's own
code; tools/check_tiled_upsampler.py runs it against that pack, when it is
installed, and requires identical output.

The upscaler network layout, key names and the 48 normalisation numbers are
LBH-123-AI's (the weights only load in the layout they were trained in; see
THIRD_PARTY_NOTICES.md). The runner is this pack's.
"""
from __future__ import annotations

import collections
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

TAG = "H3VideoUpsampler"

# --- the recipe -------------------------------------------------------------
UPSCALER_FILE = "minimax_h3_latent_upscaler_3d_bf16.safetensors"
MEGAPIXELS = 2.0            # 1 MP = 1024 x 1024 px, the only widget
SIZE_ALIGN = 32             # both edges snap to this, aspect held
CHUNK_FRAMES = 102          # pixel frames per time chunk (a multiple of 17)
CHUNK_OVERLAP = 17          # pixel frames shared by neighbouring chunks
ANCHOR_AUG = 0.999          # frame-0 pin's visual cond noise aug (H3 default)
TOKEN_BUDGET = 70000        # largest sequence one tile forward may carry
TILE_OVERLAP_PX = 128       # desired overlap between tiles, both axes
MIN_TILE_PX = 256           # no tile smaller than this
MAX_GRID_SIDE = 9           # tile grid search limit per axis
SCHEDULER, STEPS, DENOISE = "simple", 1, 0.2
SEED = 444
UPSCALER_DTYPE = torch.float16
_FLUSH_S = [0.0]           # seconds spent in sample_tile's cache flush (log only)
_LAST_OOM = [0]            # OOM retries in the most recent redraw_chunk call (log only)
_LAST_BUDGET = [TOKEN_BUDGET]  # budget redraw_chunk / upsample finished at


class _DefaultInt(int):
    """Sentinel int equal to CHUNK_FRAMES that marks an un-overridden default."""


_DEFAULT_CHUNK_FRAMES = _DefaultInt(CHUNK_FRAMES)
REGEN2K_MIN_CHUNK_FRAMES = 68
REGEN2K_SEED = 42
REGEN2K_SIGMAS = torch.tensor([0.909375, 6.0 / 7.0, 2.0 / 3.0, 0.0], dtype=torch.float32)
REGEN2K_SIGMAS_START_08 = torch.tensor([0.8, 24.0 / 35.0, 0.48, 0.0], dtype=torch.float32)
REGEN2K_SIGMAS_STYLIZED = torch.tensor([0.6, 0.4, 0.2, 0.0], dtype=torch.float32)
REGEN2K_SIGMAS_2STEP = torch.tensor([0.7, 0.494118, 0.0], dtype=torch.float32)
REGEN2K_PRESETS = ("realistic", "start_08", "stylized")


def regen2k_sigmas_for_preset(preset="realistic"):
    """Return the 4-sigma Regen2K schedule tensor for `preset`."""
    p = str(preset or "realistic").strip().lower()
    if p == "realistic":
        return REGEN2K_SIGMAS
    if p in ("start_08", "start_0.8", "stylized_08", "0.8"):
        return REGEN2K_SIGMAS_START_08
    if p == "stylized":
        return REGEN2K_SIGMAS_STYLIZED
    raise ValueError(f"{TAG}: unknown regen2k_preset {preset!r} "
                     f"(expected one of {list(REGEN2K_PRESETS)})")

# --- the guided re-draw (not part of the recipe) ----------------------------
#
# With an h3upscale LoRA on the re-draw model, the same one step also gets the
# LoRA's guide: the film at 1/factor of the output size, packed on every
# factor-th patch of each tile. The recipe's numbers do not move; the output
# size snaps to 32 * factor, tiles snap to the same grid, the guide's rows
# count against the token budget, and the conditioning is the LoRA's caption.
# Without a guide every line below behaves exactly as before.
STRIDE_KEY = "htc_grid_stride"     # upscale_guide.STRIDE_KEY; the layout patch reads it

# factor: the LoRA's reference_downscale_factor. cond: the caption
# conditioning, used for every chunk. decode(video latent) -> frames and
# encode(frames) -> video latent are the chain's VAE; decode may be None and
# upsample_chain fills in its own. pixels(frames, w_px, h_px, factor) shrinks
# canvas frames to the guide (upscale_guide.guide_pixels).
Guide = collections.namedtuple("Guide", "factor cond decode encode pixels")

FOLDER = "latent_upscale_models"
SPATIAL = 16                # H3 video VAE: pixels per latent cell
FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
FRAME_RESCALE = 5.0 / 3.0   # audio latent steps per pixel frame (40 vs 24 Hz)

# The statistics LBH-123-AI's weights were trained against (24 channels).
LATENT_MEAN = (
    0.858090341091156, -0.9606591463088989, 1.0661640167236328,
    -0.5090325474739075, -0.2727581858634949, -1.3675414323806763,
    -0.2553254961967468, -0.26907554268836975, -0.5376840829849243,
    -0.0464097298681736, 0.6657370328903198, 0.19690127670764923,
    -0.5460608005523682, -0.4035342037677765, -0.23683024942874908,
    0.25928452610969543, -0.30133944749832153, 0.211341992020607,
    -1.1206848621368408, 0.3581933379173279, -0.04225143790245056,
    0.2604829967021942, 0.22864092886447906, 0.7056031823158264,
)
LATENT_STD = (
    1.2223774194717407, 1.2767263650894165, 1.6831774711608887,
    1.7549455165863037, 1.5636216402053833, 2.194143533706665,
    0.9653137922286987, 1.0569885969161987, 0.841948926448822,
    0.7729952931404114, 1.8955937623977661, 0.946841835975647,
    0.7996809482574463, 0.44988900423049927, 0.7197399735450745,
    0.6936293244361877, 2.961095094680786, 2.7694199085235596,
    3.0496184825897217, 2.1088054180265264, 3.276226282119751,
    3.1627357006073, 2.2816812992095947, 2.6127843856811523,
)


def register_folder():
    """models/latent_upscale_models, unless something registered it already."""
    import folder_paths
    if FOLDER not in folder_paths.folder_names_and_paths:
        folder_paths.add_model_folder_path(
            FOLDER, os.path.join(folder_paths.models_dir, FOLDER))


def choices():
    try:
        import folder_paths
        names = [n for n in folder_paths.get_filename_list(FOLDER)
                 if n.endswith(".safetensors")]
    except Exception:  # noqa: BLE001
        names = []
    if UPSCALER_FILE in names:
        names.remove(UPSCALER_FILE)
    return [UPSCALER_FILE] + sorted(names)


# --- size -------------------------------------------------------------------

def target_size(src_w, src_h, megapixels, align=SIZE_ALIGN):
    """(w, h) in pixels: `megapixels` of area at the source aspect, both edges
    on the `align` grid.

    Of the nine grid points around the ideal size, the one whose aspect is
    closest wins; area only breaks ties, weighted a quarter as much, because a
    bent aspect is what people notice and the area is what they asked for.
    """
    src_w, src_h, align = max(1, int(src_w)), max(1, int(src_h)), max(1, int(align))
    aspect = src_w / float(src_h)
    area = float(megapixels) * 1024.0 * 1024.0
    ideal_h = (area / aspect) ** 0.5
    ideal_w = ideal_h * aspect
    area = ideal_w * ideal_h

    def snap(v):
        return max(align, int(round(float(v) / align)) * align)

    best = None
    for dw in (-1, 0, 1):
        w = snap(ideal_w) + dw * align
        if w < align:
            continue
        for dh in (-1, 0, 1):
            h = snap(ideal_h) + dh * align
            if h < align:
                continue
            score = (abs(w / float(h) - aspect) / aspect * 4.0
                     + abs(w * h - area) / max(1.0, area))
            if best is None or score < best[0]:
                best = (score, w, h)
    return best[1], best[2]


# --- time: frames, tokens, chunks -------------------------------------------

def frames_in(tokens):
    """Pixel frames covered by the first `tokens` video latent steps."""
    return sum(FRAME_PER_TOKEN[k % 5] for k in range(int(tokens)))


def tokens_to_reach(frames):
    """Fewest video latent steps whose frames reach at least `frames`."""
    k = got = 0
    while got < frames:
        got += FRAME_PER_TOKEN[k % 5]
        k += 1
    return k


def audio_span(f0, f1):
    """Audio latent steps [a0, a1) under pixel frames [f0, f1)."""
    return round(f0 * FRAME_RESCALE), round(f1 * FRAME_RESCALE)


def _nearest_key_step(frame, tokens):
    """The key step (index % 5 == 0) whose start frame is nearest `frame`."""
    best_k, best_f, best_d = 0, 0, frame
    for k in range(0, tokens + 1, 5):
        f = frames_in(k)
        if abs(f - frame) < best_d:
            best_k, best_f, best_d = k, f, abs(f - frame)
    return best_k, best_f


def chunks(tokens, length=CHUNK_FRAMES, overlap=CHUNK_OVERLAP):
    """[(k0, f0, k1, f1)]: each chunk's latent steps and frames.

    Boundaries sit on key steps, a chunk never starts after the previous one
    ended, and the last chunk ends exactly on the clip's last frame.
    """
    total = frames_in(tokens)
    hop = length - overlap
    out, prev_end, i = [], 0, 0
    while True:
        start, end = i * hop, min(i * hop + length, total)
        if i == 0:
            k0, f0 = 0, 0
        else:
            k0, f0 = _nearest_key_step(start, tokens)
            if k0 > prev_end:
                k0, f0 = prev_end, frames_in(prev_end)
        if end >= total:
            k1, f1 = tokens, total
        else:
            k1, f1 = _nearest_key_step(end, tokens)
            if k1 <= k0:
                k1 = k0 + 5
                f1 = frames_in(k1)
            if k1 >= tokens:
                k1, f1 = tokens, total
        out.append((k0, f0, k1, f1))
        if k1 >= tokens:
            return out
        prev_end = k1
        i += 1


# --- space: the tile grid ---------------------------------------------------

def _equal_split(total_px, count, want_overlap_px, grid=32):
    """(tile_px, overlap_px) for `count` equal tiles covering `total_px`
    exactly, the overlap on the `grid` and as close to the wanted one as the
    divisibility allows. None when no such split exists."""
    if count <= 1:
        return -(-int(total_px) // grid) * grid, 0
    lo = -(-((int(total_px) + (count - 1) * int(want_overlap_px)) // count)
           // grid) * grid
    hi = int(total_px) - grid * (count - 1)
    for tile in range(lo, hi + 1, grid):
        spare = count * tile - int(total_px)
        if spare % (count - 1) == 0:
            ov = spare // (count - 1)
            if ov % grid == 0 and 0 <= ov <= tile - grid:
                return tile, ov
    return None


def seq_len(tile_w_px, tile_h_px, tokens):
    """One forward's sequence length: 2x2-patched cells x video latent steps."""
    return (int(tile_w_px) // 32) * (int(tile_h_px) // 32) * int(tokens)


def guided_seq_len(tile_w_px, tile_h_px, tokens, factor):
    """`seq_len` plus the guide's rows: one per factor-th patch, every step."""
    f = int(factor)
    return (seq_len(tile_w_px, tile_h_px, tokens)
            + (int(tile_w_px) // (32 * f)) * (int(tile_h_px) // (32 * f)) * int(tokens))


def solve_grid(frame_w_px, frame_h_px, tokens, overlap_px=TILE_OVERLAP_PX,
               budget=TOKEN_BUDGET, min_tile_px=MIN_TILE_PX,
               max_side=MAX_GRID_SIDE, factor=1):
    """The fewest equal tiles whose forward fits `budget`.

    -> (rows, cols, tile_w_px, tile_h_px, overlap_w_px, overlap_h_px) or None.
    Among grids of the same size, the one that splits the longer frame axis
    more wins (strips, not a checkerboard), then the squarer one. A guide
    `factor` > 1 puts tiles and overlaps on the 32 * factor grid and counts the
    guide's rows in the budget.
    """
    grid = 32 * int(factor) if int(factor) > 1 else 32
    for n in range(1, max_side * max_side + 1):
        best = None
        for rows in range(1, max_side + 1):
            if n % rows:
                continue
            cols = n // rows
            if cols > max_side:
                continue
            across = _equal_split(frame_w_px, cols, overlap_px, grid)
            down = _equal_split(frame_h_px, rows, overlap_px, grid)
            if across is None or down is None:
                continue
            tw, ow = across
            th, oh = down
            if tw < min_tile_px or th < min_tile_px:
                continue
            cost = (guided_seq_len(tw, th, tokens, factor) if int(factor) > 1
                    else seq_len(tw, th, tokens))
            if cost > budget:
                continue
            long_split = rows if frame_h_px >= frame_w_px else cols
            rank = (-long_split, abs(rows - cols))
            if best is None or rank < best[0]:
                best = (rank, rows, cols, tw, th, ow, oh)
        if best is not None:
            return best[1:]
    return None


def _axis(size, tile, overlap, floor):
    """Tile starts, extents and seam overlaps along one axis, in latent cells.
    A leftover edge tile smaller than `floor` is pulled back to `floor`."""
    if size <= tile:
        return [0], [size], [0]
    stride = tile - overlap
    n = math.ceil((size - overlap) / stride)
    if (n - 1) * stride + tile < size:
        n += 1
    starts = [i * stride for i in range(n)]
    extents = [min(tile, size - s) for s in starts]
    if floor > 0 and n >= 2 and size - starts[-1] < floor:
        pulled = size - floor
        if starts[-2] < pulled < starts[-2] + extents[-2]:
            starts[-1], extents[-1] = pulled, size - pulled
    seams = [0] + [max(0, starts[i - 1] + extents[i - 1] - starts[i])
                   for i in range(1, n)]
    return starts, extents, seams


def snap_overlap(px):
    """Tile overlap in px, rounded to the 32 px patch grid, 0..512."""
    return max(0, min(512, int(round(float(px) / 32.0)) * 32))


def tile_grid(h, w, tokens, factor=1, budget=TOKEN_BUDGET, overlap_px=TILE_OVERLAP_PX):
    """The tiles for an (h, w)-cell chunk of `tokens` video steps.

    -> list of (r0, c0, rows, cols, seam_top, seam_left, top, left, bottom,
    right) in latent cells, plus the solved grid for the log. `budget` is the
    largest sequence one tile forward may carry: the recipe's TOKEN_BUDGET
    unless a caller lowers it to fit a smaller card.
    """
    got = solve_grid(w * SPATIAL, h * SPATIAL, tokens, overlap_px=snap_overlap(overlap_px),
                     budget=int(budget), factor=factor)
    if got is None:
        raise RuntimeError(
            f"{TAG}: no tile grid up to {MAX_GRID_SIDE}x{MAX_GRID_SIDE} keeps "
            f"a {w * SPATIAL}x{h * SPATIAL}px chunk of {tokens} latent steps "
            f"under {int(budget)} tokens per forward. Lower megapixels, or raise "
            f"the tile budget.")
    _r, _c, tw, th, ow, oh = got
    tw, th, ow, oh = tw // SPATIAL, th // SPATIAL, ow // SPATIAL, oh // SPATIAL
    floor = MIN_TILE_PX // SPATIAL
    r0s, rh, rseam = _axis(h, th, oh, floor)
    c0s, cw, cseam = _axis(w, tw, ow, floor)
    odd = sorted({x * SPATIAL for x in rh + cw if x % 2})
    if odd:
        raise RuntimeError(f"{TAG}: a tile side of {odd} px is not a multiple "
                           "of 32; H3 patchifies 2x2 latent cells.")
    if int(factor) > 1:
        # The guide sits on every factor-th PATCH of the tile, and a tile's
        # guide window is cut from the film's guide at (r0, c0) / factor, so
        # every edge and origin must land on a whole guide patch.
        step = 2 * int(factor)
        off = sorted({x * SPATIAL for x in r0s + rh + c0s + cw if x % step})
        if off:
            raise RuntimeError(f"{TAG}: tile edges {off} px are off the "
                               f"{step * SPATIAL} px guide grid")
    tiles = []
    nr, nc = len(r0s), len(c0s)
    for i in range(nr):
        for j in range(nc):
            tiles.append((r0s[i], c0s[j], rh[i], cw[j],
                          rseam[i] or oh, cseam[j] or ow,
                          i > 0, j > 0, i < nr - 1, j < nc - 1))
    return tiles, got


def feather(rows, cols, seam_top, seam_left, top, left, bottom, right, device):
    """A tile's weight when tiles are merged: 1 inside, a raised cosine down to
    0 across any side that has a neighbour. Raised cosine, not linear, so the
    weight arrives at 1 without a kink that would print as a faint line."""
    def ramp(n):
        i = torch.arange(n, device=device, dtype=torch.float32)
        return 0.5 * (1.0 - torch.cos(math.pi * i / n))

    m = torch.ones((rows, cols), device=device, dtype=torch.float32)
    if left and seam_left > 0:
        m[:, :seam_left] *= ramp(min(seam_left, cols))[None, :]
    if right and seam_left > 0:
        m[:, -seam_left:] *= torch.flip(ramp(min(seam_left, cols)), (0,))[None, :]
    if top and seam_top > 0:
        m[:seam_top, :] *= ramp(min(seam_top, rows))[:, None]
    if bottom and seam_top > 0:
        m[-seam_top:, :] *= torch.flip(ramp(min(seam_top, rows)), (0,))[:, None]
    return m


# --- conditioning -----------------------------------------------------------

def _resize_planes(lt, h, w, dtype=None):
    """[B, C, T, H, W] -> [B, C, T, h, w], every H x W plane bilinear."""
    b, c, t, hh, ww = lt.shape
    x = lt if dtype is None else lt.to(dtype)
    x = F.interpolate(x.reshape(b * c * t, 1, hh, ww), size=(h, w),
                      mode="bilinear", align_corners=False)
    return x.reshape(b, c, t, h, w)


def clean_refs(cond):
    """Reference blocks whose row counts come from their real latents.

    H3 reserves reference rows from a block's metadata but fills them from the
    block's latent. A block whose metadata disagrees, or a picture block with
    no latent, would reserve rows nothing fills. Rewrite the metadata from the
    latent and drop picture blocks without one.
    """
    out = []
    for emb, d in cond:
        d = dict(d)
        if d.get("minimax_refs"):
            kept = []
            for blk in d["minimax_refs"]:
                blk = dict(blk)
                lt = blk.get("latent")
                if lt is None:
                    if blk.get("kind") not in ("image", "video", "video_audio"):
                        kept.append(blk)
                    continue
                blk["latent_h"], blk["latent_w"] = int(lt.shape[3]), int(lt.shape[4])
                if blk.get("kind") in ("video", "video_audio"):
                    blk["latent_t"] = int(lt.shape[2])
                kept.append(blk)
            d["minimax_refs"] = kept
        out.append([emb, d])
    return out


def _keyframe_in(kf, f0, f1):
    """The part of keyframe `kf` lying wholly inside frames [f0, f1), re-based
    to f0, or None."""
    idx = kf["resolved_frame_index"]
    lt, al = kf.get("latent"), kf.get("audio_latent")
    if lt is None and al is None:
        return {"resolved_frame_index": idx - f0} if f0 <= idx < f1 else None
    out = {}
    if lt is not None:
        first = last = None
        pos = idx
        for k in range(lt.shape[2]):
            span = FRAME_PER_TOKEN[k % 5]
            if f0 <= pos and pos + span <= f1:
                first = k if first is None else first
                last = k + 1
            pos += span
        if first is None:
            return None
        out["latent"] = lt[:, :, first:last].contiguous()
        out["resolved_frame_index"] = idx + frames_in(first) - f0
    if al is not None:
        a0 = max(0, math.ceil((f0 - idx) * FRAME_RESCALE))
        a1 = min(al.shape[-1], math.floor((f1 - idx) / FRAME_RESCALE))
        if a1 > a0:
            out["audio_latent"] = al[..., a0:a1].contiguous()
            out.setdefault("resolved_frame_index", max(0, idx - f0))
    return out if ("latent" in out or "audio_latent" in out) else None


def cond_for_chunk(cond, f0, f1, h, w):
    """The keyframes that fall inside the chunk, re-based to it, their video
    resized to the chunk's (upscaled) grid."""
    out = []
    for emb, d in cond:
        d = dict(d)
        if d.get("minimax_keyframes"):
            kept = [k for k in (_keyframe_in(kf, f0, f1)
                                for kf in d["minimax_keyframes"]) if k is not None]
            for kf in kept:
                lt = kf.get("latent")
                if lt is not None and (lt.shape[3] != h or lt.shape[4] != w):
                    kf["latent"] = _resize_planes(lt, h, w)
            if kept:
                d["minimax_keyframes"] = kept
            else:
                d.pop("minimax_keyframes", None)
        out.append([emb, d])
    return out


def pin_chunk_start(cond, done_video, f0, aug=ANCHOR_AUG):
    """Frame 0 of this chunk pinned to what the previous chunk re-drew there."""
    k = tokens_to_reach(f0)
    if k >= done_video.shape[2]:
        raise RuntimeError(f"{TAG}: the finished chunks stop before frame {f0}")
    pin = {"resolved_frame_index": 0,
           "latent": done_video[:, :, k:k + 1].contiguous()}
    out = []
    for emb, d in cond:
        d = dict(d)
        rest = [kf for kf in (d.get("minimax_keyframes") or [])
                if kf.get("resolved_frame_index") != 0 or "latent" not in kf]
        d["minimax_keyframes"] = [pin] + rest
        d["minimax_visual_cond_noise_aug"] = max(0.0, min(1.0, float(aug)))
        out.append([emb, d])
    return out


def cond_for_tile(cond, h, w, r0, c0, rows, cols):
    """Keyframe video cropped to one tile. A keyframe on another grid is
    resized to the chunk's first, so the crop has the tile's row count."""
    out = []
    for emb, d in cond:
        d = dict(d)
        if d.get("minimax_keyframes"):
            kfs = []
            for kf in d["minimax_keyframes"]:
                kf = dict(kf)
                lt = kf.get("latent")
                f = int(kf.get(STRIDE_KEY, 1))
                if lt is not None and f > 1:
                    # The guide is on its own 1/f grid: cut the tile's window
                    # there, never resized up to the chunk's grid.
                    kf["latent"] = lt[:, :, :, r0 // f:(r0 + rows) // f,
                                      c0 // f:(c0 + cols) // f].contiguous()
                elif lt is not None:
                    if lt.shape[3] != h or lt.shape[4] != w:
                        lt = _resize_planes(lt, h, w, torch.float32)
                    kf["latent"] = lt[:, :, :, r0:r0 + rows, c0:c0 + cols].contiguous()
                kfs.append(kf)
            d["minimax_keyframes"] = kfs
        out.append([emb, d])
    return out


def encode_steps(encode, frames, steps):
    """`steps` video latent steps from the frames that cover them.

    H3's VAE encodes 17-frame clips to 5 steps each, then drops the last 3
    steps -- right for a whole clip (17g+5 frames, 5g+2 steps), short for a
    chunk cut out of a film: 102 frames come back as 27 steps, not 30. Each
    17-frame clip encodes on its own, so padding the frames to the next whole
    clip length (repeating the last frame, as the encoder itself pads) and
    keeping the first `steps` steps gives exactly what those frames encode to.
    """
    n = frames_in(steps)
    if int(frames.shape[0]) < n:
        raise RuntimeError(f"{TAG}: {int(frames.shape[0])} frames for {steps} latent "
                           f"steps, which cover {n}")
    frames = frames[:n]
    full = 5 if n <= 5 else 17 * math.ceil((n - 5) / 17) + 5
    if full > n:
        frames = torch.cat([frames, frames[-1:].expand(full - n, -1, -1, -1)], dim=0)
    lat = encode(frames)
    if int(lat.shape[2]) < steps:
        raise RuntimeError(f"{TAG}: {full} frames encoded to {int(lat.shape[2])} "
                           f"latent steps, needed {steps}")
    return lat[:, :, :steps]


def add_guide(cond, guide_latent, factor):
    """The h3upscale guide appended as a strided keyframe at frame 0. After
    `pin_chunk_start`, which drops every other latent keyframe at frame 0."""
    kf = {"latent": guide_latent, "resolved_frame_index": 0, STRIDE_KEY: int(factor)}
    out = []
    for emb, d in cond:
        d = dict(d)
        d["minimax_keyframes"] = list(d.get("minimax_keyframes") or []) + [kf]
        out.append([emb, d])
    return out


# --- the upscaler network, in the layout its weights were trained in --------

def _gn(c):
    return nn.GroupNorm(32, c)


class _Res(nn.Module):
    def __init__(self, c, emb):
        super().__init__()
        self.in_layers = nn.Sequential(_gn(c), nn.SiLU(), nn.Conv3d(c, c, 3, padding=1))
        self.emb_layers = nn.Sequential(nn.SiLU(), nn.Linear(emb, 2 * c))
        self.out_norm = _gn(c)
        self.out_layers = nn.Sequential(nn.SiLU(), nn.Dropout(0.0),
                                        nn.Conv3d(c, c, 3, padding=1))

    def forward(self, x, emb):
        h = self.in_layers(x)
        scale, shift = self.emb_layers(emb).to(h.dtype)[..., None, None, None] \
            .chunk(2, dim=1)
        return x + self.out_layers(self.out_norm(h) * (1 + scale) + shift)


class _Temporal(nn.Module):
    def __init__(self, c, k):
        super().__init__()
        self.norm = _gn(c)
        self.dwconv = nn.Conv3d(c, c, (k, 1, 1), padding=(k // 2, 0, 0), groups=c)
        self.pwconv = nn.Conv3d(c, c, 1)

    def forward(self, x):
        return x + self.pwconv(self.dwconv(F.silu(self.norm(x))))


class LatentUpscaler(nn.Module):
    def __init__(self, sd):
        super().__init__()
        c = int(sd["conv_in.weight"].shape[0])
        cin = int(sd["conv_in.weight"].shape[1])
        emb = int(sd["embed.0.weight"].shape[0])
        self.conv_in = nn.Conv3d(cin, c, 3, padding=1)
        self.embed = nn.Sequential(nn.Linear(1, emb), nn.SiLU(), nn.Linear(emb, emb))
        self.in_blocks = self._blocks(sd, "in_blocks", c, emb)
        self.out_blocks = self._blocks(sd, "out_blocks", c, emb)
        self.norm_out = _gn(c)
        self.conv_out = nn.Conv3d(c, cin, 3, padding=1)

    @staticmethod
    def _blocks(sd, name, c, emb):
        # Which slot is a residual block and which a temporal conv is read
        # from the keys, so a file with another rhythm refuses rather than
        # loading wrong.
        slots = {}
        for k in sd:
            if k.startswith(name + "."):
                i = int(k.split(".")[1])
                if k.endswith(".dwconv.weight"):
                    slots[i] = _Temporal(c, int(sd[k].shape[2]))
                elif ".in_layers." in k:
                    slots.setdefault(i, None)
        for i in [i for i, m in slots.items() if m is None]:
            slots[i] = _Res(c, emb)
        if sorted(slots) != list(range(len(slots))):
            raise RuntimeError(f"{TAG}: {name} has gaps in its block list; "
                               "not an upscaler file this node reads.")
        return nn.ModuleList(slots[i] for i in range(len(slots)))

    def forward(self, x, size, scale):
        emb = self.embed(torch.tensor([[float(scale) - 1.0]],
                                      device=x.device, dtype=x.dtype))
        emb = emb.expand(x.shape[0], -1)
        x = self.conv_in(x)
        for b in self.in_blocks:
            x = b(x, emb) if isinstance(b, _Res) else b(x)
        x = F.interpolate(x, size=size, mode="trilinear", align_corners=False)
        for b in self.out_blocks:
            x = b(x, emb) if isinstance(b, _Res) else b(x)
        return self.conv_out(F.silu(self.norm_out(x)))


_NET = {}


def load_upscaler(name):
    """The network for `name`, kept on the CPU in fp16 between chunks."""
    if name in _NET:
        return _NET[name]
    import folder_paths
    from safetensors.torch import load_file
    if not str(name).endswith(".safetensors"):
        raise RuntimeError(f"{TAG}: {name!r}: only .safetensors files are "
                           "read (no pickle).")
    path = folder_paths.get_full_path(FOLDER, name)
    if not path:
        raise RuntimeError(f"{TAG}: {name!r} is not in models/{FOLDER}. "
                           f"The recipe's file is {UPSCALER_FILE} "
                           "(huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler).")
    sd = load_file(path, device="cpu")
    if any(k.startswith("upscaler.") for k in sd):
        sd = {k[len("upscaler."):]: v for k, v in sd.items()
              if k.startswith("upscaler.")}
    if "conv_in.weight" not in sd or "embed.0.weight" not in sd:
        raise RuntimeError(f"{TAG}: {name!r} is not an H3 latent upscaler.")
    if int(sd["conv_in.weight"].shape[1]) != 24:
        raise RuntimeError(f"{TAG}: {name!r} does not take 24-channel H3 latents.")
    net = LatentUpscaler(sd)
    net.load_state_dict({k: (v.to(torch.float16) if v.dtype == torch.float8_e4m3fn
                             else v) for k, v in sd.items()}, strict=True)
    net = net.eval().requires_grad_(False).to(UPSCALER_DTYPE)
    _NET.clear()
    _NET[name] = net
    print(f"[{TAG}] upscaler {name}: "
          f"{sum(p.numel() for p in net.parameters()) / 1e6:.0f}M params", flush=True)
    return net


def _device():
    import comfy.model_management as mm
    return mm.get_torch_device()


def _free_all_gpu():
    """Unload any resident ComfyUI models (VAE, CLIP, DiT) and empty CUDA cache."""
    try:
        import gc
        import comfy.model_management as mm
        mm.unload_all_models()
        gc.collect()
        mm.soft_empty_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _is_oom(exc):
    """True when `exc` is a CUDA or ComfyUI out-of-memory error."""
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    try:
        import comfy.model_management as mm
        if isinstance(exc, getattr(mm, "OOM_EXCEPTION", ())):
            return True
    except Exception:
        pass
    msg = str(exc).lower()
    return "out of memory" in msg or "cuda error: out of memory" in msg or "alloc" in msg


def upscale_chunk(video, name, w_px, h_px):
    """One chunk's video latent resized to (w_px, h_px) by the upscaler.

    The network is told the scale actually achieved (mean of the two axes),
    and runs in fp16 end to end, normalisation included, as in the recipe.
    Unloads resident ComfyUI models before moving the 512-channel 3D Conv
    upscaler to CUDA, and automatically falls back to CPU (float32) if CUDA
    VRAM is exhausted so the latent upscale stage can never OOM.
    """
    _, _, t, h_in, w_in = video.shape
    scale = (w_px / (w_in * SPATIAL) + h_px / (h_in * SPATIAL)) / 2.0
    h_out = max(1, int(round(h_px / SPATIAL)))
    w_out = max(1, int(round(w_px / SPATIAL)))
    if scale < 1.0 and (w_out < w_in or h_out < h_in):
        raise RuntimeError(f"{TAG}: {w_px}x{h_px} is smaller than the clip "
                           f"({w_in * SPATIAL}x{h_in * SPATIAL}); this only "
                           "upscales. Raise megapixels.")
    if (h_out, w_out) == (h_in, w_in):
        return video
    dev = _device()
    if dev.type == "cuda":
        _free_all_gpu()
    net = load_upscaler(name)
    try:
        net = net.to(device=dev, dtype=UPSCALER_DTYPE)
        mean = torch.tensor(LATENT_MEAN, device=dev, dtype=UPSCALER_DTYPE).view(1, -1, 1, 1, 1)
        std = torch.tensor(LATENT_STD, device=dev, dtype=UPSCALER_DTYPE).view(1, -1, 1, 1, 1)
        with torch.inference_mode():
            x = (video.to(device=dev, dtype=UPSCALER_DTYPE, copy=True) - mean) / std
            out = net(x, (t, h_out, w_out), scale)
            del x
            out = out * std + mean
        return out.to(device="cpu", dtype=video.dtype)
    except Exception as exc:
        if dev.type != "cuda" or not _is_oom(exc):
            raise
        print(f"[{TAG}] CUDA VRAM tight during 3D latent upscale ({exc}); "
              "sharing load with CPU...", flush=True)
        try:
            net.to("cpu")
            torch.cuda.empty_cache()
        except Exception:
            pass
        cpu_dev = torch.device("cpu")
        net = net.to(device=cpu_dev, dtype=torch.float32)
        mean = torch.tensor(LATENT_MEAN, device=cpu_dev, dtype=torch.float32).view(1, -1, 1, 1, 1)
        std = torch.tensor(LATENT_STD, device=cpu_dev, dtype=torch.float32).view(1, -1, 1, 1, 1)
        with torch.inference_mode():
            x = (video.to(device=cpu_dev, dtype=torch.float32, copy=True) - mean) / std
            out = net(x, (t, h_out, w_out), scale)
            del x
            out = out * std + mean
        net.to(dtype=UPSCALER_DTYPE)
        return out.to(device="cpu", dtype=video.dtype)
    finally:
        net.to("cpu")
        if dev.type == "cuda":
            torch.cuda.empty_cache()


def upscale_chunks_batch(videos, name, w_px, h_px):
    """Upscale a list of chunk video latents `[cv_0, cv_1, ...]` in one pass so
    the 345M-parameter 3D Conv upscaler is loaded onto GPU once instead of 21
    times across a 10-hop chain."""
    if not videos:
        return []
    _, _, _t0, h_in, w_in = videos[0].shape
    scale = (w_px / (w_in * SPATIAL) + h_px / (h_in * SPATIAL)) / 2.0
    h_out = max(1, int(round(h_px / SPATIAL)))
    w_out = max(1, int(round(w_px / SPATIAL)))
    if scale < 1.0 and (w_out < w_in or h_out < h_in):
        raise RuntimeError(f"{TAG}: {w_px}x{h_px} is smaller than the clip "
                           f"({w_in * SPATIAL}x{h_in * SPATIAL}); this only "
                           "upscales. Raise megapixels.")
    if (h_out, w_out) == (h_in, w_in):
        return list(videos)
    # If a test monkeypatches `upscale_chunk`, honour the monkeypatch per chunk.
    if upscale_chunk is not _DEFAULT_UPSCALE_CHUNK:
        return [upscale_chunk(v, name, w_px, h_px) for v in videos]
    dev = _device()
    if dev.type == "cuda":
        _free_all_gpu()
    net = load_upscaler(name)
    out_list = []
    try:
        net = net.to(device=dev, dtype=UPSCALER_DTYPE)
        mean = torch.tensor(LATENT_MEAN, device=dev, dtype=UPSCALER_DTYPE).view(1, -1, 1, 1, 1)
        std = torch.tensor(LATENT_STD, device=dev, dtype=UPSCALER_DTYPE).view(1, -1, 1, 1, 1)
        for v in videos:
            t = int(v.shape[2])
            try:
                with torch.inference_mode():
                    x = (v.to(device=dev, dtype=UPSCALER_DTYPE, copy=True) - mean) / std
                    o = net(x, (t, h_out, w_out), scale)
                    del x
                    o = o * std + mean
                out_list.append(o.to(device="cpu", dtype=v.dtype))
                del o
            except Exception as exc:
                if dev.type != "cuda" or not _is_oom(exc):
                    raise
                print(f"[{TAG}] CUDA VRAM tight during 3D latent upscale ({exc}); "
                      "sharing load with CPU...", flush=True)
                net.to("cpu")
                torch.cuda.empty_cache()
                dev = torch.device("cpu")
                net = net.to(device=dev, dtype=torch.float32)
                mean = torch.tensor(LATENT_MEAN, device=dev, dtype=torch.float32).view(1, -1, 1, 1, 1)
                std = torch.tensor(LATENT_STD, device=dev, dtype=torch.float32).view(1, -1, 1, 1, 1)
                with torch.inference_mode():
                    x = (v.to(device=dev, dtype=torch.float32, copy=True) - mean) / std
                    o = net(x, (t, h_out, w_out), scale)
                    del x
                    o = o * std + mean
                out_list.append(o.to(device="cpu", dtype=v.dtype))
                del o
    finally:
        net.to(device="cpu", dtype=UPSCALER_DTYPE)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return out_list


_DEFAULT_UPSCALE_CHUNK = upscale_chunk


# --- sampling ---------------------------------------------------------------

def recipe_sigmas(model, denoise=DENOISE, steps=STEPS, start_sigma=0.0):
    """One `simple` step at `denoise`, exactly as Core's BasicScheduler.

    The recipe's 0.2 is a fraction of the SCHEDULE, not of the noise: under
    H3's video shift (12) the step starts at sigma 0.75. Below it the re-draw
    repaints less; at 0 there is no re-draw and this returns None.

    `start_sigma` > 0 (0.2.0) replaces `denoise`: the `simple` schedule over
    the model's (shifted) sigmas, cut where it drops to `start_sigma`, then
    `steps` steps spread evenly down that tail to 0. `steps` 1 with
    `start_sigma` 0 is the 0.1.0 recipe exactly.
    """
    import comfy.samplers
    steps = max(1, int(steps))
    ms = model.get_model_object("model_sampling")
    if float(start_sigma) > 0.0:
        full = comfy.samplers.calculate_sigmas(ms, SCHEDULER, 1000).cpu().float()
        tail = full[full <= float(start_sigma) + 1e-6]
        if tail.numel() < 2:
            return None
        idx = torch.linspace(0, tail.numel() - 1, steps + 1).round().long()
        idx = torch.unique_consecutive(idx)
        sig = tail[idx]
        sig[-1] = 0.0
        return sig
    if float(denoise) <= 0.0:
        return None
    total = int(steps / float(denoise))
    sig = comfy.samplers.calculate_sigmas(ms, SCHEDULER, total).cpu()
    return sig[-(steps + 1):]


def _apply_low_vram_patches(model, attn_chunks=4, ffn_chunks=2):
    """Apply AdaLN port, 4-way attention head chunking, and 2-way SwiGLU FFN
    token chunking (as in Seanaiochat.json's MiniMaxLowVRAMAttention(4) +
    MiniMaxChunkFeedForward(2, 4096)) onto `model` so tiled upsampling uses
    minimal activation VRAM even when `upsample_dit` is wired from a bare
    UNETLoader. Mathematically identical to the unchunked forward."""
    if model is None or not hasattr(model, "get_model_object"):
        return model
    try:
        import nodes as _nodes_mod
        nmap = getattr(_nodes_mod, "NODE_CLASS_MAPPINGS", {}) or {}
        if hasattr(model, "patches") and "H3AdaLNLoRAFix" in nmap:
            _res = nmap["H3AdaLNLoRAFix"].execute(model=model, mode="port")
            model = (_res.args if hasattr(_res, "args") else tuple(_res))[0]
        if int(attn_chunks) > 1 and "MiniMaxLowVRAMAttention" in nmap:
            _res = nmap["MiniMaxLowVRAMAttention"].execute(
                model=model, head_chunks=int(attn_chunks))
            model = (_res.args if hasattr(_res, "args") else tuple(_res))[0]
        if int(ffn_chunks) > 1 and "MiniMaxChunkFeedForward" in nmap:
            _res = nmap["MiniMaxChunkFeedForward"].execute(
                model=model, chunks=int(ffn_chunks), seq_threshold=4096)
            model = (_res.args if hasattr(_res, "args") else tuple(_res))[0]
    except Exception:
        pass
    return model


def sample_tile(model, cond, sampler, sigmas, video, audio, noise, seed):
    """One tile through `sigmas`, audio frozen. -> the tile's video latent.

    `noise` is a (video, audio) pair to use as is. SamplerCustomAdvanced's
    path, with a positive-only guider at cfg 1 and the x0 preview.
    """
    import comfy.model_management as mm
    import comfy.nested_tensor as nt
    import comfy.sample
    import comfy.samplers
    import comfy.utils
    import latent_preview

    samples = comfy.sample.fix_empty_latent_channels(
        model, nt.NestedTensor((video, audio)), None, None)
    mask = nt.NestedTensor((
        torch.ones((1, 1, 1, video.shape[3], video.shape[4]),
                   device=video.device, dtype=torch.float32),
        torch.zeros((1, 32, 2, audio.shape[-1]),
                    device=audio.device, dtype=audio.dtype)))
    guider = comfy.samplers.CFGGuider(model)
    guider.inner_set_conds({"positive": cond})
    callback = latent_preview.prepare_callback(
        guider.model_patcher, sigmas.shape[-1] - 1, {})
    out = guider.sample(nt.NestedTensor(noise), samples, sampler, sigmas,
                        denoise_mask=mask, callback=callback,
                        disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED,
                        seed=seed)
    res = out.to(mm.intermediate_device()).tensors[0]
    del out, guider, samples, mask
    _t = time.perf_counter()
    mm.soft_empty_cache()
    _FLUSH_S[0] += time.perf_counter() - _t
    return res


def _highpass_detail_merge(base_video, redrawn_video, sigma=8.0):
    """Keep `base_video` (full-frame 3D latent upscaler output, zero tiles) for
    tile-scale low frequencies (`> sigma` latent px, default 8.0 = 128 screen px)
    and take all mid + fine structural and texture detail from `redrawn_video`
    (tiled DiT re-draw): `base + (D - G_sigma(D))`.

    Eliminates tile-to-tile low-frequency DC/colour/contrast grid seams while
    preserving all mid-band facial/structural detail and fine texture from the
    `h3upscale` LoRA.
    """
    sigma = float(sigma or 0.0)
    if sigma <= 0.0 or base_video.shape != redrawn_video.shape or base_video.dim() < 5:
        return redrawn_video
    b, c, t, h, w = base_video.shape
    base_f = base_video.to(device=redrawn_video.device, dtype=torch.float32)
    red_f = redrawn_video.to(dtype=torch.float32)
    d = red_f - base_f
    radius = max(1, int(math.ceil(3.0 * sigma)))
    rad_h = min(radius, max(0, h - 1))
    rad_w = min(radius, max(0, w - 1))
    low = d.reshape(b * c * t, 1, h, w)
    if rad_w > 0:
        cw = torch.arange(-rad_w, rad_w + 1, device=low.device, dtype=torch.float32)
        kw = torch.exp(-0.5 * (cw / sigma) ** 2)
        kw = (kw / kw.sum()).view(1, 1, 1, -1)
        low = F.conv2d(F.pad(low, (rad_w, rad_w, 0, 0), mode="reflect"), kw)
    if rad_h > 0:
        ch = torch.arange(-rad_h, rad_h + 1, device=low.device, dtype=torch.float32)
        kh = torch.exp(-0.5 * (ch / sigma) ** 2)
        kh = (kh / kh.sum()).view(1, 1, -1, 1)
        low = F.conv2d(F.pad(low, (0, 0, rad_h, rad_h), mode="reflect"), kh)
    d_low = low.reshape(b, c, t, h, w)
    return (base_f + (d - d_low)).to(dtype=redrawn_video.dtype)


def _cond_extra_tokens(cond):
    """Extra non-strided conditioning rows (`minimax_refs` + `minimax_keyframes`)
    that ride every tile forward alongside the tile's own video tokens."""
    if not cond:
        return 0
    extra = 0
    try:
        d = cond[0][1] if len(cond[0]) > 1 and isinstance(cond[0][1], dict) else {}
        for blk in (d.get("minimax_refs") or ()):
            kind = blk.get("kind")
            if kind == "image":
                extra += max(0, int(blk.get("latent_h", 0)) // 2) * max(0, int(blk.get("latent_w", 0)) // 2)
            elif kind in ("video", "video_audio"):
                extra += (max(0, int(blk.get("latent_t", 0)))
                          * max(0, int(blk.get("latent_h", 0)) // 2)
                          * max(0, int(blk.get("latent_w", 0)) // 2))
            if kind in ("audio", "video_audio"):
                extra += max(0, int(blk.get("ref_audio_t", 0))) * 2
    except Exception:
        return 0
    return extra


def redraw_chunk(model, cond, sampler, sigmas, video, audio, seed, factor=1,
                 budget=TOKEN_BUDGET, overlap_px=TILE_OVERLAP_PX,
                 detail_sigma=0.0, noise_video=None):
    """Every tile stepped together, merged under the feather after each step.

    Step 0 hands each tile its slice of one noise field drawn at the chunk's
    full size; later steps add no noise. Audio rides along frozen. `factor`
    > 1 is a guided re-draw: tiles on the guide's grid, guide rows budgeted.
    When `detail_sigma > 0`, applies `_highpass_detail_merge(video, x, detail_sigma)`
    so low-frequency tile-grid shifts are stripped while fine detail is kept.
    Includes an automatic OOM guardrail: if a tile forward runs out of VRAM at
    `budget`, halves the tile token budget to split into smaller tiles and
    retries the chunk automatically.
    """
    import comfy.nested_tensor as nt
    import comfy.sample

    _, _, t, h, w = video.shape
    field = comfy.sample.prepare_noise(nt.NestedTensor((video, audio)), seed)
    nv, na = field.tensors[0], field.tensors[1]
    if noise_video is not None:
        # This chunk's frames of one film-wide field (noise=film): the frames
        # two chunks share get the same noise from both sides.
        nv = noise_video.to(device=nv.device, dtype=nv.dtype)
    dev, dt = video.device, video.dtype
    cur_budget = int(budget)
    _oom_retries = 0
    _LAST_OOM[0] = 0

    while True:
        tiles, grid = tile_grid(h, w, t, factor, cur_budget, overlap_px)
        prepared = []
        for (r0, c0, rows, cols, st, sl, top, left, bottom, right) in tiles:
            prepared.append((r0, c0, rows, cols,
                             cond_for_tile(cond, h, w, r0, c0, rows, cols),
                             feather(rows, cols, st, sl, top, left, bottom, right,
                                     dev)[None, None, None]))
        tokens = (seq_len(grid[2], grid[3], t) if int(factor) <= 1
                  else f"{guided_seq_len(grid[2], grid[3], t, factor)} (guide incl.)")
        _sig_s = [round(float(s), 4) for s in sigmas]
        _sig_cpu = sigmas.detach().cpu().float() if isinstance(sigmas, torch.Tensor) else torch.tensor(list(sigmas), dtype=torch.float32)
        _preset_tag = ("stylized" if _sig_cpu.shape == REGEN2K_SIGMAS_STYLIZED.shape
                       and torch.allclose(_sig_cpu, REGEN2K_SIGMAS_STYLIZED)
                       else "start_08" if _sig_cpu.shape == REGEN2K_SIGMAS_START_08.shape
                       and torch.allclose(_sig_cpu, REGEN2K_SIGMAS_START_08)
                       else "realistic" if _sig_cpu.shape == REGEN2K_SIGMAS.shape
                       and torch.allclose(_sig_cpu, REGEN2K_SIGMAS)
                       else None)
        print(f"[{TAG}] chunk {w * SPATIAL}x{h * SPATIAL}px x {t} latent steps: "
              f"{grid[0]}x{grid[1]} tiles of {grid[2]}x{grid[3]}px, overlap "
              f"{grid[4]}/{grid[5]}px, {tokens} tokens per "
              f"forward, stepped together"
              + (f", preset={_preset_tag}" if _preset_tag else "")
              + f", sigmas {_sig_s}"
              + (f", high-pass detail={float(detail_sigma):g}" if float(detail_sigma or 0.0) > 0.0 else ""),
              flush=True)

        # Multi-step (0.2.0): every step runs over ALL tiles and is merged under
        # the feather before the next step starts, so neighbours share one state
        # at every sigma. Between steps x holds the sampler's output for a CONST
        # (flow) model, i.e. x_t / (1 - sigma); the next call's zero-noise
        # noise_scaling multiplies it back by (1 - sigma), so the hand-off is
        # exact. Only step 0 draws noise, from one full-chunk field.
        _t0, _calls, _FLUSH_S[0] = time.perf_counter(), 0, 0.0
        try:
            x = video
            if len(prepared) == 1:
                # 1x1 whole-chunk tile: no spatial tile seams to feather-merge
                # between steps, so run all `sigmas` in a single `sample_tile`
                # (`CFGGuider.sample`) call instead of re-staging the 32.4 GB
                # DynamicVRAM DiT on every step.
                r0, c0, rows, cols, tcond, _m = prepared[0]
                xt = x[:, :, :, r0:r0 + rows, c0:c0 + cols].contiguous()
                noise = (nv[:, :, :, r0:r0 + rows, c0:c0 + cols].contiguous(), na)
                ot = sample_tile(model, tcond, sampler, sigmas, xt, audio, noise, seed)
                _calls += 1
                x = ot.to(device=dev, dtype=dt)
                del ot, xt
            else:
                wsum = torch.zeros((1, 1, 1, h, w), device=dev, dtype=torch.float32)
                for r0, c0, rows, cols, _tcond, m in prepared:
                    wsum[:, :, :, r0:r0 + rows, c0:c0 + cols] += m[:, :, 0]
                wsum = wsum.clamp(min=1e-8)
                for step in range(len(sigmas) - 1):
                    two = sigmas[step:step + 2]
                    acc = torch.zeros_like(x, dtype=torch.float32)
                    for r0, c0, rows, cols, tcond, m in prepared:
                        xt = x[:, :, :, r0:r0 + rows, c0:c0 + cols].contiguous()
                        if step == 0:
                            noise = (nv[:, :, :, r0:r0 + rows, c0:c0 + cols].contiguous(), na)
                        else:
                            noise = (torch.zeros_like(xt), torch.zeros_like(audio))
                        ot = sample_tile(model, tcond, sampler, two, xt, audio, noise, seed)
                        _calls += 1
                        acc[:, :, :, r0:r0 + rows, c0:c0 + cols] += \
                            ot.to(device=dev, dtype=torch.float32) * m
                        del ot, xt
                    x = (acc / wsum).to(dtype=dt)
                    del acc
                del wsum
            _dt = time.perf_counter() - _t0
            _LAST_OOM[0] = _oom_retries
            print(f"[{TAG}] chunk re-draw: {len(sigmas) - 1} step(s) x {len(prepared)} tile(s) "
                  f"({grid[0]}x{grid[1]}) in {_calls} sampler call(s), {_dt:.1f} s "
                  f"({_dt / max(1, _calls):.2f} s per call, cache flush {_FLUSH_S[0]:.1f} s), "
                  f"OOM retries {_oom_retries}"
                  + (f" (budget -> {cur_budget})" if cur_budget != int(budget) else ""),
                  flush=True)
            break
        except Exception as exc:
            if not _is_oom(exc) or cur_budget <= 16384:
                raise
            next_budget = max(16384, int(cur_budget * 0.65))
            if next_budget >= cur_budget:
                raise
            _oom_retries += 1
            print(f"[{TAG}] CUDA OOM at tile budget {cur_budget} ({exc}); "
                  f"lowering tile budget to {next_budget} (more, smaller tiles) and retrying chunk...",
                  flush=True)
            _free_all_gpu()
            cur_budget = next_budget

    if float(detail_sigma or 0.0) > 0.0:
        x = _highpass_detail_merge(video, x, sigma=float(detail_sigma))
    _LAST_BUDGET[0] = cur_budget
    return x, cur_budget


def _crossfade(a, b, dim):
    n = a.shape[dim]
    wgt = torch.linspace(0.0, 1.0, n, device=a.device, dtype=a.dtype)
    shape = [1] * a.ndim
    shape[dim] = n
    return a + (b - a) * wgt.view(shape)


def join_chunk(done_v, done_a, v, a, k0, f0):
    """Append a finished chunk, crossfading where it overlaps what is done."""
    if done_v is None:
        return v, a
    a0 = round(f0 * FRAME_RESCALE)
    out_v = torch.zeros((1, done_v.shape[1], max(done_v.shape[2], k0 + v.shape[2]),
                         done_v.shape[3], done_v.shape[4]),
                        device=done_v.device, dtype=done_v.dtype)
    out_a = torch.zeros((1, 32, 2, max(done_a.shape[-1], a0 + a.shape[-1])),
                        device=done_a.device, dtype=done_a.dtype)
    out_v[:, :, :done_v.shape[2]] = done_v
    out_a[..., :done_a.shape[-1]] = done_a
    ov = min(done_v.shape[2] - k0, v.shape[2])
    if ov > 0:
        out_v[:, :, k0:k0 + ov] = _crossfade(out_v[:, :, k0:k0 + ov].clone(),
                                             v[:, :, :ov], 2)
        v = v[:, :, ov:]
    at = k0 + max(ov, 0)
    if v.shape[2] > 0:
        out_v[:, :, at:at + v.shape[2]] = v
    ova = min(done_a.shape[-1] - a0, a.shape[-1])
    if ova > 0:
        out_a[..., a0:a0 + ova] = _crossfade(out_a[..., a0:a0 + ova].clone(),
                                             a[..., :ova], 3)
        a = a[..., ova:]
    at = a0 + max(ova, 0)
    if a.shape[-1] > 0:
        out_a[..., at:at + a.shape[-1]] = a
    return out_v, out_a


def _join_into(out_v, out_a, end_v, end_a, v, a, k0, f0):
    """Write chunk `(v, a)` into preallocated `(out_v, out_a)` with
    `join_chunk`'s exact crossfade arithmetic. -> `(end_v, end_a)`."""
    if end_v == 0 and end_a == 0:
        nv, na = int(v.shape[2]), int(a.shape[-1])
        out_v[:, :, :nv] = v
        out_a[..., :na] = a
        return nv, na
    a0 = round(f0 * FRAME_RESCALE)
    ov = min(end_v - k0, v.shape[2])
    if ov > 0:
        out_v[:, :, k0:k0 + ov] = _crossfade(out_v[:, :, k0:k0 + ov].clone(),
                                             v[:, :, :ov], 2)
        v = v[:, :, ov:]
    at = k0 + max(ov, 0)
    if v.shape[2] > 0:
        out_v[:, :, at:at + v.shape[2]] = v
    end_v = max(end_v, at + int(v.shape[2]))
    ova = min(end_a - a0, a.shape[-1])
    if ova > 0:
        out_a[..., a0:a0 + ova] = _crossfade(out_a[..., a0:a0 + ova].clone(),
                                             a[..., :ova], 3)
        a = a[..., ova:]
    at = a0 + max(ova, 0)
    if a.shape[-1] > 0:
        out_a[..., at:at + a.shape[-1]] = a
    end_a = max(end_a, at + int(a.shape[-1]))
    return end_v, end_a


def _free(model):
    import comfy.model_management as mm
    if hasattr(model, "clone_base_uuid"):
        mm.unload_model_and_clones(model, unload_additional_models=False)
        mm.soft_empty_cache()


def _regen2k_chunk_frames(w_px, h_px, budget, chunk_frames=_DEFAULT_CHUNK_FRAMES,
                          overlap_px=TILE_OVERLAP_PX, explicit=None):
    """Pick chunk length from `budget` for regen2k mode (R0).

    If `chunk_frames` was not explicitly set on `HTCUpsampleSettings` and a
    102 f full-frame chunk exceeds `budget`, shorten `chunk_frames` in steps
    of 17 down to a floor of 68 f (20 steps) before allowing spatial tiles.
    """
    if explicit is None:
        explicit = not isinstance(chunk_frames, _DefaultInt)
    cf = int(chunk_frames)
    if not explicit and cf > REGEN2K_MIN_CHUNK_FRAMES:
        while cf > REGEN2K_MIN_CHUNK_FRAMES:
            steps = max(1, (cf // 17) * 5)
            if seq_len(w_px, h_px, steps) <= int(budget):
                break
            cf = max(REGEN2K_MIN_CHUNK_FRAMES, cf - 17)
    steps = max(1, (cf // 17) * 5)
    grid = solve_grid(w_px, h_px, steps, overlap_px=snap_overlap(overlap_px),
                      budget=int(budget))
    if grid is not None and grid[0] == 1 and grid[1] == 1:
        print(f"[{TAG}] regen2k: budget {int(budget)} -> chunk {cf} f full-frame",
              flush=True)
    elif grid is not None:
        print(f"[{TAG}] regen2k: budget {int(budget)} -> chunk {cf} f "
              f"-> {grid[0]}x{grid[1]} tiles", flush=True)
    return cf, grid


def upsample(model, conditioning, latent, sampler, upscaler, megapixels,
             denoise=DENOISE, guide=None, budget=TOKEN_BUDGET, steps=STEPS,
             start_sigma=0.0, overlap_px=TILE_OVERLAP_PX, detail_sigma=0.0,
             chunk_frames=_DEFAULT_CHUNK_FRAMES, chunk_overlap=CHUNK_OVERLAP,
             anchor=ANCHOR_AUG, seed=SEED, noise="per_chunk",
             attn_chunks=4, ffn_chunks=2, sigmas=None):
    """The whole recipe on an H3 AV latent. -> the upsampled AV latent dict.

    `conditioning` is one conditioning for the whole clip, or a callable
    `(f0, f1) -> conditioning` giving each chunk its own -- the chain passes
    the latter, because each hop was generated from its own prompt.

    `guide` (a Guide) makes the re-draw a guided one: each chunk's canvas
    frames, shrunk to 1/factor of the output, ride every tile as the h3upscale
    guide, and `guide.cond` replaces `conditioning`. Same step, same sigmas.

    `budget` caps one tile forward's tokens (default: the recipe's). Lower it
    and the same frame is cut into more, smaller tiles -- less VRAM, more
    forwards, the same feathered merge.
    """
    import comfy.nested_tensor as nt

    samples = latent["samples"]
    if not (getattr(samples, "is_nested", False) and len(samples.tensors) == 2
            and samples.tensors[0].ndim == 5 and samples.tensors[0].shape[1] == 24
            and samples.tensors[1].ndim == 4 and samples.tensors[1].shape[1] == 32):
        raise RuntimeError(f"{TAG}: wants a MiniMax H3 AV latent "
                           "(video [1,24,T,H,W] + audio [1,32,2,T]).")
    video, audio = samples.tensors
    if video.shape[0] != 1:
        raise RuntimeError(f"{TAG}: one clip at a time (batch 1).")
    factor = int(guide.factor) if guide is not None else 1
    if guide is not None:
        conditioning = guide.cond
    per_chunk = callable(conditioning)
    cond = None if per_chunk else clean_refs(conditioning)
    tv, ta = video.shape[2], audio.shape[-1]
    if guide is not None:
        w_px, h_px = target_size(video.shape[4] * SPATIAL, video.shape[3] * SPATIAL,
                                 megapixels, align=32 * factor)
    else:
        w_px, h_px = target_size(video.shape[4] * SPATIAL, video.shape[3] * SPATIAL,
                                 megapixels)
    model = _apply_low_vram_patches(model, attn_chunks, ffn_chunks)
    is_regen2k = sigmas is not None
    if is_regen2k:
        sigmas = (sigmas.detach().cpu().float() if isinstance(sigmas, torch.Tensor)
                  else torch.tensor(list(sigmas), dtype=torch.float32))
        chunk_frames, _ = _regen2k_chunk_frames(w_px, h_px, budget, chunk_frames, overlap_px)
    else:
        sigmas = recipe_sigmas(model, denoise, steps, start_sigma)
    plan = chunks(tv, int(chunk_frames), int(chunk_overlap))
    print(f"[{TAG}] {video.shape[4] * SPATIAL}x{video.shape[3] * SPATIAL} -> "
          f"{w_px}x{h_px} ({w_px * h_px / 1048576.0:.2f} MP), {len(plan)} "
          f"chunk(s), "
          + (f"sigmas {[round(float(s), 4) for s in sigmas]}" if sigmas is not None
             else "no re-draw (denoise 0)")
          + (f", h3upscale guide at {w_px // factor}x{h_px // factor}"
             if guide is not None and sigmas is not None else ""), flush=True)

    # 3-Phase non-overlapping execution so VideoVAE, 3D LatentUpscaler, and the
    # 32.4 GB DynamicVRAM H3 DiT never coexist or churn in and out of VRAM
    # across 21 chunks:
    #   Phase 1: Decode & encode all `h3upscale` guide latents on VideoVAE once,
    #            store on CPU RAM (~1.5 MB/chunk), then unload VideoVAE.
    #   Phase 2: Load 3D `LatentUpscaler` onto GPU once, upscale all chunk video
    #            latents, store on CPU RAM (~6 MB/chunk), then offload upscaler.
    #   Phase 3: Stage the 32.4 GB H3 DiT onto GPU ONCE and step all chunks
    #            without unloading/re-staging the DiT between chunks.
    _tp = [time.perf_counter()]
    if _device().type == "cuda":
        _free(model)
    raw_cvs = [video[:, :, k0:k1].contiguous() for (k0, _f0, k1, _f1) in plan]
    glats = [None] * len(plan)
    if guide is not None and sigmas is not None:
        for i, cv_in in enumerate(raw_cvs):
            px = guide.decode(cv_in)[:frames_in(cv_in.shape[2])]
            glat = encode_steps(guide.encode, guide.pixels(px, w_px, h_px, factor),
                                int(cv_in.shape[2]))
            glats[i] = glat.cpu() if hasattr(glat, "cpu") else glat
            del px, glat
        if _device().type == "cuda":
            _free_all_gpu()

    _tp.append(time.perf_counter())
    upscaled_cvs = upscale_chunks_batch(raw_cvs, upscaler, w_px, h_px)
    del raw_cvs
    _tp.append(time.perf_counter())
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    film_nv = None
    _c0 = upscaled_cvs[0]
    if str(noise) == "film" and sigmas is not None and upscaled_cvs:
        film_nv = torch.randn(
            (1, int(_c0.shape[1]), tv, int(_c0.shape[3]), int(_c0.shape[4])),
            generator=torch.Generator().manual_seed(int(seed)),
            dtype=torch.float32, device="cpu").to(dtype=_c0.dtype)
    total_a = max((a0 + max(0, min(a1, int(ta)) - a0)
                   for (_k0, f0, _k1, f1) in plan
                   for a0, a1 in (audio_span(f0, f1),)), default=0)
    out_v = torch.zeros((1, int(_c0.shape[1]), int(tv), int(_c0.shape[3]), int(_c0.shape[4])),
                        dtype=_c0.dtype, device=_c0.device)
    out_a = torch.zeros((1, 32, 2, int(total_a)), dtype=audio.dtype, device=audio.device)
    end_v = end_a = 0
    _redraw_s = _join_s = 0.0
    _oom_total = 0
    init_budget = cur_budget = int(budget)
    for i, (k0, f0, k1, f1) in enumerate(plan):
        a0, a1 = audio_span(f0, f1)
        cv = upscaled_cvs[i]
        upscaled_cvs[i] = None
        ca = audio[:, :, :, a0:min(a1, ta)].contiguous()
        if sigmas is None:
            _tj = time.perf_counter()
            end_v, end_a = _join_into(out_v, out_a, end_v, end_a, cv, ca, k0, f0)
            _join_s += time.perf_counter() - _tj
            continue
        ccond = cond_for_chunk(clean_refs(conditioning(f0, f1)) if per_chunk
                               else cond, f0, f1, cv.shape[3], cv.shape[4])
        if i > 0 and not is_regen2k:
            ccond = pin_chunk_start(ccond, out_v[:, :, :end_v], f0, anchor)
        glat = glats[i]
        glats[i] = None
        if glat is not None:
            want = (int(cv.shape[2]), int(cv.shape[3]) // factor, int(cv.shape[4]) // factor)
            got = tuple(int(x) for x in glat.shape[2:])
            if got != want:
                raise RuntimeError(f"{TAG}: guide latent {got} for a chunk that "
                                   f"needs {want} (steps, rows, cols)")
            ccond = add_guide(ccond, glat.to(cv.device, cv.dtype), factor)
            del glat
        if cur_budget < init_budget:
            _gi = solve_grid(int(cv.shape[4]) * SPATIAL, int(cv.shape[3]) * SPATIAL,
                             int(cv.shape[2]), overlap_px=snap_overlap(overlap_px),
                             budget=init_budget, factor=factor)
            _gc = solve_grid(int(cv.shape[4]) * SPATIAL, int(cv.shape[3]) * SPATIAL,
                             int(cv.shape[2]), overlap_px=snap_overlap(overlap_px),
                             budget=cur_budget, factor=factor)
            if _gi is not None and _gc is not None and _gi[:2] != _gc[:2]:
                print(f"[{TAG}] chunk {i + 1}/{len(plan)}: keeping lowered budget "
                      f"{cur_budget} ({_gc[0]}x{_gc[1]} tiles; would fit "
                      f"{_gi[0]}x{_gi[1]} at {init_budget})", flush=True)
        _tr = time.perf_counter()
        _rc = redraw_chunk(model, ccond, sampler, sigmas, cv, ca, int(seed), factor,
                           cur_budget, overlap_px, detail_sigma=detail_sigma,
                           noise_video=(None if film_nv is None
                                        else film_nv[:, :, k0:k1]))
        rv, cur_budget = (_rc if isinstance(_rc, tuple) else (_rc, cur_budget))
        _redraw_s += time.perf_counter() - _tr
        _oom_total += _LAST_OOM[0]
        if i == 0:
            _, _g0 = tile_grid(int(cv.shape[3]), int(cv.shape[4]), int(cv.shape[2]),
                               factor, cur_budget, overlap_px)
            _tok0 = (seq_len(_g0[2], _g0[3], int(cv.shape[2])) if int(factor) <= 1
                     else guided_seq_len(_g0[2], _g0[3], int(cv.shape[2]), factor))
            _vram0 = (f"{torch.cuda.max_memory_allocated() / 1048576.0:.0f} MB"
                      if torch.cuda.is_available() else "0 MB")
            print(f"[{TAG}] chunk 1: peak VRAM {_vram0}, {_tok0} tokens/forward, "
                  f"budget {cur_budget}"
                  + (f"; set upsample_tile_tokens to {cur_budget} to skip the OOM next run"
                     if cur_budget < init_budget else ""),
                  flush=True)
        _tj = time.perf_counter()
        end_v, end_a = _join_into(out_v, out_a, end_v, end_a, rv, ca, k0, f0)
        _join_s += time.perf_counter() - _tj
        del cv, rv
    _LAST_BUDGET[0] = cur_budget
    if end_v < out_v.shape[2]:
        out_v = out_v[:, :, :end_v]
    if end_a < out_a.shape[-1]:
        out_a = out_a[..., :end_a]
    assert end_v == int(tv) and end_a == int(total_a), (end_v, tv, end_a, total_a)
    _peak_mb = torch.cuda.max_memory_allocated() / 1048576.0 if torch.cuda.is_available() else 0.0
    _lat_mb = (out_v.numel() * out_v.element_size()
               + out_a.numel() * out_a.element_size()) / 1048576.0
    _free(model)
    _tp.append(time.perf_counter())
    print(f"[{TAG}] timing: guide prep {_tp[1] - _tp[0]:.1f} s, 3D upscale "
          f"{_tp[2] - _tp[1]:.1f} s, re-draw {_redraw_s:.1f} s, join {_join_s:.1f} s "
          f"({len(plan)} chunk(s), OOM retries {_oom_total}, "
          f"joined latent {_lat_mb:.1f} MB"
          + (f", peak VRAM {_peak_mb:.0f} MB" if torch.cuda.is_available() else "")
          + ")", flush=True)
    return {"samples": nt.NestedTensor((out_v, out_a))}


# --- inside the chain -------------------------------------------------------
#
# HandTieClips runs the recipe once over the WHOLE film after its last hop,
# not per hop: a hop re-drawn on its own would put a fresh texture step at
# every join, while one pass over the joined film lets the recipe's own
# 17-frame chunk crossfades run across the joins. It is delivery-only -- the
# chain has handed every pin on before this starts, so nothing here can reach
# what a hop inherits.

def segments(hops):
    """Split the chain's hops at restarts. A restart is a cut by design and
    opens on frame 0 of its own VAE cycle, so it cannot be joined to what came
    before on the latent grid -- each run between restarts is upsampled alone.
    `hops` are dicts with a `start` flag. -> [[hop, ...], ...]"""
    out = []
    for h in hops:
        if h["start"] or not out:
            out.append([])
        out[-1].append(h)
    return out


def stitch(seg):
    """One continuous latent for a run of hops. -> (video, audio).

    Each hop after the first drops the `drop` frames the master dropped (its
    regenerated overlap). That only joins cleanly on the latent grid when the
    drop is a whole number of latent steps AND the first kept step sits at the
    phase the running latent is at -- otherwise a 1-frame key step would be
    read as a 4-frame one and the timing would slip. Every H3 hop is 17k+5
    frames and every shipped overlap (5 / 22 / 39) meets both, but it is
    checked rather than assumed, with a message instead of wrong frames.
    """
    vids, auds, steps = [], [], 0
    for n, h in enumerate(seg):
        v, a = h["video"], h["audio"]
        if v is None or a is None:
            raise ValueError(f"hop {h['hop']} has no stored sampler latent "
                             "(a cache entry from an older build)")
        v = v if v.ndim == 5 else v.unsqueeze(0)
        a = a if a.ndim == 4 else a.unsqueeze(0)
        drop = 0 if n == 0 else int(h["drop"])
        k = tokens_to_reach(drop)
        if frames_in(k) != drop:
            raise ValueError(f"hop {h['hop']}: an overlap of {drop} frames is not "
                             "a whole number of latent steps")
        if n and k % 5 != steps % 5:
            raise ValueError(f"hop {h['hop']}: its kept frames start at VAE "
                             f"phase {k % 5}, the joined latent is at "
                             f"{steps % 5}")
        # Hand-tie cuts the previous hop's tail instead of this hop's head.
        tail = int(h.get("tail") or 0)
        end = int(v.shape[2])
        if tail:
            end = tokens_to_reach(frames_in(end) - tail)
            if frames_in(end) != frames_in(int(v.shape[2])) - tail:
                raise ValueError(f"hop {h['hop']}: a tail cut of {tail} frames is "
                                 "not a whole number of latent steps")
        kept = v[:, :, k:end]
        vids.append(kept)
        auds.append(a[..., round(drop * FRAME_RESCALE):
                      int(a.shape[-1]) - round(tail * FRAME_RESCALE)])
        steps += int(kept.shape[2])
    return torch.cat(vids, dim=2), torch.cat(auds, dim=-1)


def cond_by_frame(seg):
    """`(f0, f1) -> conditioning`: the conditioning of the hop that holds the
    chunk's middle frame, on the joined run's frame axis."""
    spans, at = [], 0
    for h in seg:
        spans.append((at, at + int(h["frames"]), h["cond"]))
        at += int(h["frames"])

    def pick(f0, f1):
        mid = (int(f0) + int(f1)) // 2
        for a, b, c in spans:
            if a <= mid < b:
                return c
        return spans[-1][2]
    return pick


def match_levels(frames, ref, chunk=16, in_place=False):
    """Each upsampled frame's per-channel mean moved onto the matching frame
    of `ref` (the chain's tone-corrected master at the canvas size), clamped
    to 0..1.

    `tone_compensate` corrects decoded pixels, and the upsampler decodes its
    own. Every mode's correction is (or is dominated by) a per-frame level
    shift, which does not depend on resolution, so a mean match carries it
    over -- and also takes back any level the light re-draw moved. It is not
    a contrast match: lut / gain_bias slopes are not reproduced.
    """
    out = frames if in_place else torch.empty_like(frames)
    for s in range(0, int(frames.shape[0]), chunk):
        up = frames[s:s + chunk].float()
        rf = ref[s:s + chunk].float()
        shift = rf.mean(dim=(1, 2), keepdim=True) - up.mean(dim=(1, 2), keepdim=True)
        out[s:s + chunk] = (up + shift).clamp_(0.0, 1.0).to(frames.dtype)
    return out


def upsample_chain(model, sampler, vae, hops, megapixels, master=None,
                   upscaler=UPSCALER_FILE, decode=None, denoise=DENOISE,
                   alloc=None, guide=None, budget=TOKEN_BUDGET, settings=None,
                   sigmas=None):
    """The recipe over a finished chain. -> (IMAGE [N,H,W,3], note).

    `hops` are per-hop dicts: video/audio (the hop's delivered sampler
    latent), cond, start, drop, frames, hop. `master`, when given, is the
    tone-corrected canvas master to match levels against. `decode` turns a
    video latent into frames (the chain passes its VAE); left None it uses
    `vae.decode`. `alloc(n, h, w)` makes the film buffer (the chain passes
    its master allocator, which spills to disk); left None it is float32 RAM.

    Each run is written into one preallocated film and level-matched in
    place. A 2 MP, 2877 f film is ~72 GB in float32; gathering the runs with
    `torch.cat` and matching into `empty_like` held three of those at once
    and choked a 64 GB machine at the very end of a two-minute render.
    """
    import comfy.nested_tensor as nt
    if decode is None:
        def decode(v):
            img = vae.decode(v)
            return img.reshape(-1, img.shape[-3], img.shape[-2], img.shape[-1])
    if alloc is None:
        def alloc(n, h, w):
            return torch.empty((n, h, w, 3), dtype=torch.float32)
    if guide is not None and guide.decode is None:
        guide = guide._replace(decode=decode)
    st = resolve_settings(settings)
    if st["sampler"] is not None:
        sampler = st["sampler"]
    model = _apply_low_vram_patches(model, st["attn_chunks"], st["ffn_chunks"])
    # Unwired settings run the original recipe: one step from sigma 0.75
    # (`upsample_denoise=0.2` under H3's `shift=12` curve), 128 px tile overlap.
    # 3.0.0 ran 2 steps at 0.1 with 192 px; HTCUpsampleSettings can still.
    _chain_steps = int(st["steps"]) if float(denoise) > 0.0 else STEPS
    want = sum(int(h["frames"]) for h in hops)
    frames, at, runs = None, 0, 0
    _segs = segments(hops)
    _cur_budget = budget
    for seg in _segs:
        _t0 = time.perf_counter()
        v, a = stitch(seg)
        lat = {"samples": nt.NestedTensor((v, a))}
        if _cur_budget is not None:
            _LAST_BUDGET[0] = int(_cur_budget)
        _ukw = dict(
            steps=_chain_steps,
            start_sigma=(st["start_sigma"] if float(denoise) > 0.0 else 0.0),
            overlap_px=st["tile_overlap_px"], detail_sigma=st["detail_sigma"],
            chunk_frames=st["chunk_frames"], chunk_overlap=st["chunk_overlap"],
            anchor=st["anchor_strength"], seed=st["seed"], noise=st["noise"],
            attn_chunks=st["attn_chunks"], ffn_chunks=st["ffn_chunks"],
        )
        if sigmas is not None:
            _ukw["sigmas"] = sigmas
        out = upsample(model, cond_by_frame(seg), lat, sampler, upscaler,
                       megapixels, denoise, guide=guide, budget=_cur_budget,
                       **_ukw)
        if _cur_budget is not None and _LAST_BUDGET[0] < int(_cur_budget):
            _cur_budget = int(_LAST_BUDGET[0])
        vid = out["samples"].tensors[0]
        _free(model)
        if _device().type == "cuda":
            _free_all_gpu()
        _t1 = time.perf_counter()
        tv_seg = int(vid.shape[2])
        if vae is not None and tv_seg > 32:
            # Stream-decode in 30-token (102-frame) windows with a 5-token (17-frame)
            # causal warm-up prefix so ViT3DDecoder never allocates a full-film
            # 22-44 GB float32 tensor in RAM at once, writing each slice directly
            # into the disk-backed `frames` memmap.
            del out
            for k0 in range(0, tv_seg, 30):
                k1 = min(tv_seg, k0 + 30)
                k_in_start = max(0, k0 - 5) if k0 > 0 else 0
                k_in_end = min(tv_seg, k1 + 2) if k1 < tv_seg else tv_seg
                sub_z = vid[:, :, k_in_start:k_in_end].contiguous()
                part = decode(sub_z).float().cpu()
                del sub_z
                drop_head = 17 if k0 > 0 else 0
                keep_len = frames_in(k1) - frames_in(k0)
                part = part[drop_head:drop_head + keep_len]
                n = int(part.shape[0])
                if at + n > want:
                    raise ValueError(f"the upsampled film is at least {at + n} frames, "
                                     f"the chain delivered {want}")
                if frames is None:
                    frames = alloc(want, int(part.shape[1]), int(part.shape[2]))
                if master is not None:
                    part = match_levels(part, master[at:at + n], in_place=True)
                frames[at:at + n] = part.to(frames.dtype)
                del part
                at += n
            del vid
        else:
            part = decode(vid).float().cpu()
            del out, vid
            n = int(part.shape[0])
            if at + n > want:
                raise ValueError(f"the upsampled film is at least {at + n} frames, "
                                 f"the chain delivered {want}")
            if frames is None:
                frames = alloc(want, int(part.shape[1]), int(part.shape[2]))
            if master is not None:
                part = match_levels(part, master[at:at + n], in_place=True)
            frames[at:at + n] = part.to(frames.dtype)
            del part
            at += n
        runs += 1
        print(f"[{TAG}] run {runs}/{len(_segs)} (hop {seg[0]['hop']}"
              + (f"-{seg[-1]['hop']}" if len(seg) > 1 else "")
              + f"): upsample {_t1 - _t0:.1f} s, decode"
              + (" + level match" if master is not None else "")
              + f" {time.perf_counter() - _t1:.1f} s", flush=True)
    if at != want:
        raise ValueError(f"the upsampled film is {at} frames, "
                         f"the chain delivered {want}")
    note = (f"upsampled {int(frames.shape[2])}x{int(frames.shape[1])} "
            f"({frames.shape[1] * frames.shape[2] / 1048576.0:.2f} MP), "
            f"{runs} run(s)" + (", h3upscale guided" if guide is not None else ""))
    return frames, note


class HTCTiledUpsampler:
    """A finished H3 clip upscaled in latent space and lightly re-drawn in
    tiles, then decoded."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The H3 model the clip was made with."}),
                "conditioning": ("CONDITIONING", {
                    "tooltip": "The conditioning the clip was made with."}),
                "latent": ("LATENT", {"tooltip": "The finished H3 AV latent."}),
                "sampler": ("SAMPLER", {"tooltip": "The sampler for the light re-draw."}),
                "vae": ("VAE", {"tooltip": "The H3 video VAE."}),
                "audio_vae": ("VAE", {"tooltip": "The H3 audio VAE."}),
                "upscale_model": (choices(), {
                    "tooltip": "From models/latent_upscale_models. The recipe uses "
                               f"{UPSCALER_FILE}."}),
                "megapixels": ("FLOAT", {
                    "default": MEGAPIXELS, "min": 0.1, "max": 16.0, "step": 0.05,
                    "tooltip": "Output size as area, aspect kept (1 MP = 1024x1024). "
                               "1.5-2 is the usual range."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO")
    FUNCTION = "run"
    CATEGORY = "Hand Tie Clips"
    DESCRIPTION = (
        "Makes a finished MiniMax H3 clip bigger and sharper: the H3 latent "
        "upscaler per time chunk, then one light re-draw step in tiles that are "
        "stepped together, so there are no tile seams. The recipe is fixed; "
        "only the size is a widget.")

    def run(self, model, conditioning, latent, sampler, vae, audio_vae,
            upscale_model, megapixels):
        import nodes
        from comfy_extras.nodes_audio import vae_decode_audio
        out = upsample(model, conditioning, latent, sampler, str(upscale_model),
                       float(megapixels))
        images = nodes.VAEDecode().decode(vae, out)[0]
        return (images, vae_decode_audio(audio_vae, out))


# --- the chain's upsample settings -------------------------------------------
#
# HandTieClips' upsample_settings socket. Unwired, the chain runs exactly
# these defaults: the original recipe (one simple step, 102 / 17 f chunks,
# anchor 0.999, 70000-token tiles with 128 px overlap, seed 444). 3.0.0 ran
# 2 steps at 0.1 with 192 px; measured 10-25 on one 576p -> 2 MP clip, the
# recipe's single step from 0.75 kept 14% more fine detail.

SETTINGS_TYPE = "HTC_UPSAMPLE"
CHAIN_DEFAULTS = {
    "steps": STEPS, "start_sigma": 0.0, "noise": "per_chunk", "seed": SEED,
    "chunk_frames": _DEFAULT_CHUNK_FRAMES, "chunk_overlap": CHUNK_OVERLAP,
    "anchor_strength": ANCHOR_AUG, "tile_overlap_px": TILE_OVERLAP_PX, "detail_sigma": 0.0,
    "attn_chunks": 4, "ffn_chunks": 2, "sampler": None,
}


def resolve_settings(settings):
    """The settings node's dict over the chain defaults. -> a full dict."""
    out = dict(CHAIN_DEFAULTS)
    for k, v in (settings or {}).items():
        if k in out:
            out[k] = v
    if int(out["chunk_overlap"]) >= int(out["chunk_frames"]):
        raise ValueError(f"{TAG}: chunk_overlap ({out['chunk_overlap']}) must be "
                         f"smaller than chunk_frames ({out['chunk_frames']})")
    return out


class HTCUpsampleSettings:
    """Every setting of the chain's upsample, for HandTieClips' upsample_settings."""

    @classmethod
    def INPUT_TYPES(cls):
        d = CHAIN_DEFAULTS
        return {
            "required": {
                "steps": ("INT", {
                    "default": 3, "min": 1, "max": 8,
                    "tooltip": "Re-draw steps per chunk. 3 = the Regen2K 3-step schedule "
                               "[0.9094, 6/7, 2/3, 0]; 2 = the 2-step schedule [0.7, 0.4941, 0]; "
                               "1 = 1-step schedule."}),
                "start_sigma": ("FLOAT", {
                    "default": d["start_sigma"], "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "0 = the Regen2K schedule ([0.9094, 6/7, 2/3, 0] at 3 steps, or "
                               "[0.7, 0.4941, 0] at 2 steps). Above 0 the re-draw starts at this "
                               "sigma and follows the model's schedule down to 0 in `steps` steps."}),
                "noise": (["per_chunk", "film"], {
                    "default": "per_chunk",
                    "tooltip": "per_chunk: every chunk draws from its own first frame "
                               "(Regen2K default). film: one noise field for the whole "
                               "film, so the frames two chunks share get the same noise "
                               "from both sides."}),
                # Not named seed: ComfyUI hangs a randomize control on any
                # widget called seed, and the seed must stay fixed.
                "redraw_seed": ("INT", {
                    "default": REGEN2K_SEED, "min": 0, "max": 0x7FFFFFFF,
                    "tooltip": "Noise seed of the re-draw (Regen2K default: 42). Fixed: "
                               "it does not change between runs."}),
                "chunk_frames": ("INT", {
                    "default": int(d["chunk_frames"]), "min": 34, "max": 1020, "step": 17,
                    "tooltip": "Frames per time chunk, a multiple of 17. Longer chunks "
                               "mean fewer joins but more tokens per pass (more tiles "
                               "or more VRAM)."}),
                "chunk_overlap": ("INT", {
                    "default": d["chunk_overlap"], "min": 17, "max": 1003, "step": 17,
                    "tooltip": "Frames two chunks share and crossfade over, a multiple "
                               "of 17, smaller than chunk_frames."}),
                "anchor_strength": ("FLOAT", {
                    "default": d["anchor_strength"], "min": 0.0, "max": 1.0, "step": 0.001,
                    "tooltip": "How hard each chunk's first frame is pinned to what the "
                               "previous chunk drew there. 0.999 = H3's default."}),
                "tile_overlap_px": ("INT", {
                    "default": d["tile_overlap_px"], "min": 0, "max": 512, "step": 32,
                    "tooltip": "Overlap between spatial tiles, in pixels (128 = the "
                               "recipe). Only matters when a chunk splits into tiles."}),
                "detail_sigma": ("FLOAT", {
                    "default": d["detail_sigma"], "min": 0.0, "max": 32.0, "step": 0.5,
                    "tooltip": "0 = off. Above 0, keeps the 3D upscaler's large shapes "
                               "(blurred at this many latent px) and takes only the finer "
                               "detail from the re-draw. For tile-grid shifts, not a "
                               "sharpness control."}),
                "attn_chunks": ("INT", {
                    "default": d["attn_chunks"], "min": 1, "max": 16,
                    "tooltip": "Attention split into this many head groups to save VRAM. "
                               "Keep 4. Lower needs far more VRAM: 1 / 1 on a 32 GB card "
                               "at 1-2 MP spills into system RAM and stalls the PC. Same "
                               "result either way."}),
                "ffn_chunks": ("INT", {
                    "default": d["ffn_chunks"], "min": 1, "max": 16,
                    "tooltip": "Feed-forward split into this many token groups to save "
                               "VRAM. Keep 2 or raise it on smaller cards; 1 needs far "
                               "more VRAM. Same result either way."}),
            },
            "optional": {
                "sampler": ("SAMPLER", {
                    "tooltip": "Sampler for the re-draw. Unwired uses euler for Regen2K. "
                               "Multistep samplers only keep their history when a chunk "
                               "is one tile."}),
            },
        }

    RETURN_TYPES = (SETTINGS_TYPE,)
    RETURN_NAMES = ("upsample_settings",)
    FUNCTION = "build"
    CATEGORY = "Hand Tie Clips"
    DESCRIPTION = ("Settings for HandTieClips' upsample. Wire into its upsample_settings "
                   "socket; unwired, the chain uses these defaults.")

    def build(self, sampler=None, **kw):
        out = dict(kw)
        out["seed"] = out.pop("redraw_seed", REGEN2K_SEED)
        if out.get("chunk_frames") == CHUNK_FRAMES:
            out["chunk_frames"] = _DEFAULT_CHUNK_FRAMES
        out["sampler"] = sampler
        resolve_settings(out)
        return (out,)


NODE_CLASS_MAPPINGS = {"HTCTiledUpsampler": HTCTiledUpsampler,
                       "HTCUpsampleSettings": HTCUpsampleSettings}
NODE_DISPLAY_NAME_MAPPINGS = {"HTCTiledUpsampler": "H3 Tiled Upsampler",
                              "HTCUpsampleSettings": "HTC Upsample Settings"}
