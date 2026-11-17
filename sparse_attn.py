# Generated from ComfyUI-Hand-Tie-Clips (https://github.com/sandpies/ComfyUI-Hand-Tie-Clips) by tools/build_upsampler_pack.py.
# Edit it there, not here: the next build overwrites this file.
"""Clean-room MIT MiniMax-H3 Sparse & Low-VRAM Attention (`HTCSparseAttention`).

Built directly on Comfy Kitchen (`comfy_kitchen.sol_attn` & `sol_attn_chunked`)
and ComfyUI Core with zero external AGPL/GPL pack dependencies:
  * Streaming 4096-row chunked QKV projection (`ck.sol_attn_chunked`) so full
    sequence QKV tensors are never materialized in VRAM, with per-branch
    `kmean`/`vscale` statistics reused across steps.
  * Pooled-tail compensation (`tail=True` by default) + exact packed prefix rows
    (`exact_prefix_rows` keeping text, reference images, and audio exact).
  * Track-aware presets for Turbo LoRAs (`turbo_sla_balanced`, `turbo_sla_quality`,
    `turbo_sla_fast`), training-free native checkpoints (`native_sol_balanced`,
    `native_sol_fast`), FastH3-VSA (`vsa_10`), and pure Low-VRAM head-chunked
    dense attention (`dense_low_vram`, usable standalone or by `tiled_upsampler.py`).
  * Safe `double_block` + `optimized_attention_override` chaining so it composes
    cleanly with `HTCSteeringDials`, `HTCTokenLoRA` (including VSA tile order),
    `FunControl`, and `HandTieClips`'s built-in `h3_cache.py`.
"""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
from functools import partial
import logging
import math
import re
import weakref

import torch

TAG = "HTCSparseAttention"
_LOG = logging.getLogger(TAG)

HEAD_DIM = 128
BLOCK_SIZE = 64
PRODUCER_CHUNK = 4096
VSA_CUBE = (4, 4, 4)
VSA_PLAN_CACHE = 4

PRESETS = (
    "turbo_sla_balanced",
    "turbo_sla_quality",
    "turbo_sla_fast",
    "native_sol_balanced",
    "native_sol_fast",
    "vsa_10",
    "dense_low_vram",
    "manual",
)
METHODS = ("sla", "sol-attn", "vsa", "dense")
SINK_MODES = ("exact_prefix_rows", "exact_kv_and_rows", "exact_kv", "off")
DENSE_BACKENDS = ("comfy_kitchen_int8", "pytorch", "auto")

PRESET_CONFIGS: dict[str, dict] = {
    "turbo_sla_balanced": {
        "method": "sla",
        "keep_percent": 20.0,
        "keep_percent_end": -1.0,
        "tail": True,
        "extra_tokens": 0,
        "sink_mode": "exact_prefix_rows",
        "dense_first_steps": 1,
        "dense_first_percent": 0.0,
        "dense_last_steps": 1,
        "dense_layers": "0",
        "chunked_qkv": True,
        "dense_head_chunks": 1,
        "disable_fp16_accum": True,
    },
    "turbo_sla_quality": {
        "method": "sla",
        "keep_percent": 30.0,
        "keep_percent_end": -1.0,
        "tail": True,
        "extra_tokens": 0,
        "sink_mode": "exact_prefix_rows",
        "dense_first_steps": 0,
        "dense_first_percent": 0.25,
        "dense_last_steps": 1,
        "dense_layers": "0,49",
        "chunked_qkv": True,
        "dense_head_chunks": 1,
        "disable_fp16_accum": True,
    },
    "turbo_sla_fast": {
        "method": "sla",
        "keep_percent": 14.0,
        "keep_percent_end": 20.0,
        "tail": True,
        "extra_tokens": 0,
        "sink_mode": "exact_prefix_rows",
        "dense_first_steps": 1,
        "dense_first_percent": 0.0,
        "dense_last_steps": 1,
        "dense_layers": "0",
        "chunked_qkv": True,
        "dense_head_chunks": 1,
        "disable_fp16_accum": True,
    },
    "native_sol_balanced": {
        "method": "sol-attn",
        "tau": 1.35,
        "tau_end": -1.0,
        "tail": True,
        "extra_tokens": 0,
        "sink_mode": "exact_prefix_rows",
        "dense_first_steps": 0,
        "dense_first_percent": 0.20,
        "dense_last_steps": 1,
        "dense_layers": "0",
        "chunked_qkv": True,
        "dense_head_chunks": 1,
        "disable_fp16_accum": True,
    },
    "native_sol_fast": {
        "method": "sol-attn",
        "tau": 1.60,
        "tau_end": -1.0,
        "tail": True,
        "extra_tokens": 0,
        "sink_mode": "exact_prefix_rows",
        "dense_first_steps": 0,
        "dense_first_percent": 0.15,
        "dense_last_steps": 1,
        "dense_layers": "0",
        "chunked_qkv": True,
        "dense_head_chunks": 1,
        "disable_fp16_accum": True,
    },
    "vsa_10": {
        "method": "vsa",
        "keep_percent": 10.0,
        "keep_percent_end": -1.0,
        "tail": False,
        "extra_tokens": 0,
        "sink_mode": "exact_kv_and_rows",
        "dense_first_steps": 0,
        "dense_first_percent": 0.0,
        "dense_last_steps": 0,
        "dense_layers": "",
        "chunked_qkv": True,
        "dense_head_chunks": 1,
        "disable_fp16_accum": True,
    },
    "dense_low_vram": {
        "method": "dense",
        "chunked_qkv": False,
        "dense_head_chunks": 4,
        "disable_fp16_accum": False,
    },
}


def parse_block_list(text: str, n_layers: int | None = None) -> frozenset[int]:
    """Parse `'0, 1, 47-49'` -> `frozenset({0, 1, 47, 48, 49})`."""
    raw = str(text or "").strip()
    if not raw:
        return frozenset()
    leftover = re.sub(r"\d+\s*-\s*\d+|\d+|[\s,;]+", "", raw)
    if leftover:
        raise ValueError(f"{TAG}: invalid dense_layers {raw!r} (unexpected {leftover!r})")
    blocks: set[int] = set()
    for part in re.findall(r"\d+\s*-\s*\d+|\d+", raw):
        if "-" in part:
            a, b = (int(x) for x in part.split("-"))
            blocks.update(range(min(a, b), max(a, b) + 1))
        else:
            blocks.add(int(part))
    if n_layers is not None:
        bad = [i for i in blocks if i >= n_layers]
        if bad:
            raise ValueError(f"{TAG}: dense_layers {bad} out of range (model has {n_layers} blocks)")
    return frozenset(blocks)


def _scalar(val) -> float | None:
    if val is None:
        return None
    if torch.is_tensor(val):
        return float(val.detach().reshape(-1)[0].item())
    if isinstance(val, (list, tuple)):
        return _scalar(val[0]) if val else None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def resolve_step(transformer_options: dict) -> tuple[int | None, int]:
    """Return `(0-based step index, total_steps)` from `sample_sigmas` / `sigmas`."""
    sample = transformer_options.get("sample_sigmas")
    current = transformer_options.get("sigmas")
    if sample is None or current is None:
        return None, 0
    if torch.is_tensor(sample):
        sched = sample.detach().reshape(-1)
        n_steps = int(sched.numel()) - 1
        if n_steps <= 0:
            return None, 0
        cur = current.detach().reshape(-1)[0].to(sched.device) if torch.is_tensor(current) else _scalar(current)
        if cur is None:
            return None, n_steps
        idx = int((sched[:n_steps] - cur).abs().argmin().item())
        return idx, n_steps
    if isinstance(sample, (list, tuple)):
        vals = [float(v) for v in sample]
        n_steps = len(vals) - 1
        if n_steps <= 0:
            return None, 0
        cur = _scalar(current)
        if cur is None:
            return None, n_steps
        idx = min(range(n_steps), key=lambda i: abs(vals[i] - cur))
        return idx, n_steps
    return None, 0


def schedule_progress(step0: int | None, n_steps: int) -> float:
    if step0 is None or n_steps <= 1:
        return 0.0
    return max(0.0, min(1.0, float(step0) / float(n_steps - 1)))


def _lerp(a: float, b: float, t: float) -> float:
    return float(a) + (float(b) - float(a)) * float(t)


def compute_sinks(transformer_options: dict, tokens: int, sink_mode: str) -> tuple[tuple[int, int], tuple[int, int]] | None:
    """Compute `(sink_kv_blocks, sink_query_blocks)` protecting prefix / audio rows."""
    if sink_mode == "off":
        return (0, 0), (0, 0)
    layout = transformer_options.get("minimax_h3_layout")
    if layout is None or getattr(layout, "seq_len", None) != tokens:
        return None
    video = next(((a, b) for a, b, kind in layout.segments if kind == "video"), None)
    if video is None or video[0] <= 0:
        return (0, 0), (0, 0)
    pref_end = (video[0] + BLOCK_SIZE - 1) // BLOCK_SIZE
    pref_range = (0, pref_end)
    if sink_mode == "exact_prefix_rows":
        return pref_range, pref_range
    if sink_mode == "exact_kv":
        return pref_range, (0, 0)
    # exact_kv_and_rows
    audio = next(((a, b) for a, b, kind in layout.segments if kind == "audio"), None)
    if audio is None:
        return pref_range, pref_range
    return pref_range, (audio[0] // BLOCK_SIZE, pref_end)


@contextmanager
def reduction_policy(disable_fp16_accum: bool):
    if not disable_fp16_accum or not torch.cuda.is_available():
        yield
        return
    backend = torch.backends.cuda.matmul
    names = ("allow_fp16_reduced_precision_reduction", "allow_bf16_reduced_precision_reduction")
    saved = {}
    try:
        for name in names:
            if not hasattr(backend, name):
                continue
            val = getattr(backend, name)
            sk = getattr(backend, name + "_split_k", None)
            saved[name] = val if sk is None else (val, sk)
            setattr(backend, name, False)
        yield
    finally:
        for name, val in saved.items():
            setattr(backend, name, val)


def head_chunked_attention(q, k, v, heads: int, head_chunks: int, dense_fn, **kwargs):
    """Compute exact dense attention in `head_chunks` slices to cap peak VRAM."""
    chunks = max(1, int(head_chunks))
    if chunks <= 1 or heads % chunks != 0:
        return dense_fn(q, k, v, heads, **kwargs)
    skip_reshape = bool(kwargs.get("skip_reshape", False))
    skip_out = bool(kwargs.get("skip_output_reshape", False))
    dim_head = q.shape[-1] if skip_reshape else (q.shape[-1] // heads)
    sub_h = heads // chunks
    kw_inner = dict(kwargs)
    kw_inner["skip_reshape"] = True
    kw_inner["skip_output_reshape"] = True
    if skip_reshape:
        # [B, H, N, D]
        outs = []
        for g in range(chunks):
            h0, h1 = g * sub_h, (g + 1) * sub_h
            outs.append(dense_fn(q[:, h0:h1], k[:, h0:h1], v[:, h0:h1], sub_h, **kw_inner))
        res = torch.cat(outs, dim=1)
        return res if skip_out else res.transpose(1, 2).reshape(q.shape[0], q.shape[2], heads * dim_head)
    # [B, N, H*D]
    b, n, _ = q.shape
    qs = q.view(b, n, heads, dim_head)
    ks = k.view(b, -1, heads, dim_head)
    vs = v.view(b, -1, heads, dim_head)
    outs = []
    for g in range(chunks):
        h0, h1 = g * sub_h, (g + 1) * sub_h
        og = dense_fn(
            qs[:, :, h0:h1].transpose(1, 2),
            ks[:, :, h0:h1].transpose(1, 2),
            vs[:, :, h0:h1].transpose(1, 2),
            sub_h,
            **kw_inner,
        )
        outs.append(og.transpose(1, 2))
    out = torch.cat(outs, dim=2)
    return out.transpose(1, 2) if skip_out else out.reshape(b, n, heads * dim_head)


class SparseAttnPatch:
    """Immutable-config + `_`-prefixed runtime state for `_model_fingerprint` stability."""

    def __init__(
        self,
        *,
        preset: str,
        method: str,
        keep_percent: float,
        keep_percent_end: float,
        tau: float,
        tau_end: float,
        tail: bool,
        extra_tokens: int,
        sink_mode: str,
        dense_first_steps: int,
        dense_first_percent: float,
        dense_last_steps: int,
        dense_layers: frozenset[int],
        min_tokens: int,
        chunked_qkv: bool,
        dense_head_chunks: int,
        dense_backend: str,
        disable_fp16_accum: bool,
        sigma_start: float,
        sigma_end: float,
        verbose: bool,
        dense_attention,
    ):
        # Public immutable configuration fields (seen by h3_ref_chain._model_fingerprint)
        self.preset = str(preset)
        self.method = str(method)
        self.vsa = (self.method == "vsa")
        self.keep_ratio = float(keep_percent) / 100.0
        self.keep_ratio_end = (float(keep_percent_end) / 100.0) if float(keep_percent_end) >= 0.5 else None
        self.tau = float(tau)
        self.tau_end = float(tau_end) if float(tau_end) >= 0.0 else None
        self.tail = bool(tail)
        self.extra_tokens = int(extra_tokens)
        self.sink_mode = str(sink_mode)
        self.dense_first_steps = max(0, int(dense_first_steps))
        self.dense_first_percent = max(0.0, float(dense_first_percent))
        self.dense_last_steps = max(0, int(dense_last_steps))
        self.dense_layers_key = ",".join(str(x) for x in sorted(dense_layers))
        self.min_tokens = max(0, int(min_tokens))
        self.chunked_qkv = bool(chunked_qkv)
        self.dense_head_chunks = max(1, int(dense_head_chunks))
        self.dense_backend = str(dense_backend)
        self.disable_fp16_accum = bool(disable_fp16_accum)
        self.sigma_start = float(sigma_start)
        self.sigma_end = float(sigma_end)
        self.verbose = bool(verbose)

        # Private underscore runtime fields (ignored by _model_fingerprint)
        self._dense_layers = dense_layers
        self._dense_attention = dense_attention
        self._installed = set()
        self._reset_runtime()

    def _reset_runtime(self):
        self._pooled = {}
        self._vsa_plans = {}
        self._vsa_rope = None
        self._logged = set()
        self._sparse_calls = 0
        self._dense_calls = 0
        self._chunked_calls = 0
        self._last_reason = None

    def reset(self):
        self._reset_runtime()

    def _log_once(self, key, msg: str):
        if self.verbose and key not in self._logged:
            self._logged.add(key)
            _LOG.info("%s: %s", TAG, msg)

    def fingerprint_tuple(self) -> tuple:
        return (
            self.preset, self.method, round(self.keep_ratio, 5),
            None if self.keep_ratio_end is None else round(self.keep_ratio_end, 5),
            round(self.tau, 4), None if self.tau_end is None else round(self.tau_end, 4),
            self.tail, self.extra_tokens, self.sink_mode,
            self.dense_first_steps, round(self.dense_first_percent, 4),
            self.dense_last_steps, self.dense_layers_key,
            self.min_tokens, self.chunked_qkv, self.dense_head_chunks,
            self.dense_backend, self.disable_fp16_accum,
        )

    def current_params(self, transformer_options: dict) -> tuple[float, float]:
        step0, n_steps = resolve_step(transformer_options)
        t = schedule_progress(step0, n_steps)
        if self.method == "sla" and self.keep_ratio_end is not None:
            topk = _lerp(self.keep_ratio, self.keep_ratio_end, t)
        else:
            topk = self.keep_ratio if self.method != "sol-attn" else 0.0
        if self.method == "sol-attn" and self.tau_end is not None:
            tau = _lerp(self.tau, self.tau_end, t)
        else:
            tau = self.tau
        return topk, tau

    def dense_reason(self, transformer_options: dict, tokens: int, block_index: int | None) -> str | None:
        if self.method == "dense":
            return "dense_low_vram mode"
        if transformer_options.get("_htc_sparse_force_dense"):
            return transformer_options.get("_htc_sparse_dense_reason") or "scheduled dense step"
        sigmas = transformer_options.get("sigmas")
        if sigmas is not None:
            sigma = _scalar(sigmas)
            if sigma is not None and (sigma > self.sigma_start or sigma < self.sigma_end):
                return f"sigma {sigma:.3g} outside start/end window"
        if tokens < self.min_tokens:
            return f"{tokens} tokens < min_tokens {self.min_tokens}"
        if self._dense_layers and block_index is not None and block_index in self._dense_layers:
            return f"block {block_index} in dense_layers"
        return None

    def step_dense_reason(self, transformer_options: dict) -> str | None:
        if self.method == "dense":
            return "dense_low_vram mode"
        step0, n_steps = resolve_step(transformer_options)
        if step0 is None:
            if self.dense_first_steps > 0 or self.dense_last_steps > 0 or self.dense_first_percent > 0:
                return None
            return None
        if self.dense_first_steps > 0 and step0 < self.dense_first_steps:
            return f"dense_first_steps ({step0}/{n_steps})"
        if self.dense_first_percent > 0.0 and n_steps > 0:
            n_first = min(n_steps, int(math.ceil(self.dense_first_percent * n_steps - 1e-9)))
            if step0 < n_first:
                return f"dense_first_percent ({step0}/{n_steps})"
        if self.dense_last_steps > 0 and n_steps > 0 and step0 >= n_steps - self.dense_last_steps:
            return f"dense_last_steps ({step0}/{n_steps})"
        return None

    def vsa_plan(self, layout, device):
        key = (tuple(layout.signature), tuple(layout.segments), str(device))
        plan = self._vsa_plans.get(key)
        if plan is not None:
            return plan
        _text_len, latent_t, latent_h, latent_w, _audio_t = layout.signature
        grid = (int(latent_t), int(latent_h) // 2, int(latent_w) // 2)
        tiles, n_prefix = [], 0
        for a, b, kind in layout.segments:
            n = b - a
            if kind != "video":
                m = (n + BLOCK_SIZE - 1) // BLOCK_SIZE
                seg = torch.full((m * BLOCK_SIZE,), -1, dtype=torch.int64, device=device)
                seg[:n] = torch.arange(a, b, device=device)
                tiles.append(seg.view(m, BLOCK_SIZE))
                n_prefix += m
                continue
            if grid[0] * grid[1] * grid[2] != n:
                raise RuntimeError(f"VSA: video segment {n} rows != grid {grid}")
            ct, ch, cw = VSA_CUBE
            pt, ph, pw = ((g + c - 1) // c * c for g, c in zip(grid, VSA_CUBE))
            padded = torch.full((pt, ph, pw), -1, dtype=torch.int64, device=device)
            padded[:grid[0], :grid[1], :grid[2]] = torch.arange(a, b, device=device).view(*grid)
            cubes = padded.view(pt // ct, ct, ph // ch, ch, pw // cw, cw).permute(0, 2, 4, 1, 3, 5).reshape(-1, BLOCK_SIZE)
            order = torch.argsort((cubes < 0).to(torch.int8), dim=1, stable=True)
            tiles.append(torch.gather(cubes, 1, order))
        tiles = torch.cat(tiles)
        src = tiles.reshape(-1)
        live = src >= 0
        inv = torch.empty(layout.seq_len, dtype=torch.int64, device=device)
        inv[src[live]] = torch.nonzero(live).flatten()
        plan = {"n": int(src.numel()), "n_prefix": n_prefix, "src": src, "inv": inv, "block_len": (tiles >= 0).sum(1).to(torch.int32)}
        while len(self._vsa_plans) >= VSA_PLAN_CACHE:
            del self._vsa_plans[next(iter(self._vsa_plans))]
        self._vsa_plans[key] = plan
        return plan

    def vsa_rope_freqs(self, rope_freqs, plan):
        hit = self._vsa_rope
        if hit is not None and hit[0]() is rope_freqs and hit[1] is plan:
            return hit[2]
        padded = rope_freqs.new_zeros((1, plan["n"]) + tuple(rope_freqs.shape[2:]))
        padded[0, plan["inv"]] = rope_freqs[0]
        self._vsa_rope = (weakref.ref(rope_freqs), plan, padded)
        return padded


def _scale_qkv_v_spans(qkv: torch.Tensor, chunk_start: int, v_spans, inner_dim: int) -> torch.Tensor:
    """Scale only the `V` third of packed `[M, 3 * inner]` QKV chunks inside `(a, b)` spans."""
    if not v_spans:
        return qkv
    chunk_end = chunk_start + int(qkv.shape[0])
    out = None
    for a, b, scale in v_spans:
        s = float(scale)
        if s >= 0.9999:
            continue
        lo = max(int(a), chunk_start)
        hi = min(int(b), chunk_end)
        if hi <= lo:
            continue
        if out is None:
            out = qkv.clone()
        out[lo - chunk_start:hi - chunk_start, 2 * inner_dim:].mul_(s)
    return out if out is not None else qkv


def _scale_v_tensor_spans(v: torch.Tensor, v_spans, *, skip_reshape: bool) -> torch.Tensor:
    """Scale `V` rows for dense / non-chunked `ck.sol_attn` without touching Q or K."""
    if not v_spans:
        return v
    seq_len = int(v.shape[2] if skip_reshape else v.shape[1])
    out = None
    for a, b, scale in v_spans:
        s = float(scale)
        if s >= 0.9999:
            continue
        lo = max(0, int(a))
        hi = min(seq_len, int(b))
        if hi <= lo:
            continue
        if out is None:
            out = v.clone()
        if skip_reshape:
            out[:, :, lo:hi, :].mul_(s)
        else:
            out[:, lo:hi, :].mul_(s)
    return out if out is not None else v


def _ineligible(q, k, v, dim_head: int) -> str | None:
    import comfy_kitchen as ck  # noqa: PLC0415
    if q.device.type != "cuda":
        return "not on CUDA"
    if not ck.sol_attn_is_available(q.device):
        return "no compiled sol_attn kernel for this GPU"
    if q.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        return f"dtype {q.dtype}"
    if dim_head != HEAD_DIM:
        return f"head_dim {dim_head} != {HEAD_DIM}"
    if q.shape != k.shape or q.shape != v.shape:
        return "cross-attention or GQA"
    if k.dtype != q.dtype or v.dtype != q.dtype:
        return "mixed dtypes"
    return None


def _run_dense_fallback(patch: SparseAttnPatch, prev_override, func, q, k, v, heads, kwargs, reason: str | None = None):
    from comfy.ldm.modules.attention import AttentionTensorContainer, optimized_attention  # noqa: PLC0415

    patch._dense_calls += 1
    if reason:
        patch._last_reason = reason
        patch._log_once(("dense", reason), f"dense ({reason})")

    owned = isinstance(q, AttentionTensorContainer)
    has_mask = kwargs.get("mask") is not None

    if prev_override is not None and (patch.dense_head_chunks <= 1 or has_mask):
        kw_call = dict(kwargs)
        kw_call["_inside_attn_wrapper"] = True
        if owned:
            if hasattr(prev_override, "container_function"):
                return prev_override.container_function(q, k, v, heads, **kw_call)
            qt, kt, vt = q.take(), k.take(), v.take()
            return prev_override(func or optimized_attention, qt, kt, vt, heads, **kw_call)
        return prev_override(func or optimized_attention, q, k, v, heads, **kw_call)

    def base_dense(qq, kk, vv, hh, **kw):
        kw_call = dict(kw)
        kw_call["_inside_attn_wrapper"] = True
        if prev_override is not None:
            return prev_override(func or optimized_attention, qq, kk, vv, hh, **kw_call)
        if patch._dense_attention is not None and kw_call.get("mask") is None:
            return patch._dense_attention(qq, kk, vv, hh, **kw_call)
        if func is not None:
            return func(qq, kk, vv, hh, **kw)
        return optimized_attention(qq, kk, vv, hh, **kw_call)

    if owned:
        q, k, v = q.take(), k.take(), v.take()
    if patch.dense_head_chunks > 1 and not has_mask:
        return head_chunked_attention(q, k, v, heads, patch.dense_head_chunks, base_dense, **kwargs)
    return base_dense(q, k, v, heads, **kwargs)


def make_attention_override(patch: SparseAttnPatch, previous):
    from comfy.ldm.modules.attention import AttentionTensorContainer  # noqa: PLC0415
    import comfy_kitchen as ck  # noqa: PLC0415

    def override(func, q, k, v, heads, mask=None, attn_precision=None,
                 skip_reshape=False, skip_output_reshape=False, **kwargs):
        transformer_options = kwargs.get("transformer_options") or {}
        all_kw = dict(mask=mask, attn_precision=attn_precision, skip_reshape=skip_reshape,
                      skip_output_reshape=skip_output_reshape, **kwargs)

        if mask is not None or patch.vsa or patch.method == "dense":
            return _run_dense_fallback(patch, previous, func, q, k, v, heads, all_kw,
                                       reason=("dense_low_vram" if patch.method == "dense" else "mask/vsa"))

        owned = isinstance(q, AttentionTensorContainer)
        inputs = [t.peek() for t in (q, k, v)] if owned else [q, k, v]
        tokens = inputs[0].shape[2] if skip_reshape else inputs[0].shape[1]
        reason = patch.dense_reason(transformer_options, tokens, transformer_options.get("block_index"))
        if reason is not None:
            del inputs
            return _run_dense_fallback(patch, previous, func, q, k, v, heads, all_kw, reason=reason)

        if skip_reshape:
            b, _, _, dim_head = inputs[0].shape
            qs, ks, vs = (t.transpose(1, 2) for t in inputs)
        else:
            b, _, dim_head = inputs[0].shape
            dim_head //= heads
            qs, ks, vs = (t.view(b, -1, heads, dim_head) for t in inputs)

        inelig = _ineligible(qs, ks, vs, dim_head)
        if inelig is not None:
            del inputs, qs, ks, vs
            return _run_dense_fallback(patch, previous, func, q, k, v, heads, all_kw, reason=inelig)

        sinks = compute_sinks(transformer_options, tokens, patch.sink_mode)
        if sinks is None:
            del inputs, qs, ks, vs
            return _run_dense_fallback(patch, previous, func, q, k, v, heads, all_kw, reason="no minimax_h3_layout")

        # Committed to sparse path: consume single-owner containers so packed buffers can free promptly
        if owned:
            for t in (q, k, v):
                t.take()
        del inputs

        sink, sink_q = sinks
        topk, tau = patch.current_params(transformer_options)
        v_spans = transformer_options.get("_htc_ref_video_v_spans")
        if v_spans and not transformer_options.get("_htc_v_spans_applied"):
            vs = _scale_v_tensor_spans(vs, v_spans, skip_reshape=False)
        in_dtype = qs.dtype
        if in_dtype == torch.float32:
            qs, ks, vs = (t.to(torch.bfloat16) for t in (qs, ks, vs))
        out = ck.sol_attn(
            qs.contiguous(), ks.contiguous(), vs.contiguous(),
            tau=tau, scale=kwargs.get("scale"),
            sink_blocks=list(sink), sink_q=list(sink_q),
            topk_ratio=topk, tail=patch.tail, token_aug=patch.extra_tokens,
        ).to(in_dtype)
        patch._sparse_calls += 1
        patch._log_once(("sparse", tokens, round(topk, 3), round(tau, 2)),
                        f"sparse ({b}, {tokens}, {heads}, {dim_head}) method={patch.method} topk={topk:.2f} tau={tau:.2f} tail={patch.tail} sinks={sink}/{sink_q}")
        if skip_output_reshape:
            return out.transpose(1, 2)
        return out.reshape(b, -1, heads * dim_head)

    override.container_function = partial(override, None)
    return override


def install_override(patch: SparseAttnPatch, transformer_options: dict):
    current = transformer_options.get("optimized_attention_override")
    if current in patch._installed:
        return
    override = make_attention_override(patch, current)
    patch._installed.add(override)
    transformer_options["optimized_attention_override"] = override


def _h3_chunked_eligible(attn, x, rope_freqs, transformer_options: dict, patch: SparseAttnPatch, block_index: int | None) -> bool:
    if not patch.chunked_qkv or patch.method == "dense":
        return False
    import comfy_kitchen as ck  # noqa: PLC0415
    n_tokens = x.shape[0]
    if rope_freqs is None or x.dtype != torch.bfloat16 or x.device.type != "cuda" or attn.head_dim != HEAD_DIM:
        return False
    reason = patch.dense_reason(transformer_options, n_tokens, block_index)
    if reason is None and not ck.sol_attn_is_available(x.device):
        reason = "no compiled sol_attn kernel"
    if reason is not None:
        patch._log_once(("chunk_dense", block_index, reason), f"block {block_index} dense ({reason})")
        return False
    sinks = compute_sinks(transformer_options, n_tokens, patch.sink_mode)
    if sinks is None and not patch.vsa:
        return False
    if patch.vsa:
        lay = transformer_options.get("minimax_h3_layout")
        if lay is None or lay.seq_len != n_tokens:
            return False
    return True


def _h3_chunked_sparse_attention(attn, x, rope_freqs, transformer_options: dict, patch: SparseAttnPatch, block_index: int | None):
    import comfy.model_management as mm  # noqa: PLC0415
    import comfy.model_prefetch as mp  # noqa: PLC0415
    import comfy_kitchen as ck  # noqa: PLC0415

    n_tokens = x.shape[0]
    heads, head_dim = attn.heads, attn.head_dim
    qw = mm.cast_to(attn.q_norm.weight, device=x.device)
    kw = mm.cast_to(attn.k_norm.weight, device=x.device)
    extra, plan, gate = {}, None, None
    n, freqs = n_tokens, rope_freqs

    with mp.pause_malloc_graph():
        if patch.vsa:
            plan = patch.vsa_plan(transformer_options["minimax_h3_layout"], x.device)
            n = plan["n"]
            freqs = patch.vsa_rope_freqs(rope_freqs, plan)

        cw = transformer_options.get("context_window")
        cw_key = (
            (int(cw.index_list[0]), len(cw.index_list))
            if cw is not None and getattr(cw, "index_list", None)
            else ()
        )
        key = (block_index, n, cw_key, tuple(transformer_options.get("uuids", ())))
        pooled = patch._pooled.get(key)
        first = pooled is None
        if first:
            pooled = (
                torch.empty((heads, head_dim), dtype=torch.float32, device=x.device),
                torch.empty((heads, head_dim), dtype=torch.float32, device=x.device),
            )

    if patch.vsa:
        sink = sink_q = (0, plan["n_prefix"])
        extra = {"tail": False, "block_len": plan["block_len"]}
        gate = getattr(attn, "to_gate_compress", None)
        if gate is not None:
            extra["coarse_gate"] = x.new_empty(n, heads * head_dim).view(1, n, heads, head_dim)
        topk, tau = patch.keep_ratio, 0.0
    else:
        sink, sink_q = compute_sinks(transformer_options, n_tokens, patch.sink_mode) or ((0, 0), (0, 0))
        topk, tau = patch.current_params(transformer_options)
        extra = {"tail": patch.tail}

    v_spans = transformer_options.get("_htc_ref_video_v_spans")
    inner_dim = heads * head_dim

    def chunks():
        for i in range(0, n, PRODUCER_CHUNK):
            if plan is None:
                qkv = attn.qkv_proj(x[i:i + PRODUCER_CHUNK])
                if v_spans:
                    qkv = _scale_qkv_v_spans(qkv, i, v_spans, inner_dim)
                yield qkv
                continue
            idx = plan["src"][i:i + PRODUCER_CHUNK]
            xc = x[idx.clamp_min(0)] * (idx >= 0).unsqueeze(1).to(x.dtype)
            if gate is not None:
                extra["coarse_gate"].view(n, heads * head_dim)[i:i + xc.shape[0]] = gate(xc)
            qkv = attn.qkv_proj(xc)
            if v_spans:
                # Map VSA reordered positions back to original token spans
                pos_slice = idx
                need = False
                for a, b, s in v_spans:
                    if float(s) < 0.9999 and bool(((pos_slice >= int(a)) & (pos_slice < int(b))).any()):
                        need = True
                        break
                if need:
                    qkv = qkv.clone()
                    for a, b, s in v_spans:
                        fs = float(s)
                        if fs >= 0.9999:
                            continue
                        mask_p = (pos_slice >= int(a)) & (pos_slice < int(b))
                        if bool(mask_p.any()):
                            qkv[mask_p, 2 * inner_dim:].mul_(fs)
            yield qkv

    out, kmean, vscale = ck.sol_attn_chunked(
        chunks, n, heads, freqs, (qw, kw),
        kmean=None if first else pooled[0],
        vscale=None if first else pooled[1],
        tau=tau, topk_ratio=topk,
        token_aug=(0 if patch.vsa else patch.extra_tokens),
        sink_blocks=list(sink), sink_q=list(sink_q),
        rope_eps=attn.q_norm.eps, **extra,
    )
    pooled[0].copy_(kmean)
    pooled[1].copy_(vscale)
    patch._pooled[key] = pooled
    patch._chunked_calls += 1
    patch._sparse_calls += 1
    patch._log_once(
        ("chunked", n_tokens, round(topk, 3), round(tau, 2)),
        f"chunked sparse ({n_tokens} tokens, method={patch.method}, topk={topk:.2f}, tau={tau:.2f}, tail={patch.tail}, sinks={sink}/{sink_q})",
    )
    out = out.view(n, heads * head_dim)
    if plan is not None:
        out = out[plan["inv"]]
    return attn.out_proj(out)


def _make_block_patch(block, block_index: int, patch: SparseAttnPatch, prev_patch):
    def attention(h, rope_freqs=None, transformer_options={}):
        return _h3_chunked_sparse_attention(block.attn, h, rope_freqs, transformer_options, patch, block_index)

    def block_patch(args, extra):
        to = args.get("transformer_options") or {}
        if _h3_chunked_eligible(block.attn, args["img"], args["rope_freqs"], to, patch, block_index):
            args = dict(args)
            args["attention"] = attention
        if prev_patch is not None:
            return prev_patch(args, extra)
        return extra["original_block"](args)

    return block_patch


def _make_step_wrapper(patch: SparseAttnPatch):
    def wrapper(executor, x, timestep, context, transformer_options=None, *args, **kwargs):
        to = dict(transformer_options or {})
        install_override(patch, to)
        reason = patch.step_dense_reason(to)
        to["_htc_sparse_force_dense"] = (reason is not None)
        to["_htc_sparse_dense_reason"] = reason
        with reduction_policy(patch.disable_fp16_accum):
            return executor(x, timestep, context, to, *args, **kwargs)
    return wrapper


def block_forward_takes_attention(model) -> bool:
    """False when an object patch on an H3 block forward (ComfyUI-KJNodes before its
    2026-09-06 fix) lacks core's `attention` keyword, so any `double_block` replace
    would raise in the first block. Same test as tiled_upsampler's guard."""
    import inspect  # noqa: PLC0415
    try:
        from comfy.ldm.minimax.model import DiTBlock  # noqa: PLC0415
        if "attention" not in inspect.signature(DiTBlock.forward).parameters:
            return True
    except Exception:  # noqa: BLE001
        return True
    for name, obj in (getattr(model, "object_patches", None) or {}).items():
        parts = str(name).split(".")
        if (len(parts) < 4 or parts[-1] != "forward" or parts[-3] != "blocks"
                or not parts[-2].isdigit()):
            continue
        try:
            if "attention" not in inspect.signature(obj).parameters:
                return False
        except (TypeError, ValueError):
            continue
    return True


def _resolve_preset(preset: str, kwargs: dict) -> dict:
    out = dict(kwargs)
    spec = PRESET_CONFIGS.get(str(preset or "manual"))
    if spec:
        out.update(spec)
    return out


def apply_sparse_attention(
    model,
    *,
    preset: str = "turbo_sla_balanced",
    method: str = "sla",
    keep_percent: float = 20.0,
    keep_percent_end: float = -1.0,
    tau: float = 1.35,
    tau_end: float = -1.0,
    tail: bool = True,
    extra_tokens: int = 0,
    sink_mode: str = "exact_prefix_rows",
    dense_first_steps: int = 1,
    dense_first_percent: float = 0.0,
    dense_last_steps: int = 1,
    dense_layers: str = "0",
    min_tokens: int = 12288,
    chunked_qkv: bool = True,
    dense_head_chunks: int = 1,
    dense_backend: str = "comfy_kitchen_int8",
    disable_fp16_accum: bool = True,
    start_percent: float = 0.0,
    end_percent: float = 1.0,
    verbose: bool = False,
):
    import comfy.patcher_extension as pe  # noqa: PLC0415
    from comfy.ldm.minimax.model import MiniMaxH3Model  # noqa: PLC0415
    from comfy.ldm.modules.attention import get_attention_function  # noqa: PLC0415
    import comfy_kitchen as ck  # noqa: PLC0415

    cfg = _resolve_preset(preset, dict(
        method=method, keep_percent=keep_percent, keep_percent_end=keep_percent_end,
        tau=tau, tau_end=tau_end, tail=tail, extra_tokens=extra_tokens,
        sink_mode=sink_mode, dense_first_steps=dense_first_steps,
        dense_first_percent=dense_first_percent, dense_last_steps=dense_last_steps,
        dense_layers=dense_layers, min_tokens=min_tokens, chunked_qkv=chunked_qkv,
        dense_head_chunks=dense_head_chunks, dense_backend=dense_backend,
        disable_fp16_accum=disable_fp16_accum,
    ))

    eff_method = str(cfg["method"])
    if eff_method != "dense" and not ck.sol_attn_is_available():
        raise RuntimeError(f"{TAG}: Comfy Kitchen sol_attn kernel is unavailable on this device.")

    dm = model.get_model_object("diffusion_model")
    is_h3 = isinstance(dm, MiniMaxH3Model)
    if eff_method == "vsa" and not is_h3:
        raise ValueError(f"{TAG}: VSA method requires a MiniMax-H3 model.")
    if is_h3 and not block_forward_takes_attention(model):
        # The chunked/VSA paths are `double_block` patches; core calls a replaced block
        # with `attention=`, which an old KJNodes block forward cannot take.
        if eff_method == "vsa":
            raise RuntimeError(f"{TAG}: VSA needs a newer ComfyUI-KJNodes (its patched H3 block "
                               "forward has no `attention` argument). Update KJNodes or pick an SLA/Sol preset.")
        if cfg["chunked_qkv"]:
            print(f"[{TAG}] chunked_qkv off: ComfyUI-KJNodes is too old (its patched H3 block "
                  "forward has no `attention` argument); per-call sparse attention only. "
                  "Update KJNodes for the chunked path.", flush=True)
            cfg["chunked_qkv"] = False

    blocks = getattr(dm, "blocks", None)
    n_layers = len(blocks) if blocks is not None else None
    parsed_layers = parse_block_list(cfg["dense_layers"], n_layers)

    sigma_start, sigma_end = float("inf"), 0.0
    try:
        ms = model.get_model_object("model_sampling")
        if ms is not None and hasattr(ms, "percent_to_sigma"):
            sigma_start = float(ms.percent_to_sigma(float(start_percent)))
            sigma_end = float(ms.percent_to_sigma(float(end_percent)))
    except Exception:
        pass

    dense_attn = None if cfg["dense_backend"] == "auto" else get_attention_function(cfg["dense_backend"], None)
    patch = SparseAttnPatch(
        preset=preset,
        method=eff_method,
        keep_percent=cfg["keep_percent"],
        keep_percent_end=cfg["keep_percent_end"],
        tau=cfg["tau"],
        tau_end=cfg["tau_end"],
        tail=cfg["tail"],
        extra_tokens=(0 if eff_method == "vsa" else cfg["extra_tokens"]),
        sink_mode=cfg["sink_mode"],
        dense_first_steps=cfg["dense_first_steps"],
        dense_first_percent=cfg["dense_first_percent"],
        dense_last_steps=cfg["dense_last_steps"],
        dense_layers=parsed_layers,
        min_tokens=cfg["min_tokens"],
        chunked_qkv=(bool(cfg["chunked_qkv"]) and is_h3),
        dense_head_chunks=cfg["dense_head_chunks"],
        dense_backend=cfg["dense_backend"],
        disable_fp16_accum=cfg["disable_fp16_accum"],
        sigma_start=sigma_start,
        sigma_end=sigma_end,
        verbose=verbose,
        dense_attention=dense_attn,
    )

    m = model.clone()
    to = m.model_options.setdefault("transformer_options", {})
    to["htc_sparse_attn"] = patch.fingerprint_tuple()
    install_override(patch, to)

    m.remove_wrappers_with_key(pe.WrappersMP.DIFFUSION_MODEL, "htc_sparse_attn")
    m.add_wrapper_with_key(pe.WrappersMP.DIFFUSION_MODEL, "htc_sparse_attn", _make_step_wrapper(patch))
    m.add_callback_with_key(
        pe.CallbacksMP.ON_PREPARE_STATE,
        "htc_sparse_attn",
        lambda mp, ts, mo: install_override(patch, mo.setdefault("transformer_options", {})),
    )
    m.add_callback_with_key(
        pe.CallbacksMP.ON_CLEANUP,
        "htc_sparse_attn",
        lambda mp: _cleanup_log(patch),
    )

    if is_h3 and blocks is not None and (patch.chunked_qkv or eff_method == "vsa"):
        dit_rep = dict((to.get("patches_replace") or {}).get("dit") or {})
        for i, blk in enumerate(blocks):
            prev = dit_rep.get(("double_block", i))
            # Strip any prior htc_sparse_attn wrapper on this block if re-patching
            if getattr(prev, "_htc_sparse_inner", None) is not None:
                prev = prev._htc_sparse_inner
            bp = _make_block_patch(blk, i, patch, prev)
            bp._htc_sparse_inner = prev  # type: ignore[attr-defined]
            m.set_model_patch_replace(bp, "dit", "double_block", i)

    return m


def _cleanup_log(patch: SparseAttnPatch):
    if patch.verbose or patch._sparse_calls or patch._dense_calls:
        _LOG.info(
            "%s: preset=%s method=%s sparse=%d (chunked=%d) dense=%d last_dense=%s",
            TAG, patch.preset, patch.method, patch._sparse_calls,
            patch._chunked_calls, patch._dense_calls, patch._last_reason,
        )
    patch.reset()


def apply_low_vram_attention(model, head_chunks: int = 4):
    """Pure-MIT head-chunked dense attention for low-VRAM upsample passes (zero external dependency)."""
    chunks = max(1, int(head_chunks))
    if chunks <= 1 or model is None:
        return model
    return apply_sparse_attention(
        model,
        preset="dense_low_vram",
        method="dense",
        dense_head_chunks=chunks,
        dense_backend="auto",
        chunked_qkv=False,
        disable_fp16_accum=False,
    )


def apply_chunk_ffn(model, chunks: int = 2, seq_threshold: int = 4096):
    """Pure-MIT token-chunked SwiGLU FFN on MiniMax-H3 blocks (zero external dependency)."""
    chunks = max(1, int(chunks))
    seq_threshold = max(0, int(seq_threshold))
    if chunks <= 1 or model is None or not block_forward_takes_attention(model):
        return model
    dm = model.get_model_object("diffusion_model") if hasattr(model, "get_model_object") else None
    blocks = getattr(dm, "blocks", None)
    if not blocks:
        return model
    import comfy.ops  # noqa: PLC0415

    m = model.clone()
    to = m.model_options.setdefault("transformer_options", {})
    to["htc_chunk_ffn"] = (chunks, seq_threshold)
    dit_rep = dict((to.get("patches_replace") or {}).get("dit") or {})

    for i, blk in enumerate(blocks):
        prev = dit_rep.get(("double_block", i))
        if getattr(prev, "_htc_ffn_inner", None) is not None:
            prev = prev._htc_ffn_inner

        def _make_bp(block, prev_fn):
            def bp(args, extra):
                mlp = block.mlp
                orig_fwd = mlp.forward
                had = "forward" in mlp.__dict__
                saved = mlp.__dict__.get("forward")

                def chunked_fwd(x):
                    if x.shape[0] <= seq_threshold or chunks <= 1:
                        return orig_fwd(x)
                    out = None
                    off = 0
                    for c in torch.chunk(x, chunks, dim=0):
                        part = comfy.ops.linear_input_act(mlp.fc2, mlp.fc1(c), "swiglu")
                        if out is None:
                            out = part.new_empty(x.shape[0], part.shape[1])
                        out[off:off + part.shape[0]] = part
                        off += part.shape[0]
                    return out

                mlp.__dict__["forward"] = chunked_fwd
                try:
                    if prev_fn is not None:
                        return prev_fn(args, extra)
                    return extra["original_block"](args)
                finally:
                    if had:
                        mlp.__dict__["forward"] = saved
                    else:
                        mlp.__dict__.pop("forward", None)

            bp._htc_ffn_inner = prev_fn  # type: ignore[attr-defined]
            return bp

        m.set_model_patch_replace(_make_bp(blk, prev), "dit", "double_block", i)
    return m


class HTCSparseAttention:
    """Built-in MIT MiniMax-H3 Sparse / Low-VRAM Attention (Comfy Kitchen SLA / Sol-Attn / VSA)."""

    CATEGORY = "Hand Tie Clips"
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply"
    DESCRIPTION = (
        "Clean-room MIT MiniMax-H3 sparse attention on Comfy Kitchen (`ck.sol_attn_chunked` + `sol_attn`), "
        "combining streaming 4096-row QKV projection, pooled-tail compensation, exact prefix protection, "
        "step annealing, and head-chunked low-VRAM dense fallback. Chains safely with HTCSteeringDials, "
        "HTCTokenLoRA, and HandTieClips's built-in cache with zero external dependencies."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {
                    "tooltip": "MiniMax-H3 model to accelerate with SLA+ / Sol-Attn sparse attention.",
                }),
                "preset": (list(PRESETS), {
                    "default": PRESETS[0],
                    "tooltip": (
                        "Quick attention preset (overrides manual settings below except manual): "
                        "turbo_sla_balanced = best 4-8 step Turbo speed/quality balance (20% keep + step 0/last dense); "
                        "turbo_sla_quality = higher 30% keep + first 25% steps dense; "
                        "turbo_sla_fast = lighter 14%->20% keep for max speed; "
                        "native_sol_balanced / quality = training-free sparse attention for base (non-Turbo) H3 checkpoints; "
                        "vsa_10 = FastH3 3D block sparse (10% keep); "
                        "dense_low_vram = exact dense attention split into 4 head chunks; "
                        "manual = use all individual sliders below."
                    ),
                }),
            },
            "optional": {
                "method": (list(METHODS), {
                    "default": "sla",
                    "tooltip": (
                        "Manual only. Sparse attention algorithm: "
                        "sla = token-level importance keep + pooled tail compensation; "
                        "vsa = 3D spatial-temporal cube sparse (VSA); "
                        "dense = exact dense attention."
                    ),
                }),
                "keep_percent": ("FLOAT", {
                    "default": 20.0, "min": 0.5, "max": 95.0, "step": 0.5,
                    "tooltip": "Manual only. Percentage of key/value blocks each query attends to (higher = closer to dense, slower).",
                }),
                "keep_percent_end": ("FLOAT", {
                    "default": -1.0, "min": -1.0, "max": 95.0, "step": 0.5,
                    "tooltip": "Manual only. End keep_percent at the final step when annealing across sampling (-1 = fixed at keep_percent).",
                }),
                "tau": ("FLOAT", {
                    "default": 1.35, "min": 0.0, "max": 4.0, "step": 0.05,
                    "tooltip": "Manual only. Temperature/sharpness of sparse block importance scores (0 = uniform, ~1.35 = balanced).",
                }),
                "tau_end": ("FLOAT", {
                    "default": -1.0, "min": -1.0, "max": 4.0, "step": 0.05,
                    "tooltip": "Manual only. End tau at the final step when annealing (-1 = fixed at tau).",
                }),
                "tail": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Manual only. Add pooled summary of unselected tail tokens so distant context is never silently dropped.",
                }),
                "extra_tokens": ("INT", {
                    "default": 0, "min": 0, "max": 256, "step": 64,
                    "tooltip": "Manual only. Extra learned/augmented sink tokens per block (0 = training-free, recommended).",
                }),
                "sink_mode": (list(SINK_MODES), {
                    "default": SINK_MODES[0],
                    "tooltip": (
                        "Which conditioning prefix rows always receive exact dense attention: "
                        "exact_prefix_rows (protect text/ref/audio sinks), exact_kv, exact_kv_and_rows, or off."
                    ),
                }),
                "dense_first_steps": ("INT", {
                    "default": 1, "min": 0, "max": 32, "step": 1,
                    "tooltip": "Manual only. Run this many opening denoising steps in exact dense attention to lock early layout.",
                }),
                "dense_first_percent": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Manual only. Run this fraction of early steps dense (0 = rely on dense_first_steps).",
                }),
                "dense_last_steps": ("INT", {
                    "default": 1, "min": 0, "max": 16, "step": 1,
                    "tooltip": "Manual only. Run this many final steps dense to keep fine skin, lips, and edge detail sharp.",
                }),
                "dense_layers": ("STRING", {
                    "default": "0", "multiline": False,
                    "tooltip": "Manual only. Comma/range list of block indices forced to dense attention (e.g. '0' or '0-2,49').",
                }),
                "min_tokens": ("INT", {
                    "default": 12288, "min": 0, "max": 1 << 20, "step": 512,
                    "tooltip": "Manual only. Fall back cleanly to dense below this token count (short clips get zero speedup overhead).",
                }),
                "chunked_qkv": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Manual only. Stream QKV projection in 4096-row chunks directly into sol_attn_chunked (saves substantial VRAM).",
                }),
                "dense_head_chunks": ("INT", {
                    "default": 1, "min": 1, "max": 8, "step": 1,
                    "tooltip": "Manual only. Split dense fallback attention across this many head chunks (4 cuts peak VRAM on small GPUs).",
                }),
                "dense_backend": (list(DENSE_BACKENDS), {
                    "default": DENSE_BACKENDS[0],
                    "tooltip": "Manual only. Backend used on dense steps (comfy_kitchen_int8, comfy_kitchen, or torch_sdpa).",
                }),
                "disable_fp16_accum": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Disable reduced-precision fp16/bf16 matmul accumulation during H3 steps for cleaner color & seam accuracy.",
                }),
                "start_percent": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Sampling progress [0..1] where sparse attention becomes active (steps before this stay dense).",
                }),
                "end_percent": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Sampling progress [0..1] where sparse attention stops (steps after this return to dense).",
                }),
                "verbose": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Log per-step sparse vs dense attention calls to the ComfyUI console.",
                }),
            },
        }

    def apply(
        self,
        model,
        preset="turbo_sla_balanced",
        method="sla",
        keep_percent=20.0,
        keep_percent_end=-1.0,
        tau=1.35,
        tau_end=-1.0,
        tail=True,
        extra_tokens=0,
        sink_mode="exact_prefix_rows",
        dense_first_steps=1,
        dense_first_percent=0.0,
        dense_last_steps=1,
        dense_layers="0",
        min_tokens=12288,
        chunked_qkv=True,
        dense_head_chunks=1,
        dense_backend="comfy_kitchen_int8",
        disable_fp16_accum=True,
        start_percent=0.0,
        end_percent=1.0,
        verbose=False,
    ):
        patched = apply_sparse_attention(
            model,
            preset=preset,
            method=method,
            keep_percent=keep_percent,
            keep_percent_end=keep_percent_end,
            tau=tau,
            tau_end=tau_end,
            tail=tail,
            extra_tokens=extra_tokens,
            sink_mode=sink_mode,
            dense_first_steps=dense_first_steps,
            dense_first_percent=dense_first_percent,
            dense_last_steps=dense_last_steps,
            dense_layers=dense_layers,
            min_tokens=min_tokens,
            chunked_qkv=chunked_qkv,
            dense_head_chunks=dense_head_chunks,
            dense_backend=dense_backend,
            disable_fp16_accum=disable_fp16_accum,
            start_percent=start_percent,
            end_percent=end_percent,
            verbose=verbose,
        )
        return (patched,)


NODE_CLASS_MAPPINGS = {"HTCSparseAttention": HTCSparseAttention}
NODE_DISPLAY_NAME_MAPPINGS = {"HTCSparseAttention": "HTC Sparse Attention / SLA+ (H3)"}
