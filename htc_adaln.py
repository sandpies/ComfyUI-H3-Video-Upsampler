# Generated from ComfyUI-Hand-Tie-Clips (https://github.com/sandpies/ComfyUI-Hand-Tie-Clips) by tools/build_upsampler_pack.py.
# Edit it there, not here: the next build overwrites this file.
"""Built-in MIT AdaLN LoRA porting for MiniMax-H3 (`dense <-> curve-form`).

Adapted from PlagueKind's MIT `ComfyUI-H3-AdaLN-LoRA-Fix` (`adaln.py` /
`adaln_patch.py`, Copyright (c) 2026 PlagueKind). Baked directly into
`HTCTokenLoRA`, `HandTieClips` (`_boost_model`), `tiled_upsampler.py`, and
`video_upsampler.py` (plus exposed as `HTCAdaLNFix` for non-TokenLoRA graphs)
so dense and pruned H3 LoRAs apply cleanly with zero external dependency.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import struct
import uuid

import torch
import torch.nn.functional as F
from comfy_api.latest import io

TAG = "HTCAdaLN"
log = logging.getLogger(TAG)

MODES = ("port", "strip", "off")
GRID_ROWS = 1025
FREQ_DIM = 256
BASIS_SUBDIR = "h3_adaln"
GRID_FILENAME = "h3_silu_temb_grid.safetensors"
GRID_TENSOR_KEY = "silu_t_emb_grid"
MAX_RESIDUAL = 5e-3

ADALN_WEIGHT_SUFFIX = "adaln_proj.linear.weight"
TE_IN_W = "time_embedder.proj_in.weight"
TE_IN_B = "time_embedder.proj_in.bias"
TE_OUT_W = "time_embedder.proj_out.weight"
TE_OUT_B = "time_embedder.proj_out.bias"


def silu_temb_grid(proj_in_w, proj_in_b, proj_out_w, proj_out_b, rows=GRID_ROWS, freq_dim=FREQ_DIM):
    """Dense `silu(t_emb(t))` sampled on the 1025-row `[0, 1]` curve grid."""
    t = torch.arange(rows, dtype=torch.float32) / float(rows - 1)
    half = freq_dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float32) / half)
    args = t[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    hidden = F.silu(emb @ proj_in_w.float().T + proj_in_b.float())
    temb = hidden @ proj_out_w.float().T + proj_out_b.float()
    return F.silu(temb)


def solve_basis(table, grid):
    """Least-squares fit `grid[i] ~= c + V @ table[i]` -> `(c [D], V [D, k], rel_residual)`."""
    table = table.detach().float().cpu()
    grid = grid.detach().float().cpu()
    if table.shape[0] != grid.shape[0]:
        raise ValueError(f"table has {table.shape[0]} rows, grid has {grid.shape[0]}")
    ones = torch.ones(table.shape[0], 1, dtype=torch.float32)
    x = torch.cat([ones, table], dim=1)
    sol = torch.linalg.lstsq(x, grid).solution
    resid = x @ sol - grid
    denom = grid.norm()
    rel = float(resid.norm() / denom) if float(denom) > 0 else float("inf")
    return sol[0].contiguous(), sol[1:].T.contiguous(), rel


def port_delta(lora_b, lora_a, scale, c, basis_v):
    """Dense LoRA `(B [out, r], A [r, D])` -> pruned `(dW [out, k], db [out])`."""
    a = lora_a.detach().float()
    b = lora_b.detach().float()
    d_w = (b @ (a @ basis_v)) * scale
    d_b = (b @ (a @ c)) * scale
    return d_w, d_b


def pinv_basis(basis_v):
    return torch.linalg.pinv(basis_v.detach().float())


def port_factors_reverse(lora_a, scale, c, pinv_v):
    """Curve LoRA `A [r, k]` -> `(A_dense [r, D], A_c [r])`."""
    a_dense = (lora_a.detach().float() @ pinv_v) * scale
    return a_dense, a_dense @ c


def port_delta_reverse(lora_b, lora_a, scale, c, basis_v):
    a_dense, a_c = port_factors_reverse(lora_a, scale, c, pinv_basis(basis_v))
    b = lora_b.detach().float()
    return b @ a_dense, -(b @ a_c)


def grid_from_time_embedder(time_embedder, rows=GRID_ROWS):
    try:
        proj_in, proj_out = time_embedder.proj_in, time_embedder.proj_out
        tensors = (proj_in.weight, proj_in.bias, proj_out.weight, proj_out.bias)
        if any(t is None or t.device.type == "meta" for t in tensors):
            return None
        freq_dim = int(getattr(time_embedder, "freq_dim", FREQ_DIM))
        with torch.no_grad():
            return silu_temb_grid(*tensors, rows=rows, freq_dim=freq_dim).clone()
    except Exception:
        return None


def tensor_hash(tensor):
    t = tensor.detach().float().cpu().contiguous()
    h = hashlib.sha256()
    h.update(str(tuple(t.shape)).encode())
    h.update(t.numpy().tobytes())
    return h.hexdigest()[:16]


table_hash = tensor_hash


def _safetensors_header(path):
    with open(path, "rb") as fh:
        raw = fh.read(8)
        if len(raw) < 8:
            return None
        size = struct.unpack("<Q", raw)[0]
        if size <= 0 or size > (64 << 20):
            return None
        return json.loads(fh.read(size))


def _time_embedder_prefix(header):
    if not header or any(k.endswith("adaln_t_table") for k in header):
        return None
    for key in header:
        if key.endswith(TE_OUT_W):
            prefix = key[:-len(TE_OUT_W)]
            if all(prefix + n in header for n in (TE_IN_W, TE_IN_B, TE_OUT_B)):
                return prefix
    return None


def _detach_from_file(tensor):
    return tensor.to(torch.float32).clone()


def _load_time_embedder(path, prefix):
    from safetensors import safe_open
    with safe_open(path, framework="pt") as f:
        return [_detach_from_file(f.get_tensor(prefix + n)) for n in (TE_IN_W, TE_IN_B, TE_OUT_W, TE_OUT_B)]


def _grid_from_checkpoint(path, rows):
    try:
        prefix = _time_embedder_prefix(_safetensors_header(path))
        if prefix is None:
            return None
        return silu_temb_grid(*_load_time_embedder(path, prefix), rows=rows)
    except Exception:
        return None


def _load_prebaked_grid(path):
    try:
        from safetensors import safe_open
        with safe_open(path, framework="pt") as f:
            keys = list(f.keys())
            key = GRID_TENSOR_KEY if GRID_TENSOR_KEY in keys else (keys[0] if len(keys) == 1 else None)
            if key is None:
                return None
            return _detach_from_file(f.get_tensor(key))
    except Exception:
        return None


def _basis_dir():
    import folder_paths
    return os.path.join(folder_paths.models_dir, BASIS_SUBDIR)


def _candidate_grids(rows):
    try:
        import folder_paths
    except Exception:
        return
    for name in folder_paths.get_filename_list("diffusion_models"):
        if not name.lower().endswith(".safetensors"):
            continue
        path = folder_paths.get_full_path("diffusion_models", name)
        if not path:
            continue
        grid = _grid_from_checkpoint(path, rows)
        if grid is not None:
            yield os.path.basename(name), grid

    seen = set()
    search = [_basis_dir()]
    search += list(folder_paths.get_folder_paths("loras"))
    search += list(folder_paths.get_folder_paths("diffusion_models"))
    packs = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if os.path.isdir(packs):
        search += [os.path.join(packs, d) for d in os.listdir(packs) if os.path.isdir(os.path.join(packs, d))]
    for directory in search:
        path = os.path.join(directory, GRID_FILENAME)
        if path in seen or not os.path.isfile(path):
            continue
        seen.add(path)
        grid = _load_prebaked_grid(path)
        if grid is not None:
            yield f"{GRID_FILENAME} ({os.path.basename(directory)})", grid


def _cache_path(key):
    return os.path.join(_basis_dir(), f"basis_{key}.safetensors")


def _read_cache(key):
    path = _cache_path(key)
    if not os.path.isfile(path):
        return None
    try:
        from safetensors import safe_open
        with safe_open(path, framework="pt") as f:
            meta = f.metadata() or {}
            return (
                _detach_from_file(f.get_tensor("c")),
                _detach_from_file(f.get_tensor("V")),
                float(meta.get("residual", "nan")),
                meta.get("source", "unknown") + " (cached)",
            )
    except Exception:
        return None


def _write_cache(key, c, basis_v, residual, source):
    try:
        from safetensors.torch import save_file
        os.makedirs(_basis_dir(), exist_ok=True)
        save_file(
            {"c": c.contiguous(), "V": basis_v.contiguous()},
            _cache_path(key),
            metadata={"residual": repr(residual), "source": source, "table_hash": key},
        )
    except Exception:
        pass


def candidate_tables():
    try:
        import folder_paths
        from safetensors import safe_open
    except Exception:
        return
    for name in folder_paths.get_filename_list("diffusion_models"):
        if not name.lower().endswith(".safetensors"):
            continue
        path = folder_paths.get_full_path("diffusion_models", name)
        if not path:
            continue
        try:
            header = _safetensors_header(path)
            key = next((k for k in (header or ()) if k.endswith("adaln_t_table")), None)
            if key is None:
                continue
            with safe_open(path, framework="pt") as f:
                table = _detach_from_file(f.get_tensor(key))
        except Exception:
            continue
        yield os.path.basename(name), table


def get_basis_for(table, grid, cache_key, use_cache=True, source="explicit"):
    if use_cache:
        cached = _read_cache(cache_key)
        if cached is not None:
            return cached
    c, basis_v, rel = solve_basis(table, grid)
    if use_cache:
        _write_cache(cache_key, c, basis_v, rel, source)
    return c, basis_v, rel, source


def get_basis(table, use_cache=True):
    key = tensor_hash(table)
    if use_cache:
        cached = _read_cache(key)
        if cached is not None:
            return cached
    rows = int(table.shape[0])
    best = None
    for label, grid in _candidate_grids(rows):
        if int(grid.shape[0]) != rows:
            continue
        c, basis_v, rel = solve_basis(table, grid)
        if best is None or rel < best[2]:
            best = (c, basis_v, rel, label)
    if best is None:
        return None
    if use_cache and best[2] <= MAX_RESIDUAL:
        _write_cache(key, *best)
    return best


def get_basis_dense(grid, use_cache=True):
    key = "rev_" + tensor_hash(grid)
    if use_cache:
        cached = _read_cache(key)
        if cached is not None:
            return cached
    rows = int(grid.shape[0])
    best = None
    for label, table in candidate_tables():
        if int(table.shape[0]) != rows:
            continue
        c, basis_v, rel = solve_basis(table, grid)
        if best is None or rel < best[2]:
            best = (c, basis_v, rel, label)
    if best is None:
        return None
    if use_cache and best[2] <= MAX_RESIDUAL:
        _write_cache(key, *best)
    return best


def _lora_weights(value):
    weights = getattr(value, "weights", None)
    if weights is None or len(weights) < 3:
        return None
    up, down = weights[0], weights[1]
    if not (torch.is_tensor(up) and torch.is_tensor(down)):
        return None
    if up.ndim != 2 or down.ndim != 2:
        return None
    return weights


def _fits(weights, target):
    return weights[0].shape[0] == target.shape[0] and weights[1].shape[1] == target.shape[1]


def _unportable_reason(patch, weights, target, basis_v, reverse):
    if len(weights) > 3 and weights[3] is not None:
        return "locon mid weights"
    if len(weights) > 4 and weights[4] is not None:
        return "DoRA scale"
    if len(weights) > 5 and weights[5] is not None:
        return "reshape_weight"
    if patch[2] != 1.0:
        return f"strength_model={patch[2]!r} scales base weight"
    if patch[3] is not None:
        return "patch offset"
    if patch[4] is not None:
        return "patch function"
    want_lora, want_target = (basis_v.shape[1], basis_v.shape[0]) if reverse else (basis_v.shape[0], basis_v.shape[1])
    if weights[1].shape[1] != want_lora:
        return f"LoRA input dim {weights[1].shape[1]}, basis expects {want_lora}"
    if target.shape[1] != want_target:
        return f"target width {target.shape[1]}, basis produces {want_target}"
    if weights[0].shape[0] != target.shape[0]:
        return f"LoRA output dim {weights[0].shape[0]}, target {target.shape[0]}"
    return None


def patch_scale(patch, weights):
    alpha = weights[2]
    rank = weights[1].shape[0]
    return float(patch[0]) * (float(alpha) / rank if alpha is not None else 1.0)


def scan_mismatched(patches, state_dict):
    found = {}
    for key in list((patches or {}).keys()):
        if not key.endswith(ADALN_WEIGHT_SUFFIX):
            continue
        target = state_dict.get(key)
        if target is None or target.ndim != 2:
            continue
        keep, bad = [], []
        for patch in patches[key]:
            weights = _lora_weights(patch[1])
            if weights is None or _fits(weights, target):
                keep.append(patch)
            else:
                bad.append((patch, weights))
        if bad:
            found[key] = (keep, bad, target)
    return found


def _store_dtype(tensor):
    dtype = getattr(tensor, "dtype", None)
    return dtype if getattr(dtype, "is_floating_point", False) else torch.float32


def _check(basis, missing):
    if basis is None:
        return None, missing
    residual = basis[2]
    if residual != residual or residual > MAX_RESIDUAL:
        return None, f"best basis fit is {residual:.2e} (limit {MAX_RESIDUAL:.0e}), too poor to trust"
    return basis, None


def resolve_basis(diffusion_model):
    table = getattr(diffusion_model, "adaln_t_table", None)
    if not torch.is_tensor(table) or table.ndim != 2:
        return None, "model has no readable adaln_t_table"
    if table.device.type == "meta":
        return None, "adaln_t_table is not materialised"
    try:
        basis = get_basis(table)
    except Exception as exc:
        return None, f"basis lookup failed: {exc}"
    return _check(
        basis,
        f"no AdaLN basis available: put a non-pruned H3 checkpoint in models/diffusion_models, or a {GRID_FILENAME} in models/{BASIS_SUBDIR}",
    )


def resolve_basis_reverse(diffusion_model):
    embedder = getattr(diffusion_model, "time_embedder", None)
    grid = grid_from_time_embedder(embedder) if embedder is not None else None
    if grid is None:
        return None, "could not read this model time embedder"
    try:
        basis = get_basis_dense(grid)
    except Exception as exc:
        return None, f"basis lookup failed: {exc}"
    return _check(basis, "no adaln_t_table found: put a pruned (curve-form) H3 checkpoint in models/diffusion_models")


def fix_model(model, mode="port"):
    """Port or strip mismatched MiniMax-H3 AdaLN LoRA patches on `model` -> `(model, report)`."""
    report = {
        "mode": mode, "effective_mode": mode, "status": "ok", "keys": 0,
        "ported": 0, "stripped": 0, "unportable": 0, "residual": None,
        "source": None, "direction": None, "notes": [],
    }
    if mode == "off" or model is None:
        report["status"] = "disabled"
        return model, report

    diffusion_model = getattr(getattr(model, "model", None), "diffusion_model", None)
    if diffusion_model is None and hasattr(model, "get_model_object"):
        try:
            diffusion_model = model.get_model_object("diffusion_model")
        except Exception:
            diffusion_model = None
    if diffusion_model is None:
        report["status"] = "not a diffusion model, nothing to do"
        return model, report

    reverse = not getattr(diffusion_model, "use_adaln_curves", False)
    if reverse and not hasattr(diffusion_model, "time_embedder"):
        report["status"] = "not a MiniMax-H3 model, nothing to do"
        return model, report
    report["direction"] = "curve->dense" if reverse else "dense->curve"

    found = scan_mismatched(getattr(model, "patches", None), model.model_state_dict())
    if not found:
        report["status"] = "no mismatched AdaLN patches"
        return model, report
    report["keys"] = len(found)

    basis = None
    if mode == "port":
        basis, reason = resolve_basis_reverse(diffusion_model) if reverse else resolve_basis(diffusion_model)
        if basis is None:
            report["effective_mode"] = "strip"
            report["notes"].append(reason)
        else:
            report["residual"], report["source"] = basis[2], basis[3]

    from comfy.weight_adapter import LoRAAdapter  # noqa: PLC0415

    patched = model.clone()
    state_dict = patched.model_state_dict()
    new_patches = {}
    pinv_v = pinv_basis(basis[1]) if (reverse and basis is not None) else None

    for key, (keep, bad, target) in found.items():
        if keep:
            patched.patches[key] = keep
        else:
            patched.patches.pop(key, None)

        if basis is None:
            report["stripped"] += len(bad)
            continue

        c, basis_v = basis[0], basis[1]
        delta_w = delta_b = None
        ups, downs = [], []
        for patch, weights in bad:
            reason = _unportable_reason(patch, weights, target, basis_v, reverse)
            if reason is not None:
                report["unportable"] += 1
                report["notes"].append(f"{key}: {reason}")
                continue
            scale = patch_scale(patch, weights)
            if reverse:
                a_dense, a_c = port_factors_reverse(weights[1], scale, c, pinv_v)
                ups.append(weights[0])
                downs.append(a_dense.to(weights[0].dtype))
                step_b = -(weights[0].detach().float() @ a_c)
            else:
                step_w, step_b = port_delta(weights[0], weights[1], scale, c, basis_v)
                delta_w = step_w if delta_w is None else delta_w + step_w
            delta_b = step_b if delta_b is None else delta_b + step_b
            report["ported"] += 1

        if reverse and ups:
            merged = (torch.cat(ups, dim=1), torch.cat(downs, dim=0), None, None, None, None)
            new_patches[key] = LoRAAdapter(set(), merged)
        elif not reverse and delta_w is not None:
            new_patches[key] = ("diff", (delta_w.to(_store_dtype(target)),))
        else:
            continue

        bias_key = key[:-len("weight")] + "bias"
        bias = state_dict.get(bias_key)
        if bias is None:
            report["notes"].append(f"{bias_key} missing, DC term dropped")
        else:
            new_patches[bias_key] = ("diff", (delta_b.to(_store_dtype(bias)),))

    if new_patches:
        patched.add_patches(new_patches, 1.0)
    patched.patches_uuid = uuid.uuid4()
    return patched, report


def fix_model_safe(model, mode="port"):
    """`fix_model` that never fails the run: on any error the model passes through
    unported and the report says why (the 3.5.1 'not installed' behaviour)."""
    try:
        return fix_model(model, mode=mode)
    except Exception as exc:  # noqa: BLE001
        return model, {"mode": mode, "effective_mode": "off", "keys": 1,
                       "status": f"skipped ({exc.__class__.__name__}: {exc}); LoRA AdaLN keys apply unported"}


def format_report(report):
    if report["status"] != "ok":
        return f"AdaLN LoRA fix: {report['status']}"
    if report["effective_mode"] == "strip":
        line = f"AdaLN LoRA fix: stripped {report['stripped']} incompatible patch(es) across {report['keys']} key(s)"
    else:
        line = (
            f"AdaLN LoRA fix: ported {report['ported']} patch(es) across {report['keys']} key(s) "
            f"{report['direction']}, basis from {report['source']} (residual {report['residual']:.2e})"
        )
    if report["unportable"]:
        line += f", {report['unportable']} could not be ported"
    return line


class _HTCAdaLNFixImpl:
    """Built-in MiniMax-H3 AdaLN LoRA port node (zero external dependency)."""


    def apply(self, model, mode="port"):
        try:
            patched, report = fix_model(model, mode=mode)
            if report.get("keys"):
                print(f"[{TAG}] {format_report(report)}", flush=True)
            return (patched,)
        except Exception as exc:
            print(f"[{TAG}] AdaLN LoRA fix skipped ({exc}); passing model unchanged.", flush=True)
            return (model,)


class HTCAdaLNFix(io.ComfyNode, _HTCAdaLNFixImpl):
    """V3 schema for HTCAdaLNFix; the implementation lives in _HTCAdaLNFixImpl.

    Also inherits _HTCAdaLNFixImpl so existing callers of the V1 API (HTCAdaLNFix().apply(...),
    HTCAdaLNFix.<helper>) keep working; core sees only the V3 hooks below."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id='HTCAdaLNFix',
            display_name='HTC AdaLN LoRA Fix (H3)',
            category="Hand Tie Clips",
            description="Built-in MIT AdaLN LoRA porter for MiniMax-H3 (dense <-> pruned curve-form). "
                "Note: HTCTokenLoRA already runs this automatically when adaln_fix='port'.",
            inputs=[
                io.Model.Input(
                    'model',
                    tooltip="MiniMax-H3 model carrying LoRA patches on the wire (dense or curve-form pruned checkpoint).",
                ),
                io.Combo.Input(
                    'mode',
                    options=list(MODES),
                    default=MODES[0],
                    tooltip="How mismatched AdaLN LoRA patches are handled: "
                        "port (recommended) converts dense <-> curve-form AdaLN deltas onto this checkpoint's time basis; "
                        "strip removes mismatched AdaLN patches while keeping all other LoRA layers; "
                        "off passes patches unchanged.",
                ),
            ],
            outputs=[
                io.Model.Output(display_name='model'),
            ],
        )

    @classmethod
    def execute(cls, **kwargs):
        result = _HTCAdaLNFixImpl().apply(**kwargs)
        if isinstance(result, dict):
            return io.NodeOutput.from_dict(result)
        return io.NodeOutput(*result) if result is not None else io.NodeOutput()


NODE_CLASS_MAPPINGS = {"HTCAdaLNFix": HTCAdaLNFix}
NODE_DISPLAY_NAME_MAPPINGS = {"HTCAdaLNFix": "HTC AdaLN LoRA Fix (H3)"}
