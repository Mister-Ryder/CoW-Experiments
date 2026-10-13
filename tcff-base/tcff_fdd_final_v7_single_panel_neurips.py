#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tcff_fdd_compare_all.py
=======================
All-method FDD analysis with Concat as the fixed baseline/anchor.

This script is designed for the current TopCoW TCFF codebase. It supports:
  1) --family dual
     shared encoder + auxiliary decoder models in topcow_model.py / topcow_model_baselines.py
  2) --family foundation
     frozen foundation encoder + projector + fusion models in topcow_foundation_model.py

Core design
-----------
Concat is the fixed baseline in two senses:
  A. The common FDD axis is computed from the Concat model's fusion inputs.
     All methods are evaluated on the same sampled voxels and the same FDD bins.
  B. Each method's local error is compared against Concat:
        attenuation(method) = error(Concat) - error(method)

This avoids the previous limitation where the FDD axis was defined only from TCFF/TGF,
which made DSFF/BAGF/CrossAttn comparisons harder to interpret.

Typical usage
-------------

PUG is supported by the loader and method parser, but you can omit it if no checkpoint exists.
"""

import os
import sys
import csv
import json
import argparse
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scipy import ndimage

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

NUM_CLASSES = 14
LEVELS = ["deep", "mid", "shallow"]
VESSEL_NAMES = {
    1: "BA", 2: "R-PCA", 3: "L-PCA", 4: "R-ICA", 5: "R-MCA",
    6: "L-ICA", 7: "L-MCA", 8: "R-Pcom", 9: "L-Pcom", 10: "Acom",
    11: "R-ACA", 12: "L-ACA", 13: "3rd-A2",
}

# Allow env-based SAM-Med3D path before model construction.
_ENV_SAM = os.environ.get("SAM_MED3D_PATH", '/mnt/e/00-Data/Brain-TopCoW/24/topcow-TGF/SAM-Med3D')
if _ENV_SAM and _ENV_SAM not in sys.path:
    sys.path.insert(0, _ENV_SAM)

# -----------------------------------------------------------------------------
# Generic utilities
# -----------------------------------------------------------------------------

def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def write_json(obj: Any, path: str) -> None:
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_csv(rows: List[Dict[str, Any]], path: str) -> None:
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    if not rows:
        with open(path, "w", encoding="utf-8") as f:
            f.write("")
        return
    keys, seen = [], set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                keys.append(k)
                seen.add(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def safe_torch_load(path: str, device: torch.device) -> Any:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_state(model: torch.nn.Module, ckpt_path: str, device: torch.device,
               strict: bool = False, label: str = "model") -> torch.nn.Module:
    state = safe_torch_load(ckpt_path, device)
    sd = state.get("model_state_dict", state)
    if strict:
        model.load_state_dict(sd, strict=True)
    else:
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing or unexpected:
            print(f"  [WARN:{label}] missing={len(missing)} unexpected={len(unexpected)}")
            if missing:
                print(f"    first missing: {missing[:5]}")
            if unexpected:
                print(f"    first unexpected: {unexpected[:5]}")
    model.to(device).eval()
    if hasattr(model, "set_epoch"):
        model.set_epoch(999999)
    return model


def ci95(vals: List[float]) -> float:
    arr = np.asarray(vals, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 1:
        return 0.0
    return float(1.96 * arr.std(ddof=1) / np.sqrt(arr.size))


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3 or a[m].std() < 1e-12 or b[m].std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a[m], b[m])[0, 1])


def normalize_image(x: np.ndarray, p_low: float = 1.0, p_high: float = 99.5) -> np.ndarray:
    x = x.astype(np.float32)
    finite = np.isfinite(x)
    if not finite.any():
        return np.zeros_like(x, dtype=np.float32)
    lo, hi = np.percentile(x[finite], [p_low, p_high])
    if hi <= lo:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip((x - lo) / (hi - lo), 0, 1).astype(np.float32)


def compute_dice(p: np.ndarray, g: np.ndarray, smooth: float = 1e-7) -> float:
    p = p.astype(bool)
    g = g.astype(bool)
    inter = np.logical_and(p, g).sum()
    return float((2.0 * inter + smooth) / (p.sum() + g.sum() + smooth))


def compute_b0(pred: np.ndarray, gt: np.ndarray) -> int:
    def b0(x: np.ndarray) -> int:
        if x.sum() == 0:
            return 0
        _, n = ndimage.label(x.astype(np.uint8))
        return int(n)
    return abs(b0(pred > 0) - b0(gt > 0))


def setup_imports(args: argparse.Namespace) -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    for p in [script_dir, os.getcwd(), args.code_dir]:
        if p and os.path.abspath(p) not in sys.path:
            sys.path.insert(0, os.path.abspath(p))
    sam_path = args.sam_med3d_path or os.environ.get("SAM_MED3D_PATH", None)
    if sam_path and os.path.abspath(sam_path) not in sys.path:
        sys.path.insert(0, os.path.abspath(sam_path))


def get_case_name(batch: Dict[str, Any]) -> str:
    name = batch.get("case_name", "case")
    if isinstance(name, (list, tuple)):
        return str(name[0])
    return str(name)


def safe_quantile_edges(values: np.ndarray, n_bins: int) -> np.ndarray:
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.linspace(0, 1, n_bins + 1)
    vmin, vmax = float(v.min()), float(v.max())
    if vmax - vmin < 1e-12:
        eps = 1e-6 if vmax == 0 else abs(vmax) * 1e-6
        return np.linspace(vmin - eps, vmax + eps, n_bins + 1)
    edges = np.quantile(v, np.linspace(0, 1, n_bins + 1))
    if np.unique(edges).size < max(3, n_bins // 2):
        edges = np.linspace(vmin, vmax, n_bins + 1)
    edges[0] -= 1e-8
    edges[-1] += 1e-8
    return edges


def percentile_rank_map(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = np.zeros(values.shape, dtype=np.float32)
    idx = np.flatnonzero(mask.reshape(-1))
    if idx.size == 0:
        return out
    flat = values.reshape(-1)
    vals = flat[idx]
    order = np.argsort(vals, kind="mergesort")
    ranks = np.empty(vals.shape[0], dtype=np.float32)
    if vals.size == 1:
        ranks[order] = 0.5
    else:
        ranks[order] = np.linspace(0.0, 1.0, vals.size, dtype=np.float32)
    out.reshape(-1)[idx] = ranks
    return out


def cosine_divergence(seg_feat: torch.Tensor, aux_feat: torch.Tensor) -> torch.Tensor:
    if seg_feat.shape[2:] != aux_feat.shape[2:]:
        aux_feat = F.interpolate(aux_feat, size=seg_feat.shape[2:], mode="trilinear", align_corners=False)
    seg_n = F.normalize(seg_feat, p=2, dim=1, eps=1e-6)
    aux_n = F.normalize(aux_feat, p=2, dim=1, eps=1e-6)
    return (1.0 - (seg_n * aux_n).sum(dim=1, keepdim=True)).clamp(0.0, 2.0)


def upsample_map(t: torch.Tensor, shape: Tuple[int, int, int], clamp01: bool = False) -> np.ndarray:
    if t.shape[2:] != shape:
        t = F.interpolate(t, size=shape, mode="trilinear", align_corners=False)
    if clamp01:
        t = t.clamp(0.0, 1.0)
    return t[0, 0].detach().cpu().numpy().astype(np.float32)


def apply_aux_transform_if_available(fusion_module: torch.nn.Module,
                                     aux_feat: torch.Tensor,
                                     target_shape: Tuple[int, int, int]) -> torch.Tensor:
    if aux_feat.shape[2:] != target_shape:
        aux_feat = F.interpolate(aux_feat, size=target_shape, mode="trilinear", align_corners=False)
    # Dual baseline modules from fusion_baselines.py use transform/transform_aux/adapt.
    if hasattr(fusion_module, "transform"):
        return fusion_module.transform(aux_feat)
    if hasattr(fusion_module, "transform_aux"):
        return fusion_module.transform_aux(aux_feat)
    if hasattr(fusion_module, "adapt"):
        return fusion_module.adapt(aux_feat)
    return aux_feat


def fdd_from_features(main_feat: torch.Tensor,
                      aux_feat: torch.Tensor,
                      fusion_module: torch.nn.Module,
                      gt_shape: Tuple[int, int, int]) -> np.ndarray:
    aux_t = apply_aux_transform_if_available(fusion_module, aux_feat, main_feat.shape[2:])
    fdd = cosine_divergence(main_feat, aux_t)
    return upsample_map(fdd, gt_shape, clamp01=False)

# -----------------------------------------------------------------------------
# Model specs and loading
# -----------------------------------------------------------------------------

def parse_method_specs(specs: Optional[List[str]]) -> List[Dict[str, str]]:
    """Parse specs like Label=type:/path/to/best_model.pth."""
    methods = []
    for s in specs or []:
        if "=" not in s or ":" not in s.split("=", 1)[1]:
            raise ValueError(
                f"Bad --method_specs entry: {s}\n"
                "Expected format: Label=type:/path/to/best_model.pth"
            )
        label, rest = s.split("=", 1)
        mtype, ckpt = rest.split(":", 1)
        methods.append({"label": label.strip(), "type": mtype.strip().lower(), "ckpt": ckpt.strip()})
    return methods


def read_config_near_ckpt(ckpt_path: str) -> Dict[str, Any]:
    cfg_path = os.path.join(os.path.dirname(os.path.abspath(ckpt_path)), "config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def load_dual_model(ckpt_path: str, model_type: str, device: torch.device,
                    strict: bool = False, label: str = "model") -> torch.nn.Module:
    from topcow_model import TopCoWTGFModel, TopCoWUNet
    from topcow_model_baselines import TopCoWBaselineFusionModel

    model_type = model_type.lower()
    if model_type == "tgf":
        model = TopCoWTGFModel(disable_gating=False)
    elif model_type == "tgf_nogate":
        model = TopCoWTGFModel(disable_gating=True)
    elif model_type == "unet":
        model = TopCoWUNet()
    elif model_type in ["concat", "crossattn", "bagf", "pug", "dsff"]:
        model = TopCoWBaselineFusionModel(fusion_type=model_type)
    else:
        raise ValueError(f"Unknown dual model_type={model_type}")
    model = load_state(model, ckpt_path, device, strict=strict, label=label)
    print(f"  [dual:{label}/{model_type}] {ckpt_path}")
    return model


def load_foundation_model(ckpt_path: str, fusion_type: Optional[str], device: torch.device,
                          foundation_type: Optional[str], foundation_ckpt: Optional[str],
                          strict: bool = False, label: str = "model") -> torch.nn.Module:
    from topcow_foundation_model import TopCoWFoundationModel, build_foundation_encoder

    cfg = read_config_near_ckpt(ckpt_path)
    ft = foundation_type or cfg.get("foundation_type")
    fuse = fusion_type or cfg.get("fusion_type")
    fckpt = foundation_ckpt if foundation_ckpt is not None else cfg.get("foundation_ckpt")
    if ft is None or fuse is None:
        raise ValueError(
            f"Cannot infer foundation spec for {ckpt_path}. Provide --foundation_type and method fusion type."
        )
    encoder = build_foundation_encoder(ft, fckpt)
    model = TopCoWFoundationModel(encoder, fusion_type=fuse)
    model = load_state(model, ckpt_path, device, strict=strict, label=label)
    print(f"  [foundation:{label}/{fuse}] foundation={ft} foundation_ckpt={fckpt} ckpt={ckpt_path}")
    return model


def load_all_models(args: argparse.Namespace, device: torch.device) -> Tuple[torch.nn.Module, Dict[str, torch.nn.Module], List[str], Dict[str, str]]:
    method_specs = parse_method_specs(args.method_specs)
    labels = ["Concat"]
    models: Dict[str, torch.nn.Module] = {}
    method_types: Dict[str, str] = {"Concat": "concat"}

    if args.family == "dual":
        concat_model = load_dual_model(args.concat_ckpt, "concat", device, strict=args.strict_load, label="Concat")
        models["Concat"] = concat_model
        for spec in method_specs:
            if spec["label"] == "Concat":
                continue
            models[spec["label"]] = load_dual_model(spec["ckpt"], spec["type"], device,
                                                     strict=args.strict_load, label=spec["label"])
            labels.append(spec["label"])
            method_types[spec["label"]] = spec["type"]
        return concat_model, models, labels, method_types

    if args.family == "foundation":
        concat_model = load_foundation_model(args.concat_ckpt, "concat", device,
                                             args.foundation_type, args.foundation_ckpt,
                                             strict=args.strict_load, label="Concat")
        models["Concat"] = concat_model
        for spec in method_specs:
            if spec["label"] == "Concat":
                continue
            models[spec["label"]] = load_foundation_model(spec["ckpt"], spec["type"], device,
                                                           args.foundation_type, args.foundation_ckpt,
                                                           strict=args.strict_load, label=spec["label"])
            labels.append(spec["label"])
            method_types[spec["label"]] = spec["type"]
        return concat_model, models, labels, method_types

    raise ValueError(f"Unknown family={args.family}")

# -----------------------------------------------------------------------------
# Anchor FDD probing: Concat is the only FDD anchor
# -----------------------------------------------------------------------------

@torch.no_grad()
def predict_model(model: torch.nn.Module, x: torch.Tensor) -> np.ndarray:
    seg, _, _ = model(x)
    return torch.argmax(seg[0], dim=0).detach().cpu().numpy().astype(np.uint8)


@torch.no_grad()
def probe_dual_concat_anchor(concat_model: torch.nn.Module,
                             x: torch.Tensor,
                             gt_shape: Tuple[int, int, int]) -> Dict[str, Any]:
    """Manual forward for TopCoWBaselineFusionModel(fusion_type='concat'), with FDD maps."""
    enc_feats = []
    x_enc = x
    for encoder in concat_model.encoders:
        x_enc = encoder(x_enc)
        enc_feats.append(x_enc)
    enc_rev = list(reversed(enc_feats))
    bottleneck, skips = enc_rev[0], enc_rev[1:]  # [128,64,32]

    aux_x = bottleneck
    aux_feats = []
    for i, dec in enumerate(concat_model.aux_decoders):
        aux_x = dec(skips[i], aux_x)
        aux_feats.append(aux_x)

    fdd_shallow = fdd_from_features(skips[2], aux_feats[2], concat_model.fusion_shallow, gt_shape)
    modified_skip2 = concat_model.fusion_shallow(skips[2], aux_feats[2])

    main_x = bottleneck
    main_feats = []
    fdd_deep = fdd_mid = None
    for i, seg_dec in enumerate(concat_model.seg_decoders):
        main_x = seg_dec(modified_skip2 if i == 2 else skips[i], main_x)
        if i == 0:
            fdd_deep = fdd_from_features(main_x, aux_feats[0], concat_model.fusion_deep, gt_shape)
            main_x = concat_model.fusion_deep(main_x, aux_feats[0])
        elif i == 1:
            fdd_mid = fdd_from_features(main_x, aux_feats[1], concat_model.fusion_mid, gt_shape)
            main_x = concat_model.fusion_mid(main_x, aux_feats[1])
        main_feats.append(main_x)

    dec_fusion = concat_model.dropout2(concat_model.fusion_module_decoder(main_feats))
    seg_logits = concat_model.seg_final(dec_fusion)
    seg = concat_model.seg_activation(seg_logits)
    pred = torch.argmax(seg[0], dim=0).detach().cpu().numpy().astype(np.uint8)
    maps = {"fdd_deep_raw": fdd_deep, "fdd_mid_raw": fdd_mid, "fdd_shallow_raw": fdd_shallow}
    return {"pred": pred, "maps": maps}


@torch.no_grad()
def probe_foundation_concat_anchor(concat_model: torch.nn.Module,
                                   x: torch.Tensor,
                                   gt_shape: Tuple[int, int, int]) -> Dict[str, Any]:
    """Manual forward for TopCoWFoundationModel(fusion_type='concat'), with common FDD maps."""
    with torch.no_grad():
        found_feat = concat_model.foundation_encoder(x)

    enc_feats = []
    x_enc = x
    for encoder in concat_model.encoders:
        x_enc = encoder(x_enc)
        enc_feats.append(x_enc)
    enc_rev = list(reversed(enc_feats))
    bottleneck, skips = enc_rev[0], enc_rev[1:]

    target_shapes = [s.shape[2:] for s in skips]
    found_multi = concat_model.projector(found_feat, target_shapes)  # [deep, mid, shallow]

    fdd_shallow_t = cosine_divergence(skips[2], found_multi[2])
    fdd_shallow = upsample_map(fdd_shallow_t, gt_shape)
    modified_skip2 = concat_model.fusion_shallow(skips[2], found_multi[2])

    main_x = bottleneck
    main_feats = []
    fdd_deep = fdd_mid = None
    for i, seg_dec in enumerate(concat_model.seg_decoders):
        main_x = seg_dec(modified_skip2 if i == 2 else skips[i], main_x)
        if i == 0:
            fdd_deep = upsample_map(cosine_divergence(main_x, found_multi[0]), gt_shape)
            main_x = concat_model.fusion_deep(main_x, found_multi[0])
        elif i == 1:
            fdd_mid = upsample_map(cosine_divergence(main_x, found_multi[1]), gt_shape)
            main_x = concat_model.fusion_mid(main_x, found_multi[1])
        main_feats.append(main_x)

    dec_fusion = concat_model.dropout2(concat_model.fusion_module_decoder(main_feats))
    seg_logits = concat_model.seg_final(dec_fusion)
    seg = concat_model.seg_activation(seg_logits)
    pred = torch.argmax(seg[0], dim=0).detach().cpu().numpy().astype(np.uint8)
    maps = {"fdd_deep_raw": fdd_deep, "fdd_mid_raw": fdd_mid, "fdd_shallow_raw": fdd_shallow}
    return {"pred": pred, "maps": maps}

# -----------------------------------------------------------------------------
# Optional TCFF/TGF response probing for response curves
# -----------------------------------------------------------------------------

class FeatureHook:
    def __init__(self):
        self.features = None
        self.handle = None

    def hook_fn(self, module, inputs, output):
        self.features = output

    def register(self, module):
        self.handle = module.register_forward_hook(self.hook_fn)
        return self

    def remove(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


@torch.no_grad()
def probe_dual_tgf_response(model: torch.nn.Module, x: torch.Tensor,
                            gt_shape: Tuple[int, int, int]) -> Dict[str, np.ndarray]:
    """For dual TGF negative-control response. These maps are not auxiliary trust."""
    from topcow_model import NUM_CIRCUIT, NUM_CALIBER, NUM_LATERALITY

    h_main_shallow = FeatureHook().register(model.encoders[0])
    h_main_deep = FeatureHook().register(model.seg_decoders[0])
    h_main_mid = FeatureHook().register(model.seg_decoders[1])
    h_out_deep = FeatureHook().register(model.gaam)
    h_out_mid = FeatureHook().register(model.saam)
    h_out_shallow = FeatureHook().register(model.laam)
    h_wu_deep = FeatureHook().register(model.gaam.W_u)
    h_wu_mid = FeatureHook().register(model.saam.W_u)
    h_wu_shallow = FeatureHook().register(model.laam.W_u)
    hooks = [h_main_shallow, h_main_deep, h_main_mid, h_out_deep, h_out_mid, h_out_shallow,
             h_wu_deep, h_wu_mid, h_wu_shallow]
    try:
        _, aux_out, _ = model(x)
    finally:
        for h in hooks:
            h.remove()

    _, unc_deep = model._edl(aux_out["circuit_logits"], NUM_CIRCUIT)
    _, unc_mid = model._edl(aux_out["caliber_logits"], NUM_CALIBER)
    _, unc_shallow = model._edl(aux_out["laterality_logits"], NUM_LATERALITY)

    pre = {"deep": h_main_deep.features, "mid": h_main_mid.features, "shallow": h_main_shallow.features}
    post = {"deep": h_out_deep.features, "mid": h_out_mid.features, "shallow": h_out_shallow.features}
    wu = {"deep": h_wu_deep.features, "mid": h_wu_mid.features, "shallow": h_wu_shallow.features}
    unc = {"deep": unc_deep, "mid": unc_mid, "shallow": unc_shallow}

    maps = {}
    for lv in LEVELS:
        u = unc[lv]
        w = wu[lv]
        if u.shape[2:] != w.shape[2:]:
            u = F.interpolate(u, size=w.shape[2:], mode="trilinear", align_corners=False)
        reversion = (w * u).clamp(0, 1)
        conditional_weight = 1.0 - reversion
        pre_l, post_l = pre[lv], post[lv]
        if pre_l.shape[2:] != post_l.shape[2:]:
            pre_l = F.interpolate(pre_l, size=post_l.shape[2:], mode="trilinear", align_corners=False)
        delta = torch.norm(post_l - pre_l, p=2, dim=1, keepdim=True) / np.sqrt(float(post_l.shape[1]))
        maps[f"response_uncertainty_{lv}"] = upsample_map(u, gt_shape, clamp01=True)
        maps[f"response_wu_{lv}"] = upsample_map(w, gt_shape, clamp01=True)
        maps[f"response_reversion_{lv}"] = upsample_map(reversion, gt_shape, clamp01=True)
        maps[f"response_weight_{lv}"] = upsample_map(conditional_weight, gt_shape, clamp01=True)
        maps[f"response_fusion_delta_{lv}"] = upsample_map(delta, gt_shape, clamp01=False)
    return finalize_response_maps(maps)


def foundation_tgf_forward_diag(module: torch.nn.Module,
                                seg_feat: torch.Tensor,
                                aux_feat: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    if seg_feat.shape[2:] != aux_feat.shape[2:]:
        aux_feat = F.interpolate(aux_feat, size=seg_feat.shape[2:], mode="trilinear", align_corners=False)
    seg_global = seg_feat.mean(dim=[2, 3, 4])
    aux_global = aux_feat.mean(dim=[2, 3, 4])
    channel_weight = module.channel_gate(torch.cat([seg_global, aux_global], dim=1))
    aux_filtered = aux_feat * channel_weight[:, :, None, None, None]
    fused = module.fuse(torch.cat([seg_feat, aux_filtered], dim=1)) + seg_feat
    edl_logits = module.edl_head(aux_global)
    evidence = F.relu(edl_logits / module.temperature.clamp(min=0.1))
    alpha = evidence + 1.0
    S = alpha.sum(dim=1, keepdim=True)
    uncertainty = module.num_edl_classes / S
    uncertainty_map = uncertainty[:, :, None, None, None]
    w = module.W_u(fused)
    reversion = (w * uncertainty_map).clamp(0.0, 1.0)
    out = fused * (1.0 - reversion) + seg_feat * reversion
    diag = {
        "uncertainty": uncertainty_map,
        "w_u": w,
        "reversion": reversion,
        "weight": 1.0 - reversion,
        "fusion_delta": torch.norm(out - seg_feat, p=2, dim=1, keepdim=True) / np.sqrt(float(seg_feat.shape[1])),
        "channel_weight_mean": channel_weight.mean(dim=1, keepdim=True)[:, :, None, None, None],
    }
    return out, diag


@torch.no_grad()
def probe_foundation_tgf_response(model: torch.nn.Module, x: torch.Tensor,
                                  gt_shape: Tuple[int, int, int]) -> Dict[str, np.ndarray]:
    assert getattr(model, "fusion_type", None) == "tgf"
    with torch.no_grad():
        found_feat = model.foundation_encoder(x)
    enc_feats = []
    x_enc = x
    for encoder in model.encoders:
        x_enc = encoder(x_enc)
        enc_feats.append(x_enc)
    enc_rev = list(reversed(enc_feats))
    bottleneck, skips = enc_rev[0], enc_rev[1:]
    found_multi = model.projector(found_feat, [s.shape[2:] for s in skips])

    maps = {}
    modified_skip2, diag_s = foundation_tgf_forward_diag(model.fusion_shallow, skips[2], found_multi[2])
    diag_by_level = {"shallow": diag_s}
    main_x = bottleneck
    main_feats = []
    for i, seg_dec in enumerate(model.seg_decoders):
        main_x = seg_dec(modified_skip2 if i == 2 else skips[i], main_x)
        if i == 0:
            main_x, diag_d = foundation_tgf_forward_diag(model.fusion_deep, main_x, found_multi[0])
            diag_by_level["deep"] = diag_d
        elif i == 1:
            main_x, diag_m = foundation_tgf_forward_diag(model.fusion_mid, main_x, found_multi[1])
            diag_by_level["mid"] = diag_m
        main_feats.append(main_x)

    for lv in LEVELS:
        d = diag_by_level[lv]
        for k, v in d.items():
            maps[f"response_{k}_{lv}"] = upsample_map(v, gt_shape, clamp01=(k in ["uncertainty", "w_u", "reversion", "weight", "channel_weight_mean"]))
    return finalize_response_maps(maps)


def finalize_response_maps(maps: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    out = dict(maps)
    for name in ["uncertainty", "w_u", "reversion", "weight", "fusion_delta", "channel_weight_mean"]:
        keys = [f"response_{name}_{lv}" for lv in LEVELS if f"response_{name}_{lv}" in out]
        if keys:
            out[f"response_{name}_mean"] = np.stack([out[k] for k in keys], axis=0).mean(axis=0).astype(np.float32)
    return out

# -----------------------------------------------------------------------------
# Sampling and aggregation
# -----------------------------------------------------------------------------

def make_candidate_mask(gt: np.ndarray, preds: Dict[str, np.ndarray], dilate_iter: int = 2) -> np.ndarray:
    mask = ndimage.binary_dilation(gt > 0, iterations=dilate_iter)
    for p in preds.values():
        mask |= ndimage.binary_dilation(p > 0, iterations=dilate_iter)
    return mask.astype(bool)


def finalize_fdd_maps(raw_maps: Dict[str, np.ndarray], candidate: np.ndarray) -> Dict[str, np.ndarray]:
    out = dict(raw_maps)
    rank_maps = []
    raw_stack = []
    for lv in LEVELS:
        key = f"fdd_{lv}_raw"
        if key in out and out[key] is not None:
            raw_stack.append(out[key])
            rk = percentile_rank_map(out[key], candidate)
            out[f"fdd_{lv}_rank"] = rk
            rank_maps.append(rk)
    if rank_maps:
        stack = np.stack(rank_maps, axis=0)
        out["fdd_all_max"] = stack.max(axis=0).astype(np.float32)
        out["fdd_all_mean"] = stack.mean(axis=0).astype(np.float32)
    if raw_stack:
        raw = np.stack(raw_stack, axis=0)
        out["fdd_raw_all_mean"] = raw.mean(axis=0).astype(np.float32)
        out["fdd_raw_all_max"] = raw.max(axis=0).astype(np.float32)
    return out


def sanitize_label(label: str) -> str:
    return "".join(c if c.isalnum() or c in ["_", "-"] else "_" for c in label)


def sample_case_voxels(case_idx: int,
                       gt: np.ndarray,
                       preds: Dict[str, np.ndarray],
                       maps: Dict[str, np.ndarray],
                       candidate: np.ndarray,
                       labels: List[str],
                       max_voxels: int,
                       seed: int) -> Optional[Dict[str, np.ndarray]]:
    idx = np.flatnonzero(candidate.reshape(-1))
    if idx.size == 0:
        return None
    rng = np.random.default_rng(seed + case_idx)
    if max_voxels and idx.size > max_voxels:
        idx = rng.choice(idx, size=max_voxels, replace=False)
    gt_f = gt.reshape(-1)
    rec: Dict[str, np.ndarray] = {
        "case_idx": np.full(idx.shape[0], case_idx, dtype=np.int32),
        "fdd": maps["fdd_all_max"].reshape(-1)[idx].astype(np.float32),
        "fdd_mean": maps.get("fdd_all_mean", maps["fdd_all_max"]).reshape(-1)[idx].astype(np.float32),
    }
    for lv in LEVELS:
        key = f"fdd_{lv}_rank"
        if key in maps:
            rec[key] = maps[key].reshape(-1)[idx].astype(np.float32)

    # Response diagnostics: keep level-specific quantities only.
    # We intentionally do NOT sample the mean-over-level response maps here, because
    # averaging deep/mid/shallow tends to hide the level-wise behavior of TCFF.
    # We also drop nearly constant auxiliary diagnostics such as EDL uncertainty and
    # channel_weight_mean from the response plot. The response figure focuses on:
    #   - reversion: how much TCFF rolls back to the main feature,
    #   - weight: retained fusion / enhanced-feature weight,
    #   - fusion_delta: feature-space shift induced by TCFF.
    for lv in LEVELS:
        for name in ["reversion", "weight", "fusion_delta"]:
            key = f"response_{name}_{lv}"
            if key in maps:
                rec[key] = maps[key].reshape(-1)[idx].astype(np.float32)

    for label in labels:
        s = sanitize_label(label)
        p_f = preds[label].reshape(-1)
        rec[f"err_{s}"] = (p_f[idx] != gt_f[idx]).astype(np.float32)
        rec[f"fp_{s}"] = ((p_f[idx] > 0) & (gt_f[idx] == 0)).astype(np.float32)
        rec[f"fn_{s}"] = ((p_f[idx] == 0) & (gt_f[idx] > 0)).astype(np.float32)
    return rec


def concat_records(records: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    keys = records[0].keys()
    return {k: np.concatenate([r[k] for r in records], axis=0) for k in keys}


def aggregate_bins(voxel: Dict[str, np.ndarray], labels: List[str], n_bins: int,
                   min_case_voxels: int) -> List[Dict[str, Any]]:
    fdd = voxel["fdd"]
    edges = safe_quantile_edges(fdd, n_bins)
    case_ids = np.unique(voxel["case_idx"])
    rows = []
    # Keep response statistics level-specific. Do not aggregate deep/mid/shallow before plotting.
    response_keys = [
        f"response_{name}_{lv}"
        for lv in LEVELS
        for name in ["reversion", "weight", "fusion_delta"]
        if f"response_{name}_{lv}" in voxel
    ]

    for b in range(n_bins):
        bm = (fdd >= edges[b]) & (fdd < edges[b + 1])
        row: Dict[str, Any] = {
            "bin": b + 1,
            "fdd_left": float(edges[b]),
            "fdd_right": float(edges[b + 1]),
            "fdd_mean": float(np.mean(fdd[bm])) if bm.any() else float("nan"),
            "n_voxels": int(bm.sum()),
        }
        # method metrics with case-level CI
        for label in labels:
            s = sanitize_label(label)
            for metric in ["err", "fp", "fn"]:
                key = f"{metric}_{s}"
                per_case = []
                for cid in case_ids:
                    cm = bm & (voxel["case_idx"] == cid)
                    if cm.sum() >= min_case_voxels:
                        per_case.append(float(np.mean(voxel[key][cm])))
                row[f"{metric}_{s}_mean"] = float(np.mean(per_case)) if per_case else float("nan")
                row[f"{metric}_{s}_ci95"] = ci95(per_case) if per_case else float("nan")

        # attenuation against Concat
        concat_s = sanitize_label("Concat")
        for label in labels:
            if label == "Concat":
                continue
            s = sanitize_label(label)
            for metric in ["err", "fp", "fn"]:
                key_c = f"{metric}_{concat_s}"
                key_m = f"{metric}_{s}"
                per_case = []
                for cid in case_ids:
                    cm = bm & (voxel["case_idx"] == cid)
                    if cm.sum() >= min_case_voxels:
                        per_case.append(float(np.mean(voxel[key_c][cm] - voxel[key_m][cm])))
                row[f"atten_{metric}_{s}_mean"] = float(np.mean(per_case)) if per_case else float("nan")
                row[f"atten_{metric}_{s}_ci95"] = ci95(per_case) if per_case else float("nan")

        for key in response_keys:
            per_case = []
            for cid in case_ids:
                cm = bm & (voxel["case_idx"] == cid)
                if cm.sum() >= min_case_voxels:
                    per_case.append(float(np.mean(voxel[key][cm])))
            row[f"{key}_mean"] = float(np.mean(per_case)) if per_case else float("nan")
            row[f"{key}_ci95"] = ci95(per_case) if per_case else float("nan")
        rows.append(row)
    return rows


def collect_case_rows(case_name: str, gt: np.ndarray, preds: Dict[str, np.ndarray],
                      maps: Dict[str, np.ndarray], labels: List[str]) -> List[Dict[str, Any]]:
    rows = []
    for label in labels:
        pred = preds[label]
        candidate = (gt > 0) | (pred > 0)
        denom = max(1, int(candidate.sum()))
        rows.append({
            "case": case_name,
            "method": label,
            "dice": compute_dice(pred > 0, gt > 0),
            "b0": compute_b0(pred, gt),
            "local_error": float(((pred != gt) & candidate).sum() / denom),
            "fp": float(((pred > 0) & (gt == 0) & candidate).sum() / denom),
            "fn": float(((pred == 0) & (gt > 0) & candidate).sum() / denom),
            "fdd_all_max_mean": float(np.mean(maps["fdd_all_max"][(gt > 0) | (pred > 0)])) if ((gt > 0) | (pred > 0)).any() else float("nan"),
        })
    return rows

# -----------------------------------------------------------------------------
# Plotting: publication-style line figures
# -----------------------------------------------------------------------------

# The analysis output is curve-first. Case-slice image export is intentionally
# disabled in this version; the core deliverables are the FDD-response and
# method-vs-Concat line curves.
PLOT_EXTENSIONS = ("png", "pdf")


def configure_neurips_matplotlib() -> None:
    """Compact, vector-friendly matplotlib defaults for NeurIPS-style figures."""
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.035,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "DejaVu Sans",
        "font.size": 8.2,
        "axes.titlesize": 8.8,
        "axes.labelsize": 8.4,
        "axes.linewidth": 0.7,
        "xtick.labelsize": 7.6,
        "ytick.labelsize": 7.6,
        "legend.fontsize": 7.8,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
    })


configure_neurips_matplotlib()


def arr(rows: List[Dict[str, Any]], key: str) -> np.ndarray:
    return np.asarray([float(r.get(key, np.nan)) for r in rows], dtype=np.float64)


# Color-blind-friendly palette with TCFF as the visual anchor.
METHOD_COLORS = {
    "concat": "#4D4D4D",
    "tcff": "#D55E00",       # highlighted method
    "tgf": "#D55E00",        # legacy/internal name for TCFF-style fusion
    "trust": "#D55E00",
    "dsff": "#0072B2",
    "bagf": "#009E73",
    "crossattn": "#CC79A7",
    "cross_attention": "#CC79A7",
    "cross-attn": "#CC79A7",
    "pug": "#E69F00",
    "unet": "#7F7F7F",
    "tgf_nogate": "#A6761D",
    "nogate": "#A6761D",
}
FALLBACK_COLORS = ["#56B4E9", "#009E73", "#F0E442", "#0072B2", "#CC79A7", "#999999"]
METHOD_MARKERS = {
    "concat": "o",
    "tcff": "*",
    "tgf": "*",
    "dsff": "D",
    "bagf": "s",
    "crossattn": "P",
    "cross_attention": "P",
    "cross-attn": "P",
    "pug": "X",
    "unet": "v",
    "tgf_nogate": "h",
    "nogate": "h",
}


def _norm_label(label: str) -> str:
    return str(label).strip().lower().replace(" ", "_")


def _is_tcff_label(label: str) -> bool:
    z = _norm_label(label)
    return ("tcff" in z) or (z == "tgf") or ("trust" in z and "fusion" in z)


def _method_key(label: str) -> str:
    z = _norm_label(label)
    if "concat" in z:
        return "concat"
    if _is_tcff_label(label):
        return "tcff"
    if "tgf_nogate" in z or "no_gate" in z or "nogate" in z:
        return "tgf_nogate"
    if "cross" in z and "att" in z:
        return "crossattn"
    for k in ["dsff", "bagf", "pug", "unet"]:
        if k in z:
            return k
    return z


def pretty_method_label(label: str) -> str:
    key = _method_key(label)
    mapping = {
        "concat": "Concat",
        "tcff": "TCFF",
        "dsff": "DSFF",
        "bagf": "BAGF",
        "crossattn": "CrossAttn",
        "pug": "PUG",
        "unet": "UNet",
        "tgf_nogate": "TCFF w/o gate",
    }
    return mapping.get(key, str(label))


def build_method_style_map(labels: List[str]) -> Dict[str, Dict[str, Any]]:
    """Assign one stable color/marker to each method across all figures."""
    used = set()
    styles: Dict[str, Dict[str, Any]] = {}
    fallback_i = 0
    for label in labels:
        key = _method_key(label)
        color = METHOD_COLORS.get(key)
        if color is None:
            while FALLBACK_COLORS[fallback_i % len(FALLBACK_COLORS)] in used:
                fallback_i += 1
            color = FALLBACK_COLORS[fallback_i % len(FALLBACK_COLORS)]
            fallback_i += 1
        used.add(color)
        is_tcff = _is_tcff_label(label)
        styles[label] = {
            "color": color,
            "marker": METHOD_MARKERS.get(key, "o"),
            "linewidth": 2.65 if is_tcff else 1.85,
            "markersize": 7.2 if is_tcff else 4.3,
            "alpha": 1.0 if is_tcff else 0.88,
            "zorder": 8 if is_tcff else (5 if key == "concat" else 4),
            "label": pretty_method_label(label),
        }
    return styles


def _style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="both", color="#EAEAEA", linewidth=0.7)
    ax.tick_params(axis="both", length=2.6, width=0.65, color="#333333")


# def _format_fdd_axis(ax: plt.Axes) -> None:
#     ax.set_xlabel("Concat-defined FDD percentile")
#     ax.set_xlim(-0.02, 1.02)
#     ax.set_xticks(np.linspace(0, 1, 6))
#     ax.set_xticklabels(["0", "20", "40", "60", "80", "100"])
def _format_fdd_axis(ax: plt.Axes, x: Optional[np.ndarray] = None,
                     force_left: Optional[float] = None) -> None:
    """
    Format FDD percentile axis.

    x is expected to be in [0, 1], while tick labels are shown as percentiles.
    force_left can be set as 0.30 to force the axis to start from 30%.
    """
    ax.set_xlabel("Concat-defined FDD percentile")

    if force_left is not None:
        left = float(force_left)
    elif x is not None:
        xx = np.asarray(x, dtype=np.float64)
        xx = xx[np.isfinite(xx)]
        if xx.size > 0:
            # Start from the nearest lower 5-percentile tick.
            left = max(0.0, np.floor(xx.min() * 20.0) / 20.0)
        else:
            left = 0.0
    else:
        left = 0.0

    right = 1.0
    ax.set_xlim(left, right)

    # Use 10-percentile ticks after the chosen start.
    tick_start = np.ceil(left * 10.0) / 10.0
    ticks = np.arange(tick_start, 1.0001, 0.1)
    if left < tick_start - 1e-8:
        ticks = np.r_[left, ticks]

    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{int(round(t * 100))}" for t in ticks])

def save_curve_figure(fig: plt.Figure, out_dir: str, filename: str) -> None:
    ensure_dir(out_dir)
    for ext in PLOT_EXTENSIONS:
        fig.savefig(os.path.join(out_dir, f"{filename}.{ext}"), dpi=600, bbox_inches="tight", pad_inches=0.035)


def save_panel_figure(fig: plt.Figure, out_dir: str, filename: str) -> None:
    """Save standalone panels with the exact canvas ratio set by PANEL_FIGSIZE.

    Unlike save_curve_figure, this deliberately disables the global tight-bbox
    setting so the exported PNG/PDF keeps the requested ~1.2:1 panel canvas and
    leaves enough whitespace for manual cropping and later figure assembly.
    """
    ensure_dir(out_dir)
    with plt.rc_context({"savefig.bbox": None, "savefig.pad_inches": 0.0}):
        for ext in PLOT_EXTENSIONS:
            fig.savefig(os.path.join(out_dir, f"{filename}.{ext}"), dpi=600, pad_inches=0.0)


# Standalone panel exports. Each individual panel uses an approximately 1.2:1
# width-to-height ratio, which is convenient for manual cropping and later assembly
# in a NeurIPS two-column or appendix layout.
PANEL_FIGSIZE = (3.6, 3.0)        # width:height = 1.2:1
TWO_PANEL_FIGSIZE = (8.4, 3.05)   # spacious overview figure
THREE_PANEL_FIGSIZE = (12.6, 3.05)


def _panel_dir(out_dir: str) -> str:
    return ensure_dir(os.path.join(out_dir, "panels"))


def _finalize_single_panel(fig: plt.Figure, ax: plt.Axes) -> None:
    ax.yaxis.labelpad = 6.5
    ax.xaxis.labelpad = 5.0
    fig.subplots_adjust(left=0.18, right=0.965, bottom=0.24, top=0.86)


def _save_legend_only(handles: List[Any], legend_labels: List[str], out_dir: str,
                      filename: str, ncol: Optional[int] = None) -> None:
    if not handles:
        return
    ncol = ncol or min(len(handles), 6)
    width = max(3.6, 0.92 * len(handles))
    fig = plt.figure(figsize=(width, 0.55))
    fig.legend(handles, legend_labels, loc="center", ncol=ncol, frameon=False,
               handlelength=1.9, columnspacing=1.35, handletextpad=0.45)
    save_curve_figure(fig, _panel_dir(out_dir), filename)
    plt.close(fig)


def _plot_error_probability_panel(ax: plt.Axes, x: np.ndarray, bin_rows: List[Dict[str, Any]],
                                  labels: List[str], styles: Dict[str, Dict[str, Any]]) -> Tuple[List[Any], List[str]]:
    handles, legend_labels = [], []
    for label in labels:
        s = sanitize_label(label)
        y = arr(bin_rows, f"err_{s}_mean")
        e = arr(bin_rows, f"err_{s}_ci95")
        st = styles[label]
        h, = ax.plot(
            x, y,
            marker=st["marker"], markersize=st["markersize"], linewidth=st["linewidth"],
            label=st["label"], color=st["color"], alpha=st["alpha"], zorder=st["zorder"],
            markeredgewidth=0.0,
        )
        ax.fill_between(x, y - e, y + e, alpha=0.10 if _is_tcff_label(label) else 0.075,
                        color=st["color"], linewidth=0, zorder=1)
        handles.append(h)
        legend_labels.append(st["label"])
    ax.set_title("Local error probability")
    _format_fdd_axis(ax, x, force_left=0.30)
    ax.set_ylabel("Local error probability")
    _style_axes(ax)
    return handles, legend_labels


def _plot_error_attenuation_panel(ax: plt.Axes, x: np.ndarray, bin_rows: List[Dict[str, Any]],
                                  labels: List[str], styles: Dict[str, Dict[str, Any]]) -> Tuple[List[Any], List[str]]:
    handles, legend_labels = [], []
    for label in labels:
        if label == "Concat":
            continue
        s = sanitize_label(label)
        y = arr(bin_rows, f"atten_err_{s}_mean")
        e = arr(bin_rows, f"atten_err_{s}_ci95")
        st = styles[label]
        h, = ax.plot(
            x, y,
            marker=st["marker"], markersize=st["markersize"], linewidth=st["linewidth"],
            label=st["label"], color=st["color"], alpha=st["alpha"], zorder=st["zorder"],
            markeredgewidth=0.0,
        )
        ax.fill_between(x, y - e, y + e, alpha=0.10 if _is_tcff_label(label) else 0.075,
                        color=st["color"], linewidth=0, zorder=1)
        handles.append(h)
        legend_labels.append(st["label"])
    ax.axhline(0, linestyle="--", linewidth=0.9, color="#555555", zorder=2)
    ax.set_title("Error attenuation vs. Concat")
    _format_fdd_axis(ax, x, force_left=0.30)
    ax.set_ylabel("Concat error − method error")
    _style_axes(ax)
    return handles, legend_labels


def plot_all_error_curves(bin_rows: List[Dict[str, Any]], labels: List[str], out_dir: str,
                          title: str) -> None:
    x = arr(bin_rows, "fdd_mean")
    styles = build_method_style_map(labels)

    # Spacious overview figure, retained for quick inspection.
    fig, axes = plt.subplots(1, 2, figsize=TWO_PANEL_FIGSIZE, constrained_layout=False)
    handles, legend_labels = _plot_error_probability_panel(axes[0], x, bin_rows, labels, styles)
    _plot_error_attenuation_panel(axes[1], x, bin_rows, labels, styles)
    fig.suptitle(title, fontsize=9.4, y=0.985)
    fig.legend(handles, legend_labels, loc="lower center", bbox_to_anchor=(0.5, -0.035),
               ncol=min(len(handles), 6), frameon=False, handlelength=1.8, columnspacing=1.25)
    fig.subplots_adjust(left=0.080, right=0.990, top=0.78, bottom=0.29, wspace=0.58)
    save_curve_figure(fig, out_dir, "curve_all_methods_error_vs_concat")
    plt.close(fig)

    # Standalone panels for paper assembly and manual cropping.
    fig, ax = plt.subplots(figsize=PANEL_FIGSIZE, constrained_layout=False)
    handles, legend_labels = _plot_error_probability_panel(ax, x, bin_rows, labels, styles)
    _finalize_single_panel(fig, ax)
    save_panel_figure(fig, _panel_dir(out_dir), "curve_all_methods_error_probability_panel")
    _save_legend_only(handles, legend_labels, out_dir, "legend_all_methods", ncol=min(len(handles), 6))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=PANEL_FIGSIZE, constrained_layout=False)
    _plot_error_attenuation_panel(ax, x, bin_rows, labels, styles)
    _finalize_single_panel(fig, ax)
    save_panel_figure(fig, _panel_dir(out_dir), "curve_all_methods_error_attenuation_panel")
    plt.close(fig)


def _plot_component_attenuation_panel(ax: plt.Axes, x: np.ndarray, bin_rows: List[Dict[str, Any]],
                                      labels: List[str], styles: Dict[str, Dict[str, Any]],
                                      metric: str, panel_title: str, ylabel: str) -> Tuple[List[Any], List[str]]:
    handles, legend_labels = [], []
    for label in labels:
        if label == "Concat":
            continue
        s = sanitize_label(label)
        y = arr(bin_rows, f"atten_{metric}_{s}_mean")
        e = arr(bin_rows, f"atten_{metric}_{s}_ci95")
        st = styles[label]
        h, = ax.plot(
            x, y,
            marker=st["marker"], markersize=st["markersize"], linewidth=st["linewidth"],
            label=st["label"], color=st["color"], alpha=st["alpha"], zorder=st["zorder"],
            markeredgewidth=0.0,
        )
        ax.fill_between(x, y - e, y + e, alpha=0.10 if _is_tcff_label(label) else 0.075,
                        color=st["color"], linewidth=0, zorder=1)
        handles.append(h)
        legend_labels.append(st["label"])
    ax.axhline(0, linestyle="--", linewidth=0.9, color="#555555", zorder=2)
    ax.set_title(panel_title)
    _format_fdd_axis(ax, x, force_left=0.30)
    ax.set_ylabel(ylabel)
    _style_axes(ax)
    return handles, legend_labels


def plot_fp_fn_attenuation(bin_rows: List[Dict[str, Any]], labels: List[str], out_dir: str,
                           title: str) -> None:
    x = arr(bin_rows, "fdd_mean")
    styles = build_method_style_map(labels)

    specs = [
        ("fp", "False-positive attenuation", "Concat FP − method FP", "curve_all_methods_fp_attenuation_panel"),
        ("fn", "False-negative attenuation", "Concat FN − method FN", "curve_all_methods_fn_attenuation_panel"),
    ]

    # Spacious overview figure.
    fig, axes = plt.subplots(1, 2, figsize=TWO_PANEL_FIGSIZE, constrained_layout=False)
    handles, legend_labels = [], []
    for ax, (metric, panel, ylabel, _) in zip(axes, specs):
        h, l = _plot_component_attenuation_panel(ax, x, bin_rows, labels, styles, metric, panel, ylabel)
        if metric == "fp":
            handles, legend_labels = h, l
    fig.suptitle(title, fontsize=9.4, y=0.985)
    fig.legend(handles, legend_labels, loc="lower center", bbox_to_anchor=(0.5, -0.035),
               ncol=min(len(handles), 6), frameon=False, handlelength=1.8, columnspacing=1.25)
    fig.subplots_adjust(left=0.080, right=0.990, top=0.78, bottom=0.29, wspace=0.58)
    save_curve_figure(fig, out_dir, "curve_all_methods_fp_fn_attenuation")
    plt.close(fig)

    # Standalone panels.
    for metric, panel, ylabel, filename in specs:
        fig, ax = plt.subplots(figsize=PANEL_FIGSIZE, constrained_layout=False)
        handles, legend_labels = _plot_component_attenuation_panel(ax, x, bin_rows, labels, styles, metric, panel, ylabel)
        _finalize_single_panel(fig, ax)
        save_panel_figure(fig, _panel_dir(out_dir), filename)
        plt.close(fig)
    _save_legend_only(handles, legend_labels, out_dir, "legend_attenuation_methods", ncol=min(len(handles), 6))


def _delta_from_lowest_fdd_bin(y: np.ndarray) -> np.ndarray:
    """Return y - y[first finite bin]. Keeps NaNs unchanged."""
    y = np.asarray(y, dtype=np.float64)
    out = y.copy()
    finite = np.isfinite(out)
    if not finite.any():
        return out
    baseline = out[np.flatnonzero(finite)[0]]
    return out - baseline


def _maxabs_normalize(y: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Normalize a delta curve by its max absolute finite amplitude for trend display."""
    y = np.asarray(y, dtype=np.float64)
    finite = np.isfinite(y)
    if not finite.any():
        return y
    denom = np.nanmax(np.abs(y[finite]))
    if denom < eps:
        return np.zeros_like(y, dtype=np.float64)
    return y / denom


def plot_response_curve(bin_rows: List[Dict[str, Any]], out_dir: str, title: str, family: str) -> None:
    """Plot TCFF response as level-specific relative changes across Concat-FDD bins.

    Besides the spacious overview figures, this function exports each level as a
    standalone 1.2:1 panel under ``out_dir/panels``. These single-panel outputs are
    the recommended assets for manual cropping and NeurIPS paper assembly.
    """
    if not bin_rows:
        return

    response_metrics = ["reversion", "weight", "fusion_delta"]
    available = [
        f"response_{name}_{lv}_mean"
        for lv in LEVELS
        for name in response_metrics
        if f"response_{name}_{lv}_mean" in bin_rows[0]
    ]
    if not available:
        return

    x = arr(bin_rows, "fdd_mean")
    level_titles = {
        "deep": "Deep / semantic context",
        "mid": "Mid / morphology",
        "shallow": "Shallow / boundary support",
    }
    label_weight = "Retained fusion weight" if family == "foundation" else "Retained enhanced-feature weight"
    metric_specs = [
        ("reversion", "Reversion to main feature", "o", "#D55E00"),
        ("weight", label_weight, "^", "#0072B2"),
        ("fusion_delta", "Fusion-induced feature shift", "s", "#009E73"),
    ]

    def _plot_response_level_panel(ax: plt.Axes, lv: str, normalize: bool, ylabel: str) -> Tuple[List[Any], List[str]]:
        handles, labels_out = [], []
        for name, label, marker, color in metric_specs:
            key = f"response_{name}_{lv}_mean"
            if key not in bin_rows[0]:
                continue
            y = _delta_from_lowest_fdd_bin(arr(bin_rows, key))
            if normalize:
                y = _maxabs_normalize(y)
            h, = ax.plot(
                x, y,
                marker=marker, markersize=4.8, linewidth=2.15,
                label=label, color=color, alpha=0.96, markeredgewidth=0.0,
            )
            handles.append(h)
            labels_out.append(label)
        ax.axhline(0, linestyle="--", linewidth=0.85, color="#555555", zorder=1)
        ax.set_title(level_titles.get(lv, lv))
        _format_fdd_axis(ax, x, force_left=0.30)
        ax.set_ylabel(ylabel)
        _style_axes(ax)
        return handles, labels_out

    def _draw(normalize: bool, filename: str, ylabel: str) -> None:
        # Spacious overview figure, with larger gaps than the earlier version.
        fig, axes = plt.subplots(1, 3, figsize=THREE_PANEL_FIGSIZE,
                                 constrained_layout=False, sharey=True)
        handles, labels_out = [], []
        for ax, lv in zip(axes, LEVELS):
            h, l = _plot_response_level_panel(ax, lv, normalize, ylabel)
            if lv == LEVELS[0]:
                handles, labels_out = h, l
            else:
                ax.set_ylabel("")
        fig.suptitle(title, fontsize=9.4, y=0.985)
        fig.legend(handles, labels_out, loc="lower center", bbox_to_anchor=(0.5, -0.040),
                   ncol=3, frameon=False, handlelength=1.8, columnspacing=1.35)
        fig.subplots_adjust(left=0.075, right=0.990, top=0.78, bottom=0.30, wspace=0.62)
        save_curve_figure(fig, out_dir, filename)
        plt.close(fig)

        # Standalone 1.2:1 panels for manual assembly.
        for lv in LEVELS:
            fig, ax = plt.subplots(figsize=PANEL_FIGSIZE, constrained_layout=False)
            handles, labels_out = _plot_response_level_panel(ax, lv, normalize, ylabel)
            _finalize_single_panel(fig, ax)
            suffix = "norm_" if normalize else ""
            save_panel_figure(fig, _panel_dir(out_dir), f"{filename}_{suffix}{lv}_panel")
            plt.close(fig)

        legend_name = "legend_response_metrics_norm" if normalize else "legend_response_metrics"
        _save_legend_only(handles, labels_out, out_dir, legend_name, ncol=3)

    _draw(
        normalize=False,
        filename="curve_tcff_response_delta_by_level",
        ylabel="Δ response from lowest-FDD bin",
    )
    _draw(
        normalize=True,
        filename="curve_tcff_response_delta_norm_by_level",
        ylabel="Normalized Δ response",
    )

def axis_to_int(plane: str) -> int:
    return {"z": 0, "y": 1, "x": 2}[plane]


def project_slab(vol: np.ndarray, axis: int, center: int, slab: int, mode: str = "max") -> np.ndarray:
    n = vol.shape[axis]
    lo, hi = max(0, center - slab), min(n, center + slab + 1)
    sl = [slice(None)] * 3
    sl[axis] = slice(lo, hi)
    sub = vol[tuple(sl)]
    if mode == "mean":
        return sub.mean(axis=axis)
    return sub.max(axis=axis)


def best_slice(gt: np.ndarray, pred_concat: np.ndarray, pred_method: np.ndarray,
               fdd: np.ndarray, axis: int) -> int:
    gt_fg = gt > 0
    pc = pred_concat > 0
    pm = pred_method > 0
    changed = (pc != pm) & (gt_fg | pc | pm)
    if changed.sum() == 0:
        cand = (gt_fg | pc | pm) & (fdd >= np.quantile(fdd[gt_fg | pc | pm], 0.75)) if (gt_fg | pc | pm).any() else gt_fg
    else:
        cand = changed
    if cand.sum() == 0:
        cand = gt_fg
    scores = cand.sum(axis=tuple(i for i in range(3) if i != axis))
    return int(np.argmax(scores))


def bbox2d(mask: np.ndarray, margin: int = 24, min_size: int = 96) -> Tuple[int, int, int, int]:
    H, W = mask.shape
    ys, xs = np.where(mask)
    if ys.size == 0:
        return 0, H, 0, W
    r0, r1 = max(0, int(ys.min()) - margin), min(H, int(ys.max()) + 1 + margin)
    c0, c1 = max(0, int(xs.min()) - margin), min(W, int(xs.max()) + 1 + margin)
    if r1 - r0 < min_size:
        mid = (r0 + r1) // 2
        r0 = max(0, mid - min_size // 2)
        r1 = min(H, r0 + min_size)
        r0 = max(0, r1 - min_size)
    if c1 - c0 < min_size:
        mid = (c0 + c1) // 2
        c0 = max(0, mid - min_size // 2)
        c1 = min(W, c0 + min_size)
        c0 = max(0, c1 - min_size)
    return r0, r1, c0, c1


def crop(x: np.ndarray, bb: Tuple[int, int, int, int]) -> np.ndarray:
    r0, r1, c0, c1 = bb
    return x[r0:r1, c0:c1]


def overlay_err(ax, img: np.ndarray, gt: np.ndarray, pred: np.ndarray, title: str) -> None:
    ax.imshow(img, cmap="gray", vmin=0, vmax=1)
    gt_fg, pr_fg = gt > 0, pred > 0
    rgba = np.zeros((*img.shape, 4), dtype=np.float32)
    rgba[gt_fg & pr_fg] = (0.15, 0.80, 0.25, 0.52)
    rgba[(~gt_fg) & pr_fg] = (1.00, 0.10, 0.10, 0.62)
    rgba[gt_fg & (~pr_fg)] = (0.10, 0.30, 1.00, 0.62)
    ax.imshow(rgba)
    if gt_fg.any():
        ax.contour(gt_fg.astype(float), levels=[0.5], colors=["white"], linewidths=0.7)
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def plot_case_method(case_name: str, image: np.ndarray, gt: np.ndarray,
                     pred_concat: np.ndarray, pred_method: np.ndarray,
                     fdd_map: np.ndarray, method_label: str, out_path: str,
                     plane: str, slab: int, no_crop: bool,
                     center: Optional[int] = None,
                     shared_bbox: Optional[Tuple[int, int, int, int]] = None,
                     save_pdf: bool = True,
                     dpi: int = 300) -> None:
    """Plot one method-vs-Concat panel for a fixed slice.

    The previous implementation chose ``best_slice`` independently for each method,
    which could make TCFF/BAGF/DSFF figures from the same case correspond to
    different slice positions. The optional ``center`` and ``shared_bbox`` arguments
    enforce strict slice/crop alignment across all methods.
    """
    axis = axis_to_int(plane)
    if center is None:
        center = best_slice(gt, pred_concat, pred_method, fdd_map, axis)

    img2 = normalize_image(project_slab(image, axis, center, slab, "max"))
    gt2 = project_slab(gt.astype(np.int16), axis, center, slab, "max")
    pc2 = project_slab(pred_concat.astype(np.int16), axis, center, slab, "max")
    pm2 = project_slab(pred_method.astype(np.int16), axis, center, slab, "max")
    fdd2 = project_slab(fdd_map.astype(np.float32), axis, center, slab, "max")
    changed = ((pc2 > 0) != (pm2 > 0)) | ((pc2 != pm2) & ((gt2 > 0) | (pc2 > 0) | (pm2 > 0)))
    mask = (gt2 > 0) | (pc2 > 0) | (pm2 > 0) | changed
    if shared_bbox is not None:
        bb = shared_bbox
    else:
        bb = (0, img2.shape[0], 0, img2.shape[1]) if no_crop else bbox2d(mask, margin=24, min_size=96)
    img, gt_c, pc_c, pm_c, fdd_c = [crop(v, bb) for v in [img2, gt2, pc2, pm2, fdd2]]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), constrained_layout=True)
    axes = axes.ravel()
    axes[0].imshow(img, cmap="gray", vmin=0, vmax=1)
    if (gt_c > 0).any():
        axes[0].contour((gt_c > 0).astype(float), levels=[0.5], colors=["lime"], linewidths=1.0)
    axes[0].set_title(f"(A) MRA + GT\n{case_name} | {plane}-slice {center}, slab ±{slab}", fontsize=10)
    axes[0].axis("off")

    overlay_err(axes[1], img, gt_c, pc_c, "(B) Concat error")
    overlay_err(axes[2], img, gt_c, pm_c, f"(C) {method_label} error")

    axes[3].imshow(img, cmap="gray", vmin=0, vmax=1)
    fdd_show = np.ma.masked_where(~((gt_c > 0) | (pc_c > 0) | (pm_c > 0)), fdd_c)
    hm = axes[3].imshow(fdd_show, cmap="magma", alpha=0.65, vmin=0, vmax=1)
    axes[3].set_title("(D) Concat-defined FDD")
    axes[3].axis("off")
    fig.colorbar(hm, ax=axes[3], fraction=0.046, pad=0.02)

    axes[4].imshow(img, cmap="gray", vmin=0, vmax=1)
    gt_fg, pc_fg, pm_fg = gt_c > 0, pc_c > 0, pm_c > 0
    rgba = np.zeros((*img.shape, 4), dtype=np.float32)
    rgba[(~pc_fg) & gt_fg & pm_fg] = (1.0, 0.85, 0.1, 0.75)   # recovered FN
    rgba[pc_fg & (~gt_fg) & (~pm_fg)] = (0.75, 0.1, 0.9, 0.75) # reduced FP
    rgba[(~pm_fg) & gt_fg] = (0.1, 0.3, 1.0, 0.62)             # residual FN
    rgba[pm_fg & (~gt_fg) & (~pc_fg)] = (1.0, 0.1, 0.1, 0.62)  # new FP
    axes[4].imshow(rgba)
    axes[4].set_title(f"(E) {method_label} changed regions vs Concat")
    axes[4].axis("off")

    metrics = ["Err", "FP", "FN", "Dice"]
    cand_c = (gt > 0) | (pred_concat > 0)
    cand_m = (gt > 0) | (pred_method > 0)
    def local_vals(pred, cand):
        denom = max(1, int(cand.sum()))
        return [
            float(((pred != gt) & cand).sum() / denom),
            float(((pred > 0) & (gt == 0) & cand).sum() / denom),
            float(((pred == 0) & (gt > 0) & cand).sum() / denom),
            compute_dice(pred > 0, gt > 0),
        ]
    x = np.arange(len(metrics))
    axes[5].bar(x - 0.18, local_vals(pred_concat, cand_c), width=0.36, label="Concat")
    axes[5].bar(x + 0.18, local_vals(pred_method, cand_m), width=0.36, label=method_label)
    axes[5].set_xticks(x)
    axes[5].set_xticklabels(metrics)
    axes[5].set_ylim(0, 1.05)
    axes[5].set_title("(F) Case-level statistics")
    axes[5].grid(axis="y", alpha=0.25)
    axes[5].legend(frameon=False)
    legend = [
        Patch(facecolor=(0.15, 0.80, 0.25, 0.52), label="TP"),
        Patch(facecolor=(1.00, 0.10, 0.10, 0.62), label="FP / New FP"),
        Patch(facecolor=(0.10, 0.30, 1.00, 0.62), label="FN / Residual FN"),
        Patch(facecolor=(1.0, 0.85, 0.1, 0.75), label="Recovered FN"),
        Patch(facecolor=(0.75, 0.1, 0.9, 0.75), label="Reduced FP"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=5, frameon=False, fontsize=9)
    ensure_dir(os.path.dirname(os.path.abspath(out_path)))
    fig.savefig(out_path + ".png", dpi=dpi)
    if save_pdf:
        fig.savefig(out_path + ".pdf", dpi=dpi)
    plt.close(fig)


def compute_shared_bbox_for_slice(gt: np.ndarray,
                                  preds: Dict[str, np.ndarray],
                                  fdd_map: np.ndarray,
                                  plane: str,
                                  center: int,
                                  slab: int,
                                  no_crop: bool) -> Tuple[int, int, int, int]:
    """Compute one crop box shared by all methods for the same case/plane/slice."""
    axis = axis_to_int(plane)
    gt2 = project_slab(gt.astype(np.int16), axis, center, slab, "max")
    fdd2 = project_slab(fdd_map.astype(np.float32), axis, center, slab, "max")
    mask = gt2 > 0
    pred2s = {}
    for label, pred in preds.items():
        p2 = project_slab(pred.astype(np.int16), axis, center, slab, "max")
        pred2s[label] = p2
        mask |= p2 > 0
    if "Concat" in pred2s:
        pc = pred2s["Concat"]
        for label, p2 in pred2s.items():
            if label == "Concat":
                continue
            mask |= ((pc > 0) != (p2 > 0)) | ((pc != p2) & ((gt2 > 0) | (pc > 0) | (p2 > 0)))
    if mask.sum() == 0 and np.isfinite(fdd2).any():
        # Fallback for empty slices: center a crop around high-FDD regions.
        q = np.quantile(fdd2[np.isfinite(fdd2)], 0.95)
        mask = fdd2 >= q
    if no_crop:
        return (0, gt2.shape[0], 0, gt2.shape[1])
    return bbox2d(mask, margin=24, min_size=96)


def slice_content_scores(gt: np.ndarray,
                         preds: Dict[str, np.ndarray],
                         plane: str) -> np.ndarray:
    """Count foreground / predicted foreground voxels per slice for a shared case-level mask."""
    axis = axis_to_int(plane)
    mask = gt > 0
    for pred in preds.values():
        mask |= pred > 0
    other_axes = tuple(i for i in range(3) if i != axis)
    return mask.sum(axis=other_axes)


def get_slice_indices_for_case(gt: np.ndarray,
                               preds: Dict[str, np.ndarray],
                               plane: str,
                               args: argparse.Namespace) -> List[int]:
    """Select slice indices for case-level visualization.

    - all: every slice along the plane, optionally with stride/range.
    - nonempty: all slices containing GT or any method prediction.
    - explicit: use --slice_indices.
    - auto_best: a small backward-compatible mode that chooses the highest-content slices.
    """
    axis = axis_to_int(plane)
    n = int(gt.shape[axis])

    if args.slice_mode == "explicit":
        base = int(args.slice_index_base)
        raw = args.slice_indices or []
        idxs = [int(i) - base for i in raw]
    elif args.slice_mode == "all":
        idxs = list(range(n))
    elif args.slice_mode == "nonempty":
        scores = slice_content_scores(gt, preds, plane)
        idxs = [int(i) for i in np.where(scores > 0)[0]]
    elif args.slice_mode == "auto_best":
        scores = slice_content_scores(gt, preds, plane)
        k = args.max_slices_per_plane if args.max_slices_per_plane > 0 else 12
        order = np.argsort(scores)[::-1]
        idxs = sorted([int(i) for i in order[:k] if scores[i] > 0])
    else:
        raise ValueError(f"Unknown slice_mode={args.slice_mode}")

    start = 0 if args.slice_start is None else max(0, int(args.slice_start) - int(args.slice_index_base))
    end = n - 1 if args.slice_end is None else min(n - 1, int(args.slice_end) - int(args.slice_index_base))
    stride = max(1, int(args.slice_stride))
    idxs = [i for i in idxs if 0 <= i < n and start <= i <= end and ((i - start) % stride == 0)]
    idxs = sorted(set(idxs))
    if args.max_slices_per_plane > 0 and args.slice_mode != "auto_best":
        idxs = idxs[:args.max_slices_per_plane]
    return idxs


def select_cases_for_visualization(dense_cache: Dict[str, Dict[str, Any]],
                                   case_rows: List[Dict[str, Any]],
                                   args: argparse.Namespace) -> List[str]:
    """Select cases by explicit index/name first; keep legacy top_k as fallback."""
    ordered = sorted(dense_cache.items(), key=lambda kv: int(kv[1].get("case_idx", 10**9)))
    idx_to_name = {int(item.get("case_idx", -1)): name for name, item in ordered}
    selected: List[str] = []

    if args.case_indices:
        base = int(args.case_index_base)
        for raw_idx in args.case_indices:
            idx = int(raw_idx) - base
            if idx in idx_to_name:
                selected.append(idx_to_name[idx])
            else:
                print(f"  [WARN] case index {raw_idx} (base={base}) not found; available 0..{max(idx_to_name.keys()) if idx_to_name else -1}")

    if args.case_names:
        for name in args.case_names:
            if name in dense_cache:
                selected.append(name)
            else:
                print(f"  [WARN] case name not found in dense cache: {name}")

    # Backward-compatible automatic selection. Kept only as fallback.
    if not selected and args.top_k_cases > 0:
        selected = select_cases_for_plot(case_rows, args.case_names, args.top_k_cases, args.select_cases_by)

    # New behavior: if user asked to save figures without specifying cases, save all cached cases.
    if not selected:
        selected = [name for name, _ in ordered]

    # stable de-duplication
    seen = set()
    out = []
    for name in selected:
        if name not in seen:
            out.append(name)
            seen.add(name)
    return out


def save_case_slice_figures(item: Dict[str, Any],
                            labels: List[str],
                            out_dir: str,
                            args: argparse.Namespace) -> List[Dict[str, Any]]:
    """Save fixed-slice, fixed-crop figures for one selected case."""
    case_name = str(item.get("case_name", "case"))
    case_idx = int(item.get("case_idx", -1))
    image = item["image"]
    gt = item["gt"]
    preds = item["preds"]
    fdd_map = item["maps"]["fdd_all_max"]
    rows: List[Dict[str, Any]] = []

    planes = args.vis_planes or ["z", "y", "x"]
    for plane in planes:
        slice_indices = get_slice_indices_for_case(gt, preds, plane, args)
        print(f"  [Case figures] case_idx={case_idx} case={case_name} plane={plane} slices={len(slice_indices)}")
        plane_dir = ensure_dir(os.path.join(out_dir, "figures", plane))
        for center in slice_indices:
            shared_bbox = compute_shared_bbox_for_slice(gt, preds, fdd_map, plane, center, args.slab, args.no_crop)
            for label in labels:
                if label == "Concat":
                    continue
                out_prefix = os.path.join(
                    plane_dir,
                    f"{sanitize_label(case_name)}_{plane}{center:03d}_{sanitize_label(label)}_vs_Concat"
                )
                plot_case_method(
                    case_name, image, gt, preds["Concat"], preds[label], fdd_map, label,
                    out_prefix, plane, args.slab, args.no_crop,
                    center=center, shared_bbox=shared_bbox,
                    save_pdf=args.save_pdf_figures, dpi=args.figure_dpi
                )
                rows.append({
                    "case_idx": case_idx,
                    "case": case_name,
                    "plane": plane,
                    "slice": center,
                    "method": label,
                    "path_png": out_prefix + ".png",
                    "shared_bbox": str(shared_bbox),
                })
    return rows

# -----------------------------------------------------------------------------
# Main analysis
# -----------------------------------------------------------------------------

def prepare_loader(test_dir: str, num_workers: int) -> DataLoader:
    from topcow_dataset import TopCoWTestDataset
    ds = TopCoWTestDataset(test_dir, split="test")
    return DataLoader(ds, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True)


def case_selected_by_args(case_idx: int, case_name: str, args: argparse.Namespace) -> bool:
    """Whether a case is explicitly requested for visualization-only mode.

    If neither indices nor names are specified, all cases are considered selected.
    """
    has_index_filter = bool(args.case_indices)
    has_name_filter = bool(args.case_names)
    if not has_index_filter and not has_name_filter:
        return True

    matched = False
    if has_index_filter:
        base = int(args.case_index_base)
        requested = {int(i) - base for i in args.case_indices}
        matched = matched or (int(case_idx) in requested)
    if has_name_filter:
        requested_names = set(str(x) for x in args.case_names)
        matched = matched or (str(case_name) in requested_names)
    return matched


def _npz_scalar_to_str(x: Any, default: str = "") -> str:
    try:
        arr = np.asarray(x)
        if arr.shape == ():
            return str(arr.item())
        if arr.size > 0:
            return str(arr.reshape(-1)[0])
    except Exception:
        pass
    return default


def load_dense_cache_from_npz(args: argparse.Namespace) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    """Load cached dense predictions/maps for visualization without model inference.

    Expected NPZ keys are generated by this script's --save_npz mode:
      image, gt, fdd_all_max, pred_Concat, pred_TCFF, ...
    This enables later figure regeneration for specific cases/slices without
    rerunning model forward or global FDD bin aggregation.
    """
    npz_dir = args.npz_dir or os.path.join(args.output_dir, "case_npz")
    if not os.path.isdir(npz_dir):
        raise FileNotFoundError(f"NPZ directory not found: {npz_dir}. Run once with --save_npz or set --npz_dir.")

    files = sorted([f for f in os.listdir(npz_dir) if f.endswith(".npz")])
    if not files:
        raise RuntimeError(f"No .npz files found in {npz_dir}")

    dense_cache: Dict[str, Dict[str, Any]] = {}
    labels_seen: List[str] = []

    for idx, fn in enumerate(files):
        path = os.path.join(npz_dir, fn)
        data = np.load(path, allow_pickle=True)
        case_name = _npz_scalar_to_str(data["case_name"], os.path.splitext(fn)[0]) if "case_name" in data.files else os.path.splitext(fn)[0]
        case_idx = int(np.asarray(data["case_idx"]).item()) if "case_idx" in data.files else idx

        if not case_selected_by_args(case_idx, case_name, args):
            continue

        preds: Dict[str, np.ndarray] = {}
        maps: Dict[str, np.ndarray] = {}
        for k in data.files:
            if k.startswith("pred_"):
                label = k[len("pred_"):]
                preds[label] = data[k].astype(np.uint8)
                if label not in labels_seen:
                    labels_seen.append(label)
            elif k.startswith("fdd_") or k.startswith("response_"):
                maps[k] = data[k].astype(np.float32)

        if "Concat" not in preds:
            raise RuntimeError(f"{path} does not contain pred_Concat; cannot plot method-vs-Concat figures.")
        if "fdd_all_max" not in maps:
            maps["fdd_all_max"] = np.zeros_like(data["gt"], dtype=np.float32)
            print(f"  [WARN] {fn} has no fdd_all_max; using zero FDD map for visualization.")

        dense_cache[case_name] = {
            "case_idx": case_idx,
            "case_name": case_name,
            "image": data["image"].astype(np.float32),
            "gt": data["gt"].astype(np.uint8),
            "preds": preds,
            "maps": maps,
        }

    labels = ["Concat"] + [l for l in labels_seen if l != "Concat"]
    return dense_cache, labels


def run_visualization_from_npz(args: argparse.Namespace) -> None:
    """Render case figures from cached NPZ files only.

    This mode is intended for the second pass of figure selection: it skips model
    loading, prediction, voxel sampling, FDD bin aggregation, and curve plotting.
    """
    out_dir = ensure_dir(args.output_dir)
    ensure_dir(os.path.join(out_dir, "figures"))
    dense_cache, labels = load_dense_cache_from_npz(args)

    write_json({
        "args": vars(args),
        "labels": labels,
        "mode": "vis_from_npz",
    }, os.path.join(out_dir, "analysis_config.json"))

    case_manifest = []
    for name, item in sorted(dense_cache.items(), key=lambda kv: int(kv[1].get("case_idx", 10**9))):
        gt_shape = tuple(int(s) for s in item["gt"].shape)
        case_manifest.append({
            "case_idx0": int(item.get("case_idx", -1)),
            "case_idx1": int(item.get("case_idx", -1)) + 1 if int(item.get("case_idx", -1)) >= 0 else "",
            "case": name,
            "shape": str(gt_shape),
        })
    write_csv(case_manifest, os.path.join(out_dir, "case_index_manifest.csv"))

    selected = select_cases_for_visualization(dense_cache, [], args)
    print(f"[NPZ visualization] npz_dir={args.npz_dir or os.path.join(args.output_dir, 'case_npz')}")
    print(f"[Case figures] selected={selected}")
    figure_rows: List[Dict[str, Any]] = []
    for case_name in selected:
        if case_name not in dense_cache:
            print(f"  [WARN] selected case not cached: {case_name}")
            continue
        figure_rows.extend(save_case_slice_figures(dense_cache[case_name], labels, out_dir, args))
    write_csv(figure_rows, os.path.join(out_dir, "figure_manifest.csv"))

    summary = {
        "mode": "vis_from_npz",
        "n_cases_cached": len(dense_cache),
        "n_figures": len(figure_rows),
        "labels": labels,
    }
    write_json(summary, os.path.join(out_dir, "summary.json"))
    print("\n[Done]")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Outputs saved to: {out_dir}")


def choose_response_model(args: argparse.Namespace, models: Dict[str, torch.nn.Module],
                          method_types: Dict[str, str], labels: List[str]) -> Optional[Tuple[str, torch.nn.Module]]:
    # Prefer explicitly requested response label; otherwise first tgf among methods.
    if args.response_method_label:
        if args.response_method_label not in models:
            raise ValueError(f"--response_method_label={args.response_method_label} not found in loaded methods")
        return args.response_method_label, models[args.response_method_label]
    for label in labels:
        if method_types.get(label) == "tgf" and label != "Concat":
            return label, models[label]
    return None



def enforce_curve_only_mode(args: argparse.Namespace) -> None:
    """Disable case-slice image export; keep line-curve generation as the core output."""
    if getattr(args, "save_case_figures", False):
        print("[Curve-only] --save_case_figures is ignored: case-slice image export is disabled in this version.")
        args.save_case_figures = False
    if getattr(args, "save_pdf_figures", False):
        print("[Curve-only] --save_pdf_figures is ignored because case-slice image export is disabled.")
        args.save_pdf_figures = False


def run(args: argparse.Namespace) -> None:
    # Case-slice image saving is disabled; the retained visual output is line curves.
    enforce_curve_only_mode(args)
    if args.vis_from_npz:
        print("[Curve-only] --vis_from_npz is disabled because it only renders case-slice images. Run full analysis to generate curves.")
        return

    setup_imports(args)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    out_dir = ensure_dir(args.output_dir)
    if args.save_case_figures:
        ensure_dir(os.path.join(out_dir, "figures"))
    device = torch.device(args.device if args.device else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    print(f"[Device] {device}")

    concat_model, models, labels, method_types = load_all_models(args, device)
    response_pair = choose_response_model(args, models, method_types, labels)
    if response_pair:
        print(f"[Response probe] {response_pair[0]} ({method_types[response_pair[0]]})")

    loader = prepare_loader(args.test_dir, args.num_workers)
    voxel_records: List[Dict[str, np.ndarray]] = []
    case_rows: List[Dict[str, Any]] = []
    case_manifest: List[Dict[str, Any]] = []
    dense_cache: Dict[str, Dict[str, Any]] = {}

    if args.visualize_only and not (args.save_case_figures or args.save_npz):
        print("[WARN] --visualize_only is set, but neither --save_case_figures nor --save_npz is enabled.")

    with torch.no_grad():
        for case_idx, batch in enumerate(loader):
            if "multiclass" not in batch:
                print("[Skip] batch has no multiclass")
                continue
            case_name = get_case_name(batch)
            image_shape = tuple(int(s) for s in batch["multiclass"][0].shape)
            case_manifest.append({"case_idx0": int(case_idx), "case_idx1": int(case_idx + 1),
                                  "case": case_name, "shape": str(image_shape)})

            if args.visualize_only and not case_selected_by_args(case_idx, case_name, args):
                continue

            image_t = batch["image"].float().to(device, non_blocking=True)
            image_np = batch["image"][0, 0].detach().cpu().numpy().astype(np.float32)
            gt = batch["multiclass"][0].detach().cpu().numpy().astype(np.uint8)
            gt_shape = tuple(int(s) for s in gt.shape)
            print(f"[{case_idx+1:03d}/{len(loader):03d}] {case_name} shape={gt_shape}")

            if args.family == "dual":
                anchor = probe_dual_concat_anchor(concat_model, image_t, gt_shape)
            else:
                anchor = probe_foundation_concat_anchor(concat_model, image_t, gt_shape)

            preds: Dict[str, np.ndarray] = {"Concat": anchor["pred"]}
            for label in labels:
                if label == "Concat":
                    continue
                preds[label] = predict_model(models[label], image_t)

            candidate = make_candidate_mask(gt, preds, dilate_iter=args.candidate_dilate)
            maps = finalize_fdd_maps(anchor["maps"], candidate)

            if response_pair and (not args.visualize_only or args.save_npz):
                resp_label, resp_model = response_pair
                try:
                    if args.family == "foundation" and method_types[resp_label] == "tgf":
                        maps.update(probe_foundation_tgf_response(resp_model, image_t, gt_shape))
                    elif args.family == "dual" and method_types[resp_label] == "tgf":
                        maps.update(probe_dual_tgf_response(resp_model, image_t, gt_shape))
                except Exception as e:
                    print(f"  [WARN] response probe failed for {resp_label}: {e}")

            case_rows.extend(collect_case_rows(case_name, gt, preds, maps, labels))

            if not args.visualize_only:
                rec = sample_case_voxels(case_idx, gt, preds, maps, candidate, labels,
                                         max_voxels=args.max_voxels_per_case, seed=args.seed)
                if rec is not None:
                    voxel_records.append(rec)

            # Cache dense maps for case-level figure export / NPZ export.
            need_cache = args.save_case_figures or args.save_npz
            if need_cache:
                # In visualize-only mode we cache only explicitly selected cases (or all if no filter is given).
                if (not args.visualize_only) or case_selected_by_args(case_idx, case_name, args):
                    dense_cache[case_name] = {
                        "case_idx": int(case_idx), "case_name": case_name,
                        "image": image_np, "gt": gt, "preds": preds, "maps": maps
                    }

    write_csv(case_manifest, os.path.join(out_dir, "case_index_manifest.csv"))
    write_json({"args": vars(args), "labels": labels, "method_types": method_types}, os.path.join(out_dir, "analysis_config.json"))

    if args.visualize_only:
        summary = {
            "mode": "visualize_only",
            "n_cases_cached": len(dense_cache),
            "labels": labels,
        }
    else:
        if not voxel_records:
            raise RuntimeError("No voxel records collected. Check test_dir and multiclass GT.")

        voxel = concat_records(voxel_records)
        bin_rows = aggregate_bins(voxel, labels, args.n_bins, args.min_case_voxels)

        write_csv(bin_rows, os.path.join(out_dir, "bin_stats_all_methods_concat_anchor.csv"))
        write_csv(case_rows, os.path.join(out_dir, "case_records_all_methods.csv"))

        title = args.experiment_label or f"{args.family}: all methods vs Concat"
        plot_all_error_curves(bin_rows, labels, out_dir, title)
        plot_fp_fn_attenuation(bin_rows, labels, out_dir, title)
        if response_pair:
            plot_response_curve(bin_rows, out_dir, f"{title} | response={response_pair[0]}", args.family)

        # Summary
        concat_s = sanitize_label("Concat")
        summary = {
            "mode": "full_analysis",
            "n_cases": len(set(r["case"] for r in case_rows)),
            "n_voxels_sampled": int(len(voxel["fdd"])),
            "labels": labels,
            "corr_concat_fdd_error": safe_corr(voxel["fdd"], voxel[f"err_{concat_s}"]),
        }
        for label in labels:
            s = sanitize_label(label)
            method_rows = [r for r in case_rows if r["method"] == label]
            summary[f"mean_dice_{s}"] = float(np.mean([r["dice"] for r in method_rows])) if method_rows else float("nan")
            summary[f"mean_local_error_{s}"] = float(np.mean([r["local_error"] for r in method_rows])) if method_rows else float("nan")
            if label != "Concat":
                summary[f"corr_fdd_attenuation_{s}"] = safe_corr(voxel["fdd"], voxel[f"err_{concat_s}"] - voxel[f"err_{s}"])
    write_json(summary, os.path.join(out_dir, "summary.json"))

    if args.save_case_figures:
        selected = select_cases_for_visualization(dense_cache, case_rows, args)
        print(f"[Case figures] selected={selected}")
        figure_rows: List[Dict[str, Any]] = []
        for case_name in selected:
            if case_name not in dense_cache:
                print(f"  [WARN] selected case not cached: {case_name}")
                continue
            figure_rows.extend(save_case_slice_figures(dense_cache[case_name], labels, out_dir, args))
        write_csv(figure_rows, os.path.join(out_dir, "figure_manifest.csv"))

    if args.save_npz:
        map_dir = ensure_dir(os.path.join(out_dir, "case_npz"))
        for case_name, item in dense_cache.items():
            payload = {
                "case_idx": np.asarray(item.get("case_idx", -1), dtype=np.int32),
                "case_name": np.asarray(str(case_name)),
                "image": item["image"].astype(np.float32),
                "gt": item["gt"].astype(np.uint8),
            }
            for k, v in item["maps"].items():
                if isinstance(v, np.ndarray) and v.shape == item["gt"].shape:
                    payload[k] = v.astype(np.float32)
            for label, pred in item["preds"].items():
                payload[f"pred_{sanitize_label(label)}"] = pred.astype(np.uint8)
            np.savez_compressed(os.path.join(map_dir, f"{case_name}.npz"), **payload)

    print("\n[Done]")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Outputs saved to: {out_dir}")


def select_cases_for_plot(case_rows: List[Dict[str, Any]], explicit: Optional[List[str]],
                          top_k: int, select_by: str) -> List[str]:
    if explicit:
        return explicit
    if top_k <= 0:
        return []
    # Use best improvement over Concat among all non-Concat methods as a visual score.
    by_case: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for r in case_rows:
        by_case.setdefault(r["case"], {})[r["method"]] = r
    scored = []
    for case, rows in by_case.items():
        if "Concat" not in rows:
            continue
        base = rows["Concat"]
        best = -1e9
        for m, r in rows.items():
            if m == "Concat":
                continue
            if select_by == "dice_gain":
                score = r["dice"] - base["dice"]
            elif select_by == "error_reduction":
                score = base["local_error"] - r["local_error"]
            elif select_by == "bad_method":
                score = r["local_error"] - base["local_error"]
            else:
                score = (base["local_error"] - r["local_error"]) + 0.5 * (r["dice"] - base["dice"])
            best = max(best, score)
        scored.append((best, case))
    scored.sort(reverse=True)
    return [c for _, c in scored[:top_k]]


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--family", choices=["dual", "foundation"], required=True)
    p.add_argument("--test_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--experiment_label", type=str, default="")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--code_dir", type=str, default=None)
    p.add_argument("--sam_med3d_path", type=str, default=None)
    p.add_argument("--strict_load", action="store_true")

    p.add_argument("--concat_ckpt", type=str, required=True,
                   help="Concat checkpoint. This model defines both the baseline prediction and common FDD axis.")
    p.add_argument("--method_specs", nargs="*", default=[],
                   help="List of Label=type:checkpoint. type can be tgf/concat/crossattn/bagf/pug/dsff/tgf_nogate/unet for dual; tgf/concat/crossattn/bagf/pug/dsff for foundation.")

    # Shared foundation spec for all foundation models. If config.json near ckpt is complete, these can be omitted.
    p.add_argument("--foundation_type", type=str, default=None)
    p.add_argument("--foundation_ckpt", type=str, default=None)

    p.add_argument("--max_voxels_per_case", type=int, default=80000)
    p.add_argument("--n_bins", type=int, default=10)
    p.add_argument("--min_case_voxels", type=int, default=20)
    p.add_argument("--candidate_dilate", type=int, default=2)

    p.add_argument("--response_method_label", type=str, default=None,
                   help="Which tgf method to probe for response curves. Default: first loaded non-Concat method with type=tgf.")

    p.add_argument("--save_case_figures", action="store_true",
                   help="Deprecated and ignored in this curve-only version. Case-slice image export is disabled.")
    p.add_argument("--save_npz", action="store_true")
    p.add_argument("--visualize_only", "--vis_only", dest="visualize_only", action="store_true",
                   help="Only perform case-level visualization / NPZ export. Skip global FDD voxel aggregation and curve plotting, but still runs model inference to obtain predictions and FDD maps.")
    p.add_argument("--vis_from_npz", action="store_true",
                   help="Deprecated and ignored in this curve-only version because it only renders case-slice images.")
    p.add_argument("--npz_dir", type=str, default=None,
                   help="Source directory for --vis_from_npz. Default: output_dir/case_npz.")

    # Case-level dense visualization. Explicit case indices/names are preferred.
    # If neither is given and --top_k_cases=0, all cached test cases are exported.
    p.add_argument("--case_indices", nargs="*", type=int, default=None,
                   help="Specific test case indices to visualize. Default base is 0; set --case_index_base 1 for 1-based indices.")
    p.add_argument("--case_index_base", type=int, choices=[0, 1], default=0,
                   help="Interpret --case_indices as 0-based or 1-based.")
    p.add_argument("--case_names", nargs="*", default=None,
                   help="Specific case names to visualize. Can be combined with --case_indices.")
    p.add_argument("--top_k_cases", type=int, default=0,
                   help="Deprecated fallback: automatically select top-k cases if no explicit case is given.")
    p.add_argument("--select_cases_by", choices=["motivation", "dice_gain", "error_reduction", "bad_method"], default="motivation")

    p.add_argument("--vis_planes", nargs="+", choices=["z", "y", "x"], default=["z", "y", "x"],
                   help="Planes to export for selected cases. Figures are saved directly into output_dir/figures/{x|y|z}/ and distinguished by filename.")
    p.add_argument("--slice_mode", choices=["nonempty", "all", "explicit", "auto_best"], default="nonempty",
                   help="nonempty saves all slices with GT or any prediction; all saves every slice; explicit uses --slice_indices.")
    p.add_argument("--slice_indices", nargs="*", type=int, default=None,
                   help="Slice indices for --slice_mode explicit. Uses --slice_index_base.")
    p.add_argument("--slice_index_base", type=int, choices=[0, 1], default=0,
                   help="Interpret --slice_indices/--slice_start/--slice_end as 0-based or 1-based.")
    p.add_argument("--slice_start", type=int, default=None,
                   help="Optional inclusive slice start. Uses --slice_index_base.")
    p.add_argument("--slice_end", type=int, default=None,
                   help="Optional inclusive slice end. Uses --slice_index_base.")
    p.add_argument("--slice_stride", type=int, default=1)
    p.add_argument("--max_slices_per_plane", type=int, default=0,
                   help="Optional cap per plane. 0 means no cap except auto_best default.")

    # Backward-compatible single-plane option is kept but no longer used by the new dense exporter.
    p.add_argument("--plane", choices=["z", "y", "x"], default="z")
    p.add_argument("--slab", type=int, default=3)
    p.add_argument("--no_crop", action="store_true")
    p.add_argument("--figure_dpi", type=int, default=220)
    p.add_argument("--save_pdf_figures", action="store_true",
                   help="Also save PDF for every case slice figure. Disabled by default to avoid huge output.")
    return p


if __name__ == "__main__":
    run(build_argparser().parse_args())

    """"
      --vis_only \只做个体可视化，不做全局 FDD bin 统计和曲线分析。但它仍然会加载模型并推理，因为需要得到 Concat、TCFF、BAGF、DSFF 的预测结果和 FDD map。
python tcff_fdd_final_v7_single_panel_neurips.py \
  --family foundation \
  --test_dir ./ixi_processed \
  --output_dir ./analysis_fdd_foundation_all_ixi_vis3 \
  --foundation_type sam3d_turbo \
  --foundation_ckpt ./sam_med3d_turbo.pth \
  --concat_ckpt ./Runs_foundation/sam_concat/best_model.pth \
  --method_specs \
      TCFF=tgf:./Runs_foundation_v2/sam_tgf_v3_s0/best_model.pth \
      DSFF=dsff:./Runs_foundation/sam_dsff/best_model.pth \
      BAGF=bagf:./Runs_foundation/sam_bagf/best_model.pth \
      CrossAttn=crossattn:./Runs_foundation_v2/sam_crossattn/best_model.pth \
      AFF=aff:./Runs_foundation_v2/sam_aff/best_model.pth \
  --save_case_figures \
  --case_indices 1 \
  --case_index_base 1 \
  --vis_planes x y z \
  --slice_mode nonempty \
  --slice_stride 1 \
  --slab 3
  
python tcff_fdd_final_v4_vis_only_npz.py \
  --concat_ckpt ./Runs_foundation/sam_concat/best_model.pth \
  --family foundation \
  --test_dir ./ixi_processed \
  --output_dir ./analysis_fdd_foundation_all_ixi_vis \
  --vis_from_npz \
  --npz_dir ./analysis_fdd_foundation_all_ixi_vis/case_npz \
  --save_case_figures \
  --case_indices 1 \
  --case_index_base 1 \
  --vis_planes x y z \
  --slice_mode nonempty \
  --slice_stride 1 \
  --slab 3
  
    
    """
