"""Audited TCFF latency, memory, parameter, and FLOP/MAC benchmark."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
import torch

from tcff_rebuttal_runner import load_foundation_model, sha256_file, write_json
from topcow_dataset import TopCoWTestDataset
from topcow_foundation_model import TRUST_VARIANTS


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temp, index=False)
    temp.replace(path)


def benchmark(function, warmup: int, repeats: int, device: torch.device):
    with torch.inference_mode():
        for _ in range(warmup):
            function()
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        baseline_memory = torch.cuda.memory_allocated(device) / (1024 ** 2)
        samples = []
        for repeat in range(repeats):
            started = torch.cuda.Event(enable_timing=True)
            finished = torch.cuda.Event(enable_timing=True)
            started.record()
            function()
            finished.record()
            finished.synchronize()
            samples.append({
                "repeat": repeat,
                "latency_ms": float(started.elapsed_time(finished)),
            })
        peak_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    return samples, baseline_memory, peak_memory


def capture_fusion_inputs(model, x):
    captured = {}
    handles = []
    for level, module in model._tgf_levels().items():
        def hook(_, inputs, level_name=level):
            captured[level_name] = tuple(value.detach() for value in inputs[:2])
        handles.append(module.register_forward_pre_hook(hook))
    with torch.inference_mode():
        model(x)
    for handle in handles:
        handle.remove()
    missing = set(model._tgf_levels()) - set(captured)
    if missing:
        raise RuntimeError(f"Fusion hooks did not capture: {sorted(missing)}")
    return captured


def summarize_samples(samples):
    values = np.asarray([row["latency_ms"] for row in samples], dtype=float)
    return {
        "latency_mean_ms": float(values.mean()),
        "latency_sd_ms": float(values.std(ddof=1)) if len(values) > 1 else np.nan,
        "latency_median_ms": float(np.median(values)),
        "latency_p25_ms": float(np.quantile(values, 0.25)),
        "latency_p75_ms": float(np.quantile(values, 0.75)),
        "latency_p95_ms": float(np.quantile(values, 0.95)),
        "repeats": int(len(values)),
    }


def try_thop(model, x, captured):
    result = {
        "end_to_end_macs": np.nan, "end_to_end_flops": np.nan,
        "fusion_only_macs": np.nan, "fusion_only_flops": np.nan,
        "flop_profiler": "unavailable",
        "flop_profile_error": "",
    }
    try:
        from thop import profile
        with torch.inference_mode():
            macs, _ = profile(model, inputs=(x,), verbose=False)
        fusion_macs = 0.0
        for level, module in model._tgf_levels().items():
            with torch.inference_mode():
                level_macs, _ = profile(module, inputs=captured[level], verbose=False)
            fusion_macs += float(level_macs)
        result.update({
            "end_to_end_macs": float(macs),
            "end_to_end_flops": float(2.0 * macs),
            "fusion_only_macs": float(fusion_macs),
            "fusion_only_flops": float(2.0 * fusion_macs),
            "flop_profiler": "thop_0.1.1",
        })
    except Exception as exc:
        result["flop_profile_error"] = f"{type(exc).__name__}: {exc}"
    return result


def main(args):
    project_root = Path(args.project_root).resolve()
    results_root = Path(args.results_root).resolve()
    exp_dir = Path(args.exp_dir).resolve()
    dataset_dir = Path(args.dataset_dir).resolve()
    device = torch.device(args.device)
    dataset = TopCoWTestDataset(str(dataset_dir), split="test")
    sample = dataset[args.case_index]
    x = sample["image"].unsqueeze(0).float().to(device)
    model, config, checkpoint = load_foundation_model(
        exp_dir, device, trust_variant="full")
    model.eval()
    total_params = int(sum(parameter.numel() for parameter in model.parameters()))
    trainable_params = int(sum(parameter.numel() for parameter in model.parameters()
                               if parameter.requires_grad))
    fusion_params = int(sum(parameter.numel() for module in model._tgf_levels().values()
                            for parameter in module.parameters()))
    fusion_trainable_params = int(sum(
        parameter.numel() for module in model._tgf_levels().values()
        for parameter in module.parameters() if parameter.requires_grad))

    raw_rows = []
    summary_rows = []
    flop_result = None
    variants = [value.strip() for value in args.variants.split(",") if value.strip()]
    for variant in variants:
        if variant not in TRUST_VARIANTS:
            raise ValueError(f"Unknown variant: {variant}")
        model.set_rebuttal_runtime(
            trust_variant=variant,
            corruption={"type": "none", "severity": 0.0, "seed": 0},
            instrumentation=False)
        end_samples, end_baseline, end_peak = benchmark(
            lambda: model(x), args.warmup, args.repeats, device)
        for row in end_samples:
            raw_rows.append({"model": "tcff_seed42", "variant": variant,
                             "scope": "end_to_end", **row})

        captured = capture_fusion_inputs(model, x)
        level_order = [level for level in ["shallow", "mid", "deep"]
                       if level in captured]
        def fusion_forward():
            for level in level_order:
                model._tgf_levels()[level](*captured[level])
        fusion_samples, fusion_baseline, fusion_peak = benchmark(
            fusion_forward, args.warmup, args.repeats, device)
        for row in fusion_samples:
            raw_rows.append({"model": "tcff_seed42", "variant": variant,
                             "scope": "fusion_only_three_levels", **row})

        if flop_result is None:
            flop_result = try_thop(model, x, captured)
        summary_rows.append({
            "model": "tcff_seed42", "variant": variant,
            "device": torch.cuda.get_device_name(device),
            "batch_size": int(x.shape[0]),
            "input_shape": "x".join(map(str, x.shape)),
            "precision": str(x.dtype).replace("torch.", ""),
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "fusion_parameters": fusion_params,
            "fusion_trainable_parameters": fusion_trainable_params,
            "forward_count": 1,
            "auxiliary_encoder_included_end_to_end": True,
            **{"end_to_end_" + key: value
               for key, value in summarize_samples(end_samples).items()},
            "end_to_end_baseline_allocated_mb": end_baseline,
            "end_to_end_peak_allocated_mb": end_peak,
            "end_to_end_incremental_peak_mb": end_peak - end_baseline,
            **{"fusion_only_" + key: value
               for key, value in summarize_samples(fusion_samples).items()},
            "fusion_only_baseline_allocated_mb": fusion_baseline,
            "fusion_only_peak_allocated_mb": fusion_peak,
            "fusion_only_incremental_peak_mb": fusion_peak - fusion_baseline,
            **(flop_result or {}),
        })

    raw = pd.DataFrame(raw_rows)
    summary = pd.DataFrame(summary_rows)
    atomic_csv(results_root / "09_efficiency" / "efficiency_raw_repeats.csv", raw)
    atomic_csv(results_root / "09_efficiency" / "efficiency_summary.csv", summary)
    atomic_csv(results_root / "06_summary_tables" / "Table_R2_efficiency.csv", summary)
    manifest = {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "checkpoint": str(checkpoint),
        "checkpoint_hash": sha256_file(checkpoint),
        "model_config": config,
        "dataset": str(dataset_dir),
        "case_id": sample["case_name"],
        "input_shape": list(x.shape),
        "warmup": args.warmup,
        "repeats": args.repeats,
        "variants": variants,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "python": platform.python_version(),
        "flop_convention": "FLOPs=2*MACs",
        "fusion_only_definition": "sum of the three TGF module forwards on cached real inputs",
        "end_to_end_definition": "one complete model forward including frozen foundation encoder",
        "environment": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        },
    }
    write_json(results_root / "09_efficiency" / "efficiency_manifest.json", manifest)
    print(json.dumps({
        "status": "success", "rows": len(summary),
        "raw_repeats": len(raw), "device": manifest["gpu_name"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", default=str(here))
    parser.add_argument("--results_root", default=str(here / "results_tcff_rebuttal"))
    parser.add_argument("--exp_dir", default=str(here / "Runs_foundation" / "sam_tgf_v3"))
    parser.add_argument("--dataset_dir", default=str(here / "crown_processed"))
    parser.add_argument("--case_index", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--variants", default="full,residual,u_only,g_only,no_aux,additive")
    raise SystemExit(main(parser.parse_args()))
