"""Isolated, cross-method TCFF efficiency benchmark for the V2 rebuttal."""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import json
import os
from pathlib import Path
import platform
import subprocess

import numpy as np
import pandas as pd
import torch

from tcff_rebuttal_runner import load_foundation_model, sha256_file, write_json
from topcow_dataset import TopCoWTestDataset


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def compute_process_snapshot():
    command = [
        "nvidia-smi",
        "--query-compute-apps=pid,gpu_uuid,used_gpu_memory",
        "--format=csv,noheader,nounits",
    ]
    process = subprocess.run(
        command, capture_output=True, text=True, check=False)
    if process.returncode != 0:
        raise RuntimeError(
            "Unable to audit GPU process isolation: " + process.stderr.strip())
    rows = []
    for line in process.stdout.splitlines():
        values = [value.strip() for value in line.split(",")]
        if len(values) != 3 or not values[0]:
            continue
        rows.append({
            "pid": int(values[0]),
            "gpu_uuid": values[1],
            "used_gpu_memory_mb": (
                None if values[2] in {"N/A", "[N/A]"} else int(values[2])),
        })
    return rows


def require_isolated_gpu_host(snapshot):
    foreign = [row for row in snapshot if row["pid"] != os.getpid()]
    if foreign:
        raise RuntimeError(
            "Formal efficiency timing requires no other GPU compute process; "
            f"observed foreign processes: {foreign}")


def benchmark(function, warmup: int, repeats: int, device: torch.device):
    # Release Python garbage and unused allocator blocks before each scope.  The
    # end-to-end benchmark is called before fusion inputs are captured, so its
    # baseline contains only the model and the real input tensor.
    gc.collect()
    torch.cuda.empty_cache()
    with torch.inference_mode():
        for _ in range(warmup):
            function()
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        baseline = torch.cuda.memory_allocated(device) / (1024 ** 2)
        values = []
        for repeat in range(repeats):
            started = torch.cuda.Event(enable_timing=True)
            finished = torch.cuda.Event(enable_timing=True)
            started.record()
            function()
            finished.record()
            finished.synchronize()
            values.append({
                "repeat": repeat,
                "latency_ms": float(started.elapsed_time(finished)),
            })
        peak = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    return values, baseline, peak


def summarize(samples):
    values = np.asarray([item["latency_ms"] for item in samples], dtype=float)
    return {
        "latency_mean_ms": float(values.mean()),
        "latency_sd_ms": (
            float(values.std(ddof=1)) if len(values) > 1 else np.nan),
        "latency_median_ms": float(np.median(values)),
        "latency_p25_ms": float(np.quantile(values, 0.25)),
        "latency_p75_ms": float(np.quantile(values, 0.75)),
        "latency_p95_ms": float(np.quantile(values, 0.95)),
        "repeats": int(len(values)),
    }


def fusion_modules(model):
    order = ("shallow", "mid", "deep")
    return {
        level: getattr(model, f"fusion_{level}")
        for level in order
        if hasattr(model, f"fusion_{level}")
    }


def capture_fusion_inputs_and_execution(model, modules, x):
    """Capture real fusion inputs and parameters invoked by this exact forward.

    ``executed``/``active`` below deliberately means that the parameter is
    directly owned by a module whose forward was invoked.  It is independent
    of ``requires_grad`` and does not claim that every invoked parameter has a
    non-zero causal effect for every runtime ablation.
    """
    captured = {}
    executed_parameter_ids = set()
    handles = []
    for level, module in modules.items():
        def hook(_, inputs, level_name=level):
            captured[level_name] = tuple(
                value.detach() for value in inputs[:2])
        handles.append(module.register_forward_pre_hook(hook))

    for module in model.modules():
        direct_parameter_ids = tuple(
            id(parameter) for parameter in module.parameters(recurse=False))
        if not direct_parameter_ids:
            continue

        def execution_hook(
                _module, _inputs, _output,
                parameter_ids=direct_parameter_ids):
            executed_parameter_ids.update(parameter_ids)

        handles.append(module.register_forward_hook(execution_hook))

    try:
        with torch.inference_mode():
            model(x)
        torch.cuda.synchronize(x.device)
    finally:
        for handle in handles:
            handle.remove()

    missing = set(modules) - set(captured)
    if missing:
        raise RuntimeError(f"Fusion hooks did not capture: {sorted(missing)}")
    parameter_lookup = {
        id(parameter): parameter for parameter in model.parameters()}
    unknown = executed_parameter_ids - set(parameter_lookup)
    if unknown:
        raise RuntimeError(
            "Execution hooks observed parameters outside model.parameters()")
    executed_parameters = int(sum(
        parameter_lookup[parameter_id].numel()
        for parameter_id in executed_parameter_ids))
    executed_trainable_parameters = int(sum(
        parameter_lookup[parameter_id].numel()
        for parameter_id in executed_parameter_ids
        if parameter_lookup[parameter_id].requires_grad))
    return captured, executed_parameters, executed_trainable_parameters


def thop_limitations(modules):
    """Return limitations that THOP cannot quantify as a coverage fraction."""
    limitations = [
        "functional interpolation/pooling/activation operations",
        "tensor reductions, concatenation, reshaping and elementwise arithmetic",
    ]
    module_names = {
        type(module).__name__ for module in modules.values()}
    if any("CrossAttn" in name for name in module_names):
        limitations.extend([
            "CrossAttn torch.matmul(Q,K^T)",
            "CrossAttn torch.matmul(attention,V)",
            "CrossAttn functional softmax",
        ])
    return limitations


def try_thop(model, x, modules, captured):
    limitations = thop_limitations(modules)
    result = {
        "end_to_end_macs_thop_partial": np.nan,
        "end_to_end_flops_thop_partial": np.nan,
        "fusion_only_macs_thop_partial": np.nan,
        "fusion_only_flops_thop_partial": np.nan,
        "flop_profiler": "unavailable",
        "flop_profile_error": "",
        "macs_coverage": "not_profiled",
        "macs_coverage_fraction": np.nan,
        "unsupported_or_untracked_ops": "; ".join(limitations),
        "macs_strictly_cross_method_comparable": False,
        "macs_interpretation": (
            "Secondary diagnostic only; do not use as strict cross-method "
            "evidence."),
    }
    try:
        from thop import profile
        with torch.inference_mode():
            macs, _ = profile(model, inputs=(x,), verbose=False)
        fusion_macs = 0.0
        for level, module in modules.items():
            with torch.inference_mode():
                level_macs, _ = profile(
                    module, inputs=captured[level], verbose=False)
            fusion_macs += float(level_macs)
        result.update({
            "end_to_end_macs_thop_partial": float(macs),
            "end_to_end_flops_thop_partial": float(2.0 * macs),
            "fusion_only_macs_thop_partial": float(fusion_macs),
            "fusion_only_flops_thop_partial": float(2.0 * fusion_macs),
            "flop_profiler": "thop",
            "macs_coverage": (
                "incomplete_supported_module_ops_only_no_denominator"),
            "macs_interpretation": (
                "Partial THOP estimate, not a complete lower/upper bound and "
                "not strictly cross-method comparable; functional and tensor "
                "operations listed in unsupported_or_untracked_ops are absent "
                "from the estimate."),
        })
    except Exception as exc:
        result["flop_profile_error"] = f"{type(exc).__name__}: {exc}"
    return result


def skipped_thop(modules):
    limitations = thop_limitations(modules)
    return {
        "end_to_end_macs_thop_partial": np.nan,
        "end_to_end_flops_thop_partial": np.nan,
        "fusion_only_macs_thop_partial": np.nan,
        "fusion_only_flops_thop_partial": np.nan,
        "flop_profiler": "skipped",
        "flop_profile_error": "",
        "macs_coverage": "not_profiled",
        "macs_coverage_fraction": np.nan,
        "unsupported_or_untracked_ops": "; ".join(limitations),
        "macs_strictly_cross_method_comparable": False,
        "macs_interpretation": (
            "Not profiled; parameters, measured latency and measured memory "
            "are the primary cross-method evidence."),
    }


def default_entries(root: Path):
    return [
        {
            "method": "TCFF",
            "variant": variant,
            "exp_dir": root / "Runs_foundation" / "sam_tgf_v3",
        }
        for variant in ("full", "product", "g_only", "residual", "no_aux")
    ] + [
        {
            "method": "Concat",
            "variant": "native",
            "exp_dir": root / "Runs_foundation" / "sam_concat",
        },
        {
            "method": "BAGF",
            "variant": "native",
            "exp_dir": root / "Runs_foundation" / "sam_bagf",
        },
        {
            "method": "DSFF",
            "variant": "native",
            "exp_dir": root / "Runs_foundation" / "sam_dsff",
        },
        {
            "method": "CrossAttn",
            "variant": "native",
            "exp_dir": root / "Runs_foundation_v2" / "sam_crossattn",
        },
        {
            "method": "AFF",
            "variant": "native",
            "exp_dir": root / "Runs_foundation_v2" / "sam_aff",
        },
    ]


def main(args):
    root = Path(args.project_root).resolve()
    results = Path(args.results_root).resolve()
    dataset_dir = Path(args.dataset_dir).resolve()
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("Formal V2 efficiency timing requires one CUDA device")
    if args.warmup < 10 or args.repeats < 30:
        raise ValueError(
            "Formal protocol requires warmup>=10 and repeats>=30")
    torch.cuda.set_device(device)
    dataset = TopCoWTestDataset(str(dataset_dir), split="test")
    sample = dataset[args.case_index]
    x = sample["image"].unsqueeze(0).float().to(device)
    if x.shape[0] != 1 or x.dtype != torch.float32:
        raise RuntimeError("Formal protocol requires batch=1 and FP32 input")

    raw_rows = []
    summary_rows = []
    provenance = []
    entries = default_entries(root)
    requested = {
        value.strip().lower()
        for value in args.only.split(",") if value.strip()}
    if requested:
        entries = [
            entry for entry in entries
            if entry["method"].lower() in requested or
            f"{entry['method']}/{entry['variant']}".lower() in requested]
    if not entries:
        raise ValueError(f"No benchmark entries matched --only={args.only!r}")
    identities = [
        f"{entry['method']}/{entry['variant']}" for entry in entries]
    if len(identities) != len(set(identities)):
        raise RuntimeError(f"Duplicate benchmark identities: {identities}")
    processes_before = compute_process_snapshot()
    require_isolated_gpu_host(processes_before)

    for entry in entries:
        exp_dir = Path(entry["exp_dir"]).resolve()
        if not (exp_dir / "best_model.pth").exists():
            raise FileNotFoundError(exp_dir / "best_model.pth")
        runtime_variant = (
            entry["variant"] if entry["method"] == "TCFF" else "full")
        model, config, checkpoint = load_foundation_model(
            exp_dir, device, trust_variant=runtime_variant)
        model.eval()
        model.set_rebuttal_runtime(
            trust_variant=runtime_variant,
            corruption={"type": "none", "severity": 0.0, "seed": 0},
            instrumentation=False,
            temperature_scale=1.0,
            gate_transform="identity")
        parameter_dtypes = sorted({
            str(parameter.dtype).replace("torch.", "")
            for parameter in model.parameters()
            if parameter.is_floating_point()})
        if parameter_dtypes != ["float32"]:
            raise RuntimeError(
                f"Formal protocol requires FP32 parameters, got "
                f"{parameter_dtypes} for {entry['method']}/"
                f"{entry['variant']}")
        modules = fusion_modules(model)

        total_parameters = int(sum(
            parameter.numel() for parameter in model.parameters()))
        trainable_parameters = int(sum(
            parameter.numel() for parameter in model.parameters()
            if parameter.requires_grad))
        foundation_parameters = int(sum(
            parameter.numel()
            for parameter in model.foundation_encoder.parameters()))
        fusion_parameters = int(sum(
            parameter.numel() for module in modules.values()
            for parameter in module.parameters()))
        fusion_trainable_parameters = int(sum(
            parameter.numel() for module in modules.values()
            for parameter in module.parameters() if parameter.requires_grad))

        # This must precede capture_fusion_inputs_and_execution: otherwise the
        # retained encoder/fusion features inflate both the end-to-end memory
        # baseline and peak.
        # Bind the per-entry objects into the callables themselves.  Besides
        # making the measured target explicit, this prevents a later loop
        # iteration from changing the closure while CUDA work is synchronized.
        end_samples, end_baseline, end_peak = benchmark(
            lambda current_model=model, current_x=x: current_model(current_x),
            args.warmup,
            args.repeats,
            device)
        captured, executed_parameters, executed_trainable_parameters = (
            capture_fusion_inputs_and_execution(model, modules, x))
        level_order = [
            level for level in ("shallow", "mid", "deep")
            if level in modules]

        def fusion_forward(
                current_modules=modules,
                current_captured=captured,
                current_levels=tuple(level_order)):
            for level in current_levels:
                current_modules[level](*current_captured[level])

        fusion_samples, fusion_baseline, fusion_peak = benchmark(
            fusion_forward, args.warmup, args.repeats, device)
        flops = (
            try_thop(model, x, modules, captured)
            if not args.skip_flops else skipped_thop(modules))
        identity = f"{entry['method']}/{entry['variant']}"
        for row in end_samples:
            raw_rows.append({
                "method": entry["method"],
                "variant": entry["variant"],
                "identity": identity,
                "scope": "end_to_end",
                **row,
            })
        for row in fusion_samples:
            raw_rows.append({
                "method": entry["method"],
                "variant": entry["variant"],
                "identity": identity,
                "scope": "fusion_only_three_levels",
                **row,
            })
        summary_rows.append({
            "method": entry["method"],
            "variant": entry["variant"],
            "identity": identity,
            "device": torch.cuda.get_device_name(device),
            "batch_size": int(x.shape[0]),
            "input_shape": "x".join(map(str, x.shape)),
            "precision": str(x.dtype).replace("torch.", ""),
            "model_parameter_dtypes": ",".join(parameter_dtypes),
            "total_parameters": total_parameters,
            "trainable_parameters": trainable_parameters,
            "executed_parameters": executed_parameters,
            "executed_trainable_parameters": executed_trainable_parameters,
            "active_parameters": executed_parameters,
            "active_trainable_parameters": executed_trainable_parameters,
            "inactive_parameters": total_parameters - executed_parameters,
            "executed_parameter_fraction": (
                executed_parameters / total_parameters
                if total_parameters else np.nan),
            "active_parameter_definition": (
                "Unique parameters directly owned by modules invoked in the "
                "measured inference forward; includes frozen parameters and "
                "does not imply non-zero causal influence for every ablation."),
            "foundation_parameters": foundation_parameters,
            "fusion_parameters": fusion_parameters,
            "fusion_trainable_parameters": fusion_trainable_parameters,
            "nonfoundation_parameters": (
                total_parameters - foundation_parameters),
            "executed_graph_note": (
                "Counted dynamically with per-module forward hooks on the same "
                "real input. Runtime variants may execute a parameter even when "
                "their final ablation makes its output causally irrelevant."),
            "primary_cross_method_efficiency_evidence": (
                "parameter counts; measured latency; measured peak allocated "
                "CUDA memory"),
            "end_to_end_memory_measurement_state": (
                "model plus input only; fusion inputs are captured after this "
                "benchmark completes"),
            **{"end_to_end_" + key: value
               for key, value in summarize(end_samples).items()},
            "end_to_end_baseline_allocated_mb": end_baseline,
            "end_to_end_peak_allocated_mb": end_peak,
            "end_to_end_incremental_peak_mb": end_peak - end_baseline,
            **{"fusion_only_" + key: value
               for key, value in summarize(fusion_samples).items()},
            "fusion_only_baseline_allocated_mb": fusion_baseline,
            "fusion_only_peak_allocated_mb": fusion_peak,
            "fusion_only_incremental_peak_mb": fusion_peak - fusion_baseline,
            **flops,
        })
        provenance.append({
            "identity": identity,
            "exp_dir": str(exp_dir),
            "checkpoint": str(checkpoint),
            "checkpoint_hash": sha256_file(checkpoint),
            "model_config": config,
        })
        # Drop every strong reference before benchmarking the next identity.
        # Assigning None (rather than deleting loop-local names) also keeps
        # static definite-assignment analysis accurate across iterations.
        model = None
        captured = None
        modules = None
        fusion_forward = None
        gc.collect()
        torch.cuda.empty_cache()

    raw = pd.DataFrame(raw_rows)
    summary = pd.DataFrame(summary_rows)
    atomic_csv(
        results / "09_efficiency" / "v2_efficiency_raw_repeats.csv", raw)
    atomic_csv(
        results / "09_efficiency" / "v2_efficiency_summary.csv", summary)
    atomic_csv(
        results / "06_summary_tables" /
        "Table_R2_v2_cross_method_efficiency.csv", summary)
    processes_after = compute_process_snapshot()
    require_isolated_gpu_host(processes_after)
    manifest = {
        "protocol": "TCFF_V2_ISOLATED_EFFICIENCY_2026-07-30",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "dataset": str(dataset_dir),
        "case_id": sample["case_name"],
        "input_shape": list(x.shape),
        "batch_size": 1,
        "precision": "float32",
        "warmup": args.warmup,
        "repeats": args.repeats,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "python": platform.python_version(),
        "flop_convention": (
            "For explicitly suffixed THOP-partial fields only: FLOPs=2*MACs"),
        "primary_cross_method_efficiency_evidence": [
            "parameter counts",
            "measured latency",
            "measured peak allocated CUDA memory",
        ],
        "thop_evidence_role": (
            "Secondary partial estimate only. THOP does not count all "
            "functional/tensor operations, including explicit attention "
            "matmuls, so MACs/FLOPs are not strictly cross-method comparable."),
        "macs_strictly_cross_method_comparable": False,
        "active_parameter_definition": (
            "Unique parameters directly owned by modules invoked in the "
            "measured inference forward; includes frozen parameters and does "
            "not imply non-zero causal influence for every ablation."),
        "fusion_only_definition": (
            "sum of three fusion-module forwards on cached real inputs"),
        "end_to_end_definition": (
            "one complete forward including the frozen foundation encoder; "
            "measured before any fusion-input feature cache is created"),
        "concurrency_requirement": (
            "No other compute process on any GPU during formal timing"),
        "gpu_compute_processes_before": processes_before,
        "gpu_compute_processes_after": processes_after,
        "environment": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        },
        "models": provenance,
    }
    write_json(
        results / "09_efficiency" / "v2_efficiency_manifest.json", manifest)
    print(json.dumps({
        "status": "success",
        "rows": len(summary),
        "raw_repeats": len(raw),
        "device": manifest["gpu_name"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", default=str(here))
    parser.add_argument(
        "--results_root", default=str(here / "results_tcff_rebuttal"))
    parser.add_argument(
        "--dataset_dir", default=str(here / "crown_processed"))
    parser.add_argument("--case_index", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument(
        "--only", default="",
        help="Comma list of methods or method/variant identities")
    parser.add_argument("--skip_flops", action="store_true")
    raise SystemExit(main(parser.parse_args()))
