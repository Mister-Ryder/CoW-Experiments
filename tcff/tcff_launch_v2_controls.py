"""Launch low-cost V2 inference controls with audited, resumable run IDs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import queue
import socket
import subprocess
import threading
import time

from tcff_rebuttal_runner import sha256_file, split_hash


DATASETS = (
    ("crown", "TopCoW_val", "crown_processed"),
    ("lausanne", "Lausanne", "lausanne_processed"),
    ("ixi", "IXI_HH", "ixi_processed"),
)
REPRESENTATIVE_CORRUPTIONS = (
    ("gaussian", 0.20, "channel_std_multiplier"),
    ("channel_dropout", 0.50, "channel_fraction"),
    ("translation", 4.0, "feature_voxel"),
    ("donor_mix", 0.50, "mixing_weight"),
)


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def tag(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def conditions(root: Path):
    clean = []
    for dataset_tag, dataset_name, directory in DATASETS:
        clean.append({
            "dataset_tag": dataset_tag,
            "dataset_name": dataset_name,
            "dataset_dir": str(root / directory),
            "corruption": "none",
            "severity": 0.0,
            "unit": "none",
            "realization": 0,
        })
    source_shift = []
    for corruption, severity, unit in REPRESENTATIVE_CORRUPTIONS:
        for realization in range(3):
            source_shift.append({
                "dataset_tag": "crown",
                "dataset_name": "TopCoW_val",
                "dataset_dir": str(root / "crown_processed"),
                "corruption": corruption,
                "severity": severity,
                "unit": unit,
                "realization": realization,
            })
    return clean, source_shift


def condition_tag(condition: dict) -> str:
    if condition["corruption"] == "none":
        return f"clean_{condition['dataset_tag']}"
    return (
        f"{condition['corruption']}_{tag(condition['severity'])}_"
        f"r{condition['realization']}_{condition['dataset_tag']}")


def build_jobs(root: Path):
    clean, source_shift = conditions(root)
    alpha_shift = [
        condition for condition in source_shift
        if condition["corruption"] in {"gaussian", "translation", "donor_mix"}]
    jobs = []
    for scale in (0.5, 1.0, 2.0):
        for condition in clean + source_shift:
            jobs.append({
                **condition,
                "family": "temperature",
                "experiment_id": "V2_TEMPERATURE_SENSITIVITY",
                "run_id": (
                    f"full_temp_{tag(scale)}_{condition_tag(condition)}"),
                "model_variant": "full",
                "gate_scale": 1.0,
                "temperature_scale": scale,
                "gate_transform": "identity",
                "gate_alpha": None,
                "save_predictions": scale == 1.0,
            })
    for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
        for condition in clean + alpha_shift:
            jobs.append({
                **condition,
                "family": "alpha",
                "experiment_id": "V2_CONSTANT_ACCEPTANCE",
                "run_id": (
                    f"full_alpha_{tag(alpha)}_{condition_tag(condition)}"),
                "model_variant": "full",
                "gate_scale": 1.0,
                "temperature_scale": 1.0,
                "gate_transform": "identity",
                "gate_alpha": alpha,
                "save_predictions": alpha in (0.0, 1.0),
            })
    for transform in ("shuffle", "invert"):
        for condition in clean + source_shift:
            jobs.append({
                **condition,
                "family": "gate_transform",
                "experiment_id": "V2_GATE_TRANSFORM",
                "run_id": (
                    f"full_gate_{transform}_{condition_tag(condition)}"),
                "model_variant": "full",
                "gate_scale": 1.0,
                "temperature_scale": 1.0,
                "gate_transform": transform,
                "gate_alpha": None,
                "save_predictions": False,
            })
    for variant in (
            "full", "product", "g_only", "u_only", "additive",
            "residual", "no_aux"):
        for condition in clean + source_shift:
            jobs.append({
                **condition,
                "family": "policy_replay",
                "experiment_id": "V2_FIXED_POLICY_REPLAY",
                "run_id": (
                    f"{variant}_policy_{condition_tag(condition)}"),
                "model_variant": variant,
                "gate_scale": 1.0,
                "temperature_scale": 1.0,
                "gate_transform": "identity",
                "gate_alpha": None,
                "save_predictions": False,
            })
    for scale in (0.0, 0.25, 0.5, 0.75, 1.0):
        for condition in clean + alpha_shift:
            jobs.append({
                **condition,
                "family": "dose_response",
                "experiment_id": "V2_GATE_DOSE_RESPONSE",
                "run_id": (
                    f"full_scale_{tag(scale)}_{condition_tag(condition)}"),
                "model_variant": "full",
                "gate_scale": scale,
                "temperature_scale": 1.0,
                "gate_transform": "identity",
                "gate_alpha": None,
                "save_predictions": False,
            })
    for condition in clean + source_shift:
        jobs.append({
            **condition,
            "family": "mean_matched",
            "experiment_id": "V2_GATE_MEAN_MATCHED",
            "run_id": f"full_mean_matched_{condition_tag(condition)}",
            "model_variant": "full",
            "gate_scale": 1.0,
            "temperature_scale": 1.0,
            "gate_transform": "mean_matched",
            "gate_alpha": None,
            "save_predictions": False,
        })
    return jobs


def status_value(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("status")
    except Exception:
        return None


def successful_run_matches(run_dir: Path, job: dict) -> bool:
    """Reject a same-name success whose semantic condition is different."""
    try:
        manifest = json.loads(
            (run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    except Exception:
        return False
    corruption = manifest.get("corruption", {})
    intervention = manifest.get("intervention", {})
    selection = manifest.get("selection", {})
    selection_is_explicit = (
        isinstance(selection, dict) and
        "case_ids" in selection and "max_cases" in selection)
    expected_alpha = job["gate_alpha"]
    observed_alpha = intervention.get("gate_alpha")
    return (
        manifest.get("experiment_id") == job["experiment_id"] and
        manifest.get("model_variant") == job["model_variant"] and
        manifest.get("dataset") == job["dataset_name"] and
        corruption.get("type") == job["corruption"] and
        float(corruption.get("severity", float("nan"))) ==
        float(job["severity"]) and
        int(corruption.get("realization", -1)) == int(job["realization"]) and
        int(manifest.get("seed", -1)) == 0 and
        manifest.get("checkpoint_hash") == job["expected_checkpoint_hash"] and
        manifest.get("data_split_hash") == job["expected_data_split_hash"] and
        selection_is_explicit and
        selection.get("case_ids", "") == "" and
        selection.get("max_cases") == job["max_cases"] and
        observed_alpha == expected_alpha and
        float(intervention.get("gate_scale", float("nan"))) ==
        float(job["gate_scale"]) and
        float(intervention.get("temperature_scale", float("nan"))) ==
        float(job["temperature_scale"]) and
        intervention.get("gate_transform") == job["gate_transform"])


def is_recent_nonterminal(status_path: Path, seconds: int = 900) -> bool:
    if not status_path.exists():
        return False
    return (
        status_value(status_path) == "running" and
        time.time() - status_path.stat().st_mtime < seconds)


def claim_run(results: Path, job: dict):
    claim_dir = results / "02_manifests" / ".v2_run_claims"
    claim_dir.mkdir(parents=True, exist_ok=True)
    safe_name = (
        f"{job['experiment_id']}__{job['run_id']}"
        .replace("/", "_").replace("\\", "_"))
    path = claim_dir / f"{safe_name}.json"
    payload = {
        "experiment_id": job["experiment_id"],
        "run_id": job["run_id"],
        "canonical_run_id": job["base_run_id"],
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "claimed_at_epoch": time.time(),
    }
    try:
        descriptor = os.open(
            path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
            same_host = existing.get("hostname") == socket.gethostname()
            owner_pid = int(existing.get("pid", -1))
            owner_alive = owner_pid > 0
            if owner_alive:
                try:
                    os.kill(owner_pid, 0)
                except ProcessLookupError:
                    owner_alive = False
                except PermissionError:
                    owner_alive = True
            run_dir = (
                results / "03_raw_runs" / job["experiment_id"] /
                job["run_id"])
            if same_host and not owner_alive and not run_dir.exists():
                path.unlink()
                return claim_run(results, job)
        except Exception:
            pass
        return None
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    return path


def allocate_run(results: Path, job: dict):
    base = (results / "03_raw_runs" / job["experiment_id"] /
            job["run_id"])
    if not base.exists():
        return job["run_id"], None
    if (status_value(base / "status.json") == "success" and
            successful_run_matches(base, job)):
        return None, "success"
    if is_recent_nonterminal(base / "status.json"):
        return None, "active"
    retry = 1
    while True:
        candidate_id = f"{job['run_id']}__retry{retry}"
        candidate = base.parent / candidate_id
        if not candidate.exists():
            return candidate_id, "retry"
        if (status_value(candidate / "status.json") == "success" and
                successful_run_matches(candidate, job)):
            return None, "success"
        if is_recent_nonterminal(candidate / "status.json"):
            return None, "active"
        retry += 1


def main(args):
    root = Path(args.project_root).resolve()
    results = Path(args.results_root).resolve()
    selected_families = {
        value.strip() for value in args.only.split(",") if value.strip()}
    jobs = [
        job for job in build_jobs(root)
        if not selected_families or job["family"] in selected_families]
    if args.limit is not None:
        jobs = jobs[:args.limit]
    checkpoint = root / "Runs_foundation" / "sam_tgf_v3" / "best_model.pth"
    expected_checkpoint_hash = sha256_file(checkpoint)
    data_hashes = {}
    for job in jobs:
        dataset_dir = Path(job["dataset_dir"])
        key = str(dataset_dir)
        if key not in data_hashes:
            data_hashes[key] = split_hash(dataset_dir)
        job["expected_checkpoint_hash"] = expected_checkpoint_hash
        job["expected_data_split_hash"] = data_hashes[key]
        job["max_cases"] = args.max_cases

    work = queue.Queue()
    skipped = []
    deferred = []
    planned = []
    for job in jobs:
        actual_run_id, disposition = allocate_run(results, job)
        if disposition == "success":
            skipped.append(job)
            continue
        if disposition == "active":
            deferred.append(job)
            continue
        queued = dict(job)
        queued["base_run_id"] = job["run_id"]
        queued["run_id"] = actual_run_id
        queued["retry_of"] = (
            job["run_id"] if disposition == "retry" else None)
        planned.append(queued)
        work.put(queued)

    state = {
        "protocol": "TCFF_V2_CONTROLS_2026-07-30",
        "planned_total": len(jobs),
        "queued_this_launch": len(planned),
        "skipped_success": skipped,
        "deferred_active": deferred,
        "completed": [],
        "failed": [],
        "started_at_epoch": time.time(),
    }
    state_path = (
        results / "02_manifests" / f"v2_controls_{args.node_tag}.json")
    atomic_json(state_path, state)
    lock = threading.Lock()
    log_dir = results / "10_logs" / "v2_controls"
    log_dir.mkdir(parents=True, exist_ok=True)

    def worker(gpu: str):
        while True:
            try:
                job = work.get_nowait()
            except queue.Empty:
                return
            source = dict(job)
            source["run_id"] = job["base_run_id"]
            actual_run_id, disposition = allocate_run(results, source)
            if disposition in {"success", "active"}:
                with lock:
                    target = (
                        state["skipped_success"]
                        if disposition == "success"
                        else state["deferred_active"])
                    target.append(source)
                    state["remaining"] = work.qsize()
                    atomic_json(state_path, state)
                work.task_done()
                continue
            job = {
                **source,
                "base_run_id": source["run_id"],
                "run_id": actual_run_id,
                "retry_of": (
                    source["run_id"] if disposition == "retry" else None),
            }
            claim_path = claim_run(results, job)
            if claim_path is None:
                with lock:
                    state["deferred_active"].append({
                        **source,
                        "reason": "atomic claim already owned",
                    })
                    state["remaining"] = work.qsize()
                    atomic_json(state_path, state)
                work.task_done()
                continue
            try:
                command = [
                    args.python, str(root / "tcff_rebuttal_runner.py"),
                    "--project_root", str(root),
                    "--results_root", str(results),
                    "--experiment_id", job["experiment_id"],
                    "--run_id", job["run_id"],
                    "--canonical_run_id", job["base_run_id"],
                    "--exp_dir",
                    str(root / "Runs_foundation" / "sam_tgf_v3"),
                    "--dataset_dir", job["dataset_dir"],
                    "--dataset_name", job["dataset_name"],
                    "--model_variant", job["model_variant"],
                    "--corruption_type", job["corruption"],
                    "--severity", str(job["severity"]),
                    "--severity_unit", job["unit"],
                    "--realization", str(job["realization"]),
                    "--seed", "0",
                    "--device", "cuda:0",
                    "--gate_scale", str(job["gate_scale"]),
                    "--temperature_scale", str(job["temperature_scale"]),
                    "--gate_transform", job["gate_transform"],
                ]
                if job["retry_of"] is not None:
                    command.extend(["--retry_of", job["retry_of"]])
                if job["gate_alpha"] is not None:
                    command.extend(["--gate_alpha", str(job["gate_alpha"])])
                if job["save_predictions"]:
                    command.append("--save_predictions")
                if args.max_cases is not None:
                    command.extend(["--max_cases", str(args.max_cases)])
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = gpu
                env["PYTHONPATH"] = (
                    f"{root / 'SAM-Med3D'}{os.pathsep}{root}")
                env["NUMEXPR_MAX_THREADS"] = "64"
                env["OMP_NUM_THREADS"] = "8"
                conda_lib = "/root/miniconda3/envs/cu122py39/lib"
                env["LD_LIBRARY_PATH"] = (
                    conda_lib + ":" + env.get("LD_LIBRARY_PATH", ""))
                log_path = (
                    log_dir /
                    f"{job['experiment_id']}__{job['run_id']}.log")
                with log_path.open(
                        "w", encoding="utf-8", buffering=1) as log:
                    process = subprocess.run(
                        command, cwd=root, env=env, stdout=log,
                        stderr=subprocess.STDOUT, check=False)
                run_dir = (
                    results / "03_raw_runs" / job["experiment_id"] /
                    job["run_id"])
                if process.returncode == 0 and run_dir.exists():
                    (run_dir / "stdout.log").write_bytes(
                        log_path.read_bytes())
                    (run_dir / "stderr.log").touch(exist_ok=True)
                record = {
                    **job,
                    "gpu": gpu,
                    "exit_code": process.returncode,
                    "finished_at_epoch": time.time(),
                }
                with lock:
                    target = (
                        state["completed"] if process.returncode == 0
                        else state["failed"])
                    target.append(record)
                    state["remaining"] = work.qsize()
                    atomic_json(state_path, state)
                    print(json.dumps(record, sort_keys=True), flush=True)
            finally:
                claim_path.unlink(missing_ok=True)
                work.task_done()

    gpu_ids = [value.strip() for value in args.gpus.split(",") if value.strip()]
    if not gpu_ids:
        raise ValueError("At least one GPU ID is required")
    if len(gpu_ids) != len(set(gpu_ids)):
        raise ValueError(f"GPU IDs must be unique: {gpu_ids}")
    threads = [
        threading.Thread(target=worker, args=(gpu,), daemon=False)
        for gpu in gpu_ids]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    state["finished_at_epoch"] = time.time()
    state["remaining"] = work.qsize()
    atomic_json(state_path, state)
    complete = (
        not state["failed"] and not state["deferred_active"] and
        work.qsize() == 0)
    print(json.dumps({
        "status": "success" if complete else "incomplete",
        "planned_total": len(jobs),
        "queued_this_launch": len(planned),
        "skipped_success": len(skipped),
        "deferred_active": len(deferred),
        "completed": len(state["completed"]),
        "failed": len(state["failed"]),
    }, sort_keys=True))
    return 0 if complete else 2


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", default=str(here))
    parser.add_argument(
        "--results_root", default=str(here / "results_tcff_rebuttal"))
    parser.add_argument(
        "--python", default="/root/miniconda3/envs/cu122py39/bin/python")
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--node_tag", default="server8")
    parser.add_argument(
        "--only", default="",
        help=(
            "Comma-separated subset: temperature,alpha,gate_transform,"
            "policy_replay,dose_response,mean_matched"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max_cases", type=int)
    raise SystemExit(main(parser.parse_args()))
