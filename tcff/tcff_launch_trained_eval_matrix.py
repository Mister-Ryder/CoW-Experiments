"""Evaluate independently trained TCFF variants on clean and shifted data."""

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
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temp.replace(path)


def parse_assignments(text):
    jobs = []
    for item in text.split(","):
        gpu, specification = item.strip().split("=", 1)
        variant, seed = specification.split(":", 1)
        jobs.append({"gpu": gpu, "variant": variant, "seed": int(seed)})
    return jobs


def severity_tag(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def status_value(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("status")
    except Exception:
        return None


def successful_run_matches(run_dir: Path, run: dict) -> bool:
    try:
        manifest = json.loads(
            (run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    except Exception:
        return False
    corruption = manifest.get("corruption", {})
    intervention = manifest.get("intervention", {
        "gate_alpha": None,
        "temperature_scale": 1.0,
        "gate_transform": "identity",
    })
    selection = manifest.get("selection", {})
    selection_is_explicit = (
        isinstance(selection, dict) and
        "case_ids" in selection and "max_cases" in selection)
    return (
        manifest.get("experiment_id") == run["experiment_id"] and
        manifest.get("model_variant") == run["variant"] and
        manifest.get("dataset") == run["dataset_name"] and
        int(manifest.get("seed", -1)) == int(run["seed"]) and
        manifest.get("checkpoint_hash") == run["expected_checkpoint_hash"] and
        manifest.get("data_split_hash") == run["expected_data_split_hash"] and
        selection_is_explicit and
        selection.get("case_ids", "") == "" and
        selection.get("max_cases") is None and
        corruption.get("type") == run["corruption"] and
        float(corruption.get("severity", float("nan"))) ==
        float(run["severity"]) and
        int(corruption.get("realization", -1)) ==
        int(run["realization"]) and
        intervention.get("gate_alpha") is None and
        float(intervention.get("temperature_scale", 1.0)) == 1.0 and
        intervention.get("gate_transform", "identity") == "identity")


def is_recent_running(status_path: Path, seconds: int = 900) -> bool:
    return (
        status_value(status_path) == "running" and
        time.time() - status_path.stat().st_mtime < seconds
        if status_path.exists() else False)


def allocate_run(results: Path, run: dict):
    base = (
        results / "03_raw_runs" / run["experiment_id"] / run["run_id"])
    if not base.exists():
        return run["run_id"], None
    if (status_value(base / "status.json") == "success" and
            successful_run_matches(base, run)):
        return None, "success"
    if is_recent_running(base / "status.json"):
        return None, "active"
    retry = 1
    while True:
        candidate_id = f"{run['run_id']}__retry{retry}"
        candidate = base.parent / candidate_id
        if not candidate.exists():
            return candidate_id, "retry"
        if (status_value(candidate / "status.json") == "success" and
                successful_run_matches(candidate, run)):
            return None, "success"
        if is_recent_running(candidate / "status.json"):
            return None, "active"
        retry += 1


def claim_run(results: Path, run: dict):
    claim_dir = results / "02_manifests" / ".v2_run_claims"
    claim_dir.mkdir(parents=True, exist_ok=True)
    safe_name = (
        f"{run['experiment_id']}__{run['run_id']}"
        .replace("/", "_").replace("\\", "_"))
    path = claim_dir / f"{safe_name}.json"
    payload = {
        "experiment_id": run["experiment_id"],
        "run_id": run["run_id"],
        "canonical_run_id": run["base_run_id"],
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
                results / "03_raw_runs" / run["experiment_id"] /
                run["run_id"])
            if same_host and not owner_alive and not run_dir.exists():
                path.unlink()
                return claim_run(results, run)
        except Exception:
            pass
        return None
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    return path


def build_runs(root: Path, results: Path, model_job: dict,
               corruption_realizations: int, args):
    variant = model_job["variant"]
    seed = model_job["seed"]
    training_run = (
        results / "03_raw_runs" / args.training_experiment_id /
        f"{args.training_run_prefix}{variant}_seed{seed}")
    training_status = json.loads((training_run / "status.json").read_text(encoding="utf-8"))
    if training_status.get("status") != "success":
        raise RuntimeError(f"training run not successful: {training_run}")
    exp_dir = training_run / "training_output" / "model"
    checkpoint = exp_dir / "best_model.pth"
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    expected_checkpoint_hash = sha256_file(checkpoint)
    expected_data_hashes = {
        str(root / directory): split_hash(root / directory)
        for _, _, directory in DATASETS}
    runs = []
    for tag, label, directory in DATASETS:
        runs.append({
            "experiment_id": args.clean_experiment_id,
            "run_id": (
                f"{args.eval_run_prefix}{variant}_seed{seed}_clean_{tag}"),
            "exp_dir": str(exp_dir), "dataset_dir": str(root / directory),
            "dataset_name": label, "dataset_tag": tag,
            "variant": variant, "seed": seed,
            "corruption": "none", "severity": 0.0,
            "severity_unit": "none", "realization": 0,
            "expected_checkpoint_hash": expected_checkpoint_hash,
            "expected_data_split_hash": expected_data_hashes[
                str(root / directory)],
        })
    # Robustness comparisons are source-validation only; fixed-checkpoint runs
    # supply the full three-domain severity chains.
    for corruption, severity, unit in REPRESENTATIVE_CORRUPTIONS:
        for realization in range(corruption_realizations):
            runs.append({
                "experiment_id": args.corruption_experiment_id,
                "run_id": (f"{args.eval_run_prefix}{variant}_seed{seed}_{corruption}_"
                           f"{severity_tag(severity)}_r{realization}_crown"),
                "exp_dir": str(exp_dir),
                "dataset_dir": str(root / "crown_processed"),
                "dataset_name": "TopCoW_val", "dataset_tag": "crown",
                "variant": variant, "seed": seed,
                "corruption": corruption, "severity": severity,
                "severity_unit": unit, "realization": realization,
                "expected_checkpoint_hash": expected_checkpoint_hash,
                "expected_data_split_hash": expected_data_hashes[
                    str(root / "crown_processed")],
            })
    return runs


def main(args):
    root = Path(args.project_root).resolve()
    results = Path(args.results_root).resolve()
    assignments = parse_assignments(args.assignments)
    state_path = (results / "02_manifests" /
                  f"{args.manifest_prefix}_{args.node_tag}.json")
    state = {
        "node_tag": args.node_tag, "assignments": assignments,
        "corruption_realizations": args.corruption_realizations,
        "expected_total": (
            len(assignments) *
            (len(DATASETS) +
             len(REPRESENTATIVE_CORRUPTIONS) *
             args.corruption_realizations)),
        "completed": [], "failed": [], "skipped_success": [],
        "deferred_active": [],
        "started_at_epoch": time.time(),
    }
    lock = threading.Lock()
    atomic_json(state_path, state)

    def evaluate_model(model_job):
        try:
            runs = build_runs(root, results, model_job,
                              args.corruption_realizations, args)
        except Exception as exc:
            with lock:
                state["failed"].append({**model_job, "stage": "build_runs",
                                        "reason": f"{type(exc).__name__}: {exc}"})
                atomic_json(state_path, state)
            return
        for run in runs:
            actual_run_id, disposition = allocate_run(results, run)
            if disposition in {"success", "active"}:
                with lock:
                    target = (
                        state["skipped_success"]
                        if disposition == "success"
                        else state["deferred_active"])
                    target.append(run)
                    atomic_json(state_path, state)
                continue
            run = {
                **run,
                "base_run_id": run["run_id"],
                "run_id": actual_run_id,
                "retry_of": (
                    run["run_id"] if disposition == "retry" else None),
            }
            run_dir = (
                results / "03_raw_runs" / run["experiment_id"] /
                run["run_id"])
            claim_path = claim_run(results, run)
            if claim_path is None:
                with lock:
                    state["deferred_active"].append({
                        **run,
                        "reason": "atomic claim already owned",
                    })
                    atomic_json(state_path, state)
                continue
            command = [
                args.python, str(root / "tcff_rebuttal_runner.py"),
                "--project_root", str(root), "--results_root", str(results),
                "--experiment_id", run["experiment_id"],
                "--run_id", run["run_id"],
                "--canonical_run_id", run["base_run_id"],
                "--exp_dir", run["exp_dir"],
                "--dataset_dir", run["dataset_dir"],
                "--dataset_name", run["dataset_name"],
                "--model_variant", run["variant"],
                "--corruption_type", run["corruption"],
                "--severity", str(run["severity"]),
                "--severity_unit", run["severity_unit"],
                "--realization", str(run["realization"]),
                "--seed", str(run["seed"]), "--device", "cuda:0",
            ]
            if run["retry_of"] is not None:
                command.extend(["--retry_of", run["retry_of"]])
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = model_job["gpu"]
            env["PYTHONPATH"] = f"{root / 'SAM-Med3D'}{os.pathsep}{root}"
            env["NUMEXPR_MAX_THREADS"] = "64"
            env["OMP_NUM_THREADS"] = "8"
            conda_lib = "/root/miniconda3/envs/cu122py39/lib"
            env["LD_LIBRARY_PATH"] = conda_lib + ":" + env.get("LD_LIBRARY_PATH", "")
            log_dir = results / "10_logs" / "trained_eval_matrix"
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / f"{run['experiment_id']}__{run['run_id']}.log"
            try:
                with log_path.open(
                        "w", encoding="utf-8", buffering=1) as log:
                    process = subprocess.run(
                        command, cwd=root, env=env, stdout=log,
                        stderr=subprocess.STDOUT, check=False)
                if process.returncode == 0 and run_dir.exists():
                    (run_dir / "stdout.log").write_bytes(
                        log_path.read_bytes())
                    (run_dir / "stderr.log").touch(exist_ok=True)
                record = {
                    **run,
                    "gpu": model_job["gpu"],
                    "exit_code": process.returncode,
                    "finished_at_epoch": time.time(),
                }
                with lock:
                    target = (
                        state["completed"] if process.returncode == 0
                        else state["failed"])
                    target.append(record)
                    atomic_json(state_path, state)
                    print(json.dumps(record, sort_keys=True), flush=True)
            finally:
                claim_path.unlink(missing_ok=True)

    gpu_ids = [job["gpu"] for job in assignments]
    if len(gpu_ids) != len(set(gpu_ids)):
        raise ValueError(f"GPU assignments must be unique: {gpu_ids}")
    threads = [threading.Thread(target=evaluate_model, args=(job,), daemon=False)
               for job in assignments]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    state["finished_at_epoch"] = time.time()
    atomic_json(state_path, state)
    handled = len(state["completed"]) + len(state["skipped_success"])
    complete = (
        not state["failed"] and not state["deferred_active"] and
        handled == state["expected_total"])
    print(json.dumps({
        "status": "success" if complete else "incomplete",
        "expected_total": state["expected_total"],
        "handled_total": handled,
        "completed": len(state["completed"]),
        "failed": len(state["failed"]),
        "skipped_success": len(state["skipped_success"]),
        "deferred_active": len(state["deferred_active"]),
    }, sort_keys=True))
    return 0 if complete else 2


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", default=str(here))
    parser.add_argument("--results_root", default=str(here / "results_tcff_rebuttal"))
    parser.add_argument("--python", default="/root/autodl-tmp/tcff-venv/bin/python")
    parser.add_argument("--node_tag", required=True)
    parser.add_argument("--assignments", required=True,
                        help="comma list, e.g. 0=full:0,1=u_only:0")
    parser.add_argument("--corruption_realizations", type=int, default=3)
    parser.add_argument("--training_experiment_id",
                        default="P0_D_TRAIN_VARIANTS")
    parser.add_argument("--training_run_prefix", default="")
    parser.add_argument("--clean_experiment_id",
                        default="P0_D_TRAINED_EVAL_CLEAN")
    parser.add_argument("--corruption_experiment_id",
                        default="P0_D_TRAINED_EVAL_CORRUPTION")
    parser.add_argument("--eval_run_prefix", default="")
    parser.add_argument("--manifest_prefix",
                        default="p0_trained_eval_matrix")
    raise SystemExit(main(parser.parse_args()))
