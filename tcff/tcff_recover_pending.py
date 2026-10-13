"""Recover interrupted TCFF queues without overwriting historical run dirs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import queue
import subprocess
import threading
import time


DATASETS = (
    ("crown", "TopCoW_val", "crown_processed"),
    ("lausanne", "Lausanne", "lausanne_processed"),
    ("ixi", "IXI_HH", "ixi_processed"),
)
TRAINED_VARIANTS = (
    "full", "residual", "u_only", "g_only", "no_aux", "additive")
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


def severity_tag(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def trained_jobs(root: Path, results: Path):
    jobs = []
    for variant in TRAINED_VARIANTS:
        for seed in range(3):
            training = (
                results / "03_raw_runs" / "P0_D_TRAIN_VARIANTS" /
                f"{variant}_seed{seed}")
            status = json.loads(
                (training / "status.json").read_text(encoding="utf-8"))
            checkpoint = training / "training_output" / "model" / "best_model.pth"
            if status.get("status") != "success" or not checkpoint.exists():
                raise RuntimeError(f"Missing successful training asset: {training}")
            exp_dir = str(checkpoint.parent)
            for dataset_tag, dataset_name, directory in DATASETS:
                jobs.append({
                    "family": "trained",
                    "experiment_id": "P0_D_TRAINED_EVAL_CLEAN",
                    "run_id": f"{variant}_seed{seed}_clean_{dataset_tag}",
                    "exp_dir": exp_dir,
                    "dataset_dir": str(root / directory),
                    "dataset_name": dataset_name,
                    "model_variant": variant,
                    "corruption": "none",
                    "severity": 0.0,
                    "unit": "none",
                    "realization": 0,
                    "seed": seed,
                })
            for corruption, severity, unit in REPRESENTATIVE_CORRUPTIONS:
                for realization in range(3):
                    jobs.append({
                        "family": "trained",
                        "experiment_id": "P0_D_TRAINED_EVAL_CORRUPTION",
                        "run_id": (
                            f"{variant}_seed{seed}_{corruption}_"
                            f"{severity_tag(severity)}_r{realization}_crown"),
                        "exp_dir": exp_dir,
                        "dataset_dir": str(root / "crown_processed"),
                        "dataset_name": "TopCoW_val",
                        "model_variant": variant,
                        "corruption": corruption,
                        "severity": severity,
                        "unit": unit,
                        "realization": realization,
                        "seed": seed,
                    })
    return jobs


def product_jobs(root: Path, results: Path):
    plan_path = (
        results / "02_manifests" /
        "p0_product_intervention_matrix_plan.json")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    directory_for_tag = {
        dataset_tag: directory
        for dataset_tag, _, directory in DATASETS}
    jobs = []
    for source in plan["jobs"]:
        job = dict(source)
        job.update({
            "family": "product",
            "exp_dir": str(root / "Runs_foundation" / "sam_tgf_v3"),
            "dataset_dir": str(root / directory_for_tag[job["dataset_tag"]]),
            "model_variant": "product",
            "seed": 0,
        })
        jobs.append(job)
    return jobs


def status_value(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("status")
    except Exception:
        return None


def is_recent_nonterminal(status_path: Path, seconds: int = 900):
    if not status_path.exists():
        return False
    status = status_value(status_path)
    return (
        status in {"running", "partial"} and
        time.time() - status_path.stat().st_mtime < seconds)


def allocate_run(results: Path, job: dict):
    base = (
        results / "03_raw_runs" / job["experiment_id"] / job["run_id"])
    if not base.exists():
        return job["run_id"], None
    if status_value(base / "status.json") == "success":
        return None, "success"
    if is_recent_nonterminal(base / "status.json"):
        return None, "active"
    retry = 1
    while True:
        candidate_id = f"{job['run_id']}__retry{retry}"
        candidate = base.parent / candidate_id
        if not candidate.exists():
            return candidate_id, "retry"
        if status_value(candidate / "status.json") == "success":
            return None, "success"
        if is_recent_nonterminal(candidate / "status.json"):
            return None, "active"
        retry += 1


def main(args):
    root = Path(args.project_root).resolve()
    results = Path(args.results_root).resolve()
    requested = {
        value.strip() for value in args.only.split(",") if value.strip()}
    jobs = []
    if not requested or "trained" in requested:
        jobs.extend(trained_jobs(root, results))
    if not requested or "product" in requested:
        jobs.extend(product_jobs(root, results))

    skipped = []
    deferred = []
    queued = []
    work = queue.Queue()
    for source in jobs:
        actual_run_id, disposition = allocate_run(results, source)
        if disposition == "success":
            skipped.append(source)
            continue
        if disposition == "active":
            deferred.append(source)
            continue
        job = dict(source)
        job["base_run_id"] = source["run_id"]
        job["run_id"] = actual_run_id
        job["retry_of"] = (
            source["run_id"] if disposition == "retry" else None)
        queued.append(job)
        work.put(job)
        if args.limit is not None and len(queued) >= args.limit:
            break

    state = {
        "protocol": "TCFF_INTERRUPTED_QUEUE_RECOVERY_2026-07-30",
        "expected_total": len(jobs),
        "queued_this_launch": len(queued),
        "skipped_success": skipped,
        "deferred_active": deferred,
        "completed": [],
        "failed": [],
        "started_at_epoch": time.time(),
    }
    state_path = (
        results / "02_manifests" / f"v2_recovery_{args.node_tag}.json")
    atomic_json(state_path, state)
    log_dir = results / "10_logs" / "v2_recovery"
    log_dir.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()

    def worker(gpu: str):
        while True:
            try:
                job = work.get_nowait()
            except queue.Empty:
                return
            preflight = dict(job)
            preflight["run_id"] = job["base_run_id"]
            actual_run_id, disposition = allocate_run(results, preflight)
            if disposition in {"success", "active"}:
                with lock:
                    target = (
                        state["skipped_success"] if disposition == "success"
                        else state["deferred_active"])
                    target.append(preflight)
                    state["remaining"] = work.qsize()
                    atomic_json(state_path, state)
                work.task_done()
                continue
            job["run_id"] = actual_run_id
            job["retry_of"] = (
                job["base_run_id"]
                if actual_run_id != job["base_run_id"] else None)
            command = [
                args.python, str(root / "tcff_rebuttal_runner.py"),
                "--project_root", str(root),
                "--results_root", str(results),
                "--experiment_id", job["experiment_id"],
                "--run_id", job["run_id"],
                "--exp_dir", job["exp_dir"],
                "--dataset_dir", job["dataset_dir"],
                "--dataset_name", job["dataset_name"],
                "--model_variant", job["model_variant"],
                "--corruption_type", job["corruption"],
                "--severity", str(job["severity"]),
                "--severity_unit", job["unit"],
                "--realization", str(job["realization"]),
                "--seed", str(job["seed"]),
                "--device", "cuda:0",
            ]
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
                log_dir / f"{job['experiment_id']}__{job['run_id']}.log")
            with log_path.open("w", encoding="utf-8", buffering=1) as log:
                process = subprocess.run(
                    command, cwd=root, env=env, stdout=log,
                    stderr=subprocess.STDOUT, check=False)
            run_dir = (
                results / "03_raw_runs" / job["experiment_id"] /
                job["run_id"])
            if run_dir.exists():
                (run_dir / "stdout.log").write_bytes(log_path.read_bytes())
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
            work.task_done()

    gpu_ids = [value.strip() for value in args.gpus.split(",") if value.strip()]
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
    print(json.dumps({
        "status": "success" if not state["failed"] else "failed",
        "expected_total": len(jobs),
        "queued_this_launch": len(queued),
        "skipped_success": len(state["skipped_success"]),
        "deferred_active": len(state["deferred_active"]),
        "completed": len(state["completed"]),
        "failed": len(state["failed"]),
    }, sort_keys=True))
    return 0 if not state["failed"] else 2


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
        help="Comma-separated subset: trained,product")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max_cases", type=int)
    raise SystemExit(main(parser.parse_args()))
