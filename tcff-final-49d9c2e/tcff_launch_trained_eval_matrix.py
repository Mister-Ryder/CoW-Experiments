"""Evaluate independently trained TCFF variants on clean and shifted data."""

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


def build_runs(root: Path, results: Path, model_job: dict,
               corruption_realizations: int):
    variant = model_job["variant"]
    seed = model_job["seed"]
    training_run = results / "03_raw_runs" / "P0_D_TRAIN_VARIANTS" / f"{variant}_seed{seed}"
    training_status = json.loads((training_run / "status.json").read_text(encoding="utf-8"))
    if training_status.get("status") != "success":
        raise RuntimeError(f"training run not successful: {training_run}")
    exp_dir = training_run / "training_output" / "model"
    if not (exp_dir / "best_model.pth").exists():
        raise FileNotFoundError(exp_dir / "best_model.pth")
    runs = []
    for tag, label, directory in DATASETS:
        runs.append({
            "experiment_id": "P0_D_TRAINED_EVAL_CLEAN",
            "run_id": f"{variant}_seed{seed}_clean_{tag}",
            "exp_dir": str(exp_dir), "dataset_dir": str(root / directory),
            "dataset_name": label, "dataset_tag": tag,
            "variant": variant, "seed": seed,
            "corruption": "none", "severity": 0.0,
            "severity_unit": "none", "realization": 0,
        })
    # Robustness comparisons are source-validation only; fixed-checkpoint runs
    # supply the full three-domain severity chains.
    for corruption, severity, unit in REPRESENTATIVE_CORRUPTIONS:
        for realization in range(corruption_realizations):
            runs.append({
                "experiment_id": "P0_D_TRAINED_EVAL_CORRUPTION",
                "run_id": (f"{variant}_seed{seed}_{corruption}_"
                           f"{severity_tag(severity)}_r{realization}_crown"),
                "exp_dir": str(exp_dir),
                "dataset_dir": str(root / "crown_processed"),
                "dataset_name": "TopCoW_val", "dataset_tag": "crown",
                "variant": variant, "seed": seed,
                "corruption": corruption, "severity": severity,
                "severity_unit": unit, "realization": realization,
            })
    return runs


def main(args):
    root = Path(args.project_root).resolve()
    results = Path(args.results_root).resolve()
    assignments = parse_assignments(args.assignments)
    state_path = (results / "02_manifests" /
                  f"p0_trained_eval_matrix_{args.node_tag}.json")
    state = {
        "node_tag": args.node_tag, "assignments": assignments,
        "corruption_realizations": args.corruption_realizations,
        "completed": [], "failed": [], "skipped_success": [],
        "started_at_epoch": time.time(),
    }
    lock = threading.Lock()
    atomic_json(state_path, state)

    def evaluate_model(model_job):
        try:
            runs = build_runs(root, results, model_job,
                              args.corruption_realizations)
        except Exception as exc:
            with lock:
                state["failed"].append({**model_job, "stage": "build_runs",
                                        "reason": f"{type(exc).__name__}: {exc}"})
                atomic_json(state_path, state)
            return
        for run in runs:
            run_dir = results / "03_raw_runs" / run["experiment_id"] / run["run_id"]
            status_path = run_dir / "status.json"
            if status_path.exists():
                status = json.loads(status_path.read_text(encoding="utf-8")).get("status")
                if status == "success":
                    with lock:
                        state["skipped_success"].append(run)
                        atomic_json(state_path, state)
                    continue
                with lock:
                    state["failed"].append({**run,
                        "reason": "existing non-success run preserved; manual retry required"})
                    atomic_json(state_path, state)
                continue
            command = [
                args.python, str(root / "tcff_rebuttal_runner.py"),
                "--project_root", str(root), "--results_root", str(results),
                "--experiment_id", run["experiment_id"],
                "--run_id", run["run_id"], "--exp_dir", run["exp_dir"],
                "--dataset_dir", run["dataset_dir"],
                "--dataset_name", run["dataset_name"],
                "--model_variant", run["variant"],
                "--corruption_type", run["corruption"],
                "--severity", str(run["severity"]),
                "--severity_unit", run["severity_unit"],
                "--realization", str(run["realization"]),
                "--seed", str(run["seed"]), "--device", "cuda:0",
            ]
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
            with log_path.open("w", encoding="utf-8", buffering=1) as log:
                process = subprocess.run(command, cwd=root, env=env,
                                         stdout=log, stderr=subprocess.STDOUT,
                                         check=False)
            if run_dir.exists():
                (run_dir / "stdout.log").write_bytes(log_path.read_bytes())
                (run_dir / "stderr.log").touch(exist_ok=True)
            record = {**run, "gpu": model_job["gpu"],
                      "exit_code": process.returncode,
                      "finished_at_epoch": time.time()}
            with lock:
                target = state["completed"] if process.returncode == 0 else state["failed"]
                target.append(record)
                atomic_json(state_path, state)
                print(json.dumps(record, sort_keys=True), flush=True)

    threads = [threading.Thread(target=evaluate_model, args=(job,), daemon=False)
               for job in assignments]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    state["finished_at_epoch"] = time.time()
    atomic_json(state_path, state)
    print(json.dumps({
        "completed": len(state["completed"]),
        "failed": len(state["failed"]),
        "skipped_success": len(state["skipped_success"]),
    }, sort_keys=True))
    return 0 if not state["failed"] else 2


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
    raise SystemExit(main(parser.parse_args()))
