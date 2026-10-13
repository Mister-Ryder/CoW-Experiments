"""Collect confidence and entropy controls without changing hard predictions."""

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
CORRUPTIONS = (
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


def tag(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def build_jobs(root: Path, realizations: int):
    jobs = []
    for dataset_tag, dataset_name, directory in DATASETS:
        jobs.append({
            "run_id": f"full_probability_clean_{dataset_tag}",
            "dataset_tag": dataset_tag, "dataset_name": dataset_name,
            "dataset_dir": str(root / directory), "corruption": "none",
            "severity": 0.0, "unit": "none", "realization": 0,
        })
        for corruption, severity, unit in CORRUPTIONS:
            for realization in range(realizations):
                jobs.append({
                    "run_id": (f"full_probability_{corruption}_{tag(severity)}_"
                               f"r{realization}_{dataset_tag}"),
                    "dataset_tag": dataset_tag, "dataset_name": dataset_name,
                    "dataset_dir": str(root / directory),
                    "corruption": corruption, "severity": severity,
                    "unit": unit, "realization": realization,
                })
    return jobs


def main(args):
    root = Path(args.project_root).resolve()
    results = Path(args.results_root).resolve()
    jobs = build_jobs(root, args.realizations)
    work = queue.Queue()
    state = {
        "planned": len(jobs), "completed": [], "failed": [],
        "skipped_success": [], "started_at_epoch": time.time(),
    }
    for job in jobs:
        status_path = (results / "03_raw_runs" / "P0_B_PROBABILITY_SIGNALS" /
                       job["run_id"] / "status.json")
        if status_path.exists():
            status = json.loads(status_path.read_text(encoding="utf-8")).get("status")
            if status == "success":
                state["skipped_success"].append(job)
                continue
            state["failed"].append({**job,
                "reason": "existing non-success run preserved; manual retry required"})
            continue
        work.put(job)
    state["queued"] = work.qsize()
    state_path = results / "02_manifests" / "p0_probability_signal_matrix.json"
    lock = threading.Lock()
    atomic_json(state_path, state)

    def worker(gpu):
        while True:
            try:
                job = work.get_nowait()
            except queue.Empty:
                return
            command = [
                args.python, str(root / "tcff_rebuttal_runner.py"),
                "--project_root", str(root), "--results_root", str(results),
                "--experiment_id", "P0_B_PROBABILITY_SIGNALS",
                "--run_id", job["run_id"],
                "--exp_dir", str(root / "Runs_foundation" / "sam_tgf_v3"),
                "--dataset_dir", job["dataset_dir"],
                "--dataset_name", job["dataset_name"],
                "--model_variant", "full",
                "--corruption_type", job["corruption"],
                "--severity", str(job["severity"]),
                "--severity_unit", job["unit"],
                "--realization", str(job["realization"]),
                "--seed", "0", "--device", "cuda:0",
                "--save_probability_signals",
            ]
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu
            env["PYTHONPATH"] = f"{root / 'SAM-Med3D'}{os.pathsep}{root}"
            env["NUMEXPR_MAX_THREADS"] = "64"
            env["OMP_NUM_THREADS"] = "8"
            conda_lib = "/root/miniconda3/envs/cu122py39/lib"
            env["LD_LIBRARY_PATH"] = conda_lib + ":" + env.get("LD_LIBRARY_PATH", "")
            log_dir = results / "10_logs" / "probability_signal_matrix"
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / f"{job['run_id']}.log"
            with log_path.open("w", encoding="utf-8", buffering=1) as log:
                process = subprocess.run(command, cwd=root, env=env,
                                         stdout=log, stderr=subprocess.STDOUT,
                                         check=False)
            run_dir = results / "03_raw_runs" / "P0_B_PROBABILITY_SIGNALS" / job["run_id"]
            if run_dir.exists():
                (run_dir / "stdout.log").write_bytes(log_path.read_bytes())
                (run_dir / "stderr.log").touch(exist_ok=True)
            record = {**job, "gpu": gpu, "exit_code": process.returncode,
                      "finished_at_epoch": time.time()}
            with lock:
                target = state["completed"] if process.returncode == 0 else state["failed"]
                target.append(record)
                state["remaining"] = work.qsize()
                atomic_json(state_path, state)
                print(json.dumps(record, sort_keys=True), flush=True)
            work.task_done()

    gpu_ids = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    threads = [threading.Thread(target=worker, args=(gpu,), daemon=False)
               for gpu in gpu_ids]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    state["finished_at_epoch"] = time.time()
    atomic_json(state_path, state)
    print(json.dumps({
        "planned": len(jobs), "completed": len(state["completed"]),
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
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--realizations", type=int, default=3)
    raise SystemExit(main(parser.parse_args()))
