"""Rerun preregistered selected cases to persist predictions and trust maps."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import queue
import subprocess
import threading
import time

import pandas as pd


DATASETS = {
    "TopCoW_val": ("crown", "crown_processed"),
    "Lausanne": ("lausanne", "lausanne_processed"),
    "IXI_HH": ("ixi", "ixi_processed"),
}


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def main(args):
    root = Path(args.project_root).resolve()
    results = Path(args.results_root).resolve()
    selection_path = results / "08_failure_cases" / "predeclared_case_selection.csv"
    selection = pd.read_csv(selection_path)
    selection = selection[selection["availability"].eq("selected")]
    jobs = []
    for dataset, (tag, directory) in DATASETS.items():
        case_ids = sorted(selection[selection["dataset"].eq(dataset)]["case_id"]
                          .dropna().astype(str).unique().tolist())
        if not case_ids:
            continue
        jobs.append({
            "dataset": dataset, "dataset_tag": tag,
            "dataset_dir": str(root / directory), "case_ids": case_ids,
            "run_id": f"full_selected_maps_{tag}",
        })
    work = queue.Queue()
    state = {"planned": len(jobs), "completed": [], "failed": [],
             "skipped_success": [], "started_at_epoch": time.time()}
    for job in jobs:
        status_path = (results / "03_raw_runs" / "P0_G_SELECTED_CASE_MAPS" /
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
    state_path = results / "02_manifests" / "p0_selected_case_maps.json"
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
                "--experiment_id", "P0_G_SELECTED_CASE_MAPS",
                "--run_id", job["run_id"],
                "--exp_dir", str(root / "Runs_foundation" / "sam_tgf_v3"),
                "--dataset_dir", job["dataset_dir"],
                "--dataset_name", job["dataset"],
                "--model_variant", "full", "--corruption_type", "none",
                "--severity", "0", "--severity_unit", "none",
                "--realization", "0", "--seed", "0", "--device", "cuda:0",
                "--case_ids", ",".join(job["case_ids"]),
                "--save_predictions", "--save_probability_signals",
                "--save_signal_maps",
            ]
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu
            env["PYTHONPATH"] = f"{root / 'SAM-Med3D'}{os.pathsep}{root}"
            env["NUMEXPR_MAX_THREADS"] = "64"
            env["OMP_NUM_THREADS"] = "8"
            conda_lib = "/root/miniconda3/envs/cu122py39/lib"
            env["LD_LIBRARY_PATH"] = conda_lib + ":" + env.get("LD_LIBRARY_PATH", "")
            log_dir = results / "10_logs" / "selected_case_maps"
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / f"{job['run_id']}.log"
            with log_path.open("w", encoding="utf-8", buffering=1) as log:
                process = subprocess.run(command, cwd=root, env=env,
                                         stdout=log, stderr=subprocess.STDOUT,
                                         check=False)
            run_dir = results / "03_raw_runs" / "P0_G_SELECTED_CASE_MAPS" / job["run_id"]
            if run_dir.exists():
                (run_dir / "stdout.log").write_bytes(log_path.read_bytes())
                (run_dir / "stderr.log").touch(exist_ok=True)
            record = {**job, "gpu": gpu, "exit_code": process.returncode,
                      "finished_at_epoch": time.time()}
            with lock:
                target = state["completed"] if process.returncode == 0 else state["failed"]
                target.append(record)
                atomic_json(state_path, state)
                print(json.dumps(record, sort_keys=True), flush=True)
            work.task_done()

    gpu_ids = [value.strip() for value in args.gpus.split(",") if value.strip()]
    threads = [threading.Thread(target=worker, args=(gpu,), daemon=False)
               for gpu in gpu_ids]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    state["finished_at_epoch"] = time.time()
    atomic_json(state_path, state)
    print(json.dumps({"completed": len(state["completed"]),
                      "failed": len(state["failed"]),
                      "skipped_success": len(state["skipped_success"])},
                     sort_keys=True))
    return 0 if not state["failed"] else 2


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", default=str(here))
    parser.add_argument("--results_root", default=str(here / "results_tcff_rebuttal"))
    parser.add_argument("--python", default="/root/autodl-tmp/tcff-venv/bin/python")
    parser.add_argument("--gpus", default="0,1,2")
    raise SystemExit(main(parser.parse_args()))
