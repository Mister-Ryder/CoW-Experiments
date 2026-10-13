"""One-case gate for probability summaries and persisted TCFF signal maps."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True,
                                    ensure_ascii=False, default=str),
                         encoding="utf-8")
    temporary.replace(path)


def run_once(command, cwd: Path, env: dict, log_path: Path,
             status_path: Path) -> None:
    if status_path.exists():
        status = read_json(status_path).get("status")
        if status == "success":
            return
        raise RuntimeError(
            f"Existing non-success G1 run is preserved and requires a new tag: {status_path}")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", buffering=1) as log:
        process = subprocess.run(command, cwd=cwd, env=env, stdout=log,
                                 stderr=subprocess.STDOUT, check=False)
    if process.returncode:
        raise RuntimeError(
            f"G1 subprocess failed with exit {process.returncode}; see {log_path}")


def check_finite_range(frame: pd.DataFrame, column: str,
                       low: float, high: float) -> None:
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise AssertionError(f"{column} contains non-finite values")
    if (values < low).any() or (values > high).any():
        raise AssertionError(f"{column} outside [{low}, {high}]")


def main(args: argparse.Namespace) -> int:
    root = Path(args.project_root).resolve()
    results = Path(args.results_root).resolve()
    experiment = "G1_PROBABILITY_INTERFACE"
    control_id = f"{args.tag}_control"
    probability_id = f"{args.tag}_probability_maps"
    base = results / "03_raw_runs" / experiment
    control_dir = base / control_id
    probability_dir = base / probability_id

    common = [
        args.python, str(root / "tcff_rebuttal_runner.py"),
        "--project_root", str(root), "--results_root", str(results),
        "--experiment_id", experiment,
        "--exp_dir", str(root / "Runs_foundation" / "sam_tgf_v3"),
        "--dataset_dir", str(root / "crown_processed"),
        "--dataset_name", "TopCoW_val", "--model_variant", "full",
        "--corruption_type", "none", "--severity", "0",
        "--severity_unit", "none", "--realization", "0", "--seed", "0",
        "--max_cases", "1", "--device", "cuda:0", "--fail_fast",
        "--save_predictions",
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["PYTHONPATH"] = f"{root / 'SAM-Med3D'}{os.pathsep}{root}"
    env["NUMEXPR_MAX_THREADS"] = "64"
    env["OMP_NUM_THREADS"] = "8"
    conda_lib = "/root/miniconda3/envs/cu122py39/lib"
    env["LD_LIBRARY_PATH"] = conda_lib + ":" + env.get("LD_LIBRARY_PATH", "")
    log_dir = results / "10_logs" / "g1_followup"
    run_once(common + ["--run_id", control_id], root, env,
             log_dir / f"{control_id}.log", control_dir / "status.json")
    run_once(common + [
        "--run_id", probability_id, "--save_probability_signals",
        "--save_signal_maps"], root, env,
        log_dir / f"{probability_id}.log", probability_dir / "status.json")

    control_status = read_json(control_dir / "status.json")
    probability_status = read_json(probability_dir / "status.json")
    if control_status.get("status") != "success" or probability_status.get("status") != "success":
        raise AssertionError("G1 status is not success")
    control_cases = read_json(control_dir / "selected_cases.json")
    probability_cases = read_json(probability_dir / "selected_cases.json")
    if control_cases != probability_cases or len(control_cases) != 1:
        raise AssertionError("G1 case selection mismatch")
    case_id = control_cases[0]

    control_case = pd.read_csv(control_dir / "case_metrics.csv")
    probability_case = pd.read_csv(probability_dir / "case_metrics.csv")
    probability_region = pd.read_parquet(probability_dir / "region_metrics.parquet")
    required_case = {"mean_confidence", "mean_entropy", "prediction_error_rate"}
    required_region = {"confidence", "entropy", "prediction_error_rate"}
    if not required_case <= set(probability_case):
        raise AssertionError(f"Missing case columns: {sorted(required_case - set(probability_case))}")
    if not required_region <= set(probability_region):
        raise AssertionError(
            f"Missing region columns: {sorted(required_region - set(probability_region))}")
    for column in required_case:
        check_finite_range(probability_case, column, 0.0, 1.0)
    for column in required_region:
        check_finite_range(probability_region, column, 0.0, 1.0)

    control_prediction = np.load(
        control_dir / "predictions" / f"{case_id}.npz")
    probability_prediction = np.load(
        probability_dir / "predictions" / f"{case_id}.npz")
    prediction_keys = ["target", "no_aux", "residual", "gt"]
    for key in prediction_keys:
        if not np.array_equal(control_prediction[key], probability_prediction[key]):
            raise AssertionError(f"Hard prediction mismatch for {key}")
    if (control_case.loc[0, "prediction_sha256"] !=
            probability_case.loc[0, "prediction_sha256"]):
        raise AssertionError("Prediction SHA mismatch")

    map_path = probability_dir / "maps" / f"{case_id}.npz"
    maps = np.load(map_path)
    expected_map_keys = {"confidence", "entropy"}
    for level in ["shallow", "mid", "deep"]:
        expected_map_keys.update({
            f"{level}_U", f"{level}_G", f"{level}_T", f"{level}_FDD",
            f"{level}_accepted_delta_l1",
        })
    missing_maps = sorted(expected_map_keys - set(maps.files))
    if missing_maps:
        raise AssertionError(f"Missing signal maps: {missing_maps}")
    for key in expected_map_keys:
        if not np.isfinite(maps[key]).all():
            raise AssertionError(f"Non-finite signal map: {key}")
    for key in ["confidence", "entropy"]:
        if maps[key].min() < 0 or maps[key].max() > 1:
            raise AssertionError(f"Probability signal map outside [0,1]: {key}")

    report = {
        "status": "pass",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "experiment_id": experiment,
        "control_run_id": control_id,
        "probability_run_id": probability_id,
        "case_id": case_id,
        "prediction_sha256": str(probability_case.loc[0, "prediction_sha256"]),
        "case_probability_columns": sorted(required_case),
        "region_probability_columns": sorted(required_region),
        "signal_map_keys": sorted(expected_map_keys),
        "n_region_rows": int(len(probability_region)),
        "gpu": str(args.gpu),
    }
    atomic_json(results / "11_quality_audit" /
                "G1_PROBABILITY_AND_MAP_INTERFACE.json", report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", default=str(here))
    parser.add_argument("--results_root", default=str(here / "results_tcff_rebuttal"))
    parser.add_argument("--python", default="/root/autodl-tmp/tcff-venv/bin/python")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--tag", default="g1_probability_v1")
    raise SystemExit(main(parser.parse_args()))
