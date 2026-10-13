"""Wait for every preregistered TCFF evidence stream, then finalize safely.

This operational guard is deliberately stricter than ``tcff_postprocess.py
--strict``: it also requires probability-signal runs, selected-case maps, and
rendered case panels before the final 2,000-replicate analysis is allowed to
start.  It never retries or overwrites a failed experimental run.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict, Iterable, Tuple


EXPECTED_FIXED_RUNS = 162
EXPECTED_PROBABILITY_RUNS = 39
EXPECTED_TRAINING = {
    "nodeA4": 4,
    "nodeB8": 8,
    "nodeB6_second": 6,
    "nodeA3_product": 3,
}
EXPECTED_TRAINED_EVAL = {
    "nodeA4_followup": 60,
    "nodeB8_followup": 120,
    "nodeB6_second_followup": 90,
    "nodeA3_product_followup": 45,
}
EXPECTED_PRODUCT_FIXED_RUNS = 147


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def state_counts(state: Dict[str, Any], include_skipped: bool = False) -> Tuple[int, int]:
    completed = len(state.get("completed", []))
    if include_skipped:
        completed += len(state.get("skipped_success", []))
    return completed, len(state.get("failed", []))


def inspect_matrix(
    path: Path,
    expected: int,
    *,
    planned_key: str = "planned",
    include_skipped: bool = False,
) -> Dict[str, Any]:
    state = read_json(path)
    completed, failed = state_counts(state, include_skipped=include_skipped)
    planned = int(state.get(planned_key, 0) or 0)
    return {
        "path": path.as_posix(),
        "exists": path.exists(),
        "expected": expected,
        "planned": planned,
        "completed": completed,
        "failed": failed,
        "ready": bool(path.exists() and planned == expected and completed == expected and failed == 0),
        "blocked": bool(failed),
    }


def inspect_training(path: Path, expected: int) -> Dict[str, Any]:
    state = read_json(path)
    completed, failed = state_counts(state)
    planned = len(state.get("jobs", []))
    return {
        "path": path.as_posix(),
        "exists": path.exists(),
        "expected": expected,
        "planned": planned,
        "completed": completed,
        "failed": failed,
        "finished": "finished_at_epoch" in state,
        "ready": bool(path.exists() and planned == expected and completed == expected and failed == 0 and "finished_at_epoch" in state),
        "blocked": bool(failed),
    }


def inspect_trained_eval(path: Path, expected: int) -> Dict[str, Any]:
    state = read_json(path)
    completed, failed = state_counts(state, include_skipped=True)
    assignments = len(state.get("assignments", []))
    realizations = int(state.get("corruption_realizations", 3) or 3)
    planned = assignments * (3 + 4 * realizations)
    return {
        "path": path.as_posix(),
        "exists": path.exists(),
        "expected": expected,
        "planned": planned,
        "completed": completed,
        "failed": failed,
        "finished": "finished_at_epoch" in state,
        "ready": bool(path.exists() and planned == expected and completed == expected and failed == 0 and "finished_at_epoch" in state),
        "blocked": bool(failed),
    }


def inspect_efficiency(results: Path) -> Dict[str, Any]:
    summary = results / "09_efficiency" / "efficiency_summary.csv"
    manifest_path = results / "09_efficiency" / "efficiency_manifest.json"
    manifest = read_json(manifest_path)
    rows = 0
    if summary.exists():
        with summary.open("r", encoding="utf-8", newline="") as handle:
            rows = sum(1 for _ in csv.DictReader(handle))
    ready = bool(
        rows == 6
        and int(manifest.get("warmup", 0) or 0) >= 10
        and int(manifest.get("repeats", 0) or 0) >= 30
        and manifest.get("gpu_name")
    )
    return {
        "summary": summary.as_posix(),
        "manifest": manifest_path.as_posix(),
        "rows": rows,
        "warmup": manifest.get("warmup"),
        "repeats": manifest.get("repeats"),
        "gpu_name": manifest.get("gpu_name"),
        "ready": ready,
        "blocked": False,
    }


def inspect_panels(results: Path) -> Dict[str, Any]:
    index_path = results / "08_failure_cases" / "case_panel_index.csv"
    rows = []
    if index_path.exists():
        with index_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    success = sum(row.get("status") == "success" for row in rows)
    failed = sum(row.get("status") != "success" for row in rows)
    return {
        "path": index_path.as_posix(),
        "rows": len(rows),
        "successful": success,
        "failed": failed,
        "ready": bool(rows and success == len(rows) and failed == 0),
        "blocked": bool(failed),
    }


def inspect_g1(results: Path) -> Dict[str, Any]:
    path = results / "11_quality_audit" / "G1_PROBABILITY_AND_MAP_INTERFACE.json"
    report = read_json(path)
    ready = path.exists() and report.get("status") == "pass"
    return {
        "path": path.as_posix(),
        "status": report.get("status"),
        "ready": ready,
        "blocked": bool(path.exists() and not ready),
    }


def inspect_all(results: Path) -> Dict[str, Any]:
    manifests = results / "02_manifests"
    checks: Dict[str, Dict[str, Any]] = {}
    checks["fixed_inference"] = inspect_matrix(
        manifests / "p0_inference_matrix_status.json", EXPECTED_FIXED_RUNS
    )
    checks["product_fixed_intervention"] = inspect_matrix(
        manifests / "p0_product_intervention_matrix_status.json",
        EXPECTED_PRODUCT_FIXED_RUNS,
    )
    checks["probability_signals"] = inspect_matrix(
        manifests / "p0_probability_signal_matrix.json",
        EXPECTED_PROBABILITY_RUNS,
        include_skipped=True,
    )
    for tag, expected in EXPECTED_TRAINING.items():
        checks[f"training_{tag}"] = inspect_training(
            manifests / f"p0_training_matrix_{tag}.json", expected
        )
    for tag, expected in EXPECTED_TRAINED_EVAL.items():
        checks[f"trained_eval_{tag}"] = inspect_trained_eval(
            manifests / f"p0_trained_eval_matrix_{tag}.json", expected
        )
    selected = read_json(manifests / "p0_selected_case_maps.json")
    selected_expected = int(selected.get("planned", 0) or 0)
    checks["selected_case_maps"] = inspect_matrix(
        manifests / "p0_selected_case_maps.json",
        selected_expected,
        include_skipped=True,
    )
    if selected_expected <= 0:
        checks["selected_case_maps"]["ready"] = False
    checks["case_panels"] = inspect_panels(results)
    checks["efficiency"] = inspect_efficiency(results)
    checks["g1_probability_map_interface"] = inspect_g1(results)
    return {
        "checked_at": utc_now(),
        "ready": all(check["ready"] for check in checks.values()),
        "blocked": any(check["blocked"] for check in checks.values()),
        "checks": checks,
    }


def run_logged(command: Iterable[str], cwd: Path, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", buffering=1) as log:
        log.write(f"\n[{utc_now()}] RUN {' '.join(command)}\n")
        process = subprocess.run(
            list(command), cwd=cwd, stdout=log, stderr=subprocess.STDOUT, check=False
        )
        log.write(f"[{utc_now()}] EXIT {process.returncode}\n")
    return process.returncode


def finalize(args: argparse.Namespace, readiness: Dict[str, Any]) -> int:
    root = Path(args.project_root).resolve()
    results = Path(args.results_root).resolve()
    log_path = results / "10_logs" / "finalization.log"
    status_path = results / "02_manifests" / "finalization_status.json"
    script_path = Path(__file__).resolve()
    status: Dict[str, Any] = {
        "status": "running",
        "started_at": utc_now(),
        "project_root": root.as_posix(),
        "results_root": results.as_posix(),
        "finalizer_sha256": sha256_file(script_path),
        "readiness": readiness,
        "bootstrap_replicates": 2000,
    }
    atomic_json(status_path, status)
    postprocess = [
        args.python,
        str(root / "tcff_postprocess.py"),
        "--results_root",
        str(results),
        "--bootstrap",
        "2000",
        "--strict",
    ]
    verify = [
        args.python,
        str(root / "tcff_verify_quote_table.py"),
        "--results_root",
        str(results),
    ]
    post_exit = run_logged(postprocess, root, log_path)
    verify_exit = run_logged(verify, root, log_path) if post_exit == 0 else None
    verification = read_json(
        results / "11_quality_audit" / "quote_table_verification.json"
    )
    passed = bool(
        post_exit == 0
        and verify_exit == 0
        and verification.get("status") == "pass"
        and verification.get("root_ledger_matches") is True
    )
    status.update(
        {
            "status": "success" if passed else "failed",
            "finished_at": utc_now(),
            "postprocess_exit_code": post_exit,
            "quote_verifier_exit_code": verify_exit,
            "quote_verification": verification,
        }
    )
    atomic_json(status_path, status)
    return 0 if passed else 2


def main(args: argparse.Namespace) -> int:
    results = Path(args.results_root).resolve()
    start = time.monotonic()
    while True:
        state = inspect_all(results)
        compact = {
            "checked_at": state["checked_at"],
            "ready": state["ready"],
            "blocked": state["blocked"],
            "progress": {
                name: {
                    key: value
                    for key, value in check.items()
                    if key in {"planned", "completed", "failed", "rows", "successful", "ready"}
                }
                for name, check in state["checks"].items()
            },
        }
        print(json.dumps(compact, ensure_ascii=False, sort_keys=True), flush=True)
        if state["blocked"]:
            return 3
        if state["ready"]:
            return finalize(args, state) if args.finalize else 0
        if args.once:
            return 1
        if time.monotonic() - start >= args.timeout_hours * 3600:
            return 4
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", default=str(here))
    parser.add_argument(
        "--results_root", default=str(here / "results_tcff_rebuttal")
    )
    parser.add_argument(
        "--python", default="/root/autodl-tmp/tcff-venv/bin/python"
    )
    parser.add_argument("--poll_seconds", type=float, default=30.0)
    parser.add_argument("--timeout_hours", type=float, default=12.0)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--finalize", action="store_true")
    raise SystemExit(main(parser.parse_args()))
