"""Independent, raw-evidence final audit for the TCFF rebuttal package."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ORIGINAL_FIXED_EXPERIMENTS = {
    "P0_AB_CLEAN_MECHANISM",
    "P0_D_FIXED_CKPT_INTERVENTIONS",
    "P0_C1_GAUSSIAN",
    "P0_C1_CHANNEL_DROPOUT",
    "P0_C2_TRANSLATION",
    "P0_C3_DONOR_MIX",
}
PRODUCT_FIXED_EXPERIMENTS = {
    "P0_D_PRODUCT_FIXED_CLEAN",
    "P0_D_PRODUCT_FIXED_CORRUPTION",
}
TRAINING_EXPERIMENTS = {"P0_D_TRAIN_VARIANTS"}
TRAINED_EVAL_EXPERIMENTS = {
    "P0_D_TRAINED_EVAL_CLEAN",
    "P0_D_TRAINED_EVAL_CORRUPTION",
}
PROBABILITY_EXPERIMENTS = {"P0_B_PROBABILITY_SIGNALS"}
SELECTED_MAP_EXPERIMENTS = {"P0_G_SELECTED_CASE_MAPS"}
FORMAL_INFERENCE_EXPERIMENTS = (
    ORIGINAL_FIXED_EXPERIMENTS
    | PRODUCT_FIXED_EXPERIMENTS
    | TRAINED_EVAL_EXPERIMENTS
    | PROBABILITY_EXPERIMENTS
    | SELECTED_MAP_EXPERIMENTS
)
EXPECTED = {
    "original_fixed_inference": 162,
    "original_training_jobs": 18,
    "original_trained_evaluations": 270,
    "probability_runs": 39,
    "product_fixed_inference": 147,
    "product_training_jobs": 3,
    "product_trained_evaluations": 45,
    "efficiency_rows": 6,
    "selected_map_runs": 3,
    "case_panels": 24,
    "case_panel_files": 48,
    "bootstrap_replicates": 2000,
}
TOLERANCE = 1e-5


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return default


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def acceptance_policy(manifest: Dict[str, Any]) -> str:
    intervention = manifest.get("intervention") or {}
    if intervention.get("gate_alpha") is not None:
        return "constant_acceptance"
    transform = intervention.get("gate_transform", "identity")
    if transform not in (None, "", "identity"):
        return f"gate_transform:{transform}"
    variant = manifest.get("model_variant")
    if variant == "full":
        return "submitted_full"
    if variant == "product":
        return "product"
    return str(variant or "unknown")


def discover_raw_runs(root: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    raw_root = (root / "03_raw_runs").resolve()
    for manifest_path in sorted(raw_root.glob("*/*/run_manifest.json")):
        run_dir = manifest_path.parent.resolve()
        if not inside(run_dir, raw_root):
            raise RuntimeError(f"Run outside raw root: {run_dir}")
        manifest = read_json(manifest_path, {}) or {}
        status = read_json(run_dir / "status.json", {}) or {}
        records.append({
            "experiment_id": str(manifest.get("experiment_id") or run_dir.parent.name),
            "run_id": str(manifest.get("run_id") or run_dir.name),
            "canonical_run_id": str(
                manifest.get("canonical_run_id") or manifest.get("run_id") or run_dir.name
            ),
            "model_variant": str(manifest.get("model_variant") or "unknown"),
            "manifest_status": manifest.get("status"),
            "status": status.get("status", manifest.get("status", "unknown")),
            "checkpoint_hash": manifest.get("checkpoint_hash"),
            "git_commit": manifest.get("git_commit"),
            "acceptance_policy": acceptance_policy(manifest),
            "manifest": manifest,
            "run_dir": run_dir,
        })
    return records


def scoped(records: Iterable[Dict[str, Any]], experiments: set[str]) -> List[Dict[str, Any]]:
    return [row for row in records if row["experiment_id"] in experiments]


def count_success(records: Iterable[Dict[str, Any]]) -> int:
    return sum(row["status"] == "success" for row in records)


def count_non_success(records: Iterable[Dict[str, Any]]) -> int:
    return sum(row["status"] != "success" for row in records)


def update_max(current: float, values: pd.Series) -> float:
    finite = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if finite.notna().any():
        return max(current, float(finite.max()))
    return current


def raw_signal_audit(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "runs_checked": 0,
        "rows_checked": 0,
        "input_range_failures": 0,
        "formula_T_gt_G_failures": 0,
        "stored_T_formula_failures": 0,
        "submitted_full_identity_failures": 0,
        "product_identity_failures": 0,
        "submitted_full_rows_checked": 0,
        "product_rows_checked": 0,
        "explicit_acceptance_intervention_rows": 0,
        "max_abs_stored_T_formula_residual": 0.0,
        "max_abs_submitted_full_identity_residual": 0.0,
        "max_abs_product_identity_residual": 0.0,
    }
    for row in records:
        if row["status"] != "success" or row["experiment_id"] not in FORMAL_INFERENCE_EXPERIMENTS:
            continue
        parquet_path = row["run_dir"] / "internal_signals.parquet"
        if not parquet_path.is_file():
            raise RuntimeError(f"Missing internal signals for successful run: {parquet_path}")
        names = set(pq.ParquetFile(parquet_path).schema_arrow.names)
        if not {"U", "G"} <= names:
            raise RuntimeError(f"Missing U/G in {parquet_path}")
        acceptance_column = "A_impl" if "A_impl" in names else "T"
        if acceptance_column not in names:
            raise RuntimeError(f"Missing executed acceptance in {parquet_path}")
        columns = ["U", "G", acceptance_column]
        if "T_formula" in names:
            columns.append("T_formula")
        frame = pd.read_parquet(parquet_path, columns=columns)
        u = pd.to_numeric(frame["U"], errors="coerce")
        g = pd.to_numeric(frame["G"], errors="coerce")
        a_impl = pd.to_numeric(frame[acceptance_column], errors="coerce")
        finite = np.isfinite(u) & np.isfinite(g) & np.isfinite(a_impl)
        formula = g * (1.0 - u)
        result["runs_checked"] += 1
        result["rows_checked"] += int(finite.sum())
        result["input_range_failures"] += int((
            finite
            & ((u < -TOLERANCE) | (u > 1.0 + TOLERANCE)
               | (g < -TOLERANCE) | (g > 1.0 + TOLERANCE))
        ).sum())
        result["formula_T_gt_G_failures"] += int(
            (finite & (formula > g + TOLERANCE)).sum()
        )
        if "T_formula" in frame:
            stored = pd.to_numeric(frame["T_formula"], errors="coerce")
            residual = (stored - formula).abs()
            result["stored_T_formula_failures"] += int(
                (finite & np.isfinite(stored) & (residual > TOLERANCE)).sum()
            )
            result["max_abs_stored_T_formula_residual"] = update_max(
                result["max_abs_stored_T_formula_residual"], residual[finite]
            )
        policy = row["acceptance_policy"]
        if policy == "submitted_full":
            expected = 1.0 - (1.0 - g) * u
            residual = (a_impl - expected).abs()
            result["submitted_full_rows_checked"] += int(finite.sum())
            result["submitted_full_identity_failures"] += int(
                (finite & (residual > TOLERANCE)).sum()
            )
            result["max_abs_submitted_full_identity_residual"] = update_max(
                result["max_abs_submitted_full_identity_residual"], residual[finite]
            )
        elif policy == "product":
            residual = (a_impl - formula).abs()
            result["product_rows_checked"] += int(finite.sum())
            result["product_identity_failures"] += int(
                (finite & (residual > TOLERANCE)).sum()
            )
            result["max_abs_product_identity_residual"] = update_max(
                result["max_abs_product_identity_residual"], residual[finite]
            )
        elif policy == "constant_acceptance" or policy.startswith("gate_transform:"):
            result["explicit_acceptance_intervention_rows"] += int(finite.sum())
    failures = [
        result["input_range_failures"],
        result["formula_T_gt_G_failures"],
        result["stored_T_formula_failures"],
        result["submitted_full_identity_failures"],
        result["product_identity_failures"],
    ]
    result["status"] = "pass" if not any(failures) else "fail"
    return result


def gate_status(path: Path) -> str:
    text = path.read_text(encoding="utf-8") if path.is_file() else ""
    return "PASS" if "Status: **PASS**" in text else "FAIL"


def main(args: argparse.Namespace) -> int:
    root = Path(args.results_root).resolve()
    records = discover_raw_runs(root)
    original_fixed = scoped(records, ORIGINAL_FIXED_EXPERIMENTS)
    product_fixed = scoped(records, PRODUCT_FIXED_EXPERIMENTS)
    training = scoped(records, TRAINING_EXPERIMENTS)
    trained_eval = scoped(records, TRAINED_EVAL_EXPERIMENTS)
    probability = scoped(records, PROBABILITY_EXPERIMENTS)
    selected_maps = scoped(records, SELECTED_MAP_EXPERIMENTS)
    selected_map_state = read_json(
        root / "02_manifests" / "p0_selected_case_maps.json", {}
    ) or {}
    planned_map_run_ids = {
        str(item.get("run_id"))
        for key in ["completed", "skipped_success"]
        for item in (selected_map_state.get(key) or [])
        if item.get("run_id")
    }
    planned_selected_maps = [
        row for row in selected_maps if row["run_id"] in planned_map_run_ids
    ]
    supplemental_selected_maps = [
        row for row in selected_maps if row["run_id"] not in planned_map_run_ids
    ]

    original_training = [r for r in training if r["model_variant"] != "product"]
    product_training = [r for r in training if r["model_variant"] == "product"]
    original_eval = [r for r in trained_eval if r["model_variant"] != "product"]
    product_eval = [r for r in trained_eval if r["model_variant"] == "product"]
    groups = {
        "original_fixed_inference": original_fixed,
        "original_training_jobs": original_training,
        "original_trained_evaluations": original_eval,
        "probability_runs": probability,
        "product_fixed_inference": product_fixed,
        "product_training_jobs": product_training,
        "product_trained_evaluations": product_eval,
        "selected_map_runs": planned_selected_maps,
    }
    counts = {name: count_success(rows) for name, rows in groups.items()}
    non_success = {name: count_non_success(rows) for name, rows in groups.items()}
    duplicate_identities = len(records) - len({
        (row["experiment_id"], row["run_id"]) for row in records
    })
    status_manifest_mismatches = sum(
        row["manifest_status"] is not None and row["manifest_status"] != row["status"]
        for row in records
    )
    raw_status_counts: Dict[str, int] = {}
    for row in records:
        raw_status_counts[row["status"]] = raw_status_counts.get(row["status"], 0) + 1

    efficiency_path = root / "09_efficiency" / "efficiency_summary.csv"
    efficiency = pd.read_csv(efficiency_path) if efficiency_path.is_file() else pd.DataFrame()
    counts["efficiency_rows"] = len(efficiency)

    panel_path = root / "08_failure_cases" / "case_panel_index.csv"
    panels = pd.read_csv(panel_path) if panel_path.is_file() else pd.DataFrame()
    successful_panels = panels[panels.get("status", pd.Series(dtype=str)).eq("success")]
    counts["case_panels"] = len(successful_panels)
    panel_files = []
    panel_file_failures = []
    for row in successful_panels.itertuples(index=False):
        for column in ["png", "pdf"]:
            relative = str(getattr(row, column, ""))
            path = (root / relative).resolve()
            valid = bool(relative and inside(path, root) and path.is_file() and path.stat().st_size > 0)
            panel_files.append({"path": relative, "valid": valid})
            if not valid:
                panel_file_failures.append(relative)
    counts["case_panel_files"] = sum(item["valid"] for item in panel_files)

    map_file_failures = []
    for row in selected_maps:
        if row["status"] != "success":
            continue
        maps = list((row["run_dir"] / "maps").glob("*.npz"))
        predictions = list((row["run_dir"] / "predictions").glob("*.npz"))
        if not maps or not predictions:
            map_file_failures.append(str(row["run_dir"]))

    post_manifest = read_json(root / "02_manifests" / "postprocessing_manifest.json", {}) or {}
    finalization = read_json(root / "02_manifests" / "finalization_status.json", {}) or {}
    quote_verification = read_json(
        root / "11_quality_audit" / "quote_table_verification.json", {}
    ) or {}
    semantic_audit = read_json(
        root / "11_quality_audit" / "TRUST_SIGNAL_SEMANTICS_AUDIT.json", {}
    ) or {}
    counts["bootstrap_replicates"] = int(post_manifest.get("bootstrap_replicates", 0) or 0)

    ledger = root / "06_summary_tables" / "exact_numbers_safe_to_quote.csv"
    ledger_rows = len(pd.read_csv(ledger)) if ledger.is_file() else 0
    ledger_sha = sha256_file(ledger) if ledger.is_file() else None
    quote_check = {
        "status": quote_verification.get("status"),
        "n_claims": quote_verification.get("n_claims"),
        "ledger_rows": ledger_rows,
        "n_failed": quote_verification.get("n_failed"),
        "root_ledger_matches": quote_verification.get("root_ledger_matches"),
        "ledger_sha256": ledger_sha,
        "verification_ledger_sha256": quote_verification.get("ledger_sha256"),
        "pass": bool(
            quote_verification.get("status") == "pass"
            and int(quote_verification.get("n_failed", -1)) == 0
            and quote_verification.get("root_ledger_matches") is True
            and int(quote_verification.get("n_claims", -1)) == ledger_rows
            and ledger_sha == quote_verification.get("ledger_sha256")
        ),
    }

    findings_path = root / "11_quality_audit" / "data_quality_findings.csv"
    findings = pd.read_csv(findings_path) if findings_path.is_file() else pd.DataFrame()
    high_critical_failures = findings[
        findings.get("severity", pd.Series(dtype=str)).isin(["critical", "high"])
        & findings.get("status", pd.Series(dtype=str)).eq("fail")
    ] if not findings.empty else pd.DataFrame()

    gates = {
        name: gate_status(root / "02_manifests" / f"{name}_ACCEPTANCE.md")
        for name in ["G2", "G3", "G5"]
    }
    signal_audit = raw_signal_audit(records)

    fixed_hashes = {
        row["checkpoint_hash"] for row in original_fixed + product_fixed
        if row["status"] == "success" and row["checkpoint_hash"]
    }
    code_checks = {
        "finalizer_sha_matches": bool(
            finalization.get("finalizer_sha256")
            == sha256_file(Path(args.project_root).resolve() / "tcff_finalize_when_complete.py")
        ),
        "postprocess_sha_matches": bool(
            post_manifest.get("script_sha256")
            == sha256_file(Path(args.project_root).resolve() / "tcff_postprocess.py")
        ),
    }

    failures: List[str] = []
    for name, expected in EXPECTED.items():
        if counts.get(name) != expected:
            failures.append(f"{name}: expected {expected}, got {counts.get(name)}")
    for name, value in non_success.items():
        if value:
            failures.append(f"{name}: {value} non-success raw runs")
    if duplicate_identities:
        failures.append(f"duplicate experiment/run identities: {duplicate_identities}")
    if status_manifest_mismatches:
        failures.append(f"status/manifest mismatches: {status_manifest_mismatches}")
    if map_file_failures:
        failures.append(f"selected-map file failures: {len(map_file_failures)}")
    if int(selected_map_state.get("planned", 0) or 0) != EXPECTED["selected_map_runs"]:
        failures.append(
            "selected-map manifest planned count: "
            f"{selected_map_state.get('planned')}"
        )
    if len(planned_map_run_ids) != EXPECTED["selected_map_runs"]:
        failures.append(f"selected-map planned run identities: {len(planned_map_run_ids)}")
    supplemental_map_non_success = count_non_success(supplemental_selected_maps)
    if supplemental_map_non_success:
        failures.append(
            f"supplemental formula-panel map runs non-success: {supplemental_map_non_success}"
        )
    if panel_file_failures:
        failures.append(f"panel file failures: {len(panel_file_failures)}")
    if len(fixed_hashes) != 1:
        failures.append(f"fixed intervention checkpoint hashes: {len(fixed_hashes)}")
    if any(value != "PASS" for value in gates.values()):
        failures.append(f"decision gates: {gates}")
    if not high_critical_failures.empty:
        failures.append(f"high/critical audit failures: {len(high_critical_failures)}")
    if semantic_audit.get("status") != "pass":
        failures.append("derived trust-semantics audit is not pass")
    if signal_audit.get("status") != "pass":
        failures.append("independent raw signal audit is not pass")
    if not quote_check["pass"]:
        failures.append("quote ledger verification mismatch")
    if finalization.get("status") != "success":
        failures.append("finalization status is not success")
    if not all(code_checks.values()):
        failures.append(f"code hash checks: {code_checks}")

    report = {
        "status": "pass" if not failures else "fail",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "results_root": str(root),
        "expected": EXPECTED,
        "successful_counts": counts,
        "non_success_counts": non_success,
        "raw_run_inventory": {
            "total": len(records),
            "status_counts": raw_status_counts,
            "duplicate_identities": duplicate_identities,
            "status_manifest_mismatches": status_manifest_mismatches,
        },
        "selected_maps": {
            "plan_declared": int(selected_map_state.get("planned", 0) or 0),
            "planned_run_ids": sorted(planned_map_run_ids),
            "planned_successful_runs": count_success(planned_selected_maps),
            "supplemental_successful_runs_for_formula_panels": count_success(
                supplemental_selected_maps
            ),
            "supplemental_run_ids": sorted(
                row["run_id"] for row in supplemental_selected_maps
                if row["status"] == "success"
            ),
        },
        "fixed_checkpoint_hash_count": len(fixed_hashes),
        "decision_gates": gates,
        "high_critical_audit_failures": len(high_critical_failures),
        "derived_semantics_status": semantic_audit.get("status"),
        "raw_signal_audit": signal_audit,
        "quote_ledger": quote_check,
        "code_hash_checks": code_checks,
        "selected_map_file_failures": map_file_failures,
        "case_panel_file_failures": panel_file_failures,
        "failures": failures,
    }
    if args.output:
        atomic_json(Path(args.output).resolve(), report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0 if not failures else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    parser.add_argument("--project_root", default=str(here))
    parser.add_argument("--results_root", required=True)
    parser.add_argument("--output")
    raise SystemExit(main(parser.parse_args()))
