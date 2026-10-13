"""Audit, aggregate, analyse, and plot the TCFF rebuttal experiment package.

The script is intentionally read-only with respect to raw runs.  Every derived
file is written under the numbered result folders and can be regenerated from
``03_raw_runs``.  Statistical resampling is case-clustered throughout.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import re
import warnings
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score


BOOTSTRAP_SEED = 20260729
DEFAULT_BOOTSTRAPS = 2000
EXPECTED_CASES = {"TopCoW_val": 20, "Lausanne": 20, "IXI_HH": 20}
DATASET_ORDER = ["TopCoW_val", "Lausanne", "IXI_HH"]
LEVEL_ORDER = ["shallow", "mid", "deep"]
# Raw runs produced before the semantic audit exported the concrete model's
# interpolation/acceptance coefficient under the ambiguous name ``T``.  Keep
# that raw field immutable for provenance, expose it as ``A_impl``, and derive
# the manuscript-defined trust score from same-row U and G.
FORMULA_TRUST = "T_formula"
IMPLEMENTED_ACCEPTANCE = "A_impl"
LEGACY_T = "T"
SIGNAL_SEMANTICS_VERSION = "2026-07-30.v2-product-intervention"
SIGNALS = {
    "1-U": lambda frame: 1.0 - frame["U"].astype(float),
    "G": lambda frame: frame["G"].astype(float),
    FORMULA_TRUST: lambda frame: frame[FORMULA_TRUST].astype(float),
    IMPLEMENTED_ACCEPTANCE: lambda frame: frame[IMPLEMENTED_ACCEPTANCE].astype(float),
    "1-FDD/2": lambda frame: (1.0 - frame["FDD"].astype(float) / 2.0).clip(0.0, 1.0),
}
PROBABILITY_SIGNALS = {
    "confidence": lambda frame: frame["confidence"].astype(float).clip(0.0, 1.0),
    "1-entropy": lambda frame: (1.0 - frame["entropy"].astype(float)).clip(0.0, 1.0),
}
TARGETS = {
    "component": ("B_component", "Q_component", "vessel_component_roi"),
    "voxel": ("B_vox", "Q_vox", "vessel_component_roi"),
    "centerline": ("B_centerline", "Q_centerline", "centerline_segment_roi"),
}
REQUIRED_MANIFEST = {
    "experiment_id", "run_id", "git_commit", "config_hash",
    "checkpoint_hash", "data_split_hash", "dataset", "model_variant",
    "corruption", "seed", "started_at", "finished_at", "status",
    "failure_reason",
}
REQUIRED_CASE = {
    "experiment_id", "run_id", "dataset", "case_id", "seed",
    "model_variant", "corruption_type", "severity", "dice", "cldice",
    "hd95", "beta0_error", "clsavg", "cls_beta0", "det_f1", "mean_U",
    "mean_G", "mean_T", "accepted_delta_l1", "accepted_delta_l2",
    "runtime_sec", "peak_gpu_mem_mb",
}
REQUIRED_REGION = {
    "experiment_id", "run_id", "dataset", "case_id", "region_id",
    "region_type", "feature_level", "vessel_class", "size_bin",
    "centerline_or_boundary", "U", "G", "T", "FDD", "delta_l1",
    "accepted_delta_l1", "Q_vox", "Q_centerline", "Q_component",
    "Q_topology", "B_vox", "B_centerline", "B_component", "trust_bin",
}

BLUE = "#4C78A8"
GOLD = "#F2B134"
ORANGE = "#E07A32"
OLIVE = "#7A8B36"
PINK = "#C76D9A"
CHARCOAL = "#25313C"
GREY = "#A7AFB7"
PALETTE = [BLUE, GOLD, ORANGE, OLIVE, PINK]


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def stable_int(*parts: object) -> int:
    value = "|".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "little")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path, default=None):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True,
                               ensure_ascii=False, default=str),
                    encoding="utf-8")
    temp.replace(path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temp, index=False)
    temp.replace(path)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(value, encoding="utf-8")
    temp.replace(path)


def atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temp, index=False)
    temp.replace(path)


def finite_number(value) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def relative_to(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def discover_runs(results_root: Path) -> Tuple[pd.DataFrame, List[dict]]:
    records: List[dict] = []
    raw: List[dict] = []
    for manifest_path in sorted(results_root.glob("03_raw_runs/*/*/run_manifest.json")):
        run_dir = manifest_path.parent
        manifest = read_json(manifest_path, {}) or {}
        status_payload = read_json(run_dir / "status.json", {}) or {}
        config_path = run_dir / "run_config.yaml"
        selected = read_json(run_dir / "selected_cases.json", []) or []
        failures = read_json(run_dir / "case_failures.json", []) or []
        runtime = read_json(run_dir / "runtime.json", {}) or {}
        corruption = manifest.get("corruption") or {}
        intervention = manifest.get("intervention") or {}
        gate_alpha = intervention.get("gate_alpha")
        gate_transform = intervention.get("gate_transform", "identity")
        model_variant = manifest.get("model_variant")
        if gate_alpha is not None:
            acceptance_policy = "constant_acceptance"
        elif gate_transform not in (None, "", "identity"):
            acceptance_policy = f"gate_transform:{gate_transform}"
        elif model_variant == "full":
            acceptance_policy = "submitted_full"
        elif model_variant == "product":
            acceptance_policy = "product"
        else:
            acceptance_policy = str(model_variant or "unknown")
        record = {
            "experiment_id": manifest.get("experiment_id", run_dir.parent.name),
            "run_id": manifest.get("run_id", run_dir.name),
            "run_dir": relative_to(run_dir, results_root),
            "manifest_path": relative_to(manifest_path, results_root),
            "status": status_payload.get("status", manifest.get("status", "unknown")),
            "manifest_status": manifest.get("status"),
            "dataset": manifest.get("dataset"),
            "model_variant": model_variant,
            "acceptance_policy": acceptance_policy,
            "gate_alpha": gate_alpha,
            "gate_transform": gate_transform,
            "temperature_scale": intervention.get("temperature_scale"),
            "corruption_type": corruption.get("type", "none"),
            "severity": corruption.get("severity", 0.0),
            "severity_unit": corruption.get("unit", "none"),
            "realization": corruption.get("realization", 0),
            "seed": manifest.get("seed"),
            "git_commit": manifest.get("git_commit"),
            "config_hash": manifest.get("config_hash"),
            "checkpoint_hash": manifest.get("checkpoint_hash"),
            "data_split_hash": manifest.get("data_split_hash"),
            "started_at": manifest.get("started_at"),
            "finished_at": manifest.get("finished_at"),
            "failure_reason": manifest.get("failure_reason"),
            "n_selected": len(selected),
            "n_case_failures": len(failures),
            "runtime_n_success": runtime.get("n_success"),
            "runtime_n_failed": runtime.get("n_failed"),
            "manifest_missing_fields": ";".join(sorted(REQUIRED_MANIFEST - set(manifest))),
            "has_config": config_path.exists(),
            "has_case_metrics": (run_dir / "case_metrics.csv").exists(),
            "has_class_metrics": (run_dir / "class_metrics.csv").exists(),
            "has_internal_signals": (run_dir / "internal_signals.parquet").exists(),
            "has_runtime": (run_dir / "runtime.json").exists(),
            "has_status": (run_dir / "status.json").exists(),
            "has_stdout": (run_dir / "stdout.log").exists(),
            "has_stderr": (run_dir / "stderr.log").exists(),
        }
        records.append(record)
        raw.append({"manifest": manifest, "status": status_payload,
                    "runtime": runtime, "selected": selected,
                    "failures": failures, "run_dir": run_dir})
    return pd.DataFrame(records), raw


def load_successful_outputs(results_root: Path, inventory: pd.DataFrame,
                            include_partial: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    accepted = {"success"}
    if include_partial:
        accepted.add("partial")
    case_frames: List[pd.DataFrame] = []
    class_frames: List[pd.DataFrame] = []
    region_frames: List[pd.DataFrame] = []
    for row in inventory.itertuples(index=False):
        if row.status not in accepted:
            continue
        run_dir = results_root / row.run_dir
        metadata = {
            "source_run_dir": row.run_dir,
            "source_manifest": row.manifest_path,
            "git_commit": row.git_commit,
            "config_hash": row.config_hash,
            "checkpoint_hash": row.checkpoint_hash,
            "data_split_hash": row.data_split_hash,
            "severity_unit": row.severity_unit,
            "acceptance_policy": row.acceptance_policy,
            "gate_alpha": row.gate_alpha,
            "gate_transform": row.gate_transform,
            "temperature_scale": row.temperature_scale,
        }
        for name, frames, reader in (
                ("case_metrics.csv", case_frames, pd.read_csv),
                ("class_metrics.csv", class_frames, pd.read_csv),
                ("internal_signals.parquet", region_frames, pd.read_parquet)):
            path = run_dir / name
            if not path.exists() or path.stat().st_size == 0:
                continue
            frame = reader(path)
            for key, value in metadata.items():
                frame[key] = value
            frames.append(frame)
    empty = pd.DataFrame()
    return (
        pd.concat(case_frames, ignore_index=True, sort=False) if case_frames else empty.copy(),
        pd.concat(class_frames, ignore_index=True, sort=False) if class_frames else empty.copy(),
        pd.concat(region_frames, ignore_index=True, sort=False) if region_frames else empty.copy(),
    )


def add_signal_semantics(case: pd.DataFrame,
                         region: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Add unambiguous derived aliases without mutating raw-run evidence.

    Raw files use ``T`` for the coefficient actually executed by that run:
    ``1-(1-G)U`` for submitted ``full`` and ``G(1-U)`` for the explicit
    ``product`` intervention.  The manuscript-formula diagnostic is always
    recomputed from same-row U/G, so every downstream claim states which
    quantity and intervention it evaluates.
    """
    case_out = case.copy()
    region_out = region.copy()
    if "mean_T" in case_out:
        case_out["mean_A_impl"] = pd.to_numeric(
            case_out["mean_T"], errors="coerce")
    if region_out.empty:
        return case_out, region_out

    for column in ["U", "G", LEGACY_T]:
        if column not in region_out:
            region_out[column] = np.nan
        region_out[column] = pd.to_numeric(region_out[column], errors="coerce")
    region_out[IMPLEMENTED_ACCEPTANCE] = region_out[LEGACY_T]
    region_out[FORMULA_TRUST] = (
        region_out["G"] * (1.0 - region_out["U"])
    )
    region_out["A_impl_bin"] = pd.cut(
        region_out[IMPLEMENTED_ACCEPTANCE], bins=np.linspace(0.0, 1.0, 11),
        labels=False, include_lowest=True).astype("Int64")
    region_out["T_formula_bin"] = pd.cut(
        region_out[FORMULA_TRUST], bins=np.linspace(0.0, 1.0, 11),
        labels=False, include_lowest=True).astype("Int64")
    region_out["legacy_T_semantics"] = "implemented_acceptance"

    if {"clean_U", "clean_G"} <= set(region_out.columns):
        clean_u = pd.to_numeric(region_out["clean_U"], errors="coerce")
        clean_g = pd.to_numeric(region_out["clean_G"], errors="coerce")
        region_out["clean_T_formula"] = clean_g * (1.0 - clean_u)
    if "clean_T" in region_out:
        region_out["clean_A_impl"] = pd.to_numeric(
            region_out["clean_T"], errors="coerce")
    return case_out, region_out


def write_signal_semantics_note(results_root: Path) -> None:
    note = f"""# Trust-signal semantics correction

Version: `{SIGNAL_SEMANTICS_VERSION}`
Generated: {utc_now()}

## Immutable raw evidence

Raw-run CSV, Parquet, NPZ, checkpoints, predictions, and manifests are not
rewritten. Historical raw field `T` is retained verbatim and exposed in
derived tables as `A_impl`, the interpolation coefficient actually executed
by the concrete model.

For the unmodified submitted `full` acceptance policy, the audited
implementation identity is:

`A_impl = 1 - (1-G)U`.

Runs that explicitly replace this policy (for example, constant-acceptance or
gate-transform controls) retain their intervention label and are audited in a
separate acceptance-policy stratum; they are never tested against or described
as executing the unmodified submitted identity.

## Manuscript-formula signal

Every downstream formula-trust analysis derives, on the same region/feature
level support:

`T_formula = G(1-U)`.

The mandatory invariant is `0 <= T_formula <= G <= 1`. The raw `T` column is
never used as `T_formula` after this correction.

`T_formula` derived from a historical full run is a diagnostic score only. It
does not convert that run into product-gated performance. The separately
labelled `product` intervention executes `A_product=G(1-U)` inside each fusion
module; its audit additionally requires `A_impl == T_formula` on every
available row.

## Regenerated outputs

Signal validity, calibration, risk-coverage, frozen-threshold action rates,
failure-case selection, controlled-intervention summaries, Figures R2-R4,
case panels, quality reports, and the exact-number quote ledger must all be
regenerated from the immutable raw runs. Historical segmentation predictions
and metrics remain unchanged. Claims comparing product-gated performance use
only the new `P0_D_PRODUCT_*` runs and never a renamed historical full run.
"""
    atomic_text(results_root / "11_quality_audit" /
                "TRUST_SIGNAL_SEMANTICS_CORRECTION.md", note)


def add_finding(findings: List[dict], check_id: str, severity: str,
                status: str, evidence: str, impact: str,
                remediation: str = "", confidence: str = "high") -> None:
    findings.append({
        "check_id": check_id,
        "severity": severity,
        "status": status,
        "evidence": evidence,
        "impact": impact,
        "remediation": remediation,
        "confidence": confidence,
    })


def audit_outputs(results_root: Path, inventory: pd.DataFrame,
                  case: pd.DataFrame, region: pd.DataFrame) -> pd.DataFrame:
    findings: List[dict] = []
    if inventory.empty:
        add_finding(findings, "runs_present", "critical", "fail",
                    "No run manifests were discovered.",
                    "No quantitative result is auditable.",
                    "Restore or complete raw runs.")
        return pd.DataFrame(findings)

    duplicate_runs = int(inventory.duplicated(["experiment_id", "run_id"]).sum())
    add_finding(findings, "unique_run_ids",
                "critical" if duplicate_runs else "low",
                "fail" if duplicate_runs else "pass",
                f"Duplicate experiment/run identities: {duplicate_runs}.",
                "Duplicates can silently double-count experiments.")

    missing_manifest = inventory["manifest_missing_fields"].astype(str).str.len() > 0
    add_finding(findings, "manifest_schema",
                "high" if missing_manifest.any() else "low",
                "fail" if missing_manifest.any() else "pass",
                f"Runs missing one or more required manifest fields: {int(missing_manifest.sum())}/{len(inventory)}.",
                "Missing identity fields weaken provenance.",
                "Repair only derived manifests; never overwrite raw evidence.")

    status_mismatch = inventory[
        inventory["manifest_status"].notna() &
        (inventory["status"] != inventory["manifest_status"])]
    add_finding(findings, "status_consistency",
                "high" if len(status_mismatch) else "low",
                "fail" if len(status_mismatch) else "pass",
                f"Status/manifest disagreements: {len(status_mismatch)}.",
                "A mismatched status can admit incomplete data.")

    status_counts = inventory["status"].value_counts(dropna=False).to_dict()
    non_success = inventory[~inventory["status"].isin(["success", "running"])]
    add_finding(findings, "failed_partial_runs",
                "high" if len(non_success) else "low",
                "fail" if len(non_success) else "pass",
                f"Run status counts: {status_counts}; failed/partial/unknown={len(non_success)}.",
                "Non-success runs must remain visible and cannot enter primary pooling.")

    inference_experiments = {
        "G1_PILOT_INTERFACE", "P0_AB_CLEAN_MECHANISM",
        "P0_D_FIXED_CKPT_INTERVENTIONS", "P0_C1_GAUSSIAN",
        "P0_C1_CHANNEL_DROPOUT", "P0_C2_TRANSLATION",
        "P0_C3_DONOR_MIX", "P0_D_TRAINED_EVAL_CLEAN",
        "P0_D_TRAINED_EVAL_CORRUPTION",
        "P0_B_PROBABILITY_SIGNALS", "P0_G_SELECTED_CASE_MAPS",
    }
    result_runs = inventory[inventory["experiment_id"].isin(inference_experiments)]
    required_file_cols = ["has_config", "has_case_metrics", "has_class_metrics",
                          "has_internal_signals", "has_runtime", "has_status",
                          "has_stdout", "has_stderr"]
    successful_result_runs = result_runs[result_runs["status"].eq("success")]
    missing_files = int((~successful_result_runs[required_file_cols].fillna(False)).sum().sum())
    add_finding(findings, "run_file_completeness",
                "high" if missing_files else "low",
                "fail" if missing_files else "pass",
                f"Missing required files across successful inference runs: {missing_files}.",
                "A successful run without its evidence bundle is not reproducible.")

    if case.empty:
        add_finding(findings, "case_table_present", "critical", "fail",
                    "No case metrics were loaded from successful runs.",
                    "Headline performance cannot be computed.")
    else:
        missing_case_cols = sorted(REQUIRED_CASE - set(case.columns))
        add_finding(findings, "case_schema",
                    "high" if missing_case_cols else "low",
                    "fail" if missing_case_cols else "pass",
                    "Missing case columns: " + (", ".join(missing_case_cols) or "none"),
                    "Missing fields can invalidate a requested metric.")
        dup_case = int(case.duplicated(["experiment_id", "run_id", "case_id"]).sum())
        add_finding(findings, "unique_case_rows",
                    "critical" if dup_case else "low",
                    "fail" if dup_case else "pass",
                    f"Duplicate case rows within runs: {dup_case}.",
                    "Duplicate cases bias every aggregate.")
        numeric = case.select_dtypes(include=[np.number])
        inf_count = int(np.isinf(numeric.to_numpy(dtype=float, na_value=np.nan)).sum())
        add_finding(findings, "case_finite_values",
                    "high" if inf_count else "low",
                    "fail" if inf_count else "pass",
                    f"Infinite numeric case values: {inf_count}. Missing values are audited separately.",
                    "Infinite values break summaries and confidence intervals.")
        range_fail = 0
        for column in ["dice", "cldice", "clsavg", "det_f1"]:
            if column in case:
                values = pd.to_numeric(case[column], errors="coerce")
                range_fail += int(((values < 0) | (values > 1)).sum())
        add_finding(findings, "metric_ranges",
                    "high" if range_fail else "low",
                    "fail" if range_fail else "pass",
                    f"Out-of-range Dice/clDice/class-average/Det-F1 values: {range_fail}.",
                    "Out-of-range values imply a metric or parsing error.")

    if region.empty:
        add_finding(findings, "region_table_present", "critical", "fail",
                    "No internal signal rows were loaded.",
                    "Mechanism and reliability analyses cannot run.")
    else:
        missing_region_cols = sorted(REQUIRED_REGION - set(region.columns))
        add_finding(findings, "region_schema",
                    "high" if missing_region_cols else "low",
                    "fail" if missing_region_cols else "pass",
                    "Missing region columns: " + (", ".join(missing_region_cols) or "none"),
                    "Missing fields can invalidate mechanism analyses.")
        region_key = ["experiment_id", "run_id", "case_id", "region_id", "feature_level"]
        dup_region = int(region.duplicated(region_key).sum()) if set(region_key) <= set(region) else -1
        add_finding(findings, "unique_region_rows",
                    "critical" if dup_region > 0 else "low",
                    "fail" if dup_region > 0 else "pass",
                    f"Duplicate region/level rows within runs: {dup_region}.",
                    "Duplicates can create pseudo-replication.")
        numeric = region.select_dtypes(include=[np.number])
        inf_count = int(np.isinf(numeric.to_numpy(dtype=float, na_value=np.nan)).sum())
        add_finding(findings, "region_finite_values",
                    "high" if inf_count else "low",
                    "fail" if inf_count else "pass",
                    f"Infinite internal-signal values: {inf_count}.",
                    "Infinite values invalidate signal ranking.")
        na_rows = []
        for column in ["U", "G", "T", "FDD", "Q_vox", "Q_centerline", "Q_component"]:
            if column in region:
                na_rows.append(f"{column}={int(region[column].isna().sum())}")
        add_finding(findings, "region_missingness", "low", "pass_with_note",
                    "Missing counts: " + ", ".join(na_rows) + ".",
                    "Undefined centerlines and masks remain NA rather than being imputed.")

        semantic_columns = {"U", "G", LEGACY_T, FORMULA_TRUST,
                            IMPLEMENTED_ACCEPTANCE}
        if not semantic_columns <= set(region.columns):
            missing_semantics = sorted(semantic_columns - set(region.columns))
            add_finding(
                findings, "trust_signal_semantics", "critical", "fail",
                "Missing semantic columns: " + ", ".join(missing_semantics) + ".",
                "Paper-formula trust cannot be separated from the executed interpolation coefficient.",
                "Run add_signal_semantics before audit and downstream analysis.")
        else:
            tolerance = 1e-5
            u = pd.to_numeric(region["U"], errors="coerce")
            g = pd.to_numeric(region["G"], errors="coerce")
            legacy = pd.to_numeric(region[LEGACY_T], errors="coerce")
            formula = pd.to_numeric(region[FORMULA_TRUST], errors="coerce")
            implemented = pd.to_numeric(
                region[IMPLEMENTED_ACCEPTANCE], errors="coerce")
            computable = np.isfinite(u) & np.isfinite(g)
            with_legacy = computable & np.isfinite(legacy) & np.isfinite(implemented)
            policy = region.get(
                "acceptance_policy", pd.Series("unknown", index=region.index)
            ).astype(str)
            # The submitted identity applies only when the acceptance rule was
            # not intentionally replaced.  Constant-acceptance and transformed-
            # gate experiments keep model_variant="full" to preserve checkpoint
            # identity, but by design do not execute the submitted full rule.
            full = with_legacy & policy.eq("submitted_full")
            product = with_legacy & policy.eq("product")
            explicit_intervention = with_legacy & (
                policy.eq("constant_acceptance") |
                policy.str.startswith("gate_transform:")
            )
            input_range_fail = int((
                computable & ((u < -tolerance) | (u > 1.0 + tolerance) |
                              (g < -tolerance) | (g > 1.0 + tolerance))
            ).sum())
            formula_range_fail = int((
                computable & ((formula < -tolerance) |
                              (formula > 1.0 + tolerance))
            ).sum())
            formula_gt_g = int((computable & (formula > g + tolerance)).sum())
            alias_mismatch = int((
                with_legacy & ((implemented - legacy).abs() > tolerance)
            ).sum())
            expected_impl = 1.0 - (1.0 - g) * u
            impl_residual = (implemented - expected_impl).abs()
            impl_identity_fail = int((full & (impl_residual > tolerance)).sum())
            product_residual = (implemented - formula).abs()
            product_identity_fail = int((
                product & (product_residual > tolerance)).sum())
            legacy_gt_g = int((with_legacy & (legacy > g + tolerance)).sum())
            legacy_vs_formula = (legacy - formula).abs()
            legacy_formula_mismatch = int((
                with_legacy & (legacy_vs_formula > tolerance)
            ).sum())
            semantic_audit = {
                "created_at": utc_now(),
                "raw_field": LEGACY_T,
                "raw_field_meaning": "implemented interpolation coefficient A_impl",
                "paper_formula_field": FORMULA_TRUST,
                "paper_formula": "G * (1 - U)",
                "full_variant_implemented_formula": "1 - (1 - G) * U",
                "tolerance": tolerance,
                "n_region_rows": int(len(region)),
                "n_formula_computable_rows": int(computable.sum()),
                "formula_coverage_fraction": (
                    float(computable.mean()) if len(region) else 0.0),
                "n_rows_with_legacy_T": int(with_legacy.sum()),
                "n_full_rows_checked": int(full.sum()),
                "n_explicit_acceptance_intervention_rows": int(
                    explicit_intervention.sum()),
                "acceptance_policy_counts": {
                    str(key): int(value) for key, value in
                    policy[with_legacy].value_counts().sort_index().items()
                },
                "input_range_failures": input_range_fail,
                "formula_range_failures": formula_range_fail,
                "formula_T_gt_G_failures": formula_gt_g,
                "A_impl_alias_mismatches": alias_mismatch,
                "full_A_impl_identity_failures": impl_identity_fail,
                "max_abs_full_A_impl_identity_residual": (
                    float(impl_residual[full].max()) if full.any() else None),
                "n_product_rows_checked": int(product.sum()),
                "product_A_impl_equals_T_formula_failures": product_identity_fail,
                "max_abs_product_A_impl_T_formula_residual": (
                    float(product_residual[product].max()) if product.any() else None),
                "legacy_T_gt_G_rows": legacy_gt_g,
                "legacy_T_vs_formula_mismatch_rows": legacy_formula_mismatch,
                "max_abs_legacy_T_vs_formula_difference": (
                    float(legacy_vs_formula[with_legacy].max())
                    if with_legacy.any() else None),
                "remediation": (
                    "Raw T is preserved for provenance and aliased to A_impl; "
                    "all manuscript-formula analyses use T_formula derived from same-row U and G."),
            }
            semantic_audit["status"] = (
                "pass" if not (input_range_fail or formula_range_fail or
                                formula_gt_g or alias_mismatch or
                                impl_identity_fail or product_identity_fail) else "fail")
            atomic_json(results_root / "11_quality_audit" /
                        "TRUST_SIGNAL_SEMANTICS_AUDIT.json", semantic_audit)
            atomic_csv(results_root / "11_quality_audit" /
                       "TRUST_SIGNAL_SEMANTICS_AUDIT.csv",
                       pd.DataFrame([semantic_audit]))
            add_finding(
                findings, "paper_formula_trust_invariant",
                "critical" if formula_gt_g or formula_range_fail else "low",
                "fail" if formula_gt_g or formula_range_fail else "pass",
                (f"Derived {FORMULA_TRUST}=G(1-U) on {int(computable.sum())}/"
                 f"{len(region)} rows; T_formula>G violations={formula_gt_g}; "
                 f"range violations={formula_range_fail}."),
                "Violations would invalidate every paper-formula trust analysis.")
            add_finding(
                findings, "legacy_t_semantic_remap", "high",
                "fail" if impl_identity_fail or product_identity_fail or alias_mismatch else "pass",
                (f"Historical raw T is retained as A_impl. Unmodified submitted-"
                  f"full rows "
                  f"checked={int(full.sum())}; A_impl identity failures="
                  f"{impl_identity_fail}; product A_impl=T_formula failures="
                  f"{product_identity_fail}; legacy T>G rows={legacy_gt_g}; "
                  f"legacy/formula mismatches={legacy_formula_mismatch}; "
                  f"explicit acceptance-intervention rows separately scoped="
                  f"{int(explicit_intervention.sum())}."),
                "Ambiguous legacy T must not be quoted as G(1-U).",
                "Use T_formula for manuscript-formula analyses and A_impl for executed interpolation/accepted-update analyses.")

    formal = inventory[inventory["experiment_id"].str.startswith("P0_", na=False)]
    completed_counts = formal[formal["status"].eq("success")].groupby(
        ["experiment_id", "dataset", "model_variant"], dropna=False).size()
    atomic_csv(results_root / "11_quality_audit" / "seed_and_run_coverage.csv",
               completed_counts.rename("n_successful_runs").reset_index())

    fixed = formal[formal["experiment_id"].isin([
        "P0_AB_CLEAN_MECHANISM", "P0_D_FIXED_CKPT_INTERVENTIONS",
        "P0_C1_GAUSSIAN", "P0_C1_CHANNEL_DROPOUT",
        "P0_C2_TRANSLATION", "P0_C3_DONOR_MIX",
        "P0_D_PRODUCT_FIXED_CLEAN", "P0_D_PRODUCT_FIXED_CORRUPTION"])]
    unique_ckpt = fixed["checkpoint_hash"].dropna().nunique()
    unique_commit = fixed["git_commit"].dropna().nunique()
    add_finding(findings, "fixed_checkpoint_identity",
                "critical" if unique_ckpt > 1 else "low",
                "fail" if unique_ckpt > 1 else "pass",
                f"Distinct checkpoint hashes in fixed-checkpoint P0 runs: {unique_ckpt}; code commits: {unique_commit}.",
                "A checkpoint mismatch would confound mechanism comparisons.")

    # The executable configs contain no threshold or calibration argument.  The
    # separately frozen protocol supplies all thresholds from source validation.
    protocol = results_root / "01_configs" / "POSTPROCESSING_PROTOCOL.md"
    add_finding(findings, "external_tuning_guard",
                "high" if not protocol.exists() else "low",
                "fail" if not protocol.exists() else "pass",
                ("Frozen source-only protocol SHA256=" + sha256_file(protocol))
                if protocol.exists() else "Frozen post-processing protocol missing.",
                "Without a frozen source-only rule, external results could influence thresholds.",
                "Restore the preregistered protocol before external analysis.")

    integrity = read_json(results_root / "00_project_discovery" / "DATA_INTEGRITY_AUDIT.json")
    if integrity:
        overlaps = int(integrity.get("cross_dataset_exact_duplicates", -1))
        add_finding(findings, "split_overlap",
                    "critical" if overlaps > 0 else "low",
                    "fail" if overlaps > 0 else "pass",
                    f"Exact preprocessed-image duplicates across dataset folders: {overlaps}.",
                    "Cross-cohort duplicates would leak cases across evaluation domains.")
    else:
        add_finding(findings, "split_overlap", "medium", "not_machine_readable",
                    "The discovery report records the check, but a machine-readable integrity audit is absent.",
                    "The check cannot be independently recomputed from this result snapshot.",
                    "Persist DATA_INTEGRITY_AUDIT.json with hashes and overlap counts.")

    return pd.DataFrame(findings)


def clean_binary_arrays(y: Sequence, score: Sequence) -> Tuple[np.ndarray, np.ndarray]:
    y_array = pd.to_numeric(pd.Series(y), errors="coerce").to_numpy(dtype=float)
    score_array = pd.to_numeric(pd.Series(score), errors="coerce").to_numpy(dtype=float)
    keep = np.isfinite(y_array) & np.isfinite(score_array)
    return y_array[keep].astype(int), score_array[keep]


def safe_auprc(y: Sequence, score: Sequence) -> float:
    labels, values = clean_binary_arrays(y, score)
    if len(labels) == 0 or labels.sum() == 0:
        return float("nan")
    return float(average_precision_score(labels, values))


def safe_auroc(y: Sequence, score: Sequence) -> float:
    labels, values = clean_binary_arrays(y, score)
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, values))


def brier(y: Sequence, probability: Sequence) -> float:
    labels, values = clean_binary_arrays(y, probability)
    if len(labels) == 0:
        return float("nan")
    values = np.clip(values, 0.0, 1.0)
    return float(np.mean((values - labels) ** 2))


def binary_nll(y: Sequence, probability: Sequence) -> float:
    labels, values = clean_binary_arrays(y, probability)
    if len(labels) == 0:
        return float("nan")
    values = np.clip(values, 1e-7, 1.0 - 1e-7)
    return float(-np.mean(labels * np.log(values) +
                          (1 - labels) * np.log(1 - values)))


def risk_coverage_curve(y_beneficial: Sequence,
                        beneficial_score: Sequence) -> pd.DataFrame:
    labels, scores = clean_binary_arrays(y_beneficial, beneficial_score)
    if len(labels) == 0:
        return pd.DataFrame(columns=["threshold", "coverage", "harmful_risk",
                                     "n_accepted", "n_total"])
    table = pd.DataFrame({"beneficial": labels, "score": scores})
    grouped = table.groupby("score", sort=False)["beneficial"].agg(
        n="size", beneficial="sum").sort_index(ascending=False)
    grouped["n_accepted"] = grouped["n"].cumsum()
    grouped["beneficial_accepted"] = grouped["beneficial"].cumsum()
    grouped["coverage"] = grouped["n_accepted"] / len(table)
    grouped["harmful_risk"] = (
        (grouped["n_accepted"] - grouped["beneficial_accepted"]) /
        grouped["n_accepted"])
    curve = grouped.reset_index().rename(columns={"score": "threshold"})
    origin = pd.DataFrame([{
        "threshold": float("inf"), "n": 0, "beneficial": 0,
        "n_accepted": 0, "beneficial_accepted": 0,
        "coverage": 0.0, "harmful_risk": 0.0,
    }])
    curve = pd.concat([origin, curve], ignore_index=True, sort=False)
    curve["n_total"] = len(table)
    return curve


def aurc(y_beneficial: Sequence, beneficial_score: Sequence) -> float:
    curve = risk_coverage_curve(y_beneficial, beneficial_score)
    if curve.empty:
        return float("nan")
    return float(np.trapz(curve["harmful_risk"].to_numpy(),
                          curve["coverage"].to_numpy()))


def cluster_bootstrap(frame: pd.DataFrame, metric: Callable[[pd.DataFrame], float],
                      n_boot: int, seed: int,
                      cluster_col: str = "case_id") -> Tuple[float, float, int]:
    if frame.empty or cluster_col not in frame:
        return float("nan"), float("nan"), 0
    clusters = sorted(frame[cluster_col].dropna().astype(str).unique().tolist())
    if not clusters:
        return float("nan"), float("nan"), 0
    grouped = {cluster: frame[frame[cluster_col].astype(str).eq(cluster)]
               for cluster in clusters}
    rng = np.random.default_rng(seed)
    values: List[float] = []
    for _ in range(n_boot):
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        pieces = []
        for draw, cluster in enumerate(sampled):
            part = grouped[str(cluster)].copy()
            # A synthetic cluster id preserves repeated draws as independent
            # bootstrap clusters without changing any row-level metric.
            part["_bootstrap_case"] = f"{draw}:{cluster}"
            pieces.append(part)
        value = finite_number(metric(pd.concat(pieces, ignore_index=True)))
        if value is not None:
            values.append(value)
    if not values:
        return float("nan"), float("nan"), 0
    low, high = np.quantile(values, [0.025, 0.975])
    return float(low), float(high), len(values)


def bootstrap_case_weights(n_boot: int, n_cases: int, seed: int) -> np.ndarray:
    if n_boot <= 0 or n_cases <= 0:
        return np.empty((0, n_cases), dtype=np.float32)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_cases, size=(n_boot, n_cases))
    weights = np.zeros((n_boot, n_cases), dtype=np.float32)
    row_index = np.repeat(np.arange(n_boot), n_cases)
    np.add.at(weights, (row_index, draws.reshape(-1)), 1.0)
    return weights


def finite_interval(values: np.ndarray) -> Tuple[float, float, int]:
    valid = np.asarray(values, dtype=float)
    valid = valid[np.isfinite(valid)]
    if len(valid) == 0:
        return float("nan"), float("nan"), 0
    low, high = np.quantile(valid, [0.025, 0.975])
    return float(low), float(high), int(len(valid))


def bootstrap_mean_ci(frame: pd.DataFrame, value_col: str, n_boot: int,
                      seed: int) -> Tuple[float, float, int]:
    grouped = frame.groupby("case_id")[value_col].agg(["sum", "count"])
    if grouped.empty:
        return float("nan"), float("nan"), 0
    weights = bootstrap_case_weights(n_boot, len(grouped), seed)
    numerator = weights @ grouped["sum"].to_numpy(dtype=float)
    denominator = weights @ grouped["count"].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        values = numerator / denominator
    return finite_interval(values)


def bootstrap_binary_metric_cis(work: pd.DataFrame, label_col: str,
                                 probability_col: str, n_boot: int,
                                 seed: int) -> Dict[str, Tuple[float, float, int]]:
    metrics = ["auprc_beneficial", "auprc_harmful", "auroc_beneficial",
               "brier_beneficial", "nll_beneficial",
               "aurc_harmful_acceptance"]
    empty = {name: (np.nan, np.nan, 0) for name in metrics}
    if work.empty or n_boot <= 0:
        return empty
    frame = work[["case_id", label_col, probability_col]].copy()
    frame[label_col] = pd.to_numeric(frame[label_col], errors="coerce")
    frame[probability_col] = pd.to_numeric(frame[probability_col], errors="coerce")
    frame = frame[np.isfinite(frame[label_col]) &
                  np.isfinite(frame[probability_col])]
    if frame.empty:
        return empty
    y = frame[label_col].to_numpy(dtype=int)
    probability = frame[probability_col].clip(0.0, 1.0).to_numpy(dtype=float)
    cases, case_code = np.unique(frame["case_id"].astype(str).to_numpy(),
                                 return_inverse=True)
    unique_scores, score_code_ascending = np.unique(probability, return_inverse=True)
    score_code = len(unique_scores) - 1 - score_code_ascending
    n_cases = len(cases)
    n_groups = len(unique_scores)
    total_by_case_group = np.zeros((n_cases, n_groups), dtype=np.float32)
    positive_by_case_group = np.zeros((n_cases, n_groups), dtype=np.float32)
    np.add.at(total_by_case_group, (case_code, score_code), 1.0)
    np.add.at(positive_by_case_group, (case_code, score_code), y.astype(np.float32))
    squared_error_by_case = np.bincount(
        case_code, weights=(probability - y) ** 2, minlength=n_cases)
    clipped = np.clip(probability, 1e-7, 1.0 - 1e-7)
    nll_by_case = np.bincount(
        case_code,
        weights=-(y * np.log(clipped) + (1 - y) * np.log(1 - clipped)),
        minlength=n_cases)
    weights = bootstrap_case_weights(n_boot, n_cases, seed)
    group_total = weights @ total_by_case_group
    group_positive = weights @ positive_by_case_group
    group_harmful = group_total - group_positive
    total = group_total.sum(axis=1)
    total_positive = group_positive.sum(axis=1)
    total_harmful = group_harmful.sum(axis=1)
    cumulative_total = np.cumsum(group_total, axis=1)
    cumulative_positive = np.cumsum(group_positive, axis=1)
    cumulative_harmful = np.cumsum(group_harmful, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.divide(
            cumulative_positive, cumulative_total,
            out=np.zeros_like(cumulative_positive),
            where=cumulative_total > 0)
        ap_beneficial = np.divide(
            np.sum(precision * group_positive, axis=1), total_positive,
            out=np.full(n_boot, np.nan), where=total_positive > 0)
        reverse_total = group_total[:, ::-1]
        reverse_harmful = group_harmful[:, ::-1]
        cumulative_reverse_harmful = np.cumsum(reverse_harmful, axis=1)
        cumulative_reverse_total = np.cumsum(reverse_total, axis=1)
        harm_precision = np.divide(
            cumulative_reverse_harmful, cumulative_reverse_total,
            out=np.zeros_like(cumulative_reverse_harmful),
            where=cumulative_reverse_total > 0)
        ap_harmful = np.divide(
            np.sum(harm_precision * reverse_harmful, axis=1), total_harmful,
            out=np.full(n_boot, np.nan), where=total_harmful > 0)
        harmful_below = total_harmful[:, None] - cumulative_harmful
        auroc = (np.sum(
            group_positive * (harmful_below + 0.5 * group_harmful), axis=1) /
            (total_positive * total_harmful))
        brier_values = (weights @ squared_error_by_case) / total
        nll_values = (weights @ nll_by_case) / total
        coverage = np.divide(
            cumulative_total, total[:, None],
            out=np.zeros_like(cumulative_total), where=total[:, None] > 0)
        risk = np.divide(
            cumulative_harmful, cumulative_total,
            out=np.zeros_like(cumulative_harmful),
            where=cumulative_total > 0)
        previous_coverage = np.concatenate(
            [np.zeros((n_boot, 1)), coverage[:, :-1]], axis=1)
        previous_risk = np.concatenate(
            [np.zeros((n_boot, 1)), risk[:, :-1]], axis=1)
        aurc_values = np.sum(
            0.5 * (previous_risk + risk) *
            (coverage - previous_coverage), axis=1)
    distributions = {
        "auprc_beneficial": ap_beneficial,
        "auprc_harmful": ap_harmful,
        "auroc_beneficial": auroc,
        "brier_beneficial": brier_values,
        "nll_beneficial": nll_values,
        "aurc_harmful_acceptance": aurc_values,
    }
    return {name: finite_interval(values)
            for name, values in distributions.items()}


def bootstrap_action_rate_cis(frame: pd.DataFrame, threshold: float,
                               n_boot: int, seed: int) -> Dict[str, Tuple[float, float, int]]:
    names = ["har", "brr", "coverage", "balanced_action_error"]
    empty = {name: (np.nan, np.nan, 0) for name in names}
    if frame.empty or n_boot <= 0 or not math.isfinite(float(threshold)):
        return empty
    work = frame[["case_id", "B_component", "score"]].copy()
    work["B_component"] = pd.to_numeric(work["B_component"], errors="coerce")
    work["score"] = pd.to_numeric(work["score"], errors="coerce")
    work = work[np.isfinite(work["B_component"]) & np.isfinite(work["score"])]
    if work.empty:
        return empty
    work["accepted"] = work["score"] > threshold
    work["harmful"] = work["B_component"].eq(0)
    work["beneficial"] = work["B_component"].eq(1)
    work["harmful_accepted"] = work["harmful"] & work["accepted"]
    work["beneficial_rejected"] = work["beneficial"] & ~work["accepted"]
    work["one"] = 1
    by_case = work.groupby("case_id").agg(
        harmful=("harmful", "sum"),
        harmful_accepted=("harmful_accepted", "sum"),
        beneficial=("beneficial", "sum"),
        beneficial_rejected=("beneficial_rejected", "sum"),
        accepted=("accepted", "sum"), total=("one", "sum"))
    weights = bootstrap_case_weights(n_boot, len(by_case), seed)
    aggregate = {column: weights @ by_case[column].to_numpy(dtype=float)
                 for column in by_case.columns}
    with np.errstate(divide="ignore", invalid="ignore"):
        har = aggregate["harmful_accepted"] / aggregate["harmful"]
        brr = aggregate["beneficial_rejected"] / aggregate["beneficial"]
        coverage = aggregate["accepted"] / aggregate["total"]
        balanced = 0.5 * (har + brr)
    return {name: finite_interval(values) for name, values in {
        "har": har, "brr": brr, "coverage": coverage,
        "balanced_action_error": balanced}.items()}


def summarize_clustered(frame: pd.DataFrame, value_col: str, n_boot: int,
                        seed_parts: Sequence[object],
                        binary: bool = False) -> dict:
    work = frame[["case_id", value_col]].copy()
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    work = work[np.isfinite(work[value_col])]
    if work.empty:
        return {
            "mean": np.nan, "sd": np.nan, "median": np.nan,
            "q25": np.nan, "q75": np.nan, "ci_low": np.nan,
            "ci_high": np.nan, "n_rows": 0, "n_cases": 0,
            "n_boot_valid": 0,
        }
    low, high, valid = bootstrap_mean_ci(
        work, value_col, n_boot,
        stable_int(BOOTSTRAP_SEED, *seed_parts))
    values = work[value_col]
    return {
        "mean": float(values.mean()),
        "sd": float(values.std(ddof=1)) if len(values) > 1 else np.nan,
        "median": float(values.median()),
        "q25": float(values.quantile(0.25)),
        "q75": float(values.quantile(0.75)),
        "ci_low": low,
        "ci_high": high,
        "n_rows": int(len(values)),
        "n_cases": int(work["case_id"].nunique()),
        "n_boot_valid": valid,
        "is_binary": bool(binary),
    }


def metric_bundle(frame: pd.DataFrame, label_col: str, probability: pd.Series,
                  n_boot: int, seed_parts: Sequence[object],
                  compute_ci: bool = True) -> dict:
    work = frame[["case_id", label_col]].copy()
    work["probability"] = pd.to_numeric(probability, errors="coerce")
    work[label_col] = pd.to_numeric(work[label_col], errors="coerce")
    work = work[np.isfinite(work[label_col]) & np.isfinite(work["probability"])]
    if work.empty:
        return {
            "n_rows": 0, "n_cases": 0, "prevalence_beneficial": np.nan,
            "prevalence_harmful": np.nan,
        }
    work[label_col] = work[label_col].astype(int)
    metric_functions = {
        "auprc_beneficial": lambda part: safe_auprc(part[label_col], part["probability"]),
        "auprc_harmful": lambda part: safe_auprc(1 - part[label_col], 1 - part["probability"]),
        "auroc_beneficial": lambda part: safe_auroc(part[label_col], part["probability"]),
        "brier_beneficial": lambda part: brier(part[label_col], part["probability"]),
        "nll_beneficial": lambda part: binary_nll(part[label_col], part["probability"]),
        "aurc_harmful_acceptance": lambda part: aurc(part[label_col], part["probability"]),
    }
    output = {
        "n_rows": int(len(work)),
        "n_cases": int(work["case_id"].nunique()),
        "prevalence_beneficial": float(work[label_col].mean()),
        "prevalence_harmful": float(1.0 - work[label_col].mean()),
    }
    intervals = (bootstrap_binary_metric_cis(
        work, label_col, "probability", n_boot,
        stable_int(BOOTSTRAP_SEED, *seed_parts, "joint_metrics"))
        if compute_ci else {})
    for name, function in metric_functions.items():
        output[name] = function(work)
        low, high, valid = intervals.get(name, (np.nan, np.nan, 0))
        output[name + "_ci_low"] = low
        output[name + "_ci_high"] = high
        output[name + "_n_boot_valid"] = valid
    output["ci_computed"] = bool(compute_ci)
    return output


def choose_source_threshold(labels: Sequence, scores: Sequence) -> dict:
    y, s = clean_binary_arrays(labels, scores)
    if len(y) == 0 or len(np.unique(y)) < 2:
        return {"threshold": np.nan, "har": np.nan, "brr": np.nan,
                "coverage": np.nan, "balanced_action_error": np.nan,
                "n_rows": len(y)}
    unique = np.unique(s)
    epsilon = max(1e-12, float(np.ptp(unique)) * 1e-12)
    candidates = np.concatenate(([unique.min() - epsilon], unique,
                                 [unique.max() + epsilon]))
    options = []
    harmful = y == 0
    beneficial = y == 1
    for threshold in candidates:
        accepted = s > threshold
        har = float(accepted[harmful].mean())
        brr = float((~accepted[beneficial]).mean())
        coverage = float(accepted.mean())
        options.append((0.5 * (har + brr), har + brr, -coverage,
                        float(threshold), har, brr, coverage))
    best = min(options)
    return {
        "threshold": best[3], "har": best[4], "brr": best[5],
        "coverage": best[6], "balanced_action_error": best[0],
        "n_rows": int(len(y)),
    }


def action_rates(labels: Sequence, scores: Sequence, threshold: float) -> dict:
    y, s = clean_binary_arrays(labels, scores)
    if len(y) == 0 or not math.isfinite(float(threshold)):
        return {"har": np.nan, "brr": np.nan, "coverage": np.nan,
                "balanced_action_error": np.nan, "n_rows": len(y),
                "n_harmful": int((y == 0).sum()),
                "n_beneficial": int((y == 1).sum())}
    accepted = s > threshold
    harmful = y == 0
    beneficial = y == 1
    har = float(accepted[harmful].mean()) if harmful.any() else np.nan
    brr = float((~accepted[beneficial]).mean()) if beneficial.any() else np.nan
    balanced = float(np.nanmean([har, brr])) if np.isfinite([har, brr]).any() else np.nan
    return {
        "har": har, "brr": brr, "coverage": float(accepted.mean()),
        "balanced_action_error": balanced, "n_rows": int(len(y)),
        "n_harmful": int(harmful.sum()), "n_beneficial": int(beneficial.sum()),
    }


def holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    adjusted = np.full_like(values, np.nan)
    finite = np.flatnonzero(np.isfinite(values))
    if len(finite) == 0:
        return adjusted
    order = finite[np.argsort(values[finite])]
    running = 0.0
    m = len(order)
    for rank, index in enumerate(order):
        candidate = min(1.0, (m - rank) * values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def formal_full_regions(region: pd.DataFrame) -> pd.DataFrame:
    if region.empty:
        return region.copy()
    allowed = {
        "P0_AB_CLEAN_MECHANISM", "P0_C1_GAUSSIAN",
        "P0_C1_CHANNEL_DROPOUT", "P0_C2_TRANSLATION",
        "P0_C3_DONOR_MIX",
    }
    return region[
        region["experiment_id"].isin(allowed) &
        region["model_variant"].eq("full")].copy()


def clean_full_regions(region: pd.DataFrame) -> pd.DataFrame:
    frame = formal_full_regions(region)
    return frame[
        frame["experiment_id"].eq("P0_AB_CLEAN_MECHANISM") &
        frame["corruption_type"].isin(["none", "clean"]) &
        pd.to_numeric(frame["severity"], errors="coerce").fillna(0).eq(0)
    ].copy()


def target_rows(region: pd.DataFrame, target_name: str) -> pd.DataFrame:
    label, utility, roi_type = TARGETS[target_name]
    frame = region[region["region_type"].eq(roi_type)].copy()
    frame[label] = pd.to_numeric(frame[label], errors="coerce")
    frame[utility] = pd.to_numeric(frame[utility], errors="coerce")
    return frame[np.isfinite(frame[label]) & np.isfinite(frame[utility])]


def analyse_p0_a(region: pd.DataFrame, results_root: Path,
                 n_boot: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    clean = target_rows(clean_full_regions(region), "component")
    # Q_component is output-level and intentionally repeated for level-specific
    # trust analysis.  Collapse those copies for the problem-existence result.
    key = ["dataset", "case_id", "vessel_class", "vessel_name",
           "vessel_group", "size_bin", "Q_component", "B_component"]
    rows = clean[key].drop_duplicates().copy()
    rows["harmful"] = 1 - rows["B_component"].astype(int)
    rows = rows.sort_values(["dataset", "case_id", "vessel_class"]).reset_index(drop=True)
    atomic_csv(results_root / "06_summary_tables" / "Figure_R1_source.csv", rows)

    summaries: List[dict] = []
    grouping_sets = [
        ("dataset_vessel_group", ["dataset", "vessel_group"]),
        ("dataset", ["dataset"]),
        ("vessel_group", ["vessel_group"]),
        ("overall", []),
    ]
    for scope, columns in grouping_sets:
        iterator = rows.groupby(columns, dropna=False) if columns else [((), rows)]
        for keys, group in iterator:
            if not isinstance(keys, tuple):
                keys = (keys,)
            base = {"scope": scope}
            base.update(dict(zip(columns, keys)))
            q_summary = summarize_clustered(
                group, "Q_component", n_boot,
                ["P0A", scope, *keys, "Q_component"])
            harmful_summary = summarize_clustered(
                group, "harmful", n_boot,
                ["P0A", scope, *keys, "harmful"], binary=True)
            base.update({"utility_" + key: value for key, value in q_summary.items()})
            base.update({"harmful_rate_" + key: value
                         for key, value in harmful_summary.items()})
            summaries.append(base)
    summary = pd.DataFrame(summaries)
    atomic_csv(results_root / "06_summary_tables" / "p0_a_utility_summary.csv", summary)

    case_rates = rows.groupby(["dataset", "case_id"], as_index=False).agg(
        harmful_update_rate=("harmful", "mean"),
        mean_component_utility=("Q_component", "mean"),
        n_vessels=("vessel_class", "size"),
        n_small_vessels=("vessel_group", lambda x: int((x == "communicating_or_variant").sum())),
    )
    atomic_csv(results_root / "04_case_metrics" / "p0_a_case_harmful_rates.csv", case_rates)
    return rows, summary


def condition_labels(frame: pd.DataFrame) -> pd.Series:
    corruption = frame["corruption_type"].fillna("none").astype(str)
    return np.where(corruption.isin(["none", "clean"]), "clean", corruption)


def analyse_p0_b(region: pd.DataFrame, results_root: Path,
                 n_boot: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    full = formal_full_regions(region)
    full["condition"] = condition_labels(full)
    metric_rows: List[dict] = []
    calibration_rows: List[dict] = []
    conditions = ["clean", "all_corruption"] + sorted(
        value for value in full["condition"].dropna().unique() if value != "clean")

    for target_name, (label_col, _, _) in TARGETS.items():
        target = target_rows(full, target_name)
        for condition in conditions:
            subset = (target[target["condition"].ne("clean")]
                      if condition == "all_corruption"
                      else target[target["condition"].eq(condition)])
            for (dataset, level), group in subset.groupby(
                    ["dataset", "feature_level"], dropna=False):
                for signal_name, transform in SIGNALS.items():
                    probability = transform(group)
                    compute_ci = (target_name == "component" and
                                  condition in {"clean", "all_corruption"})
                    bundle = metric_bundle(
                        group, label_col, probability, n_boot,
                        ["P0B", target_name, condition, dataset, level, signal_name],
                        compute_ci=compute_ci)
                    metric_rows.append({
                        "target": target_name,
                        "condition": condition,
                        "dataset": dataset,
                        "feature_level": level,
                        "signal": signal_name,
                        "higher_score_means": "beneficial_update",
                        "ci_scope": ("primary_component_clean_or_pooled_corruption"
                                     if compute_ci else "point_estimate_only"),
                        **bundle,
                    })

                    if condition == "clean":
                        cal = group[["case_id", label_col]].copy()
                        cal["probability"] = pd.to_numeric(probability, errors="coerce")
                        cal[label_col] = pd.to_numeric(cal[label_col], errors="coerce")
                        cal = cal[np.isfinite(cal[label_col]) & np.isfinite(cal["probability"])]
                        cal["probability"] = cal["probability"].clip(0.0, 1.0)
                        cal["bin"] = pd.cut(
                            cal["probability"], bins=np.linspace(0, 1, 11),
                            labels=False, include_lowest=True).astype("Int64")
                        for bin_id, bin_frame in cal.groupby("bin", dropna=False):
                            calibration_rows.append({
                                "target": target_name,
                                "dataset": dataset,
                                "feature_level": level,
                                "signal": signal_name,
                                "bin": int(bin_id) if pd.notna(bin_id) else None,
                                "bin_lower": (int(bin_id) / 10.0) if pd.notna(bin_id) else np.nan,
                                "bin_upper": ((int(bin_id) + 1) / 10.0) if pd.notna(bin_id) else np.nan,
                                "mean_predicted_benefit": float(bin_frame["probability"].mean()),
                                "observed_beneficial_rate": float(bin_frame[label_col].mean()),
                                "n_rows": int(len(bin_frame)),
                                "n_cases": int(bin_frame["case_id"].nunique()),
                            })

    probability_region = region[
        region["experiment_id"].eq("P0_B_PROBABILITY_SIGNALS") &
        region["model_variant"].eq("full")].copy()
    if (not probability_region.empty and
            {"confidence", "entropy"} <= set(probability_region.columns)):
        probability_region["condition"] = condition_labels(probability_region)
        probability_conditions = ["clean", "all_corruption"] + sorted(
            value for value in probability_region["condition"].dropna().unique()
            if value != "clean")
        for target_name, (label_col, _, _) in TARGETS.items():
            target = target_rows(probability_region, target_name)
            for condition in probability_conditions:
                subset = (target[target["condition"].ne("clean")]
                          if condition == "all_corruption"
                          else target[target["condition"].eq(condition)])
                for (dataset, level), group in subset.groupby(
                        ["dataset", "feature_level"], dropna=False):
                    for signal_name, transform in PROBABILITY_SIGNALS.items():
                        probability = transform(group)
                        compute_ci = (target_name == "component" and
                                      condition in {"clean", "all_corruption"})
                        bundle = metric_bundle(
                            group, label_col, probability, n_boot,
                            ["P0B_probability", target_name, condition,
                             dataset, level, signal_name],
                            compute_ci=compute_ci)
                        metric_rows.append({
                            "target": target_name, "condition": condition,
                            "dataset": dataset, "feature_level": level,
                            "signal": signal_name,
                            "higher_score_means": "beneficial_update",
                            "ci_scope": ("primary_component_clean_or_pooled_corruption"
                                         if compute_ci else "point_estimate_only"),
                            **bundle,
                        })
                        if condition == "clean":
                            cal = group[["case_id", label_col]].copy()
                            cal["probability"] = pd.to_numeric(
                                probability, errors="coerce")
                            cal[label_col] = pd.to_numeric(
                                cal[label_col], errors="coerce")
                            cal = cal[np.isfinite(cal[label_col]) &
                                      np.isfinite(cal["probability"])]
                            cal["probability"] = cal["probability"].clip(0.0, 1.0)
                            cal["bin"] = pd.cut(
                                cal["probability"], bins=np.linspace(0, 1, 11),
                                labels=False, include_lowest=True).astype("Int64")
                            for bin_id, bin_frame in cal.groupby("bin", dropna=False):
                                calibration_rows.append({
                                    "target": target_name, "dataset": dataset,
                                    "feature_level": level, "signal": signal_name,
                                    "bin": int(bin_id) if pd.notna(bin_id) else None,
                                    "bin_lower": int(bin_id) / 10.0 if pd.notna(bin_id) else np.nan,
                                    "bin_upper": (int(bin_id) + 1) / 10.0 if pd.notna(bin_id) else np.nan,
                                    "mean_predicted_benefit": float(
                                        bin_frame["probability"].mean()),
                                    "observed_beneficial_rate": float(
                                        bin_frame[label_col].mean()),
                                    "n_rows": int(len(bin_frame)),
                                    "n_cases": int(bin_frame["case_id"].nunique()),
                                })

    validity = pd.DataFrame(metric_rows)
    calibration = pd.DataFrame(calibration_rows)
    atomic_csv(results_root / "06_summary_tables" / "p0_b_signal_validity.csv", validity)
    atomic_csv(results_root / "06_summary_tables" / "p0_b_calibration_bins.csv", calibration)

    component_clean = target_rows(clean_full_regions(region), "component")
    phase_ug = make_phase_table(component_clean, "G")
    phase_ufdd = make_phase_table(component_clean, "FDD")
    atomic_csv(results_root / "06_summary_tables" / "p0_b_phase_UxG.csv", phase_ug)
    atomic_csv(results_root / "06_summary_tables" / "p0_b_phase_UxFDD.csv", phase_ufdd)
    return validity, calibration, phase_ug, phase_ufdd


def source_ecdf_percentile(values: pd.Series, reference: np.ndarray) -> np.ndarray:
    output = np.full(len(values), np.nan)
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(numeric)
    if len(reference) == 0:
        return output
    output[valid] = np.searchsorted(reference, numeric[valid], side="right") / len(reference)
    return output


def quartile_from_percentile(percentile: np.ndarray) -> pd.Series:
    values = np.ceil(np.clip(percentile, 1e-12, 1.0) * 4)
    values[~np.isfinite(percentile)] = np.nan
    return pd.Series(values, dtype="Int64")


def make_phase_table(component_clean: pd.DataFrame, second_signal: str) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    for level, level_frame in component_clean.groupby("feature_level"):
        source = level_frame[level_frame["dataset"].eq("TopCoW_val")]
        u_reference = np.sort(pd.to_numeric(source["U"], errors="coerce").dropna().to_numpy())
        s_reference = np.sort(pd.to_numeric(source[second_signal], errors="coerce").dropna().to_numpy())
        work = level_frame.copy()
        work["U_quartile"] = quartile_from_percentile(
            source_ecdf_percentile(work["U"], u_reference)).to_numpy()
        work[second_signal + "_quartile"] = quartile_from_percentile(
            source_ecdf_percentile(work[second_signal], s_reference)).to_numpy()
        rows.append(work)
    if not rows:
        return pd.DataFrame()
    work = pd.concat(rows, ignore_index=True)
    work["harmful"] = 1 - work["B_component"].astype(int)
    grouped = work.groupby(
        ["dataset", "feature_level", "U_quartile", second_signal + "_quartile"],
        dropna=False, as_index=False).agg(
            sample_count=("B_component", "size"),
            case_count=("case_id", "nunique"),
            harmful_rate=("harmful", "mean"),
            mean_utility=("Q_component", "mean"),
            mean_T_formula=(FORMULA_TRUST, "mean"),
            mean_A_impl=(IMPLEMENTED_ACCEPTANCE, "mean"),
            small_vessel_proportion=("vessel_group", lambda x: float(
                (x == "communicating_or_variant").mean())),
        )
    grouped["second_signal"] = second_signal
    return grouped


def analyse_p0_f_and_g(region: pd.DataFrame, case: pd.DataFrame,
                       results_root: Path, n_boot: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    full = formal_full_regions(region)
    full["condition"] = condition_labels(full)
    component = target_rows(full, "component")
    source_clean = component[
        component["dataset"].eq("TopCoW_val") &
        component["condition"].eq("clean")]

    thresholds: List[dict] = []
    for level, level_frame in source_clean.groupby("feature_level"):
        for signal_name, transform in SIGNALS.items():
            selected = choose_source_threshold(
                level_frame["B_component"], transform(level_frame))
            thresholds.append({
                "target": "component", "feature_level": level,
                "signal": signal_name,
                "selection_dataset": "TopCoW_val",
                "selection_condition": "clean",
                "selection_rule": "minimize_balanced_action_error_then_total_error_then_max_coverage_then_lower_tau",
                **selected,
            })
    threshold_table = pd.DataFrame(thresholds)
    atomic_csv(results_root / "06_summary_tables" / "p0_f_source_thresholds.csv",
               threshold_table)

    action_rows: List[dict] = []
    curve_rows: List[pd.DataFrame] = []
    quadrant_rows: List[dict] = []
    row_level_quadrants: List[pd.DataFrame] = []
    for threshold_row in threshold_table.itertuples(index=False):
        level_frame = component[component["feature_level"].eq(threshold_row.feature_level)].copy()
        level_frame["score"] = SIGNALS[threshold_row.signal](level_frame)
        level_frame["accepted"] = level_frame["score"] > threshold_row.threshold
        level_frame["harmful"] = 1 - level_frame["B_component"].astype(int)
        level_frame["utility_quadrant"] = np.select(
            [
                level_frame["accepted"] & level_frame["B_component"].eq(1),
                (~level_frame["accepted"]) & level_frame["B_component"].eq(0),
                (~level_frame["accepted"]) & level_frame["B_component"].eq(1),
            ],
            ["correct_acceptance", "correct_suppression", "over_suppression"],
            default="incorrect_acceptance")

        condition_sets = [("clean", level_frame[level_frame["condition"].eq("clean")]),
                          ("all_corruption", level_frame[level_frame["condition"].ne("clean")])]
        condition_sets.extend(
            (name, level_frame[level_frame["condition"].eq(name)])
            for name in sorted(level_frame["condition"].unique()) if name != "clean")
        for condition, condition_frame in condition_sets:
            for dataset, group in condition_frame.groupby("dataset"):
                rates = action_rates(group["B_component"], group["score"],
                                     threshold_row.threshold)
                base = {
                    "target": "component", "signal": threshold_row.signal,
                    "feature_level": threshold_row.feature_level,
                    "threshold": threshold_row.threshold,
                    "dataset": dataset, "condition": condition,
                    **rates,
                    "n_cases": int(group["case_id"].nunique()),
                }
                intervals = bootstrap_action_rate_cis(
                    group[["case_id", "B_component", "score"]],
                    threshold_row.threshold, n_boot,
                    stable_int(BOOTSTRAP_SEED, "P0F", threshold_row.signal,
                               threshold_row.feature_level, dataset, condition,
                               "joint_action_rates"))
                for metric_name in ["har", "brr", "coverage", "balanced_action_error"]:
                    low, high, valid = intervals[metric_name]
                    base[metric_name + "_ci_low"] = low
                    base[metric_name + "_ci_high"] = high
                    base[metric_name + "_n_boot_valid"] = valid
                action_rows.append(base)

                curve = risk_coverage_curve(group["B_component"], group["score"])
                curve.insert(0, "condition", condition)
                curve.insert(0, "dataset", dataset)
                curve.insert(0, "feature_level", threshold_row.feature_level)
                curve.insert(0, "signal", threshold_row.signal)
                curve_rows.append(curve)

                counts = group.groupby("utility_quadrant").agg(
                    n_rows=("B_component", "size"), n_cases=("case_id", "nunique")).reset_index()
                counts["share"] = counts["n_rows"] / counts["n_rows"].sum()
                for count in counts.itertuples(index=False):
                    quadrant_rows.append({
                        "signal": threshold_row.signal,
                        "feature_level": threshold_row.feature_level,
                        "threshold": threshold_row.threshold,
                        "dataset": dataset,
                        "condition": condition,
                        "utility_quadrant": count.utility_quadrant,
                        "n_rows": int(count.n_rows),
                        "n_cases": int(count.n_cases),
                        "share": float(count.share),
                    })
        if threshold_row.signal == FORMULA_TRUST:
            row_level_quadrants.append(level_frame)

    action = pd.DataFrame(action_rows)
    curves = pd.concat(curve_rows, ignore_index=True) if curve_rows else pd.DataFrame()
    quadrants = pd.DataFrame(quadrant_rows)
    atomic_csv(results_root / "06_summary_tables" / "p0_f_har_brr.csv", action)
    atomic_csv(results_root / "06_summary_tables" / "p0_f_risk_coverage.csv", curves)
    atomic_csv(results_root / "06_summary_tables" / "p0_g_quadrants.csv", quadrants)

    quadrant_detail = (pd.concat(row_level_quadrants, ignore_index=True)
                       if row_level_quadrants else pd.DataFrame())
    selections = select_failure_cases(quadrant_detail, case)
    atomic_csv(results_root / "08_failure_cases" / "predeclared_case_selection.csv",
               selections)
    return threshold_table, action, curves, quadrants, selections


def select_failure_cases(quadrant_detail: pd.DataFrame,
                         case: pd.DataFrame) -> pd.DataFrame:
    candidates: List[dict] = []
    clean = quadrant_detail[quadrant_detail["condition"].eq("clean")].copy()
    if not clean.empty:
        clean["formula_trust_utility_disagreement"] = np.abs(
            pd.to_numeric(clean[FORMULA_TRUST], errors="coerce") -
            pd.to_numeric(clean["B_component"], errors="coerce"))
        per_case = clean.groupby(["dataset", "case_id"], as_index=False).agg(
            formula_trust_utility_disagreement=(
                "formula_trust_utility_disagreement", "mean"),
            har=("utility_quadrant", lambda x: float(
                (x == "incorrect_acceptance").sum() /
                max(1, ((x == "incorrect_acceptance") |
                        (x == "correct_suppression")).sum()))),
            brr=("utility_quadrant", lambda x: float(
                (x == "over_suppression").sum() /
                max(1, ((x == "over_suppression") |
                        (x == "correct_acceptance")).sum()))),
            n_regions=("utility_quadrant", "size"),
        )
        for dataset, group in per_case.groupby("dataset"):
            for rule, column, n in [
                    ("top5_formula_trust_utility_disagreement",
                     "formula_trust_utility_disagreement", 5),
                    ("highest_HAR", "har", 1),
                    ("highest_BRR", "brr", 1)]:
                chosen = group.sort_values([column, "case_id"], ascending=[False, True]).head(n)
                for rank, row in enumerate(chosen.itertuples(index=False), start=1):
                    candidates.append({
                        "dataset": dataset, "case_id": row.case_id,
                        "selection_rule": rule, "rank": rank,
                        "selection_value": float(getattr(row, column)),
                        "availability": "selected",
                    })

    if not case.empty:
        clean_case = case[
            case["corruption_type"].isin(["none", "clean"]) &
            pd.to_numeric(case["severity"], errors="coerce").fillna(0).eq(0)]
        full = clean_case[clean_case["model_variant"].eq("full")][
            ["dataset", "case_id", "det_f1"]].rename(columns={"det_f1": "full_det_f1"})
        noaux = clean_case[clean_case["model_variant"].eq("no_aux")][
            ["dataset", "case_id", "det_f1"]].rename(columns={"det_f1": "noaux_det_f1"})
        paired = full.merge(noaux, on=["dataset", "case_id"], how="inner")
        paired["det_f1_delta_vs_no_aux"] = paired["full_det_f1"] - paired["noaux_det_f1"]
        for dataset, group in paired.groupby("dataset"):
            for rule, ascending in [("largest_DetF1_improvement", False),
                                    ("largest_DetF1_degradation", True)]:
                chosen = group.sort_values(
                    ["det_f1_delta_vs_no_aux", "case_id"],
                    ascending=[ascending, True]).head(1)
                for row in chosen.itertuples(index=False):
                    candidates.append({
                        "dataset": dataset, "case_id": row.case_id,
                        "selection_rule": rule, "rank": 1,
                        "selection_value": float(row.det_f1_delta_vs_no_aux),
                        "availability": "selected",
                    })

    candidates.append({
        "dataset": "all", "case_id": "",
        "selection_rule": "representative_common_mode_failure", "rank": 1,
        "selection_value": np.nan,
        "availability": "unavailable_common_mode_intervention_not_run",
    })
    return pd.DataFrame(candidates)


def mechanism_case_level(region: pd.DataFrame, case: pd.DataFrame) -> pd.DataFrame:
    full_region = formal_full_regions(region)
    component = target_rows(full_region, "component")
    component["harmful_rate"] = 1 - component["B_component"].astype(int)
    id_cols = ["dataset", "case_id", "realization", "corruption_type", "severity"]
    signal_level = component.groupby(id_cols + ["feature_level"], as_index=False).agg(
        Q_component=("Q_component", "mean"),
        harmful_rate=("harmful_rate", "mean"),
        U=("U", "mean"), G=("G", "mean"),
        T_formula=(FORMULA_TRUST, "mean"),
        A_impl=(IMPLEMENTED_ACCEPTANCE, "mean"),
        accepted_delta_l1=("accepted_delta_l1", "mean"),
        n_component_rows=("B_component", "size"),
        n_vessels=("vessel_class", "nunique"),
    )
    signal_level["1-U"] = 1.0 - signal_level["U"]
    signal_all = signal_level.groupby(id_cols, as_index=False).agg(
        Q_component=("Q_component", "mean"),
        harmful_rate=("harmful_rate", "mean"),
        U=("U", "mean"), G=("G", "mean"),
        T_formula=(FORMULA_TRUST, "mean"),
        A_impl=(IMPLEMENTED_ACCEPTANCE, "mean"),
        **{"1-U": ("1-U", "mean")},
        accepted_delta_l1=("accepted_delta_l1", "mean"),
        n_component_rows=("n_component_rows", "sum"),
        n_vessels=("n_vessels", "max"),
    )
    signal_all["feature_level"] = "all"
    signals = pd.concat([signal_level, signal_all], ignore_index=True)

    allowed = {
        "P0_AB_CLEAN_MECHANISM", "P0_C1_GAUSSIAN",
        "P0_C1_CHANNEL_DROPOUT", "P0_C2_TRANSLATION",
        "P0_C3_DONOR_MIX",
    }
    full_case = case[
        case["experiment_id"].isin(allowed) & case["model_variant"].eq("full")].copy()
    performance_cols = id_cols + [
        "dice", "cldice", "hd95", "beta0_error", "clsavg", "cls_beta0",
        "det_f1", "runtime_sec", "peak_gpu_mem_mb"]
    performance = full_case[[column for column in performance_cols
                             if column in full_case.columns]].copy()
    merged = signals.merge(performance, on=id_cols, how="left", validate="many_to_one")

    corrupted = merged[~merged["corruption_type"].isin(["none", "clean"])].copy()
    clean = merged[
        merged["corruption_type"].isin(["none", "clean"]) &
        pd.to_numeric(merged["severity"], errors="coerce").fillna(0).eq(0)].copy()
    pieces: List[pd.DataFrame] = []
    for corruption_type in sorted(corrupted["corruption_type"].dropna().unique()):
        current = corrupted[corrupted["corruption_type"].eq(corruption_type)].copy()
        baseline = clean.copy()
        baseline["corruption_type"] = corruption_type
        baseline["severity"] = 0.0
        baseline["realization"] = -1
        pieces.extend([baseline, current])
    output = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
    return output.sort_values(
        ["dataset", "corruption_type", "feature_level", "severity",
         "case_id", "realization"]).reset_index(drop=True)


def analyse_p0_c(region: pd.DataFrame, case: pd.DataFrame,
                 results_root: Path, n_boot: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    case_level = mechanism_case_level(region, case)
    atomic_csv(results_root / "04_case_metrics" / "p0_c_mechanism_case_level.csv",
               case_level)
    if case_level.empty:
        empty = pd.DataFrame()
        atomic_csv(results_root / "06_summary_tables" / "Figure_R3_source.csv", empty)
        atomic_csv(results_root / "06_summary_tables" / "p0_c_degradation_slopes.csv", empty)
        return case_level, empty, empty

    metrics = [
        "Q_component", "harmful_rate", "U", "1-U", "G",
        FORMULA_TRUST, IMPLEMENTED_ACCEPTANCE,
        "accepted_delta_l1", "dice", "cldice", "hd95", "beta0_error",
        "clsavg", "cls_beta0", "det_f1",
    ]
    metrics = [metric for metric in metrics if metric in case_level]
    summary_rows: List[dict] = []
    for keys, group in case_level.groupby(
            ["dataset", "corruption_type", "feature_level", "severity"],
            dropna=False):
        dataset, corruption_type, level, severity = keys
        for metric in metrics:
            summary = summarize_clustered(
                group, metric, n_boot,
                ["P0C", dataset, corruption_type, level, severity, metric],
                binary=(metric == "harmful_rate"))
            summary_rows.append({
                "dataset": dataset, "corruption_type": corruption_type,
                "feature_level": level, "severity": float(severity),
                "metric": metric, **summary,
            })
    summary_table = pd.DataFrame(summary_rows)
    atomic_csv(results_root / "06_summary_tables" / "Figure_R3_source.csv",
               summary_table)

    slope_rows: List[dict] = []
    averaged = case_level.groupby(
        ["dataset", "corruption_type", "feature_level", "case_id", "severity"],
        as_index=False)[metrics].mean(numeric_only=True)
    for keys, group in averaged.groupby(
            ["dataset", "corruption_type", "feature_level"], dropna=False):
        dataset, corruption_type, level = keys
        for metric in metrics:
            per_case = []
            for case_id, case_frame in group.groupby("case_id"):
                x = pd.to_numeric(case_frame["severity"], errors="coerce").to_numpy(dtype=float)
                y = pd.to_numeric(case_frame[metric], errors="coerce").to_numpy(dtype=float)
                valid = np.isfinite(x) & np.isfinite(y)
                if valid.sum() < 2 or len(np.unique(x[valid])) < 2:
                    continue
                slope = float(np.polyfit(x[valid], y[valid], 1)[0])
                per_case.append({"case_id": case_id, "slope": slope})
            slope_frame = pd.DataFrame(per_case)
            if slope_frame.empty:
                continue
            result = summarize_clustered(
                slope_frame, "slope", n_boot,
                ["P0C_slope", dataset, corruption_type, level, metric])
            slope_rows.append({
                "dataset": dataset, "corruption_type": corruption_type,
                "feature_level": level, "metric": metric,
                "severity_unit": {
                    "gaussian": "channel_std_multiplier",
                    "channel_dropout": "channel_fraction",
                    "translation": "feature_voxel",
                    "donor_mix": "mixing_weight",
                }.get(corruption_type, "unknown"),
                **{"slope_" + key: value for key, value in result.items()},
            })
    slopes = pd.DataFrame(slope_rows)
    atomic_csv(results_root / "06_summary_tables" / "p0_c_degradation_slopes.csv",
               slopes)
    return case_level, summary_table, slopes


def paired_bootstrap_difference(paired: pd.DataFrame, value_col: str,
                                reference_col: str, n_boot: int,
                                seed_parts: Sequence[object]) -> dict:
    work = paired[["case_id", value_col, reference_col]].copy()
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    work[reference_col] = pd.to_numeric(work[reference_col], errors="coerce")
    work = work[np.isfinite(work[value_col]) & np.isfinite(work[reference_col])]
    work["difference"] = work[value_col] - work[reference_col]
    if work.empty:
        return {"paired_difference": np.nan, "paired_ci_low": np.nan,
                "paired_ci_high": np.nan, "n_paired_cases": 0,
                "wilcoxon_p": np.nan}
    low, high, _ = bootstrap_mean_ci(
        work[["case_id", "difference"]], "difference", n_boot,
        stable_int(BOOTSTRAP_SEED, *seed_parts))
    case_differences = work.groupby("case_id")["difference"].mean()
    try:
        if np.allclose(case_differences, 0):
            p_value = 1.0
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                p_value = float(stats.wilcoxon(case_differences).pvalue)
    except ValueError:
        p_value = np.nan
    return {
        "paired_difference": float(work["difference"].mean()),
        "paired_ci_low": low, "paired_ci_high": high,
        "n_paired_cases": int(work["case_id"].nunique()),
        "wilcoxon_p": p_value,
    }


def analyse_variants(case: pd.DataFrame, results_root: Path,
                     n_boot: int) -> pd.DataFrame:
    if case.empty:
        output = pd.DataFrame()
        atomic_csv(results_root / "06_summary_tables" / "Table_R1_components.csv", output)
        return output
    fixed_full = case[case["experiment_id"].eq("P0_AB_CLEAN_MECHANISM")].copy()
    fixed_other = case[case["experiment_id"].isin([
        "P0_D_FIXED_CKPT_INTERVENTIONS", "P0_D_PRODUCT_FIXED_CLEAN"
    ])].copy()
    fixed = pd.concat([fixed_full, fixed_other], ignore_index=True)
    fixed = fixed[
        fixed["corruption_type"].isin(["none", "clean"]) &
        pd.to_numeric(fixed["severity"], errors="coerce").fillna(0).eq(0)]
    fixed["analysis_type"] = "fixed_checkpoint_intervention"

    trained = case[case["experiment_id"].str.startswith("P0_D_TRAINED_EVAL", na=False)].copy()
    trained = trained[
        trained["corruption_type"].isin(["none", "clean"]) &
        pd.to_numeric(trained["severity"], errors="coerce").fillna(0).eq(0)]
    trained["analysis_type"] = "independently_trained_variant"
    all_variants = pd.concat([fixed, trained], ignore_index=True, sort=False)
    metrics = [metric for metric in ["dice", "cldice", "clsavg", "det_f1", "hd95", "beta0_error"]
               if metric in all_variants]
    rows: List[dict] = []
    for analysis_type, analysis_frame in all_variants.groupby("analysis_type"):
        for dataset, dataset_frame in analysis_frame.groupby("dataset"):
            reference = dataset_frame[dataset_frame["model_variant"].eq("full")]
            for variant, variant_frame in dataset_frame.groupby("model_variant"):
                for metric in metrics:
                    values = pd.to_numeric(variant_frame[metric], errors="coerce")
                    valid = variant_frame[np.isfinite(values)].copy()
                    if valid.empty:
                        continue
                    base = {
                        "analysis_type": analysis_type, "dataset": dataset,
                        "variant": variant, "metric": metric,
                        "mean": float(pd.to_numeric(valid[metric]).mean()),
                        "sd": float(pd.to_numeric(valid[metric]).std(ddof=1)),
                        "median": float(pd.to_numeric(valid[metric]).median()),
                        "q25": float(pd.to_numeric(valid[metric]).quantile(0.25)),
                        "q75": float(pd.to_numeric(valid[metric]).quantile(0.75)),
                        "n_case_seed_rows": int(len(valid)),
                        "n_cases": int(valid["case_id"].nunique()),
                        "n_seeds": int(valid["seed"].nunique()),
                    }
                    join_keys = ["case_id"]
                    if (analysis_type == "independently_trained_variant" and
                            "seed" in valid and "seed" in reference):
                        join_keys.append("seed")
                    paired = valid[join_keys + [metric]].rename(
                        columns={metric: "variant_value"}).merge(
                            reference[join_keys + [metric]].rename(
                                columns={metric: "reference_value"}),
                            on=join_keys, how="inner")
                    base.update(paired_bootstrap_difference(
                        paired, "variant_value", "reference_value", n_boot,
                        ["P0D", analysis_type, dataset, variant, metric]))
                    rows.append(base)
    output = pd.DataFrame(rows)
    if not output.empty:
        output["wilcoxon_p_holm"] = np.nan
        for _, indices in output.groupby(
                ["analysis_type", "dataset", "metric"]).groups.items():
            output.loc[list(indices), "wilcoxon_p_holm"] = holm_adjust(
                output.loc[list(indices), "wilcoxon_p"].to_numpy())
        variant_order = {name: index for index, name in enumerate(
            ["no_aux", "residual", "u_only", "g_only", "additive", "full",
             "product"])}
        output["_variant_order"] = output["variant"].map(variant_order).fillna(99)
        output = output.sort_values(
            ["analysis_type", "dataset", "metric", "_variant_order"]).drop(
                columns="_variant_order").reset_index(drop=True)
    atomic_csv(results_root / "06_summary_tables" / "Table_R1_components.csv", output)
    return output


def analyse_trained_corruption(case: pd.DataFrame, results_root: Path,
                               n_boot: int) -> pd.DataFrame:
    clean = case[
        case["experiment_id"].eq("P0_D_TRAINED_EVAL_CLEAN") &
        case["dataset"].eq("TopCoW_val")].copy()
    corrupted = case[
        case["experiment_id"].eq("P0_D_TRAINED_EVAL_CORRUPTION") &
        case["dataset"].eq("TopCoW_val")].copy()
    output_path = results_root / "06_summary_tables" / "p0_d_trained_corruption.csv"
    if clean.empty or corrupted.empty:
        output = pd.DataFrame()
        atomic_csv(output_path, output)
        return output
    metrics = [metric for metric in ["dice", "cldice", "clsavg", "det_f1",
                                      "hd95", "beta0_error"]
               if metric in corrupted and metric in clean]
    corrupt_mean = corrupted.groupby(
        ["model_variant", "seed", "case_id", "corruption_type", "severity"],
        as_index=False)[metrics].mean(numeric_only=True)
    clean_mean = clean.groupby(
        ["model_variant", "seed", "case_id"], as_index=False)[metrics].mean(numeric_only=True)
    rows = []
    for keys, group in corrupt_mean.groupby(
            ["model_variant", "corruption_type", "severity"], dropna=False):
        variant, corruption_type, severity = keys
        reference = clean_mean[clean_mean["model_variant"].eq(variant)]
        paired = group.merge(reference, on=["model_variant", "seed", "case_id"],
                             suffixes=("_corrupted", "_clean"), how="inner")
        for metric in metrics:
            paired["difference"] = (paired[metric + "_corrupted"] -
                                    paired[metric + "_clean"])
            summary = summarize_clustered(
                paired[["case_id", "difference"]], "difference", n_boot,
                ["P0D_trained_corruption", variant, corruption_type,
                 severity, metric])
            rows.append({
                "dataset": "TopCoW_val", "variant": variant,
                "corruption_type": corruption_type, "severity": severity,
                "metric": metric,
                "corrupted_mean": float(paired[metric + "_corrupted"].mean()),
                "clean_mean": float(paired[metric + "_clean"].mean()),
                **{"delta_" + key: value for key, value in summary.items()},
                "n_seeds": int(paired["seed"].nunique()),
            })
    output = pd.DataFrame(rows)
    atomic_csv(output_path, output)
    return output


def analyse_product_fixed_corruption(case: pd.DataFrame, results_root: Path,
                                     n_boot: int) -> pd.DataFrame:
    """Compare truly executed G(1-U) against submitted A_impl on one checkpoint."""
    output_path = (results_root / "06_summary_tables" /
                   "Table_R1_product_fixed_corruption.csv")
    product = case[case["experiment_id"].eq(
        "P0_D_PRODUCT_FIXED_CORRUPTION")].copy()
    submitted = case[case["experiment_id"].isin([
        "P0_C1_GAUSSIAN", "P0_C1_CHANNEL_DROPOUT",
        "P0_C2_TRANSLATION", "P0_C3_DONOR_MIX",
    ]) & case["model_variant"].eq("full")].copy()
    if product.empty or submitted.empty:
        output = pd.DataFrame()
        atomic_csv(output_path, output)
        return output
    keys = ["dataset", "case_id", "corruption_type", "severity",
            "realization"]
    metrics = [metric for metric in ["cldice", "det_f1"]
               if metric in product and metric in submitted]
    paired = product[keys + metrics].merge(
        submitted[keys + metrics], on=keys, how="inner",
        suffixes=("_product", "_submitted"))
    rows: List[dict] = []
    for group_keys, group in paired.groupby(
            ["dataset", "corruption_type", "severity"], dropna=False):
        dataset, corruption_type, severity = group_keys
        for metric in metrics:
            product_col = metric + "_product"
            submitted_col = metric + "_submitted"
            summary = paired_bootstrap_difference(
                group, product_col, submitted_col, n_boot,
                ["P0D_product_fixed", dataset, corruption_type,
                 severity, metric])
            rows.append({
                "analysis_type": "fixed_checkpoint_product_intervention",
                "dataset": dataset,
                "corruption_type": corruption_type,
                "severity": severity,
                "metric": metric,
                "product_mean": float(pd.to_numeric(
                    group[product_col], errors="coerce").mean()),
                "submitted_A_impl_mean": float(pd.to_numeric(
                    group[submitted_col], errors="coerce").mean()),
                **summary,
                "n_cases": int(group["case_id"].nunique()),
                "n_realizations": int(group["realization"].nunique()),
                "comparison_scope": (
                    "same submitted checkpoint; executed acceptance changed "
                    "from 1-(1-G)U to G(1-U)"),
            })
    output = pd.DataFrame(rows).sort_values(
        ["dataset", "corruption_type", "severity", "metric"]
    ).reset_index(drop=True)
    atomic_csv(output_path, output)
    return output


def analyse_variant_reliability(region: pd.DataFrame,
                                results_root: Path) -> pd.DataFrame:
    if region.empty:
        output = pd.DataFrame()
        atomic_csv(results_root / "06_summary_tables" /
                   "p0_d_variant_reliability.csv", output)
        return output
    fixed = region[region["experiment_id"].isin(
        ["P0_AB_CLEAN_MECHANISM", "P0_D_FIXED_CKPT_INTERVENTIONS",
         "P0_D_PRODUCT_FIXED_CLEAN"])].copy()
    fixed["analysis_type"] = "fixed_checkpoint_intervention"
    trained = region[
        region["experiment_id"].eq("P0_D_TRAINED_EVAL_CLEAN")].copy()
    trained["analysis_type"] = "independently_trained_variant"
    frame = pd.concat([fixed, trained], ignore_index=True, sort=False)
    frame = frame[
        frame["corruption_type"].isin(["none", "clean"]) &
        pd.to_numeric(frame["severity"], errors="coerce").fillna(0).eq(0)]
    frame = target_rows(frame, "component")
    rows = []
    for (analysis_type, variant, level), level_frame in frame.groupby(
            ["analysis_type", "model_variant", "feature_level"], dropna=False):
        source = level_frame[level_frame["dataset"].eq("TopCoW_val")]
        threshold = choose_source_threshold(
            source["B_component"], source[IMPLEMENTED_ACCEPTANCE])
        for dataset, group in level_frame.groupby("dataset"):
            rates = action_rates(group["B_component"],
                                 group[IMPLEMENTED_ACCEPTANCE],
                                 threshold["threshold"])
            rows.append({
                "analysis_type": analysis_type, "variant": variant,
                "dataset": dataset, "feature_level": level,
                "n_rows": int(len(group)),
                "n_cases": int(group["case_id"].nunique()),
                "n_seeds": int(group["seed"].nunique()),
                "beneficial_prevalence": float(group["B_component"].mean()),
                "action_signal": IMPLEMENTED_ACCEPTANCE,
                "A_impl_auprc_beneficial": safe_auprc(
                    group["B_component"], group[IMPLEMENTED_ACCEPTANCE]),
                "A_impl_auroc_beneficial": safe_auroc(
                    group["B_component"], group[IMPLEMENTED_ACCEPTANCE]),
                "A_impl_aurc": aurc(
                    group["B_component"], group[IMPLEMENTED_ACCEPTANCE]),
                "source_frozen_threshold": threshold["threshold"],
                "HAR": rates["har"], "BRR": rates["brr"],
                "coverage": rates["coverage"],
            })
    output = pd.DataFrame(rows)
    atomic_csv(results_root / "06_summary_tables" /
               "p0_d_variant_reliability.csv", output)
    return output


def build_table_r1(clean_metrics: pd.DataFrame,
                   reliability: pd.DataFrame,
                   trained_corruption: pd.DataFrame,
                   results_root: Path) -> pd.DataFrame:
    if clean_metrics.empty:
        output = pd.DataFrame()
        atomic_csv(results_root / "06_summary_tables" / "Table_R1_rebuttal.csv",
                   output)
        return output
    clean = clean_metrics.pivot_table(
        index=["analysis_type", "dataset", "variant"], columns="metric",
        values=["mean", "paired_difference", "paired_ci_low", "paired_ci_high",
                "n_cases", "n_seeds"], aggfunc="first")
    clean.columns = [f"clean_{metric}_{stat}" for stat, metric in clean.columns]
    clean = clean.reset_index()
    if not reliability.empty:
        reliability_summary = reliability.groupby(
            ["analysis_type", "dataset", "variant"], as_index=False).agg(
                A_impl_AUPRC_mean_levels=("A_impl_auprc_beneficial", "mean"),
                A_impl_AUROC_mean_levels=("A_impl_auroc_beneficial", "mean"),
                A_impl_AURC_mean_levels=("A_impl_aurc", "mean"),
                HAR_mean_levels=("HAR", "mean"),
                BRR_mean_levels=("BRR", "mean"),
                coverage_mean_levels=("coverage", "mean"),
            )
        clean = clean.merge(reliability_summary,
                            on=["analysis_type", "dataset", "variant"],
                            how="left")
    if not trained_corruption.empty:
        selected = trained_corruption[
            trained_corruption["metric"].isin(["cldice", "det_f1"])].copy()
        selected["column"] = (
            "trained_" + selected["corruption_type"].astype(str) + "_" +
            selected["metric"].astype(str) + "_delta")
        corruption_wide = selected.pivot_table(
            index="variant", columns="column", values="delta_mean",
            aggfunc="first").reset_index()
        mask = clean["analysis_type"].eq("independently_trained_variant")
        trained_rows = clean[mask].merge(corruption_wide, on="variant", how="left")
        clean = pd.concat([clean[~mask], trained_rows], ignore_index=True,
                          sort=False)
    clean["comparison_scope"] = np.where(
        clean["analysis_type"].eq("fixed_checkpoint_intervention"),
        "same submitted checkpoint with inference-time fusion intervention",
        "independent training with matched schedule and seed set")
    output = clean.sort_values(
        ["analysis_type", "dataset", "variant"]).reset_index(drop=True)
    atomic_csv(results_root / "06_summary_tables" / "Table_R1_rebuttal.csv",
               output)
    return output


def apply_plot_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.edgecolor": CHARCOAL,
        "axes.labelcolor": CHARCOAL,
        "xtick.color": CHARCOAL,
        "ytick.color": CHARCOAL,
        "text.color": CHARCOAL,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#D9DEE3",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.7,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "legend.frameon": False,
    })


def save_figure(fig: plt.Figure, results_root: Path, stem: str) -> None:
    folder = results_root / "07_figures"
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / f"{stem}.png", dpi=220, bbox_inches="tight",
                facecolor="white")
    fig.savefig(folder / f"{stem}.pdf", bbox_inches="tight",
                facecolor="white")
    plt.close(fig)


def plot_r1(rows: pd.DataFrame, summary: pd.DataFrame,
            results_root: Path) -> None:
    if rows.empty:
        return
    apply_plot_style()
    combos = []
    for dataset in DATASET_ORDER:
        for vessel_group in ["major", "communicating_or_variant"]:
            subset = rows[
                rows["dataset"].eq(dataset) & rows["vessel_group"].eq(vessel_group)]
            if not subset.empty:
                combos.append((dataset, vessel_group, subset))
    if not combos:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2),
                             gridspec_kw={"width_ratios": [1.6, 1.0]})
    ax = axes[0]
    distributions = [item[2]["Q_component"].dropna().to_numpy() for item in combos]
    violin = ax.violinplot(distributions, positions=np.arange(len(combos)),
                           widths=0.78, showmeans=False, showmedians=True,
                           showextrema=False)
    for body, (_, group, _) in zip(violin["bodies"], combos):
        body.set_facecolor(BLUE if group == "major" else GOLD)
        body.set_edgecolor(CHARCOAL)
        body.set_alpha(0.68)
    violin["cmedians"].set_color(CHARCOAL)
    violin["cmedians"].set_linewidth(1.2)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    for position, (_, group, subset) in enumerate(combos):
        values = subset["Q_component"].dropna().to_numpy()
        if len(values) <= 250:
            jitter = rng.uniform(-0.16, 0.16, size=len(values))
            ax.scatter(position + jitter, values, s=8,
                       color=BLUE if group == "major" else GOLD,
                       edgecolor="white", linewidth=0.2, alpha=0.42)
    ax.axhline(0, color=CHARCOAL, linewidth=1.0, linestyle="--")
    ax.set_xticks(np.arange(len(combos)))
    ax.set_xticklabels([
        f"{dataset}\n{'major' if group == 'major' else 'small/variant'}"
        for dataset, group, _ in combos], rotation=0)
    ax.set_ylabel("Component utility (residual − no auxiliary)")
    ax.set_title("A. Auxiliary-update utility distribution")

    ax = axes[1]
    bars = summary[summary["scope"].eq("dataset_vessel_group")].copy()
    order = {(dataset, group): index for index, (dataset, group, _) in enumerate(combos)}
    bars["position"] = [order.get((row.dataset, row.vessel_group), np.nan)
                        for row in bars.itertuples(index=False)]
    bars = bars[np.isfinite(bars["position"])].sort_values("position")
    colors = [BLUE if group == "major" else GOLD for group in bars["vessel_group"]]
    means = bars["harmful_rate_mean"].to_numpy(dtype=float)
    errors = np.vstack([
        means - bars["harmful_rate_ci_low"].to_numpy(dtype=float),
        bars["harmful_rate_ci_high"].to_numpy(dtype=float) - means,
    ])
    ax.bar(np.arange(len(bars)), means, color=colors, edgecolor=CHARCOAL,
           linewidth=0.6, yerr=errors, capsize=2)
    ax.set_xticks(np.arange(len(bars)))
    ax.set_xticklabels([
        f"{row.dataset}\n{'major' if row.vessel_group == 'major' else 'small/variant'}"
        for row in bars.itertuples(index=False)])
    ax.set_ylim(0, min(1.0, max(0.15, float(np.nanmax(means + errors[1])) * 1.15)))
    ax.set_ylabel("Harmful-update rate")
    ax.set_title("B. Fraction with negative component utility")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    fig.suptitle("Figure R1. Local benefit and harm of auxiliary updates", fontsize=12,
                 color=CHARCOAL, y=1.02)
    fig.text(0.01, -0.02,
             "One row per case and vessel class; error bars are 95% case-clustered bootstrap CIs.",
             fontsize=8, color="#59636D")
    fig.tight_layout()
    save_figure(fig, results_root, "Figure_R1_utility_distribution")


def plot_r2(validity: pd.DataFrame, calibration: pd.DataFrame,
            phase_ug: pd.DataFrame, results_root: Path) -> None:
    if validity.empty:
        return
    apply_plot_style()
    primary_dataset = "TopCoW_val" if "TopCoW_val" in validity["dataset"].values else validity["dataset"].iloc[0]
    primary = validity[
        validity["target"].eq("component") &
        validity["condition"].eq("clean") &
        validity["dataset"].eq(primary_dataset)].copy()
    if primary.empty:
        return
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.2))
    signal_order = [signal for signal in [*SIGNALS, *PROBABILITY_SIGNALS]
                    if signal in primary["signal"].values]
    width = 0.24
    x = np.arange(len(signal_order))
    for index, level in enumerate(LEVEL_ORDER):
        level_frame = primary[primary["feature_level"].eq(level)].set_index("signal")
        values = [level_frame.loc[s, "auprc_beneficial"]
                  if s in level_frame.index else np.nan for s in signal_order]
        axes[0, 0].bar(x + (index - 1) * width, values, width,
                       label=level, color=PALETTE[index], edgecolor=CHARCOAL,
                       linewidth=0.5)
        aurc_values = [level_frame.loc[s, "aurc_harmful_acceptance"]
                       if s in level_frame.index else np.nan for s in signal_order]
        axes[0, 1].bar(x + (index - 1) * width, aurc_values, width,
                       label=level, color=PALETTE[index], edgecolor=CHARCOAL,
                       linewidth=0.5)
    for ax, title, ylabel in [
            (axes[0, 0], "A. Beneficial-update ranking", "AUPRC (higher is better)"),
            (axes[0, 1], "B. Harmful risk under coverage", "AURC (lower is better)")]:
        ax.set_xticks(x)
        ax.set_xticklabels(signal_order, rotation=22, ha="right", fontsize=8)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_ylim(bottom=0)
    axes[0, 0].legend(ncol=3, loc="upper center")

    cal = calibration[
        calibration["target"].eq("component") &
        calibration["signal"].eq(FORMULA_TRUST)].copy()
    ax = axes[1, 0]
    ax.plot([0, 1], [0, 1], color=CHARCOAL, linestyle="--", linewidth=1,
            label="ideal")
    for index, dataset in enumerate(DATASET_ORDER):
        data = cal[cal["dataset"].eq(dataset)]
        if data.empty:
            continue
        pooled = data.groupby("bin", as_index=False).apply(
            lambda group: pd.Series({
                "predicted": np.average(group["mean_predicted_benefit"],
                                         weights=group["n_rows"]),
                "observed": np.average(group["observed_beneficial_rate"],
                                        weights=group["n_rows"]),
                "n_rows": group["n_rows"].sum(),
            })).reset_index(drop=True)
        ax.plot(pooled["predicted"], pooled["observed"], marker="o",
                markersize=4, color=PALETTE[index], label=dataset)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Mean formula trust score")
    ax.set_ylabel("Observed beneficial-update rate")
    ax.set_title("C. Formula T=G(1-U) reliability (levels pooled)")
    ax.legend(loc="lower right")

    ax = axes[1, 1]
    phase = phase_ug[phase_ug["dataset"].eq(primary_dataset)].copy()
    if not phase.empty:
        phase = phase.groupby(["U_quartile", "G_quartile"], as_index=False).apply(
            lambda group: pd.Series({
                "harmful_rate": np.average(group["harmful_rate"],
                                            weights=group["sample_count"]),
                "sample_count": group["sample_count"].sum(),
            })).reset_index(drop=True)
        matrix = phase.pivot(index="U_quartile", columns="G_quartile",
                             values="harmful_rate").reindex(index=[4, 3, 2, 1],
                                                             columns=[1, 2, 3, 4])
        image = ax.imshow(matrix.to_numpy(dtype=float), cmap="Blues", vmin=0, vmax=1,
                          aspect="auto")
        for row_index in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                value = matrix.iloc[row_index, column_index]
                if np.isfinite(value):
                    ax.text(column_index, row_index, f"{value:.0%}", ha="center",
                            va="center", color="white" if value > 0.55 else CHARCOAL,
                            fontsize=8)
        ax.set_xticks(range(4), [1, 2, 3, 4])
        ax.set_yticks(range(4), [4, 3, 2, 1])
        ax.set_xlabel("G source-reference quartile")
        ax.set_ylabel("U source-reference quartile")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04,
                     label="Harmful-update rate")
    ax.set_title("D. U × G phase diagram")
    fig.suptitle(f"Figure R2. Trust construct validation on {primary_dataset}",
                 fontsize=12, y=1.01)
    fig.text(0.01, -0.005,
             "AUPRC is interpreted with prevalence in the source table; intervals are case-clustered. FDD uses 1−FDD/2.",
             fontsize=8, color="#59636D")
    fig.tight_layout()
    save_figure(fig, results_root, "Figure_R2_trust_validation")


def plot_r3(summary: pd.DataFrame, results_root: Path) -> None:
    if summary.empty:
        return
    apply_plot_style()
    source = summary[
        summary["dataset"].eq("TopCoW_val") & summary["feature_level"].eq("all")].copy()
    if source.empty:
        source = summary[summary["feature_level"].eq("all")].copy()
    if source.empty:
        return
    corruption_order = [name for name in ["gaussian", "channel_dropout", "translation", "donor_mix"]
                        if name in source["corruption_type"].values]
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.5))
    panels = [
        (axes[0, 0], ["harmful_rate"], "A. Harmful-update rate", "Rate"),
        (axes[0, 1], ["1-U", "G", FORMULA_TRUST, IMPLEMENTED_ACCEPTANCE],
         "B. Formula trust and executed acceptance", "Mean signal"),
        (axes[1, 0], ["accepted_delta_l1"], "C. Executed update magnitude",
         "Mean |A_impl Δ|"),
        (axes[1, 1], ["cldice", "det_f1"], "D. Output robustness", "Change from clean"),
    ]
    line_styles = {"1-U": ":", "G": "--", FORMULA_TRUST: "-.",
                   IMPLEMENTED_ACCEPTANCE: "-",
                   "cldice": "-", "det_f1": "--"}
    markers = {"1-U": "^", "G": "s", FORMULA_TRUST: "D",
               IMPLEMENTED_ACCEPTANCE: "o",
               "cldice": "o", "det_f1": "s"}
    for ax, metrics, title, ylabel in panels:
        for color_index, corruption in enumerate(corruption_order):
            corruption_frame = source[source["corruption_type"].eq(corruption)]
            max_severity = float(corruption_frame["severity"].max())
            for metric in metrics:
                data = corruption_frame[corruption_frame["metric"].eq(metric)].sort_values("severity")
                if data.empty:
                    continue
                x = data["severity"].to_numpy(dtype=float) / max_severity if max_severity else 0
                y = data["mean"].to_numpy(dtype=float)
                if title.startswith("D."):
                    y = y - y[0]
                label = corruption if len(metrics) == 1 else "_nolegend_"
                ax.plot(x, y, color=PALETTE[color_index],
                        linestyle=line_styles.get(metric, "-"),
                        marker=markers.get(metric, "o"), markersize=4,
                        linewidth=1.5, label=label)
        ax.set_title(title)
        ax.set_xlabel("Normalized severity within corruption family")
        ax.set_ylabel(ylabel)
        ax.axhline(0, color=CHARCOAL, linewidth=0.8,
                   linestyle="--" if title.startswith("D.") else "-")
        if len(metrics) == 1:
            ax.legend(fontsize=7)
        else:
            corruption_handles = [
                Line2D([0], [0], color=PALETTE[index], linewidth=2,
                       label=corruption)
                for index, corruption in enumerate(corruption_order)
            ]
            metric_handles = [
                Line2D([0], [0], color=CHARCOAL,
                       linestyle=line_styles.get(metric, "-"),
                       marker=markers.get(metric, "o"), markersize=4,
                       label=metric)
                for metric in metrics
            ]
            if title.startswith("B."):
                ax.legend(handles=corruption_handles + metric_handles,
                          fontsize=6.7, ncol=4, loc="center",
                          framealpha=0.92)
            else:
                ax.legend(handles=corruption_handles + metric_handles,
                          fontsize=6.7, ncol=3, loc="lower left",
                          framealpha=0.92)
    axes[0, 0].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    fig.suptitle("Figure R3. Controlled auxiliary-source intervention chain",
                 fontsize=12, y=1.01)
    fig.text(0.01, -0.005,
             "Severity is normalized only for display; unnormalized values, units, confidence intervals, and case counts are in Figure_R3_source.csv.",
             fontsize=8, color="#59636D")
    fig.tight_layout()
    save_figure(fig, results_root, "Figure_R3_controlled_interventions")


def plot_r4(quadrants: pd.DataFrame, selections: pd.DataFrame,
            results_root: Path) -> None:
    data = quadrants[
        quadrants["signal"].eq(FORMULA_TRUST) &
        quadrants["condition"].eq("clean")].copy()
    if data.empty:
        return
    apply_plot_style()
    categories = ["correct_acceptance", "correct_suppression",
                  "over_suppression", "incorrect_acceptance"]
    colors = [BLUE, "#B7C8D9", GOLD, ORANGE]
    pivot = data.pivot_table(
        index=["dataset", "feature_level"], columns="utility_quadrant",
        values="share", aggfunc="sum", fill_value=0).reindex(columns=categories,
                                                               fill_value=0)
    order = [(dataset, level) for dataset in DATASET_ORDER for level in LEVEL_ORDER
             if (dataset, level) in pivot.index]
    pivot = pivot.reindex(order)
    selected = selections[selections["availability"].eq("selected")].copy()
    lines = []
    for dataset in DATASET_ORDER:
        group = selected[selected["dataset"].eq(dataset)]
        if group.empty:
            continue
        lines.append(f"{dataset}")
        disagreement = group[group["selection_rule"].eq(
            "top5_formula_trust_utility_disagreement")].sort_values("rank")
        for row in disagreement.itertuples(index=False):
            lines.append(f"  disagreement #{row.rank}: {row.case_id} ({row.selection_value:.3f})")
        compact = []
        for rule, label in [
                ("highest_HAR", "HAR"), ("highest_BRR", "BRR"),
                ("largest_DetF1_improvement", "ΔDet+"),
                ("largest_DetF1_degradation", "ΔDet−")]:
            item = group[group["selection_rule"].eq(rule)]
            if not item.empty:
                row = item.iloc[0]
                compact.append(f"{label}: {row.case_id} ({row.selection_value:.3f})")
        lines.extend("  " + " · ".join(compact[index:index + 2])
                     for index in range(0, len(compact), 2))
        lines.append("")
    unavailable = selections[~selections["availability"].eq("selected")]
    if not unavailable.empty:
        lines.append("Common-mode case: unavailable (intervention not run).")

    height = max(5.3, 0.25 * max(len(lines), len(pivot) * 2) + 1.8)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, height),
                             gridspec_kw={"width_ratios": [1.45, 1.0]})
    ax = axes[0]
    left = np.zeros(len(pivot))
    for category, color in zip(categories, colors):
        values = pivot[category].to_numpy(dtype=float)
        ax.barh(np.arange(len(pivot)), values, left=left, color=color,
                edgecolor="white", linewidth=0.5, label=category.replace("_", " "))
        left += values
    ax.set_yticks(np.arange(len(pivot)),
                  [f"{dataset} · {level}" for dataset, level in pivot.index])
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel("")
    ax.set_title("A. Source-frozen formula-T threshold quadrants")
    ax.invert_yaxis()
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2,
              fontsize=8)

    ax = axes[1]
    ax.axis("off")
    ax.set_title("B. Mechanically selected audit cases", loc="left")
    ax.text(0.0, 0.97, "\n".join(lines), va="top", ha="left",
            fontsize=7.5, family="DejaVu Sans Mono", linespacing=1.3)
    fig.suptitle("Figure R4. Correct decisions, failure quadrants, and audit cases",
                 fontsize=12, y=1.01)
    fig.text(0.01, 0.012,
             "Thresholds were selected only on clean TopCoW validation and then frozen. Case IDs follow preregistered ranking rules.",
             fontsize=8, color="#59636D")
    fig.tight_layout(rect=(0, 0.08, 1, 0.97))
    save_figure(fig, results_root, "Figure_R4_failure_quadrants")


def baseline_reproduction_table(results_root: Path) -> pd.DataFrame:
    source = (results_root / "00_project_discovery" / "baseline_reproduction" /
              "comparison.json")
    payload = read_json(source, {}) or {}
    rows = []
    dataset_map = {
        "tcff_crown": "TopCoW_val", "tcff_lausanne": "Lausanne",
        "tcff_ixi": "IXI_HH", "concat_crown": "TopCoW_val",
    }
    variant_map = {
        "tcff_crown": "full_tcff_seed42", "tcff_lausanne": "full_tcff_seed42",
        "tcff_ixi": "full_tcff_seed42", "concat_crown": "concat_baseline",
    }
    for run_name, metrics in (payload.get("comparisons") or {}).items():
        for metric, values in metrics.items():
            rows.append({
                "run_name": run_name, "dataset": dataset_map.get(run_name, run_name),
                "variant": variant_map.get(run_name, run_name), "metric": metric,
                "historical": values.get("historical"),
                "reproduced": values.get("reproduced"),
                "absolute_difference": values.get("abs_diff"),
            })
    table = pd.DataFrame(rows)
    atomic_csv(results_root / "06_summary_tables" / "G0_baseline_reproduction.csv",
               table)
    return table


def append_claim(claims: List[dict], source_frame: pd.DataFrame,
                 source_path: Path, results_root: Path, source_index: int,
                 source_column: str, description: str, unit: str,
                 dataset: str, variant: str,
                 comparison: str = "", ci_low_column: Optional[str] = None,
                 ci_high_column: Optional[str] = None) -> None:
    row = source_frame.loc[source_index]
    value = finite_number(row.get(source_column))
    if value is None:
        return
    claims.append({
        "claim_id": "",
        "description": description,
        "value": value,
        "unit": unit,
        "ci_low": finite_number(row.get(ci_low_column)) if ci_low_column else None,
        "ci_high": finite_number(row.get(ci_high_column)) if ci_high_column else None,
        "dataset": dataset,
        "variant": variant,
        "comparison": comparison,
        "source_file": relative_to(source_path, results_root),
        "source_row": int(source_index) + 2,
        "source_column": source_column,
        "verified_by": "tcff_postprocess.py independent recomputation",
    })


def make_exact_numbers(results_root: Path, baseline: pd.DataFrame,
                       p0a: pd.DataFrame, validity: pd.DataFrame,
                       action: pd.DataFrame, variants: pd.DataFrame,
                       mechanism: pd.DataFrame,
                       product_fixed: pd.DataFrame) -> pd.DataFrame:
    claims: List[dict] = []
    baseline_path = results_root / "06_summary_tables" / "G0_baseline_reproduction.csv"
    for index, row in baseline.iterrows():
        if row.get("variant") != "full_tcff_seed42" or row.get("metric") not in {
                "binary_dice", "binary_cldice", "class_avg_dice", "grp2_detection_f1"}:
            continue
        metric_label = str(row.metric).replace("_", " ")
        append_claim(
            claims, baseline, baseline_path, results_root, index, "reproduced",
            f"Reproduced submitted-checkpoint {metric_label}", "fraction",
            str(row.dataset), str(row.variant), "historical result")

    p0a_path = results_root / "06_summary_tables" / "p0_a_utility_summary.csv"
    for index, row in p0a[p0a["scope"].eq("dataset_vessel_group")].iterrows():
        group_label = "major vessels" if row.get("vessel_group") == "major" else "communicating/variant vessels"
        append_claim(
            claims, p0a, p0a_path, results_root, index, "harmful_rate_mean",
            f"Harmful component-update rate for {group_label}", "fraction",
            str(row.get("dataset")), "full_tcff_fixed_checkpoint", "residual vs no auxiliary",
            "harmful_rate_ci_low", "harmful_rate_ci_high")
        append_claim(
            claims, p0a, p0a_path, results_root, index, "utility_mean",
            f"Mean component utility for {group_label}", "Dice difference",
            str(row.get("dataset")), "full_tcff_fixed_checkpoint", "residual vs no auxiliary",
            "utility_ci_low", "utility_ci_high")

    validity_path = results_root / "06_summary_tables" / "p0_b_signal_validity.csv"
    valid_primary = validity[
        validity["target"].eq("component") & validity["condition"].eq("clean")]
    for index, row in valid_primary.iterrows():
        append_claim(
            claims, validity, validity_path, results_root, index,
            "auprc_beneficial",
            f"{row.get('signal')} AUPRC for beneficial component updates at {row.get('feature_level')} level",
            "AUPRC", str(row.get("dataset")), "full_tcff_fixed_checkpoint",
            "beneficial prevalence=" + str(row.get("prevalence_beneficial")),
            "auprc_beneficial_ci_low", "auprc_beneficial_ci_high")
        append_claim(
            claims, validity, validity_path, results_root, index,
            "aurc_harmful_acceptance",
            f"{row.get('signal')} harmful-update AURC at {row.get('feature_level')} level",
            "AURC", str(row.get("dataset")), "full_tcff_fixed_checkpoint",
            "accept from highest beneficial score",
            "aurc_harmful_acceptance_ci_low", "aurc_harmful_acceptance_ci_high")

    action_path = results_root / "06_summary_tables" / "p0_f_har_brr.csv"
    for index, row in action[
            action["signal"].eq(FORMULA_TRUST) &
            action["condition"].eq("clean")].iterrows():
        for column, label in [("har", "Harmful acceptance rate"),
                              ("brr", "Beneficial rejection rate")]:
            append_claim(
                claims, action, action_path, results_root, index, column,
                f"{label} using source-frozen formula-T threshold at {row.get('feature_level')} level",
                "fraction", str(row.get("dataset")), "full_tcff_fixed_checkpoint",
                "threshold=" + str(row.get("threshold")),
                column + "_ci_low", column + "_ci_high")

    variant_path = results_root / "06_summary_tables" / "Table_R1_components.csv"
    for index, row in variants.iterrows():
        append_claim(
            claims, variants, variant_path, results_root, index, "mean",
            f"{row.get('variant')} {row.get('metric')} ({row.get('analysis_type')})",
            "metric value", str(row.get("dataset")), str(row.get("variant")),
            "paired against independently matching full row where available")
        if row.get("variant") != "full":
            append_claim(
                claims, variants, variant_path, results_root, index,
                "paired_difference",
                f"Paired {row.get('metric')} difference for {row.get('variant')} versus full",
                "metric difference", str(row.get("dataset")), str(row.get("variant")),
                "full", "paired_ci_low", "paired_ci_high")

    mechanism_path = results_root / "06_summary_tables" / "Figure_R3_source.csv"
    primary = mechanism[
        mechanism["feature_level"].eq("all") &
        mechanism["metric"].isin(
            ["harmful_rate", FORMULA_TRUST, IMPLEMENTED_ACCEPTANCE,
             "cldice", "det_f1"])]
    for (dataset, corruption, metric), group in primary.groupby(
            ["dataset", "corruption_type", "metric"]):
        max_rows = group[group["severity"].eq(group["severity"].max())]
        for index, row in max_rows.iterrows():
            append_claim(
                claims, mechanism, mechanism_path, results_root, index, "mean",
                f"{metric} at maximum preregistered {corruption} severity",
                "metric value", str(dataset), "full_tcff_fixed_checkpoint",
                f"severity={row.get('severity')}", "ci_low", "ci_high")

    product_path = (results_root / "06_summary_tables" /
                    "Table_R1_product_fixed_corruption.csv")
    if not product_fixed.empty:
        maximum = product_fixed[
            product_fixed["severity"].eq(product_fixed.groupby(
                ["dataset", "corruption_type", "metric"]
            )["severity"].transform("max"))]
        for index, row in maximum.iterrows():
            scope = (f"submitted A_impl at severity={row.get('severity')} "
                     f"for {row.get('corruption_type')}")
            append_claim(
                claims, product_fixed, product_path, results_root, index,
                "product_mean",
                f"Product-intervention {row.get('metric')} at maximum tested severity",
                "metric value", str(row.get("dataset")), "product", scope)
            append_claim(
                claims, product_fixed, product_path, results_root, index,
                "paired_difference",
                f"Product minus submitted-A_impl {row.get('metric')} at maximum tested severity",
                "metric difference", str(row.get("dataset")), "product",
                scope, "paired_ci_low", "paired_ci_high")

    output = pd.DataFrame(claims)
    if not output.empty:
        output.insert(1, "description_hash", [
            hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
            for text in output["description"].astype(str)])
        output["claim_id"] = [f"TCFF-{index + 1:04d}" for index in range(len(output))]
    required = ["claim_id", "description", "value", "unit", "ci_low", "ci_high",
                "dataset", "variant", "comparison", "source_file", "source_row",
                "source_column", "verified_by"]
    for column in required:
        if column not in output:
            output[column] = pd.Series(dtype=object)
    output = output[[column for column in ["claim_id", "description_hash", *required[1:]]
                     if column in output]]
    atomic_csv(results_root / "06_summary_tables" / "exact_numbers_safe_to_quote.csv",
               output)
    return output


def write_chart_map(results_root: Path) -> None:
    rows = [
        {
            "figure": "Figure R1", "section": "Problem existence",
            "question": "Are auxiliary updates locally harmful despite aggregate benefit?",
            "family": "distribution and comparison", "chart_type": "violin plus interval bars",
            "fields": "dataset,vessel_group,Q_component,harmful_rate",
            "supported_claim": "Utility has a measurable negative tail and harmful-rate heterogeneity.",
            "palette_policy": "hard two-root cap", "artifact": "07_figures/Figure_R1_utility_distribution.png",
            "source": "06_summary_tables/Figure_R1_source.csv",
        },
        {
            "figure": "Figure R2", "section": "Trust construct validation",
            "question": "Do 1-U, G, formula T, A_impl, and FDD rank observed update utility?",
            "family": "comparison, calibration, matrix", "chart_type": "grouped bars, reliability line, heatmap",
            "fields": "signal,feature_level,AUPRC,AURC,predicted,observed,U_quartile,G_quartile",
            "supported_claim": "Formula trust and executed acceptance are explicitly separated and differ in utility ranking and calibration across levels.",
            "palette_policy": "relaxed multi-category for levels only", "artifact": "07_figures/Figure_R2_trust_validation.png",
            "source": "06_summary_tables/p0_b_signal_validity.csv",
        },
        {
            "figure": "Figure R3", "section": "Controlled interventions",
            "question": "Does auxiliary degradation propagate through trust and output quality?",
            "family": "ordered response", "chart_type": "faceted multi-series line",
            "fields": "corruption_type,severity,utility,harmful_rate,1-U,G,T_formula,A_impl,accepted_delta,clDice,Det-F1",
            "supported_claim": "The full mechanism chain is visible without pooling corruption units.",
            "palette_policy": "four declared roots plus line styles", "artifact": "07_figures/Figure_R3_controlled_interventions.png",
            "source": "06_summary_tables/Figure_R3_source.csv",
        },
        {
            "figure": "Figure R4", "section": "Failure boundaries",
            "question": "Where does the frozen trust action accept harm or suppress benefit?",
            "family": "composition", "chart_type": "100% stacked horizontal bars plus selected-case list",
            "fields": "dataset,feature_level,quadrant,share,case_id,selection_rule",
            "supported_claim": "Correct decisions and both failure modes remain explicitly visible.",
            "palette_policy": "four semantic roots with direct legend", "artifact": "07_figures/Figure_R4_failure_quadrants.png",
            "source": "06_summary_tables/p0_g_quadrants.csv",
        },
    ]
    atomic_csv(results_root / "07_figures" / "CHART_MAP.csv", pd.DataFrame(rows))


def matrix_completion(results_root: Path, inventory: pd.DataFrame) -> dict:
    plan = read_json(results_root / "02_manifests" / "p0_inference_matrix_plan.json", {}) or {}
    planned = int(plan.get("n_jobs", 0) or 0)
    experiment_ids = {
        "P0_AB_CLEAN_MECHANISM", "P0_D_FIXED_CKPT_INTERVENTIONS",
        "P0_C1_GAUSSIAN", "P0_C1_CHANNEL_DROPOUT",
        "P0_C2_TRANSLATION", "P0_C3_DONOR_MIX",
    }
    formal = inventory[inventory["experiment_id"].isin(experiment_ids)]
    success = int(formal["status"].eq("success").sum())
    failed = int(formal["status"].isin(["failed", "partial", "unknown"]).sum())
    running = int(formal["status"].eq("running").sum())
    return {
        "planned": planned, "successful": success, "failed_or_partial": failed,
        "running": running,
        "complete": bool(planned > 0 and success == planned and failed == 0),
    }


def product_matrix_completion(results_root: Path,
                              inventory: pd.DataFrame) -> dict:
    plan = read_json(
        results_root / "02_manifests" /
        "p0_product_intervention_matrix_plan.json", {}) or {}
    planned = int(plan.get("n_jobs", 0) or 0)
    formal = inventory[inventory["experiment_id"].isin([
        "P0_D_PRODUCT_FIXED_CLEAN", "P0_D_PRODUCT_FIXED_CORRUPTION"
    ])]
    success = int(formal["status"].eq("success").sum())
    failed = int(formal["status"].isin(
        ["failed", "partial", "unknown"]).sum())
    running = int(formal["status"].eq("running").sum())
    return {
        "planned": planned, "successful": success,
        "failed_or_partial": failed, "running": running,
        "complete": bool(planned > 0 and success == planned and failed == 0),
        "scope": ("supplemental same-checkpoint product intervention; "
                  "A_product=G(1-U) is actually executed"),
    }
def workflow_completion(results_root: Path, inventory: pd.DataFrame) -> dict:
    """Summarize the fixed and trained evidence streams without hiding gaps."""
    fixed = matrix_completion(results_root, inventory)
    product_fixed = product_matrix_completion(results_root, inventory)

    training_jobs = set()
    for path in sorted((results_root / "02_manifests").glob(
            "p0_training_matrix_*.json")):
        plan = read_json(path, {}) or {}
        for job in plan.get("jobs", []):
            if "variant" in job and "seed" in job:
                training_jobs.add((str(job["variant"]), int(job["seed"])))
    training = inventory[inventory["experiment_id"].eq(
        "P0_D_TRAIN_VARIANTS")].copy()
    training_success = set()
    training_non_success = set()
    for row in training.itertuples(index=False):
        key = (str(row.model_variant), int(row.seed))
        if row.status == "success":
            training_success.add(key)
        elif row.status in {"failed", "partial", "unknown"}:
            training_non_success.add(key)
    training_summary = {
        "planned": len(training_jobs),
        "successful": len(training_jobs & training_success),
        "failed_or_partial": len(training_jobs & training_non_success),
        "running_or_not_started": len(training_jobs - training_success -
                                      training_non_success),
    }
    training_summary["complete"] = bool(
        training_jobs and training_jobs <= training_success and
        not (training_jobs & training_non_success))

    evaluated_model_jobs = set()
    eval_realizations = {}
    for path in sorted((results_root / "02_manifests").glob(
            "p0_trained_eval_matrix_*.json")):
        state = read_json(path, {}) or {}
        realizations = int(state.get("corruption_realizations", 3) or 3)
        for assignment in state.get("assignments", []):
            if "variant" not in assignment or "seed" not in assignment:
                continue
            key = (str(assignment["variant"]), int(assignment["seed"]))
            evaluated_model_jobs.add(key)
            eval_realizations[key] = realizations
    expected_eval_runs = sum(3 + 4 * eval_realizations[key]
                             for key in evaluated_model_jobs)
    trained_eval = inventory[inventory["experiment_id"].isin(
        ["P0_D_TRAINED_EVAL_CLEAN", "P0_D_TRAINED_EVAL_CORRUPTION"])]
    eval_success = int(trained_eval["status"].eq("success").sum())
    eval_failed = int(trained_eval["status"].isin(
        ["failed", "partial", "unknown"]).sum())
    eval_running = int(trained_eval["status"].eq("running").sum())
    trained_eval_summary = {
        "planned_model_seed_jobs": len(evaluated_model_jobs),
        "planned_runs": expected_eval_runs,
        "successful_runs": eval_success,
        "failed_or_partial_runs": eval_failed,
        "running_runs": eval_running,
        "complete": bool(expected_eval_runs > 0 and
                         eval_success == expected_eval_runs and
                         eval_failed == 0 and eval_running == 0),
    }

    efficiency_path = results_root / "09_efficiency" / "efficiency_summary.csv"
    efficiency_ready = False
    if efficiency_path.exists():
        try:
            efficiency_ready = not pd.read_csv(efficiency_path).empty
        except Exception:
            efficiency_ready = False

    probability_state = read_json(
        results_root / "02_manifests" / "p0_probability_signal_matrix.json", {}) or {}
    probability_planned = int(probability_state.get("planned", 0) or 0)
    probability_done = (len(probability_state.get("completed", [])) +
                        len(probability_state.get("skipped_success", [])))
    probability_failed = len(probability_state.get("failed", []))

    selected_state = read_json(
        results_root / "02_manifests" / "p0_selected_case_maps.json", {}) or {}
    selected_planned = int(selected_state.get("planned", 0) or 0)
    selected_done = (len(selected_state.get("completed", [])) +
                     len(selected_state.get("skipped_success", [])))
    selected_failed = len(selected_state.get("failed", []))

    core_complete = bool(
        fixed["complete"] and product_fixed["complete"] and
        training_summary["complete"] and
        trained_eval_summary["complete"] and efficiency_ready)
    return {
        "fixed_checkpoint": fixed,
        "product_fixed_intervention": product_fixed,
        "training": training_summary,
        "trained_evaluation": trained_eval_summary,
        "efficiency_ready": efficiency_ready,
        "probability_signals": {
            "planned": probability_planned, "completed": probability_done,
            "failed": probability_failed,
            "complete": bool(probability_planned > 0 and
                             probability_done == probability_planned and
                             probability_failed == 0),
        },
        "selected_case_maps": {
            "planned": selected_planned, "completed": selected_done,
            "failed": selected_failed,
            "complete": bool(selected_planned > 0 and
                             selected_done == selected_planned and
                             selected_failed == 0),
        },
        "core_complete": core_complete,
    }


def markdown_table(frame: pd.DataFrame, columns: Sequence[str],
                   max_rows: int = 30) -> str:
    available = [column for column in columns if column in frame]
    if not available or frame.empty:
        return "_No rows available._"
    view = frame[available].head(max_rows).copy()
    for column in view.select_dtypes(include=[np.number]).columns:
        view[column] = view[column].map(
            lambda value: "" if pd.isna(value) else f"{float(value):.6g}")
    header = "| " + " | ".join(available) + " |"
    divider = "| " + " | ".join(["---"] * len(available)) + " |"
    lines = [header, divider]
    for row in view.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value).replace("|", "\\|")
                                         for value in row) + " |")
    return "\n".join(lines)


def write_quality_reports(results_root: Path, inventory: pd.DataFrame,
                          case: pd.DataFrame, region: pd.DataFrame,
                          findings: pd.DataFrame, n_boot: int) -> None:
    as_of = utc_now()
    completion = matrix_completion(results_root, inventory)
    workflow = workflow_completion(results_root, inventory)
    failures = findings[findings["status"].eq("fail")] if not findings.empty else findings
    critical_high = failures[failures["severity"].isin(["critical", "high"])] if not failures.empty else failures
    quality_path = results_root / "11_quality_audit" / "DATA_QUALITY_REPORT.md"
    quality_text = f"""# TCFF result data-quality report

Generated: {as_of}

## Dataset and grain summary

- Run inventory: {len(inventory)} manifests.
- Loaded case rows: {len(case)} at one row per run and case.
- Loaded internal rows: {len(region)} at one row per run, case, ROI, and feature level.
- Fixed-checkpoint matrix: {completion['successful']}/{completion['planned']} successful; {completion['running']} running; {completion['failed_or_partial']} failed or partial.
- Supplemental executed-product matrix: {workflow['product_fixed_intervention']['successful']}/{workflow['product_fixed_intervention']['planned']} successful; {workflow['product_fixed_intervention']['running']} running; {workflow['product_fixed_intervention']['failed_or_partial']} failed or partial.
- Training matrix: {workflow['training']['successful']}/{workflow['training']['planned']} successful; {workflow['training']['running_or_not_started']} running or not started; {workflow['training']['failed_or_partial']} failed or partial.
- Trained evaluation matrix: {workflow['trained_evaluation']['successful_runs']}/{workflow['trained_evaluation']['planned_runs']} successful; {workflow['trained_evaluation']['running_runs']} running; {workflow['trained_evaluation']['failed_or_partial_runs']} failed or partial.
- Bootstrap replicates in this refresh: {n_boot} (final requirement: {DEFAULT_BOOTSTRAPS}).
- Quantitative pooling includes successful runs only. Failed and partial evidence remains in the inventory.

## Checks performed

Manifest identity, required files, status agreement, case and region keys, schema, finite values, metric ranges, checkpoint identity, run/seed coverage, frozen external-tuning guard, machine-readable split-overlap evidence, and the `T_formula<=G` signal-semantic invariant.

## Findings

{markdown_table(findings, ['check_id', 'severity', 'status', 'evidence', 'impact'], 50)}

## Analytical risk and remediation

{len(critical_high)} critical/high failing checks remain. A result is safe to quote only when its source run is successful, its row appears in `exact_numbers_safe_to_quote.csv`, and the independent verifier passes. Missing centerline or downsampled-mask signal values remain NA by design.

## Assumptions and open checks

- Dataset roles follow the executable project split: TopCoW training data for fitting, Crown/TopCoW validation for checkpoint and threshold selection, and Lausanne/IXI-HH for evaluation only.
- HD95 is reported in evaluator voxel units.
- The exact-duplicate scan is considered machine-verifiable only when `DATA_INTEGRITY_AUDIT.json` is present.
"""
    quality_path.parent.mkdir(parents=True, exist_ok=True)
    temp = quality_path.with_suffix(".md.tmp")
    temp.write_text(quality_text, encoding="utf-8")
    temp.replace(quality_path)

    overall = "Ready to share"
    if (len(critical_high) or not workflow["core_complete"] or
            n_boot < DEFAULT_BOOTSTRAPS):
        overall = "Needs revision"
    elif len(failures):
        overall = "Share with caveats"
    validation_path = results_root / "11_quality_audit" / "ANALYSIS_VALIDATION_REPORT.md"
    validation_text = f"""# Validation report

## Overall assessment: {overall}

### Question and evidence boundary

The analysis asks whether fixed-checkpoint auxiliary updates show local benefit and harm, whether `1-U`, `G`, manuscript-formula `T_formula=G(1-U)`, executed acceptance `A_impl`, and FDD rank observed paired inference-time utility, how controlled auxiliary corruption propagates through the mechanism chain, and whether trained components outperform simple alternatives. It establishes within-model response under declared interventions, not real-world causal effects.

### Methodology review

- Source-only thresholding is frozen in `01_configs/POSTPROCESSING_PROTOCOL.md`.
- All confidence intervals use case-clustered bootstrap resampling ({DEFAULT_BOOTSTRAPS} planned replicates by default).
- Utility rows repeated across feature levels are collapsed for problem-existence summaries and retained only when evaluating level-specific trust.
- External cohorts are never pooled for tuning or checkpoint selection and are reported separately.
- AUPRC is accompanied by beneficial/harmful prevalence; AURC accepts tied scores together.
- Historical raw `T` is preserved for provenance as `A_impl`; all manuscript-formula analyses use a newly derived same-row `T_formula=G(1-U)`.

### Issues found

{markdown_table(failures, ['severity', 'check_id', 'evidence', 'remediation'], 50)}

### Calculation spot-checks

- Case/run uniqueness: {'verified' if findings[findings['check_id'].eq('unique_case_rows')]['status'].eq('pass').all() else 'not verified'}.
- Fixed-checkpoint identity: {'verified' if findings[findings['check_id'].eq('fixed_checkpoint_identity')]['status'].eq('pass').all() else 'not verified'}.
- Quote-table values: must be rechecked with `tcff_verify_quote_table.py` after each refresh.

### Visualization review

Figures are regenerated from named source CSVs, use honest axes and explicit units, and preserve negative or mixed outcomes. Final visual inspection is required after the full matrix replaces the incremental snapshot.

### Required caveats

- Confidence and entropy are unavailable unless probability summaries are collected in dedicated runs.
- The current implementation exports level-global `U`; spatial discrimination in formula trust is supplied by `G`, while executed interpolation is reported separately as `A_impl`.
- Common-mode failure and full per-case trust maps require dedicated follow-up runs if they remain absent.
"""
    temp = validation_path.with_suffix(".md.tmp")
    temp.write_text(validation_text, encoding="utf-8")
    temp.replace(validation_path)


def verdict_table(p0a: pd.DataFrame, validity: pd.DataFrame,
                  variants: pd.DataFrame, mechanism: pd.DataFrame,
                  quadrants: pd.DataFrame, completion: dict) -> pd.DataFrame:
    rows = []
    if p0a.empty:
        h1 = "Not evaluated"
        h1_caveat = "Clean mechanism rows unavailable."
    else:
        overall = p0a[p0a["scope"].eq("overall")]
        harmful = float(overall["harmful_rate_mean"].iloc[0]) if not overall.empty else np.nan
        mean_q = float(overall["utility_mean"].iloc[0]) if not overall.empty else np.nan
        h1 = ("Strongly supported" if completion["complete"] and
              mean_q > 0 and harmful > 0 else "Partially supported")
        h1_caveat = "Observed paired inference-time utility; not a causal ground truth."
    rows.append({"hypothesis": "H1 Local auxiliary-update risk exists",
                 "verdict": h1, "evidence_files": "Figure_R1_source.csv; p0_a_utility_summary.csv",
                 "caveat": h1_caveat})

    source_validity = validity[
        validity["dataset"].eq("TopCoW_val") & validity["condition"].eq("clean") &
        validity["target"].eq("component")]
    if source_validity.empty:
        h2 = "Not evaluated"
    else:
        informative = (source_validity["auprc_beneficial"] >
                       source_validity["prevalence_beneficial"]).mean()
        h2 = "Strongly supported" if informative >= 0.75 else (
            "Partially supported" if informative > 0 else "Not supported")
    rows.append({"hypothesis": "H2 Saved trust signals rank observed utility",
                 "verdict": h2, "evidence_files": "p0_b_signal_validity.csv; Figure_R2_trust_validation.png",
                 "caveat": "Ranking and raw calibration are distinct; level-global U limits spatial claims."})

    trained = variants[variants["analysis_type"].eq("independently_trained_variant")] if not variants.empty else variants
    rows.append({"hypothesis": "H3 G adds information beyond U",
                 "verdict": "Not evaluated" if trained.empty else "See measured comparisons",
                 "evidence_files": "Table_R1_components.csv",
                 "caveat": "A full verdict requires completed three-seed trained variants."})
    rows.append({"hypothesis": "H4 Executed product rule outperforms simple alternatives",
                  "verdict": "Not evaluated" if trained.empty else "See measured comparisons",
                  "evidence_files": ("Table_R1_components.csv; "
                                     "Table_R1_product_fixed_corruption.csv"),
                  "caveat": ("Submitted A_impl, fixed product intervention, and "
                             "independently trained variants are reported separately.")})

    h5 = "Not evaluated" if mechanism.empty else (
        "Partially supported" if not completion["complete"] else "See measured intervention slopes")
    rows.append({"hypothesis": "H5 Controlled degradation forms a mechanism-consistent chain",
                 "verdict": h5, "evidence_files": "Figure_R3_source.csv; p0_c_degradation_slopes.csv",
                 "caveat": "Corruption severities have different physical/statistical units and are not pooled numerically."})
    rows.append({"hypothesis": "H6 Gains are not explained by capacity or global attenuation",
                 "verdict": "Not evaluated", "evidence_files": "Table_R1_components.csv; Table_R2_efficiency.csv",
                 "caveat": "Parameter-matched and magnitude-matched controls were not in the minimum 12-hour matrix."})

    failure_share = quadrants[
        quadrants["utility_quadrant"].isin(["over_suppression", "incorrect_acceptance"])]
    h7 = "Not evaluated" if quadrants.empty else (
        "Strongly supported" if completion["complete"] and
        failure_share["n_rows"].sum() > 0 else
        "Partially supported" if failure_share["n_rows"].sum() > 0 else
        "Not supported")
    rows.append({"hypothesis": "H7 Failure boundaries remain measurable",
                 "verdict": h7, "evidence_files": "p0_g_quadrants.csv; predeclared_case_selection.csv",
                 "caveat": "Common-mode failures require a dedicated intervention and are not inferred from other corruptions."})
    return pd.DataFrame(rows)


def write_final_summary(results_root: Path, inventory: pd.DataFrame,
                        findings: pd.DataFrame, p0a: pd.DataFrame,
                        validity: pd.DataFrame, variants: pd.DataFrame,
                        mechanism: pd.DataFrame, quadrants: pd.DataFrame,
                        exact: pd.DataFrame, n_boot: int) -> None:
    completion = matrix_completion(results_root, inventory)
    workflow = workflow_completion(results_root, inventory)
    verdicts = verdict_table(p0a, validity, variants, mechanism, quadrants, completion)
    atomic_csv(results_root / "06_summary_tables" / "hypothesis_verdicts.csv", verdicts)
    commits = sorted(inventory["git_commit"].dropna().astype(str).unique())
    checkpoints = sorted(inventory["checkpoint_hash"].dropna().astype(str).unique())
    split_hashes = sorted(inventory["data_split_hash"].dropna().astype(str).unique())
    seeds = sorted(pd.to_numeric(inventory["seed"], errors="coerce").dropna().astype(int).unique().tolist())
    failures = inventory[inventory["status"].isin(["failed", "partial", "unknown"])]
    ready_state = ("complete" if workflow["core_complete"] and
                   n_boot >= DEFAULT_BOOTSTRAPS else
                   "incremental — formal jobs or final statistics still outstanding")
    negative_notes = []
    if not validity.empty:
        primary_validity = validity[
            validity["target"].eq("component") &
            validity["condition"].isin(["clean", "all_corruption"])]
        below = primary_validity[
            primary_validity["auprc_beneficial"] <=
            primary_validity["prevalence_beneficial"]]
        negative_notes.append(
            f"{len(below)}/{len(primary_validity)} primary component clean/pooled-corruption signal rows did not exceed the prevalence AUPRC baseline.")
    if not quadrants.empty:
        primary_quadrants = quadrants[
            quadrants["signal"].eq(FORMULA_TRUST) &
            quadrants["condition"].isin(["clean", "all_corruption"])]
        incorrect = primary_quadrants[
            primary_quadrants["utility_quadrant"].eq(
                "incorrect_acceptance")]["n_rows"].sum()
        over = primary_quadrants[
            primary_quadrants["utility_quadrant"].eq(
                "over_suppression")]["n_rows"].sum()
        negative_notes.append(
            f"Level-specific formula-T decisions on clean plus pooled-corruption strata include {int(incorrect)} incorrect-acceptance rows and {int(over)} over-suppression rows.")
    if not negative_notes:
        negative_notes.append("Formal mixed-result summaries are not yet available.")
    unresolved = []
    if not completion["complete"]:
        unresolved.append(
            f"Formal fixed-checkpoint completeness: {completion['successful']}/{completion['planned']}.")
    if not workflow["product_fixed_intervention"]["complete"]:
        unresolved.append(
            "Executed-product fixed-checkpoint intervention remains incomplete: "
            f"{workflow['product_fixed_intervention']['successful']}/"
            f"{workflow['product_fixed_intervention']['planned']} successful.")
    if not workflow["training"]["complete"]:
        unresolved.append(
            "Planned training jobs remain incomplete: "
            f"{workflow['training']['successful']}/{workflow['training']['planned']} successful.")
    if not workflow["trained_evaluation"]["complete"]:
        unresolved.append(
            "Trained-variant evaluation remains incomplete: "
            f"{workflow['trained_evaluation']['successful_runs']}/"
            f"{workflow['trained_evaluation']['planned_runs']} successful runs.")
    if not workflow["probability_signals"]["complete"]:
        unresolved.append("Confidence/entropy probability summaries remain incomplete.")
    if not workflow["selected_case_maps"]["complete"]:
        unresolved.append("Full selected-case trust-map panels remain incomplete.")
    if not workflow["efficiency_ready"]:
        unresolved.append("Same-device efficiency benchmark remains incomplete.")
    if n_boot < DEFAULT_BOOTSTRAPS:
        unresolved.append(
            f"This refresh uses {n_boot} bootstrap replicates; final requires {DEFAULT_BOOTSTRAPS}.")
    unresolved.extend([
        "A parameter-matched ordinary gate and dedicated common-mode corruption are not part of the completed minimum package unless separately listed as successful runs.",
        "Data-integrity overlap evidence must remain machine-readable in DATA_INTEGRITY_AUDIT.json.",
    ])

    text = f"""# Final Result Summary

Status: **{ready_state}**  
Generated: {utc_now()}

## Reproducibility identity

- execution git commit(s): {', '.join(commits) or 'unavailable'}
- checkpoint hashes observed: {len(checkpoints)} ({'; '.join(checkpoints)})
- data split hashes observed: {len(split_hashes)} ({'; '.join(split_hashes)})
- seeds observed: {seeds}
- failed/partial/unknown runs: {len(failures)}
- fixed-checkpoint matrix: {completion['successful']}/{completion['planned']} successful
- executed-product fixed-checkpoint matrix: {workflow['product_fixed_intervention']['successful']}/{workflow['product_fixed_intervention']['planned']} successful
- training matrix: {workflow['training']['successful']}/{workflow['training']['planned']} successful
- trained-evaluation matrix: {workflow['trained_evaluation']['successful_runs']}/{workflow['trained_evaluation']['planned_runs']} successful
- case-clustered bootstrap replicates: {n_boot}

## H1–H7 verdict

{markdown_table(verdicts, ['hypothesis', 'verdict', 'evidence_files', 'caveat'], 20)}

## Primary findings

### Finding 1: problem existence

See `06_summary_tables/p0_a_utility_summary.csv` and Figure R1. The table preserves mean utility, its full negative tail, harmful-update rate, vessel-group strata, case counts, and case-clustered intervals.

### Finding 2: U validity and limits

See `06_summary_tables/p0_b_signal_validity.csv`. `1-U`, `G`, manuscript-formula `T_formula=G(1-U)`, executed acceptance `A_impl`, and the preregistered FDD orientation are reported with AUPRC, AUROC, Brier, NLL, AURC, prevalence, and bootstrap intervals. Historical raw `T` is never treated as formula trust. Missing confidence/entropy is explicit rather than imputed.

### Finding 3: G incremental role

See Table R1. Fixed-checkpoint interventions are separated from independently trained variants so that output intervention effects are not conflated with training adaptation.

### Finding 4: executed full interpolation vs alternatives

See Table R1, `Table_R1_product_fixed_corruption.csv`, and `p0_f_har_brr.csv`. The submitted `full` row always denotes executed `A_impl=1-(1-G)U`; `product` denotes a separate run that actually executes `G(1-U)`. Both performance and action-quality trade-offs are required before preferring a variant.

### Finding 5: controlled intervention chain

See Figure R3 and `p0_c_degradation_slopes.csv`. Each corruption family retains its original severity unit; the plot normalization is display-only.

### Finding 6: capacity/regularization controls

The original minimum matrix covers no auxiliary, residual, U-only, G-only, additive, and submitted-full controls. The supplemental matrix adds product-gated fixed-checkpoint runs plus three independently trained product seeds. Parameter-matched and global-attenuation controls remain outside the completed package unless separately listed.

### Finding 7: failure boundaries

See Figure R4 and `08_failure_cases/predeclared_case_selection.csv`. Incorrect acceptance and over-suppression remain visible; a common-mode representative is not claimed without a common-mode run.

## Reviewer mapping

### pCat

Use source-to-external signal rows, controlled corruptions, confidence/entropy controls, and Table R2 to address proxy stability, shift behavior, and cost.

### Rkxt

Use paired update utility, explicit `T_formula`/`A_impl` comparisons, fixed and retrained alternatives, HAR/BRR, and Figure R4 to address whether TCFF is more than a renamed gate and where it fails.

### mDyq

Use the mild-to-severe corruption curves, component table, three-seed coverage, and the same-device efficiency benchmark for the requested engineering validation.

### Area Chair

G0 reproduction, multi-seed component retraining, controlled robustness, reliability/action quality, simple alternatives, and efficiency directly address robustness, contribution, and computational cost.

## Main tables and figures

- Figure R1: utility distribution and harmful-update rate.
- Figure R2: formula-trust and executed-acceptance AUPRC/AURC, reliability, and U×G phase diagram.
- Figure R3: controlled corruption mechanism chain.
- Table R1: fixed interventions and independently trained variants.
- Table R2: efficiency and estimator/reliability metrics when benchmark output is present.
- Figure R4: action quadrants and preregistered case selection.

## Negative or mixed findings

{chr(10).join('- ' + note for note in negative_notes)}

## Recommended claim scope

Describe the results as paired inference-time auxiliary-update utility and within-model response under controlled interventions. Do not call the utility a causal ground truth, do not generalize beyond the measured cohorts/corruptions, and do not claim that TCFF wins every metric or every failure regime.

## Exact numbers safe to quote

The authoritative ledger contains {len(exact)} values: `06_summary_tables/exact_numbers_safe_to_quote.csv`. Every quoted number must pass `tcff_verify_quote_table.py` after the final refresh.

## Unresolved issues

{chr(10).join('- ' + item for item in unresolved)}
"""
    path = results_root / "FINAL_RESULT_SUMMARY.md"
    temp = path.with_suffix(".md.tmp")
    temp.write_text(text, encoding="utf-8")
    temp.replace(path)


def normalized_verdict_status(verdict: str) -> str:
    value = str(verdict).strip().lower()
    if value.startswith("strongly supported"):
        return "supported"
    if value.startswith("partially supported"):
        return "partially_supported"
    if value.startswith("not supported"):
        return "not_supported"
    return "inconclusive"


def write_handoff_artifacts(results_root: Path, inventory: pd.DataFrame,
                            findings: pd.DataFrame, p0a: pd.DataFrame,
                            validity: pd.DataFrame, variants: pd.DataFrame,
                            mechanism: pd.DataFrame, action: pd.DataFrame,
                            quadrants: pd.DataFrame, selections: pd.DataFrame,
                            exact: pd.DataFrame, n_boot: int) -> None:
    """Write the compact bridge-file deliverables used by the reply writer."""
    workflow = workflow_completion(results_root, inventory)
    fixed = workflow["fixed_checkpoint"]
    verdicts = verdict_table(p0a, validity, variants, mechanism, quadrants, fixed)

    status = inventory.copy()
    if not status.empty:
        status["included_in_quantitative_pooling"] = status["status"].eq("success")
    atomic_csv(results_root / "experiment_status.csv", status)
    atomic_csv(results_root / "exact_numbers_safe_to_quote.csv", exact)

    recommended = {
        "H1": ("Auxiliary updates show measurable local heterogeneity, including "
               "beneficial and harmful component-level outcomes under the executed protocol."),
        "H2": ("The saved signals are informative to the extent shown by their "
               "case-clustered ranking and action metrics; U is not complete reliability."),
        "H3": ("Describe G's incremental role only through the reported U-only, G-only, "
               "and full comparisons; do not claim that both factors are universally necessary."),
        "H4": ("Separate the manuscript-formula diagnostic score T_formula=G(1-U) "
               "from submitted-model interpolation A_impl; product-performance claims use "
               "only the explicitly executed product intervention and its matched retraining."),
        "H5": ("Controlled auxiliary perturbations provide mechanism-consistent, "
               "within-model evidence under the tested corruption families."),
        "H6": ("Capacity and generic-gating explanations are only partially addressed "
               "without a parameter-matched ordinary gate."),
        "H7": ("Incorrect acceptance and over-suppression are measurable boundaries; "
               "common-mode failure is not claimed without a dedicated run."),
    }
    handoff_rows = []
    for row in verdicts.itertuples(index=False):
        match = re.match(r"(H\d+)", str(row.hypothesis))
        hypothesis_id = match.group(1) if match else str(row.hypothesis)
        handoff_rows.append({
            "hypothesis_id": hypothesis_id,
            "status": normalized_verdict_status(row.verdict),
            "supporting_experiments": row.evidence_files,
            "primary_effect": "See the named audited source tables; no value is copied from logs.",
            "ci": ("Case-clustered percentile 95% CI where reported; "
                   f"this refresh uses {n_boot} replicates."),
            "limitations": row.caveat,
            "recommended_wording": recommended.get(hypothesis_id, row.caveat),
        })
    handoff_verdicts = pd.DataFrame(handoff_rows)
    atomic_csv(results_root / "hypothesis_verdicts.csv", handoff_verdicts)

    verdict_lookup = (handoff_verdicts.set_index("hypothesis_id")["status"].to_dict()
                      if not handoff_verdicts.empty else {})
    claims = pd.DataFrame([
        {
            "claim_id": "C1", "claim_text": recommended["H1"],
            "evidence_level": verdict_lookup.get("H1", "inconclusive"),
            "experiments": "P0-A",
            "figures_tables": "Figure R1; p0_a_utility_summary.csv",
            "safe_to_state": "yes_with_caveat" if not p0a.empty else "no",
            "required_caveat": "Observed paired inference-time utility, not causal ground truth.",
        },
        {
            "claim_id": "C2", "claim_text": recommended["H2"],
            "evidence_level": verdict_lookup.get("H2", "inconclusive"),
            "experiments": "P0-B; P0-F",
            "figures_tables": "Figure R2; p0_b_signal_validity.csv; p0_f_har_brr.csv",
            "safe_to_state": "yes_with_caveat" if not validity.empty else "no",
            "required_caveat": "Ranking, calibration, and operational trust are distinct.",
        },
        {
            "claim_id": "C3", "claim_text": recommended["H3"],
            "evidence_level": verdict_lookup.get("H3", "inconclusive"),
            "experiments": "P0-D",
            "figures_tables": "Table R1",
            "safe_to_state": "measured_comparison_only" if not variants.empty else "no",
            "required_caveat": "Separate fixed-checkpoint interventions from retrained variants.",
        },
        {
            "claim_id": "C4", "claim_text": recommended["H4"],
            "evidence_level": verdict_lookup.get("H4", "inconclusive"),
            "experiments": "P0-D",
            "figures_tables": "Table R1",
            "safe_to_state": "measured_comparison_only" if not variants.empty else "no",
            "required_caveat": "No claim of theoretical or universal optimality.",
        },
        {
            "claim_id": "C5", "claim_text": recommended["H5"],
            "evidence_level": verdict_lookup.get("H5", "inconclusive"),
            "experiments": "P0-C",
            "figures_tables": "Figure R3; p0_c_degradation_slopes.csv",
            "safe_to_state": "yes_with_caveat" if not mechanism.empty else "no",
            "required_caveat": "Within-model controlled interventions, not real-world causal proof.",
        },
        {
            "claim_id": "C6", "claim_text": recommended["H6"],
            "evidence_level": verdict_lookup.get("H6", "inconclusive"),
            "experiments": "P0-D; P0-H",
            "figures_tables": "Table R1; Table R2",
            "safe_to_state": ("measured_comparison_only"
                              if workflow["efficiency_ready"] else "no"),
            "required_caveat": "A parameter-matched ordinary gate was not completed.",
        },
        {
            "claim_id": "C7", "claim_text": recommended["H7"],
            "evidence_level": verdict_lookup.get("H7", "inconclusive"),
            "experiments": "P0-G",
            "figures_tables": "Figure R4; p0_g_quadrants.csv; predeclared_case_selection.csv",
            "safe_to_state": "yes_with_caveat" if not quadrants.empty else "no",
            "required_caveat": "Only executed failure modes are represented.",
        },
    ])
    atomic_csv(results_root / "claim_support_matrix.csv", claims)

    reviewer_map = f"""# Reviewer evidence map

Generated: {utc_now()}

This file maps one unified evidence system to each reviewer; it does not create reviewer-specific experimental conclusions.

## Area Chair

- Reliability and robustness: Figures R1-R3; `06_summary_tables/p0_a_utility_summary.csv`, `p0_b_signal_validity.csv`, and `p0_c_degradation_slopes.csv`.
- Component contribution and simple alternatives: `06_summary_tables/Table_R1_components.csv`, `Table_R1_rebuttal.csv`, and `Table_R1_product_fixed_corruption.csv`.
- Reliability of reported numbers: `11_quality_audit/DATA_QUALITY_REPORT.md`, `ANALYSIS_VALIDATION_REPORT.md`, and the independently verified quote ledger.
- Efficiency: `09_efficiency/efficiency_summary.csv` and `06_summary_tables/Table_R2_efficiency.csv` when present.

## Rkxt

- Whether TCFF is more than a renamed gate: paired update utility (Figure R1), explicit `1-U`/`G`/`T_formula`/`A_impl` construct tests (Figure R2), fixed and retrained alternatives (Table R1), and frozen-threshold HAR/BRR.
- Falsifiability: Figure R4 retains incorrect acceptance and over-suppression; no missing common-mode run is inferred.
- Formula scope: `T_formula=G(1-U)` is tested as a diagnostic score on historical runs; performance of the product rule comes only from separately labelled runs in which `A_product=G(1-U)` is actually executed.

## pCat

- Proxy stability and shift: source-versus-external rows in Figure R1/R2 source tables and controlled corruption curves in Figure R3.
- Probability controls: confidence and entropy rows in `p0_b_signal_validity.csv` when the dedicated probability matrix completes.
- Cost: the same-device, batch-1, FP32 efficiency manifest and Table R2.

## mDyq

- Mild-to-severe auxiliary corruption: Gaussian, channel dropout, spatial translation, and donor mixing are kept in separate physical/statistical units.
- Ablation and engineering evidence: Table R1, run manifests, three-seed retraining coverage, and Table R2.

## Completion snapshot

- Fixed-checkpoint runs: {fixed['successful']}/{fixed['planned']} successful.
- Executed-product fixed-checkpoint runs: {workflow['product_fixed_intervention']['successful']}/{workflow['product_fixed_intervention']['planned']} successful.
- Training jobs: {workflow['training']['successful']}/{workflow['training']['planned']} successful.
- Trained evaluation runs: {workflow['trained_evaluation']['successful_runs']}/{workflow['trained_evaluation']['planned_runs']} successful.
- Bootstrap replicates in this refresh: {n_boot}.
"""
    atomic_text(results_root / "reviewer_evidence_map.md", reviewer_map)

    negative = [
        "# Negative and mixed findings",
        "",
        f"Generated: {utc_now()}",
        "",
        "These findings are retained by construction and must be reflected in the final wording.",
        "",
    ]
    if not validity.empty:
        primary_validity = validity[
            validity["target"].eq("component") &
            validity["condition"].isin(["clean", "all_corruption"])]
        below = primary_validity[
            primary_validity["auprc_beneficial"] <=
            primary_validity["prevalence_beneficial"]]
        negative.append(
            f"- {len(below)}/{len(primary_validity)} primary component clean/pooled-corruption signal rows did not exceed the beneficial-prevalence AUPRC baseline.")
    if not quadrants.empty:
        primary_quadrants = quadrants[
            quadrants["signal"].eq(FORMULA_TRUST) &
            quadrants["condition"].isin(["clean", "all_corruption"])]
        incorrect = int(primary_quadrants[primary_quadrants["utility_quadrant"].eq(
            "incorrect_acceptance")]["n_rows"].sum())
        over = int(primary_quadrants[primary_quadrants["utility_quadrant"].eq(
            "over_suppression")]["n_rows"].sum())
        negative.append(
            f"- For level-specific formula-T decisions on clean plus pooled-corruption strata, frozen source-selected thresholds yield {incorrect} incorrect-acceptance rows and {over} over-suppression rows in the currently completed runs.")
    if not mechanism.empty:
        families = sorted(mechanism["corruption_type"].dropna().astype(str).unique())
        negative.append(
            "- Corruption-family responses are heterogeneous and retain separate units; "
            f"no pooled monotonicity claim is made across {', '.join(families)}.")
    if not variants.empty:
        negative.append(
            "- Clean performance, corruption robustness, and action quality can rank variants differently; Table R1 is not reduced to a single winner label.")
    failed_runs = inventory[inventory["status"].isin(["failed", "partial", "unknown"])]
    negative.append(
        f"- Failed/partial/unknown runs preserved in the audit inventory: {len(failed_runs)}.")
    if not workflow["probability_signals"]["complete"]:
        negative.append("- Confidence/entropy controls are incomplete in this refresh.")
    if not workflow["selected_case_maps"]["complete"]:
        negative.append("- Full selected-case trust-map panels are incomplete in this refresh.")
    negative.append(
        "- A parameter-matched ordinary gate, shuffled/inverted trust, oracle selection, and a dedicated common-mode corruption are outside the completed minimum package unless separately listed as successful runs.")
    atomic_text(results_root / "negative_and_mixed_findings.md",
                "\n".join(negative) + "\n")

    claim_scope = f"""# Recommended claim scope

Generated: {utc_now()}

## Safe framing

- TCFF is evaluated as uncertainty-aware reliability control for local auxiliary feature intervention in topology-sensitive CoW segmentation.
- `U` is an evidence-insufficiency signal, `G` is task-conditioned compatibility, and manuscript-formula `T_formula=G(1-U)` is evaluated separately from the executed interpolation coefficient `A_impl`.
- Auxiliary-update utility is an observed paired inference-time outcome; it is not causal ground truth or a true reliability label.
- Controlled auxiliary corruptions provide intervention-supported, mechanism-consistent evidence within the tested model and cohorts.
- External cohorts are evaluation-only and must be named separately.

## Claims requiring measured qualification

- Do not claim that `T_formula` universally outperforms `G`, `1-U`, or `A_impl`; use the exact ranking/action rows for each dataset, level, target, and condition.
- Do not describe historical `full` performance as product-gated. Any product-performance statement must cite the separately labelled product intervention/retraining rows, and no empirical ranking establishes theoretical optimality.
- Do not claim robustness across all domains, corruption types, metrics, or anatomical structures when the reported effects are heterogeneous.
- Do not equate level-global `U` with spatial reliability; spatial discrimination in `T_formula` is supplied by `G`, and output interpolation is separately described by `A_impl`.
- Do not claim that every alternative explanation has been eliminated without a parameter-matched ordinary-gate control.

## Current readiness

- Core workflow complete: {workflow['core_complete']}.
- Fixed-checkpoint matrix complete: {fixed['complete']}.
- Final {DEFAULT_BOOTSTRAPS}-replicate bootstrap used: {n_boot >= DEFAULT_BOOTSTRAPS}.
- Final OpenReview wording must draw numeric values only from `exact_numbers_safe_to_quote.csv` after independent verification.
"""
    atomic_text(results_root / "recommended_claim_scope.md", claim_scope)

    required_variants = {"full", "product", "residual", "u_only", "g_only"}
    observed_variants = set(variants.get("variant", pd.Series(dtype=str)).dropna())
    mechanism_families = set(
        mechanism.get("corruption_type", pd.Series(dtype=str)).dropna())
    high_failures = findings[
        findings["severity"].isin(["critical", "high"]) &
        findings["status"].eq("fail")]
    g2_checks = {
        "fixed_checkpoint_matrix_complete": fixed["complete"],
        "executed_product_matrix_complete": workflow[
            "product_fixed_intervention"]["complete"],
        "p0_a_available": not p0a.empty,
        "p0_b_available": not validity.empty,
        "at_least_three_corruption_families": len(mechanism_families) >= 3,
        "required_variants_present": required_variants <= observed_variants,
        "risk_action_analysis_available": not action.empty,
        "efficiency_available": workflow["efficiency_ready"],
        "no_high_or_critical_quality_failure": high_failures.empty,
    }
    g2_status = "PASS" if all(g2_checks.values()) else "PENDING"
    g2_text = "# Decision gate G2\n\nStatus: **" + g2_status + "**\n\n" + "\n".join(
        f"- [{'x' if value else ' '}] {key}" for key, value in g2_checks.items()) + "\n"
    atomic_text(results_root / "02_manifests" / "G2_ACCEPTANCE.md", g2_text)

    g3_checks = {
        "g2_passed": g2_status == "PASS",
        "failure_quadrants_available": not quadrants.empty,
        "predeclared_case_selection_available": not selections.empty,
        "negative_findings_preserved": True,
    }
    g3_status = "PASS" if all(g3_checks.values()) else "PENDING"
    g3_text = "# Decision gate G3\n\nStatus: **" + g3_status + "**\n\n" + "\n".join(
        f"- [{'x' if value else ' '}] {key}" for key, value in g3_checks.items()) + "\n\n"
    g3_text += ("Scope note: continuous gate-dose analysis and dedicated common-mode "
                "corruption remain optional follow-ups and are not inferred from other runs.\n")
    atomic_text(results_root / "02_manifests" / "G3_ACCEPTANCE.md", g3_text)

    ledger_path = results_root / "06_summary_tables" / "exact_numbers_safe_to_quote.csv"
    verification = read_json(
        results_root / "11_quality_audit" / "quote_table_verification.json", {}) or {}
    verified_current = bool(
        ledger_path.exists() and verification.get("status") == "pass" and
        verification.get("ledger_sha256") == sha256_file(ledger_path))
    g5_checks = {
        "core_workflow_complete": workflow["core_complete"],
        "bootstrap_2000": n_boot >= DEFAULT_BOOTSTRAPS,
        "no_high_or_critical_quality_failure": high_failures.empty,
        "quote_ledger_nonempty": not exact.empty,
        "current_quote_ledger_independently_verified": verified_current,
        "claim_support_matrix_written": True,
        "negative_and_mixed_findings_written": True,
        "recommended_claim_scope_written": True,
    }
    g5_status = "PASS" if all(g5_checks.values()) else "PENDING"
    g5_text = "# Decision gate G5\n\nStatus: **" + g5_status + "**\n\n" + "\n".join(
        f"- [{'x' if value else ' '}] {key}" for key, value in g5_checks.items()) + "\n"
    atomic_text(results_root / "02_manifests" / "G5_ACCEPTANCE.md", g5_text)


def postprocessing_manifest(results_root: Path, inventory: pd.DataFrame,
                            script_path: Path, n_boot: int) -> None:
    manifest_hashes = []
    for relative in inventory.get("manifest_path", pd.Series(dtype=str)).dropna():
        path = results_root / str(relative)
        if path.exists():
            manifest_hashes.append({"path": str(relative), "sha256": sha256_file(path)})
    protocol = results_root / "01_configs" / "POSTPROCESSING_PROTOCOL.md"
    atomic_json(results_root / "02_manifests" / "postprocessing_manifest.json", {
        "created_at": utc_now(),
        "script": script_path.name,
        "script_sha256": sha256_file(script_path),
        "protocol_sha256": sha256_file(protocol) if protocol.exists() else None,
        "bootstrap_replicates": n_boot,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "signal_semantics_version": SIGNAL_SEMANTICS_VERSION,
        "legacy_T_interpretation": IMPLEMENTED_ACCEPTANCE,
        "paper_formula_trust_field": FORMULA_TRUST,
        "successful_runs_included": int(inventory["status"].eq("success").sum()),
        "raw_run_manifests": manifest_hashes,
    })


def ensure_output_folders(results_root: Path) -> None:
    for name in [
            "00_project_discovery", "01_configs", "02_manifests", "03_raw_runs",
            "04_case_metrics", "05_region_metrics", "06_summary_tables",
            "07_figures", "08_failure_cases", "09_efficiency", "10_logs",
            "11_quality_audit"]:
        (results_root / name).mkdir(parents=True, exist_ok=True)


def write_start_here(results_root: Path, inventory: pd.DataFrame,
                     findings: pd.DataFrame, n_boot: int) -> None:
    workflow = workflow_completion(results_root, inventory)
    high_failures = findings[
        findings["severity"].isin(["critical", "high"]) &
        findings["status"].eq("fail")]
    final_ready = bool(
        workflow["core_complete"] and n_boot >= DEFAULT_BOOTSTRAPS and
        high_failures.empty)
    state = "FINAL VERIFIED CANDIDATE" if final_ready else "INCREMENTAL — NOT FINAL"
    panel_path = results_root / "08_failure_cases" / "case_panel_index.csv"
    panel_count = 0
    if panel_path.exists():
        try:
            panel_index = pd.read_csv(panel_path)
            panel_count = int(panel_index.get(
                "status", pd.Series(dtype=str)).eq("success").sum())
        except Exception:
            panel_count = 0
    text = f"""# START HERE

Status: **{state}**

Generated: {utc_now()}

This package uses trust-signal semantics version
`{SIGNAL_SEMANTICS_VERSION}`. Historical raw `T` remains immutable and is
reported only as executed acceptance `A_impl`; manuscript-formula trust is
recomputed as `T_formula=G(1-U)` on the same row/support. See
`11_quality_audit/TRUST_SIGNAL_SEMANTICS_CORRECTION.md` and
`TRUST_SIGNAL_SEMANTICS_AUDIT.json` before interpreting signal results.

## Current completion

- fixed inference: {workflow['fixed_checkpoint']['successful']}/{workflow['fixed_checkpoint']['planned']}
- executed-product fixed inference: {workflow['product_fixed_intervention']['successful']}/{workflow['product_fixed_intervention']['planned']}
- probability runs: {workflow['probability_signals']['completed']}/{workflow['probability_signals']['planned']}
- training jobs: {workflow['training']['successful']}/{workflow['training']['planned']}
- trained evaluations: {workflow['trained_evaluation']['successful_runs']}/{workflow['trained_evaluation']['planned_runs']}
- efficiency ready: {workflow['efficiency_ready']}
- corrected case panels currently indexed: {panel_count}
- bootstrap replicates in this refresh: {n_boot}
- high/critical audit failures: {len(high_failures)}

## Read first

1. `11_quality_audit/TRUST_SIGNAL_SEMANTICS_AUDIT.json`
2. `11_quality_audit/DATA_QUALITY_REPORT.md`
3. `06_summary_tables/exact_numbers_safe_to_quote.csv`
4. `06_summary_tables/Table_R1_product_fixed_corruption.csv`
5. `06_summary_tables/p0_b_signal_validity.csv`
6. `06_summary_tables/Figure_R3_source.csv`
7. `07_figures/Figure_R2_trust_validation.png`
8. `07_figures/Figure_R3_controlled_interventions.png`
9. `08_failure_cases/case_panel_index.csv`

No raw run, checkpoint, prediction, or legacy map is overwritten by this
refresh. Do not use this package as final until the status above is final and
the independent quote-ledger verifier passes.
"""
    atomic_text(results_root / "START_HERE.md", text)


def main(args: argparse.Namespace) -> int:
    results_root = Path(args.results_root).resolve()
    ensure_output_folders(results_root)
    inventory, _ = discover_runs(results_root)
    atomic_csv(results_root / "11_quality_audit" / "run_inventory.csv", inventory)
    case, class_metrics, region = load_successful_outputs(
        results_root, inventory, include_partial=args.include_partial)
    case, region = add_signal_semantics(case, region)
    write_signal_semantics_note(results_root)
    atomic_csv(results_root / "04_case_metrics" / "all_case_metrics.csv", case)
    atomic_csv(results_root / "04_case_metrics" / "all_class_metrics.csv", class_metrics)
    if not region.empty:
        atomic_parquet(results_root / "05_region_metrics" / "all_region_metrics.parquet",
                       region)

    findings = audit_outputs(results_root, inventory, case, region)
    atomic_csv(results_root / "11_quality_audit" / "data_quality_findings.csv",
               findings)

    baseline = baseline_reproduction_table(results_root)
    empty = pd.DataFrame()
    p0a_rows = p0a_summary = validity = calibration = phase_ug = phase_ufdd = empty
    mechanism_case = mechanism_summary = slopes = empty
    thresholds = action = curves = quadrants = selections = empty
    variants = trained_corruption = product_fixed = variant_reliability = table_r1 = empty
    if not region.empty:
        p0a_rows, p0a_summary = analyse_p0_a(region, results_root, args.bootstrap)
        validity, calibration, phase_ug, phase_ufdd = analyse_p0_b(
            region, results_root, args.bootstrap)
        mechanism_case, mechanism_summary, slopes = analyse_p0_c(
            region, case, results_root, args.bootstrap)
        thresholds, action, curves, quadrants, selections = analyse_p0_f_and_g(
            region, case, results_root, args.bootstrap)
        variants = analyse_variants(case, results_root, args.bootstrap)
        trained_corruption = analyse_trained_corruption(
            case, results_root, args.bootstrap)
        product_fixed = analyse_product_fixed_corruption(
            case, results_root, args.bootstrap)
        variant_reliability = analyse_variant_reliability(region, results_root)
        table_r1 = build_table_r1(variants, variant_reliability,
                                  trained_corruption, results_root)
        plot_r1(p0a_rows, p0a_summary, results_root)
        plot_r2(validity, calibration, phase_ug, results_root)
        plot_r3(mechanism_summary, results_root)
        plot_r4(quadrants, selections, results_root)
    else:
        variants = analyse_variants(case, results_root, args.bootstrap)
        trained_corruption = analyse_trained_corruption(
            case, results_root, args.bootstrap)
        product_fixed = analyse_product_fixed_corruption(
            case, results_root, args.bootstrap)
        variant_reliability = analyse_variant_reliability(region, results_root)
        table_r1 = build_table_r1(variants, variant_reliability,
                                  trained_corruption, results_root)

    exact = make_exact_numbers(
        results_root, baseline, p0a_summary, validity, action, variants,
        mechanism_summary, product_fixed)
    write_chart_map(results_root)
    write_quality_reports(results_root, inventory, case, region, findings,
                          args.bootstrap)
    write_final_summary(results_root, inventory, findings, p0a_summary,
                        validity, variants, mechanism_summary, quadrants, exact,
                        args.bootstrap)
    write_handoff_artifacts(
        results_root, inventory, findings, p0a_summary, validity, variants,
        mechanism_summary, action, quadrants, selections, exact,
        args.bootstrap)
    postprocessing_manifest(results_root, inventory, Path(__file__).resolve(),
                            args.bootstrap)
    write_start_here(results_root, inventory, findings, args.bootstrap)

    completion = matrix_completion(results_root, inventory)
    package = workflow_completion(results_root, inventory)
    summary = {
        "status": ("complete" if package["core_complete"] and
                   args.bootstrap >= DEFAULT_BOOTSTRAPS else "incremental"),
        "runs": len(inventory),
        "case_rows": len(case),
        "region_rows": len(region),
        "fixed_matrix": completion,
        "workflow": package,
        "quality_failures": int(findings["status"].eq("fail").sum()),
        "quote_values": len(exact),
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    if args.strict and (not package["core_complete"] or
                        args.bootstrap < DEFAULT_BOOTSTRAPS or
                        findings[
                            findings["severity"].isin(["critical", "high"]) &
                            findings["status"].eq("fail")].shape[0]):
        return 2
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_root", default=str(
        Path(__file__).resolve().parent / "results_tcff_rebuttal"))
    parser.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAPS)
    parser.add_argument("--include_partial", action="store_true")
    parser.add_argument("--strict", action="store_true")
    raise SystemExit(main(parser.parse_args()))
