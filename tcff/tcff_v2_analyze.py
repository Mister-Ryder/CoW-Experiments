"""Read-only, failure-tolerant analysis for the TCFF V2 rebuttal controls.

The script scans only successful raw runs and writes a new, self-contained
``12_v2_analysis`` directory.  It never edits raw runs or the legacy derived
tables.  Statistical resampling is clustered by ``seed/case_id``; region rows
are first collapsed to one observation per run, case, seed and feature level.

The implementation intentionally accepts partially populated result trees:
missing groups, columns, parquet engines, or reference conditions are recorded
in the failure/negative-finding tables instead of being silently dropped.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


PROTOCOL = "TCFF_V2_ANALYSIS_2026-07-30"
DEFAULT_SEED = 20260730
HIGHER_IS_BETTER = {
    "dice": True,
    "cldice": True,
    "hd95": False,
    "beta0_error": False,
    "clsavg": True,
    "cls_beta0": False,
    "det_f1": True,
}
CASE_METRICS = tuple(HIGHER_IS_BETTER)
V2_EXPERIMENTS = {
    "V2_PROBABILITY_CORRECTED",
    "V2_TEMPERATURE_SENSITIVITY",
    "V2_CONSTANT_ACCEPTANCE",
    "V2_GATE_TRANSFORM",
    "V2_FIXED_POLICY_REPLAY",
    "V2_GATE_DOSE_RESPONSE",
    "V2_GATE_MEAN_MATCHED",
    "V2_PRODUCT_RETRAIN",
    "V2_PRODUCT_TRAINED_EVAL_CLEAN",
    "V2_PRODUCT_TRAINED_EVAL_CORRUPTION",
}
ANALYSIS_EXPERIMENTS = V2_EXPERIMENTS | {
    "P0_AB_CLEAN_MECHANISM",
    "P0_B_PROBABILITY_SIGNALS",
    "P0_C1_CHANNEL_DROPOUT",
    "P0_C1_GAUSSIAN",
    "P0_C2_TRANSLATION",
    "P0_C3_DONOR_MIX",
    "P0_D_FIXED_CKPT_INTERVENTIONS",
    "P0_D_PRODUCT_FIXED_CLEAN",
    "P0_D_PRODUCT_FIXED_CORRUPTION",
    "P0_D_TRAINED_EVAL_CLEAN",
    "P0_D_TRAINED_EVAL_CORRUPTION",
}


def read_json(path: Path) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def finite(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def scalar_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, ensure_ascii=False)
    return str(value)


def canonical_run_id(run_id: str) -> str:
    return re.sub(r"__retry\d+$", "", str(run_id or ""))


def canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(payload).hexdigest()


def atomic_csv(path: Path, frame: pd.DataFrame,
               columns: Optional[Sequence[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if frame.empty and columns is not None:
        frame = pd.DataFrame(columns=list(columns))
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def status_is_success(run_dir: Path) -> bool:
    status = read_json(run_dir / "status.json")
    manifest = read_json(run_dir / "run_manifest.json")
    return (
        status.get("status") == "success" and
        manifest.get("status", "success") == "success")


def semantic_identity(manifest: Mapping[str, Any], run_dir: Path) -> str:
    corruption = manifest.get("corruption") or {}
    intervention = manifest.get("intervention") or {}
    identity = {
        "experiment_id": manifest.get("experiment_id", run_dir.parent.name),
        "canonical_run_id": canonical_run_id(
            manifest.get("canonical_run_id") or
            manifest.get("run_id") or run_dir.name),
        "dataset": manifest.get("dataset"),
        "model_variant": manifest.get("model_variant"),
        "checkpoint_hash": manifest.get("checkpoint_hash"),
        "seed": manifest.get("seed"),
        "corruption": {
            "type": corruption.get("type"),
            "severity": corruption.get("severity"),
            "realization": corruption.get("realization"),
        },
        "intervention": {
            "gate_alpha": intervention.get("gate_alpha"),
            "gate_scale": intervention.get("gate_scale"),
            "temperature_scale": intervention.get("temperature_scale"),
            "gate_transform": intervention.get("gate_transform"),
        },
    }
    return canonical_hash(identity)


def scan_success_runs(results_root: Path) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    raw_root = results_root / "03_raw_runs"
    attempts: Dict[str, List[Dict[str, Any]]] = {}
    failures: List[Dict[str, Any]] = []
    if not raw_root.exists():
        failures.append({
            "severity": "high",
            "category": "input",
            "finding": f"raw run root does not exist: {raw_root}",
            "evidence": str(raw_root),
        })
        return pd.DataFrame(), failures
    for status_path in sorted(raw_root.rglob("status.json")):
        run_dir = status_path.parent
        if not (run_dir / "run_manifest.json").exists():
            continue
        if not status_is_success(run_dir):
            continue
        manifest = read_json(run_dir / "run_manifest.json")
        if manifest.get("experiment_id") not in ANALYSIS_EXPERIMENTS:
            continue
        identity = semantic_identity(manifest, run_dir)
        finished = (
            manifest.get("finished_at") or
            read_json(status_path).get("finished_at") or "")
        row = {
            "semantic_id": identity,
            "experiment_id": manifest.get(
                "experiment_id", run_dir.parent.name),
            "run_id": manifest.get("run_id", run_dir.name),
            "canonical_run_id": canonical_run_id(
                manifest.get("canonical_run_id") or
                manifest.get("run_id") or run_dir.name),
            "dataset": manifest.get("dataset"),
            "model_variant": manifest.get("model_variant"),
            "seed": manifest.get("seed"),
            "checkpoint_hash": manifest.get("checkpoint_hash"),
            "data_split_hash": manifest.get("data_split_hash"),
            "finished_at": finished,
            "mtime_ns": run_dir.stat().st_mtime_ns,
            "run_dir": str(run_dir),
            "has_case_metrics": (run_dir / "case_metrics.csv").exists(),
            "has_internal_signals": (
                run_dir / "internal_signals.parquet").exists(),
            "has_aggregate_metrics": (
                run_dir / "aggregate_metrics.json").exists(),
            "manifest": manifest,
        }
        attempts.setdefault(identity, []).append(row)
    selected: List[Dict[str, Any]] = []
    for identity, rows in attempts.items():
        rows.sort(key=lambda item: (
            scalar_text(item["finished_at"]), int(item["mtime_ns"])))
        chosen = dict(rows[-1])
        chosen["duplicate_success_attempts"] = len(rows)
        chosen["superseded_success_dirs"] = ";".join(
            item["run_dir"] for item in rows[:-1])
        selected.append(chosen)
        if len(rows) > 1:
            failures.append({
                "severity": "warning",
                "category": "duplicate_success",
                "finding": (
                    f"{len(rows)} successful attempts share semantic identity "
                    f"{identity}; newest selected"),
                "evidence": ";".join(item["run_dir"] for item in rows),
            })
    selected.sort(key=lambda item: (
        scalar_text(item["experiment_id"]),
        scalar_text(item["canonical_run_id"])))
    return pd.DataFrame(selected), failures


def manifest_defaults(manifest: Mapping[str, Any]) -> Dict[str, Any]:
    corruption = manifest.get("corruption") or {}
    intervention = manifest.get("intervention") or {}
    return {
        "experiment_id": manifest.get("experiment_id"),
        "run_id": manifest.get("run_id"),
        "canonical_run_id": canonical_run_id(
            manifest.get("canonical_run_id") or manifest.get("run_id")),
        "dataset": manifest.get("dataset"),
        "model_variant": manifest.get("model_variant"),
        "seed": manifest.get("seed"),
        "corruption_type": corruption.get("type", "none"),
        "severity": corruption.get("severity", 0.0),
        "realization": corruption.get("realization", 0),
        "gate_alpha": intervention.get("gate_alpha"),
        "gate_scale": intervention.get("gate_scale", 1.0),
        "temperature_scale": intervention.get("temperature_scale", 1.0),
        "gate_transform": intervention.get("gate_transform", "identity"),
        "checkpoint_hash": manifest.get("checkpoint_hash"),
    }


def fill_defaults(frame: pd.DataFrame, defaults: Mapping[str, Any]) -> pd.DataFrame:
    result = frame.copy()
    for key, value in defaults.items():
        if key not in result:
            result[key] = value
        else:
            result[key] = result[key].where(result[key].notna(), value)
    return result


REVIEWER_CORE_SIGNAL_EXPERIMENTS = {
    "V2_PROBABILITY_CORRECTED",
    "P0_C1_CHANNEL_DROPOUT",
    "P0_C1_GAUSSIAN",
    "P0_C2_TRANSLATION",
    "P0_C3_DONOR_MIX",
}


def load_artifacts(
        inventory: pd.DataFrame,
        signal_experiment_ids: Optional[set[str]] = None) -> Tuple[
        pd.DataFrame, pd.DataFrame, List[Dict[str, Any]]]:
    cases: List[pd.DataFrame] = []
    signals: List[pd.DataFrame] = []
    failures: List[Dict[str, Any]] = []
    if inventory.empty:
        return pd.DataFrame(), pd.DataFrame(), failures
    for _, item in inventory.iterrows():
        run_dir = Path(item["run_dir"])
        manifest = item["manifest"]
        defaults = manifest_defaults(manifest)
        defaults.update({
            "semantic_id": item["semantic_id"],
            "run_dir": str(run_dir),
        })
        case_path = run_dir / "case_metrics.csv"
        if case_path.exists():
            try:
                frame = fill_defaults(pd.read_csv(case_path), defaults)
                cases.append(frame)
            except Exception as exc:
                failures.append({
                    "severity": "high",
                    "category": "case_metrics_read",
                    "finding": f"{type(exc).__name__}: {exc}",
                    "evidence": str(case_path),
                })
        signal_path = run_dir / "internal_signals.parquet"
        load_signal = (
            signal_experiment_ids is None or
            str(item.get("experiment_id")) in signal_experiment_ids)
        if load_signal and signal_path.exists():
            try:
                frame = fill_defaults(pd.read_parquet(signal_path), defaults)
                signals.append(frame)
            except Exception as exc:
                failures.append({
                    "severity": "high",
                    "category": "internal_signals_read",
                    "finding": f"{type(exc).__name__}: {exc}",
                    "evidence": str(signal_path),
                })
    case = pd.concat(cases, ignore_index=True, sort=False) if cases else pd.DataFrame()
    signal = (
        pd.concat(signals, ignore_index=True, sort=False)
        if signals else pd.DataFrame())
    for frame in (case, signal):
        if frame.empty:
            continue
        for column in (
                "seed", "realization", "severity", "gate_alpha", "gate_scale",
                "temperature_scale"):
            if column in frame:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return case, signal, failures


def explicit_signal_semantics_audit(
        inventory: pd.DataFrame, raw_signal: pd.DataFrame) -> Dict[str, Any]:
    """Prove that corrected runs natively persisted the V2 signal fields.

    This audit intentionally runs before ``normalize_signal_columns`` so that
    its verdict cannot be satisfied by any backward-compatibility fallback.
    """
    required_fields = (
        "U", "G", "T_formula", "A_impl", "A_impl_raw",
        "FDD_paper", "FDD_legacy",
    )
    expected_runs = 39
    corrected_inventory = inventory[
        inventory.get(
            "experiment_id", pd.Series(index=inventory.index, dtype=str)
        ).eq("V2_PROBABILITY_CORRECTED")
    ].copy()
    corrected_ids = set(
        corrected_inventory.get(
            "semantic_id", pd.Series(dtype=str)
        ).dropna().astype(str))
    corrected_signal = (
        raw_signal[
            raw_signal.get(
                "experiment_id",
                pd.Series(index=raw_signal.index, dtype=str),
            ).eq("V2_PROBABILITY_CORRECTED")
        ].copy()
        if not raw_signal.empty else pd.DataFrame()
    )
    if {"U", "G"}.issubset(corrected_signal.columns):
        u_numeric = pd.to_numeric(
            corrected_signal["U"], errors="coerce")
        g_numeric = pd.to_numeric(
            corrected_signal["G"], errors="coerce")
        eligible_mask = pd.Series(
            np.isfinite(u_numeric.to_numpy(dtype=float)) &
            np.isfinite(g_numeric.to_numpy(dtype=float)),
            index=corrected_signal.index)
    else:
        eligible_mask = pd.Series(
            False, index=corrected_signal.index, dtype=bool)
    field_audit: Dict[str, Dict[str, Any]] = {}
    for field in required_fields:
        column_present = field in corrected_signal.columns
        numeric = (
            pd.to_numeric(corrected_signal[field], errors="coerce")
            if column_present else
            pd.Series(np.nan, index=corrected_signal.index, dtype=float)
        )
        finite_mask = pd.Series(
            np.isfinite(numeric.to_numpy(dtype=float)),
            index=corrected_signal.index)
        runs_all_finite = 0
        if (
                column_present and "semantic_id" in corrected_signal.columns
                and not corrected_signal.empty):
            eligible = corrected_signal[eligible_mask].assign(
                _explicit_field_finite=finite_mask[eligible_mask])
            per_run = eligible.groupby(
                    "semantic_id", dropna=False)["_explicit_field_finite"].agg(
                        ["size", "all"])
            runs_all_finite = int(
                ((per_run["size"] > 0) & per_run["all"]).sum())
        missing_on_eligible = int(
            (eligible_mask & ~finite_mask).sum())
        field_audit[field] = {
            "column_present_before_fallback": bool(column_present),
            "finite_rows": int(finite_mask.sum()),
            "total_rows": int(len(corrected_signal)),
            "eligible_rows_with_native_U_and_G": int(
                eligible_mask.sum()),
            "missing_on_eligible_rows": missing_on_eligible,
            "canonical_runs_all_eligible_rows_finite": runs_all_finite,
            "expected_canonical_runs": expected_runs,
            "pass": bool(
                column_present and int(eligible_mask.sum()) > 0 and
                missing_on_eligible == 0 and
                runs_all_finite == expected_runs),
        }
    observed_signal_ids = set(
        corrected_signal.get(
            "semantic_id", pd.Series(dtype=str)
        ).dropna().astype(str))
    run_coverage_ok = bool(
        len(corrected_inventory) == expected_runs and
        len(corrected_ids) == expected_runs and
        observed_signal_ids == corrected_ids)
    status = bool(
        run_coverage_ok and field_audit and
        all(item["pass"] for item in field_audit.values()))
    return {
        "status": "PASS" if status else "FAIL",
        "audited_before_backward_compatibility_fallback": True,
        "experiment_id": "V2_PROBABILITY_CORRECTED",
        "expected_canonical_runs": expected_runs,
        "inventory_canonical_runs": int(len(corrected_ids)),
        "signal_canonical_runs": int(len(observed_signal_ids)),
        "signal_rows": int(len(corrected_signal)),
        "run_identity_coverage_pass": run_coverage_ok,
        "required_native_fields": list(required_fields),
        "fields": field_audit,
    }


def experiment_family(experiment_id: Any) -> str:
    value = str(experiment_id or "")
    mapping = {
        "V2_PROBABILITY_CORRECTED": "corrected_probability",
        "V2_TEMPERATURE_SENSITIVITY": "temperature",
        "V2_CONSTANT_ACCEPTANCE": "constant_acceptance",
        "V2_GATE_TRANSFORM": "gate_transform",
        "V2_FIXED_POLICY_REPLAY": "fixed_policy_replay",
        "V2_GATE_DOSE_RESPONSE": "gate_dose_response",
        "V2_GATE_MEAN_MATCHED": "mean_matched",
        "V2_PRODUCT_RETRAIN": "product_retrain",
        "V2_PRODUCT_TRAINED_EVAL_CLEAN": "product_trained_eval",
        "V2_PRODUCT_TRAINED_EVAL_CORRUPTION": "product_trained_eval",
        "P0_AB_CLEAN_MECHANISM": "p0_clean",
        "P0_D_FIXED_CKPT_INTERVENTIONS": "p0_fixed",
        "P0_D_PRODUCT_FIXED_CLEAN": "p0_product_fixed",
        "P0_D_PRODUCT_FIXED_CORRUPTION": "p0_product_fixed",
        "P0_D_TRAINED_EVAL_CLEAN": "trained_variant",
        "P0_D_TRAINED_EVAL_CORRUPTION": "trained_variant",
        "P0_B_PROBABILITY_SIGNALS": "legacy_probability",
        "P0_C1_CHANNEL_DROPOUT": "p0_corruption",
        "P0_C1_GAUSSIAN": "p0_corruption",
        "P0_C2_TRANSLATION": "p0_corruption",
        "P0_C3_DONOR_MIX": "p0_corruption",
    }
    return mapping.get(value, value.lower() or "unknown")


def normalize_case_columns(case: pd.DataFrame) -> pd.DataFrame:
    if case.empty:
        return case
    result = case.copy()
    result["family"] = result["experiment_id"].map(experiment_family)
    aliases = {
        "mean_T_formula": ("mean_T_formula",),
        "mean_A_impl": ("mean_A_impl", "mean_T"),
        "mean_A_impl_raw": ("mean_A_impl_raw", "mean_A_impl", "mean_T"),
        "mean_FDD_paper": ("mean_FDD_paper", "mean_FDD"),
        "mean_FDD_legacy": ("mean_FDD_legacy", "mean_FDD"),
    }
    for target, candidates in aliases.items():
        if target not in result:
            result[target] = np.nan
        for source in candidates:
            if source in result:
                result[target] = result[target].where(
                    result[target].notna(), result[source])
    if {"mean_G", "mean_U"}.issubset(result.columns):
        derived = (
            pd.to_numeric(result["mean_G"], errors="coerce") *
            (1.0 - pd.to_numeric(result["mean_U"], errors="coerce")))
        result["mean_T_formula"] = result["mean_T_formula"].where(
            result["mean_T_formula"].notna(), derived)
    for column in (
            *CASE_METRICS, "mean_U", "mean_G", "mean_T_formula",
            "mean_A_impl", "mean_A_impl_raw", "mean_FDD_paper",
            "mean_FDD_legacy",
            "mean_confidence", "mean_entropy",
            "probability_sum_max_abs_error", "probability_min",
            "probability_max", "A_transform_invariant_residual_max"):
        if column in result:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    for column, default in (
            ("corruption_type", "none"), ("severity", 0.0),
            ("realization", 0), ("seed", 0), ("gate_scale", 1.0),
            ("temperature_scale", 1.0), ("gate_transform", "identity")):
        if column not in result:
            result[column] = default
        result[column] = result[column].fillna(default)
    result["cluster_id"] = (
        result["seed"].astype(str) + "|" + result["case_id"].astype(str))
    return result


def normalize_signal_columns(signal: pd.DataFrame) -> pd.DataFrame:
    if signal.empty:
        return signal
    result = signal.copy()
    result["family"] = result["experiment_id"].map(experiment_family)
    result["T_formula_observed"] = (
        result["T_formula"].notna()
        if "T_formula" in result.columns
        else False)
    if "T_formula" not in result:
        result["T_formula"] = np.nan
    if {"G", "U"}.issubset(result.columns):
        derived = (
            pd.to_numeric(result["G"], errors="coerce") *
            (1.0 - pd.to_numeric(result["U"], errors="coerce")))
        result["T_formula"] = result["T_formula"].where(
            result["T_formula"].notna(), derived)
    # The explicit raw field was added with the V2 instrumentation.  Legacy
    # runs only have ``T`` (the executed variant-specific acceptance), so they
    # cannot support the submitted-full-policy algebra invariant.
    result["A_impl_raw_observed"] = (
        result["A_impl_raw"].notna()
        if "A_impl_raw" in result.columns
        else False)
    if "A_impl" not in result:
        result["A_impl"] = result["T"] if "T" in result else np.nan
    if "A_impl_raw" not in result:
        result["A_impl_raw"] = result["A_impl"]
    result["FDD_paper_observed"] = (
        result["FDD_paper"].notna()
        if "FDD_paper" in result.columns
        else False)
    result["FDD_legacy_observed"] = (
        result["FDD_legacy"].notna()
        if "FDD_legacy" in result.columns
        else False)
    if "FDD_paper" not in result:
        result["FDD_paper"] = result["FDD"] if "FDD" in result else np.nan
    if "FDD_legacy" not in result:
        result["FDD_legacy"] = result["FDD"] if "FDD" in result else np.nan
    for column in (
            "U", "G", "T_formula", "A_impl", "A_impl_raw", "FDD_paper",
            "FDD_legacy",
            "confidence", "entropy", "Q_vox", "Q_component", "Q_centerline",
            "B_vox", "B_component", "B_centerline"):
        if column in result:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    for column, default in (
            ("feature_level", "unknown"), ("seed", 0),
            ("realization", 0), ("severity", 0.0),
            ("corruption_type", "none")):
        if column not in result:
            result[column] = default
        result[column] = result[column].fillna(default)
    result["cluster_id"] = (
        result["seed"].astype(str) + "|" + result["case_id"].astype(str))
    return result


def bootstrap_mean(values: np.ndarray, clusters: np.ndarray, n_boot: int,
                   rng: np.random.Generator) -> Tuple[float, float]:
    keep = np.isfinite(values)
    values = values[keep]
    clusters = clusters[keep]
    if not len(values):
        return np.nan, np.nan
    unique = np.unique(clusters)
    sums = np.asarray([values[clusters == key].sum() for key in unique])
    counts = np.asarray([(clusters == key).sum() for key in unique])
    draws = rng.integers(0, len(unique), size=(n_boot, len(unique)))
    estimates = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    return tuple(np.quantile(estimates, [0.025, 0.975]).tolist())


def aggregate_long(frame: pd.DataFrame, family_name: str, label_column: str,
                   group_columns: Sequence[str], reference_label: Any,
                   n_boot: int, rng: np.random.Generator) -> pd.DataFrame:
    columns = [
        "analysis_family", *group_columns, "label", "reference_label",
        "metric", "higher_is_better", "n_rows", "n_cases", "n_seeds",
        "mean", "sd", "ci_low", "ci_high", "paired_n",
        "paired_difference", "paired_ci_low", "paired_ci_high",
    ]
    if frame.empty or label_column not in frame:
        return pd.DataFrame(columns=columns)
    use = frame.copy()
    use["label"] = use[label_column].map(scalar_text)
    reference_text = scalar_text(reference_label)
    rows: List[Dict[str, Any]] = []
    grouping = list(group_columns) + ["label"]
    for keys, group in use.groupby(grouping, dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        metadata = dict(zip(grouping, keys))
        condition = {
            key: metadata[key] for key in group_columns}
        reference = use.copy()
        for key, value in condition.items():
            if pd.isna(value):
                reference = reference[reference[key].isna()]
            else:
                reference = reference[reference[key] == value]
        reference = reference[reference["label"] == reference_text]
        join_keys = [
            key for key in (
                "dataset", "case_id", "seed", "corruption_type", "severity",
                "realization")
            if key in group.columns and key in reference.columns]
        for metric in CASE_METRICS:
            if metric not in group:
                continue
            values = pd.to_numeric(group[metric], errors="coerce")
            valid = values.notna()
            point_values = values[valid].to_numpy(dtype=float)
            point_clusters = group.loc[
                valid, "cluster_id"].to_numpy(dtype=str)
            ci_low, ci_high = bootstrap_mean(
                point_values, point_clusters, n_boot, rng)
            row = {
                "analysis_family": family_name,
                **condition,
                "label": metadata["label"],
                "reference_label": reference_text,
                "metric": metric,
                "higher_is_better": HIGHER_IS_BETTER[metric],
                "n_rows": int(valid.sum()),
                "n_cases": int(group.loc[valid, "case_id"].nunique()),
                "n_seeds": int(group.loc[valid, "seed"].nunique()),
                "mean": (
                    float(point_values.mean()) if len(point_values) else np.nan),
                "sd": (
                    float(point_values.std(ddof=1))
                    if len(point_values) > 1 else np.nan),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "paired_n": 0,
                "paired_difference": np.nan,
                "paired_ci_low": np.nan,
                "paired_ci_high": np.nan,
            }
            if (
                    metadata["label"] != reference_text and
                    not reference.empty and join_keys):
                left = group[join_keys + [metric]].copy()
                right = reference[join_keys + [metric]].copy()
                left = left.groupby(join_keys, dropna=False)[metric].mean()
                right = right.groupby(join_keys, dropna=False)[metric].mean()
                paired = pd.concat(
                    [left.rename("target"), right.rename("reference")],
                    axis=1, join="inner").dropna()
                if not paired.empty:
                    paired = paired.reset_index()
                    paired["difference"] = (
                        paired["target"] - paired["reference"])
                    paired["cluster_id"] = (
                        paired["seed"].astype(str) + "|" +
                        paired["case_id"].astype(str))
                    diff = paired["difference"].to_numpy(dtype=float)
                    clusters = paired["cluster_id"].to_numpy(dtype=str)
                    low, high = bootstrap_mean(diff, clusters, n_boot, rng)
                    row.update({
                        "paired_n": int(len(paired)),
                        "paired_difference": float(diff.mean()),
                        "paired_ci_low": low,
                        "paired_ci_high": high,
                    })
            rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def make_practical_tables(case: pd.DataFrame, n_boot: int,
                          rng: np.random.Generator) -> Dict[str, pd.DataFrame]:
    common = ["dataset", "corruption_type", "severity"]
    tables: Dict[str, pd.DataFrame] = {}
    temp = case[case["family"] == "temperature"].copy()
    tables["temperature"] = aggregate_long(
        temp, "temperature", "temperature_scale", common, 1.0, n_boot, rng)

    constant = case[case["family"] == "constant_acceptance"].copy()
    constant_table = aggregate_long(
        constant, "constant_acceptance", "gate_alpha", common, 1.0,
        n_boot, rng)
    dose = case[case["family"] == "gate_dose_response"].copy()
    dose_table = aggregate_long(
        dose, "alpha_times_policy", "gate_scale", common, 1.0, n_boot, rng)
    tables["dose"] = pd.concat(
        [constant_table, dose_table], ignore_index=True, sort=False)

    replay_parts: List[pd.DataFrame] = []
    for family in ("p0_fixed", "fixed_policy_replay"):
        part = case[case["family"] == family].copy()
        replay_parts.append(aggregate_long(
            part, family, "model_variant", common, "full", n_boot, rng))

    transforms = case[
        case["family"].isin(["gate_transform", "mean_matched"])].copy()
    transforms["policy_label"] = transforms["gate_transform"].astype(str)
    identity = case[
        (case["family"] == "fixed_policy_replay") &
        (case["model_variant"] == "full")].copy()
    if identity.empty:
        identity = case[
            (case["model_variant"] == "full") &
            (case["gate_transform"] == "identity") &
            (case["gate_scale"] == 1.0) &
            (case["temperature_scale"] == 1.0) &
            (case["gate_alpha"].isna())].copy()
    identity["policy_label"] = "identity"
    transform_pool = pd.concat(
        [transforms, identity], ignore_index=True, sort=False)
    replay_parts.append(aggregate_long(
        transform_pool, "gate_transform", "policy_label", common,
        "identity", n_boot, rng))

    trained = case[
        case["family"].isin(["trained_variant", "product_trained_eval"])
    ].copy()
    replay_parts.append(aggregate_long(
        trained, "trained_variant", "model_variant", common, "full",
        n_boot, rng))
    tables["formula"] = pd.concat(
        replay_parts, ignore_index=True, sort=False)
    tables["practical"] = pd.concat(
        [tables["temperature"], tables["dose"], tables["formula"]],
        ignore_index=True, sort=False)
    return tables


def average_precision(y: np.ndarray, score: np.ndarray) -> float:
    positives = int(y.sum())
    if positives == 0:
        return np.nan
    order = np.argsort(-score, kind="mergesort")
    ranked = y[order]
    precision = np.cumsum(ranked) / np.arange(1, len(ranked) + 1)
    return float((precision * ranked).sum() / positives)


def auroc(y: np.ndarray, score: np.ndarray) -> float:
    positives = y == 1
    negatives = y == 0
    n_pos, n_neg = int(positives.sum()), int(negatives.sum())
    if not n_pos or not n_neg:
        return np.nan
    ranks = pd.Series(score).rank(method="average").to_numpy(dtype=float)
    value = (
        ranks[positives].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(value)


def aurc(y: np.ndarray, score: np.ndarray) -> float:
    if not len(y):
        return np.nan
    order = np.argsort(-score, kind="mergesort")
    risk = 1.0 - y[order].astype(float)
    cumulative = np.cumsum(risk) / np.arange(1, len(risk) + 1)
    return float(cumulative.mean())


def binary_metrics(y: np.ndarray, score: np.ndarray) -> Dict[str, float]:
    keep = np.isfinite(y) & np.isfinite(score)
    y = y[keep].astype(int)
    score = np.clip(score[keep].astype(float), 0.0, 1.0)
    if not len(y):
        return {
            "AUPRC": np.nan, "AUROC": np.nan, "Brier": np.nan,
            "NLL": np.nan, "AURC": np.nan, "prevalence": np.nan}
    epsilon = 1e-6
    clipped = np.clip(score, epsilon, 1.0 - epsilon)
    return {
        "AUPRC": average_precision(y, score),
        "AUROC": auroc(y, score),
        "Brier": float(np.mean((score - y) ** 2)),
        "NLL": float(np.mean(
            -(y * np.log(clipped) + (1.0 - y) * np.log(1.0 - clipped)))),
        "AURC": aurc(y, score),
        "prevalence": float(y.mean()),
    }


def bootstrap_binary(y: np.ndarray, score: np.ndarray, clusters: np.ndarray,
                     metric: str, n_boot: int,
                     rng: np.random.Generator) -> Tuple[float, float]:
    unique = np.unique(clusters)
    if len(unique) < 2:
        return np.nan, np.nan
    by_cluster = {
        key: np.flatnonzero(clusters == key) for key in unique}
    values = []
    for _ in range(n_boot):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([by_cluster[key] for key in sampled])
        value = binary_metrics(y[indices], score[indices])[metric]
        if math.isfinite(value):
            values.append(value)
    if not values:
        return np.nan, np.nan
    return tuple(np.quantile(values, [0.025, 0.975]).tolist())


def select_signal_source(signal: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    priorities = (
        ("corrected_probability",),
        ("legacy_probability",),
        ("p0_clean",),
        ("fixed_policy_replay",),
    )
    for families in priorities:
        chosen = signal[signal["family"].isin(families)].copy()
        if not chosen.empty:
            return chosen, families[0]
    return signal.copy(), "all_available"


def collapse_signal_units(signal: pd.DataFrame) -> Tuple[
        pd.DataFrame, Dict[str, Any]]:
    if signal.empty:
        return pd.DataFrame(), {"raw_region_rows": 0, "case_level_units": 0}
    chosen, source = select_signal_source(signal)
    keys = [
        key for key in (
            "semantic_id", "experiment_id", "family", "dataset", "case_id",
            "seed", "corruption_type", "severity", "realization",
            "feature_level")
        if key in chosen]
    numeric = [
        key for key in (
            "U", "G", "T_formula", "A_impl", "A_impl_raw", "FDD_paper",
            "FDD_legacy",
            "confidence", "entropy", "Q_vox", "Q_component", "Q_centerline")
        if key in chosen]
    units = chosen.groupby(keys, dropna=False)[numeric].mean().reset_index()
    if "Q_component" in units:
        units["target_component"] = (
            pd.to_numeric(units["Q_component"], errors="coerce") > 0).astype(int)
    if "Q_vox" in units:
        units["target_vox"] = (
            pd.to_numeric(units["Q_vox"], errors="coerce") > 0).astype(int)
    units["cluster_id"] = (
        units["seed"].astype(str) + "|" + units["case_id"].astype(str))
    audit = {
        "source_family": source,
        "raw_region_rows": int(len(chosen)),
        "case_level_units": int(len(units)),
        "unique_seed_cases": int(units["cluster_id"].nunique()),
        "analysis_unit": "one mean observation per run/seed/case/feature_level",
    }
    return units, audit


def reliability_scores(units: pd.DataFrame) -> Dict[str, Tuple[pd.Series, str]]:
    scores: Dict[str, Tuple[pd.Series, str]] = {}
    if "U" in units:
        scores["U"] = (1.0 - units["U"], "1-U (higher=more reliable)")
    if "G" in units:
        scores["G"] = (units["G"], "identity")
    if "T_formula" in units:
        scores["T"] = (units["T_formula"], "T_formula=G(1-U)")
    if "A_impl" in units:
        scores["A"] = (units["A_impl"], "executed A_impl")
    if "FDD_paper" in units:
        scores["FDDpaper"] = (
            1.0 - units["FDD_paper"].clip(0, 1),
            "1-clip(FDD_paper,0,1)")
    if "FDD_legacy" in units:
        scores["FDDlegacy"] = (
            1.0 - units["FDD_legacy"].clip(0, 2) / 2.0,
            "1-clip(FDD_legacy,0,2)/2")
    if "confidence" in units:
        scores["confidence"] = (units["confidence"], "identity")
    if "entropy" in units:
        scores["1-entropy"] = (1.0 - units["entropy"], "1-entropy")
    return scores


def signal_validity(units: pd.DataFrame, n_boot: int,
                    rng: np.random.Generator) -> pd.DataFrame:
    columns = [
        "dataset", "corruption_type", "severity", "target", "signal",
        "score_transform", "metric", "value", "ci_low", "ci_high",
        "prevalence", "n_units", "n_cases", "n_seeds",
        "analysis_unit", "bootstrap_cluster",
    ]
    if units.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    grouping = [
        key for key in ("dataset", "corruption_type", "severity")
        if key in units]
    targets = [
        key for key in ("target_component", "target_vox") if key in units]
    for keys, group in units.groupby(grouping, dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        metadata = dict(zip(grouping, keys))
        for target in targets:
            y_all = pd.to_numeric(group[target], errors="coerce")
            for name, (series, transform) in reliability_scores(group).items():
                score_all = pd.to_numeric(series, errors="coerce")
                keep = y_all.notna() & score_all.notna()
                y = y_all[keep].to_numpy(dtype=float)
                score = score_all[keep].to_numpy(dtype=float)
                clusters = group.loc[
                    keep, "cluster_id"].to_numpy(dtype=str)
                point = binary_metrics(y, score)
                for metric in ("AUPRC", "AUROC", "Brier", "AURC"):
                    low, high = bootstrap_binary(
                        y, score, clusters, metric, n_boot, rng)
                    rows.append({
                        **metadata,
                        "target": target,
                        "signal": name,
                        "score_transform": transform,
                        "metric": metric,
                        "value": point[metric],
                        "ci_low": low,
                        "ci_high": high,
                        "prevalence": point["prevalence"],
                        "n_units": int(len(y)),
                        "n_cases": int(group.loc[keep, "case_id"].nunique()),
                        "n_seeds": int(group.loc[keep, "seed"].nunique()),
                        "analysis_unit": (
                            "run/seed/case/feature_level mean; no region or "
                            "voxel treated as an independent U sample"),
                        "bootstrap_cluster": "seed/case_id",
                    })
    return pd.DataFrame(rows, columns=columns)


def calibration_diagnostics(
        y: np.ndarray, score: np.ndarray) -> Tuple[float, float, str]:
    keep = np.isfinite(y) & np.isfinite(score)
    y = y[keep].astype(float)
    score = score[keep].astype(float)
    if len(y) < 4:
        return np.nan, np.nan, "inconclusive_too_few_units"
    if len(np.unique(y)) < 2:
        return np.nan, np.nan, "inconclusive_single_class"
    logit = np.log(
        np.clip(score, 1e-6, 1.0 - 1e-6) /
        np.clip(1.0 - score, 1e-6, 1.0))
    if not np.isfinite(logit).all() or float(np.std(logit)) < 1e-8:
        return np.nan, np.nan, "inconclusive_constant_score"
    beta = logistic_fit(logit.reshape(-1, 1), y, ridge=1e-6)
    if len(beta) != 2 or not np.isfinite(beta).all():
        return np.nan, np.nan, "inconclusive_fit_failure"
    return float(beta[0]), float(beta[1]), "estimable_diagnostic_only"


def action_metrics(
        y: np.ndarray, score: np.ndarray,
        threshold: Optional[float]) -> Dict[str, float]:
    keep = np.isfinite(y) & np.isfinite(score)
    y = y[keep].astype(int)
    score = score[keep].astype(float)
    if not len(y) or threshold is None or not math.isfinite(threshold):
        return {"HAR": np.nan, "BRR": np.nan, "coverage": np.nan}
    accepted = score >= threshold
    harmful = y == 0
    beneficial = y == 1
    return {
        "HAR": (
            float(accepted[harmful].mean()) if harmful.any() else np.nan),
        "BRR": (
            float((~accepted[beneficial]).mean())
            if beneficial.any() else np.nan),
        "coverage": float(accepted.mean()),
    }


def choose_source_threshold(
        y: np.ndarray, score: np.ndarray) -> Tuple[Optional[float], str]:
    keep = np.isfinite(y) & np.isfinite(score)
    y = y[keep].astype(int)
    score = np.clip(score[keep].astype(float), 0.0, 1.0)
    if len(y) < 4 or len(np.unique(y)) < 2:
        return None, "inconclusive_source_classes"
    candidates = np.unique(np.concatenate([
        np.array([0.0, 0.5, 1.0]), score]))
    choices = []
    for threshold in candidates:
        rates = action_metrics(y, score, float(threshold))
        if not all(math.isfinite(rates[key]) for key in ("HAR", "BRR")):
            continue
        choices.append((
            0.5 * (rates["HAR"] + rates["BRR"]),
            rates["HAR"] + rates["BRR"],
            -rates["coverage"],
            float(threshold),
        ))
    if not choices:
        return None, "inconclusive_threshold_fit"
    return min(choices)[3], "TopCoW_val_clean_frozen"


def hierarchical_bootstrap_values(
        frame: pd.DataFrame, statistic: Any, n_boot: int,
        rng: np.random.Generator) -> List[float]:
    if frame.empty or "case_id" not in frame or "seed" not in frame:
        return []
    seed_values = frame["seed"].dropna().unique()
    if not len(seed_values):
        return []
    values: List[float] = []
    for _ in range(n_boot):
        sampled_seeds = (
            rng.choice(seed_values, size=len(seed_values), replace=True)
            if len(seed_values) > 1 else seed_values)
        blocks = []
        for sampled_seed in sampled_seeds:
            seed_frame = frame[frame["seed"] == sampled_seed]
            case_values = seed_frame["case_id"].dropna().unique()
            if not len(case_values):
                continue
            sampled_cases = rng.choice(
                case_values, size=len(case_values), replace=True)
            blocks.extend(
                seed_frame[seed_frame["case_id"] == case_value]
                for case_value in sampled_cases)
        if not blocks:
            continue
        value = statistic(pd.concat(blocks, ignore_index=True))
        if value is not None and math.isfinite(float(value)):
            values.append(float(value))
    return values


def formal_reliability_endpoints(
        units: pd.DataFrame, n_boot: int,
        rng: np.random.Generator) -> pd.DataFrame:
    columns = [
        "dataset", "corruption_type", "severity", "target", "signal",
        "score_transform", "metric", "value", "ci_low", "ci_high",
        "prevalence_beneficial", "delta_AUPRC_over_prevalence",
        "threshold", "threshold_source", "endpoint_status",
        "eligible_for_calibration_claim", "n_units", "n_cases", "n_seeds",
        "analysis_unit", "bootstrap_protocol",
    ]
    if units.empty:
        return pd.DataFrame(columns=columns)
    targets = [
        name for name in ("target_component", "target_vox")
        if name in units]
    if not targets:
        return pd.DataFrame(columns=columns)

    score_catalogue = reliability_scores(units)
    thresholds: Dict[Tuple[str, str], Tuple[Optional[float], str]] = {}
    clean_types = {"none", "clean", "identity"}
    source_mask = (
        units["dataset"].astype(str).eq("TopCoW_val") &
        units["corruption_type"].astype(str).str.lower().isin(clean_types) &
        (pd.to_numeric(units["severity"], errors="coerce").fillna(0.0) == 0))
    for target in targets:
        y_source_all = pd.to_numeric(units[target], errors="coerce")
        for signal_name, (score_series, _) in score_catalogue.items():
            if signal_name == "A":
                thresholds[(target, signal_name)] = (
                    0.5, "predefined_executed_acceptance_0.5")
                continue
            source_keep = (
                source_mask & y_source_all.notna() &
                pd.to_numeric(score_series, errors="coerce").notna())
            thresholds[(target, signal_name)] = choose_source_threshold(
                y_source_all[source_keep].to_numpy(dtype=float),
                pd.to_numeric(
                    score_series[source_keep], errors="coerce").to_numpy(
                        dtype=float))

    grouping = [
        key for key in ("dataset", "corruption_type", "severity")
        if key in units]
    rows: List[Dict[str, Any]] = []
    for group_keys, group in units.groupby(
            grouping, dropna=False, sort=True):
        if not isinstance(group_keys, tuple):
            group_keys = (group_keys,)
        metadata = dict(zip(grouping, group_keys))
        for target in targets:
            y_all = pd.to_numeric(group[target], errors="coerce")
            for signal_name, (series, transform) in reliability_scores(
                    group).items():
                score_all = pd.to_numeric(series, errors="coerce")
                keep = y_all.notna() & score_all.notna()
                data = group.loc[keep].copy()
                if data.empty:
                    continue
                data["_endpoint_y"] = y_all[keep].to_numpy(dtype=float)
                data["_endpoint_score"] = score_all[keep].to_numpy(dtype=float)
                y = data["_endpoint_y"].to_numpy(dtype=float)
                score = np.clip(
                    data["_endpoint_score"].to_numpy(dtype=float), 0.0, 1.0)
                threshold, threshold_source = thresholds.get(
                    (target, signal_name),
                    (None, "inconclusive_source_missing"))
                basic = binary_metrics(y, score)
                action = action_metrics(y, score, threshold)
                intercept, slope, calibration_status = (
                    calibration_diagnostics(y, score))
                endpoints = {
                    "AUPRC_beneficial": basic["AUPRC"],
                    "AUROC": basic["AUROC"],
                    "AURC": basic["AURC"],
                    "Brier": basic["Brier"],
                    "NLL": basic["NLL"],
                    "calibration_intercept": intercept,
                    "calibration_slope": slope,
                    "HAR": action["HAR"],
                    "BRR": action["BRR"],
                    "coverage": action["coverage"],
                }

                def statistic(metric_name: str, sampled: pd.DataFrame) -> float:
                    sampled_y = sampled["_endpoint_y"].to_numpy(dtype=float)
                    sampled_score = np.clip(
                        sampled["_endpoint_score"].to_numpy(dtype=float),
                        0.0, 1.0)
                    if metric_name == "calibration_intercept":
                        return calibration_diagnostics(
                            sampled_y, sampled_score)[0]
                    if metric_name == "calibration_slope":
                        return calibration_diagnostics(
                            sampled_y, sampled_score)[1]
                    if metric_name in {"HAR", "BRR", "coverage"}:
                        return action_metrics(
                            sampled_y, sampled_score, threshold)[metric_name]
                    lookup = (
                        "AUPRC" if metric_name == "AUPRC_beneficial"
                        else metric_name)
                    return binary_metrics(
                        sampled_y, sampled_score)[lookup]

                for metric_name, value in endpoints.items():
                    boot = hierarchical_bootstrap_values(
                        data,
                        lambda sampled, name=metric_name: statistic(
                            name, sampled),
                        n_boot, rng)
                    ci_low = ci_high = np.nan
                    if boot:
                        ci_low, ci_high = np.quantile(
                            boot, [0.025, 0.975]).tolist()
                    endpoint_status = "estimable"
                    if metric_name.startswith("calibration_"):
                        endpoint_status = calibration_status
                    elif metric_name in {"HAR", "BRR", "coverage"} and (
                            threshold is None):
                        endpoint_status = threshold_source
                    elif not math.isfinite(float(value)):
                        endpoint_status = "inconclusive_not_estimable"
                    rows.append({
                        **metadata,
                        "target": target,
                        "signal": signal_name,
                        "score_transform": transform,
                        "metric": metric_name,
                        "value": value,
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                        "prevalence_beneficial": basic["prevalence"],
                        "delta_AUPRC_over_prevalence": (
                            basic["AUPRC"] - basic["prevalence"]
                            if math.isfinite(basic["AUPRC"]) and
                            math.isfinite(basic["prevalence"]) else np.nan),
                        "threshold": threshold,
                        "threshold_source": threshold_source,
                        "endpoint_status": endpoint_status,
                        "eligible_for_calibration_claim": bool(
                            endpoint_status in {
                                "estimable", "estimable_diagnostic_only"}),
                        "n_units": int(len(data)),
                        "n_cases": int(data["case_id"].nunique()),
                        "n_seeds": int(data["seed"].nunique()),
                        "analysis_unit": (
                            "one run/seed/case/feature_level mean; all levels "
                            "remain inside the resampled case block"),
                        "bootstrap_protocol": (
                            "outer model-seed / inner case bootstrap"
                            if data["seed"].nunique() > 1 else
                            "single-seed case bootstrap"),
                    })
    return pd.DataFrame(rows, columns=columns)


def sigmoid(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-value))


def logistic_fit(x: np.ndarray, y: np.ndarray,
                 ridge: float = 1e-3) -> np.ndarray:
    design = np.column_stack([np.ones(len(x)), x])
    beta = np.zeros(design.shape[1], dtype=float)
    penalty = np.eye(design.shape[1]) * ridge
    penalty[0, 0] = 0.0
    for _ in range(80):
        probability = sigmoid(design @ beta)
        weight = np.clip(probability * (1.0 - probability), 1e-6, None)
        gradient = design.T @ (probability - y) + penalty @ beta
        hessian = (design.T * weight) @ design + penalty
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(hessian) @ gradient
        beta -= step
        if np.max(np.abs(step)) < 1e-8:
            break
    return beta


def prepare_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if "U" in result:
        result["feat_U"] = 1.0 - result["U"]
        result["feat_U_raw"] = result["U"]
    if "G" in result:
        result["feat_G"] = result["G"]
    if {"G", "U"}.issubset(result.columns):
        result["feat_GxU"] = result["G"] * result["U"]
    if "T_formula" in result:
        result["feat_T"] = result["T_formula"]
    if "A_impl" in result:
        result["feat_A"] = result["A_impl"]
    if "FDD_paper" in result:
        result["feat_FDD"] = 1.0 - result["FDD_paper"].clip(0, 1)
    if "confidence" in result:
        result["feat_confidence"] = result["confidence"]
    if "entropy" in result:
        result["feat_entropy"] = 1.0 - result["entropy"]
    return result


def grouped_folds(groups: np.ndarray, n_splits: int,
                  rng: np.random.Generator) -> List[np.ndarray]:
    unique = np.unique(groups)
    shuffled = unique.copy()
    rng.shuffle(shuffled)
    buckets = np.array_split(shuffled, min(n_splits, len(shuffled)))
    return [np.isin(groups, bucket) for bucket in buckets if len(bucket)]


def standardize_fit(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    median = np.nanmedian(x, axis=0)
    filled = np.where(np.isfinite(x), x, median)
    mean = filled.mean(axis=0)
    scale = filled.std(axis=0)
    scale[scale < 1e-8] = 1.0
    return mean, scale


def transform_features(x: np.ndarray, mean: np.ndarray,
                       scale: np.ndarray) -> np.ndarray:
    filled = np.where(np.isfinite(x), x, mean)
    return (filled - mean) / scale


def incremental_value(units: pd.DataFrame, n_boot: int,
                      rng: np.random.Generator,
                      failures: List[Dict[str, Any]]) -> pd.DataFrame:
    columns = [
        "evaluation", "dataset", "target", "model", "features", "metric",
        "value", "ci_low", "ci_high", "reference_model",
        "increment_vs_G", "increment_ci_low", "increment_ci_high",
        "n_units", "n_cases", "n_seeds", "fit_protocol",
    ]
    if units.empty or "target_component" not in units:
        return pd.DataFrame(columns=columns)
    frame = prepare_features(units)
    feature_sets = {
        "G": ["feat_G"],
        "U": ["feat_U"],
        "T": ["feat_T"],
        "A": ["feat_A"],
        "G+U": ["feat_G", "feat_U"],
        "G+U+G*U": ["feat_G", "feat_U_raw", "feat_GxU"],
        "G+U+FDDpaper": ["feat_G", "feat_U", "feat_FDD"],
        "all_low_cost": [
            "feat_G", "feat_U", "feat_FDD", "feat_confidence",
            "feat_entropy"],
    }
    feature_sets = {
        name: columns_ for name, columns_ in feature_sets.items()
        if all(column in frame for column in columns_)}
    source = frame[frame["dataset"] == "TopCoW_val"].copy()
    if source["case_id"].nunique() < 5 or source["target_component"].nunique() < 2:
        failures.append({
            "severity": "warning",
            "category": "incremental_value",
            "finding": "insufficient TopCoW cases/classes for grouped fitting",
            "evidence": (
                f"cases={source['case_id'].nunique()}, "
                f"classes={source['target_component'].nunique()}"),
        })
        return pd.DataFrame(columns=columns)
    predictions: Dict[Tuple[str, str], pd.DataFrame] = {}
    y_source = source["target_component"].to_numpy(dtype=float)
    groups = source["cluster_id"].to_numpy(dtype=str)
    folds = grouped_folds(groups, 5, rng)
    for model_name, features in feature_sets.items():
        x_source = source[features].to_numpy(dtype=float)
        oof = np.full(len(source), np.nan)
        for test_mask in folds:
            train_mask = ~test_mask
            if (
                    train_mask.sum() < 3 or test_mask.sum() == 0 or
                    len(np.unique(y_source[train_mask])) < 2):
                continue
            mean, scale = standardize_fit(x_source[train_mask])
            train_x = transform_features(
                x_source[train_mask], mean, scale)
            test_x = transform_features(x_source[test_mask], mean, scale)
            beta = logistic_fit(train_x, y_source[train_mask])
            oof[test_mask] = sigmoid(
                np.column_stack([np.ones(test_x.shape[0]), test_x]) @ beta)
        predictions[(model_name, "TopCoW_val_OOF")] = pd.DataFrame({
            "y": y_source,
            "score": oof,
            "cluster_id": groups,
            "case_id": source["case_id"].to_numpy(),
            "seed": source["seed"].to_numpy(),
        })
        mean, scale = standardize_fit(x_source)
        fitted_x = transform_features(x_source, mean, scale)
        beta = logistic_fit(fitted_x, y_source)
        for dataset in sorted(
                value for value in frame["dataset"].dropna().unique()
                if value != "TopCoW_val"):
            external = frame[frame["dataset"] == dataset].copy()
            x_external = transform_features(
                external[features].to_numpy(dtype=float), mean, scale)
            score = sigmoid(
                np.column_stack([
                    np.ones(x_external.shape[0]), x_external]) @ beta)
            predictions[(model_name, str(dataset))] = pd.DataFrame({
                "y": external["target_component"].to_numpy(dtype=float),
                "score": score,
                "cluster_id": external["cluster_id"].to_numpy(dtype=str),
                "case_id": external["case_id"].to_numpy(),
                "seed": external["seed"].to_numpy(),
            })
    rows = []
    for (model_name, evaluation), prediction in predictions.items():
        reference = predictions.get(("G", evaluation))
        keep = prediction[["y", "score"]].notna().all(axis=1)
        data = prediction[keep]
        if data.empty:
            continue
        y = data["y"].to_numpy(dtype=float)
        score = data["score"].to_numpy(dtype=float)
        clusters = data["cluster_id"].to_numpy(dtype=str)
        point = binary_metrics(y, score)
        for metric in ("AUPRC", "AUROC", "Brier", "NLL", "AURC"):
            low, high = bootstrap_binary(
                y, score, clusters, metric, n_boot, rng)
            increment = increment_low = increment_high = np.nan
            if model_name != "G" and reference is not None:
                merged = (
                    prediction.reset_index().merge(
                        reference.reset_index(), on="index",
                        suffixes=("_target", "_reference"))
                    .dropna(subset=[
                        "y_target", "score_target", "score_reference"]))
                if not merged.empty:
                    y_pair = merged["y_target"].to_numpy(dtype=float)
                    target_score = merged["score_target"].to_numpy(dtype=float)
                    reference_score = merged[
                        "score_reference"].to_numpy(dtype=float)
                    pair_clusters = merged[
                        "cluster_id_target"].to_numpy(dtype=str)
                    target_value = binary_metrics(
                        y_pair, target_score)[metric]
                    ref_value = binary_metrics(
                        y_pair, reference_score)[metric]
                    increment = target_value - ref_value
                    unique = np.unique(pair_clusters)
                    indices = {
                        key: np.flatnonzero(pair_clusters == key)
                        for key in unique}
                    boot = []
                    for _ in range(n_boot):
                        sampled = rng.choice(
                            unique, size=len(unique), replace=True)
                        pick = np.concatenate([
                            indices[key] for key in sampled])
                        a = binary_metrics(
                            y_pair[pick], target_score[pick])[metric]
                        b = binary_metrics(
                            y_pair[pick], reference_score[pick])[metric]
                        if math.isfinite(a) and math.isfinite(b):
                            boot.append(a - b)
                    if boot:
                        increment_low, increment_high = np.quantile(
                            boot, [0.025, 0.975]).tolist()
            rows.append({
                "evaluation": evaluation,
                "dataset": (
                    "TopCoW_val" if evaluation == "TopCoW_val_OOF"
                    else evaluation),
                "target": "target_component",
                "model": model_name,
                "features": ",".join(feature_sets[model_name]),
                "metric": metric,
                "value": point[metric],
                "ci_low": low,
                "ci_high": high,
                "reference_model": "G",
                "increment_vs_G": increment,
                "increment_ci_low": increment_low,
                "increment_ci_high": increment_high,
                "n_units": int(len(data)),
                "n_cases": int(data["case_id"].nunique()),
                "n_seeds": int(data["seed"].nunique()),
                "fit_protocol": (
                    "5-fold seed/case-grouped TopCoW OOF; one full TopCoW "
                    "fit applied to each external domain with zero retuning"),
            })
    return pd.DataFrame(rows, columns=columns)


def hierarchical_mean_ci(
        frame: pd.DataFrame, value_column: str, n_boot: int,
        rng: np.random.Generator) -> Tuple[float, float, float]:
    values = pd.to_numeric(frame[value_column], errors="coerce")
    valid = frame.loc[values.notna()].copy()
    valid[value_column] = values[values.notna()].to_numpy(dtype=float)
    if valid.empty:
        return np.nan, np.nan, np.nan
    # All formal callers target one seed/case observation.  Collapse any
    # accidental within-case replicas, then resample model seeds outside and
    # cases inside each sampled seed.  The former generic implementation
    # rebuilt and concatenated pandas frames on every replicate; this numeric
    # implementation has the identical hierarchy without turning a few
    # hundred predeclared endpoints into an hour-long CPU job.
    collapsed = (
        valid.groupby(["seed", "case_id"], dropna=False)[value_column]
        .mean().reset_index())
    point = float(collapsed[value_column].mean())
    by_seed = [
        pd.to_numeric(group[value_column], errors="coerce")
        .dropna().to_numpy(dtype=float)
        for _, group in collapsed.groupby("seed", dropna=False, sort=True)
    ]
    by_seed = [array for array in by_seed if len(array)]
    boot: List[float] = []
    if by_seed:
        seed_count = len(by_seed)
        for _ in range(n_boot):
            selected = (
                rng.integers(0, seed_count, size=seed_count)
                if seed_count > 1 else np.zeros(1, dtype=int))
            total = 0.0
            count = 0
            for seed_index in selected:
                seed_values = by_seed[int(seed_index)]
                sampled = seed_values[
                    rng.integers(
                        0, len(seed_values), size=len(seed_values))]
                total += float(sampled.sum())
                count += int(len(sampled))
            if count:
                boot.append(total / count)
    if not boot:
        return point, np.nan, np.nan
    low, high = np.quantile(boot, [0.025, 0.975]).tolist()
    return point, float(low), float(high)


def multiseed_case_comparisons(
        case: pd.DataFrame, n_boot: int,
        rng: np.random.Generator) -> pd.DataFrame:
    columns = [
        "analysis_kind", "family", "dataset", "corruption_type", "severity",
        "metric", "model_variant", "reference_variant", "estimate",
        "difference_vs_reference", "ci_low", "ci_high", "n_cases",
        "n_model_seeds", "analysis_unit", "bootstrap_protocol",
    ]
    required = {
        "family", "dataset", "corruption_type", "severity", "model_variant",
        "seed", "case_id"}
    if case.empty or not required.issubset(case.columns):
        return pd.DataFrame(columns=columns)
    metrics = [metric for metric in CASE_METRICS if metric in case]
    if not metrics:
        return pd.DataFrame(columns=columns)
    keys = [
        "family", "dataset", "corruption_type", "severity",
        "model_variant", "seed", "case_id"]
    collapsed = (
        case.groupby(keys, dropna=False)[metrics].mean().reset_index())
    rows: List[Dict[str, Any]] = []
    condition_keys = [
        "family", "dataset", "corruption_type", "severity"]
    for condition_values, condition in collapsed.groupby(
            condition_keys, dropna=False, sort=True):
        if not isinstance(condition_values, tuple):
            condition_values = (condition_values,)
        metadata = dict(zip(condition_keys, condition_values))
        variants = sorted(
            condition["model_variant"].dropna().astype(str).unique())
        for variant in variants:
            variant_frame = condition[
                condition["model_variant"].astype(str) == variant]
            if variant_frame["seed"].nunique() < 2:
                continue
            for metric in metrics:
                estimate, low, high = hierarchical_mean_ci(
                    variant_frame, metric, n_boot, rng)
                rows.append({
                    "analysis_kind": "multiseed_estimate",
                    **metadata,
                    "metric": metric,
                    "model_variant": variant,
                    "reference_variant": "",
                    "estimate": estimate,
                    "difference_vs_reference": np.nan,
                    "ci_low": low,
                    "ci_high": high,
                    "n_cases": int(variant_frame["case_id"].nunique()),
                    "n_model_seeds": int(
                        variant_frame["seed"].nunique()),
                    "analysis_unit": (
                        "one mean per model_seed/case after averaging "
                        "corruption realizations"),
                    "bootstrap_protocol": (
                        "outer model-seed / inner case bootstrap"),
                })
        if "full" not in variants:
            continue
        reference = condition[
            condition["model_variant"].astype(str) == "full"]
        pair_keys = ["seed", "case_id"]
        for variant in variants:
            if variant == "full":
                continue
            target = condition[
                condition["model_variant"].astype(str) == variant]
            for metric in metrics:
                paired = target[pair_keys + [metric]].merge(
                    reference[pair_keys + [metric]],
                    on=pair_keys, suffixes=("_target", "_reference"))
                paired["effect"] = (
                    pd.to_numeric(
                        paired[f"{metric}_target"], errors="coerce") -
                    pd.to_numeric(
                        paired[f"{metric}_reference"], errors="coerce"))
                paired = paired.dropna(subset=["effect"])
                if paired.empty or paired["seed"].nunique() < 2:
                    continue
                effect, low, high = hierarchical_mean_ci(
                    paired, "effect", n_boot, rng)
                rows.append({
                    "analysis_kind": "paired_variant_difference",
                    **metadata,
                    "metric": metric,
                    "model_variant": variant,
                    "reference_variant": "full",
                    "estimate": np.nan,
                    "difference_vs_reference": effect,
                    "ci_low": low,
                    "ci_high": high,
                    "n_cases": int(paired["case_id"].nunique()),
                    "n_model_seeds": int(paired["seed"].nunique()),
                    "analysis_unit": (
                        "within-seed/case paired model difference"),
                    "bootstrap_protocol": (
                        "outer model-seed / inner paired-case bootstrap"),
                })
    return pd.DataFrame(rows, columns=columns)


def corruption_manipulation_checks(
        signal: pd.DataFrame, n_boot: int,
        rng: np.random.Generator) -> pd.DataFrame:
    columns = [
        "check_id", "status", "experiment_id", "run_id", "dataset",
        "case_id", "seed", "observed", "tolerance", "details",
        "corruption_type", "severity", "target_level",
        "manipulated_quantity", "expected_direction", "observed_effect",
        "ci_low", "ci_high", "check_status",
        "eligible_for_mechanism_claim", "n_pairs", "n_cases", "n_seeds",
        "pairing_protocol", "bootstrap_protocol", "notes",
    ]
    required = {
        "experiment_id", "family", "dataset", "case_id", "seed",
        "corruption_type", "severity", "feature_level"}
    if signal.empty or not required.issubset(signal.columns):
        return pd.DataFrame(columns=columns)

    clean_types = {"none", "clean", "identity"}
    corruption_specs = {
        "channel_dropout": [
            ("evidence_insufficiency_U", "increase", "U", "clean_U",
             "delta_U_from_clean", None)],
        "dropout": [
            ("evidence_insufficiency_U", "increase", "U", "clean_U",
             "delta_U_from_clean", None)],
        "attenuation": [
            ("evidence_insufficiency_U", "increase", "U", "clean_U",
             "delta_U_from_clean", None)],
        "gaussian": [
            ("evidence_insufficiency_U", "increase", "U", "clean_U",
             "delta_U_from_clean", None)],
        "translation": [
            ("translation_FDD_paper", "increase", "FDD_paper",
             "clean_FDD_paper", "delta_FDD_paper_from_clean",
             "FDD_paper_observed")],
        "donor_mix": [
            ("donor_mix_FDD_paper", "increase", "FDD_paper",
             "clean_FDD_paper", "delta_FDD_paper_from_clean",
             "FDD_paper_observed"),
            ("donor_mix_semantic_compatibility_G", "decrease", "G",
             "clean_G", "delta_G_from_clean", None),
        ],
        "missing_level": [
            ("evidence_insufficiency_U", "increase", "U", "clean_U",
             "delta_U_from_clean", None),
            ("accepted_update_l1", "decrease", "accepted_delta_l1",
             "clean_accepted_delta_l1",
             "delta_accepted_delta_l1_from_clean", None),
        ],
    }
    frame = signal.copy()
    # The formal manipulation endpoint is intentionally narrower than the
    # general result inventory.  Only the predeclared controlled-corruption
    # matrices and the corrected-probability matrix can support a clean-vs-
    # corrupt mechanism statement.  Replaying this bootstrap for temperature,
    # gate-transform, dose, fixed-policy, and trained-variant families both
    # changes the estimand and multiplies thousands of irrelevant contexts.
    frame = frame[
        frame["family"].isin(
            {"p0_corruption", "corrected_probability"})].copy()
    if "model_variant" in frame:
        frame = frame[
            frame["model_variant"].fillna("full").astype(str).str.lower()
            == "full"].copy()
    if "temperature_scale" in frame:
        values = pd.to_numeric(
            frame["temperature_scale"], errors="coerce")
        frame = frame[
            values.isna() | np.isclose(values, 1.0)].copy()
    if "gate_transform" in frame:
        frame = frame[
            frame["gate_transform"].fillna("identity").astype(
                str).str.lower() == "identity"].copy()
    if "gate_scale" in frame:
        values = pd.to_numeric(frame["gate_scale"], errors="coerce")
        frame = frame[
            values.isna() | np.isclose(values, 1.0)].copy()
    if "gate_alpha" in frame:
        values = pd.to_numeric(frame["gate_alpha"], errors="coerce")
        frame = frame[
            values.isna() | np.isclose(values, 1.0)].copy()
    if frame.empty:
        return pd.DataFrame(columns=columns)
    frame["_corruption_key"] = (
        frame["corruption_type"].astype(str).str.lower())
    observed_types = set(frame["_corruption_key"].unique())
    relevant_types = observed_types.intersection(corruption_specs)
    if not relevant_types:
        return pd.DataFrame(columns=columns)

    context_columns = [
        column for column in (
            "family", "model_variant", "temperature_scale",
            "gate_transform", "gate_scale", "gate_alpha")
        if column in frame]
    unit_keys = [
        "experiment_id", *context_columns, "dataset", "corruption_type",
        "_corruption_key", "severity", "feature_level", "seed", "case_id"]
    numeric_candidates = sorted({
        field
        for specs in corruption_specs.values()
        for spec in specs
        for field in spec[2:5]
        if field
    })
    numeric_candidates = [
        column for column in numeric_candidates if column in frame]
    numeric_frame = frame.copy()
    for column in numeric_candidates:
        numeric_frame[column] = pd.to_numeric(
            numeric_frame[column], errors="coerce")
    units = (
        numeric_frame.groupby(unit_keys, dropna=False)[numeric_candidates]
        .mean().reset_index()
        if numeric_candidates else
        numeric_frame.groupby(unit_keys, dropna=False).size().reset_index(
            name="_rows"))
    for availability in ("FDD_paper_observed",):
        if availability in frame:
            available = (
                frame.groupby(unit_keys, dropna=False)[availability]
                .agg(lambda values: bool(pd.Series(values).fillna(False).all()))
                .reset_index())
            units = units.merge(available, on=unit_keys, how="left")

    clean_mask = (
        frame["_corruption_key"].isin(clean_types) &
        (pd.to_numeric(frame["severity"], errors="coerce").fillna(0.0) == 0))
    clean = frame[clean_mask].copy()
    clean_key = [
        *context_columns, "dataset", "feature_level", "seed", "case_id"]
    clean_numeric = [
        column for column in ("U", "G", "FDD_paper", "accepted_delta_l1")
        if column in clean]
    for column in clean_numeric:
        clean[column] = pd.to_numeric(clean[column], errors="coerce")
    clean_units = (
        clean.groupby(clean_key, dropna=False)[clean_numeric].mean().reset_index()
        if not clean.empty and clean_numeric else pd.DataFrame())
    clean_units = clean_units.rename(columns={
        "U": "_paired_clean_U",
        "G": "_paired_clean_G",
        "FDD_paper": "_paired_clean_FDD_paper",
        "accepted_delta_l1": "_paired_clean_accepted_delta_l1",
    })

    condition_keys = [
        "experiment_id", *context_columns, "dataset", "corruption_type",
        "_corruption_key", "severity", "feature_level"]
    rows: List[Dict[str, Any]] = []
    for condition_values, condition in units[
            units["_corruption_key"].isin(relevant_types)].groupby(
                condition_keys, dropna=False, sort=True):
        if not isinstance(condition_values, tuple):
            condition_values = (condition_values,)
        metadata = dict(zip(condition_keys, condition_values))
        corruption_key = str(metadata["_corruption_key"])
        for (
                quantity, direction, current_field, clean_field, delta_field,
                availability_field) in corruption_specs[corruption_key]:
            effects = condition[["seed", "case_id"]].copy()
            reason = ""
            if (
                    availability_field and
                    (availability_field not in condition or
                     not condition[availability_field].fillna(False).all())):
                reason = f"missing_native_field:{current_field}"
                effects["effect"] = np.nan
            elif (
                    delta_field in condition and
                    pd.to_numeric(
                        condition[delta_field], errors="coerce").notna().any()):
                effects["effect"] = pd.to_numeric(
                    condition[delta_field], errors="coerce")
            elif current_field not in condition:
                reason = f"missing_field:{current_field}"
                effects["effect"] = np.nan
            elif clean_field in condition and pd.to_numeric(
                    condition[clean_field], errors="coerce").notna().any():
                effects["effect"] = (
                    pd.to_numeric(
                        condition[current_field], errors="coerce") -
                    pd.to_numeric(
                        condition[clean_field], errors="coerce"))
            elif not clean_units.empty:
                paired = condition.merge(
                    clean_units, on=clean_key, how="left")
                paired_clean = f"_paired_clean_{current_field}"
                effects = paired[["seed", "case_id"]].copy()
                if paired_clean not in paired:
                    reason = f"missing_clean_field:{current_field}"
                    effects["effect"] = np.nan
                else:
                    effects["effect"] = (
                        pd.to_numeric(
                            paired[current_field], errors="coerce") -
                        pd.to_numeric(
                            paired[paired_clean], errors="coerce"))
            else:
                reason = f"no_clean_pair:{current_field}"
                effects["effect"] = np.nan
            effects = effects.dropna(subset=["effect"])
            point, ci_low, ci_high = hierarchical_mean_ci(
                effects, "effect", n_boot, rng)
            tolerance = 1e-6
            status = "inconclusive"
            if reason or effects.empty or not math.isfinite(point):
                status = "inconclusive"
            elif abs(point) <= tolerance:
                status = "failed_manipulation"
                reason = "observed_effect_within_zero_tolerance"
            elif direction == "increase" and point < 0:
                status = "failed_manipulation"
                reason = "point_estimate_opposite_expected_direction"
            elif direction == "decrease" and point > 0:
                status = "failed_manipulation"
                reason = "point_estimate_opposite_expected_direction"
            elif not (math.isfinite(ci_low) and math.isfinite(ci_high)):
                status = "inconclusive"
                reason = "clustered_ci_not_estimable"
            elif (
                    (direction == "increase" and ci_low > 0) or
                    (direction == "decrease" and ci_high < 0)):
                status = "pass"
                reason = "paired_effect_ci_excludes_zero_expected_direction"
            else:
                status = "inconclusive"
                reason = "expected_direction_but_ci_crosses_zero"
            context_token = ",".join(
                f"{column}={metadata.get(column)}"
                for column in context_columns)
            check_id = (
                f"corruption_manipulation::{corruption_key}::{quantity}::"
                f"{metadata.get('dataset')}::{metadata.get('severity')}::"
                f"{metadata.get('feature_level')}::{context_token}")
            details = (
                f"quantity={quantity}; direction={direction}; reason={reason}; "
                f"context={context_token}; clean-vs-corrupt paired before "
                "clustered inference")
            rows.append({
                "check_id": check_id,
                "status": status,
                "experiment_id": metadata.get("experiment_id"),
                "run_id": "aggregate_clean_vs_corrupt",
                "dataset": metadata.get("dataset"),
                "case_id": "all_paired_cases",
                "seed": "all_model_seeds",
                "observed": point,
                "tolerance": tolerance,
                "details": details,
                "corruption_type": metadata.get("corruption_type"),
                "severity": metadata.get("severity"),
                "target_level": metadata.get("feature_level"),
                "manipulated_quantity": quantity,
                "expected_direction": direction,
                "observed_effect": point,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "check_status": status,
                "eligible_for_mechanism_claim": bool(status == "pass"),
                "n_pairs": int(len(effects)),
                "n_cases": int(effects["case_id"].nunique()),
                "n_seeds": int(effects["seed"].nunique()),
                "pairing_protocol": (
                    "clean-vs-corrupt within model context, seed, case and "
                    "feature level; explicit raw delta preferred"),
                "bootstrap_protocol": (
                    "outer model-seed / inner paired-case bootstrap"
                    if effects["seed"].nunique() > 1 else
                    "single-seed paired-case bootstrap"),
                "notes": reason,
            })
    return pd.DataFrame(rows, columns=columns)


def manipulation_checks(case: pd.DataFrame, signal: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "check_id", "status", "experiment_id", "run_id", "dataset",
        "case_id", "seed", "observed", "tolerance", "details",
    ]
    rows: List[Dict[str, Any]] = []
    corrected = case[case["family"] == "corrected_probability"].copy()
    legacy = case[case["family"] != "corrected_probability"].copy()
    old_lookup: Dict[Tuple[Any, ...], List[str]] = {}
    match_columns = (
        "dataset", "case_id", "model_variant", "corruption_type",
        "severity", "realization")
    if not legacy.empty and "prediction_sha256" in legacy:
        for _, item in legacy.iterrows():
            key = tuple(item.get(column) for column in match_columns)
            value = item.get("prediction_sha256")
            if pd.notna(value):
                old_lookup.setdefault(key, []).append(str(value))
    for _, item in corrected.iterrows():
        base = {
            key: item.get(key) for key in (
                "experiment_id", "run_id", "dataset", "case_id", "seed")}
        checks = []
        value = finite(item.get("probability_sum_max_abs_error"))
        checks.append((
            "probability_normalization", value is not None and value <= 1e-4,
            value, 1e-4, "max |sum_c p_c - 1|"))
        minimum = finite(item.get("probability_min"))
        maximum = finite(item.get("probability_max"))
        range_ok = (
            minimum is not None and maximum is not None and
            minimum >= -1e-6 and maximum <= 1.0 + 1e-6)
        checks.append((
            "probability_range", range_ok,
            f"min={minimum},max={maximum}", "[-1e-6,1+1e-6]",
            "normalized model probability range"))
        legacy_equal = item.get("legacy_double_softmax_argmax_equal")
        if isinstance(legacy_equal, str):
            legacy_equal = legacy_equal.lower() == "true"
        if pd.notna(legacy_equal):
            checks.append((
                "same_runtime_legacy_argmax", bool(legacy_equal),
                legacy_equal, True,
                "corrected and quarantined double-softmax argmax"))
        submitted = item.get(
            "submitted_argmax_sha256", item.get("prediction_sha256"))
        legacy_sha = item.get("legacy_double_softmax_argmax_sha256")
        if pd.notna(legacy_sha):
            checks.append((
                "same_runtime_hard_pred_sha",
                pd.notna(submitted) and
                str(submitted) == str(legacy_sha),
                f"{submitted}|{legacy_sha}", "equal",
                "hard prediction SHA within corrected run"))
        key = tuple(item.get(column) for column in match_columns)
        candidates = sorted(set(old_lookup.get(key, [])))
        if candidates:
            observed = f"corrected={submitted};old={';'.join(candidates)}"
            # These historical and corrected matrices were produced under
            # different commits/GPU models. Preserve the comparison as
            # descriptive provenance; it is not a valid same-runtime
            # softmax-only invariant.
            status = "na"
        else:
            observed = f"corrected={submitted};old=unavailable"
            status = "na"
        rows.append({
            "check_id": "corrected_vs_old_hard_pred_sha",
            "status": status,
            **base,
            "observed": observed,
            "tolerance": "exact",
            "details": "condition-matched old probability/control group",
        })
        for check_id, passed, observed, tolerance, details in checks:
            rows.append({
                "check_id": check_id,
                "status": "pass" if passed else "fail",
                **base,
                "observed": observed,
                "tolerance": tolerance,
                "details": details,
            })
    if not case.empty and "A_transform_invariant_residual_max" in case:
        for _, item in case[
                case["A_transform_invariant_residual_max"].notna()].iterrows():
            value = finite(item["A_transform_invariant_residual_max"])
            rows.append({
                "check_id": "gate_transform_runtime_invariant",
                "status": (
                    "pass" if value is not None and value <= 1e-5 else "fail"),
                **{key: item.get(key) for key in (
                    "experiment_id", "run_id", "dataset", "case_id", "seed")},
                "observed": value,
                "tolerance": 1e-5,
                "details": (
                    f"transform={item.get('gate_transform')}, "
                    f"scale={item.get('gate_scale')}"),
            })
    if not signal.empty:
        keys = [
            key for key in (
                "semantic_id", "experiment_id", "run_id", "dataset",
                "case_id", "seed", "feature_level")
            if key in signal]
        numeric = [
            key for key in ("U", "G", "T_formula", "A_impl_raw")
            if key in signal]
        collapsed = signal.groupby(keys, dropna=False)[numeric].mean().reset_index()
        if {"U", "G", "T_formula"}.issubset(collapsed.columns):
            residual = (
                collapsed["T_formula"] -
                collapsed["G"] * (1.0 - collapsed["U"])).abs()
            for index, item in collapsed.iterrows():
                value = finite(residual.loc[index])
                rows.append({
                    "check_id": "T_formula_equals_G_times_1_minus_U",
                    # Each persisted scalar is a separate ROI mean.  In
                    # general E[G(1-U)] != E[G](1-E[U]), so this residual
                    # cannot test the voxelwise formula and must never be
                    # promoted to a pass/fail invariant.
                    "status": "na",
                    **{key: item.get(key) for key in (
                        "experiment_id", "run_id", "dataset", "case_id",
                        "seed")},
                    "observed": value,
                    "tolerance": "not identifiable from separate ROI means",
                    "details": (
                        f"feature_level={item.get('feature_level')}; "
                        "descriptive mean-product discrepancy only"),
                })
        explicit_raw = (
            signal[signal["A_impl_raw_observed"].fillna(False)].copy()
            if "A_impl_raw_observed" in signal else pd.DataFrame())
        if not explicit_raw.empty:
            if "model_variant" in explicit_raw:
                explicit_raw = explicit_raw[
                    explicit_raw["model_variant"] == "full"]
            if "gate_alpha" in explicit_raw:
                explicit_raw = explicit_raw[
                    explicit_raw["gate_alpha"].isna()]
            if "T_formula_observed" in explicit_raw:
                explicit_raw = explicit_raw[
                    explicit_raw["T_formula_observed"].fillna(False)]
        if (
                not explicit_raw.empty and
                {"U", "G", "T_formula", "A_impl_raw"}.issubset(
                    explicit_raw.columns)):
            raw_collapsed = (
                explicit_raw.groupby(keys, dropna=False)[
                    ["U", "G", "T_formula", "A_impl_raw"]]
                .mean().reset_index())
            # For the submitted full policy the two voxelwise definitions
            # imply T_formula + A_impl_raw = 1 + G - U.  Unlike multiplying
            # separate ROI means, this linear identity remains exact after
            # regional averaging.
            residual = (
                raw_collapsed["T_formula"] +
                raw_collapsed["A_impl_raw"] -
                (1.0 + raw_collapsed["G"] - raw_collapsed["U"])).abs()
            for index, item in raw_collapsed.iterrows():
                value = finite(residual.loc[index])
                rows.append({
                    "check_id": "A_impl_raw_identity",
                    "status": (
                        "pass" if value is not None and value <= 1e-5
                        else "fail"),
                    **{key: item.get(key) for key in (
                        "experiment_id", "run_id", "dataset", "case_id",
                        "seed")},
                    "observed": value,
                    "tolerance": 1e-5,
                    "details": (
                        f"feature_level={item.get('feature_level')}; "
                        "mean-preserving joint identity "
                        "T_formula+A_impl_raw=1+G-U"),
                })
        if "U" in signal:
            spread = signal.groupby(keys, dropna=False)["U"].agg(
                lambda values: float(
                    pd.to_numeric(values, errors="coerce").max() -
                    pd.to_numeric(values, errors="coerce").min())
                if pd.to_numeric(values, errors="coerce").notna().any()
                else np.nan).reset_index(name="spread")
            for _, item in spread.iterrows():
                value = finite(item["spread"])
                rows.append({
                    "check_id": "U_not_replicated_as_independent_region_signal",
                    # U is summarized inside distinct ROIs and therefore may
                    # legitimately vary between region rows.  Independence is
                    # controlled structurally by the downstream case/level
                    # collapse, not by requiring identical ROI means.
                    "status": "na",
                    **{key: item.get(key) for key in (
                        "experiment_id", "run_id", "dataset", "case_id",
                        "seed")},
                    "observed": value,
                    "tolerance": "not applicable to ROI-specific means",
                    "details": (
                        f"feature_level={item.get('feature_level')}; "
                        "observed ROI spread is descriptive; analysis "
                        "collapses region rows before statistical inference"),
                })
    return pd.DataFrame(rows, columns=columns)


def negative_findings(
        inventory: pd.DataFrame, checks: pd.DataFrame,
        validity: pd.DataFrame, practical: pd.DataFrame,
        load_failures: Sequence[Mapping[str, Any]],
        corruption_checks: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    columns = ["severity", "category", "finding", "evidence"]
    rows = [dict(item) for item in load_failures]
    if not checks.empty:
        failed = checks[checks["status"] == "fail"]
        for check_id, group in failed.groupby("check_id"):
            rows.append({
                "severity": "high",
                "category": "failed_invariant",
                "finding": f"{check_id}: {len(group)} failed rows",
                "evidence": ";".join(
                    group["run_id"].dropna().astype(str).unique()[:10]),
            })
    if not validity.empty:
        weak = validity[
            (validity["metric"] == "AUPRC") &
            validity["value"].notna() &
            validity["prevalence"].notna() &
            (validity["prevalence"] > 0) &
            (validity["prevalence"] < 1) &
            (validity["value"] < validity["prevalence"] - 1e-6)]
        for _, item in weak.iterrows():
            rows.append({
                "severity": "mixed",
                "category": "signal_validity",
                "finding": (
                    f"{item['signal']} AUPRC={item['value']:.4g} does not "
                    f"exceed prevalence={item['prevalence']:.4g} for "
                    f"{item['dataset']}/{item['target']}"),
                "evidence": (
                    f"corruption={item['corruption_type']},"
                    f"severity={item['severity']}"),
            })
    if not practical.empty:
        paired = practical[
            practical["paired_difference"].notna() &
            practical["paired_ci_low"].notna() &
            practical["paired_ci_high"].notna()]
        for _, item in paired.iterrows():
            worse = (
                item["paired_ci_high"] < 0
                if bool(item["higher_is_better"])
                else item["paired_ci_low"] > 0)
            if worse:
                rows.append({
                    "severity": "mixed",
                    "category": "practical_effect",
                    "finding": (
                        f"{item['analysis_family']} label={item['label']} "
                        f"is worse than {item['reference_label']} on "
                        f"{item['metric']} (paired CI excludes 0)"),
                    "evidence": (
                        f"{item.get('dataset')}/"
                        f"{item.get('corruption_type')}/"
                        f"{item.get('severity')}"),
                })
    if corruption_checks is not None and not corruption_checks.empty:
        nonpassing = corruption_checks[
            corruption_checks["check_status"] != "pass"]
        for _, item in nonpassing.iterrows():
            rows.append({
                "severity": (
                    "high" if
                    item["check_status"] == "failed_manipulation"
                    else "warning"),
                "category": "corruption_manipulation",
                "finding": (
                    f"{item['corruption_type']}/"
                    f"{item['manipulated_quantity']} is "
                    f"{item['check_status']} and is not eligible for a "
                    "mechanism claim"),
                "evidence": (
                    f"dataset={item['dataset']},severity={item['severity']},"
                    f"level={item['target_level']},notes={item['notes']}"),
            })
    observed = set(
        inventory["experiment_id"].dropna().astype(str)) if not inventory.empty else set()
    for experiment in sorted(V2_EXPERIMENTS - observed):
        rows.append({
            "severity": "pending",
            "category": "missing_group",
            "finding": f"no successful raw run observed for {experiment}",
            "evidence": "current results_root snapshot",
        })
    return pd.DataFrame(rows, columns=columns).drop_duplicates()


def endpoint_manifest_gates(
        reliability: pd.DataFrame, incremental: pd.DataFrame,
        multiseed: pd.DataFrame,
        corruption_checks: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    required_reliability = {
        "AUPRC_beneficial", "AUROC", "AURC", "Brier", "NLL",
        "calibration_intercept", "calibration_slope", "HAR", "BRR",
        "coverage"}
    observed_reliability = (
        set(reliability["metric"].dropna().astype(str))
        if not reliability.empty else set())
    missing_reliability = sorted(
        required_reliability - observed_reliability)
    reliability_gate = {
        "status": (
            "PASS" if not missing_reliability and not reliability.empty
            else "INCONCLUSIVE"),
        "required_metrics": sorted(required_reliability),
        "observed_metrics": sorted(observed_reliability),
        "missing_metrics": missing_reliability,
        "rows": int(len(reliability)),
        "calibration_unidentifiable_rows": int(
            reliability["endpoint_status"].astype(str).str.startswith(
                "inconclusive").sum()) if not reliability.empty else 0,
        "note": (
            "Calibration intercept/slope are diagnostic evaluation regressions "
            "and may be explicitly inconclusive for single-class or constant-"
            "score strata; no external calibrator or threshold is fitted."),
    }

    interaction = (
        incremental[incremental["model"] == "G+U+G*U"]
        if not incremental.empty and "model" in incremental
        else pd.DataFrame())
    required_increment_metrics = {"AUPRC", "Brier", "AURC"}
    observed_increment_metrics = (
        set(interaction["metric"].dropna().astype(str))
        if not interaction.empty else set())
    interaction_gate = {
        "status": (
            "PASS"
            if required_increment_metrics.issubset(observed_increment_metrics)
            else "INCONCLUSIVE"),
        "model": "G+U+G*U",
        "required_held_out_metrics": sorted(required_increment_metrics),
        "observed_metrics": sorted(observed_increment_metrics),
        "rows": int(len(interaction)),
        "fit_protocol": (
            "TopCoW case-grouped cross-fitting; external domains receive the "
            "locked source fit without retuning"),
    }

    valid_multiseed = (
        multiseed[
            pd.to_numeric(
                multiseed["n_model_seeds"], errors="coerce").fillna(0) >= 2]
        if not multiseed.empty else pd.DataFrame())
    multiseed_gate = {
        "status": "PASS" if not valid_multiseed.empty else "INCONCLUSIVE",
        "rows": int(len(multiseed)),
        "rows_with_at_least_two_seeds": int(len(valid_multiseed)),
        "bootstrap_protocol": "outer model-seed / inner case bootstrap",
    }

    missing_pattern = r"missing_|no_clean_pair"
    missing_inputs = (
        corruption_checks[
            corruption_checks["notes"].astype(str).str.contains(
                missing_pattern, regex=True, na=False)]
        if not corruption_checks.empty else pd.DataFrame())
    manipulation_gate = {
        "status": (
            "PASS"
            if not corruption_checks.empty and missing_inputs.empty
            else "INCONCLUSIVE"),
        "rows": int(len(corruption_checks)),
        "pass_rows": int(
            (corruption_checks["check_status"] == "pass").sum())
        if not corruption_checks.empty else 0,
        "inconclusive_rows": int(
            (corruption_checks["check_status"] == "inconclusive").sum())
        if not corruption_checks.empty else 0,
        "failed_manipulation_rows": int(
            (corruption_checks[
                "check_status"] == "failed_manipulation").sum())
        if not corruption_checks.empty else 0,
        "missing_input_rows": int(len(missing_inputs)),
        "claim_rule": (
            "Only check_status=pass is eligible_for_mechanism_claim; "
            "scientific null/opposite outcomes are retained as negative "
            "findings rather than treated as pipeline errors."),
    }
    return {
        "reliability_endpoint_gate": reliability_gate,
        "incremental_interaction_gate": interaction_gate,
        "multiseed_resampling_gate": multiseed_gate,
        "corruption_manipulation_gate": manipulation_gate,
    }


def summary_markdown(
        results_root: Path, output_dir: Path, inventory: pd.DataFrame,
        case: pd.DataFrame, signal: pd.DataFrame, units: pd.DataFrame,
        unit_audit: Mapping[str, Any], checks: pd.DataFrame,
        validity: pd.DataFrame, reliability: pd.DataFrame,
        incremental: pd.DataFrame, multiseed: pd.DataFrame,
        corruption_checks: pd.DataFrame, practical: pd.DataFrame,
        negatives: pd.DataFrame, gates: Mapping[str, Mapping[str, Any]],
        bootstrap: int, seed: int) -> str:
    experiment_counts = (
        inventory.groupby("experiment_id").size().sort_index().to_dict()
        if not inventory.empty else {})
    failed_checks = (
        int((checks["status"] == "fail").sum()) if not checks.empty else 0)
    passed_checks = (
        int((checks["status"] == "pass").sum()) if not checks.empty else 0)
    lines = [
        "# TCFF V2 Results Summary",
        "",
        f"- Protocol: `{PROTOCOL}`",
        f"- Generated UTC: `{dt.datetime.now(dt.timezone.utc).isoformat()}`",
        f"- Results root: `{results_root}`",
        f"- Output directory: `{output_dir}`",
        f"- Bootstrap: `{bootstrap}` seed/case-cluster replicates",
        f"- RNG seed: `{seed}`",
        f"- Unique successful runs selected: `{len(inventory)}`",
        f"- Case rows: `{len(case)}`",
        f"- Raw region rows: `{len(signal)}`",
        f"- Collapsed case/level units: `{len(units)}`",
        (
            "- Signal analysis unit: "
            "`run/seed/case/feature_level mean`; region/voxel copies of U "
            "are never treated as independent observations."
        ),
        "",
        "## Successful run inventory",
        "",
    ]
    if experiment_counts:
        lines.extend(
            f"- `{key}`: {value}" for key, value in experiment_counts.items())
    else:
        lines.append("- No successful raw runs were found.")
    lines.extend([
        "",
        "## Invariants and probability audit",
        "",
        f"- Passed checks: `{passed_checks}`",
        f"- Failed checks: `{failed_checks}`",
        (
            f"- Corrected probability rows: "
            f"`{int((case.get('family', pd.Series(dtype=str)) == 'corrected_probability').sum())}`"
        ),
        "",
        "## Signal validity",
        "",
        f"- Output rows: `{len(validity)}`",
        f"- Source selection: `{unit_audit.get('source_family', 'none')}`",
        (
            "- Scores are operational reliability rankings. U is evaluated "
            "as 1-U, FDDpaper as 1-clip(FDDpaper,0,1), and FDDlegacy "
            "as 1-clip(FDDlegacy,0,2)/2."
        ),
        (
            f"- Formal calibration/action endpoint rows: "
            f"`{len(reliability)}`; calibration slopes/intercepts are "
            "diagnostic only and never refit an external-domain predictor."
        ),
        "",
        "## Incremental value",
        "",
        f"- Output rows: `{len(incremental)}`",
        (
            "- Models are fit only on TopCoW with seed/case-grouped folds; "
            "external cohorts receive the locked full-source fit."
        ),
        "- The prespecified interaction model is `G+U+G*U`.",
        "",
        "## Multi-seed and corruption inference",
        "",
        f"- Multi-seed case-level rows: `{len(multiseed)}`",
        (
            "- Multi-seed intervals use an outer model-seed / inner case "
            "bootstrap; single-seed checks use paired-case bootstrap."
        ),
        f"- Formal corruption-manipulation rows: `{len(corruption_checks)}`",
        (
            "- Only manipulation rows with `check_status=pass` are eligible "
            "for a mechanism claim; missing fields and nonpassing results are "
            "retained explicitly."
        ),
        "- Manifest endpoint gates: " + ", ".join(
            f"`{name}={gate.get('status')}`"
            for name, gate in gates.items()),
        "",
        "## Practical effects",
        "",
        f"- Output rows: `{len(practical)}`",
        (
            "- `constant_acceptance` and `alpha_times_policy` are deliberately "
            "separate analyses."
        ),
        "",
        "## Negative, mixed, or pending findings",
        "",
        f"- Findings: `{len(negatives)}`",
    ])
    if not negatives.empty:
        for _, item in negatives.head(30).iterrows():
            lines.append(
                f"- **{item['severity']} / {item['category']}**: "
                f"{item['finding']}")
    lines.extend([
        "",
        "## Files",
        "",
        "- `v2_run_inventory.csv`",
        "- `v2_temperature_sensitivity.csv`",
        "- `v2_formula_gate_replay.csv`",
        "- `v2_gate_dose_response.csv`",
        "- `v2_manipulation_checks.csv`",
        "- `v2_corruption_manipulation_checks.csv`",
        "- `v2_signal_validity.csv`",
        "- `v2_reliability_calibration.csv`",
        "- `v2_incremental_value.csv`",
        "- `v2_multiseed_case_comparisons.csv`",
        "- `v2_practical_effects.csv`",
        "- `v2_negative_mixed_findings.csv`",
        "",
    ])
    return "\n".join(lines)


def self_test() -> int:
    unit_rows = []
    case_rows = []
    for seed in range(3):
        for case_index in range(20):
            beneficial = int(case_index % 2 == 0)
            g_value = 0.82 if beneficial else 0.18
            u_value = 0.22 if beneficial else 0.78
            case_id = f"case_{case_index:02d}"
            unit_rows.append({
                "dataset": "TopCoW_val",
                "corruption_type": "none",
                "severity": 0.0,
                "target_component": beneficial,
                "target_vox": beneficial,
                "U": u_value,
                "G": g_value,
                "T_formula": g_value * (1.0 - u_value),
                "A_impl": 1.0 - (1.0 - g_value) * u_value,
                "FDD_paper": 1.0 - g_value,
                "confidence": g_value,
                "entropy": 1.0 - g_value,
                "feature_level": "deep",
                "seed": seed,
                "case_id": case_id,
                "cluster_id": f"{seed}|{case_id}",
            })
            case_rows.append({
                "family": "trained_variant",
                "dataset": "TopCoW_val",
                "corruption_type": "none",
                "severity": 0.0,
                "model_variant": "full",
                "seed": seed,
                "case_id": case_id,
                "dice": 0.8 + 0.01 * beneficial,
                "cldice": 0.75 + 0.01 * beneficial,
                "hd95": 2.0 - 0.1 * beneficial,
                "beta0_error": 1.0 - 0.1 * beneficial,
                "clsavg": 0.7 + 0.01 * beneficial,
                "cls_beta0": 1.0 - 0.1 * beneficial,
                "det_f1": 0.7 + 0.01 * beneficial,
            })
    units = pd.DataFrame(unit_rows)
    reliability = formal_reliability_endpoints(
        units, 40, np.random.default_rng(18))
    required_metrics = {
        "NLL", "calibration_intercept", "calibration_slope",
        "HAR", "BRR", "coverage"}
    if not required_metrics.issubset(set(reliability["metric"])):
        raise AssertionError("formal reliability endpoints are incomplete")

    failures: List[Dict[str, Any]] = []
    incremental = incremental_value(
        units, 40, np.random.default_rng(19), failures)
    interaction = incremental[
        incremental["model"] == "G+U+G*U"]
    if interaction.empty or not {"AUPRC", "Brier", "AURC"}.issubset(
            set(interaction["metric"])):
        raise AssertionError("G+U+G*U held-out model is missing")

    multiseed = multiseed_case_comparisons(
        pd.DataFrame(case_rows), 40, np.random.default_rng(20))
    if multiseed.empty or int(multiseed["n_model_seeds"].min()) != 3:
        raise AssertionError("multi-seed hierarchical summary is missing")

    signal_rows = []
    specifications = {
        "channel_dropout": {
            "U": 0.55, "delta_U_from_clean": 0.30},
        "translation": {
            "FDD_paper": 0.55, "clean_FDD_paper": 0.15,
            "delta_FDD_paper_from_clean": 0.40},
        "donor_mix": {
            "FDD_paper": 0.60, "clean_FDD_paper": 0.15,
            "delta_FDD_paper_from_clean": 0.45,
            "G": 0.35, "clean_G": 0.80,
            "delta_G_from_clean": -0.45},
        "missing_level": {
            "U": 0.70, "delta_U_from_clean": 0.45,
            "accepted_delta_l1": 0.10,
            "delta_accepted_delta_l1_from_clean": -0.40},
    }
    for corruption_type, values in specifications.items():
        for seed in range(3):
            for case_index in range(12):
                row = {
                    "experiment_id": "SYNTHETIC_MANIPULATION",
                    "family": "p0_corruption",
                    "model_variant": "full",
                    "dataset": "TopCoW_val",
                    "case_id": f"case_{case_index:02d}",
                    "seed": seed,
                    "corruption_type": corruption_type,
                    "severity": 1.0,
                    "feature_level": "deep",
                    "FDD_paper_observed": True,
                }
                row.update(values)
                signal_rows.append(row)
    corruption_checks = corruption_manipulation_checks(
        pd.DataFrame(signal_rows), 40, np.random.default_rng(21))
    if corruption_checks.empty or not (
            corruption_checks["check_status"] == "pass").all():
        raise AssertionError(
            "positive corruption manipulation directions did not pass")

    missing = pd.DataFrame([{
        "experiment_id": "SYNTHETIC_MISSING_FIELD",
        "family": "p0_corruption",
        "model_variant": "full",
        "dataset": "TopCoW_val",
        "case_id": "case_missing",
        "seed": 0,
        "corruption_type": "translation",
        "severity": 1.0,
        "feature_level": "deep",
    }])
    missing_result = corruption_manipulation_checks(
        missing, 20, np.random.default_rng(22))
    if (
            len(missing_result) != 1 or
            missing_result.iloc[0]["check_status"] != "inconclusive" or
            "missing_native_field" not in str(
                missing_result.iloc[0]["notes"])):
        raise AssertionError("missing raw field was not a strict inconclusive")

    gates = endpoint_manifest_gates(
        reliability, incremental, multiseed, corruption_checks)
    if any(gate["status"] != "PASS" for gate in gates.values()):
        raise AssertionError(f"positive endpoint gates failed: {gates}")
    print(json.dumps({
        "status": "SELF_TEST_PASS",
        "reliability_rows": len(reliability),
        "interaction_rows": len(interaction),
        "multiseed_rows": len(multiseed),
        "manipulation_rows": len(corruption_checks),
        "missing_field_status": missing_result.iloc[0]["check_status"],
    }, sort_keys=True))
    return 0


def main(args: argparse.Namespace) -> int:
    results_root = Path(args.results_root).resolve()
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir else results_root / "12_v2_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    inventory, scan_failures = scan_success_runs(results_root)
    signal_experiment_ids = (
        REVIEWER_CORE_SIGNAL_EXPERIMENTS
        if args.signal_scope == "reviewer_core" else None)
    case, signal, load_failures = load_artifacts(
        inventory, signal_experiment_ids=signal_experiment_ids)
    explicit_semantics = explicit_signal_semantics_audit(
        inventory, signal)
    case = normalize_case_columns(case)
    signal = normalize_signal_columns(signal)

    inventory_public = inventory.drop(
        columns=["manifest", "mtime_ns"], errors="ignore")
    atomic_csv(
        output_dir / "v2_run_inventory.csv", inventory_public,
        columns=[
            "semantic_id", "experiment_id", "run_id", "canonical_run_id",
            "dataset", "model_variant", "seed", "checkpoint_hash",
            "data_split_hash", "finished_at", "run_dir",
            "has_case_metrics", "has_internal_signals",
            "has_aggregate_metrics", "duplicate_success_attempts",
            "superseded_success_dirs",
        ])

    practical_tables = make_practical_tables(
        case, args.bootstrap, rng)
    atomic_csv(
        output_dir / "v2_temperature_sensitivity.csv",
        practical_tables["temperature"])
    atomic_csv(
        output_dir / "v2_formula_gate_replay.csv",
        practical_tables["formula"])
    atomic_csv(
        output_dir / "v2_gate_dose_response.csv",
        practical_tables["dose"])
    atomic_csv(
        output_dir / "v2_practical_effects.csv",
        practical_tables["practical"])

    invariant_checks = manipulation_checks(case, signal)
    corruption_checks = corruption_manipulation_checks(
        signal, args.bootstrap, rng)
    checks = pd.concat(
        [invariant_checks, corruption_checks],
        ignore_index=True, sort=False)
    atomic_csv(
        output_dir / "v2_manipulation_checks.csv", checks,
        columns=[
            "check_id", "status", "experiment_id", "run_id", "dataset",
            "case_id", "seed", "observed", "tolerance", "details",
            "corruption_type", "severity", "target_level",
            "manipulated_quantity", "expected_direction", "observed_effect",
            "ci_low", "ci_high", "check_status",
            "eligible_for_mechanism_claim", "n_pairs", "n_cases", "n_seeds",
            "pairing_protocol", "bootstrap_protocol", "notes",
        ])
    atomic_csv(
        output_dir / "v2_corruption_manipulation_checks.csv",
        corruption_checks)

    units, unit_audit = collapse_signal_units(signal)
    validity = signal_validity(units, args.bootstrap, rng)
    atomic_csv(
        output_dir / "v2_signal_validity.csv", validity)
    reliability = formal_reliability_endpoints(
        units, args.bootstrap, rng)
    atomic_csv(
        output_dir / "v2_reliability_calibration.csv", reliability)
    all_failures = list(scan_failures) + list(load_failures)
    incremental = incremental_value(
        units, args.bootstrap, rng, all_failures)
    atomic_csv(
        output_dir / "v2_incremental_value.csv", incremental)
    multiseed = multiseed_case_comparisons(
        case, args.bootstrap, rng)
    atomic_csv(
        output_dir / "v2_multiseed_case_comparisons.csv", multiseed)
    gates = endpoint_manifest_gates(
        reliability, incremental, multiseed, corruption_checks)
    negatives = negative_findings(
        inventory, checks, validity, practical_tables["practical"],
        all_failures, corruption_checks)
    atomic_csv(
        output_dir / "v2_negative_mixed_findings.csv", negatives,
        columns=["severity", "category", "finding", "evidence"])

    summary = summary_markdown(
        results_root, output_dir, inventory, case, signal, units, unit_audit,
        checks, validity, reliability, incremental, multiseed,
        corruption_checks, practical_tables["practical"], negatives, gates,
        args.bootstrap, args.seed)
    atomic_text(output_dir / "V2_RESULTS_SUMMARY.md", summary)
    manifest = {
        "protocol": PROTOCOL,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "results_root": str(results_root),
        "output_dir": str(output_dir),
        "bootstrap": args.bootstrap,
        "seed": args.seed,
        "signal_load_scope": args.signal_scope,
        "signal_experiment_ids": (
            sorted(signal_experiment_ids)
            if signal_experiment_ids is not None else "all"),
        "unique_success_runs": int(len(inventory)),
        "case_rows": int(len(case)),
        "raw_region_rows": int(len(signal)),
        "collapsed_signal_units": int(len(units)),
        "signal_unit_audit": unit_audit,
        "failed_checks": (
            (int((checks["status"] == "fail").sum())
             if not checks.empty else 0)
            + (0 if explicit_semantics["status"] == "PASS" else 1)
            + sum(
                1 for gate in gates.values()
                if gate.get("status") != "PASS")),
        "explicit_signal_semantics_audit": explicit_semantics,
        "endpoint_gates": gates,
        "formal_reliability_rows": int(len(reliability)),
        "formal_corruption_manipulation_rows": int(len(corruption_checks)),
        "multiseed_case_comparison_rows": int(len(multiseed)),
        "negative_mixed_pending_findings": int(len(negatives)),
        "outputs": sorted(
            path.name for path in output_dir.iterdir()
            if path.is_file()),
    }
    atomic_text(
        output_dir / "v2_analysis_manifest.json",
        json.dumps(manifest, indent=2, sort_keys=True))
    print(json.dumps({
        "status": "success",
        "output_dir": str(output_dir),
        "unique_success_runs": len(inventory),
        "case_rows": len(case),
        "raw_region_rows": len(signal),
        "collapsed_signal_units": len(units),
        "failed_checks": manifest["failed_checks"],
        "findings": len(negatives),
    }, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Analyze successful TCFF V2 raw runs without mutation.")
    parser.add_argument(
        "--results_root", default=str(here / "results_tcff_rebuttal"))
    parser.add_argument(
        "--output_dir", default="",
        help="Default: <results_root>/12_v2_analysis")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--signal_scope",
        choices=("all", "reviewer_core"),
        default="all",
        help=(
            "Use 'reviewer_core' for the urgent response analysis: signal "
            "artifacts are loaded only from corrected probability and the "
            "four predeclared controlled-corruption matrices. Case metrics "
            "remain complete. The default 'all' is retained for the later "
            "full audit workflow."))
    parser.add_argument(
        "--self-test", action="store_true",
        help="Run synthetic positive and missing-field regression fixtures.")
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    if arguments.self_test:
        raise SystemExit(self_test())
    if arguments.bootstrap < 1:
        raise ValueError("--bootstrap must be positive")
    raise SystemExit(main(arguments))
