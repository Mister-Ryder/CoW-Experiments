"""Strict, read-only audit of the TCFF V2 rebuttal experiment results.

The experiment tree under ``results_root`` is treated as immutable input.  This
script only writes audit artifacts under ``12_v2_analysis`` (or ``--output_dir``).
Retries are collapsed by their canonical base run ID, while every physical
attempt remains visible in the attempt ledger.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


AUDIT_PROTOCOL = "TCFF_V2_STRICT_AUDIT_2026-07-30"
RETRY_SUFFIX = re.compile(r"(?:__retry[0-9]+)+$")
MISSING = object()

DATASETS = (
    ("crown", "TopCoW_val"),
    ("lausanne", "Lausanne"),
    ("ixi", "IXI_HH"),
)
CORRUPTIONS = (
    ("gaussian", 0.20, "channel_std_multiplier"),
    ("channel_dropout", 0.50, "channel_fraction"),
    ("translation", 4.0, "feature_voxel"),
    ("donor_mix", 0.50, "mixing_weight"),
)
TRAINED_VARIANTS = (
    "full", "residual", "u_only", "g_only", "no_aux", "additive")
EFFICIENCY_IDENTITIES = (
    "TCFF/full",
    "TCFF/product",
    "TCFF/g_only",
    "TCFF/residual",
    "TCFF/no_aux",
    "Concat/native",
    "BAGF/native",
    "DSFF/native",
    "CrossAttn/native",
    "AFF/native",
)
V2_FAMILY_ORDER = (
    "corrected_probability",
    "temperature_sensitivity",
    "constant_acceptance",
    "gate_transform",
    "fixed_policy_replay",
    "gate_dose_response",
    "gate_mean_matched",
    "product_retrain",
    "product_trained_eval_clean",
    "product_trained_eval_corruption",
    "efficiency",
)
EXPECTED_V2_COUNTS = {
    "corrected_probability": 39,
    "temperature_sensitivity": 45,
    "constant_acceptance": 60,
    "gate_transform": 30,
    "fixed_policy_replay": 105,
    "gate_dose_response": 60,
    "gate_mean_matched": 15,
    "product_retrain": 3,
    "product_trained_eval_clean": 9,
    "product_trained_eval_corruption": 36,
    "efficiency": 10,
}


@dataclass(frozen=True)
class ExpectedRun:
    scope: str
    family: str
    experiment_id: str
    base_run_id: str
    kind: str
    manifest_expected: Mapping[Tuple[str, ...], Any]
    config_expected: Mapping[Tuple[str, ...], Any]
    probability_required: bool = False
    training_seed: Optional[int] = None

    @property
    def key(self) -> Tuple[str, str]:
        return self.experiment_id, self.base_run_id


def severity_tag(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def canonical_base_run_id(run_id: str) -> str:
    """Collapse one or more launcher retry suffixes."""
    return RETRY_SUFFIX.sub("", run_id)


def nested_get(value: Mapping[str, Any], path: Sequence[str]) -> Any:
    current: Any = value
    for part in path:
        if not isinstance(current, Mapping) or part not in current:
            return MISSING
        current = current[part]
    return current


def finite_number(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def values_equal(observed: Any, expected: Any) -> bool:
    if observed is MISSING:
        return False
    if isinstance(expected, float):
        actual = finite_number(observed)
        return (
            actual is not None and
            math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-10))
    if isinstance(expected, int) and not isinstance(expected, bool):
        actual = finite_number(observed)
        return (
            actual is not None and actual.is_integer() and
            int(actual) == expected)
    return observed == expected


def display_value(value: Any) -> str:
    if value is MISSING:
        return "<missing>"
    return repr(value)


def condition(
        dataset_tag: str,
        dataset_name: str,
        corruption: str = "none",
        severity: float = 0.0,
        unit: str = "none",
        realization: int = 0) -> Dict[str, Any]:
    return {
        "dataset_tag": dataset_tag,
        "dataset_name": dataset_name,
        "corruption": corruption,
        "severity": severity,
        "unit": unit,
        "realization": realization,
    }


def condition_tag(item: Mapping[str, Any]) -> str:
    if item["corruption"] == "none":
        return f"clean_{item['dataset_tag']}"
    return (
        f"{item['corruption']}_{severity_tag(float(item['severity']))}_"
        f"r{item['realization']}_{item['dataset_tag']}")


def inference_expected(
        scope: str,
        family: str,
        experiment_id: str,
        base_run_id: str,
        item: Mapping[str, Any],
        variant: str,
        seed: int,
        gate_alpha: Optional[float] = None,
        gate_scale: Optional[float] = None,
        temperature_scale: float = 1.0,
        gate_transform: str = "identity",
        probability_required: bool = False,
        save_predictions: Optional[bool] = None,
        training_seed: Optional[int] = None) -> ExpectedRun:
    manifest = {
        ("experiment_id",): experiment_id,
        ("model_variant",): variant,
        ("dataset",): item["dataset_name"],
        ("seed",): seed,
        ("corruption", "type"): item["corruption"],
        ("corruption", "severity"): float(item["severity"]),
        ("corruption", "unit"): item["unit"],
        ("corruption", "realization"): int(item["realization"]),
        ("intervention", "gate_alpha"): gate_alpha,
        ("intervention", "temperature_scale"): temperature_scale,
        ("intervention", "gate_transform"): gate_transform,
    }
    config: Dict[Tuple[str, ...], Any] = {
        ("experiment_id",): experiment_id,
        ("model_variant",): variant,
        ("dataset_name",): item["dataset_name"],
        ("seed",): seed,
        ("corruption_type",): item["corruption"],
        ("severity",): float(item["severity"]),
        ("severity_unit",): item["unit"],
        ("realization",): int(item["realization"]),
        ("gate_alpha",): gate_alpha,
        ("temperature_scale",): temperature_scale,
        ("gate_transform",): gate_transform,
    }
    # ``gate_scale`` was added after the original V2 core runs began.  It is
    # intentionally required only when an expected family passes it here.
    if gate_scale is not None:
        manifest[("intervention", "gate_scale")] = gate_scale
        config[("gate_scale",)] = gate_scale
    if probability_required:
        config[("save_probability_signals",)] = True
    if save_predictions is not None:
        config[("save_predictions",)] = save_predictions
    return ExpectedRun(
        scope=scope,
        family=family,
        experiment_id=experiment_id,
        base_run_id=base_run_id,
        kind="inference",
        manifest_expected=manifest,
        config_expected=config,
        probability_required=probability_required,
        training_seed=training_seed,
    )


def build_v2_expected_runs() -> List[ExpectedRun]:
    expected: List[ExpectedRun] = []

    # Corrected probability interface: clean plus four corruptions on all
    # three domains, with three deterministic realizations per corruption.
    for dataset_tag, dataset_name in DATASETS:
        clean = condition(dataset_tag, dataset_name)
        expected.append(inference_expected(
            "v2", "corrected_probability", "V2_PROBABILITY_CORRECTED",
            f"v2_probability_clean_{dataset_tag}", clean, "full", 0,
            probability_required=True))
        for corruption, severity, unit in CORRUPTIONS:
            for realization in range(3):
                item = condition(
                    dataset_tag, dataset_name, corruption, severity, unit,
                    realization)
                expected.append(inference_expected(
                    "v2", "corrected_probability",
                    "V2_PROBABILITY_CORRECTED",
                    (
                        f"v2_probability_{corruption}_"
                        f"{severity_tag(severity)}_r{realization}_"
                        f"{dataset_tag}"
                    ),
                    item, "full", 0, probability_required=True))

    clean_conditions = [
        condition(dataset_tag, dataset_name)
        for dataset_tag, dataset_name in DATASETS]
    crown_shift = [
        condition(
            "crown", "TopCoW_val", corruption, severity, unit, realization)
        for corruption, severity, unit in CORRUPTIONS
        for realization in range(3)
    ]

    for scale in (0.5, 1.0, 2.0):
        for item in clean_conditions + crown_shift:
            expected.append(inference_expected(
                "v2", "temperature_sensitivity",
                "V2_TEMPERATURE_SENSITIVITY",
                f"full_temp_{severity_tag(scale)}_{condition_tag(item)}",
                item, "full", 0,
                temperature_scale=scale,
                save_predictions=(scale == 1.0)))

    alpha_shift = [
        item for item in crown_shift
        if item["corruption"] in {"gaussian", "translation", "donor_mix"}]
    for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
        for item in clean_conditions + alpha_shift:
            expected.append(inference_expected(
                "v2", "constant_acceptance", "V2_CONSTANT_ACCEPTANCE",
                f"full_alpha_{severity_tag(alpha)}_{condition_tag(item)}",
                item, "full", 0, gate_alpha=alpha,
                save_predictions=(alpha in (0.0, 1.0))))

    for transform in ("shuffle", "invert"):
        for item in clean_conditions + crown_shift:
            expected.append(inference_expected(
                "v2", "gate_transform", "V2_GATE_TRANSFORM",
                f"full_gate_{transform}_{condition_tag(item)}",
                item, "full", 0, gate_transform=transform,
                save_predictions=False))

    for variant in (
            "full", "product", "g_only", "u_only", "additive",
            "residual", "no_aux"):
        for item in clean_conditions + crown_shift:
            expected.append(inference_expected(
                "v2", "fixed_policy_replay", "V2_FIXED_POLICY_REPLAY",
                f"{variant}_policy_{condition_tag(item)}",
                item, variant, 0, gate_scale=1.0,
                temperature_scale=1.0, gate_transform="identity",
                save_predictions=False))

    for scale in (0.0, 0.25, 0.5, 0.75, 1.0):
        for item in clean_conditions + alpha_shift:
            expected.append(inference_expected(
                "v2", "gate_dose_response", "V2_GATE_DOSE_RESPONSE",
                f"full_scale_{severity_tag(scale)}_{condition_tag(item)}",
                item, "full", 0, gate_scale=scale,
                temperature_scale=1.0, gate_transform="identity",
                save_predictions=False))

    for item in clean_conditions + crown_shift:
        expected.append(inference_expected(
            "v2", "gate_mean_matched", "V2_GATE_MEAN_MATCHED",
            f"full_mean_matched_{condition_tag(item)}",
            item, "full", 0, gate_scale=1.0,
            temperature_scale=1.0, gate_transform="mean_matched",
            save_predictions=False))

    for seed in range(3):
        run_id = f"v2_product_seed{seed}"
        expected.append(ExpectedRun(
            scope="v2",
            family="product_retrain",
            experiment_id="V2_PRODUCT_RETRAIN",
            base_run_id=run_id,
            kind="training",
            manifest_expected={
                ("experiment_id",): "V2_PRODUCT_RETRAIN",
                ("dataset",): "TopCoW_train+TopCoW_val",
                ("model_variant",): "product",
                ("seed",): seed,
                ("corruption", "type"): "none",
                ("corruption", "severity"): 0.0,
                ("corruption", "unit"): "none",
            },
            config_expected={
                ("experiment_id",): "V2_PRODUCT_RETRAIN",
                ("trust_variant",): "product",
                ("seed",): seed,
                ("epochs",): 200,
                ("batch_size",): 2,
                ("training_protocol",):
                    "independent_single_gpu_fixed_schedule",
            },
            training_seed=seed,
        ))

        for dataset_tag, dataset_name in DATASETS:
            item = condition(dataset_tag, dataset_name)
            expected.append(inference_expected(
                "v2", "product_trained_eval_clean",
                "V2_PRODUCT_TRAINED_EVAL_CLEAN",
                f"v2_product_seed{seed}_clean_{dataset_tag}",
                item, "product", seed, training_seed=seed))
        for corruption, severity, unit in CORRUPTIONS:
            for realization in range(3):
                item = condition(
                    "crown", "TopCoW_val", corruption, severity, unit,
                    realization)
                expected.append(inference_expected(
                    "v2", "product_trained_eval_corruption",
                    "V2_PRODUCT_TRAINED_EVAL_CORRUPTION",
                    (
                        f"v2_product_seed{seed}_{corruption}_"
                        f"{severity_tag(severity)}_r{realization}_crown"
                    ),
                    item, "product", seed, training_seed=seed))

    return expected


def build_legacy_trained_corruption_expected() -> List[ExpectedRun]:
    expected: List[ExpectedRun] = []
    for variant in TRAINED_VARIANTS:
        for seed in range(3):
            for corruption, severity, unit in CORRUPTIONS:
                for realization in range(3):
                    item = condition(
                        "crown", "TopCoW_val", corruption, severity, unit,
                        realization)
                    expected.append(inference_expected(
                        "legacy", "legacy_p0_trained_corruption",
                        "P0_D_TRAINED_EVAL_CORRUPTION",
                        (
                            f"{variant}_seed{seed}_{corruption}_"
                            f"{severity_tag(severity)}_r{realization}_crown"
                        ),
                        item, variant, seed, training_seed=seed))
    return expected


def read_json_file(path: Path) -> Tuple[Optional[Any], Optional[str]]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def build_legacy_product_expected(
        results_root: Path) -> Tuple[List[ExpectedRun], List[str]]:
    plan_path = (
        results_root / "02_manifests" /
        "p0_product_intervention_matrix_plan.json")
    plan, error = read_json_file(plan_path)
    notes: List[str] = []
    if error is not None:
        notes.append(f"legacy product plan unavailable: {error}")
        return [], notes
    jobs = plan.get("jobs") if isinstance(plan, Mapping) else None
    if not isinstance(jobs, list):
        notes.append("legacy product plan has no jobs list")
        return [], notes
    expected: List[ExpectedRun] = []
    for index, source in enumerate(jobs):
        if not isinstance(source, Mapping):
            notes.append(f"legacy product plan job {index} is not an object")
            continue
        required = (
            "experiment_id", "run_id", "dataset_name", "corruption",
            "severity", "unit", "realization")
        missing = [key for key in required if key not in source]
        if missing:
            notes.append(
                f"legacy product plan job {index} missing {missing}")
            continue
        item = condition(
            str(source.get("dataset_tag", "")),
            str(source["dataset_name"]),
            str(source["corruption"]),
            float(source["severity"]),
            str(source["unit"]),
            int(source["realization"]),
        )
        expected.append(inference_expected(
            "legacy", "legacy_p0_product_fixed",
            str(source["experiment_id"]), str(source["run_id"]),
            item, "product", 0))
    if len(expected) != 147:
        notes.append(
            f"legacy product plan contains {len(expected)} valid jobs; "
            "the historical target was 147")
    return expected, notes


def load_yaml_file(path: Path) -> Tuple[Optional[Any], Optional[str]]:
    try:
        import yaml
    except Exception as exc:
        return None, f"PyYAML unavailable: {type(exc).__name__}: {exc}"
    try:
        with path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def canonical_json_hash(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def append_mismatch(
        errors: List[str],
        source_name: str,
        path: Sequence[str],
        observed: Any,
        expected: Any) -> None:
    errors.append(
        f"{source_name}.{'.'.join(path)}: expected "
        f"{display_value(expected)}, observed {display_value(observed)}")


def validate_expected_fields(
        source: Mapping[str, Any],
        expectations: Mapping[Tuple[str, ...], Any],
        source_name: str,
        errors: List[str],
        legacy: bool = False) -> None:
    for path, expected in expectations.items():
        observed = nested_get(source, path)
        if (
                legacy and path[:1] == ("intervention",) and
                observed is MISSING and
                expected in (None, 1.0, "identity")):
            continue
        if not values_equal(observed, expected):
            append_mismatch(errors, source_name, path, observed, expected)


def corrected_probability_intervention_compatibility(
        manifest: Mapping[str, Any],
        config: Mapping[str, Any],
        expected: ExpectedRun) -> bool:
    """Accept one narrowly defined pre-intervention-manifest schema.

    Early corrected-probability runs were produced before the three runtime
    intervention fields were copied into ``run_manifest.json``.  Their hashed
    YAML config is still authoritative, but this compatibility path is valid
    only when the entire manifest intervention object is absent and every
    expected config field (including the three explicit intervention fields)
    passes.
    """
    if expected.family != "corrected_probability":
        return False
    if nested_get(manifest, ("intervention",)) is not MISSING:
        return False
    intervention_paths = (
        ("gate_alpha",),
        ("temperature_scale",),
        ("gate_transform",),
    )
    if any(nested_get(config, path) is MISSING for path in intervention_paths):
        return False
    if any(
            not values_equal(nested_get(config, path), wanted)
            for path, wanted in expected.config_expected.items()):
        return False
    if manifest.get("config_hash") != canonical_json_hash(config):
        return False
    return True


def compare_manifest_config(
        manifest: Mapping[str, Any],
        config: Mapping[str, Any],
        errors: List[str]) -> None:
    pairs = [
        (("experiment_id",), ("experiment_id",)),
        (("run_id",), ("run_id",)),
        (("seed",), ("seed",)),
        (("dataset",), ("dataset_name",)),
        (("corruption", "type"), ("corruption_type",)),
        (("corruption", "severity"), ("severity",)),
        (("corruption", "unit"), ("severity_unit",)),
        (("corruption", "realization"), ("realization",)),
        (("intervention", "gate_alpha"), ("gate_alpha",)),
        (("intervention", "gate_scale"), ("gate_scale",)),
        (("intervention", "temperature_scale"), ("temperature_scale",)),
        (("intervention", "gate_transform"), ("gate_transform",)),
    ]
    manifest_variant = nested_get(manifest, ("model_variant",))
    config_variant = nested_get(config, ("model_variant",))
    if config_variant is MISSING:
        config_variant = nested_get(config, ("trust_variant",))
    if (
            manifest_variant is not MISSING and
            config_variant is not MISSING and
            not values_equal(config_variant, manifest_variant)):
        errors.append(
            "manifest/config model variant mismatch: "
            f"{display_value(manifest_variant)} vs "
            f"{display_value(config_variant)}")
    for manifest_path, config_path in pairs:
        left = nested_get(manifest, manifest_path)
        right = nested_get(config, config_path)
        if left is MISSING or right is MISSING:
            continue
        if not values_equal(right, left):
            errors.append(
                "manifest/config mismatch for "
                f"{'.'.join(manifest_path)}: "
                f"{display_value(left)} vs {display_value(right)}")


def read_csv_rows(path: Path) -> Tuple[List[Dict[str, str]], Optional[str]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle)), None
    except Exception as exc:
        return [], f"{type(exc).__name__}: {exc}"


def validate_inference_artifacts(
        run_dir: Path,
        actual_run_id: str,
        expected: ExpectedRun,
        status_payload: Mapping[str, Any],
        errors: List[str]) -> Dict[str, Any]:
    details: Dict[str, Any] = {}
    required = (
        "case_metrics.csv",
        "class_metrics.csv",
        "internal_signals.parquet",
        "case_failures.json",
        "aggregate_metrics.json",
        "selected_cases.json",
        "runtime.json",
    )
    for filename in required:
        artifact = run_dir / filename
        if not artifact.is_file():
            errors.append(f"missing success artifact: {filename}")
        elif artifact.stat().st_size == 0:
            errors.append(f"empty success artifact: {filename}")

    case_path = run_dir / "case_metrics.csv"
    rows: List[Dict[str, str]] = []
    if case_path.is_file():
        rows, csv_error = read_csv_rows(case_path)
        if csv_error:
            errors.append(f"case_metrics.csv unreadable: {csv_error}")
        elif not rows:
            errors.append("case_metrics.csv has no data rows")
    details["case_metric_rows"] = len(rows)

    selected_path = run_dir / "selected_cases.json"
    selected, selected_error = read_json_file(selected_path)
    if selected_path.is_file():
        if selected_error:
            errors.append(f"selected_cases.json unreadable: {selected_error}")
        elif not isinstance(selected, list) or not selected:
            errors.append("selected_cases.json must be a non-empty list")
        elif rows and len(selected) != len(rows):
            errors.append(
                f"selected case count {len(selected)} != case metric rows "
                f"{len(rows)}")

    n_success = finite_number(status_payload.get("n_success"))
    if n_success is not None and rows and int(n_success) != len(rows):
        errors.append(
            f"status.n_success={int(n_success)} != case metric rows "
            f"{len(rows)}")
    n_failed = finite_number(status_payload.get("n_failed"))
    if n_failed is not None and int(n_failed) != 0:
        errors.append(f"successful status reports n_failed={int(n_failed)}")

    failures_path = run_dir / "case_failures.json"
    failures, failures_error = read_json_file(failures_path)
    if failures_path.is_file():
        if failures_error:
            errors.append(f"case_failures.json unreadable: {failures_error}")
        elif failures != []:
            errors.append(
                "successful inference must have an empty case_failures.json")

    manifest_expected = expected.manifest_expected
    for index, row in enumerate(rows):
        row_pairs = {
            "experiment_id": manifest_expected.get(("experiment_id",)),
            "run_id": actual_run_id,
            "dataset": manifest_expected.get(("dataset",)),
            "model_variant": manifest_expected.get(("model_variant",)),
            "corruption_type": manifest_expected.get(
                ("corruption", "type")),
            "severity": manifest_expected.get(("corruption", "severity")),
            "seed": manifest_expected.get(("seed",)),
            "realization": manifest_expected.get(
                ("corruption", "realization")),
        }
        for column, wanted in row_pairs.items():
            if column not in row:
                errors.append(f"case_metrics.csv missing column {column}")
                break
            if not values_equal(row[column], wanted):
                errors.append(
                    f"case_metrics row {index} {column}: expected "
                    f"{display_value(wanted)}, observed "
                    f"{display_value(row[column])}")
                break

    if expected.probability_required:
        probability_errors: List[str] = []
        observed_values: List[float] = []
        if not rows:
            probability_errors.append("probability run has no case rows")
        for index, row in enumerate(rows):
            if "probability_sum_max_abs_error" not in row:
                probability_errors.append(
                    "case_metrics.csv missing "
                    "probability_sum_max_abs_error")
                break
            value = finite_number(row["probability_sum_max_abs_error"])
            if value is None:
                probability_errors.append(
                    f"probability row {index} has non-finite sum error")
            else:
                observed_values.append(value)
                if abs(value) > 1e-4:
                    probability_errors.append(
                        f"probability row {index} sum error {value:.8g} "
                        "exceeds 1e-4")
        errors.extend(probability_errors)
        details["probability_rows_checked"] = len(observed_values)
        details["probability_sum_max_abs_error"] = (
            max(map(abs, observed_values)) if observed_values else None)

    save_predictions = expected.config_expected.get(("save_predictions",))
    if save_predictions is True and rows:
        prediction_dir = run_dir / "predictions"
        prediction_count = (
            len(list(prediction_dir.glob("*.npz")))
            if prediction_dir.is_dir() else 0)
        details["prediction_files"] = prediction_count
        if prediction_count != len(rows):
            errors.append(
                f"save_predictions=True but found {prediction_count} "
                f"prediction files for {len(rows)} case rows")
    return details


def validate_training_artifacts(
        run_dir: Path,
        manifest: Mapping[str, Any],
        errors: List[str],
        skip_checkpoint_hash: bool) -> Dict[str, Any]:
    details: Dict[str, Any] = {}
    checkpoint = run_dir / "training_output" / "model" / "best_model.pth"
    runtime_path = run_dir / "runtime.json"
    if not checkpoint.is_file():
        errors.append(
            "missing success artifact: training_output/model/best_model.pth")
    if not runtime_path.is_file():
        errors.append("missing success artifact: runtime.json")
    runtime, runtime_error = read_json_file(runtime_path)
    if runtime_path.is_file() and runtime_error:
        errors.append(f"runtime.json unreadable: {runtime_error}")
    if isinstance(runtime, Mapping):
        exit_code = runtime.get("exit_code")
        if exit_code != 0:
            errors.append(
                f"training runtime exit_code is {display_value(exit_code)}")
        runtime_hash = runtime.get("final_checkpoint_hash")
        manifest_hash = manifest.get("trained_checkpoint_hash")
        if runtime_hash != manifest_hash:
            errors.append(
                "runtime/manifest trained checkpoint hash mismatch")
    expected_hash = manifest.get("trained_checkpoint_hash")
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        errors.append("manifest.trained_checkpoint_hash is missing or invalid")
    elif checkpoint.is_file() and not skip_checkpoint_hash:
        observed_hash = sha256_file(checkpoint)
        details["checkpoint_hash_checked"] = True
        details["checkpoint_sha256"] = observed_hash
        if observed_hash != expected_hash:
            errors.append(
                "best_model.pth SHA256 does not match "
                "manifest.trained_checkpoint_hash")
    else:
        details["checkpoint_hash_checked"] = False
        details["checkpoint_sha256"] = expected_hash
    return details


def load_attempt(
        run_dir: Path,
        expected: Optional[ExpectedRun],
        stale_seconds: int,
        skip_checkpoint_hash: bool,
        misplaced_expected: Optional[ExpectedRun] = None) -> Dict[str, Any]:
    experiment_id = run_dir.parent.name
    actual_run_id = run_dir.name
    canonical_id = canonical_base_run_id(actual_run_id)
    errors: List[str] = []
    warnings: List[str] = []

    manifest_path = run_dir / "run_manifest.json"
    status_path = run_dir / "status.json"
    config_path = run_dir / "run_config.yaml"
    manifest, manifest_error = read_json_file(manifest_path)
    status, status_error = read_json_file(status_path)
    config, config_error = load_yaml_file(config_path)

    if manifest_error:
        errors.append(f"run_manifest.json unreadable: {manifest_error}")
        manifest = {}
    if status_error:
        errors.append(f"status.json unreadable: {status_error}")
        status = {}
    if config_error:
        message = f"run_config.yaml unreadable: {config_error}"
        if expected is not None and expected.scope == "legacy":
            warnings.append(message)
        else:
            errors.append(message)
        config = {}
    if not isinstance(manifest, Mapping):
        errors.append("run_manifest.json root is not an object")
        manifest = {}
    if not isinstance(status, Mapping):
        errors.append("status.json root is not an object")
        status = {}
    if not isinstance(config, Mapping):
        errors.append("run_config.yaml root is not an object")
        config = {}

    manifest_run = manifest.get("run_id", MISSING)
    manifest_experiment = manifest.get("experiment_id", MISSING)
    if manifest_run != actual_run_id:
        errors.append(
            f"manifest.run_id/path mismatch: "
            f"{display_value(manifest_run)} vs {actual_run_id!r}")
    if manifest_experiment != experiment_id:
        errors.append(
            f"manifest.experiment_id/path mismatch: "
            f"{display_value(manifest_experiment)} vs {experiment_id!r}")
    config_run = config.get("run_id", MISSING)
    config_experiment = config.get("experiment_id", MISSING)
    if config and config_run != actual_run_id:
        errors.append(
            f"config.run_id/path mismatch: "
            f"{display_value(config_run)} vs {actual_run_id!r}")
    if config and config_experiment != experiment_id:
        errors.append(
            f"config.experiment_id/path mismatch: "
            f"{display_value(config_experiment)} vs {experiment_id!r}")

    status_value = status.get("status")
    manifest_status = manifest.get("status")
    if status_value != manifest_status:
        errors.append(
            f"manifest/status state mismatch: "
            f"{display_value(manifest_status)} vs "
            f"{display_value(status_value)}")
    if config:
        config_hash = canonical_json_hash(config)
        if manifest.get("config_hash") != config_hash:
            errors.append(
                "manifest.config_hash does not match canonical "
                "run_config.yaml")
        compare_manifest_config(manifest, config, errors)

    if misplaced_expected is not None:
        errors.append(
            "canonical V2 run is in the wrong experiment: expected "
            f"{misplaced_expected.experiment_id}, observed {experiment_id}")

    artifact_details: Dict[str, Any] = {}
    if expected is not None:
        probability_manifest_compatibility = (
            corrected_probability_intervention_compatibility(
                manifest, config, expected))
        manifest_expectations = expected.manifest_expected
        if probability_manifest_compatibility:
            manifest_expectations = {
                path: wanted
                for path, wanted in expected.manifest_expected.items()
                if path[:1] != ("intervention",)
            }
            warnings.append(
                "backward-compatible corrected-probability manifest: "
                "intervention object absent; explicit hashed run_config.yaml "
                "fields gate_alpha=None, temperature_scale=1.0 and "
                "gate_transform=identity were validated")
        validate_expected_fields(
            manifest, manifest_expectations, "manifest", errors,
            legacy=expected.scope == "legacy")
        if config:
            validate_expected_fields(
                config, expected.config_expected, "config", errors,
                legacy=expected.scope == "legacy")
        if status_value == "success":
            if expected.kind == "inference":
                artifact_details = validate_inference_artifacts(
                    run_dir, actual_run_id, expected, status, errors)
            else:
                artifact_details = validate_training_artifacts(
                    run_dir, manifest, errors, skip_checkpoint_hash)

    status_age_seconds: Optional[float] = None
    stale_running = False
    if status_path.exists():
        status_age_seconds = max(0.0, time.time() - status_path.stat().st_mtime)
        stale_running = (
            status_value == "running" and
            status_age_seconds >= stale_seconds)
    if status_value == "running":
        warnings.append(
            "abandoned running (stale)"
            if stale_running else "active running")
    elif status_value == "partial":
        warnings.append("partial attempt")
    elif status_value == "failed":
        warnings.append("failed attempt")
    elif status_value not in {"success", "running", "partial", "failed"}:
        errors.append(
            f"unknown or missing status {display_value(status_value)}")

    return {
        "scope": expected.scope if expected else (
            misplaced_expected.scope if misplaced_expected else "v2_extra"),
        "family": expected.family if expected else (
            misplaced_expected.family if misplaced_expected
            else "unexpected_v2_run"),
        "experiment_id": experiment_id,
        "actual_run_id": actual_run_id,
        "canonical_base_run_id": canonical_id,
        "expected_experiment_id": (
            expected.experiment_id if expected else
            misplaced_expected.experiment_id
            if misplaced_expected else None),
        "path": str(run_dir),
        "status": status_value,
        "manifest_status": manifest_status,
        "stale_running": stale_running,
        "status_age_seconds": status_age_seconds,
        "validation_ok": not errors,
        "validation_errors": errors,
        "warnings": warnings,
        "checkpoint_hash": manifest.get("checkpoint_hash"),
        "trained_checkpoint_hash": manifest.get(
            "trained_checkpoint_hash"),
        "artifact_details": artifact_details,
        "_manifest": dict(manifest),
    }


def choose_unit_status(
        expected: ExpectedRun,
        attempts: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    successes = [item for item in attempts if item.get("status") == "success"]
    valid_successes = [
        item for item in successes if item.get("validation_ok")]
    failed_attempts = [
        item for item in attempts if item.get("status") == "failed"]
    partial_attempts = [
        item for item in attempts if item.get("status") == "partial"]
    stale_attempts = [
        item for item in attempts if item.get("stale_running")]
    active_attempts = [
        item for item in attempts
        if item.get("status") == "running" and
        not item.get("stale_running")]

    issues: List[str] = []
    warnings: List[str] = []
    selected: Optional[Mapping[str, Any]] = None
    if len(successes) > 1:
        status = "conflict_multiple_success"
        issues.append(
            "same canonical base_run_id has multiple success attempts: " +
            ", ".join(str(item["actual_run_id"]) for item in successes))
        if valid_successes:
            selected = valid_successes[-1]
        else:
            selected = successes[-1]
    elif len(successes) == 1:
        selected = successes[0]
        if selected.get("validation_ok"):
            status = "success"
        else:
            status = "invalid_success"
            issues.extend(selected.get("validation_errors", []))
    elif not attempts:
        status = "missing"
        issues.append("no run directory found")
    elif stale_attempts:
        status = "abandoned_running_stale"
        selected = stale_attempts[-1]
        issues.append("no success; at least one stale running attempt")
    elif active_attempts:
        status = "running"
        selected = active_attempts[-1]
        issues.append("no success; run is still active")
    elif partial_attempts:
        status = "partial"
        selected = partial_attempts[-1]
        issues.append("no success; latest available attempt is partial")
    elif failed_attempts:
        status = "failed"
        selected = failed_attempts[-1]
        issues.append("no success; all terminal attempts failed")
    else:
        status = "invalid_attempt"
        selected = attempts[-1]
        issues.append("no usable attempt")

    if status == "success":
        if failed_attempts:
            warnings.append(
                f"{len(failed_attempts)} failed attempt(s) resolved by retry")
        if partial_attempts:
            warnings.append(
                f"{len(partial_attempts)} partial attempt(s) resolved by retry")
        if stale_attempts:
            warnings.append(
                f"{len(stale_attempts)} stale running attempt(s) coexist "
                "with the selected success")
        if active_attempts:
            warnings.append(
                f"{len(active_attempts)} active running attempt(s) coexist "
                "with the selected success")

    return {
        "scope": expected.scope,
        "family": expected.family,
        "experiment_id": expected.experiment_id,
        "canonical_base_run_id": expected.base_run_id,
        "kind": expected.kind,
        "status": status,
        "complete": status == "success",
        "attempt_count": len(attempts),
        "success_attempt_count": len(successes),
        "selected_actual_run_id": (
            selected.get("actual_run_id") if selected else None),
        "selected_path": selected.get("path") if selected else None,
        "issues": issues,
        "warnings": warnings,
        "probability_sum_max_abs_error": (
            (selected.get("artifact_details") or {}).get(
                "probability_sum_max_abs_error")
            if selected else None),
        "checkpoint_hash": (
            selected.get("checkpoint_hash") if selected else None),
        "trained_checkpoint_hash": (
            selected.get("trained_checkpoint_hash") if selected else None),
        "_selected_attempt": selected,
        "_training_seed": expected.training_seed,
    }


def validate_product_eval_checkpoint_lineage(
        units: List[Dict[str, Any]]) -> None:
    training_hashes: Dict[int, str] = {}
    for unit in units:
        if (
                unit["family"] == "product_retrain" and
                unit["status"] == "success"):
            seed = unit.get("_training_seed")
            checkpoint_hash = unit.get("trained_checkpoint_hash")
            if seed is not None and isinstance(checkpoint_hash, str):
                training_hashes[int(seed)] = checkpoint_hash
    for unit in units:
        if unit["family"] not in {
                "product_trained_eval_clean",
                "product_trained_eval_corruption"}:
            continue
        seed = unit.get("_training_seed")
        trained_hash = training_hashes.get(int(seed)) if seed is not None else None
        eval_hash = unit.get("checkpoint_hash")
        if unit["status"] != "success":
            continue
        if trained_hash is None:
            unit["status"] = "invalid_success"
            unit["complete"] = False
            unit["issues"].append(
                f"successful evaluation has no audited successful product "
                f"training checkpoint for seed {seed}")
        elif eval_hash != trained_hash:
            unit["status"] = "invalid_success"
            unit["complete"] = False
            unit["issues"].append(
                "evaluation checkpoint_hash does not match the audited "
                f"product training checkpoint for seed {seed}")


def validate_efficiency(results_root: Path) -> Tuple[
        List[Dict[str, Any]], Dict[str, Any]]:
    efficiency_dir = results_root / "09_efficiency"
    summary_path = efficiency_dir / "v2_efficiency_summary.csv"
    raw_path = efficiency_dir / "v2_efficiency_raw_repeats.csv"
    manifest_path = efficiency_dir / "v2_efficiency_manifest.json"
    global_errors: List[str] = []
    global_warnings: List[str] = []

    manifest, manifest_error = read_json_file(manifest_path)
    if manifest_error:
        global_errors.append(
            f"v2_efficiency_manifest.json unreadable: {manifest_error}")
        manifest = {}
    if not isinstance(manifest, Mapping):
        global_errors.append("efficiency manifest root is not an object")
        manifest = {}

    summary_rows, summary_error = read_csv_rows(summary_path)
    if summary_error:
        global_errors.append(
            f"v2_efficiency_summary.csv unreadable: {summary_error}")
    raw_rows, raw_error = read_csv_rows(raw_path)
    if raw_error:
        global_errors.append(
            f"v2_efficiency_raw_repeats.csv unreadable: {raw_error}")

    if manifest:
        if manifest.get("protocol") != (
                "TCFF_V2_ISOLATED_EFFICIENCY_2026-07-30"):
            global_errors.append("efficiency manifest protocol mismatch")
        if finite_number(manifest.get("warmup")) is None or int(
                float(manifest.get("warmup", 0))) < 10:
            global_errors.append("efficiency warmup must be >=10")
        if finite_number(manifest.get("repeats")) is None or int(
                float(manifest.get("repeats", 0))) < 30:
            global_errors.append("efficiency repeats must be >=30")
        if manifest.get("batch_size") != 1:
            global_errors.append("efficiency batch_size must be 1")
        if manifest.get("precision") != "float32":
            global_errors.append("efficiency precision must be float32")
        if manifest.get("input_shape") != [1, 1, 64, 160, 160]:
            global_errors.append(
                "efficiency input_shape must be [1, 1, 64, 160, 160]")
        model_entries = manifest.get("models")
        model_identities = (
            [item.get("identity") for item in model_entries
             if isinstance(item, Mapping)]
            if isinstance(model_entries, list) else [])
        if (
                len(model_identities) != len(set(model_identities)) or
                set(model_identities) != set(EFFICIENCY_IDENTITIES)):
            global_errors.append(
                "efficiency manifest model identities are not the exact "
                "10-identity protocol")
        if isinstance(model_entries, list):
            for item in model_entries:
                if not isinstance(item, Mapping):
                    continue
                checkpoint_hash = item.get("checkpoint_hash")
                if (
                        not isinstance(checkpoint_hash, str) or
                        len(checkpoint_hash) != 64):
                    global_errors.append(
                        "efficiency manifest contains a missing/invalid "
                        "checkpoint hash")
                    break
        for snapshot_name in (
                "gpu_compute_processes_before",
                "gpu_compute_processes_after"):
            snapshot = manifest.get(snapshot_name)
            if not isinstance(snapshot, list):
                global_errors.append(
                    f"efficiency manifest missing {snapshot_name}")
                continue
            pids = {
                item.get("pid") for item in snapshot
                if isinstance(item, Mapping) and item.get("pid") is not None}
            if len(pids) > 1:
                global_errors.append(
                    f"{snapshot_name} records more than one compute PID")
        process_pids = {
            item.get("pid")
            for snapshot_name in (
                "gpu_compute_processes_before",
                "gpu_compute_processes_after")
            for item in (
                manifest.get(snapshot_name)
                if isinstance(manifest.get(snapshot_name), list) else [])
            if isinstance(item, Mapping) and item.get("pid") is not None
        }
        if len(process_pids) > 1:
            global_errors.append(
                "efficiency before/after snapshots do not refer to one "
                "isolated benchmark process")

    by_identity: Dict[str, List[Dict[str, str]]] = {}
    for row in summary_rows:
        identity = row.get("identity", "")
        by_identity.setdefault(identity, []).append(row)
    unexpected_summary = sorted(
        set(by_identity) - set(EFFICIENCY_IDENTITIES))
    if unexpected_summary:
        global_errors.append(
            "unexpected efficiency summary identities: " +
            ", ".join(unexpected_summary))

    raw_counts: Dict[Tuple[str, str], int] = {}
    raw_repeat_ids: Dict[Tuple[str, str], set] = {}
    for row in raw_rows:
        identity = row.get("identity", "")
        scope = row.get("scope", "")
        key = identity, scope
        raw_counts[key] = raw_counts.get(key, 0) + 1
        raw_repeat_ids.setdefault(key, set()).add(row.get("repeat"))
        latency = finite_number(row.get("latency_ms"))
        if latency is None or latency <= 0:
            global_errors.append(
                f"non-positive/non-finite raw latency for {identity}/{scope}")
            break
    expected_repeats = finite_number(manifest.get("repeats")) if manifest else None
    minimum_repeats = max(30, int(expected_repeats or 30))
    unexpected_raw_keys = sorted(
        key for key in raw_counts
        if key[0] not in EFFICIENCY_IDENTITIES or
        key[1] not in {"end_to_end", "fusion_only_three_levels"})
    if unexpected_raw_keys:
        global_errors.append(
            "unexpected identity/scope pairs in raw efficiency CSV: " +
            ", ".join(f"{identity}/{scope}" for identity, scope
                      in unexpected_raw_keys))

    units: List[Dict[str, Any]] = []
    for identity in EFFICIENCY_IDENTITIES:
        errors = list(global_errors)
        rows = by_identity.get(identity, [])
        selected = rows[0] if len(rows) == 1 else None
        if not rows:
            status = "missing"
            errors.append("identity missing from efficiency summary")
        elif len(rows) > 1:
            status = "conflict_multiple_success"
            errors.append("identity occurs multiple times in efficiency summary")
        else:
            required_text = {
                "identity": identity,
                "precision": "float32",
            }
            expected_method, expected_variant = identity.split("/", 1)
            required_text.update({
                "method": expected_method,
                "variant": expected_variant,
                "input_shape": "1x1x64x160x160",
                "model_parameter_dtypes": "float32",
            })
            for field, wanted in required_text.items():
                if selected.get(field) != wanted:
                    errors.append(
                        f"summary {field}: expected {wanted!r}, observed "
                        f"{selected.get(field)!r}")
            batch_size = finite_number(selected.get("batch_size"))
            if batch_size != 1:
                errors.append(
                    f"summary batch_size must be 1, observed "
                    f"{selected.get('batch_size')!r}")
            for field, minimum in (
                    ("end_to_end_repeats", 30),
                    ("fusion_only_repeats", 30),
                    ("total_parameters", 1),
                    ("active_parameters", 1),
                    ("end_to_end_latency_median_ms", 0),
                    ("end_to_end_latency_p95_ms", 0),
                    ("fusion_only_latency_median_ms", 0),
                    ("fusion_only_latency_p95_ms", 0)):
                number = finite_number(selected.get(field))
                if number is None or (
                        number < minimum if minimum > 0 else number <= minimum):
                    errors.append(
                        f"summary {field} is missing, non-finite, or below "
                        f"the required bound {minimum}")
            total_parameters = finite_number(
                selected.get("total_parameters"))
            active_parameters = finite_number(
                selected.get("active_parameters"))
            if (
                    total_parameters is not None and
                    active_parameters is not None and
                    active_parameters > total_parameters):
                errors.append(
                    "summary active_parameters exceeds total_parameters")
            for prefix in ("end_to_end", "fusion_only"):
                median = finite_number(
                    selected.get(f"{prefix}_latency_median_ms"))
                p95 = finite_number(
                    selected.get(f"{prefix}_latency_p95_ms"))
                if (
                        median is not None and p95 is not None and
                        p95 < median):
                    errors.append(
                        f"summary {prefix} p95 latency is below median")
            for scope_name in (
                    "end_to_end", "fusion_only_three_levels"):
                key = identity, scope_name
                if raw_counts.get(key, 0) != minimum_repeats:
                    errors.append(
                        f"raw repeats for {scope_name} are "
                        f"{raw_counts.get(key, 0)}; expected exactly "
                        f"{minimum_repeats}")
                if len(raw_repeat_ids.get(key, set())) != raw_counts.get(
                        key, 0):
                    errors.append(
                        f"raw repeat IDs are not unique for {scope_name}")
            status = "success" if not errors else "invalid_success"
        units.append({
            "scope": "v2",
            "family": "efficiency",
            "experiment_id": "V2_ISOLATED_EFFICIENCY",
            "canonical_base_run_id": identity,
            "kind": "efficiency",
            "status": status,
            "complete": status == "success",
            "attempt_count": len(rows),
            "success_attempt_count": 1 if len(rows) == 1 else len(rows),
            "selected_actual_run_id": identity if selected else None,
            "selected_path": str(summary_path) if selected else None,
            "issues": errors,
            "warnings": list(global_warnings),
            "probability_sum_max_abs_error": None,
            "checkpoint_hash": None,
            "trained_checkpoint_hash": None,
        })
    return units, {
        "summary_path": str(summary_path),
        "raw_path": str(raw_path),
        "manifest_path": str(manifest_path),
        "global_errors": global_errors,
        "global_warnings": global_warnings,
        "summary_rows": len(summary_rows),
        "raw_rows": len(raw_rows),
    }


def summarize_units(
        units: Sequence[Mapping[str, Any]],
        family_order: Sequence[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for family in family_order:
        selected = [unit for unit in units if unit["family"] == family]
        counts: Dict[str, int] = {}
        for unit in selected:
            status = str(unit["status"])
            counts[status] = counts.get(status, 0) + 1
        rows.append({
            "family": family,
            "expected": len(selected),
            "success": counts.get("success", 0),
            "incomplete_or_invalid": (
                len(selected) - counts.get("success", 0)),
            "complete": bool(selected) and all(
                unit["status"] == "success" for unit in selected),
            "status_counts": counts,
        })
    return rows


def public_unit(unit: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        key: value for key, value in unit.items()
        if not key.startswith("_")
    }


def public_attempt(attempt: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        key: value for key, value in attempt.items()
        if not key.startswith("_")
    }


def flatten_for_csv(row: Mapping[str, Any]) -> Dict[str, Any]:
    flattened: Dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, (dict, list, tuple)):
            flattened[key] = json.dumps(
                value, sort_keys=True, ensure_ascii=False)
        elif value is None:
            flattened[key] = ""
        else:
            flattened[key] = value
    return flattened


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(
        path,
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    materialized = [flatten_for_csv(row) for row in rows]
    fieldnames: List[str] = []
    for row in materialized:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(materialized)
    temporary.replace(path)


def markdown_report(audit: Mapping[str, Any]) -> str:
    lines = [
        "# TCFF V2 严格只读审计",
        "",
        f"- 审计协议：`{audit['protocol']}`",
        f"- 生成时间：`{audit['created_at']}`",
        f"- results root：`{audit['results_root']}`",
        f"- strict 结论：**{audit['strict_verdict']}**",
        (
            f"- V2 完成：**{audit['v2_totals']['success']} / "
            f"{audit['v2_totals']['expected']}**"
        ),
        (
            "- retry 口径：按去除 `__retryN` 后的 canonical "
            "`base_run_id` 计数；物理尝试仍保留在 attempts ledger。"
        ),
        "",
        "## V2 分项",
        "",
        "| family | expected | success | incomplete / invalid | complete |",
        "|---|---:|---:|---:|:---:|",
    ]
    for row in audit["v2_family_summary"]:
        lines.append(
            f"| {row['family']} | {row['expected']} | {row['success']} | "
            f"{row['incomplete_or_invalid']} | "
            f"{'yes' if row['complete'] else 'no'} |")

    lines.extend([
        "",
        "## 严格失败原因",
        "",
    ])
    strict_issues = audit["strict_issues"]
    if strict_issues:
        lines.extend(f"- {issue}" for issue in strict_issues)
    else:
        lines.append("- 无。")

    lines.extend([
        "",
        "## 非成功 V2 单元",
        "",
        "| family | canonical base_run_id | status | selected attempt | issue |",
        "|---|---|---|---|---|",
    ])
    incomplete = [
        unit for unit in audit["v2_units"]
        if unit["status"] != "success"]
    if incomplete:
        for unit in incomplete:
            issue = "; ".join(unit["issues"]).replace("|", "\\|")
            lines.append(
                f"| {unit['family']} | `{unit['canonical_base_run_id']}` | "
                f"{unit['status']} | "
                f"`{unit.get('selected_actual_run_id') or ''}` | "
                f"{issue} |")
    else:
        lines.append("| — | — | — | — | 全部通过 |")

    lines.extend([
        "",
        "## Legacy P0（不进入 V2 denominator）",
        "",
        "| family | expected from protocol/plan | success | "
        "incomplete / invalid |",
        "|---|---:|---:|---:|",
    ])
    for row in audit["legacy_family_summary"]:
        lines.append(
            f"| {row['family']} | {row['expected']} | {row['success']} | "
            f"{row['incomplete_or_invalid']} |")
    for note in audit["legacy_notes"]:
        lines.append(f"- {note}")

    lines.extend([
        "",
        "## 审计文件",
        "",
        "- `v2_audit.json`：完整机器可读结论与所有问题。",
        "- `v2_audit_units.csv`：412 个正式 V2 canonical 单元。",
        "- `v2_audit_attempts.csv`：物理 run/retry 尝试及其状态。",
        "- `v2_audit_summary.csv`：V2 分项汇总。",
        "- `legacy_p0_audit_units.csv`：独立的旧 P0 审计，不混入 V2。",
        "",
    ])
    return "\n".join(lines)


def assert_expected_counts(expected: Sequence[ExpectedRun]) -> None:
    counts: Dict[str, int] = {}
    keys = set()
    for item in expected:
        counts[item.family] = counts.get(item.family, 0) + 1
        if item.key in keys:
            raise RuntimeError(f"duplicate expected run key: {item.key}")
        keys.add(item.key)
    for family, wanted in EXPECTED_V2_COUNTS.items():
        if family == "efficiency":
            continue
        observed = counts.get(family, 0)
        if observed != wanted:
            raise RuntimeError(
                f"internal expected-count error for {family}: "
                f"{observed} != {wanted}")
    if len(EFFICIENCY_IDENTITIES) != EXPECTED_V2_COUNTS["efficiency"]:
        raise RuntimeError("internal efficiency identity-count error")


def main(args: argparse.Namespace) -> int:
    results_root = Path(args.results_root).resolve()
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir else results_root / "12_v2_analysis")
    raw_root = results_root / "03_raw_runs"

    v2_expected = build_v2_expected_runs()
    assert_expected_counts(v2_expected)
    legacy_product, legacy_notes = build_legacy_product_expected(results_root)
    legacy_expected = (
        build_legacy_trained_corruption_expected() + legacy_product)
    all_expected = v2_expected + legacy_expected
    expected_by_key = {item.key: item for item in all_expected}
    v2_by_base: Dict[str, List[ExpectedRun]] = {}
    for item in v2_expected:
        v2_by_base.setdefault(item.base_run_id, []).append(item)
    v2_experiments = {item.experiment_id for item in v2_expected}
    legacy_experiments = {item.experiment_id for item in legacy_expected}

    attempts: List[Dict[str, Any]] = []
    unexpected_v2_attempts: List[Dict[str, Any]] = []
    if raw_root.is_dir():
        for experiment_dir in sorted(raw_root.iterdir()):
            if not experiment_dir.is_dir():
                continue
            experiment_id = experiment_dir.name
            for run_dir in sorted(experiment_dir.iterdir()):
                if not run_dir.is_dir():
                    continue
                canonical_id = canonical_base_run_id(run_dir.name)
                expected = expected_by_key.get(
                    (experiment_id, canonical_id))
                misplaced: Optional[ExpectedRun] = None
                if expected is None:
                    candidates = v2_by_base.get(canonical_id, [])
                    if len(candidates) == 1:
                        misplaced = candidates[0]
                relevant = (
                    expected is not None or misplaced is not None or
                    experiment_id in v2_experiments or
                    experiment_id in legacy_experiments)
                if not relevant:
                    continue
                attempt = load_attempt(
                    run_dir, expected, args.stale_seconds,
                    args.skip_checkpoint_hash, misplaced_expected=misplaced)
                if (
                        expected is None and misplaced is None and
                        experiment_id in v2_experiments):
                    attempt["validation_ok"] = False
                    attempt["validation_errors"].append(
                        "unexpected canonical run in formal V2 experiment")
                    unexpected_v2_attempts.append(attempt)
                elif misplaced is not None:
                    unexpected_v2_attempts.append(attempt)
                attempts.append(attempt)

    attempts_by_key: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for attempt in attempts:
        key = (
            str(attempt["experiment_id"]),
            str(attempt["canonical_base_run_id"]),
        )
        attempts_by_key.setdefault(key, []).append(attempt)

    v2_units = [
        choose_unit_status(item, attempts_by_key.get(item.key, []))
        for item in v2_expected]
    validate_product_eval_checkpoint_lineage(v2_units)
    efficiency_units, efficiency_audit = validate_efficiency(results_root)
    v2_units.extend(efficiency_units)

    legacy_units = [
        choose_unit_status(item, attempts_by_key.get(item.key, []))
        for item in legacy_expected]

    v2_family_summary = summarize_units(v2_units, V2_FAMILY_ORDER)
    legacy_family_order = (
        "legacy_p0_trained_corruption",
        "legacy_p0_product_fixed",
    )
    legacy_family_summary = summarize_units(
        legacy_units, legacy_family_order)
    v2_success = sum(unit["status"] == "success" for unit in v2_units)
    v2_expected_total = sum(EXPECTED_V2_COUNTS.values())
    if len(v2_units) != v2_expected_total:
        raise RuntimeError(
            f"internal V2 unit denominator error: {len(v2_units)} != "
            f"{v2_expected_total}")

    strict_issues: List[str] = []
    for row in v2_family_summary:
        if not row["complete"]:
            strict_issues.append(
                f"{row['family']}: {row['success']}/{row['expected']} "
                "canonical units passed")
    for attempt in unexpected_v2_attempts:
        strict_issues.append(
            "unexpected/misplaced V2 attempt: "
            f"{attempt['experiment_id']}/{attempt['actual_run_id']}")
    strict_verdict = "PASS" if not strict_issues else "FAIL"

    created_at = dt.datetime.now(dt.timezone.utc).isoformat()
    audit = {
        "protocol": AUDIT_PROTOCOL,
        "created_at": created_at,
        "results_root": str(results_root),
        "output_dir": str(output_dir),
        "strict_requested": bool(args.strict),
        "strict_verdict": strict_verdict,
        "strict_issues": strict_issues,
        "stale_seconds": args.stale_seconds,
        "checkpoint_hash_validation_skipped": bool(
            args.skip_checkpoint_hash),
        "canonicalization": {
            "retry_suffix_regex": RETRY_SUFFIX.pattern,
            "rule": (
                "strip one or more terminal __retryN suffixes; count the "
                "canonical base run once"),
            "multiple_success_policy": (
                "more than one success attempt for a canonical run is a "
                "conflict, even if both are semantically valid"),
        },
        "probability_invariant": {
            "required_column": "probability_sum_max_abs_error",
            "threshold": 1e-4,
            "policy": "every case row in every corrected probability run",
        },
        "v2_expected_counts": EXPECTED_V2_COUNTS,
        "v2_totals": {
            "expected": v2_expected_total,
            "success": v2_success,
            "incomplete_or_invalid": v2_expected_total - v2_success,
            "unexpected_or_misplaced_attempts": len(
                unexpected_v2_attempts),
        },
        "v2_family_summary": v2_family_summary,
        "v2_units": [public_unit(item) for item in v2_units],
        "attempts": [public_attempt(item) for item in attempts],
        "unexpected_or_misplaced_v2_attempts": [
            public_attempt(item) for item in unexpected_v2_attempts],
        "efficiency_audit": efficiency_audit,
        "legacy_denominator_policy": (
            "Legacy P0 units are audited separately and never enter the "
            "412-unit V2 denominator."),
        "legacy_notes": legacy_notes,
        "legacy_family_summary": legacy_family_summary,
        "legacy_units": [public_unit(item) for item in legacy_units],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(output_dir / "v2_audit.json", audit)
    atomic_csv(
        output_dir / "v2_audit_units.csv",
        [public_unit(item) for item in v2_units])
    atomic_csv(
        output_dir / "v2_audit_attempts.csv",
        [public_attempt(item) for item in attempts])
    atomic_csv(output_dir / "v2_audit_summary.csv", v2_family_summary)
    atomic_csv(
        output_dir / "legacy_p0_audit_units.csv",
        [public_unit(item) for item in legacy_units])
    atomic_text(
        output_dir / "V2_AUDIT_REPORT.md", markdown_report(audit))

    print(json.dumps({
        "protocol": AUDIT_PROTOCOL,
        "strict_verdict": strict_verdict,
        "v2_success": v2_success,
        "v2_expected": v2_expected_total,
        "unexpected_or_misplaced_v2_attempts": len(
            unexpected_v2_attempts),
        "output_dir": str(output_dir),
    }, sort_keys=True))
    if args.strict and strict_verdict != "PASS":
        return 2
    return 0


def build_parser() -> argparse.ArgumentParser:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Audit TCFF V2 results without modifying run directories."))
    parser.add_argument(
        "--results_root",
        default=str(here / "results_tcff_rebuttal"))
    parser.add_argument(
        "--output_dir",
        help=(
            "Audit output directory; defaults to "
            "<results_root>/12_v2_analysis"))
    parser.add_argument(
        "--stale_seconds",
        type=int,
        default=900,
        help="Age after which a running status is classified as abandoned")
    parser.add_argument(
        "--skip_checkpoint_hash",
        action="store_true",
        help=(
            "Skip reading product best_model.pth bytes; manifest/runtime "
            "hash consistency is still checked"))
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return exit code 2 unless all 412 V2 units pass")
    return parser


if __name__ == "__main__":
    try:
        raise SystemExit(main(build_parser().parse_args()))
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        print(
            json.dumps({
                "protocol": AUDIT_PROTOCOL,
                "fatal_error": f"{type(exc).__name__}: {exc}",
            }, sort_keys=True),
            file=sys.stderr)
        raise
