"""Strict equivalence audit for the TCFF probability-interface repair.

The historical ``P0_B_PROBABILITY_SIGNALS`` runs are paired with the
``V2_PROBABILITY_CORRECTED`` reruns by 39 predeclared canonical conditions.
The audit establishes two separate evidence tracks:

1. hard-output SHA/metric agreement is reported descriptively and becomes a
   strict invariant only when both sides share the same commit and GPU model;
2. only the corrected runs carry per-case evidence that their probability
   tensors are finite, normalized, and bounded.

Absence of the latter evidence makes the historical soft-probability summaries
inadmissible for rebuttal claims.  It does *not* by itself assert that a
particular historical numeric value is wrong, nor does it invalidate historical
hard predictions or hard metrics.

The results tree is read-only input.  JSON, CSV, and Markdown audit artifacts
are written atomically under ``RESULTS_ROOT/12_v2_analysis``.
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
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


PROTOCOL = "TCFF_PROBABILITY_REPAIR_AUDIT_V1_2026-07-30"
OLD_EXPERIMENT = "P0_B_PROBABILITY_SIGNALS"
NEW_EXPERIMENT = "V2_PROBABILITY_CORRECTED"
OLD_PREFIX = "full_probability"
NEW_PREFIX = "v2_probability"
RETRY_SUFFIX = re.compile(r"(?:__retry[0-9]+)+$")
SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")

HARD_METRICS: Tuple[str, ...] = (
    "dice",
    "cldice",
    "hd95",
    "beta0_error",
    "clsavg",
    "cls_beta0",
    "det_f1",
)
PROBABILITY_AUDIT_FIELDS: Tuple[str, ...] = (
    "probability_sum_max_abs_error",
    "probability_min",
    "probability_max",
)
DATASETS: Tuple[Tuple[str, str], ...] = (
    ("crown", "TopCoW_val"),
    ("lausanne", "Lausanne"),
    ("ixi", "IXI_HH"),
)
CORRUPTIONS: Tuple[Tuple[str, float, str], ...] = (
    ("gaussian", 0.20, "channel_std_multiplier"),
    ("channel_dropout", 0.50, "channel_fraction"),
    ("translation", 4.0, "feature_voxel"),
    ("donor_mix", 0.50, "mixing_weight"),
)


@dataclass(frozen=True)
class CanonicalCondition:
    """One of the 39 fixed probability-repair comparisons."""

    dataset_tag: str
    dataset_name: str
    corruption_type: str
    severity: float
    severity_unit: str
    realization: int

    @property
    def suffix(self) -> str:
        if self.corruption_type == "none":
            return f"clean_{self.dataset_tag}"
        return (
            f"{self.corruption_type}_{number_tag(self.severity)}_"
            f"r{self.realization}_{self.dataset_tag}"
        )

    @property
    def condition_id(self) -> str:
        return self.suffix

    @property
    def old_run_id(self) -> str:
        return f"{OLD_PREFIX}_{self.suffix}"

    @property
    def new_run_id(self) -> str:
        return f"{NEW_PREFIX}_{self.suffix}"

    def manifest_signature(self) -> Dict[str, Any]:
        return {
            "dataset": self.dataset_name,
            "model_variant": "full",
            "seed": 0,
            "corruption_type": self.corruption_type,
            "severity": float(self.severity),
            "severity_unit": self.severity_unit,
            "realization": int(self.realization),
        }


@dataclass
class LoadedRun:
    """Parsed artifacts for the selected physical attempt."""

    experiment_id: str
    expected_run_id: str
    run_dir: Optional[Path]
    actual_run_id: Optional[str]
    candidate_attempts: List[Dict[str, Any]]
    manifest: Dict[str, Any]
    status: Dict[str, Any]
    selected_cases: List[str]
    case_rows: List[Dict[str, str]]
    case_header: List[str]
    config: Optional[Dict[str, Any]]
    runtime: Dict[str, Any]
    single_softmax_config_proof: bool
    single_softmax_config_evidence: List[str]
    errors: List[str]
    warnings: List[str]


def number_tag(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def canonical_run_id(run_id: str) -> str:
    return RETRY_SUFFIX.sub("", run_id)


def build_conditions() -> List[CanonicalCondition]:
    conditions: List[CanonicalCondition] = []
    for dataset_tag, dataset_name in DATASETS:
        conditions.append(CanonicalCondition(
            dataset_tag=dataset_tag,
            dataset_name=dataset_name,
            corruption_type="none",
            severity=0.0,
            severity_unit="none",
            realization=0,
        ))
        for corruption_type, severity, severity_unit in CORRUPTIONS:
            for realization in range(3):
                conditions.append(CanonicalCondition(
                    dataset_tag=dataset_tag,
                    dataset_name=dataset_name,
                    corruption_type=corruption_type,
                    severity=severity,
                    severity_unit=severity_unit,
                    realization=realization,
                ))
    if len(conditions) != 39:
        raise RuntimeError(
            f"Internal canonical-condition error: {len(conditions)} != 39")
    return conditions


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def canonical_json_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def canonical_list_hash(values: Sequence[str]) -> str:
    payload = json.dumps(
        list(values),
        sort_keys=False,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_json(path: Path) -> Tuple[Optional[Any], Optional[str]]:
    if not path.is_file():
        return None, "missing"
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def read_yaml(path: Path) -> Tuple[Optional[Any], Optional[str]]:
    if not path.is_file():
        return None, "missing"
    try:
        import yaml
    except Exception as exc:
        return None, f"PyYAML unavailable: {type(exc).__name__}: {exc}"
    try:
        with path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def read_csv(path: Path) -> Tuple[List[Dict[str, str]], List[str], Optional[str]]:
    if not path.is_file():
        return [], [], "missing"
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            header = list(reader.fieldnames or [])
            return list(reader), header, None
    except Exception as exc:
        return [], [], f"{type(exc).__name__}: {exc}"


def finite_float(value: Any) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def numeric_equal(
        left: Any,
        right: Any,
        *,
        atol: float,
        rtol: float) -> bool:
    left_number = finite_float(left)
    right_number = finite_float(right)
    return (
        left_number is not None
        and right_number is not None
        and math.isclose(
            left_number, right_number, abs_tol=atol, rel_tol=rtol)
    )


def semantic_equal(left: Any, right: Any) -> bool:
    """Compare manifest/config scalars, allowing numeric JSON/YAML spellings."""

    if isinstance(left, bool) or isinstance(right, bool):
        return left is right
    if left is None or right is None:
        return left is None and right is None
    left_number = finite_float(left)
    right_number = finite_float(right)
    if left_number is not None and right_number is not None:
        return math.isclose(
            left_number, right_number, rel_tol=0.0, abs_tol=1e-12)
    return left == right


def nested_get(source: Mapping[str, Any], path: Sequence[str]) -> Any:
    current: Any = source
    for part in path:
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def normalize_optional_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"none", "null", "nan"} else text


def unique_case_rows(
        rows: Sequence[Mapping[str, str]]) -> Tuple[Dict[str, Dict[str, str]], List[str]]:
    indexed: Dict[str, Dict[str, str]] = {}
    errors: List[str] = []
    for index, row in enumerate(rows):
        case_id = str(row.get("case_id", "")).strip()
        if not case_id:
            errors.append(f"case_metrics row {index} has an empty case_id")
            continue
        if case_id in indexed:
            errors.append(f"case_metrics has duplicate case_id {case_id!r}")
            continue
        indexed[case_id] = dict(row)
    return indexed, errors


def attempt_status(run_dir: Path) -> Dict[str, Any]:
    status, status_error = read_json(run_dir / "status.json")
    manifest, manifest_error = read_json(run_dir / "run_manifest.json")
    status_value = (
        status.get("status") if isinstance(status, Mapping) else None)
    manifest_value = (
        manifest.get("status") if isinstance(manifest, Mapping) else None)
    return {
        "actual_run_id": run_dir.name,
        "path": str(run_dir),
        "status": status_value,
        "manifest_status": manifest_value,
        "status_read_error": status_error,
        "manifest_read_error": manifest_error,
        "is_success": (
            status_value == "success" and manifest_value == "success"),
    }


def discover_attempts(
        experiment_dir: Path,
        expected_run_ids: Iterable[str],
        prefix: str) -> Tuple[Dict[str, List[Path]], List[str]]:
    expected = set(expected_run_ids)
    grouped: Dict[str, List[Path]] = {run_id: [] for run_id in expected}
    unexpected: List[str] = []
    if not experiment_dir.is_dir():
        return grouped, unexpected
    for path in sorted(experiment_dir.iterdir()):
        if not path.is_dir() or not path.name.startswith(prefix + "_"):
            continue
        base = canonical_run_id(path.name)
        if base in expected:
            grouped[base].append(path)
        else:
            unexpected.append(path.name)
    return grouped, unexpected


def choose_attempt(
        candidates: Sequence[Path],
        *,
        experiment_id: str,
        expected_run_id: str) -> Tuple[Optional[Path], List[Dict[str, Any]], List[str]]:
    records = [attempt_status(path) for path in candidates]
    successes = [
        path for path, record in zip(candidates, records)
        if record["is_success"]
    ]
    errors: List[str] = []
    if not candidates:
        errors.append(
            f"{experiment_id}/{expected_run_id}: no physical attempt found")
        return None, records, errors
    if not successes:
        states = ", ".join(
            f"{record['actual_run_id']}="
            f"{record['status']}/{record['manifest_status']}"
            for record in records)
        errors.append(
            f"{experiment_id}/{expected_run_id}: no successful attempt "
            f"({states})")
        return None, records, errors
    if len(successes) > 1:
        errors.append(
            f"{experiment_id}/{expected_run_id}: multiple successful "
            "attempts for one canonical condition: "
            + ", ".join(path.name for path in successes))
    exact = [path for path in successes if path.name == expected_run_id]
    selected = exact[0] if exact else sorted(successes)[-1]
    return selected, records, errors


def validate_manifest_condition(
        manifest: Mapping[str, Any],
        condition: CanonicalCondition,
        *,
        experiment_id: str,
        actual_run_id: str,
        errors: List[str]) -> Dict[str, Any]:
    observed = {
        "dataset": manifest.get("dataset"),
        "model_variant": manifest.get("model_variant"),
        "seed": manifest.get("seed"),
        "corruption_type": nested_get(manifest, ("corruption", "type")),
        "severity": nested_get(manifest, ("corruption", "severity")),
        "severity_unit": nested_get(manifest, ("corruption", "unit")),
        "realization": nested_get(manifest, ("corruption", "realization")),
    }
    expected = condition.manifest_signature()
    for field, wanted in expected.items():
        if not semantic_equal(observed.get(field), wanted):
            errors.append(
                f"manifest condition {field}: expected {wanted!r}, "
                f"observed {observed.get(field)!r}")
    if manifest.get("experiment_id") != experiment_id:
        errors.append(
            "manifest experiment/path mismatch: "
            f"{manifest.get('experiment_id')!r} vs {experiment_id!r}")
    if manifest.get("run_id") != actual_run_id:
        errors.append(
            "manifest run/path mismatch: "
            f"{manifest.get('run_id')!r} vs {actual_run_id!r}")
    for field in ("checkpoint_hash", "data_split_hash"):
        value = manifest.get(field)
        if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
            errors.append(f"manifest {field} is missing or not a SHA256")
    return observed


def validate_case_row_condition(
        row: Mapping[str, str],
        condition: CanonicalCondition,
        *,
        experiment_id: str,
        actual_run_id: str,
        row_index: int,
        errors: List[str]) -> None:
    expected_text = {
        "experiment_id": experiment_id,
        "run_id": actual_run_id,
        "dataset": condition.dataset_name,
        "model_variant": "full",
        "corruption_type": condition.corruption_type,
    }
    for field, wanted in expected_text.items():
        if str(row.get(field, "")) != wanted:
            errors.append(
                f"case row {row_index} {field}: expected {wanted!r}, "
                f"observed {row.get(field)!r}")
    expected_numeric = {
        "seed": 0,
        "realization": condition.realization,
        "severity": condition.severity,
    }
    for field, wanted in expected_numeric.items():
        if not semantic_equal(row.get(field), wanted):
            errors.append(
                f"case row {row_index} {field}: expected {wanted!r}, "
                f"observed {row.get(field)!r}")


def validate_new_config(
        run: LoadedRun,
        condition: CanonicalCondition) -> None:
    proof_issues: List[str] = []
    if run.config is None:
        run.errors.append(
            "corrected run requires readable run_config.yaml")
        run.single_softmax_config_evidence = [
            "missing readable run_config.yaml"]
        return
    config = run.config
    manifest_hash = run.manifest.get("config_hash")
    observed_hash = canonical_json_hash(config)
    if manifest_hash != observed_hash:
        message = (
            "manifest.config_hash does not match canonical run_config.yaml")
        run.errors.append(message)
        proof_issues.append(message)
    expected: Dict[str, Any] = {
        "experiment_id": NEW_EXPERIMENT,
        "run_id": run.actual_run_id,
        "dataset_name": condition.dataset_name,
        "model_variant": "full",
        "corruption_type": condition.corruption_type,
        "severity": condition.severity,
        "severity_unit": condition.severity_unit,
        "realization": condition.realization,
        "seed": 0,
        "save_probability_signals": True,
        "gate_alpha": None,
        "temperature_scale": 1.0,
        "gate_transform": "identity",
    }
    for field, wanted in expected.items():
        observed = config.get(field)
        if not semantic_equal(observed, wanted):
            message = (
                f"run_config {field}: expected {wanted!r}, "
                f"observed {observed!r}")
            run.errors.append(message)
            proof_issues.append(message)

    probability_protocol = nested_get(
        config, ("protocol_notes", "probability_signals"))
    if probability_protocol != "persisted_region_means":
        message = (
            "run_config protocol_notes.probability_signals must be "
            "'persisted_region_means'")
        run.errors.append(message)
        proof_issues.append(message)
    missing_schema_fields = [
        field for field in PROBABILITY_AUDIT_FIELDS
        if field not in run.case_header
    ]
    if missing_schema_fields:
        message = (
            "corrected case schema is missing direct probability-interface "
            "audit fields: " + ", ".join(missing_schema_fields))
        run.errors.append(message)
        proof_issues.append(message)

    intervention = run.manifest.get("intervention")
    if intervention is None:
        run.warnings.append(
            "run_manifest intervention object absent; explicit hashed "
            "run_config intervention fields were used")
    elif not isinstance(intervention, Mapping):
        run.errors.append("run_manifest intervention is not an object")
    else:
        intervention_expected = {
            "gate_alpha": None,
            "temperature_scale": 1.0,
            "gate_transform": "identity",
        }
        for field, wanted in intervention_expected.items():
            if not semantic_equal(intervention.get(field), wanted):
                message = (
                    f"manifest intervention.{field}: expected {wanted!r}, "
                    f"observed {intervention.get(field)!r}")
                run.errors.append(message)
                proof_issues.append(message)

    run.single_softmax_config_proof = not proof_issues
    run.single_softmax_config_evidence = (
        [
            "canonical run_config.yaml hash matches manifest.config_hash",
            "save_probability_signals=true",
            (
                "protocol_notes.probability_signals="
                "persisted_region_means"
            ),
            "gate_alpha=null, temperature_scale=1, gate_transform=identity",
            (
                "case schema persists probability_sum_max_abs_error, "
                "probability_min, and probability_max"
            ),
        ]
        if not proof_issues else proof_issues
    )


def load_selected_run(
        selected_dir: Optional[Path],
        candidate_records: List[Dict[str, Any]],
        selection_errors: Sequence[str],
        *,
        experiment_id: str,
        expected_run_id: str,
        condition: CanonicalCondition,
        require_corrected_config: bool) -> LoadedRun:
    run = LoadedRun(
        experiment_id=experiment_id,
        expected_run_id=expected_run_id,
        run_dir=selected_dir,
        actual_run_id=selected_dir.name if selected_dir else None,
        candidate_attempts=candidate_records,
        manifest={},
        status={},
        selected_cases=[],
        case_rows=[],
        case_header=[],
        config=None,
        runtime={},
        single_softmax_config_proof=False,
        single_softmax_config_evidence=[],
        errors=list(selection_errors),
        warnings=[],
    )
    if selected_dir is None:
        return run

    manifest_value, manifest_error = read_json(
        selected_dir / "run_manifest.json")
    status_value, status_error = read_json(selected_dir / "status.json")
    selected_value, selected_error = read_json(
        selected_dir / "selected_cases.json")
    rows, header, csv_error = read_csv(selected_dir / "case_metrics.csv")
    runtime_value, runtime_error = read_json(selected_dir / "runtime.json")

    if manifest_error or not isinstance(manifest_value, Mapping):
        run.errors.append(
            f"run_manifest.json unreadable or invalid: {manifest_error}")
    else:
        run.manifest = dict(manifest_value)
    if status_error or not isinstance(status_value, Mapping):
        run.errors.append(
            f"status.json unreadable or invalid: {status_error}")
    else:
        run.status = dict(status_value)
    if selected_error or not isinstance(selected_value, list):
        run.errors.append(
            f"selected_cases.json unreadable or invalid: {selected_error}")
    elif not selected_value or not all(
            isinstance(case_id, str) and case_id
            for case_id in selected_value):
        run.errors.append(
            "selected_cases.json must be a non-empty list of strings")
    else:
        run.selected_cases = list(selected_value)
    if csv_error:
        run.errors.append(f"case_metrics.csv unreadable: {csv_error}")
    elif not rows:
        run.errors.append("case_metrics.csv has no case rows")
    else:
        run.case_rows = rows
        run.case_header = header
    if runtime_error is None and isinstance(runtime_value, Mapping):
        run.runtime = dict(runtime_value)
    elif runtime_error != "missing":
        run.warnings.append(f"runtime.json unreadable: {runtime_error}")
    else:
        run.warnings.append("runtime.json missing")

    if run.manifest:
        validate_manifest_condition(
            run.manifest,
            condition,
            experiment_id=experiment_id,
            actual_run_id=selected_dir.name,
            errors=run.errors,
        )
        if run.manifest.get("status") != "success":
            run.errors.append(
                f"manifest status is {run.manifest.get('status')!r}, "
                "not 'success'")
        declared_canonical = run.manifest.get("canonical_run_id")
        if (
                declared_canonical is not None
                and declared_canonical != expected_run_id):
            run.errors.append(
                "manifest canonical_run_id mismatch: "
                f"{declared_canonical!r} vs {expected_run_id!r}")
    if run.status:
        if run.status.get("status") != "success":
            run.errors.append(
                f"status.json status is {run.status.get('status')!r}, "
                "not 'success'")
        n_failed = finite_float(run.status.get("n_failed"))
        if n_failed is not None and n_failed != 0:
            run.errors.append(
                f"status.json reports n_failed={n_failed:g}")

    if run.case_rows:
        indexed, duplicate_errors = unique_case_rows(run.case_rows)
        run.errors.extend(duplicate_errors)
        row_order = [
            str(row.get("case_id", "")).strip() for row in run.case_rows]
        if run.selected_cases and row_order != run.selected_cases:
            run.errors.append(
                "case_metrics case order does not exactly match "
                "selected_cases.json")
        if run.selected_cases and len(indexed) != len(run.selected_cases):
            run.errors.append(
                f"unique case row count {len(indexed)} != selected case "
                f"count {len(run.selected_cases)}")
        n_success = finite_float(run.status.get("n_success"))
        if n_success is not None and n_success != len(run.case_rows):
            run.errors.append(
                f"status.json n_success={n_success:g} != case row count "
                f"{len(run.case_rows)}")
        for index, row in enumerate(run.case_rows):
            validate_case_row_condition(
                row,
                condition,
                experiment_id=experiment_id,
                actual_run_id=selected_dir.name,
                row_index=index,
                errors=run.errors,
            )

    failures_value, failures_error = read_json(
        selected_dir / "case_failures.json")
    if failures_error is None and failures_value != []:
        run.errors.append("case_failures.json is not empty")
    elif failures_error not in (None, "missing"):
        run.errors.append(
            f"case_failures.json unreadable: {failures_error}")

    config_value, config_error = read_yaml(
        selected_dir / "run_config.yaml")
    if config_error is None and isinstance(config_value, Mapping):
        run.config = dict(config_value)
    elif config_error != "missing":
        run.errors.append(f"run_config.yaml unreadable: {config_error}")
    elif require_corrected_config:
        run.errors.append("corrected run is missing run_config.yaml")
    else:
        run.warnings.append(
            "historical run has no self-contained run_config.yaml")

    if require_corrected_config:
        validate_new_config(run, condition)
    elif run.config is not None:
        expected_hash = run.manifest.get("config_hash")
        observed_hash = canonical_json_hash(run.config)
        if expected_hash != observed_hash:
            run.errors.append(
                "historical manifest.config_hash does not match present "
                "run_config.yaml")
    return run


def load_experiment_runs(
        raw_root: Path,
        conditions: Sequence[CanonicalCondition],
        *,
        experiment_id: str,
        prefix: str,
        old_side: bool) -> Tuple[Dict[str, LoadedRun], List[str]]:
    expected_ids = [
        condition.old_run_id if old_side else condition.new_run_id
        for condition in conditions
    ]
    grouped, unexpected = discover_attempts(
        raw_root / experiment_id, expected_ids, prefix)
    loaded: Dict[str, LoadedRun] = {}
    for condition, expected_id in zip(conditions, expected_ids):
        selected, records, selection_errors = choose_attempt(
            grouped[expected_id],
            experiment_id=experiment_id,
            expected_run_id=expected_id,
        )
        loaded[condition.condition_id] = load_selected_run(
            selected,
            records,
            selection_errors,
            experiment_id=experiment_id,
            expected_run_id=expected_id,
            condition=condition,
            require_corrected_config=not old_side,
        )
    return loaded, unexpected


def runtime_identity(run: LoadedRun) -> Dict[str, Optional[str]]:
    """Return the execution identity relevant to exact reproducibility."""

    commit = run.manifest.get("git_commit")
    gpu_name = run.runtime.get("gpu_name")
    return {
        "git_commit": (
            str(commit).strip() if isinstance(commit, str) and commit.strip()
            else None),
        "gpu_name": (
            str(gpu_name).strip()
            if isinstance(gpu_name, str) and gpu_name.strip() else None),
        "torch": (
            str(run.runtime.get("torch"))
            if run.runtime.get("torch") is not None else None),
        "cuda": (
            str(run.runtime.get("cuda"))
            if run.runtime.get("cuda") is not None else None),
    }


def hard_equivalence_applicability(
        old_run: LoadedRun,
        new_run: LoadedRun) -> Dict[str, Any]:
    """Decide whether byte-exact hard-output equivalence is a valid gate."""

    old_identity = runtime_identity(old_run)
    new_identity = runtime_identity(new_run)
    commit_known = (
        old_identity["git_commit"] is not None
        and new_identity["git_commit"] is not None)
    gpu_known = (
        old_identity["gpu_name"] is not None
        and new_identity["gpu_name"] is not None)
    same_commit = (
        commit_known
        and old_identity["git_commit"] == new_identity["git_commit"])
    same_gpu = (
        gpu_known
        and old_identity["gpu_name"] == new_identity["gpu_name"])
    applicable = bool(same_commit and same_gpu)
    if applicable:
        status = "applicable_same_runtime"
        reason = (
            "git commit and GPU model match; exact prediction SHA is a "
            "strict invariant")
    elif commit_known and gpu_known:
        status = "not_applicable_cross_runtime"
        reason = (
            "git commit and/or GPU model differ; SHA and metric deltas are "
            "descriptive only")
    else:
        status = "not_applicable_runtime_metadata_missing"
        reason = (
            "commit or GPU metadata is missing; exact cross-run causality "
            "cannot be established")
    return {
        "status": status,
        "exact_sha_strict_gate": applicable,
        "same_git_commit": same_commit,
        "same_gpu_model": same_gpu,
        "old_runtime": old_identity,
        "new_runtime": new_identity,
        "reason": reason,
    }


def compare_case(
        condition: CanonicalCondition,
        case_id: str,
        position: int,
        old_row: Optional[Mapping[str, str]],
        new_row: Optional[Mapping[str, str]],
        *,
        atol: float,
        rtol: float,
        probability_sum_tolerance: float) -> Dict[str, Any]:
    issues: List[str] = []
    output: Dict[str, Any] = {
        "condition_id": condition.condition_id,
        "dataset_tag": condition.dataset_tag,
        "dataset": condition.dataset_name,
        "corruption_type": condition.corruption_type,
        "severity": condition.severity,
        "severity_unit": condition.severity_unit,
        "realization": condition.realization,
        "case_position": position,
        "case_id": case_id,
        "old_case_present": old_row is not None,
        "new_case_present": new_row is not None,
    }
    if old_row is None:
        issues.append("case missing from historical run")
    if new_row is None:
        issues.append("case missing from corrected run")
    if old_row is None or new_row is None:
        output.update({
            "donor_case_equal": False,
            "old_prediction_sha256": (
                old_row.get("prediction_sha256") if old_row else None),
            "new_prediction_sha256": (
                new_row.get("prediction_sha256") if new_row else None),
            "prediction_sha256_equal": False,
            "hard_metrics_within_tolerance": False,
            "new_probability_audit_pass": False,
            "descriptive_hard_equivalence_pass": False,
            "repair_gate_case_pass": False,
            "row_pass": False,
            "issues": "; ".join(issues),
        })
        for metric in HARD_METRICS:
            output[f"old_{metric}"] = old_row.get(metric) if old_row else None
            output[f"new_{metric}"] = new_row.get(metric) if new_row else None
            output[f"abs_diff_{metric}"] = None
            output[f"{metric}_within_tolerance"] = False
        for field in PROBABILITY_AUDIT_FIELDS:
            output[f"new_{field}"] = (
                new_row.get(field) if new_row else None)
        return output

    old_donor = normalize_optional_text(old_row.get("donor_case_id"))
    new_donor = normalize_optional_text(new_row.get("donor_case_id"))
    donor_equal = old_donor == new_donor
    if not donor_equal:
        issues.append(
            f"donor_case_id mismatch: {old_donor!r} vs {new_donor!r}")
    output["donor_case_equal"] = donor_equal

    old_sha = str(old_row.get("prediction_sha256", "")).strip()
    new_sha = str(new_row.get("prediction_sha256", "")).strip()
    sha_valid = (
        SHA256_RE.fullmatch(old_sha) is not None
        and SHA256_RE.fullmatch(new_sha) is not None
    )
    sha_equal = sha_valid and old_sha == new_sha
    if not sha_equal:
        issues.append(
            "prediction_sha256 missing, invalid, or unequal")
    output["old_prediction_sha256"] = old_sha
    output["new_prediction_sha256"] = new_sha
    output["prediction_sha256_equal"] = sha_equal

    metric_passes: List[bool] = []
    for metric in HARD_METRICS:
        old_value = finite_float(old_row.get(metric))
        new_value = finite_float(new_row.get(metric))
        within = numeric_equal(
            old_value, new_value, atol=atol, rtol=rtol)
        metric_passes.append(within)
        output[f"old_{metric}"] = old_value
        output[f"new_{metric}"] = new_value
        output[f"abs_diff_{metric}"] = (
            abs(old_value - new_value)
            if old_value is not None and new_value is not None
            else None)
        output[f"{metric}_within_tolerance"] = within
        if not within:
            issues.append(
                f"{metric} not finite/equal within atol={atol:g}, "
                f"rtol={rtol:g}: {old_row.get(metric)!r} vs "
                f"{new_row.get(metric)!r}")
    hard_metrics_pass = all(metric_passes)
    output["hard_metrics_within_tolerance"] = hard_metrics_pass

    probability_error = finite_float(
        new_row.get("probability_sum_max_abs_error"))
    probability_min = finite_float(new_row.get("probability_min"))
    probability_max = finite_float(new_row.get("probability_max"))
    probability_error_pass = (
        probability_error is not None
        and 0.0 <= probability_error <= probability_sum_tolerance
    )
    probability_range_pass = (
        probability_min is not None
        and probability_max is not None
        and 0.0 <= probability_min <= probability_max <= 1.0
    )
    probability_pass = probability_error_pass and probability_range_pass
    output["new_probability_sum_max_abs_error"] = probability_error
    output["new_probability_min"] = probability_min
    output["new_probability_max"] = probability_max
    output["new_probability_sum_pass"] = probability_error_pass
    output["new_probability_range_pass"] = probability_range_pass
    output["new_probability_audit_pass"] = probability_pass
    if not probability_error_pass:
        issues.append(
            "corrected probability_sum_max_abs_error is missing, "
            f"non-finite, negative, or > {probability_sum_tolerance:g}")
    if not probability_range_pass:
        issues.append(
            "corrected probability_min/max are missing, non-finite, "
            "unordered, or outside [0, 1]")

    output["descriptive_hard_equivalence_pass"] = (
        donor_equal and sha_equal and hard_metrics_pass)
    output["repair_gate_case_pass"] = donor_equal and probability_pass
    # ``row_pass`` is retained as a compatibility alias for the strict repair
    # gate.  Cross-runtime SHA/metric deltas remain fully reported above.
    output["row_pass"] = output["repair_gate_case_pass"]
    output["issues"] = "; ".join(issues)
    return output


def compare_condition(
        condition: CanonicalCondition,
        old_run: LoadedRun,
        new_run: LoadedRun,
        *,
        atol: float,
        rtol: float,
        probability_sum_tolerance: float) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    issues = list(old_run.errors) + list(new_run.errors)
    observations: List[str] = []
    old_manifest = old_run.manifest
    new_manifest = new_run.manifest
    equivalence_policy = hard_equivalence_applicability(old_run, new_run)

    old_signature = {
        "dataset": old_manifest.get("dataset"),
        "model_variant": old_manifest.get("model_variant"),
        "seed": old_manifest.get("seed"),
        "corruption_type": nested_get(
            old_manifest, ("corruption", "type")),
        "severity": nested_get(old_manifest, ("corruption", "severity")),
        "severity_unit": nested_get(old_manifest, ("corruption", "unit")),
        "realization": nested_get(
            old_manifest, ("corruption", "realization")),
    } if old_manifest else {}
    new_signature = {
        "dataset": new_manifest.get("dataset"),
        "model_variant": new_manifest.get("model_variant"),
        "seed": new_manifest.get("seed"),
        "corruption_type": nested_get(
            new_manifest, ("corruption", "type")),
        "severity": nested_get(new_manifest, ("corruption", "severity")),
        "severity_unit": nested_get(new_manifest, ("corruption", "unit")),
        "realization": nested_get(
            new_manifest, ("corruption", "realization")),
    } if new_manifest else {}
    condition_metadata_equal = (
        bool(old_signature)
        and bool(new_signature)
        and all(
            semantic_equal(old_signature.get(field), new_signature.get(field))
            for field in condition.manifest_signature())
    )
    if not condition_metadata_equal:
        issues.append("historical/corrected manifest conditions differ")

    selected_cases_equal = (
        bool(old_run.selected_cases)
        and old_run.selected_cases == new_run.selected_cases)
    if not selected_cases_equal:
        issues.append(
            "historical/corrected selected_cases lists differ")

    old_checkpoint = old_manifest.get("checkpoint_hash")
    new_checkpoint = new_manifest.get("checkpoint_hash")
    checkpoint_equal = (
        isinstance(old_checkpoint, str)
        and SHA256_RE.fullmatch(old_checkpoint) is not None
        and old_checkpoint == new_checkpoint)
    if not checkpoint_equal:
        issues.append("historical/corrected checkpoint hashes differ")

    old_split = old_manifest.get("data_split_hash")
    new_split = new_manifest.get("data_split_hash")
    split_equal = (
        isinstance(old_split, str)
        and SHA256_RE.fullmatch(old_split) is not None
        and old_split == new_split)
    if not split_equal:
        issues.append("historical/corrected data split hashes differ")

    old_index, old_index_errors = unique_case_rows(old_run.case_rows)
    new_index, new_index_errors = unique_case_rows(new_run.case_rows)
    issues.extend(old_index_errors)
    issues.extend(new_index_errors)
    case_ids = list(old_run.selected_cases)
    for case_id in new_run.selected_cases:
        if case_id not in case_ids:
            case_ids.append(case_id)
    for case_id in sorted(set(old_index) | set(new_index)):
        if case_id not in case_ids:
            case_ids.append(case_id)

    case_rows = [
        compare_case(
            condition,
            case_id,
            position,
            old_index.get(case_id),
            new_index.get(case_id),
            atol=atol,
            rtol=rtol,
            probability_sum_tolerance=probability_sum_tolerance,
        )
        for position, case_id in enumerate(case_ids)
    ]
    if not case_rows:
        issues.append("no cases available for comparison")
    case_set_equal = (
        bool(old_index)
        and set(old_index) == set(new_index)
        and selected_cases_equal)
    prediction_equal = bool(case_rows) and all(
        row["prediction_sha256_equal"] for row in case_rows)
    hard_metrics_equal = bool(case_rows) and all(
        row["hard_metrics_within_tolerance"] for row in case_rows)
    probability_pass = bool(case_rows) and all(
        row["new_probability_audit_pass"] for row in case_rows)
    donor_equal = bool(case_rows) and all(
        row["donor_case_equal"] for row in case_rows)
    if not case_set_equal:
        issues.append("case sets/order are not identical")
    if not prediction_equal:
        message = "one or more prediction SHA256 values differ"
        if equivalence_policy["exact_sha_strict_gate"]:
            issues.append(message)
        else:
            observations.append(
                message + "; descriptive only because exact hard "
                "equivalence is not applicable across runtimes")
    if not hard_metrics_equal:
        observations.append(
            "one or more hard metrics differ beyond the reporting "
            "tolerance; these are descriptive cross-run deltas, not a "
            "strict repair gate")
    if not probability_pass:
        issues.append("one or more corrected probability audits fail")
    if not donor_equal:
        issues.append("one or more donor-case assignments differ")

    old_missing_fields = [
        field for field in PROBABILITY_AUDIT_FIELDS
        if field not in old_run.case_header
    ]
    old_probability_evidence_missing = (
        bool(old_run.case_rows) and bool(old_missing_fields))
    if not old_probability_evidence_missing:
        observations.append(
            "historical run unexpectedly contains all corrected "
            "probability-audit fields; legacy invalidity cannot be "
            "established from the declared evidence-gap criterion")

    max_abs_diffs: Dict[str, Optional[float]] = {}
    for metric in HARD_METRICS:
        values = [
            row[f"abs_diff_{metric}"]
            for row in case_rows
            if row.get(f"abs_diff_{metric}") is not None
        ]
        max_abs_diffs[metric] = max(values) if values else None

    condition_pass = (
        not old_run.errors
        and not new_run.errors
        and condition_metadata_equal
        and selected_cases_equal
        and checkpoint_equal
        and split_equal
        and case_set_equal
        and donor_equal
        and probability_pass
        and new_run.single_softmax_config_proof
        and (
            prediction_equal
            if equivalence_policy["exact_sha_strict_gate"] else True)
    )
    summary: Dict[str, Any] = {
        "condition_id": condition.condition_id,
        "dataset_tag": condition.dataset_tag,
        "dataset": condition.dataset_name,
        "corruption_type": condition.corruption_type,
        "severity": condition.severity,
        "severity_unit": condition.severity_unit,
        "realization": condition.realization,
        "old_expected_run_id": condition.old_run_id,
        "old_actual_run_id": old_run.actual_run_id,
        "new_expected_run_id": condition.new_run_id,
        "new_actual_run_id": new_run.actual_run_id,
        "old_attempt_count": len(old_run.candidate_attempts),
        "new_attempt_count": len(new_run.candidate_attempts),
        "old_status": old_run.status.get("status"),
        "new_status": new_run.status.get("status"),
        "old_case_count": len(old_run.case_rows),
        "new_case_count": len(new_run.case_rows),
        "condition_metadata_equal": condition_metadata_equal,
        "selected_cases_equal": selected_cases_equal,
        "selected_cases_sha256": (
            canonical_list_hash(old_run.selected_cases)
            if selected_cases_equal else None),
        "checkpoint_hash_equal": checkpoint_equal,
        "checkpoint_sha256": old_checkpoint if checkpoint_equal else None,
        "data_split_hash_equal": split_equal,
        "data_split_sha256": old_split if split_equal else None,
        "case_set_equal": case_set_equal,
        "donor_case_assignments_equal": donor_equal,
        "hard_equivalence": equivalence_policy["status"],
        "exact_sha_strict_gate": equivalence_policy[
            "exact_sha_strict_gate"],
        "same_git_commit": equivalence_policy["same_git_commit"],
        "same_gpu_model": equivalence_policy["same_gpu_model"],
        "old_git_commit": equivalence_policy[
            "old_runtime"]["git_commit"],
        "new_git_commit": equivalence_policy[
            "new_runtime"]["git_commit"],
        "old_gpu_name": equivalence_policy["old_runtime"]["gpu_name"],
        "new_gpu_name": equivalence_policy["new_runtime"]["gpu_name"],
        "prediction_sha256_all_equal": prediction_equal,
        "prediction_sha256_match_count": sum(
            bool(row["prediction_sha256_equal"]) for row in case_rows),
        "hard_metrics_all_within_tolerance": hard_metrics_equal,
        "new_probability_audit_pass": probability_pass,
        "single_softmax_config_proof": (
            new_run.single_softmax_config_proof),
        "single_softmax_config_evidence": (
            "; ".join(new_run.single_softmax_config_evidence)),
        "old_probability_evidence_missing": (
            old_probability_evidence_missing),
        "old_missing_probability_audit_fields": ",".join(
            old_missing_fields),
        "old_run_config_present": old_run.config is not None,
        "new_run_config_present": new_run.config is not None,
        "old_warnings": "; ".join(old_run.warnings),
        "new_warnings": "; ".join(new_run.warnings),
        "condition_pass": condition_pass,
        "issues": "; ".join(dict.fromkeys(issues)),
        "observations": "; ".join(dict.fromkeys(observations)),
    }
    for metric, value in max_abs_diffs.items():
        summary[f"max_abs_diff_{metric}"] = value
    return summary, case_rows


def load_g1_arm(run_dir: Path) -> Dict[str, Any]:
    manifest_value, manifest_error = read_json(
        run_dir / "run_manifest.json")
    runtime_value, runtime_error = read_json(run_dir / "runtime.json")
    selected_value, selected_error = read_json(
        run_dir / "selected_cases.json")
    rows, header, rows_error = read_csv(run_dir / "case_metrics.csv")
    errors: List[str] = []
    manifest = (
        dict(manifest_value) if isinstance(manifest_value, Mapping) else {})
    runtime = (
        dict(runtime_value) if isinstance(runtime_value, Mapping) else {})
    selected = (
        list(selected_value) if isinstance(selected_value, list) else [])
    if manifest_error or not manifest:
        errors.append(f"manifest invalid: {manifest_error}")
    if runtime_error or not runtime:
        errors.append(f"runtime invalid: {runtime_error}")
    if selected_error or not selected:
        errors.append(f"selected cases invalid: {selected_error}")
    if rows_error or not rows:
        errors.append(f"case metrics invalid: {rows_error}")
    return {
        "run_id": run_dir.name,
        "path": str(run_dir),
        "manifest": manifest,
        "runtime": runtime,
        "selected_cases": selected,
        "case_rows": rows,
        "case_header": header,
        "errors": errors,
    }


def inspect_g1_pair(
        raw_root: Path,
        experiment_id: str,
        control_base: str,
        probability_base: str) -> Optional[Dict[str, Any]]:
    experiment_dir = raw_root / experiment_id
    if not experiment_dir.is_dir():
        return None
    control_candidates = [
        path for path in experiment_dir.iterdir()
        if path.is_dir() and canonical_run_id(path.name) == control_base]
    probability_candidates = [
        path for path in experiment_dir.iterdir()
        if path.is_dir() and canonical_run_id(path.name) == probability_base]
    control_path, control_attempts, control_errors = choose_attempt(
        sorted(control_candidates),
        experiment_id=experiment_id,
        expected_run_id=control_base,
    )
    probability_path, probability_attempts, probability_errors = (
        choose_attempt(
            sorted(probability_candidates),
            experiment_id=experiment_id,
            expected_run_id=probability_base,
        ))
    if control_path is None or probability_path is None:
        return None
    control = load_g1_arm(control_path)
    probability = load_g1_arm(probability_path)
    errors = (
        control_errors + probability_errors
        + control["errors"] + probability["errors"])

    control_manifest = control["manifest"]
    probability_manifest = probability["manifest"]
    control_runtime = control["runtime"]
    probability_runtime = probability["runtime"]
    same_commit = (
        bool(control_manifest.get("git_commit"))
        and control_manifest.get("git_commit")
        == probability_manifest.get("git_commit"))
    same_gpu = (
        bool(control_runtime.get("gpu_name"))
        and control_runtime.get("gpu_name")
        == probability_runtime.get("gpu_name"))
    same_checkpoint = (
        bool(control_manifest.get("checkpoint_hash"))
        and control_manifest.get("checkpoint_hash")
        == probability_manifest.get("checkpoint_hash"))
    same_split = (
        bool(control_manifest.get("data_split_hash"))
        and control_manifest.get("data_split_hash")
        == probability_manifest.get("data_split_hash"))
    same_cases = (
        bool(control["selected_cases"])
        and control["selected_cases"] == probability["selected_cases"])
    control_index, control_index_errors = unique_case_rows(
        control["case_rows"])
    probability_index, probability_index_errors = unique_case_rows(
        probability["case_rows"])
    errors.extend(control_index_errors)
    errors.extend(probability_index_errors)
    sha_equal = (
        same_cases
        and set(control_index) == set(probability_index)
        and all(
            str(control_index[case_id].get("prediction_sha256", "")).strip()
            == str(
                probability_index[case_id].get(
                    "prediction_sha256", "")).strip()
            and SHA256_RE.fullmatch(
                str(control_index[case_id].get(
                    "prediction_sha256", "")).strip()) is not None
            for case_id in control["selected_cases"])
    )
    hard_metrics_exact = (
        same_cases
        and set(control_index) == set(probability_index)
        and all(
            numeric_equal(
                control_index[case_id].get(metric),
                probability_index[case_id].get(metric),
                atol=0.0,
                rtol=0.0,
            )
            for case_id in control["selected_cases"]
            for metric in HARD_METRICS)
    )
    probability_capture_configured = (
        all(
            field in probability["case_header"]
            for field in ("mean_confidence", "mean_entropy"))
        and all(
            field not in control["case_header"]
            for field in ("mean_confidence", "mean_entropy"))
    )
    passed = (
        not errors
        and same_commit
        and same_gpu
        and same_checkpoint
        and same_split
        and same_cases
        and sha_equal
        and hard_metrics_exact
        and probability_capture_configured)
    return {
        "status": "pass" if passed else "fail",
        "experiment_id": experiment_id,
        "control_run_id": control["run_id"],
        "probability_run_id": probability["run_id"],
        "same_git_commit": same_commit,
        "git_commit": (
            control_manifest.get("git_commit") if same_commit else None),
        "same_gpu_model": same_gpu,
        "gpu_name": (
            control_runtime.get("gpu_name") if same_gpu else None),
        "same_checkpoint_hash": same_checkpoint,
        "same_data_split_hash": same_split,
        "same_selected_cases": same_cases,
        "case_count": len(control["selected_cases"]),
        "prediction_sha256_exact": sha_equal,
        "hard_metrics_exact": hard_metrics_exact,
        "probability_capture_configured": probability_capture_configured,
        "control_attempts": control_attempts,
        "probability_attempts": probability_attempts,
        "errors": errors,
        "scope": (
            "Same-runtime G1 checks that enabling probability capture does "
            "not alter hard predictions. It does not make the historical "
            "and corrected 39-run matrices a causal softmax ablation."),
    }


def inspect_same_runtime_g1(raw_root: Path) -> Dict[str, Any]:
    """Find and validate a real same-runtime probability-capture invariant."""

    candidate_specs: List[Tuple[str, str, str]] = [
        ("V2_G1_SMOKE", "v2_full_control", "v2_full_probability"),
    ]
    legacy_dir = raw_root / "G1_PROBABILITY_INTERFACE"
    if legacy_dir.is_dir():
        bases = {
            canonical_run_id(path.name)
            for path in legacy_dir.iterdir() if path.is_dir()
        }
        for control_base in sorted(
                base for base in bases if base.endswith("_control")):
            stem = control_base[:-len("_control")]
            probability_base = stem + "_probability_maps"
            if probability_base in bases:
                candidate_specs.append((
                    "G1_PROBABILITY_INTERFACE",
                    control_base,
                    probability_base,
                ))

    inspected: List[Dict[str, Any]] = []
    for experiment_id, control_base, probability_base in candidate_specs:
        evidence = inspect_g1_pair(
            raw_root, experiment_id, control_base, probability_base)
        if evidence is not None:
            inspected.append(evidence)
            if evidence["status"] == "pass":
                return {
                    **evidence,
                    "recommendation": (
                        "Retain this same-runtime G1 as the hard-output "
                        "invariance evidence; do not use the cross-runtime "
                        "39-run deltas as a causal softmax comparison."),
                    "inspected_candidates": len(inspected),
                }
    if inspected:
        evidence = inspected[0]
        return {
            **evidence,
            "recommendation": (
                "Run a one-case control and corrected probability-capture "
                "arm back-to-back from the same checkout on the same GPU, "
                "with identical checkpoint/split/case/config except the "
                "capture flag; require exact prediction SHA and hard metrics "
                "and validate corrected sum/min/max fields."),
            "inspected_candidates": len(inspected),
        }
    return {
        "status": "not_found",
        "scope": (
            "No same-runtime G1 pair was found; no invariant is inferred or "
            "fabricated."),
        "recommendation": (
            "Run a one-case control and corrected probability-capture arm "
            "back-to-back from the same checkout on the same GPU, with "
            "identical checkpoint/split/case/config except the capture flag; "
            "require exact prediction SHA and hard metrics and validate "
            "corrected sum/min/max fields."),
        "inspected_candidates": 0,
    }


def csv_scalar(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return value


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(
            value,
            handle,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        handle.write("\n")
    temporary.replace(path)


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        if not text.endswith("\n"):
            handle.write("\n")
    temporary.replace(path)


def ordered_fieldnames(
        rows: Sequence[Mapping[str, Any]],
        preferred: Sequence[str]) -> List[str]:
    observed = {
        key for row in rows for key in row.keys()
    }
    result = [field for field in preferred if field in observed]
    result.extend(sorted(observed - set(result)))
    return result


def atomic_csv(
        path: Path,
        rows: Sequence[Mapping[str, Any]],
        preferred_fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ordered_fieldnames(rows, preferred_fields)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({
                field: csv_scalar(row.get(field)) for field in fields})
    temporary.replace(path)


def markdown_report(audit: Mapping[str, Any]) -> str:
    totals = audit["totals"]
    legacy = audit["legacy_probability_evidence"]
    invariant = audit["corrected_probability_invariant"]
    hard = audit["hard_equivalence_details"]
    config_proof = audit["single_softmax_configuration"]
    g1 = audit["same_runtime_g1_invariant"]
    lines = [
        "# TCFF probability repair strict audit",
        "",
        f"- verdict: **{audit['strict_verdict']}**",
        f"- protocol: `{audit['protocol']}`",
        f"- canonical conditions: "
        f"{totals['conditions_passed']}/{totals['conditions_expected']}",
        f"- cases passing the strict repair gate: "
        f"{totals['strict_case_rows_passed']}/"
        f"{totals['case_rows_compared']}",
        f"- corrected configurations proving the audited probability path: "
        f"{config_proof['conditions_passed']}/"
        f"{config_proof['conditions_checked']}",
        "",
        "The strict gate covers canonical condition identity, checkpoint and "
        "data-split hashes, selected cases (including donor assignment), the "
        "hashed corrected probability-path configuration, and every corrected "
        "normalization/range invariant. Exact prediction SHA is additionally "
        "a strict gate only for a same-commit, same-GPU comparison.",
        "",
        "## Hard-output comparison and runtime applicability",
        "",
        f"- hard equivalence: **`{hard['status']}`**",
        f"- exact hard-prediction SHA matches: "
        f"{totals['prediction_sha_matches']}/"
        f"{totals['case_rows_compared']} "
        f"({hard['prediction_sha256_match_rate']:.2%})",
        f"- hard-metric rows within tolerance: "
        f"{totals['hard_metric_rows_within_tolerance']}/"
        f"{totals['case_rows_compared']}",
        f"- historical commits: `{', '.join(hard['old_git_commits'])}`",
        f"- corrected commits: `{', '.join(hard['new_git_commits'])}`",
        f"- historical GPUs: `{', '.join(hard['old_gpu_models'])}`",
        f"- corrected GPUs: `{', '.join(hard['new_gpu_models'])}`",
        "",
        "Maximum absolute hard-metric deltas (descriptive):",
        "",
    ]
    for metric in HARD_METRICS:
        lines.append(
            f"- `{metric}`: `{hard['max_absolute_metric_differences'][metric]}`")
    lines.extend([
        "",
        (
            "**Causal-scope warning:** because the matrix sides were executed "
            "under different commits and/or GPU models, SHA mismatches and "
            "hard-metric deltas cannot be attributed to the softmax repair. "
            "They are reported only as cross-runtime reproducibility deltas."
            if hard["status"] == "not_applicable_cross_runtime"
            else (
                "**Causal-scope warning:** hard-output deltas are not evidence "
                "of the softmax repair unless the comparison is explicitly "
                "marked `applicable_same_runtime`."
            )
        ),
        "",
        "## Corrected probability interface",
        "",
        (
            "Every corrected case must have a finite, non-negative "
            "`probability_sum_max_abs_error` no greater than "
            f"`{invariant['sum_tolerance']:.8g}`, plus finite ordered "
            "`probability_min`/`probability_max` values inside `[0, 1]`."
        ),
        "",
        f"- corrected cases passing: "
        f"{invariant['cases_passed']}/{invariant['cases_checked']}",
        f"- maximum observed sum-to-one absolute error: "
        f"`{invariant['max_observed_sum_error']}`",
        f"- observed probability envelope: "
        f"`[{invariant['min_observed_probability']}, "
        f"{invariant['max_observed_probability']}]`",
        f"- hashed single-softmax-path configurations passing: "
        f"{config_proof['conditions_passed']}/"
        f"{config_proof['conditions_checked']}",
        "",
        (
            "Here “configuration proof” means that the manifest hash matches "
            "the self-contained run configuration, "
            "`save_probability_signals=true`, the probability-summary "
            "protocol is selected with identity interventions, and the "
            "direct sum/min/max audit schema is present. It is not inferred "
            "from the numerical value of confidence or entropy."
        ),
        "",
        "## Why the historical probability summaries are invalid",
        "",
        (
            "The historical P0 soft-probability summaries are not "
            "admissible evidence because their saved case schema does not "
            "carry the normalization/range audit fields needed to establish "
            "that the tensor used for confidence and entropy was a valid "
            "class-probability tensor."
        ),
        "",
        f"- historical conditions with a probability-audit evidence gap: "
        f"{legacy['conditions_with_evidence_gap']}/"
        f"{legacy['conditions_checked']}",
        f"- historical conditions missing self-contained `run_config.yaml`: "
        f"{legacy['conditions_missing_run_config']}/"
        f"{legacy['conditions_checked']}",
        (
            "- scope guard: this evidence-gap finding does **not** assume "
            "that any particular historical soft-probability number is "
            "numerically wrong. Hard predictions and hard metrics are "
            "assessed independently above."
        ),
        "",
        "## Same-runtime G1 invariant",
        "",
        f"- status: **`{g1['status']}`**",
        f"- recommendation: {g1['recommendation']}",
    ])
    if g1["status"] != "not_found":
        lines.extend([
            f"- experiment: `{g1['experiment_id']}`",
            f"- control: `{g1['control_run_id']}`",
            f"- probability capture: `{g1['probability_run_id']}`",
            f"- same commit/GPU: "
            f"`{g1['same_git_commit']}/{g1['same_gpu_model']}`",
            f"- exact prediction SHA: `{g1['prediction_sha256_exact']}`",
            f"- exact hard metrics: `{g1['hard_metrics_exact']}`",
        ])
    lines.extend([
        "",
        (
            "This G1 can establish that enabling the audited probability "
            "capture path leaves hard output invariant *within one runtime*. "
            "It does not convert the cross-runtime 39-condition comparison "
            "into a causal softmax ablation, and no missing G1 result is "
            "fabricated."
        ),
        "",
        "## Condition ledger",
        "",
        (
            "| condition | hard equivalence | cases | SHA rate | max Dice "
            "diff | probability audit | config proof | strict pass |"
        ),
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in audit["conditions"]:
        condition_case_count = row["new_case_count"]
        condition_sha_rate = (
            row["prediction_sha256_match_count"] / condition_case_count
            if condition_case_count else 0.0)
        lines.append(
            f"| `{row['condition_id']}` | "
            f"`{row['hard_equivalence']}` | "
            f"{condition_case_count} | "
            f"{condition_sha_rate:.1%} | "
            f"{row['max_abs_diff_dice']} | "
            f"{'yes' if row['new_probability_audit_pass'] else 'no'} | "
            f"{'yes' if row['single_softmax_config_proof'] else 'no'} | "
            f"{'yes' if row['condition_pass'] else 'no'} |"
        )
    strict_issues = audit["strict_issues"]
    if strict_issues:
        lines.extend(["", "## Strict failures", ""])
        for issue in strict_issues:
            lines.append(f"- {issue}")
    lines.extend([
        "",
        "## Machine-readable artifacts",
        "",
        "- `probability_repair_audit.json`: complete audit and provenance.",
        (
            "- `probability_repair_condition_audit.csv`: one row per "
            "canonical condition."
        ),
        (
            "- `probability_repair_case_audit.csv`: one row per paired case, "
            "including every hard metric and corrected probability invariant."
        ),
    ])
    return "\n".join(lines)


def main(args: argparse.Namespace) -> int:
    results_root = Path(args.results_root).resolve()
    raw_root = results_root / "03_raw_runs"
    output_dir = results_root / "12_v2_analysis"
    conditions = build_conditions()

    old_runs, unexpected_old = load_experiment_runs(
        raw_root,
        conditions,
        experiment_id=OLD_EXPERIMENT,
        prefix=OLD_PREFIX,
        old_side=True,
    )
    new_runs, unexpected_new = load_experiment_runs(
        raw_root,
        conditions,
        experiment_id=NEW_EXPERIMENT,
        prefix=NEW_PREFIX,
        old_side=False,
    )

    condition_rows: List[Dict[str, Any]] = []
    case_rows: List[Dict[str, Any]] = []
    for condition in conditions:
        condition_row, condition_case_rows = compare_condition(
            condition,
            old_runs[condition.condition_id],
            new_runs[condition.condition_id],
            atol=args.atol,
            rtol=args.rtol,
            probability_sum_tolerance=args.probability_sum_tolerance,
        )
        condition_rows.append(condition_row)
        case_rows.extend(condition_case_rows)

    strict_issues: List[str] = []
    if unexpected_old:
        strict_issues.append(
            f"{OLD_EXPERIMENT} has unexpected {OLD_PREFIX}_* canonical runs: "
            + ", ".join(unexpected_old))
    if unexpected_new:
        strict_issues.append(
            f"{NEW_EXPERIMENT} has unexpected {NEW_PREFIX}_* canonical runs: "
            + ", ".join(unexpected_new))
    for row in condition_rows:
        if not row["condition_pass"]:
            strict_issues.append(
                f"{row['condition_id']}: {row['issues'] or 'failed'}")

    probability_errors = [
        row["new_probability_sum_max_abs_error"]
        for row in case_rows
        if row.get("new_probability_sum_max_abs_error") is not None
    ]
    probability_mins = [
        row["new_probability_min"]
        for row in case_rows
        if row.get("new_probability_min") is not None
    ]
    probability_maxs = [
        row["new_probability_max"]
        for row in case_rows
        if row.get("new_probability_max") is not None
    ]
    same_runtime_g1 = inspect_same_runtime_g1(raw_root)
    legacy_gap_conditions = sum(
        bool(row["old_probability_evidence_missing"])
        for row in condition_rows)
    legacy_missing_config = sum(
        not bool(row["old_run_config_present"])
        for row in condition_rows)

    totals = {
        "conditions_expected": len(conditions),
        "conditions_passed": sum(
            bool(row["condition_pass"]) for row in condition_rows),
        "case_rows_compared": len(case_rows),
        "strict_case_rows_passed": sum(
            bool(row["row_pass"]) for row in case_rows),
        "prediction_sha_matches": sum(
            bool(row["prediction_sha256_equal"]) for row in case_rows),
        "hard_metric_rows_within_tolerance": sum(
            bool(row["hard_metrics_within_tolerance"])
            for row in case_rows),
    }
    # Compatibility alias; unlike the original audit version this now means
    # the strict repair gate, not byte-exact cross-runtime equivalence.
    totals["case_rows_passed"] = totals["strict_case_rows_passed"]

    applicability_statuses = {
        str(row["hard_equivalence"]) for row in condition_rows}
    if "not_applicable_cross_runtime" in applicability_statuses:
        global_hard_status = "not_applicable_cross_runtime"
    elif "not_applicable_runtime_metadata_missing" in applicability_statuses:
        global_hard_status = "not_applicable_runtime_metadata_missing"
    elif applicability_statuses == {"applicable_same_runtime"}:
        global_hard_status = "applicable_same_runtime"
    else:
        global_hard_status = "mixed_applicability"
    metric_maxima: Dict[str, Optional[float]] = {}
    for metric in HARD_METRICS:
        values = [
            row.get(f"abs_diff_{metric}")
            for row in case_rows
            if row.get(f"abs_diff_{metric}") is not None
        ]
        metric_maxima[metric] = max(values) if values else None
    old_commits = sorted({
        str(row["old_git_commit"]) for row in condition_rows
        if row.get("old_git_commit")})
    new_commits = sorted({
        str(row["new_git_commit"]) for row in condition_rows
        if row.get("new_git_commit")})
    old_gpus = sorted({
        str(row["old_gpu_name"]) for row in condition_rows
        if row.get("old_gpu_name")})
    new_gpus = sorted({
        str(row["new_gpu_name"]) for row in condition_rows
        if row.get("new_gpu_name")})
    config_proof_passed = sum(
        bool(row["single_softmax_config_proof"])
        for row in condition_rows)
    verdict = "PASS" if not strict_issues else "FAIL"
    audit: Dict[str, Any] = {
        "protocol": PROTOCOL,
        "created_at": utc_now(),
        "strict_verdict": verdict,
        "strict_issues": strict_issues,
        "results_root": str(results_root),
        "output_dir": str(output_dir),
        "pairing": {
            "old_experiment": OLD_EXPERIMENT,
            "old_run_prefix": OLD_PREFIX,
            "new_experiment": NEW_EXPERIMENT,
            "new_run_prefix": NEW_PREFIX,
            "canonical_condition_count": 39,
            "retry_suffix_regex": RETRY_SUFFIX.pattern,
            "retry_policy": (
                "physical retries are collapsed by terminal __retryN; "
                "exactly one successful attempt is required per side and "
                "canonical condition"),
            "unexpected_old_runs": unexpected_old,
            "unexpected_new_runs": unexpected_new,
        },
        "hard_equivalence": global_hard_status,
        "hard_equivalence_details": {
            "status": global_hard_status,
            "strict_gate_policy": (
                "Exact prediction SHA is a strict gate only when a paired "
                "condition has the same manifest git_commit and runtime GPU "
                "model. Hard metrics are always reported descriptively."),
            "prediction_sha256_policy": (
                "exact 64-hex SHA256 equality for every selected case"),
            "prediction_sha256_match_count": totals[
                "prediction_sha_matches"],
            "prediction_sha256_case_count": totals["case_rows_compared"],
            "prediction_sha256_match_rate": (
                totals["prediction_sha_matches"]
                / totals["case_rows_compared"]
                if totals["case_rows_compared"] else 0.0),
            "hard_metrics": list(HARD_METRICS),
            "absolute_tolerance": args.atol,
            "relative_tolerance": args.rtol,
            "finite_values_required": True,
            "max_absolute_metric_differences": metric_maxima,
            "old_git_commits": old_commits,
            "new_git_commits": new_commits,
            "old_gpu_models": old_gpus,
            "new_gpu_models": new_gpus,
            "causal_scope": (
                "When commit or GPU differs, observed SHA mismatches and "
                "metric deltas cannot be attributed to the softmax repair."),
        },
        "single_softmax_configuration": {
            "conditions_checked": len(condition_rows),
            "conditions_passed": config_proof_passed,
            "strict_gate": True,
            "operational_definition": (
                "manifest.config_hash matches run_config.yaml; "
                "save_probability_signals=true; probability summary protocol "
                "is persisted_region_means; gate_alpha=null, "
                "temperature_scale=1, gate_transform=identity; and direct "
                "probability sum/min/max audit fields are present"),
            "scope_guard": (
                "The audit does not infer single-softmax behavior from "
                "confidence/entropy numbers alone."),
        },
        "corrected_probability_invariant": {
            "required_fields": list(PROBABILITY_AUDIT_FIELDS),
            "sum_tolerance": args.probability_sum_tolerance,
            "sum_policy": (
                "finite and 0 <= probability_sum_max_abs_error <= tolerance"),
            "range_policy": (
                "finite and 0 <= probability_min <= probability_max <= 1"),
            "cases_checked": len(case_rows),
            "cases_passed": sum(
                bool(row["new_probability_audit_pass"])
                for row in case_rows),
            "max_observed_sum_error": (
                max(probability_errors) if probability_errors else None),
            "min_observed_probability": (
                min(probability_mins) if probability_mins else None),
            "max_observed_probability": (
                max(probability_maxs) if probability_maxs else None),
        },
        "legacy_probability_evidence": {
            "verdict": (
                "INVALID_FOR_SOFT_PROBABILITY_CLAIMS"
                if legacy_gap_conditions == len(conditions)
                else "EVIDENCE_GAP_NOT_CONFIRMED_FOR_EVERY_CONDITION"),
            "conditions_checked": len(conditions),
            "conditions_with_evidence_gap": legacy_gap_conditions,
            "conditions_missing_run_config": legacy_missing_config,
            "required_but_missing_audit_fields": list(
                PROBABILITY_AUDIT_FIELDS),
            "reason": (
                "Historical case outputs do not provide the normalization "
                "and range audit fields required to establish a valid "
                "class-probability interface for confidence/entropy claims."),
            "scope_guard": (
                "This is an evidence/provenance finding. It does not assume "
                "that any particular historical probability number is wrong "
                "and does not invalidate hard predictions or hard metrics."),
        },
        "same_runtime_g1_invariant": same_runtime_g1,
        "totals": totals,
        "conditions": condition_rows,
        "cases": case_rows,
        "attempts": {
            "old": {
                condition.condition_id:
                    old_runs[condition.condition_id].candidate_attempts
                for condition in conditions
            },
            "new": {
                condition.condition_id:
                    new_runs[condition.condition_id].candidate_attempts
                for condition in conditions
            },
        },
    }

    condition_preferred = [
        "condition_id", "dataset_tag", "dataset", "corruption_type",
        "severity", "severity_unit", "realization",
        "old_expected_run_id", "old_actual_run_id",
        "new_expected_run_id", "new_actual_run_id",
        "old_case_count", "new_case_count",
        "condition_metadata_equal", "selected_cases_equal",
        "checkpoint_hash_equal", "data_split_hash_equal",
        "case_set_equal", "donor_case_assignments_equal",
        "hard_equivalence", "exact_sha_strict_gate",
        "same_git_commit", "same_gpu_model",
        "old_git_commit", "new_git_commit",
        "old_gpu_name", "new_gpu_name",
        "prediction_sha256_match_count",
        "prediction_sha256_all_equal",
        "hard_metrics_all_within_tolerance",
        "new_probability_audit_pass",
        "single_softmax_config_proof",
        "old_probability_evidence_missing",
        "condition_pass", "issues", "observations",
    ]
    case_preferred = [
        "condition_id", "dataset_tag", "dataset", "corruption_type",
        "severity", "severity_unit", "realization", "case_position", "case_id",
        "old_case_present", "new_case_present", "donor_case_equal",
        "old_prediction_sha256", "new_prediction_sha256",
        "prediction_sha256_equal",
    ]
    for metric in HARD_METRICS:
        case_preferred.extend([
            f"old_{metric}", f"new_{metric}", f"abs_diff_{metric}",
            f"{metric}_within_tolerance",
        ])
    case_preferred.extend([
        "hard_metrics_within_tolerance",
        "new_probability_sum_max_abs_error",
        "new_probability_min", "new_probability_max",
        "new_probability_sum_pass", "new_probability_range_pass",
        "new_probability_audit_pass",
        "descriptive_hard_equivalence_pass",
        "repair_gate_case_pass", "row_pass", "issues",
    ])

    atomic_json(output_dir / "probability_repair_audit.json", audit)
    atomic_csv(
        output_dir / "probability_repair_condition_audit.csv",
        condition_rows,
        condition_preferred,
    )
    atomic_csv(
        output_dir / "probability_repair_case_audit.csv",
        case_rows,
        case_preferred,
    )
    atomic_text(
        output_dir / "PROBABILITY_REPAIR_AUDIT.md",
        markdown_report(audit),
    )

    print(json.dumps({
        "protocol": PROTOCOL,
        "strict_verdict": verdict,
        "conditions_passed": totals["conditions_passed"],
        "conditions_expected": totals["conditions_expected"],
        "strict_case_rows_passed": totals["strict_case_rows_passed"],
        "case_rows_compared": totals["case_rows_compared"],
        "hard_equivalence": global_hard_status,
        "output_dir": str(output_dir),
    }, ensure_ascii=False, sort_keys=True))
    return 0 if verdict == "PASS" else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Strictly compare the 39 historical and corrected TCFF "
            "probability-signal conditions."))
    parser.add_argument(
        "--results_root",
        required=True,
        help="TCFF rebuttal results root containing 03_raw_runs.",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-9,
        help="Absolute tolerance for hard metrics (default: 1e-9).",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-9,
        help="Relative tolerance for hard metrics (default: 1e-9).",
    )
    parser.add_argument(
        "--probability_sum_tolerance",
        type=float,
        default=1e-4,
        help=(
            "Maximum corrected per-case probability sum-to-one absolute "
            "error (default: 1e-4)."),
    )
    parsed_args = parser.parse_args()
    if parsed_args.atol < 0 or parsed_args.rtol < 0:
        parser.error("--atol and --rtol must be non-negative")
    if parsed_args.probability_sum_tolerance < 0:
        parser.error("--probability_sum_tolerance must be non-negative")
    raise SystemExit(main(parsed_args))
