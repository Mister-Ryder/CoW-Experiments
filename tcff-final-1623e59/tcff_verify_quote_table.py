"""Independently verify every value in exact_numbers_safe_to_quote.csv."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED = {
    "claim_id", "description", "value", "unit", "ci_low", "ci_high",
    "dataset", "variant", "comparison", "source_file", "source_row",
    "source_column", "verified_by",
}


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def update_g5_acceptance(root: Path, verification_passed: bool) -> None:
    path = root / "02_manifests" / "G5_ACCEPTANCE.md"
    if not path.exists():
        return
    lines = path.read_text(encoding="utf-8").splitlines()
    updated = []
    for line in lines:
        if "current_quote_ledger_independently_verified" in line:
            mark = "x" if verification_passed else " "
            line = f"- [{mark}] current_quote_ledger_independently_verified"
        updated.append(line)
    checks = [line for line in updated if line.startswith("- [")]
    passed = bool(checks and all(line.startswith("- [x]") for line in checks))
    for index, line in enumerate(updated):
        if line.startswith("Status: **"):
            updated[index] = f"Status: **{'PASS' if passed else 'PENDING'}**"
            break
    atomic_text(path, "\n".join(updated) + "\n")


def inside_root(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def finite(value) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def main(args: argparse.Namespace) -> int:
    root = Path(args.results_root).resolve()
    ledger_path = root / "06_summary_tables" / "exact_numbers_safe_to_quote.csv"
    if not ledger_path.exists():
        raise FileNotFoundError(ledger_path)
    ledger = pd.read_csv(ledger_path)
    ledger_hash = sha256_file(ledger_path)
    root_ledger = root / "exact_numbers_safe_to_quote.csv"
    root_ledger_hash = sha256_file(root_ledger) if root_ledger.exists() else None
    root_ledger_matches = root_ledger_hash == ledger_hash
    missing = sorted(REQUIRED - set(ledger.columns))
    checks = []
    cache = {}
    duplicate_ids = set(ledger.loc[ledger["claim_id"].duplicated(False),
                                   "claim_id"].astype(str))
    for index, row in ledger.iterrows():
        errors = []
        claim_id = str(row.get("claim_id", ""))
        if missing:
            errors.append("missing ledger columns: " + ",".join(missing))
        if claim_id in duplicate_ids:
            errors.append("duplicate claim_id")
        if not finite(row.get("value")):
            errors.append("non-finite claim value")
        source_relative = str(row.get("source_file", ""))
        source = root / source_relative
        if not inside_root(source, root):
            errors.append("source path escapes results root")
        elif source.suffix.lower() != ".csv":
            errors.append("source is not a CSV")
        elif not source.exists():
            errors.append("source file missing")
        else:
            if source not in cache:
                cache[source] = pd.read_csv(source)
            frame = cache[source]
            try:
                source_line = int(row.get("source_row"))
                source_index = source_line - 2
            except (TypeError, ValueError):
                source_index = -1
            column = str(row.get("source_column", ""))
            if source_index < 0 or source_index >= len(frame):
                errors.append("source_row outside CSV")
            elif column not in frame.columns:
                errors.append("source_column missing")
            else:
                source_value = frame.iloc[source_index][column]
                if not finite(source_value):
                    errors.append("source value non-finite")
                elif not np.isclose(float(source_value), float(row["value"]),
                                    rtol=args.rtol, atol=args.atol):
                    errors.append(
                        f"value mismatch ledger={row['value']} source={source_value}")
        ci_low = row.get("ci_low")
        ci_high = row.get("ci_high")
        if pd.notna(ci_low) or pd.notna(ci_high):
            if not (finite(ci_low) and finite(ci_high)):
                errors.append("only one CI endpoint is finite")
            elif float(ci_low) > float(ci_high):
                errors.append("CI endpoints reversed")
        checks.append({
            "claim_id": claim_id,
            "source_file": source_relative,
            "source_row": row.get("source_row"),
            "source_column": row.get("source_column"),
            "status": "pass" if not errors else "fail",
            "errors": "; ".join(errors),
        })
    check_frame = pd.DataFrame(checks)
    output_csv = root / "11_quality_audit" / "quote_table_verification.csv"
    atomic_csv(output_csv, check_frame)
    n_failed = int(check_frame["status"].eq("fail").sum()) if not check_frame.empty else 0
    result = {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "ledger": ledger_path.relative_to(root).as_posix(),
        "n_claims": int(len(ledger)),
        "n_passed": int(len(ledger) - n_failed),
        "n_failed": n_failed,
        "status": ("pass" if n_failed == 0 and not missing and
                   root_ledger_matches else "fail"),
        "ledger_sha256": ledger_hash,
        "root_ledger_sha256": root_ledger_hash,
        "root_ledger_matches": root_ledger_matches,
        "relative_tolerance": args.rtol,
        "absolute_tolerance": args.atol,
        "missing_required_columns": missing,
    }
    atomic_json(root / "11_quality_audit" / "quote_table_verification.json",
                result)
    update_g5_acceptance(root, result["status"] == "pass")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result["status"] == "pass" else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_root", default=str(
        Path(__file__).resolve().parent / "results_tcff_rebuttal"))
    parser.add_argument("--rtol", type=float, default=1e-12)
    parser.add_argument("--atol", type=float, default=1e-12)
    raise SystemExit(main(parser.parse_args()))
