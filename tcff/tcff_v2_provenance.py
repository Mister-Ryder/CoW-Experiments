"""Generate conservative, stdlib-only TCFF provenance ledgers.

Only small text/config/manifest files are read.  Checkpoints are stat'ed but
never loaded or hashed, and parquet/NPZ/NIfTI files are never scanned.
Unverified claims are explicitly labelled ``pending_verification``.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import xml.etree.ElementTree as ET
import zipfile


PROTOCOL = "TCFF_V2_PROVENANCE_LEDGER_2026-07-30"
TEXT_SUFFIXES = {
    ".md", ".txt", ".rst", ".tex", ".py", ".json", ".yaml", ".yml"}
MAX_TEXT_BYTES = 5 * 1024 * 1024

BASELINE_SPECS = (
    ("WingsNet", "published_baseline", (r"\bWingsNet\b",), None),
    ("NaviAirway", "published_baseline", (r"\bNaviAirway\b",), None),
    ("PSAR", "published_baseline", (r"\bPSAR\b",), None),
    ("ERNet", "published_baseline", (r"\bERNet\b",), None),
    ("ROSE", "published_baseline", (r"\bROSE\b",), None),
    ("FANN", "published_baseline", (r"\bFANN\b",), None),
    ("TCFF", "proposed_method", (r"\bTCFF\b", r"trust_variant"), "sam_tgf_v3"),
    ("Concat", "fusion_baseline", (r"\bConcat\b", r"\bconcat\b"), "sam_concat"),
    ("BAGF", "fusion_baseline", (r"\bBAGF\b", r"\bbagf\b"), "sam_bagf"),
    ("DSFF", "fusion_baseline", (r"\bDSFF\b", r"\bdsff\b"), "sam_dsff"),
    (
        "CrossAttn", "fusion_baseline",
        (r"\bCrossAttn\b", r"\bcrossattn\b"), "sam_crossattn"),
    ("AFF", "fusion_baseline", (r"\bAFF\b", r"\baff\b"), "sam_aff"),
)
TRAINING_VARIANTS = (
    "full", "product", "g_only", "u_only", "additive", "residual", "no_aux")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def read_text(path: Path) -> Optional[str]:
    try:
        if path.stat().st_size > MAX_TEXT_BYTES:
            return None
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None


def read_docx_text(path: Path) -> Optional[str]:
    try:
        with zipfile.ZipFile(path) as archive:
            payload = archive.read("word/document.xml")
        root = ET.fromstring(payload)
        words = [
            node.text or "" for node in root.iter()
            if node.tag.endswith("}t")]
        return " ".join(words)
    except Exception:
        return None


def relative_or_absolute(path: Path, project_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(project_root.resolve()))
    except ValueError:
        return str(path.resolve())


def document_candidates(project_root: Path) -> List[Path]:
    candidates: List[Path] = []
    generator_path = Path(__file__).resolve()
    for path in project_root.iterdir():
        if (
                path.is_file() and
                path.resolve() != generator_path and
                path.suffix.lower() in TEXT_SUFFIXES):
            candidates.append(path)
    for path in project_root.rglob("*README*"):
        if path.is_file() and path.stat().st_size <= MAX_TEXT_BYTES:
            candidates.append(path)
    paper_root = project_root.parent / "paper and review"
    if paper_root.is_dir():
        for path in paper_root.iterdir():
            if (
                    path.is_file() and
                    (path.suffix.lower() in TEXT_SUFFIXES or
                     path.suffix.lower() == ".docx")):
                candidates.append(path)
    return sorted(set(candidates))


def load_evidence_documents(
        project_root: Path) -> Tuple[List[Tuple[Path, str]], List[str]]:
    loaded: List[Tuple[Path, str]] = []
    notes: List[str] = []
    paper_root = project_root.parent / "paper and review"
    if not paper_root.is_dir():
        notes.append(
            "Sibling 'paper and review' directory was not available; "
            "paper/review claims remain pending where no other source exists.")
    else:
        pdfs = sorted(paper_root.glob("*.pdf"))
        if pdfs:
            notes.append(
                "PDF files were intentionally not text-extracted by this "
                "stdlib-only generator: " +
                ", ".join(path.name for path in pdfs))
    for path in document_candidates(project_root):
        text = (
            read_docx_text(path)
            if path.suffix.lower() == ".docx" else read_text(path))
        if text is None:
            notes.append(f"Unreadable/oversized text source skipped: {path}")
        else:
            loaded.append((path, text))
    return loaded, notes


def compact_excerpt(text: str, match: re.Match, width: int = 90) -> str:
    start = max(0, match.start() - width)
    end = min(len(text), match.end() + width)
    value = re.sub(r"\s+", " ", text[start:end]).strip()
    return value[:240]


def find_mentions(
        patterns: Sequence[str],
        documents: Sequence[Tuple[Path, str]],
        project_root: Path,
        suffix_filter: Optional[Sequence[str]] = None) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    allowed = set(suffix_filter or ())
    compiled = [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
    for path, text in documents:
        if allowed and path.suffix.lower() not in allowed:
            continue
        for pattern in compiled:
            match = pattern.search(text)
            if match:
                rows.append({
                    "source": relative_or_absolute(path, project_root),
                    "excerpt": compact_excerpt(text, match),
                })
                break
        if len(rows) >= 8:
            break
    return rows


def method_paths(
        project_root: Path, directory_name: Optional[str]) -> Tuple[
            List[Path], List[Path]]:
    if directory_name is None:
        return [], []
    candidate_dirs = (
        project_root / "Runs_foundation" / directory_name,
        project_root / "Runs_foundation_v2" / directory_name,
    )
    checkpoints: List[Path] = []
    configs: List[Path] = []
    for directory in candidate_dirs:
        checkpoint = directory / "best_model.pth"
        if checkpoint.is_file():
            checkpoints.append(checkpoint)
        for name in ("config.json", "config.yaml", "config.yml"):
            config = directory / name
            if config.is_file():
                configs.append(config)
    return checkpoints, configs


def manifest_records(results_root: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    raw_root = results_root / "03_raw_runs"
    if not raw_root.is_dir():
        return records
    for path in sorted(raw_root.rglob("run_manifest.json")):
        manifest = read_json(path)
        if not manifest:
            continue
        records.append({
            "path": path,
            "manifest": manifest,
            "status": read_json(path.parent / "status.json").get("status"),
        })
    return records


def manifest_matches_method(manifest: Mapping[str, Any], method: str) -> bool:
    variant = str(manifest.get("model_variant") or "").lower()
    if method == "TCFF":
        return variant in TRAINING_VARIANTS
    experiment = str(manifest.get("experiment_id") or "").lower()
    run_id = str(manifest.get("run_id") or "").lower()
    token = method.lower()
    return token in experiment or token in run_id


def join_values(values: Iterable[str]) -> str:
    return ";".join(sorted({value for value in values if value}))


def baseline_rows(
        project_root: Path,
        documents: Sequence[Tuple[Path, str]],
        manifests: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for method, category, patterns, directory_name in BASELINE_SPECS:
        mentions = find_mentions(patterns, documents, project_root)
        source_mentions = find_mentions(
            patterns, documents, project_root, suffix_filter=(".py",))
        checkpoints, configs = method_paths(project_root, directory_name)
        matched_manifests = [
            record for record in manifests
            if manifest_matches_method(record["manifest"], method)]
        concrete = bool(source_mentions and checkpoints and configs)
        verification_status = (
            "verified_local_artifact_presence"
            if concrete else "pending_verification")
        rows.append({
            "method": method,
            "category": category,
            "verification_status": verification_status,
            "documentation_mentions_found": len(mentions),
            "documentation_sources": join_values(
                item["source"] for item in mentions),
            "implementation_mentions_found": len(source_mentions),
            "implementation_sources": join_values(
                item["source"] for item in source_mentions),
            "checkpoint_paths": join_values(
                relative_or_absolute(path, project_root)
                for path in checkpoints),
            "config_paths": join_values(
                relative_or_absolute(path, project_root)
                for path in configs),
            "run_manifest_records": len(matched_manifests),
            "hash_fields_observed": sum(
                bool(record["manifest"].get("checkpoint_hash"))
                for record in matched_manifests),
            "claim_boundary": (
                "Local code/config/checkpoint existence only; no performance, "
                "paper-equivalence, or official-baseline claim is inferred."
                if concrete else
                "Insufficient direct local implementation/config/checkpoint "
                "evidence; pending_verification."),
        })
    for variant in TRAINING_VARIANTS:
        method = f"TCFF/{variant}"
        matched = [
            record for record in manifests
            if str(record["manifest"].get("model_variant")) == variant]
        checkpoints, configs = method_paths(
            project_root, "sam_tgf_v3")
        rows.append({
            "method": method,
            "category": "training_or_runtime_variant",
            "verification_status": (
                "verified_local_artifact_presence"
                if checkpoints and configs and matched
                else "pending_verification"),
            "documentation_mentions_found": 0,
            "documentation_sources": "",
            "implementation_mentions_found": 1,
            "implementation_sources": "topcow_foundation_model.py",
            "checkpoint_paths": join_values(
                relative_or_absolute(path, project_root)
                for path in checkpoints),
            "config_paths": join_values(
                relative_or_absolute(path, project_root)
                for path in configs),
            "run_manifest_records": len(matched),
            "hash_fields_observed": sum(
                bool(record["manifest"].get("checkpoint_hash"))
                for record in matched),
            "claim_boundary": (
                "Variant token and local TCFF artifacts observed; independent "
                "training is established only by a corresponding successful "
                "training manifest with trained_checkpoint_hash."),
        })
    return rows


def base_ledger_rows(project_root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    identities = [
        (method, method, "", directory_name)
        for method, _, _, directory_name in BASELINE_SPECS
    ] + [
        (f"TCFF/{variant}", "TCFF", variant, "sam_tgf_v3")
        for variant in TRAINING_VARIANTS
    ]
    for identity, method, variant, directory_name in identities:
        checkpoints, configs = method_paths(project_root, directory_name)
        if not checkpoints and not configs:
            rows.append({
                "identity": identity,
                "method": method,
                "variant": variant,
                "evidence_kind": "required_placeholder",
                "experiment_id": "",
                "run_id": "",
                "seed": "",
                "status": "",
                "checkpoint_path": "",
                "checkpoint_exists": False,
                "checkpoint_size_bytes": "",
                "checkpoint_hash": "",
                "checkpoint_hash_source": "",
                "config_path": "",
                "config_exists": False,
                "config_file_sha256": "",
                "manifest_path": "",
                "manifest_config_hash": "",
                "data_split_hash": "",
                "git_commit": "",
                "protocol_status": "pending_verification",
                "notes": "No local checkpoint/config mapping was verified.",
            })
            continue
        checkpoint = checkpoints[0] if checkpoints else None
        config = configs[0] if configs else None
        rows.append({
            "identity": identity,
            "method": method,
            "variant": variant,
            "evidence_kind": "local_config_checkpoint_pair",
            "experiment_id": "",
            "run_id": "",
            "seed": "",
            "status": "",
            "checkpoint_path": (
                relative_or_absolute(checkpoint, project_root)
                if checkpoint else ""),
            "checkpoint_exists": bool(checkpoint),
            "checkpoint_size_bytes": (
                checkpoint.stat().st_size if checkpoint else ""),
            "checkpoint_hash": "",
            "checkpoint_hash_source": (
                "not_computed_large_binary" if checkpoint else ""),
            "config_path": (
                relative_or_absolute(config, project_root) if config else ""),
            "config_exists": bool(config),
            "config_file_sha256": sha256_file(config) if config else "",
            "manifest_path": "",
            "manifest_config_hash": "",
            "data_split_hash": "",
            "git_commit": "",
            "protocol_status": (
                "artifact_paths_verified"
                if checkpoint and config else "pending_verification"),
            "notes": (
                "Existence and small-config SHA256 verified; checkpoint "
                "content was not loaded or hashed."),
        })
    return rows


def identity_from_manifest(manifest: Mapping[str, Any]) -> Tuple[str, str, str]:
    variant = str(manifest.get("model_variant") or "")
    if variant in TRAINING_VARIANTS:
        return f"TCFF/{variant}", "TCFF", variant
    run_id = str(manifest.get("run_id") or "").lower()
    experiment = str(manifest.get("experiment_id") or "").lower()
    for method, _, _, _ in BASELINE_SPECS:
        if method == "TCFF":
            continue
        if method.lower() in run_id or method.lower() in experiment:
            return method, method, ""
    return "unmapped", "unmapped", variant


def manifest_ledger_rows(
        project_root: Path,
        records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for record in records:
        manifest = record["manifest"]
        identity, method, variant = identity_from_manifest(manifest)
        if identity == "unmapped":
            continue
        path = record["path"]
        checkpoint_hash = (
            manifest.get("trained_checkpoint_hash") or
            manifest.get("checkpoint_hash") or "")
        rows.append({
            "identity": identity,
            "method": method,
            "variant": variant,
            "evidence_kind": "run_manifest",
            "experiment_id": manifest.get("experiment_id", ""),
            "run_id": manifest.get("run_id", path.parent.name),
            "seed": manifest.get("seed", ""),
            "status": record.get("status") or manifest.get("status", ""),
            "checkpoint_path": "",
            "checkpoint_exists": "",
            "checkpoint_size_bytes": "",
            "checkpoint_hash": checkpoint_hash,
            "checkpoint_hash_source": (
                "trained_checkpoint_hash"
                if manifest.get("trained_checkpoint_hash")
                else "checkpoint_hash"
                if manifest.get("checkpoint_hash") else ""),
            "config_path": relative_or_absolute(
                path.parent / "run_config.yaml", project_root)
            if (path.parent / "run_config.yaml").is_file() else "",
            "config_exists": (path.parent / "run_config.yaml").is_file(),
            "config_file_sha256": (
                sha256_file(path.parent / "run_config.yaml")
                if (path.parent / "run_config.yaml").is_file() else ""),
            "manifest_path": relative_or_absolute(path, project_root),
            "manifest_config_hash": manifest.get("config_hash", ""),
            "data_split_hash": manifest.get("data_split_hash", ""),
            "git_commit": manifest.get("git_commit", ""),
            "protocol_status": (
                "manifest_success_with_hash_fields"
                if (
                    (record.get("status") or manifest.get("status")) ==
                    "success" and checkpoint_hash and
                    manifest.get("config_hash"))
                else "pending_verification"),
            "notes": (
                "Values transcribed from run_manifest/status; checkpoint "
                "bytes were not independently rehashed."),
        })
    return rows


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def provenance_notes(
        project_root: Path,
        results_root: Path,
        baseline: Sequence[Mapping[str, Any]],
        ledger: Sequence[Mapping[str, Any]],
        source_notes: Sequence[str]) -> str:
    verified = sum(
        row["verification_status"] != "pending_verification"
        for row in baseline)
    pending = len(baseline) - verified
    manifest_rows = sum(
        row["evidence_kind"] == "run_manifest" for row in ledger)
    lines = [
        "# TCFF V2 Provenance Notes",
        "",
        f"- Protocol: `{PROTOCOL}`",
        f"- Generated UTC: `{dt.datetime.now(dt.timezone.utc).isoformat()}`",
        f"- Project root: `{project_root}`",
        f"- Results root: `{results_root}`",
        f"- Baseline/variant entries: `{len(baseline)}`",
        f"- Local-artifact entries: `{verified}`",
        f"- Pending-verification entries: `{pending}`",
        f"- Run-manifest ledger rows: `{manifest_rows}`",
        "",
        "## Claim boundary",
        "",
        (
            "This generator records only literal source/document mentions, "
            "existing paths, small-config SHA256 values, and hash/protocol "
            "fields already stored in run manifests. It does not infer "
            "architectures, training schedules, paper equivalence, official "
            "baseline status, or performance from a method name."
        ),
        (
            "Checkpoint binaries are not loaded and are not hashed here. "
            "A manifest-provided hash is labelled as such and is not presented "
            "as an independent re-verification."
        ),
        (
            "Every unsupported or incomplete claim remains exactly "
            "`pending_verification`."
        ),
        "",
        "## Source limitations",
        "",
    ]
    lines.extend(
        f"- {note}" for note in source_notes
    )
    if not source_notes:
        lines.append("- No source-discovery warnings.")
    lines.extend([
        "",
        "## Output files",
        "",
        "- `baseline_provenance.csv`",
        "- `checkpoint_protocol_ledger.csv`",
        "- `V2_PROVENANCE_NOTES.md`",
        "",
    ])
    return "\n".join(lines)


def main(args: argparse.Namespace) -> int:
    project_root = Path(args.project_root).resolve()
    results_root = Path(args.results_root).resolve()
    output_root = results_root / "00_project_discovery"
    documents, source_notes = load_evidence_documents(project_root)
    manifests = manifest_records(results_root)
    baseline = baseline_rows(
        project_root, documents, manifests)
    ledger = (
        base_ledger_rows(project_root) +
        manifest_ledger_rows(project_root, manifests))
    write_csv(output_root / "baseline_provenance.csv", baseline)
    write_csv(output_root / "checkpoint_protocol_ledger.csv", ledger)
    write_text(
        output_root / "V2_PROVENANCE_NOTES.md",
        provenance_notes(
            project_root, results_root, baseline, ledger, source_notes))
    print(json.dumps({
        "protocol": PROTOCOL,
        "baseline_entries": len(baseline),
        "ledger_rows": len(ledger),
        "manifest_rows_scanned": len(manifests),
        "output_root": str(output_root),
    }, sort_keys=True))
    return 0


def parser() -> argparse.ArgumentParser:
    here = Path(__file__).resolve().parent
    value = argparse.ArgumentParser(
        description="Generate conservative TCFF V2 provenance ledgers.")
    value.add_argument("--project_root", default=str(here))
    value.add_argument(
        "--results_root", default=str(here / "results_tcff_rebuttal"))
    return value


if __name__ == "__main__":
    raise SystemExit(main(parser().parse_args()))
