#!/usr/bin/env python3
"""Validate the compact project-state authority layer.

The checker is intentionally narrow.  It validates the new state layer and the
registry artifact pointers; it does not parse historical reports.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = [
    "AGENTS.md",
    "CURRENT_STATE.md",
    "README.md",
    "CONTEXT_MANAGEMENT.md",
    "PROJECT_PROMPT_CN.md",
    "docs/research/research_state.md",
    "docs/research/open_questions.md",
    "docs/state/THEOREM_REGISTRY.md",
    "docs/state/RESULT_REGISTRY.json",
    "docs/state/DECISION_LOG.md",
    "docs/state/FILE_STATUS.json",
    "docs/state/STATE_MIGRATION_REPORT.md",
    "active/README.md",
    "active/TASK_TEMPLATE.md",
    "active/CONTEXT_TEMPLATE.md",
    "active/STATE_DELTA_TEMPLATE.md",
]

AUTHORITY_FILES = [
    "README.md",
    "AGENTS.md",
    "CURRENT_STATE.md",
    "CONTEXT_MANAGEMENT.md",
    "PROJECT_PROMPT_CN.md",
    "docs/research/research_state.md",
    "docs/research/open_questions.md",
    "docs/state/THEOREM_REGISTRY.md",
    "docs/state/DECISION_LOG.md",
    "docs/state/STATE_MIGRATION_REPORT.md",
]

LINE_LIMITS = {
    "AGENTS.md": 250,
    "CURRENT_STATE.md": 350,
    "docs/state/THEOREM_REGISTRY.md": 600,
    "active/CONTEXT_TEMPLATE.md": 300,
}


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def _load_json(path: str):
    return json.loads(_read(path))


def _ids_from_markdown(path: str, prefix: str) -> list[str]:
    text = _read(path)
    return re.findall(rf"^##\s+({re.escape(prefix)}[A-Z0-9-]+)\b", text, flags=re.MULTILINE)


def _check_unique(ids: Iterable[str], label: str, errors: list[str]) -> None:
    seen: set[str] = set()
    for item in ids:
        if item in seen:
            errors.append(f"duplicate {label}: {item}")
        seen.add(item)


def _looks_like_path(token: str) -> bool:
    cleaned = _clean_path(token)
    known_files = {
        "AGENTS.md", "CURRENT_STATE.md", "README.md", "CONTEXT_MANAGEMENT.md",
        "PROJECT_PROMPT_CN.md",
    }
    known_prefixes = (
        "active/", "docs/", "round3_redesign/", "src/", "tests/",
        "formalization/", "scripts/",
    )
    return cleaned in known_files or cleaned.startswith(known_prefixes)


def _clean_path(token: str) -> str:
    cleaned = token.strip().strip("`'\".,;:)")
    cleaned = cleaned.split("#", 1)[0]
    if cleaned.startswith("/"):
        cleaned = cleaned[1:]
    return cleaned


def _markdown_paths(path: str) -> list[str]:
    text = _read(path)
    paths: list[str] = []
    for match in re.findall(r"\[[^\]\n]+\]\(([^)\n]+)\)", text):
        paths.append(_clean_path(match))
    for match in re.findall(r"`([^`]+)`", text):
        if _looks_like_path(match):
            paths.append(_clean_path(match))
    return [p for p in paths if p and not p.startswith("<") and "MECH-001/C011" not in p]


def _path_exists(path: str, base: Path = ROOT) -> bool:
    if "*" in path:
        return bool(list(ROOT.glob(path)))
    candidate = base / path
    if candidate.exists():
        return True
    return (ROOT / path).exists()


def _check_path_references(errors: list[str]) -> None:
    for doc in [
        "CURRENT_STATE.md",
        "docs/state/THEOREM_REGISTRY.md",
        "docs/state/DECISION_LOG.md",
    ]:
        base = (ROOT / doc).parent
        for ref in _markdown_paths(doc):
            if ref.startswith("http"):
                continue
            if not _path_exists(ref, base):
                errors.append(f"dead path reference in {doc}: {ref}")

    for row in _load_json("docs/state/RESULT_REGISTRY.json"):
        for key in ("artifact_path", "supporting_artifact_paths"):
            value = row.get(key)
            values = value if isinstance(value, list) else [value]
            for ref in values:
                if ref and not _path_exists(str(ref)):
                    errors.append(f"dead result artifact path {row.get('id')}: {ref}")

    for row in _load_json("docs/state/FILE_STATUS.json"):
        ref = str(row.get("path_or_glob", ""))
        if row.get("historical_git_only"):
            continue
        if not ref or ref.endswith("/**"):
            continue
        if "*" in ref:
            if not list(ROOT.glob(ref)):
                errors.append(f"FILE_STATUS glob matches nothing: {ref}")
        elif not (ROOT / ref).exists():
            errors.append(f"FILE_STATUS path missing: {ref}")


def _bad_current_claim(line: str) -> bool:
    lower = line.lower()
    safe_words = ("not", "historical", "superseded", "compatibility", "closed history")
    if "latent-001" in lower and "current" in lower and not any(word in lower for word in safe_words):
        return True
    if "semantic latent" in lower and "current mainline" in lower and not any(word in lower for word in safe_words):
        return True
    if "round-2 quotient" in lower and "current" in lower and not any(word in lower for word in safe_words):
        return True
    if "task 1" in lower and "task 2" in lower and "pending" in lower:
        return True
    if "repair gate" in lower and "blocked" in lower:
        return True
    return False


def _check_stale_authority(errors: list[str]) -> None:
    for path in AUTHORITY_FILES:
        for index, line in enumerate(_read(path).splitlines(), start=1):
            if _bad_current_claim(line):
                errors.append(f"stale current-state claim in {path}:{index}: {line}")


def _boot_context_stats() -> dict[str, int]:
    files = ["AGENTS.md", "CURRENT_STATE.md"]
    if (ROOT / "active/TASK.md").exists():
        files.append("active/TASK.md")
    if (ROOT / "active/CONTEXT.md").exists():
        files.append("active/CONTEXT.md")
    line_count = 0
    byte_count = 0
    for path in files:
        text = _read(path)
        line_count += len(text.splitlines())
        byte_count += len(text.encode("utf-8"))
    return {
        "file_count": len(files),
        "line_count": line_count,
        "byte_count": byte_count,
        "approximate_token_count": max(1, byte_count // 4),
    }


def check() -> dict[str, object]:
    errors: list[str] = []
    warnings: list[str] = []

    for path in REQUIRED_FILES:
        if not (ROOT / path).exists():
            errors.append(f"missing required file: {path}")

    result_registry = _load_json("docs/state/RESULT_REGISTRY.json")
    file_status = _load_json("docs/state/FILE_STATUS.json")
    if not isinstance(result_registry, list):
        errors.append("RESULT_REGISTRY.json must be a list")
    if not isinstance(file_status, list):
        errors.append("FILE_STATUS.json must be a list")

    theorem_ids = _ids_from_markdown("docs/state/THEOREM_REGISTRY.md", "T-")
    decision_ids = _ids_from_markdown("docs/state/DECISION_LOG.md", "D-")
    _check_unique(theorem_ids, "theorem id", errors)
    _check_unique(decision_ids, "decision id", errors)

    required_theorems = {
        "T-3A-A", "T-3C-PI", "T-3C-COMMON-BASE", "T-3D-IDENTIFIABILITY",
        "T-3E-INFO-FLOOR", "T-3E-SPECTRAL-SLACK", "T-3E-AFFINE-REGRET",
    }
    missing_theorems = required_theorems.difference(theorem_ids)
    if missing_theorems:
        errors.append(f"missing theorem ids: {sorted(missing_theorems)}")

    required_decisions = {
        "D-SEMANTIC-LATENT-SUPERSEDED", "D-ROUND2-QUOTIENT-SUPERSEDED",
        "D-REG-EXPOSURE-CONTAINMENT-REJECTED", "D-ACTUAL-VS-COMMON-BASE-SEPARATION",
        "D-SHARP-SLACK-REPLACES-E0", "D-TASK3-NEXT", "D-MECH001-C011-DO-NOT-RESTORE",
    }
    missing_decisions = required_decisions.difference(decision_ids)
    if missing_decisions:
        errors.append(f"missing decision ids: {sorted(missing_decisions)}")

    current = _read("CURRENT_STATE.md")
    if "Task 3 is next" not in current:
        errors.append("CURRENT_STATE.md must identify Task 3 as next")
    if "Task 3 is not executed" not in current:
        errors.append("CURRENT_STATE.md must say Task 3 is not executed")

    for bad in ("CURRENT_STATE_v2.md", "LATEST_STATE.md", "STATE_SUMMARY.md", "PROJECT_STATE_NEW.md"):
        if (ROOT / bad).exists():
            errors.append(f"competing current-state file exists: {bad}")

    mech_records = [
        row for row in file_status
        if "MECH-001/C011" in str(row.get("path_or_glob", ""))
    ]
    if not mech_records or not all(row.get("do_not_restore") for row in mech_records):
        errors.append("MECH-001/C011 must be marked do_not_restore")

    _check_path_references(errors)
    _check_stale_authority(errors)

    for path, limit in LINE_LIMITS.items():
        if (ROOT / path).exists():
            lines = len(_read(path).splitlines())
            if lines > limit:
                warnings.append(f"{path} has {lines} lines over target {limit}")

    reconstruction = {
        "current_project_stage": "Round 3 redesign after Task 1, Task 2, and repair evidence gate",
        "frozen_theory_ids": sorted(required_theorems),
        "rejected_or_superseded": [
            "D-SEMANTIC-LATENT-SUPERSEDED",
            "D-ROUND2-QUOTIENT-SUPERSEDED",
            "D-REG-EXPOSURE-CONTAINMENT-REJECTED",
            "D-SHARP-SLACK-REPLACES-E0",
        ],
        "next_task": "Task 3 applicability",
    }

    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "theorem_id_count": len(theorem_ids),
        "decision_id_count": len(decision_ids),
        "result_id_count": len(result_registry) if isinstance(result_registry, list) else 0,
        "boot_context": _boot_context_stats(),
        "state_reconstruction": reconstruction,
    }


def main() -> int:
    result = check()
    print(json.dumps(result, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
