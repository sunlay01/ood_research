"""Runner for TASK3-OOD-CAPABILITY-DECOMPOSITION-FIRST-ROUND."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

from .task3_ood_capability_decomposition.analysis import (
    ALLOWED_DOMINANT_BOTTLENECKS,
    ALLOWED_VERDICTS,
    paired_capability_rows,
    summarize_capabilities,
)
from .task3_ood_capability_decomposition.artifacts import (
    CheckpointRecord,
    build_feature_bundle,
    load_checkpoint_manifest,
    sha256_file,
)
from .task3_ood_capability_decomposition.experiments import RANKS, RIDGE, CapabilityRows, run_capability_experiments


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs/task3_cmnist_cpu_minimal.json"
MANIFEST_PATH = ROOT / "round3_redesign/task3_cmnist_counterfactual_audit/results/checkpoint_manifest.csv"
OUT_DIR = ROOT / "round3_redesign/ood_capability_decomposition"
RESULTS_DIR = OUT_DIR / "results"
STATE_DELTA_PATH = ROOT / "active/STATE_DELTA.md"
TASK_ID = "TASK3-OOD-CAPABILITY-DECOMPOSITION-FIRST-ROUND"
SEEDS = [10, 11, 12, 13, 14]
METHODS = ["ERM", "IRMv1"]
EXECUTED_EXPERIMENTS = ["A_COVERAGE", "B_SEPARABILITY", "C_SELECTION"]
FORBIDDEN_EXPERIMENTS = ["D_SOURCE_SIDE_IDENTIFICATION", "E_OPTIMIZATION_RESPONSE_ABILITY"]
TARGET_GAP_MIN = 0.20


def _run_git(args: list[str]) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return completed.stdout.strip()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_config() -> dict[str, Any]:
    with CONFIG_PATH.open(encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("task_id") != "TASK3-CMNIST-CPU-MINIMAL":
        raise ValueError("capability decomposition requires corrected CPU-minimal config")
    if config.get("stage_b", {}).get("seeds") != SEEDS:
        raise ValueError("capability decomposition is fixed to corrected Stage B seeds 10..14")
    return config


def _json_clean(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _json_clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_clean(item) for item in value]
    return value


def _csv_value(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN"
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_clean(payload), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    seen: list[str] = []
    for row in rows:
        for key in row:
            if key not in seen:
                seen.append(key)
    return seen


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = fieldnames or _fieldnames(rows)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field, "NA")) for field in names})


def _torch_version() -> str:
    return str(torch.__version__)


def _cpu_model() -> str:
    try:
        return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    except Exception:
        return platform.processor() or "unknown"


def write_preregistration(config: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "task_id": TASK_ID,
        "written_at_unix": time.time(),
        "git_head_before_run": _run_git(["rev-parse", "HEAD"]),
        "git_status_before_run": _run_git(["status", "--short"]),
        "config_path": str(CONFIG_PATH.relative_to(ROOT)),
        "config_sha256": _sha256(CONFIG_PATH),
        "checkpoint_manifest": str(MANIFEST_PATH.relative_to(ROOT)),
        "seeds": SEEDS,
        "methods": METHODS,
        "ridge": RIDGE,
        "color_subspace_ranks": RANKS,
        "executed_experiments": EXECUTED_EXPERIMENTS,
        "forbidden_experiments": FORBIDDEN_EXPERIMENTS,
        "verdicts": sorted(ALLOWED_VERDICTS),
        "dominant_bottlenecks": sorted(ALLOWED_DOMINANT_BOTTLENECKS),
        "exact_config": config,
    }
    text = f"""# TASK3-OOD-CAPABILITY-DECOMPOSITION-FIRST-ROUND Preregistered Design

task_id: `{TASK_ID}`
git_head_before_run: `{payload['git_head_before_run']}`
config_sha256: `{payload['config_sha256']}`
checkpoint_manifest: `{payload['checkpoint_manifest']}`
seeds: `10,11,12,13,14`
methods: `ERM`, `IRMv1`
ridge: `{RIDGE}`
color_subspace_ranks: `{','.join(str(rank) for rank in RANKS)}`

## Question

Run a first-round diagnostic decomposition of the corrected CPU-minimal ERM-vs-IRMv1 OOD gap into tested capabilities `C_coverage`, `C_separate`, and `C_select`.

## Inputs

- Corrected CPU-minimal config: `configs/task3_cmnist_cpu_minimal.json`.
- Corrected checkpoint manifest: `round3_redesign/task3_cmnist_counterfactual_audit/results/checkpoint_manifest.csv`.
- Corrected reconstructed checkpoints for ERM/IRMv1 seeds `10..14`.
- Corrected target counterfactual construction from `task3_cmnist_counterfactual_audit.probe`.

## A: Feature Coverage

Freeze each encoder. Fit a deterministic closed-form ridge probe on source frozen features with noisy source labels, and separately fit two-fold oracle clean ridge heads on color-balanced target counterfactual features using clean label `digit < 5`. The oracle rows are diagnostic and are not source-only evidence.

## B: Feature Separability

Estimate the color-response subspace from the uncentered second moment of `phi(x_G)-phi(x_R)` on the full target counterfactual set. For ranks `0,1,2,4,8,16,32,64`, project out top color directions and report shortcut suppression, task loss, clean oracle coverage, and red/green color-probe accuracy.

## C: Feature Selection

Discard the original head and train head-only `HEAD_ERM`, head-only `HEAD_IRMv1`, and diagnostic `ORACLE_CLEAN` heads from identical initialization for each frozen encoder. Source-trained heads use only CPU-minimal source features, source noisy labels, and the fixed batch schedule.

## Forbidden Scope

No new regularizer, no algorithm claim, no source-identifiability claim, no causal/additive decomposition claim, no theory validation, and no D/E execution.

## Decision Logic

Allowed verdicts are `FIRST-ROUND-CAPABILITY-ISOLATED`, `FIRST-ROUND-CAPABILITY-PARTIAL`, `FIRST-ROUND-CAPABILITY-INCONCLUSIVE`, and `FIRST-ROUND-AUDIT-INVALID`. A bottleneck can be isolated only if an oracle/intervention gap is at least `20pp` in at least `4/5` paired seeds and competing tested capabilities do not explain the same gap.
"""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "preregistered_design.md").write_text(text, encoding="utf-8")
    return payload


def _load_checkpoint_metrics(record: CheckpointRecord) -> dict[str, float]:
    checkpoint = torch.load(record.path, map_location="cpu")
    metrics = checkpoint.get("official_metrics")
    if not isinstance(metrics, dict):
        raise ValueError(f"checkpoint lacks official_metrics: {record.path}")
    return {key: float(value) for key, value in metrics.items()}


def verify_inputs(records: list[CheckpointRecord], *, config_sha256: str) -> dict[str, Any]:
    manifest_rows: list[dict[str, Any]] = []
    target_by_method: dict[str, list[float]] = {method: [] for method in METHODS}
    for record in records:
        actual_sha = sha256_file(record.path)
        if actual_sha != record.checkpoint_sha256:
            raise ValueError(f"checkpoint SHA mismatch: {record.path}")
        if record.config_sha256 != config_sha256:
            raise ValueError(f"config SHA mismatch: {record.seed}/{record.method}")
        metrics = _load_checkpoint_metrics(record)
        target_by_method[record.method].append(float(metrics["final_target_acc"]))
        manifest_rows.append({
            "seed": record.seed,
            "method": record.method,
            "checkpoint_path": str(record.path.relative_to(ROOT)),
            "checkpoint_sha256": record.checkpoint_sha256,
            "checkpoint_sha256_verified": actual_sha == record.checkpoint_sha256,
            "parameter_hash": record.parameter_hash,
            "config_sha256": record.config_sha256,
            "config_sha256_verified": record.config_sha256 == config_sha256,
            "git_commit": record.git_commit,
            **metrics,
        })
    erm_mean = sum(target_by_method["ERM"]) / len(target_by_method["ERM"])
    irm_mean = sum(target_by_method["IRMv1"]) / len(target_by_method["IRMv1"])
    gap = irm_mean - erm_mean
    if gap < TARGET_GAP_MIN:
        raise ValueError(f"corrected checkpoint target gap too small: {gap:.6f}")
    return {
        "manifest_rows": manifest_rows,
        "target_gap_verified": True,
        "erm_target_acc_mean": erm_mean,
        "irmv1_target_acc_mean": irm_mean,
        "irmv1_minus_erm_target_acc_mean": gap,
    }


def _append_rows(target: CapabilityRows, source: CapabilityRows) -> None:
    target.coverage.extend(source.coverage)
    target.separability.extend(source.separability)
    target.selection.extend(source.selection)


def run_analysis(config: dict[str, Any], records: list[CheckpointRecord], *, config_sha256: str) -> CapabilityRows:
    rows = CapabilityRows(coverage=[], separability=[], selection=[])
    for record in records:
        bundle = build_feature_bundle(record, config, config_sha256=config_sha256, data_root=ROOT / "data", download=False)
        _append_rows(rows, run_capability_experiments([bundle], config=config))
    return rows


def write_reports(summary: dict[str, Any], input_verification: dict[str, Any], paired_rows: list[dict[str, Any]], prereg: dict[str, Any]) -> None:
    protocol = f"""# TASK3-OOD-CAPABILITY-DECOMPOSITION-FIRST-ROUND Protocol

## Scope

This is an isolated first-round capability audit for corrected CPU-minimal ColoredMNIST. It evaluates only `C_coverage`, `C_separate`, and `C_select` on frozen ERM/IRMv1 encoders from seeds `10..14`.

## Guardrails

- No new regularizer.
- No algorithm claim.
- No source-identifiability claim.
- No causal/additive decomposition claim.
- No theory validation.
- No D/E execution.

## Input Gate

The checkpoint manifest, checkpoint SHA256 values, config SHA256, model architecture, parameter hashes, and corrected ERM/IRMv1 target gap are checked before capability metrics are computed. Verified mean target accuracy: ERM `{input_verification['erm_target_acc_mean']:.6f}`, IRMv1 `{input_verification['irmv1_target_acc_mean']:.6f}`, gap `{input_verification['irmv1_minus_erm_target_acc_mean']:.6f}`.

## Experiments

- A coverage: source ridge probe plus diagnostic oracle clean ridge coverage.
- B separability: color-response subspace removal at ranks `{RANKS}`.
- C selection: head-only ERM, head-only IRMv1, and diagnostic oracle clean heads with frozen encoders.
"""
    report = f"""# OOD Capability Decomposition Report

## Question

Can the corrected CPU-minimal ERM-vs-IRMv1 OOD gap be separated into feature coverage, feature separability, and head-selection bottlenecks?

## Result

- verdict: `{summary['verdict']}`
- dominant_bottleneck: `{summary['dominant_bottleneck']}`
- mean IRMv1-ERM target gap in loaded checkpoints: `{summary['target_gap_mean']:.6f}`
- coverage oracle gap on ERM encoders: `{summary['coverage_oracle_gap_erm_mean']:.6f}`
- best separability gain on ERM encoders: `{summary['separability_best_gain_erm_mean']:.6f}`
- selection oracle gap on ERM encoders: `{summary['selection_oracle_gap_erm_mean']:.6f}`

## Row Counts

- feature_coverage rows: `{summary['row_counts']['coverage']}`
- separability_curve rows: `{summary['row_counts']['separability']}`
- head_selection rows: `{summary['row_counts']['selection']}`
- paired_capability_summary rows: `{summary['row_counts']['paired']}`

## Isolation Votes

- coverage: `{summary['vote_counts']['coverage']}/5`
- separability: `{summary['vote_counts']['separability']}/5`
- selection: `{summary['vote_counts']['selection']}/5`

## Interpretation

This report is diagnostic only. It makes no new regularizer claim, no algorithm claim, no source-identifiability claim, no causal/additive decomposition claim, no theory-validation claim, and no D/E execution claim.

## Paired Seed Snapshot

| seed | IRMv1-ERM target | coverage vote | separability vote | selection vote |
|---:|---:|:---:|:---:|:---:|
"""
    for row in paired_rows:
        report += f"| {row['seed']} | {row['irmv1_minus_erm_target_acc']:.6f} | {row['coverage_isolation_vote']} | {row['separability_isolation_vote']} | {row['selection_isolation_vote']} |\n"
    limitations = """# OOD Capability Decomposition Limitations

- This first round tests only A/B/C; D source-side identification and E optimization/response ability are not executed.
- Oracle clean heads and target counterfactual projections are diagnostic interventions, not deployable source-only procedures.
- The decomposition is not causal or additive; overlapping explanations remain possible.
- The audit uses only corrected CPU-minimal ColoredMNIST checkpoints for ERM/IRMv1 seeds `10..14`.
- A `PARTIAL` or `INCONCLUSIVE` verdict should not be used as evidence for a new algorithm or theory validation.
"""
    provenance = {
        "task_id": TASK_ID,
        "written_at_unix": time.time(),
        "git_head": _run_git(["rev-parse", "HEAD"]),
        "git_status": _run_git(["status", "--short"]),
        "python": sys.version,
        "platform": platform.platform(),
        "cpu": _cpu_model(),
        "torch": _torch_version(),
        "config_path": str(CONFIG_PATH.relative_to(ROOT)),
        "config_sha256": prereg["config_sha256"],
        "checkpoint_manifest": str(MANIFEST_PATH.relative_to(ROOT)),
        "executed_experiments": EXECUTED_EXPERIMENTS,
        "forbidden_experiments": FORBIDDEN_EXPERIMENTS,
        "canonical_state_edited": False,
        "new_regularizer": False,
        "algorithm_claim": False,
        "source_identifiability_claim": False,
        "causal_additive_decomposition_claim": False,
        "theory_validation_claim": False,
        "d_or_e_execution": False,
    }
    (OUT_DIR / "protocol.md").write_text(protocol, encoding="utf-8")
    (OUT_DIR / "capability_decomposition_report.md").write_text(report, encoding="utf-8")
    (OUT_DIR / "limitations.md").write_text(limitations, encoding="utf-8")
    _write_json(OUT_DIR / "provenance.json", provenance)


def write_state_delta(summary: dict[str, Any], input_verification: dict[str, Any]) -> None:
    text = f"""# Proposed State Delta: TASK3-OOD-CAPABILITY-DECOMPOSITION-FIRST-ROUND

state_write_authorized: false
canonical_state_updated: false

## Proposed Result

- result_id: `TASK3-OOD-CAPABILITY-DECOMPOSITION-FIRST-ROUND`
- verdict: `{summary['verdict']}`
- dominant_bottleneck: `{summary['dominant_bottleneck']}`
- loaded checkpoint target gap verified: ERM `{input_verification['erm_target_acc_mean']:.6f}`, IRMv1 `{input_verification['irmv1_target_acc_mean']:.6f}`, gap `{input_verification['irmv1_minus_erm_target_acc_mean']:.6f}`
- outputs: `round3_redesign/ood_capability_decomposition/`

## Proposed Decision Impact

This audit should be treated as first-round diagnostic evidence only. It does not close Task 3, does not authorize a new algorithm claim, does not validate frozen theory, and does not execute source-side identification D or optimization/response E.

## Proposed Current-State Wording If Accepted Later

`A first-round corrected CPU-minimal OOD capability decomposition over ERM/IRMv1 seeds 10..14 has been run as diagnostic evidence only, with verdict {summary['verdict']} and dominant_bottleneck {summary['dominant_bottleneck']}. No canonical theory, algorithm, or Task 3 scientific verdict changes are authorized by this file alone.`
"""
    STATE_DELTA_PATH.write_text(text, encoding="utf-8")


def run() -> dict[str, Any]:
    config = _load_config()
    prereg = write_preregistration(config)
    config_sha256 = prereg["config_sha256"]
    records = load_checkpoint_manifest(MANIFEST_PATH, seeds=SEEDS, methods=METHODS, root=ROOT)
    input_verification = verify_inputs(records, config_sha256=config_sha256)
    _write_csv(RESULTS_DIR / "input_checkpoint_manifest.csv", input_verification["manifest_rows"])
    capability_rows = run_analysis(config, records, config_sha256=config_sha256)
    paired_rows = paired_capability_rows(capability_rows.coverage, capability_rows.separability, capability_rows.selection)
    summary = summarize_capabilities(
        capability_rows.coverage,
        capability_rows.separability,
        capability_rows.selection,
        valid=bool(input_verification["target_gap_verified"]),
    )
    _write_csv(RESULTS_DIR / "feature_coverage.csv", capability_rows.coverage)
    _write_csv(RESULTS_DIR / "separability_curve.csv", capability_rows.separability)
    _write_csv(RESULTS_DIR / "head_selection.csv", capability_rows.selection)
    _write_csv(RESULTS_DIR / "paired_capability_summary.csv", paired_rows)
    _write_json(RESULTS_DIR / "summary.json", {**summary, "input_verification": input_verification})
    write_reports(summary, input_verification, paired_rows, prereg)
    write_state_delta(summary, input_verification)
    return {**summary, "input_verification": input_verification}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    try:
        summary = run()
    except Exception as exc:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        payload = {
            "task_id": TASK_ID,
            "verdict": "FIRST-ROUND-AUDIT-INVALID",
            "dominant_bottleneck": "invalid",
            "error": f"{type(exc).__name__}: {exc}",
            "canonical_state_edited": False,
            "d_or_e_execution": False,
        }
        _write_json(RESULTS_DIR / "summary.json", payload)
        STATE_DELTA_PATH.write_text(
            f"# Proposed State Delta: {TASK_ID}\n\nstate_write_authorized: false\ncanonical_state_updated: false\n\nverdict: `FIRST-ROUND-AUDIT-INVALID`\nerror: `{type(exc).__name__}: {exc}`\n",
            encoding="utf-8",
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 1
    print(json.dumps(_json_clean(summary), indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
