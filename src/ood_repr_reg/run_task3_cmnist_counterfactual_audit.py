"""Runner for TASK3-CMNIST-COUNTERFACTUAL-DIAGNOSTIC-PORT."""

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

from .task3_cmnist_counterfactual_audit.analysis import (
    ALLOWED_VERDICTS,
    PAIRED_METRICS,
    build_paired_effects,
    summarize_audit,
)
from .task3_cmnist_counterfactual_audit.diagnostics import model_counterfactual_diagnostics
from .task3_cmnist_counterfactual_audit.probe import build_counterfactual_probe, probe_invariant_checks
from .task3_cmnist_cpu_minimal.data import build_task3_data
from .task3_cmnist_cpu_minimal.evaluation import evaluate_checkpoint
from .task3_cmnist_cpu_minimal.model import build_model_from_config, linear_layer_count, parameter_hash
from .task3_cmnist_cpu_minimal.trainer import train_one_method


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs/task3_cmnist_cpu_minimal.json"
CPU_MINIMAL_MAIN_RUNS = ROOT / "round3_redesign/task3_cmnist_cpu_minimal/results/main_runs.csv"
OUT_DIR = ROOT / "round3_redesign/task3_cmnist_counterfactual_audit"
RESULTS_DIR = OUT_DIR / "results"
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"
STATE_DELTA_PATH = ROOT / "active/STATE_DELTA.md"
TASK_ID = "TASK3-CMNIST-COUNTERFACTUAL-DIAGNOSTIC-PORT"
SEEDS = [10, 11, 12, 13, 14]
METHODS = ["ERM", "IRMv1"]
RECONCILIATION_TOLERANCE = 1e-6
RELATIVE_WHITENING_TOLERANCE = 1e-5


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


def _json_clean(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _json_clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_clean(item) for item in value]
    if isinstance(value, tuple):
        return [_json_clean(item) for item in value]
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_clean(payload), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _csv_value(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN"
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field, "NA")) for field in fieldnames})


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_config() -> dict[str, Any]:
    with CONFIG_PATH.open(encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("task_id") != "TASK3-CMNIST-CPU-MINIMAL":
        raise ValueError("counterfactual audit requires the corrected CPU-minimal config")
    if config.get("stage_b", {}).get("seeds") != SEEDS:
        raise ValueError("counterfactual audit is fixed to CPU-minimal Stage B seeds 10..14")
    return config


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
        "seeds": SEEDS,
        "methods": METHODS,
        "whitening_relative_tolerance": RELATIVE_WHITENING_TOLERANCE,
        "reconciliation_tolerance": RECONCILIATION_TOLERANCE,
        "checkpoint_policy": "reconstruct_if_exact_corrected_cpu_minimal_checkpoints_absent",
        "outcome_use": "held_out_target_split_post_hoc_only",
    }
    text = f"""# TASK3-CMNIST-COUNTERFACTUAL-DIAGNOSTIC-PORT Preregistered Design

task_id: `{TASK_ID}`
git_head_before_run: `{payload['git_head_before_run']}`
config_sha256: `{payload['config_sha256']}`
methods: `ERM`, `IRMv1`
seeds: `10,11,12,13,14`

## Question

Decompose the corrected CPU-minimal ERM/IRMv1 target-accuracy gap into representation content versus final-head color usage.

## Fixed Inputs

- Config: `configs/task3_cmnist_cpu_minimal.json`
- Reference corrected run table: `round3_redesign/task3_cmnist_cpu_minimal/results/main_runs.csv`
- Data/model/trainer: `src/ood_repr_reg/task3_cmnist_cpu_minimal/`

## Reconstruction Gate

If exact final checkpoints are absent, reconstruct only ERM/IRMv1 seeds `10..14` with the corrected CPU-minimal trainer, then require final source env0 accuracy, source env1 accuracy, target accuracy, and prediction-color agreement to match the existing corrected run within `1e-6`.

## Counterfactual Probe

Use the full held-out target split post-hoc. Recover grayscale by summing the two 14x14 color channels and construct red `(g,0)` and green `(0,g)` inputs. Original target color is not used in the intervention.

## Diagnostics

- Representation content: whitened latent color response, task signal, task/color overlap, balanced clean accuracy.
- Head usage: scalar-logit color response, sigmoid probability color response, task-head margin, counterfactual consistency, counterfactual flip rate.

## Descriptive Category Rule

Before observing this audit's diagnostics, fixed descriptive gates are: representation is material if IRMv1 has >=1.25x task signal, >=5pp balanced clean accuracy gain, or <=0.80x latent color response in at least 4/5 paired seeds. Head usage is material if IRMv1 has <=0.80x scalar/probability color response, >=5pp higher counterfactual consistency, or >=5pp lower flip rate in at least 4/5 paired seeds. Both material means `MIXED-DECOMPOSITION`; neither means `DESCRIPTIVE-INCONCLUSIVE`.

## Interpretation Ceiling

This audit is descriptive only. It does not establish causality, source identifiability, a new objective, frozen theory validation, novelty, or restoration of old CMNIST empirical results.
"""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "preregistered_design.md").write_text(text, encoding="utf-8")
    return payload


def _reference_rows() -> dict[tuple[int, str], dict[str, str]]:
    rows = _read_csv(CPU_MINIMAL_MAIN_RUNS)
    selected = {
        (int(row["seed"]), row["method"]): row
        for row in rows
        if int(row["seed"]) in SEEDS and row["method"] in METHODS
    }
    missing = [(seed, method) for seed in SEEDS for method in METHODS if (seed, method) not in selected]
    if missing:
        raise ValueError(f"missing corrected CPU-minimal reference rows: {missing}")
    return selected


def _fresh_model(config: dict[str, Any], seed: int):
    torch.manual_seed(int(seed))
    model = build_model_from_config(config)
    if linear_layer_count(model) != 3:
        raise ValueError("model identity check failed: expected exactly three linear layers")
    return model, parameter_hash(model)


def _official_metric_map(metrics: dict[str, float]) -> dict[str, float]:
    return {
        "final_source_env0_acc": float(metrics["source_env0_acc"]),
        "final_source_env1_acc": float(metrics["source_env1_acc"]),
        "final_source_mean_acc": float(metrics["source_mean_acc"]),
        "final_target_acc": float(metrics["target_acc"]),
        "final_prediction_color_agreement": float(metrics["prediction_color_agreement"]),
    }


def _reconciliation(
    observed: dict[str, float],
    reference: dict[str, str],
) -> tuple[bool, float, dict[str, float]]:
    deltas = {
        key: abs(float(reference[key]) - float(observed[key]))
        for key in (
            "final_source_env0_acc",
            "final_source_env1_acc",
            "final_source_mean_acc",
            "final_target_acc",
            "final_prediction_color_agreement",
        )
    }
    max_delta = max(deltas.values())
    return max_delta <= RECONCILIATION_TOLERANCE, max_delta, deltas


def _save_checkpoint(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return _sha256(path)


def _reconstruct_and_diagnose(config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    reference = _reference_rows()
    diagnostic_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    errors: list[str] = []
    config_sha = _sha256(CONFIG_PATH)
    git_head = _run_git(["rev-parse", "HEAD"])

    for seed in SEEDS:
        data = build_task3_data(config, int(seed))
        probe = build_counterfactual_probe(data.target_env)
        probe_checks = probe_invariant_checks(probe)
        if not all(probe_checks.values()):
            errors.append(f"probe invariant failed for seed {seed}: {probe_checks}")
            continue
        for method in METHODS:
            started = time.time()
            model, initial_hash = _fresh_model(config, int(seed))
            result = train_one_method(
                model=model,
                source_envs=data.source_envs,
                batch_schedule=data.batch_schedule,
                method=method,
                config=config,
                seed=int(seed),
                initial_parameter_hash=initial_hash,
            )
            if not result.finite:
                errors.append(f"training reconstruction failed for seed={seed} method={method}: {result.invalid_reason}")
                continue
            official_metrics = _official_metric_map(evaluate_checkpoint(result.model, data.source_envs, data.target_env, device=config["device"]))
            rec_ok, rec_max, rec_deltas = _reconciliation(official_metrics, reference[(int(seed), method)])
            if not rec_ok:
                errors.append(f"reconciliation failed for seed={seed} method={method}: {rec_deltas}")
            before_hash = parameter_hash(result.model)
            diagnostics = model_counterfactual_diagnostics(
                result.model,
                probe,
                device=config["device"],
                relative_tolerance=RELATIVE_WHITENING_TOLERANCE,
            )
            after_hash = parameter_hash(result.model)
            if before_hash != after_hash:
                errors.append(f"diagnostics mutated parameters for seed={seed} method={method}")
            checkpoint_path = CHECKPOINT_DIR / f"seed_{int(seed)}_{method}.pt"
            checkpoint_sha = _save_checkpoint(
                checkpoint_path,
                {
                    "task_id": TASK_ID,
                    "checkpoint_source": "reconstructed",
                    "seed": int(seed),
                    "method": method,
                    "config_sha256": config_sha,
                    "git_commit": git_head,
                    "state_dict": {key: value.detach().cpu().clone() for key, value in result.model.state_dict().items()},
                    "parameter_hash": before_hash,
                    "official_metrics": official_metrics,
                },
            )
            manifest_rows.append(
                {
                    "seed": int(seed),
                    "method": method,
                    "checkpoint_source": "reconstructed",
                    "path": str(checkpoint_path.relative_to(ROOT)),
                    "checkpoint_sha256": checkpoint_sha,
                    "parameter_hash": before_hash,
                    "config_sha256": config_sha,
                    "git_commit": git_head,
                    "initial_parameter_hash": initial_hash,
                    "batch_schedule_hash": result.batch_schedule_hash,
                    "reconciliation_status": "PASS" if rec_ok else "FAIL",
                    "reconciliation_max_abs_delta": rec_max,
                    "wall_seconds": time.time() - started,
                }
            )
            diagnostic_rows.append(
                {
                    "seed": int(seed),
                    "method": method,
                    "checkpoint_sha256": checkpoint_sha,
                    "parameter_hash": before_hash,
                    **official_metrics,
                    **diagnostics,
                    "reconciliation_status": "PASS" if rec_ok else "FAIL",
                    "reconciliation_max_abs_delta": rec_max,
                }
            )
    return diagnostic_rows, manifest_rows, errors


def _paired_fieldnames() -> list[str]:
    fields = ["seed"]
    for metric in PAIRED_METRICS:
        fields.extend([f"erm_{metric}", f"irmv1_{metric}", f"delta_{metric}", f"ratio_{metric}"])
    return fields


def _diagnostic_fieldnames() -> list[str]:
    return [
        "seed",
        "method",
        "checkpoint_sha256",
        "parameter_hash",
        "final_source_env0_acc",
        "final_source_env1_acc",
        "final_source_mean_acc",
        "final_target_acc",
        "final_prediction_color_agreement",
        "latent_color_response",
        "task_signal",
        "task_color_overlap",
        "prediction_color_response",
        "probability_color_response",
        "task_head_margin",
        "balanced_clean_accuracy",
        "counterfactual_prediction_consistency",
        "counterfactual_prediction_flip_rate",
        "n_probe_examples",
        "finite",
        "task_color_overlap_degenerate",
        "whitener_retained_rank",
        "whitener_threshold",
        "reconciliation_status",
        "reconciliation_max_abs_delta",
    ]


def _manifest_fieldnames() -> list[str]:
    return [
        "seed",
        "method",
        "checkpoint_source",
        "path",
        "checkpoint_sha256",
        "parameter_hash",
        "config_sha256",
        "git_commit",
        "initial_parameter_hash",
        "batch_schedule_hash",
        "reconciliation_status",
        "reconciliation_max_abs_delta",
        "wall_seconds",
    ]


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        if math.isfinite(value):
            return f"{value:.6g}"
        return "NaN"
    return str(value)


def write_report(summary: dict[str, Any], provenance: dict[str, Any], errors: list[str]) -> None:
    verdict = str(summary.get("verdict", "AUDIT-INVALID"))
    if verdict not in ALLOWED_VERDICTS:
        verdict = "AUDIT-INVALID"
    acc = summary.get("official_accuracy_gap", {})
    rep = summary.get("representation_content", {})
    head = summary.get("head_usage", {})
    next_experiment = {
        "HEAD-USAGE-DOMINANT": "A later frozen-encoder / re-trained-head intervention would best separate head use from representation content.",
        "REPRESENTATION-DOMINANT": "A later representation intervention such as color-subspace removal would best test whether representation content drives the gap.",
        "MIXED-DECOMPOSITION": "A later causal intervention must independently alter encoder content and head use to separate the explanations.",
        "DESCRIPTIVE-INCONCLUSIVE": "A later causal intervention is needed before choosing between representation and head-use explanations.",
        "AUDIT-INVALID": "No next experiment is justified until the validity gate is repaired.",
    }[verdict]
    lines = [
        "# TASK3-CMNIST-COUNTERFACTUAL-DIAGNOSTIC-PORT Report",
        "",
        "## 1. Validity and provenance",
        f"Git HEAD: `{provenance.get('git_head')}`",
        f"Config SHA256: `{provenance.get('config_sha256')}`",
        f"Checkpoint source: `{summary.get('checkpoint_source')}`",
        f"Reconciliation status: `{'PASS' if summary.get('valid') else 'FAIL'}`",
        f"Diagnostic rows: `{summary.get('row_count')}`",
        f"Test results: `{provenance.get('test_results', 'external validation commands run after this report is generated')}`",
        f"Errors: `{errors if errors else 'none'}`",
        "",
        "## 2. Existing corrected performance gap",
        f"ERM mean target accuracy: `{_fmt(acc.get('erm_mean'))}`",
        f"IRMv1 mean target accuracy: `{_fmt(acc.get('irmv1_mean'))}`",
        f"IRMv1 minus ERM: `{_fmt(acc.get('delta_mean'))}`",
        f"Paired target-accuracy deltas: `{[_fmt(v) for v in acc.get('paired_deltas', [])]}`",
        "",
        "## 3. Representation content",
    ]
    for metric in ("latent_color_response", "task_signal", "task_color_overlap", "balanced_clean_accuracy"):
        item = rep.get(metric, {})
        lines.append(f"{metric}: ERM mean `{_fmt(item.get('erm_mean'))}`, IRMv1 mean `{_fmt(item.get('irmv1_mean'))}`, delta mean `{_fmt(item.get('delta_mean'))}`, ratio mean `{_fmt(item.get('ratio_mean'))}`")
    lines.extend([
        "",
        "## 4. Head usage",
    ])
    for metric in ("prediction_color_response", "probability_color_response", "task_head_margin", "counterfactual_prediction_consistency", "counterfactual_prediction_flip_rate"):
        item = head.get(metric, {})
        lines.append(f"{metric}: ERM mean `{_fmt(item.get('erm_mean'))}`, IRMv1 mean `{_fmt(item.get('irmv1_mean'))}`, delta mean `{_fmt(item.get('delta_mean'))}`, ratio mean `{_fmt(item.get('ratio_mean'))}`")
    lines.extend([
        "",
        "## 5. Paired ERM-IRM decomposition",
        "Is the large target-accuracy gap accompanied mainly by representation differences, head-use differences, or both? The per-seed deltas and ratios are saved in `results/paired_effects.csv`; this report uses only those saved values.",
        "",
        "## 6. Verdict",
        verdict,
        "",
        "## 7. What this does not establish",
        "This does not establish causality, source identifiability, a new objective, frozen theory validation, novelty, target-risk lower bounds, finite-sample guarantees, or restoration of old CMNIST empirical results.",
        "",
        "## 8. Next experiment only if justified",
        next_experiment,
        "",
        "Historical reopen: none for empirical evidence; old diagnostic code was formula reference only.",
    ])
    (OUT_DIR / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_state_delta(summary: dict[str, Any]) -> None:
    verdict = str(summary.get("verdict", "AUDIT-INVALID"))
    text = f"""# Proposed State Delta

task_id: TASK3-CMNIST-COUNTERFACTUAL-DIAGNOSTIC-PORT
verdict: {verdict}
state_write_authorized: false

Proposed updates:

1. Old CMNIST empirical feature-probe results remain invalid as current evidence.
2. This task ported only the counterfactual diagnostic construction onto the corrected CPU-minimal ColoredMNIST ERM/IRMv1 runs.
3. Corrected ERM/IRMv1 counterfactual evidence is new descriptive evidence under `round3_redesign/task3_cmnist_counterfactual_audit/`.
4. No frozen theorem status changes are proposed.
5. No old Task1/Task2 CMNIST evidence is restored by this task.

Interpretation ceiling: descriptive diagnostic only; no causality, source-identifiability, new objective, theory validation, or novelty claim.

No canonical state file was edited.
"""
    STATE_DELTA_PATH.write_text(text, encoding="utf-8")


def write_provenance(config: dict[str, Any], prereg: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "task_id": TASK_ID,
        "git_head": _run_git(["rev-parse", "HEAD"]),
        "git_status": _run_git(["status", "--short"]),
        "config_sha256": prereg["config_sha256"],
        "python_version": sys.version,
        "pytorch_version": torch.__version__,
        "device": config["device"],
        "cpu_model": _cpu_model(),
        "checkpoint_source": summary.get("checkpoint_source"),
        "seeds": SEEDS,
        "methods": METHODS,
        "old_empirical_results_used": False,
        "old_code_runtime_dependency": False,
        "target_used_for_training_or_selection": False,
        "original_target_color_used_for_intervention": False,
        "test_results": "pending_external_validation",
    }
    _write_json(OUT_DIR / "provenance.json", payload)
    return payload


def run() -> dict[str, Any]:
    config = _load_config()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    prereg = write_preregistration(config)
    diagnostic_rows, manifest_rows, errors = _reconstruct_and_diagnose(config)
    valid = (
        not errors
        and len(diagnostic_rows) == 10
        and all(row.get("method") in METHODS and int(row.get("seed")) in SEEDS for row in diagnostic_rows)
        and all(bool(row.get("finite")) for row in diagnostic_rows)
        and all(row.get("reconciliation_status") == "PASS" for row in diagnostic_rows)
    )
    paired_rows = build_paired_effects(diagnostic_rows, seeds=SEEDS) if len(diagnostic_rows) == 10 else []
    summary = summarize_audit(
        diagnostic_rows=diagnostic_rows,
        paired_rows=paired_rows,
        valid=valid,
        checkpoint_source="reconstructed",
        seeds=SEEDS,
    ) if paired_rows else {
        "task_id": TASK_ID,
        "valid": False,
        "checkpoint_source": "reconstructed",
        "seeds": SEEDS,
        "methods": METHODS,
        "row_count": len(diagnostic_rows),
        "verdict": "AUDIT-INVALID",
        "interpretation_ceiling": "descriptive diagnostic only",
        "errors": errors,
    }
    if errors:
        summary["errors"] = errors
    _write_csv(RESULTS_DIR / "checkpoint_manifest.csv", manifest_rows, _manifest_fieldnames())
    _write_csv(RESULTS_DIR / "diagnostics.csv", diagnostic_rows, _diagnostic_fieldnames())
    _write_csv(RESULTS_DIR / "paired_effects.csv", paired_rows, _paired_fieldnames())
    _write_json(RESULTS_DIR / "summary.json", summary)
    provenance = write_provenance(config, prereg, summary)
    write_report(summary, provenance, errors)
    write_state_delta(summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.parse_args(argv)
    summary = run()
    print(summary["verdict"])
    return 0 if summary.get("valid") else 1


if __name__ == "__main__":
    raise SystemExit(main())
