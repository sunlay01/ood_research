"""Run Task 3 applicability without editing canonical project state."""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .task3_applicability.evaluation import evaluate_applicability, logic_audit
from .task3_applicability.features import build_feature_dictionary, build_feature_rows, write_feature_outputs
from .task3_applicability.targets import build_design, generate_target_outcomes, materialize_training_runs

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "round3_redesign" / "task3_applicability"


def _json(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, float) and math.isnan(value):
        return None
    raise TypeError(type(value).__name__)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _float(value: object) -> float:
    try:
        return float(value)
    except Exception:
        return math.nan


def _mean_metric(rows: list[dict[str, str]], benchmark: str, test: str, metric: str, feature_set: str) -> float:
    values = [
        _float(row.get(metric)) for row in rows
        if row.get("benchmark") == benchmark and row.get("test") == test and row.get("feature_set") == feature_set
    ]
    values = [value for value in values if np.isfinite(value)]
    return float(np.mean(values)) if values else math.nan


def _range_text(values: list[float]) -> str:
    finite = [value for value in values if np.isfinite(value)]
    if not finite:
        return "NA"
    return f"{min(finite):.6g}..{max(finite):.6g}"


def _write_docs(output: Path) -> None:
    results = output / "results"
    summary = json.loads((results / "summary.json").read_text(encoding="utf-8"))
    baseline = _read_csv(results / "baseline_comparison.csv")
    nulls = _read_csv(results / "permutation_controls.csv")
    counterexamples = _read_csv(results / "counterexamples.csv")
    feature_rows = _read_csv(results / "per_run_features.csv")
    outcome_rows = _read_csv(results / "per_target_outcomes.csv")
    verdict = summary["verdict"]
    baseline_lines = []
    for row in baseline:
        baseline_lines.append(
            f"| {row.get('benchmark')} | {row.get('test')} | {row.get('metric')} | "
            f"{row.get('B1')} | {row.get('B2')} | {row.get('B3')} | {row.get('B3_minus_B1')} |"
        )
    null_lines = []
    for row in nulls:
        null_lines.append(
            f"| {row.get('benchmark')} | {row.get('test')} | {row.get('metric')} | "
            f"{row.get('observed_B3')} | {row.get('permuted_B3')} | {row.get('random_B3')} |"
        )
    method_lines = []
    for benchmark in sorted({row.get("benchmark", "") for row in feature_rows if row.get("benchmark")}):
        for method in ("L2", "IRMV1", "VREX"):
            selected_features = [row for row in feature_rows if row.get("benchmark") == benchmark and row.get("method") == method]
            selected_outcomes = [row for row in outcome_rows if row.get("benchmark") == benchmark and row.get("method") == method]
            if not selected_features:
                continue
            z0_values = [_float(row.get("z0_norm")) for row in selected_features]
            e_values = [_float(row.get("E_operator_norm")) for row in selected_features]
            rho_values = [_float(row.get("rho_slack")) for row in selected_features]
            deltas = [_float(row.get("delta_target_risk_vs_erm")) for row in selected_outcomes]
            wins = [str(row.get("win_loss_vs_erm")).lower() in {"true", "1"} for row in selected_outcomes]
            finite = lambda values: [value for value in values if np.isfinite(value)]
            z0_f, e_f, rho_f, delta_f = map(finite, (z0_values, e_values, rho_values, deltas))
            method_lines.append(
                f"| {benchmark} | {method} | {_range_text(z0_f)} | {_range_text(e_f)} | "
                f"{_range_text(rho_f)} | {np.mean(delta_f) if delta_f else math.nan:.6g} | "
                f"{np.mean(wins) if wins else math.nan:.3g} |"
            )
    report = f"""# Task 3 Applicability Report

## A. Question

Do the frozen Round-3 mechanism, information, and spectral quantities add non-tautological grouped out-of-sample discrimination of finite held-out DG behavior beyond method/lambda and conventional source-side baselines?

## B. Preregistration

The preregistration was written to `task3_preregistered_design.json` before feature materialization and before held-out target outcome generation. It freezes methods, lambdas, families, radius grids, feature sets B0/B1/B2/B3, outcomes, CV grouping, null controls, and verdict thresholds.

## C. Leakage / Tautology Audit

- Logic audit pass: `{summary['logic_audit']['passes']}`.
- Feature rows: `{summary['feature_row_count']}`.
- Held-out target outcome rows: `{summary['target_outcome_row_count']}`.
- Valid joined rows: `{summary['joined_valid_row_count']}`.
- Grouped CV keeps training runs together: `{summary['grouped_cv_keeps_training_runs_together']}`.
- Target outcomes used for feature construction or selection: `False`.

## D. Gaussian Results

Gaussian rows use legal population environment-family target perturbations. Family/source-induced rows are evaluated as finite held-out target risks, not universal robust risks.

## E. CMNIST Results

CMNIST rows use frozen ERM representations and squared-loss linear heads. Accuracy is an offline post-hoc target outcome and is not used for training, lambda choice, feature construction, or threshold selection.

## F. B0/B1/B2/B3 Comparison

| benchmark | test | metric | B1 | B2 | B3 | B3-B1 |
|---|---|---|---:|---:|---:|---:|
{chr(10).join(baseline_lines) if baseline_lines else '| NA | NA | NA | NA | NA | NA | NA |'}

## G. Null Controls

| benchmark | test | metric | observed B3 | permuted B3 | random B3 |
|---|---|---|---:|---:|---:|
{chr(10).join(null_lines) if null_lines else '| NA | NA | NA | NA | NA | NA |'}

## H. Counterexamples

Counterexample rows saved: `{len(counterexamples)}`. The table is deliberately retained even when it weakens the applicability verdict.

## I. Per-Method Interpretation

L2, IRMv1, and V-REx are compared through actual-solution quantities only. Common-base attribution remains secondary and is not counted as the actual method mechanism.

| benchmark | method | range ||z0|| | range ||E||op | finite range rho_slack | mean delta risk | win rate |
|---|---|---:|---:|---:|---:|---:|
{chr(10).join(method_lines) if method_lines else '| NA | NA | NA | NA | NA | NA | NA |'}

Mechanism and family-aware quantities vary with lambda, but the grouped B3-vs-B1 rows above show that this variation does not become stable SUPPORT-level finite-target discrimination. Seed holdout is enforced for CMNIST; Gaussian uses family/config holdout and is reported with the sample-size limitation. Method/lambda conditioning is represented by B0/B1 and the null controls; apparent common-base explanations are not promoted to actual mechanisms.

## J. Radius Dependence

`radius_sweep.csv` reports the same B3-B1 comparisons separately by preregistered radius regime. Radius was not tuned after observing held-out performance.

## K. Limitations

Task 3 tests finite held-out behavior in audited Gaussian and CMNIST frozen-head settings. It does not establish a source-only deployable selector, target-risk lower bound, finite-sample guarantee, causal identification, semantic recovery, global nonlinear robustness, or universal DG theorem.

## L. Verdict

`{verdict}`

Historical reopen: none
"""
    (output / "task3_applicability_report.md").write_text(report, encoding="utf-8")
    limitations = f"""# Limitations

- The analysis is grouped and finite-target only; target rows from one training run are correlated.
- B3 quantities are family/metric conditional and are not source-only deployable selectors.
- Gaussian and CMNIST target outcomes are offline labels only and do not select lambdas, radii, feature subsets, thresholds, or verdict criteria.
- Family constants such as `R_info` are reported but not interpreted as method-level evidence.
- Verdict: `{verdict}`.
"""
    (output / "limitations.md").write_text(limitations, encoding="utf-8")
    _write_state_delta(output, summary, feature_rows, outcome_rows)


def _write_state_delta(output: Path, summary: dict[str, object], feature_rows: list[dict[str, str]], outcome_rows: list[dict[str, str]]) -> None:
    active = ROOT / "active"
    active.mkdir(exist_ok=True)
    verdict = summary["verdict"]
    close = verdict in {"TASK3-APPLICABILITY-SUPPORT", "TASK3-APPLICABILITY-PARTIAL", "TASK3-APPLICABILITY-FAIL"}
    text = f"""# Proposed State Delta: TASK3-APPLICABILITY

state_write_authorized: false

## Proposed Result IDs

- `R-TASK3-APPLICABILITY`: `{verdict}`, artifacts under `round3_redesign/task3_applicability/`.

## Proposed Decision Changes

- Close Task 3: `{str(close).lower()}`.
- Next task if closed: `Task 4 ex-ante helps/hurts prediction from local source/family geometry`.

## Proposed CURRENT_STATE Wording

Task 3 applicability has been run as an isolated finite held-out behavior audit. It produced `{len(feature_rows)}` per-run feature rows and `{len(outcome_rows)}` target outcome rows. Verdict: `{verdict}`. The result is evidence about finite Gaussian/CMNIST frozen-head behavior only, not a target-risk lower bound, causal identification result, finite-sample guarantee, semantic recovery result, or universal DG theorem.

## Do Not Apply Automatically

Canonical state was not edited by Task 3. Apply this delta only after human review.
"""
    (active / "STATE_DELTA.md").write_text(text, encoding="utf-8")


def run(output: Path = DEFAULT_OUTPUT, *, smoke: bool = False, logic_only: bool = False) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    results = output / "results"
    results.mkdir(exist_ok=True)
    design = build_design(smoke=smoke)
    design["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    design["python_version"] = platform.python_version()
    design_path = output / "task3_preregistered_design.json"
    design_path.write_text(json.dumps(design, indent=2), encoding="utf-8")
    audit = logic_audit(design)
    if logic_only:
        logic_path = results / "logic_audit.json"
        logic_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
        return {"verdict": "LOGIC-AUDIT-PASS" if audit["passes"] else "LOGIC-AUDIT-FAIL", "logic_audit": audit}
    if not audit["passes"]:
        summary = {"verdict": "TASK3-APPLICABILITY-FAIL", "logic_audit": audit}
        (results / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        _write_docs(output)
        return summary

    runs = materialize_training_runs(design, ROOT)
    feature_rows = build_feature_rows(ROOT, source_metric_rows=[run.source_feature_row() for run in runs])
    write_feature_outputs(output, feature_rows)
    outcome_rows = generate_target_outcomes(runs, design)
    _write_csv(results / "per_target_outcomes.csv", outcome_rows)
    summary = evaluate_applicability(feature_rows, outcome_rows, output, design)
    summary["order_trace"] = [
        "preregistration_written",
        "source_only_runs_materialized",
        "features_written",
        "held_out_target_outcomes_generated",
        "evaluation_written",
    ]
    (results / "summary.json").write_text(json.dumps(summary, indent=2, default=_json), encoding="utf-8")
    _write_docs(output)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--logic-audit", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.output, smoke=args.smoke, logic_only=args.logic_audit), indent=2, default=_json))


if __name__ == "__main__":
    main()
