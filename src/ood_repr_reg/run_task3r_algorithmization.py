"""Run the isolated Task 3R source-only exact Gaussian probe."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
from pathlib import Path

import numpy as np

from .task3r_algorithmization.experiment import (
    evaluate_held_out,
    gate_summary,
    prepare_source_fits,
    preregistration,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "round3_redesign" / "task3r_algorithmization"


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def _method_means(rows: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for method in sorted({str(row["method"]) for row in rows}):
        values = [row for row in rows if row["method"] == method]
        result[method] = {
            "target_risk": float(np.mean([float(row["finite_target_risk"]) for row in values])),
            "estimated_E": float(np.mean([float(row["estimated_E_norm"]) for row in values])),
            "oracle_E": float(np.mean([float(row["oracle_E_norm"]) for row in values])),
            "source_risk": float(np.mean([float(row["source_risk"]) for row in values])),
        }
    return result


def _write_report(output: Path, summary: dict[str, object], rows: list[dict[str, object]]) -> None:
    means = _method_means(rows)
    configs = summary["configuration_results"]
    config_lines = "\n".join(
        f"- `{item['config_id']}`: response reduction `{item['response_relative_reduction']:.4g}`, "
        f"held-out risk improvement `{item['held_out_risk_improvement']:.4g}`, "
        f"H1/H2/H3 = `{item['H1_response']}/{item['H2_held_out']}/{item['H3_direction']}`."
        for item in configs
    )
    method_lines = "\n".join(
        f"- `{method}`: target risk `{value['target_risk']:.8g}`, estimated residual "
        f"`{value['estimated_E']:.8g}`, oracle residual `{value['oracle_E']:.8g}`."
        for method, value in means.items()
    )
    report = f"""# Task 3R source-only algorithmization

Verdict: `{summary['verdict']}`

## A. Prior-work design bridge

The reusable pattern is to create virtual train/test domain splits from source domains, derive a local object, and then optimize a computable surrogate. MLDG supplies the pseudo-domain protocol; Fish/Fishr, Hessian Alignment, and CMA show how gradient/Hessian objects can be approximated or reduced; Transferability keeps the distinction between a local quantity and held-out behavior. The selected quadratic reduction is labeled `{summary['prior_work_equivalence_label']}` and carries no novelty claim.

## B. Controlled theorem object

The frozen chain is `A_rec, O_S, Pi -> E -> spectral slack / affine regret`. This probe replaces unavailable target `A_rec` during training with a finite source-domain LOO response target. `Pi` remains `-(H_R+lambda K)^-1(B_R+lambda C)` and is never optimized as a free matrix.

## C. Source estimability

Source moments, head optima, Hessians, task-state contrasts, pseudo-target risk, and finite LOO response are source-only. Oracle `A_rec`, true held-out risk, and slack diagnostics are post-hoc only. The machine rows record both target-use flags as false.

## D. Selected objective

The prototype learns a low-rank forcing block `C=UV^T` in a centered quadratic regularizer. It minimizes source pseudo-target risk plus a directional response mismatch over LOO source folds. Centering makes `z0=0` at every meta-source reference. `K=I`; exact FOC differentiation yields the actual response.

## E. Relation to prior methods

The finite source split is MLDG-like, but the optimized object is a constrained exact head response rather than a one-step model gradient. Under quadratic assumptions it overlaps derivative/moment alignment and is therefore a special case, not established as a distinct general algorithm. It does not inherit target-risk bounds from Transferability or Hessian Alignment.

## F. Exact Gaussian result

All optimizer rows were stable: `{summary['optimizer_stable']}`. Exact IFT versus retraining passed: `{summary['ift_ok']}`, maximum relative error `{summary['max_ift_fd_relative_error']:.3e}`. However, the preregistered Gaussian gate passed: `{summary['gaussian_gate_pass']}`.

{config_lines}

Aggregate method diagnostics:

{method_lines}

## G. Mechanism intervention

The matched-norm random-response control was trained with the same source-only protocol. It did not validate the proposed causal mechanism: H3 may pass locally because the random control is harmful, but H1 and H2 did not pass in the required independent configurations. Reduced source exposure also failed the preregistered H4 gate: `{summary['H4_source_information']}`.

## H. Approximation fidelity

`{summary['approximation_fidelity']}`. This Phase-I implementation uses exact population moments and exact quadratic derivatives; no cheap approximation was introduced.

## I. Failure decomposition

- Theory/objective: the frozen theorem remains an affine geometry result; this source pseudo-target surrogate is not implied to improve arbitrary finite target risk.
- A-estimation: the finite LOO response did not reliably reduce the post-hoc residual.
- Learner realizability: the low-rank `C=UV^T` class was optimized stably but did not reach the desired improvement.
- Optimization: no numerical failure was observed.
- Approximation: not applicable in the exact probe.

The bottleneck is: {summary['bottleneck']} CMNIST was not run because its preregistered gate did not pass.

## J. Scope

This is deterministic population Gaussian evidence. It is not a target-risk theorem, causal result, finite-sample guarantee, semantic recovery claim, or universal DG result.

Historical reopen: none.
"""
    (output / "task3r_report.md").write_text(report, encoding="utf-8")


def _write_state_delta(summary: dict[str, object]) -> None:
    text = f"""# Proposed State Delta: TASK3R-ALGORITHMIZATION

state_write_authorized: false

## Proposed Result ID

- `R-TASK3R-ALGORITHM`: `{summary['verdict']}`; exact Gaussian source-only probe under `round3_redesign/task3r_algorithmization/`.

## Proposed Decisions

- Task 3R closes as an exact controlled probe: `true`.
- Previous Task 3 applicability remains diagnostic only: `true`.
- Gaussian H1-H4 gate passed: `{str(summary['gaussian_gate_pass']).lower()}`.
- Broader CMNIST or large-benchmark algorithm testing justified now: `{str(summary['gaussian_gate_pass']).lower()}`.

## Proposed CURRENT_STATE wording

`Task 3R tested one exact source-only leave-one-source-domain-out response-matching prototype. The implementation and IFT audit were valid, but the preregistered Gaussian gate passed={str(summary['gaussian_gate_pass']).lower()}; verdict {summary['verdict']}. No CMNIST or broader benchmark was run.`

## Do Not Apply Automatically

Canonical state and registries were not edited.

Historical reopen: none.
"""
    (ROOT / "active" / "STATE_DELTA.md").write_text(text, encoding="utf-8")


def run(output: Path = DEFAULT_OUTPUT) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    results = output / "results"
    results.mkdir(exist_ok=True)
    prior_work = output / "prior_work_bridge.md"
    if not prior_work.exists() or "novelty claim" not in prior_work.read_text(encoding="utf-8").lower():
        raise RuntimeError("prior-work bridge must be completed before the probe")

    design = preregistration()
    design_path = output / "preregistered_algorithm_probe.json"
    design_path.write_text(json.dumps(design, indent=2), encoding="utf-8")

    fits, ablations = prepare_source_fits()
    # This is the only transition at which held-out target environments are
    # generated.  The fitted objects above contain source states only.
    rows = evaluate_held_out(fits)
    summary = gate_summary(rows, ablations, design)
    summary.update({
        "git_commit": _git_commit(),
        "python_version": platform.python_version(),
        "runner": "ood_repr_reg.run_task3r_algorithmization",
        "preregistration_path": str(design_path.relative_to(ROOT)),
        "preregistration_written_before_outcomes": True,
    })
    _write_csv(results / "exact_probe_rows.csv", rows)
    _write_csv(results / "ablation_rows.csv", ablations)
    (results / "summary.json").write_text(
        json.dumps(summary, indent=2, default=_json_default), encoding="utf-8"
    )
    _write_report(output, summary, rows)
    _write_state_delta(summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(args.output), indent=2, default=_json_default))


if __name__ == "__main__":
    main()
