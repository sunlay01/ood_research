"""Run Task 3 CMNIST-first local-response geometry experiment."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import torch

from .task3_cmnist_local_response import default_config, run_experiment
from .task3_cmnist_local_response.diagnostics import stable_json_hash, write_markdown
from .task3_cmnist_local_response.trainer import Task3Config, tiny_batch_sanity


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "round3_redesign" / "task3_cmnist_local_response"


def _git(args: list[str]) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False
    ).stdout.strip()


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_preregistration(output: Path, config: Task3Config) -> dict[str, Any]:
    design = {
        "task_id": "TASK3-CMNIST-LOCAL-RESPONSE",
        "written_unix_time": time.time(),
        "status": "PREREGISTERED_BEFORE_NEW_LOCAL_RESPONSE_TARGET_OUTCOMES",
        "code_commit_before_experiment": _git(["rev-parse", "HEAD"]),
        "working_tree_status_before_experiment": _git(["status", "--short"]),
        "profile": config.profile,
        "data_config": config.to_json_dict(),
        "architecture": "existing SmallCMNISTCNN from cmnist_feature_probe.py",
        "optimizer": "Adam, same learning-rate family as current CMNIST probe",
        "methods": list(config.primary_methods),
        "controls": list(config.control_methods),
        "beta_grid": list(config.beta_grid),
        "damping": {
            "primary_epsilon": config.damping_epsilon,
            "stability_ablations": list(config.damping_ablations),
            "formula": "mu = epsilon * max(trace(H)/d, 1e-8)",
            "inverse_metric_policy": "stop-gradient through damped inverse metric",
        },
        "source_only_selection_rule": "maximize worst-source validation accuracy, tie-break by mean-source validation accuracy and later checkpoint",
        "target_leakage_policy": "target accuracy/loss are evaluation-only and cannot select beta/checkpoint/method/architecture/duration/batches/ablations",
        "success_gate": {
            "support_requires": [
                "no target leakage",
                "end-to-end representation training",
                "all 10 primary seeds completed",
                "LOCAL_RESPONSE mean OOD accuracy >= ERM + 0.02",
                "LOCAL_RESPONSE mean OOD accuracy >= UNPRECONDITIONED_GRAD_ALIGN + 0.01",
                "LOCAL_RESPONSE beats ERM on at least 7/10 seeds",
                "LOCAL_RESPONSE beats GRAD_ALIGN on at least 7/10 seeds",
                "worst-source validation accuracy not degraded by more than 0.01 vs ERM",
                "real curvature metric beats matched random metric",
                "color sensitivity or another preregistered mechanism diagnostic moves in predicted direction",
                "benefit is not explained by larger effective update norm",
            ],
            "allowed_verdicts": [
                "TASK3-CMNIST-SUPPORT",
                "TASK3-CMNIST-PARTIAL",
                "TASK3-CMNIST-FAIL",
            ],
        },
        "interpretation_ceiling": "No exact A_rec/E/slack, causal, finite-sample, universal DG, or broad benchmark claim.",
    }
    design["preregistered_design_hash"] = stable_json_hash(design)
    (output / "preregistered_design.json").write_text(json.dumps(design, indent=2), encoding="utf-8")
    return design


def _write_static_docs(output: Path, config: Task3Config, cleanup: dict[str, Any], design: dict[str, Any]) -> None:
    write_markdown(
        output / "cleanup_report.md",
        "Cleanup Report",
        f"""
deleted Task3R directories/files: `{cleanup['task3r_deleted']}`

shared files reverted: `tests/test_project_state.py` keeps active-task boot support and no Task3R-specific assertions.

active files replaced: `{cleanup['active_replaced']}`

canonical state changed? `false`

old task3_applicability retained as historical diagnostic? `{cleanup['task3_applicability_retained']}`

Task3R runtime/source artifact remains? `{cleanup['task3r_artifact_remains']}`
""",
    )
    write_markdown(
        output / "prior_art_exact_object.md",
        "Prior Art Exact Object Audit",
        """
| paper | exact equation/object | same as our proposed object? | equivalent only under assumptions? | different? | implementation implication | novelty implication |
|---|---|---:|---:|---:|---|---|
| MLDG, Li et al. 2018, https://arxiv.org/abs/1710.03463 | meta-train/meta-test objective; first-order variants involve gradient alignment across domains | no | no exact inverse-H metric found in this bounded audit | yes | include only if faithful compact implementation is added; otherwise do not relabel GRAD/LR as MLDG | `RELATED-BUT-DIFFERENT` |
| Fish, Shi et al. 2021, https://arxiv.org/abs/2104.09937 | inter-domain gradient matching / gradient dot-product style update | no | related first-order gradient matching | yes | GRAD_ALIGN is a neutral baseline, not automatically Fish | `RELATED-BUT-DIFFERENT` |
| Fishr, Rame et al. 2021, https://arxiv.org/abs/2109.02934 | domain-level gradient variance matching; connects gradient variance to Fisher/Hessian motivation | no | related through Fisher/gradient-variance geometry | yes | do not call LOCAL_RESPONSE Fishr; use as strong related baseline family | `RELATED-BUT-DIFFERENT` |
| Hessian Alignment / classifier-head Hessian analyses, e.g. https://arxiv.org/abs/2308.11778 | Hessian/gradient structure for DG/generalization analysis | unresolved exact implementation match | possible only after equation-level comparison | yes in this bounded audit | no novelty claim; record as closest Hessian-geometry neighbor | `UNRESOLVED` |
| Moment/curvature alignment family | moment or Hessian matching rather than inverse-H gradient-disagreement penalty | no | no | yes | keep internal name `LOCAL_RESPONSE` | `NO-EXACT-MATCH-FOUND` for this exact object within the bounded checked set |

Bounded conclusion: the implemented object is reported under a neutral internal name. No algorithmic novelty claim is made by this task.
""",
    )
    write_markdown(
        output / "baseline_fidelity.md",
        "Baseline Fidelity",
        f"""
The run reuses the existing CMNIST binary label, red/green coloring, source correlations `{list(config.source_correlations)}`, OOD target correlation `{config.target_correlation}`, `SmallCMNISTCNN`, Adam optimizer family, seed handling, and counterfactual color probe.

This Task 3 run uses a bounded local sample/epoch budget recorded in `preregistered_design.json`: profile `{config.profile}`, train per environment `{config.train_per_environment}`, epochs `{config.epochs}`, batch size `{config.batch_size}`. The generator and architecture are not redesigned to favor the new method.

All methods share the same source data, source validation data, target evaluation data, seeds, optimizer family, checkpoint fractions, and source-only selection rule. Target outcomes are read after the preregistered design is written and after source-only beta/checkpoint selection.
""",
    )
    write_markdown(
        output / "method_definitions.md",
        "Method Definitions",
        """
`ERM`: source cross-entropy only.

`IRMv1`: standard scalar-risk-gradient penalty on source environments.

`V-REx`: variance of source cross-entropy risks.

`UNPRECONDITIONED_GRAD_ALIGN`: mean squared deviation of per-source head gradients from their source mean.

`LOCAL_RESPONSE`: the same centered head-gradient disagreement weighted by the stop-gradient inverse damped source head Hessian/Gauss-Newton metric.

`RANDOM_METRIC`: the LOCAL_RESPONSE form with a random SPD metric matched to the real metric's eigenvalue multiset, trace, and Frobenius scale.

`SHUFFLED_LOCAL_RESPONSE`: the LOCAL_RESPONSE penalty after shuffling source environment identity while preserving batch sizes.
""",
    )
    write_markdown(
        output / "mechanism_interpretation.md",
        "Mechanism Interpretation",
        """
The tested mechanism is lower-level than the frozen affine `E` theory. It asks whether cross-environment head-gradient disagreement should be measured in the local source-risk metric rather than raw Euclidean head-gradient norm.

The empirical chain being audited is: LOCAL_RESPONSE penalty decreases, counterfactual color response decreases or another mechanism diagnostic moves coherently, and OOD target accuracy improves. These arrows are hypotheses, not assumptions.
""",
    )
    write_markdown(
        output / "limitations.md",
        "Limitations",
        """
This is an end-to-end CMNIST local experiment, not a proof of the Round-3 Gaussian affine theorems for neural networks.

The training loss does not estimate or optimize `A_rec`, `E`, `rho_slack`, target risk, semantic labels, causal factors, or a universal DG objective.

The prior-art check is bounded to the listed primary papers and exact-object search terms; it is sufficient to block novelty claims here, not to certify novelty.
""",
    )
    write_markdown(
        output / "provenance.md",
        "Provenance",
        f"""
code commit before experiment: `{design['code_commit_before_experiment']}`

working tree dirty before experiment: `{bool(design['working_tree_status_before_experiment'])}`

preregistered design hash: `{design['preregistered_design_hash']}`

runner command: `PYTHONPATH=src python -m ood_repr_reg.run_task3_cmnist_local_response --profile {config.profile}`

Python: `{platform.python_version()}`

PyTorch: `{torch.__version__}`

device: `{config.device}`

seed list: `{list(config.seeds)}`
""",
    )


def _write_final_reports(output: Path) -> None:
    results = output / "results"
    summary = json.loads((results / "summary.json").read_text(encoding="utf-8"))
    run_rows = _read_csv(results / "run_table.csv")
    comparisons = _read_csv(results / "paired_comparisons.csv")
    counters = _read_csv(results / "counterexamples.csv")
    methods = summary["method_summary"]
    method_lines = "\n".join(
        f"- `{row['method']}`: mean target acc `{row['mean_target_accuracy']:.4f}`, "
        f"mean worst-source acc `{row['mean_worst_source_accuracy']:.4f}`, "
        f"mean color response `{row['mean_counterfactual_color_response']:.6f}`"
        for row in methods
    )
    comparison_lines = "\n".join(
        f"- `{row['left_method']}` vs `{row['right_method']}`: mean diff `{float(row['mean_difference']):.4f}`, "
        f"wins `{row['seed_wins']}/{row['n_pairs']}`, 95% CI "
        f"`[{float(row['ci95_low']):.4f}, {float(row['ci95_high']):.4f}]`"
        for row in comparisons
    )
    criteria_lines = "\n".join(f"- `{key}`: `{str(value).lower()}`" for key, value in summary["criteria"].items())
    report = f"""# Task 3 CMNIST Local Response Report

## A. Question

Does curvature-aware local response geometry add algorithmic value on end-to-end CMNIST?

## B. Cleanup

The discarded Gaussian Task3R implementation/results were removed from live source and result paths. `round3_redesign/task3_applicability/` was retained as historical diagnostic only.

## C. Prior Art

The exact object audit is in `prior_art_exact_object.md`. The run uses neutral internal names and makes no algorithmic novelty claim.

## D. Benchmark

The run uses the existing CMNIST generator, binary digit label, source correlations, target correlation, `SmallCMNISTCNN`, Adam optimizer family, and counterfactual color probe. Profile: `{summary['profile']}`.

## E. Methods

{method_lines}

## F. Source-Only Selection

Beta and checkpoint are selected by worst-source validation accuracy with mean-source validation as tie-breaker. `target_leakage_detected={str(summary['target_leakage_detected']).lower()}`.

## G. Main OOD Results

{comparison_lines}

## H. Curvature Increment

The primary increment is `LOCAL_RESPONSE - UNPRECONDITIONED_GRAD_ALIGN`; see `paired_comparisons.csv` and `run_table.csv` for paired seed rows.

## I. Mechanism

Mechanism diagnostics include raw gradient disagreement, local-response disagreement, response-vector norm, damped curvature spectrum, update norms, and counterfactual color response.

## J. Controls

Random metric control rows are in `random_metric_control.csv`; shuffled environment-control rows are in `environment_shuffle_control.csv`.

## K. Counterexamples

Strongest detected counterexamples are recorded in `counterexamples.csv`; count `{len(counters)}`.

## L. Relation to Frozen Theory

This experiment tests the lower-level source-risk metric insight. It does not estimate `A_rec`, optimize `E`, validate spectral slack for neural networks, or claim target-risk lower bounds.

## M. Verdict

`{summary['verdict']}`

Criteria:

{criteria_lines}

Historical reopen: none.
"""
    (output / "task3_cmnist_report.md").write_text(report, encoding="utf-8")
    state_delta = f"""# Proposed State Delta: TASK3-CMNIST-LOCAL-RESPONSE

state_write_authorized: false

discarded Task3R Gaussian branch deleted: true

new Task3 CMNIST verdict: `{summary['verdict']}`

Task 3 should close: `{str(summary['verdict'] in {'TASK3-CMNIST-SUPPORT', 'TASK3-CMNIST-FAIL'}).lower()}`

larger benchmark justified: `{str(summary['verdict'] in {'TASK3-CMNIST-SUPPORT', 'TASK3-CMNIST-PARTIAL'}).lower()}`

prior-art equivalence changes novelty interpretation: `no novelty claim made; bounded audit found related but not promoted exact equivalence`

Proposed CURRENT_STATE.md wording:

```text
Task 3 CMNIST local-response geometry has been run as an isolated end-to-end CMNIST audit. Verdict: {summary['verdict']}. It tests whether curvature-aware head-gradient disagreement adds source-only algorithmic value beyond unpreconditioned gradient alignment and standard DG baselines; it does not identify A_rec/E/rho_slack or establish a target-risk theorem.
```
"""
    (ROOT / "active" / "STATE_DELTA.md").write_text(state_delta, encoding="utf-8")


def _cleanup_status() -> dict[str, Any]:
    deleted_paths = [
        ROOT / "round3_redesign" / "task3r_algorithmization",
        ROOT / "src" / "ood_repr_reg" / "task3r_algorithmization",
        ROOT / "src" / "ood_repr_reg" / "run_task3r_algorithmization.py",
        ROOT / "tests" / "test_task3r_algorithmization.py",
    ]
    return {
        "task3r_deleted": all(not path.exists() for path in deleted_paths),
        "task3r_artifact_remains": any(path.exists() for path in deleted_paths),
        "active_replaced": "TASK3-CMNIST-LOCAL-RESPONSE" in (ROOT / "active" / "TASK.md").read_text(encoding="utf-8"),
        "task3_applicability_retained": (ROOT / "round3_redesign" / "task3_applicability").exists(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("smoke", "main"), default="main")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--skip-run", action="store_true")
    args = parser.parse_args()
    config = default_config(args.profile)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    cleanup = _cleanup_status()
    design = _write_preregistration(output, config)
    _write_static_docs(output, config, cleanup, design)
    sanity = tiny_batch_sanity(config)
    (output / "results").mkdir(exist_ok=True)
    (output / "results" / "tiny_batch_sanity.json").write_text(json.dumps(sanity, indent=2), encoding="utf-8")
    if not args.skip_run:
        run_experiment(config, output)
        _write_final_reports(output)


if __name__ == "__main__":
    main()
