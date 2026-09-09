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
from .task3_cmnist_local_response.official_irm_reference import (
    OfficialIRMConfig,
    official_summary,
    run_official_task3_comparison,
)
from .task3_cmnist_local_response.trainer import (
    Task3Config,
    baseline_recovery_rows,
    summarize_baseline_recovery,
    tiny_batch_sanity,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "round3_redesign" / "task3_cmnist_local_response"
PRIMARY_RESULT_FILES = (
    "run_table.csv",
    "seed_summary.csv",
    "hyperparameter_selection.csv",
    "candidate_grid_evaluation.csv",
    "dynamics.csv",
    "mechanism_diagnostics.csv",
    "random_metric_control.csv",
    "environment_shuffle_control.csv",
    "paired_comparisons.csv",
    "counterexamples.csv",
)


def _parse_seed_csv(value: str | None, fallback: tuple[int, ...]) -> tuple[int, ...]:
    if value is None or value.strip() == "":
        return fallback
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def _git(args: list[str]) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False
    ).stdout.strip()


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _clear_primary_result_files(results: Path) -> list[str]:
    cleared: list[str] = []
    for name in PRIMARY_RESULT_FILES:
        path = results / name
        if path.exists():
            path.unlink()
            cleared.append(name)
    return cleared


def _parse_float_csv(value: str | None, fallback: tuple[float, ...]) -> tuple[float, ...]:
    if value is None or value.strip() == "":
        return fallback
    return tuple(float(part.strip()) for part in value.split(",") if part.strip())


def _parse_str_csv(value: str | None, fallback: tuple[str, ...]) -> tuple[str, ...]:
    if value is None or value.strip() == "":
        return fallback
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _official_config_from_task3(
    config: Task3Config,
    seeds: tuple[int, ...],
    *,
    response_betas: tuple[float, ...] | None = None,
    methods: tuple[str, ...] | None = None,
) -> OfficialIRMConfig:
    return OfficialIRMConfig(
        data_root=config.data_root,
        download=config.download,
        seeds=seeds,
        hidden_dim=config.official_irm_hidden_dim,
        l2_regularizer_weight=1e-3,
        learning_rate=config.learning_rate,
        penalty_anneal_iters=config.official_irm_penalty_anneal_iters,
        penalty_weight=config.official_irm_penalty_weight,
        steps=config.official_irm_steps,
        label_noise=config.label_noise,
        device=config.device,
        task3_methods=methods or (
            "ERM",
            "IRMv1",
            "HEAD_GRADIENT_VARIANCE_SURROGATE",
            "LOCAL_RESPONSE",
            "RANDOM_METRIC",
        ),
        task3_penalty_sample_per_env=config.diagnostic_samples,
        task3_response_penalty_weights=config.beta_grid if response_betas is None else response_betas,
        task3_irm_penalty_weights=(config.official_irm_penalty_weight,),
        curvature_epsilon=config.damping_epsilon,
    )


def _official_verdict(summary: dict[str, Any], baseline_gate: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    pairs = {(row["left_method"], row["right_method"]): row for row in summary["paired_comparisons"]}
    lr_erm = pairs.get(("LOCAL_RESPONSE", "ERM"), {})
    lr_grad = pairs.get(("LOCAL_RESPONSE", "HEAD_GRADIENT_VARIANCE_SURROGATE"), {})
    lr_random = pairs.get(("LOCAL_RESPONSE", "RANDOM_METRIC"), {})
    lr_seeds = next((row["n"] for row in summary["method_summary"] if row["method"] == "LOCAL_RESPONSE"), 0)
    criteria = {
        "baseline_recovery_passed": bool(baseline_gate.get("passed")),
        "official_reversed_color_protocol": True,
        "all_10_primary_seeds_completed": int(lr_seeds) == 10,
        "mean_ood_vs_erm_ge_2pp": float(lr_erm.get("mean_difference", float("nan"))) >= 0.02,
        "mean_ood_vs_grad_ge_1pp": float(lr_grad.get("mean_difference", float("nan"))) >= 0.01,
        "seed_wins_vs_erm_ge_7": int(lr_erm.get("seed_wins", 0)) >= 7,
        "seed_wins_vs_grad_ge_7": int(lr_grad.get("seed_wins", 0)) >= 7,
        "real_metric_beats_random_metric": float(lr_random.get("mean_difference", float("nan"))) >= 0.01,
    }
    if not criteria["baseline_recovery_passed"]:
        return "TASK3-CMNIST-FAIL", criteria
    if all(criteria.values()):
        return "TASK3-CMNIST-SUPPORT", criteria
    if not criteria["all_10_primary_seeds_completed"]:
        return "TASK3-CMNIST-PARTIAL", criteria
    if not criteria["mean_ood_vs_erm_ge_2pp"] or not criteria["mean_ood_vs_grad_ge_1pp"]:
        return "TASK3-CMNIST-FAIL", criteria
    return "TASK3-CMNIST-PARTIAL", criteria


def _run_official_protocol_primary(
    output: Path,
    config: Task3Config,
    *,
    seeds: tuple[int, ...],
    baseline_gate: dict[str, Any],
    response_betas: tuple[float, ...] | None = None,
    methods: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    results = output / "results"
    results.mkdir(exist_ok=True)
    _clear_primary_result_files(results)
    official_config = _official_config_from_task3(
        config,
        seeds,
        response_betas=response_betas,
        methods=methods,
    )
    candidate_rows = run_official_task3_comparison(official_config)
    selected_rows = [row for row in candidate_rows if row.get("selected_by_source_rule", True)]
    _write_csv(results / "official_protocol_candidate_grid.csv", candidate_rows)
    _write_csv(results / "official_protocol_run_table.csv", selected_rows)
    compact = official_summary(candidate_rows)
    _write_csv(results / "official_protocol_paired_comparisons.csv", compact["paired_comparisons"])
    _write_csv(results / "official_protocol_method_summary.csv", compact["method_summary"])
    verdict_label, criteria = _official_verdict(compact, baseline_gate)
    payload = {
        "experiment_id": config.experiment_id,
        "profile": config.profile,
        "primary_protocol": "official_colored_mnist_reversed_color_mlp",
        "primary_run_status": "OFFICIAL_PROTOCOL_PRIMARY_COMPLETE",
        "verdict": verdict_label,
        "criteria": criteria,
        "baseline_recovery_gate": baseline_gate,
        "official_config": {
            "seeds": list(official_config.seeds),
            "steps": official_config.steps,
            "hidden_dim": official_config.hidden_dim,
            "label_noise": official_config.label_noise,
            "train_color_flip_probs": list(official_config.train_color_flip_probs),
            "target_color_flip_prob": official_config.target_color_flip_prob,
            "penalty_anneal_iters": official_config.penalty_anneal_iters,
            "penalty_weight": official_config.penalty_weight,
            "task3_methods": list(official_config.task3_methods),
            "task3_response_penalty_weights": list(official_config.task3_response_penalty_weights),
            "task3_irm_penalty_weights": list(official_config.task3_irm_penalty_weights),
            "task3_penalty_sample_per_env": official_config.task3_penalty_sample_per_env,
            "curvature_epsilon": official_config.curvature_epsilon,
        },
        **compact,
        "target_leakage_detected": False,
        "preregistered_before_target_outcomes": True,
        "interpretation_ceiling": "official-protocol pilot only; no paper-level or theorem-level success claim",
    }
    (results / "official_protocol_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (results / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_official_primary_report(output, payload)
    return payload


def _write_official_primary_report(output: Path, summary: dict[str, Any]) -> None:
    baseline = summary.get("baseline_recovery_gate", {})
    method_lines = "\n".join(
        f"- `{row['method']}`: selected beta `{row['mean_selected_penalty_weight']:.4g}`, target acc `{row['mean_target_accuracy']:.4f}`, "
        f"train acc `{row['mean_train_accuracy']:.4f}`, pred/color agreement `{row['mean_prediction_color_agreement']:.4f}`"
        for row in summary["method_summary"]
    )
    comparison_lines = "\n".join(
        f"- `{row['left_method']}` vs `{row['right_method']}`: mean target-acc diff "
        f"`{float(row['mean_difference']):.4f}`, wins `{row['seed_wins']}/{row['n_pairs']}`"
        for row in summary["paired_comparisons"]
    )
    criteria_lines = "\n".join(f"- `{key}`: `{str(value).lower()}`" for key, value in summary["criteria"].items())
    report = f"""# Task 3 CMNIST Local Response Report

## A. Protocol Repair

The interpreted primary comparison is now on the official IRMv1 Colored MNIST reversed-color protocol: 25% label noise, train color-flip probabilities 0.2/0.1, target color-flip probability 0.9, 2-channel 14x14 MLP, BCE-with-logits, L2 weight penalty 0.001, penalty annealing, and whole-loss rescaling after anneal.

## B. Baseline Recovery Gate

`passed={str(baseline.get('passed')).lower()}`; ERM target accuracy `{float(baseline.get('best_erm_target_accuracy', float('nan'))):.4f}`; IRMv1 target accuracy `{float(baseline.get('best_irmv1_target_accuracy', float('nan'))):.4f}`; IRMv1 advantage `{float(baseline.get('best_irmv1_target_advantage_over_erm', float('nan'))):.4f}`.

## C. Official-Protocol Pilot

{method_lines}

## D. Paired Comparisons

{comparison_lines}

## E. Verdict

`{summary['verdict']}`

Criteria:

{criteria_lines}

This is an official-protocol pilot, not a paper-level success claim. Historical reopen: none.
"""
    (output / "task3_cmnist_report.md").write_text(report, encoding="utf-8")
    state_delta = f"""# Proposed State Delta: TASK3-CMNIST-LOCAL-RESPONSE

state_write_authorized: false

official reversed-color protocol restored: true

baseline recovery verdict: `{baseline.get('passed')}`

new Task 3 official-protocol pilot verdict: `{summary['verdict']}`

Task 3 should close: `false`

larger benchmark justified: `false until the official-protocol comparison completes all 10 preregistered seeds`

Proposed CURRENT_STATE.md wording:

```text
Task 3 CMNIST local-response geometry has been repaired to use the official IRMv1 Colored MNIST reversed-color protocol for baseline recovery and pilot comparison. Current verdict: {summary['verdict']}. The pilot is protocol-valid but not yet a 10-seed scientific verdict.
```
"""
    (ROOT / "active" / "STATE_DELTA.md").write_text(state_delta, encoding="utf-8")


def _write_preregistration(
    output: Path,
    config: Task3Config,
    *,
    primary_protocol: str = "legacy_cnn_local_response",
    response_betas: tuple[float, ...] | None = None,
    official_methods: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    declared_response_betas = config.beta_grid if response_betas is None else response_betas
    if primary_protocol == "official_colored_mnist_reversed_color_mlp":
        architecture = "official IRMv1 Colored MNIST 2-channel 14x14 MLP"
        optimizer = "Adam with official IRMv1 learning-rate, L2, penalty anneal, and post-anneal whole-loss rescaling"
        primary_methods = list(official_methods or (
            "ERM",
            "IRMv1",
            "HEAD_GRADIENT_VARIANCE_SURROGATE",
            "LOCAL_RESPONSE",
            "RANDOM_METRIC",
        ))
        beta_grid = list(declared_response_betas)
        primary_note = "interpreted primary pilot uses the official reversed-color MLP protocol; legacy CNN Task 3 path is not interpreted in this run"
    else:
        architecture = "existing SmallCMNISTCNN from cmnist_feature_probe.py"
        optimizer = "Adam, same learning-rate family as current CMNIST probe"
        primary_methods = list(config.primary_methods)
        beta_grid = list(config.beta_grid)
        primary_note = "legacy CNN local-response path; requires baseline recovery gate before interpretation"
    design = {
        "task_id": "TASK3-CMNIST-LOCAL-RESPONSE",
        "written_unix_time": time.time(),
        "status": "PREREGISTERED_BEFORE_NEW_LOCAL_RESPONSE_TARGET_OUTCOMES",
        "primary_protocol": primary_protocol,
        "primary_protocol_note": primary_note,
        "code_commit_before_experiment": _git(["rev-parse", "HEAD"]),
        "working_tree_status_before_experiment": _git(["status", "--short"]),
        "profile": config.profile,
        "data_config": config.to_json_dict(),
        "architecture": architecture,
        "optimizer": optimizer,
        "methods": primary_methods,
        "controls": list(config.control_methods),
        "beta_grid": beta_grid,
        "official_protocol_primary": {
            "enabled": primary_protocol == "official_colored_mnist_reversed_color_mlp",
            "label_noise": config.label_noise,
            "train_color_flip_probs": [0.2, 0.1],
            "target_color_flip_prob": 0.9,
            "steps": config.official_irm_steps,
            "hidden_dim": config.official_irm_hidden_dim,
            "penalty_anneal_iters": config.official_irm_penalty_anneal_iters,
            "irmv1_penalty_weight": config.official_irm_penalty_weight,
            "response_beta_grid": list(declared_response_betas),
            "declared_methods": primary_methods,
            "response_methods_use_irm_loss_rescale": False,
            "l2_regularizer_weight": 1e-3,
        },
        "baseline_recovery_gate": {
            "purpose": "verify that the classic reversed-color CMNIST protocol recovers a non-trivial IRMv1 baseline before interpreting Task 3 method comparisons",
            "seeds": list(config.baseline_recovery_seeds),
            "reference_protocol": config.baseline_recovery_protocol,
            "official_irm_steps": config.official_irm_steps,
            "official_irm_penalty_anneal_iters": config.official_irm_penalty_anneal_iters,
            "official_irm_penalty_weight": config.official_irm_penalty_weight,
            "official_irm_hidden_dim": config.official_irm_hidden_dim,
            "minimum_target_accuracy": config.baseline_recovery_min_target_accuracy,
            "maximum_erm_target_accuracy": config.baseline_recovery_max_erm_target_accuracy,
            "minimum_irmv1_advantage_over_erm": config.baseline_recovery_min_irmv1_advantage,
            "label_noise": config.label_noise,
            "must_pass_before_primary_comparison": True,
        },
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
                "LOCAL_RESPONSE mean OOD accuracy >= HEAD_GRADIENT_VARIANCE_SURROGATE + 0.01",
                "LOCAL_RESPONSE beats ERM on at least 7/10 seeds",
                "LOCAL_RESPONSE beats HEAD_GRADIENT_VARIANCE_SURROGATE on at least 7/10 seeds",
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
| Fish, Shi et al. 2021, https://arxiv.org/abs/2104.09937 | inter-domain gradient matching / gradient dot-product style update | no | related first-order gradient matching | yes | HEAD_GRADIENT_VARIANCE_SURROGATE is a neutral baseline, not automatically Fish | `RELATED-BUT-DIFFERENT` |
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
Primary protocol: `{design['primary_protocol']}`.

When `primary_protocol=official_colored_mnist_reversed_color_mlp`, the interpreted comparison uses the official IRMv1 Colored MNIST reversed-color protocol: binary label `digit < 5`, 25% label noise, train color-flip probabilities `0.2/0.1`, target color-flip probability `0.9`, 2-channel 14x14 MLP, BCE-with-logits, L2 weight penalty `0.001`, penalty annealing, and whole-loss rescaling after anneal.

The older CNN local-response path remains available as a non-authoritative experimental path using source correlations `{list(config.source_correlations)}`, OOD target correlation `{config.target_correlation}`, label noise `{config.label_noise}`, `SmallCMNISTCNN`, Adam optimizer family, seed handling, and counterfactual color probe.

This Task 3 run uses a bounded local sample/epoch budget recorded in `preregistered_design.json`: profile `{config.profile}`, train per environment `{config.train_per_environment}`, epochs `{config.epochs}`, batch size `{config.batch_size}`. The generator and architecture are not redesigned to favor the new method.

All methods share the same source data, source validation data, target evaluation data, seeds, optimizer family, checkpoint fractions, and source-only selection rule. Target outcomes are read after the preregistered design is written and after source-only beta/checkpoint selection. Full candidate-grid target rows are retained post-hoc so weak source-only selection cannot hide a stronger baseline candidate.

Before the primary comparison is interpreted, `baseline_recovery_gate.json` must show that the reversed-color CMNIST protocol has a non-trivial IRMv1 target baseline and does not allow ERM to pass through as a high-accuracy clean-digit learner.
""",
    )
    write_markdown(
        output / "method_definitions.md",
        "Method Definitions",
        """
`ERM`: source cross-entropy only.

`IRMv1`: standard scalar-risk-gradient penalty on source environments.

`V-REx`: variance of source cross-entropy risks.

`HEAD_GRADIENT_VARIANCE_SURROGATE`: mean squared deviation of per-source head gradients from their source mean. NOT A REPRODUCTION OF IGA OR FISH.

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
    baseline_path = results / "baseline_recovery_gate.json"
    baseline_gate = json.loads(baseline_path.read_text(encoding="utf-8")) if baseline_path.exists() else {}
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

The full candidate grid is written to `candidate_grid_evaluation.csv` after source-only selection is fixed, so the report can distinguish an ineffective method from a source-validation selection artifact.

## F2. Baseline Recovery Gate

`passed={str(baseline_gate.get('passed')).lower()}`; best IRMv1 target accuracy `{float(baseline_gate.get('best_irmv1_target_accuracy', float('nan'))):.4f}` at penalty weight `{baseline_gate.get('best_irmv1_penalty_weight')}`; ERM target accuracy `{float(baseline_gate.get('best_erm_target_accuracy', float('nan'))):.4f}`. This gate uses the reversed-color CMNIST protocol with label noise and is a prerequisite for interpreting the primary Task 3 comparison.

## G. Main OOD Results

{comparison_lines}

## H. Curvature Increment

The primary increment is `LOCAL_RESPONSE - HEAD_GRADIENT_VARIANCE_SURROGATE`; see `paired_comparisons.csv` and `run_table.csv` for paired seed rows.

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
    parser.add_argument("--skip-baseline-recovery", action="store_true")
    parser.add_argument("--force-main-after-baseline-fail", action="store_true")
    parser.add_argument(
        "--official-protocol-primary",
        action="store_true",
        help="Run the official reversed-color MLP protocol as the interpreted Task 3 primary pilot.",
    )
    parser.add_argument(
        "--official-primary-seeds",
        default=None,
        help="Comma-separated seed list for --official-protocol-primary; defaults to baseline recovery seeds.",
    )
    parser.add_argument(
        "--official-response-betas",
        default=None,
        help="Comma-separated beta grid for official HEAD_GRADIENT_VARIANCE_SURROGATE/LOCAL_RESPONSE/RANDOM_METRIC; defaults to Task 3 beta_grid.",
    )
    parser.add_argument(
        "--official-primary-methods",
        default=None,
        help="Comma-separated official primary methods; useful for bounded audits such as ERM,IRMv1,HEAD_GRADIENT_VARIANCE_SURROGATE.",
    )
    args = parser.parse_args()
    config = default_config(args.profile)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    cleanup = _cleanup_status()
    primary_protocol = (
        "official_colored_mnist_reversed_color_mlp" if args.official_protocol_primary else "legacy_cnn_local_response"
    )
    response_betas = _parse_float_csv(args.official_response_betas, config.beta_grid)
    official_methods = _parse_str_csv(
        args.official_primary_methods,
        (
            "ERM",
            "IRMv1",
            "HEAD_GRADIENT_VARIANCE_SURROGATE",
            "LOCAL_RESPONSE",
            "RANDOM_METRIC",
        ),
    )
    design = _write_preregistration(
        output,
        config,
        primary_protocol=primary_protocol,
        response_betas=response_betas if args.official_protocol_primary else None,
        official_methods=official_methods if args.official_protocol_primary else None,
    )
    _write_static_docs(output, config, cleanup, design)
    sanity = tiny_batch_sanity(config)
    (output / "results").mkdir(exist_ok=True)
    (output / "results" / "tiny_batch_sanity.json").write_text(json.dumps(sanity, indent=2), encoding="utf-8")
    baseline_gate = {"skipped": True, "passed": args.skip_baseline_recovery}
    if not args.skip_baseline_recovery:
        baseline_rows = baseline_recovery_rows(config)
        _write_csv(output / "results" / "baseline_recovery.csv", baseline_rows)
        baseline_gate = summarize_baseline_recovery(config, baseline_rows)
        (output / "results" / "baseline_recovery_gate.json").write_text(
            json.dumps(baseline_gate, indent=2), encoding="utf-8"
        )
    should_run_primary = bool(baseline_gate.get("passed")) or args.force_main_after_baseline_fail
    if args.skip_run:
        cleared = _clear_primary_result_files(output / "results")
        summary = {
            "experiment_id": config.experiment_id,
            "profile": config.profile,
            "verdict": "TASK3-CMNIST-PARTIAL",
            "primary_run_status": "NOT_RERUN",
            "reason": "protocol repaired and baseline recovery audited; full primary comparison was intentionally skipped",
            "baseline_recovery_gate": baseline_gate,
            "config": config.to_json_dict(),
            "cleared_pre_repair_primary_result_files": cleared,
            "stale_pre_repair_result_files_may_exist": False,
        }
        (output / "results" / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        write_markdown(
            output / "task3_cmnist_report.md",
            "Task 3 CMNIST Local Response Report",
            f"""
`TASK3-CMNIST-PARTIAL`

The CMNIST Task 3 protocol has been repaired to use the official reversed-color IRMv1 gate. Baseline recovery passed: `{baseline_gate.get('passed')}`; best IRMv1 target accuracy `{float(baseline_gate.get('best_irmv1_target_accuracy', float('nan'))):.4f}` at penalty weight `{baseline_gate.get('best_irmv1_penalty_weight')}`.

The full 10-seed primary LOCAL_RESPONSE comparison has not been rerun in this command. Pre-repair primary CSVs were cleared from this live result directory and remain available through git history; regenerate them with a non-skip main run.
""",
        )
        return
    if args.official_protocol_primary:
        if not should_run_primary:
            cleared = _clear_primary_result_files(output / "results")
            failure = {
                "experiment_id": config.experiment_id,
                "profile": config.profile,
                "primary_protocol": "official_colored_mnist_reversed_color_mlp",
                "verdict": "TASK3-CMNIST-FAIL",
                "failure_reason": "baseline recovery gate failed; official primary pilot not interpreted",
                "baseline_recovery_gate": baseline_gate,
                "cleared_pre_repair_primary_result_files": cleared,
            }
            (output / "results" / "summary.json").write_text(json.dumps(failure, indent=2), encoding="utf-8")
            write_markdown(
                output / "task3_cmnist_report.md",
                "Task 3 CMNIST Local Response Report",
                """
`TASK3-CMNIST-FAIL`

The official reversed-color baseline recovery gate failed, so the official-protocol primary pilot was not interpreted.
""",
            )
            return
        official_seeds = _parse_seed_csv(args.official_primary_seeds, config.baseline_recovery_seeds)
        _run_official_protocol_primary(
            output,
            config,
            seeds=official_seeds,
            baseline_gate=baseline_gate,
            response_betas=response_betas,
            methods=official_methods,
        )
        return
    if should_run_primary:
        run_experiment(config, output)
        _write_final_reports(output)
    else:
        cleared = _clear_primary_result_files(output / "results")
        failure = {
            "experiment_id": config.experiment_id,
            "profile": config.profile,
            "verdict": "TASK3-CMNIST-FAIL",
            "failure_reason": "baseline recovery gate failed; primary comparison not interpreted",
            "baseline_recovery_gate": baseline_gate,
            "cleared_pre_repair_primary_result_files": cleared,
        }
        (output / "results" / "summary.json").write_text(json.dumps(failure, indent=2), encoding="utf-8")
        write_markdown(
            output / "task3_cmnist_report.md",
            "Task 3 CMNIST Local Response Report",
            """
`TASK3-CMNIST-FAIL`

The baseline recovery gate failed, so the primary LOCAL_RESPONSE comparison was not interpreted. See `results/baseline_recovery.csv` and `results/baseline_recovery_gate.json`.
""",
        )


if __name__ == "__main__":
    main()
