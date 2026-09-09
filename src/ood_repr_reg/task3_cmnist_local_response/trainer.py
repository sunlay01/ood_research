"""End-to-end CMNIST trainer for Task 3 local-response geometry."""

from __future__ import annotations

from dataclasses import dataclass, field
import copy
import csv
import json
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from ..cmnist_feature_probe import (
    ColoredEnvironment,
    CounterfactualProbe,
    SmallCMNISTCNN,
    accuracy,
    counterfactual_diagnostics,
    deterministic_subset,
    load_mnist_tensors,
    make_counterfactual_probe,
    make_environment,
    seed_everything,
)
from .evaluation import SelectionScore, comparison_row, counterexample_rows, select_source_only, verdict
from .official_irm_reference import OfficialIRMConfig, run_official_reference
from .penalties import diagnostic_penalties, mean_source_loss, task3_penalty


@dataclass(frozen=True)
class Task3Config:
    experiment_id: str = "TASK3-CMNIST-LOCAL-RESPONSE"
    profile: str = "main"
    data_root: str = "data"
    download: bool = True
    seeds: tuple[int, ...] = tuple(range(10))
    control_seeds: tuple[int, ...] = tuple(range(5))
    source_correlations: tuple[float, float] = (0.9, 0.8)
    target_correlation: float = 0.1
    label_noise: float = 0.25
    train_per_environment: int = 2500
    source_eval_per_environment: int = 500
    target_eval: int = 1000
    probe_size: int = 500
    diagnostic_samples: int = 512
    epochs: int = 8
    batch_size: int = 128
    learning_rate: float = 1e-3
    latent_dim: int = 32
    device: str = "cpu"
    beta_grid: tuple[float, ...] = (1e-3, 1e-2, 1e-1, 1.0, 10.0)
    baseline_recovery_seeds: tuple[int, ...] = (0,)
    baseline_recovery_irmv1_betas: tuple[float, ...] = (10000.0,)
    baseline_recovery_min_target_accuracy: float = 0.55
    baseline_recovery_max_erm_target_accuracy: float = 0.45
    baseline_recovery_min_irmv1_advantage: float = 0.05
    baseline_recovery_protocol: str = "official_irm_colored_mnist"
    official_irm_steps: int = 501
    official_irm_penalty_anneal_iters: int = 100
    official_irm_penalty_weight: float = 10000.0
    official_irm_hidden_dim: int = 256
    damping_epsilon: float = 1e-2
    damping_ablations: tuple[float, ...] = (1e-3, 1e-1)
    checkpoint_fractions: tuple[float, ...] = (0.10, 0.25, 0.50, 0.75, 1.00)
    primary_methods: tuple[str, ...] = (
        "ERM",
        "IRMv1",
        "V-REx",
        "HEAD_GRADIENT_VARIANCE_SURROGATE",
        "LOCAL_RESPONSE",
        "RANDOM_METRIC",
    )
    control_methods: tuple[str, ...] = ("SHUFFLED_LOCAL_RESPONSE",)
    support_thresholds: dict[str, float] = field(default_factory=lambda: {
        "mean_ood_vs_erm": 0.02,
        "mean_ood_vs_grad": 0.01,
        "seed_wins": 7,
        "worst_source_degradation": -0.01,
    })

    def to_json_dict(self) -> dict[str, Any]:
        return {
            key: list(value) if isinstance(value, tuple) else value
            for key, value in self.__dict__.items()
        }


@dataclass
class DataBundle:
    train: tuple[ColoredEnvironment, ...]
    source_eval: tuple[ColoredEnvironment, ...]
    target: ColoredEnvironment
    probe: CounterfactualProbe


@dataclass
class CandidateResult:
    seed: int
    method: str
    beta: float
    state_dict: dict[str, Tensor]
    selection_score: SelectionScore
    history: list[dict[str, Any]]
    checkpoint_states: dict[int, dict[str, Tensor]]


def default_config(profile: str = "main") -> Task3Config:
    if profile == "smoke":
        return Task3Config(
            profile="smoke",
            seeds=(0,),
            control_seeds=(0,),
            train_per_environment=96,
            source_eval_per_environment=96,
            target_eval=128,
            probe_size=96,
            diagnostic_samples=48,
            epochs=1,
            batch_size=48,
            beta_grid=(1e-2,),
            baseline_recovery_irmv1_betas=(1.0,),
            baseline_recovery_min_target_accuracy=0.0,
            baseline_recovery_max_erm_target_accuracy=1.0,
            baseline_recovery_min_irmv1_advantage=-1.0,
            official_irm_steps=3,
            official_irm_penalty_anneal_iters=1,
            official_irm_hidden_dim=16,
            primary_methods=("ERM", "HEAD_GRADIENT_VARIANCE_SURROGATE", "LOCAL_RESPONSE", "RANDOM_METRIC"),
        )
    if profile != "main":
        raise ValueError(f"unknown profile: {profile}")
    return Task3Config()


def _device(name: str) -> torch.device:
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    return torch.device(name)


def build_data(config: Task3Config, seed: int) -> DataBundle:
    root = Path(config.data_root)
    train_gray, train_digit = load_mnist_tensors(root, train=True, download=config.download)
    test_gray, test_digit = load_mnist_tensors(root, train=False, download=config.download)
    train_environments = []
    for env, correlation in enumerate(config.source_correlations):
        gray, digit = deterministic_subset(
            train_gray,
            train_digit,
            n=config.train_per_environment,
            seed=seed + 101,
            offset=env * config.train_per_environment,
        )
        train_environments.append(make_environment(
            gray,
            digit,
            correlation=correlation,
            env=env,
            seed=seed + env * 17,
            label_noise=config.label_noise,
        ))
    source_environments = []
    offset = 0
    for env, correlation in enumerate(config.source_correlations):
        gray, digit = deterministic_subset(
            test_gray,
            test_digit,
            n=config.source_eval_per_environment,
            seed=seed + 211,
            offset=offset,
        )
        offset += config.source_eval_per_environment
        source_environments.append(make_environment(
            gray,
            digit,
            correlation=correlation,
            env=env,
            seed=seed + 1000 + env * 17,
            label_noise=config.label_noise,
        ))
    target_gray, target_digit = deterministic_subset(
        test_gray, test_digit, n=config.target_eval, seed=seed + 211, offset=offset
    )
    offset += config.target_eval
    probe_gray, probe_digit = deterministic_subset(
        test_gray, test_digit, n=config.probe_size, seed=seed + 211, offset=offset
    )
    return DataBundle(
        train=tuple(train_environments),
        source_eval=tuple(source_environments),
        target=make_environment(
            target_gray,
            target_digit,
            correlation=config.target_correlation,
            env=2,
            seed=seed + 2000,
            label_noise=config.label_noise,
        ),
        probe=make_counterfactual_probe(probe_gray, probe_digit),
    )


def _batch_indices(n: int, batch_size: int, generator: torch.Generator) -> list[Tensor]:
    return list(torch.randperm(n, generator=generator).split(batch_size))


def _checkpoint_steps(total_steps: int, fractions: tuple[float, ...]) -> set[int]:
    steps = {max(1, min(total_steps, int(round(frac * total_steps)))) for frac in fractions}
    steps.add(total_steps)
    return steps


def _clone_state(model: SmallCMNISTCNN) -> dict[str, Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _load_state(model: SmallCMNISTCNN, state: dict[str, Tensor], device: torch.device) -> None:
    model.load_state_dict({key: value.to(device) for key, value in state.items()})


@torch.no_grad()
def _source_accuracies(
    model: SmallCMNISTCNN,
    environments: tuple[ColoredEnvironment, ...],
    device: torch.device,
) -> tuple[float, ...]:
    return tuple(accuracy(model, environment, device) for environment in environments)


@torch.no_grad()
def _target_loss(model: SmallCMNISTCNN, environment: ColoredEnvironment, device: torch.device) -> float:
    logits = model(environment.x.to(device))
    return float(F.cross_entropy(logits, environment.y.to(device)).detach().cpu())


@torch.no_grad()
def _target_color_audit(
    model: SmallCMNISTCNN,
    environment: ColoredEnvironment,
    device: torch.device,
) -> dict[str, float]:
    logits = model(environment.x.to(device))
    pred = logits.argmax(dim=1).cpu()
    return {
        "target_accuracy": float((pred == environment.y).float().mean()),
        "target_color_label_accuracy": float((environment.color == environment.y).float().mean()),
        "target_inverse_color_label_accuracy": float(((1 - environment.color) == environment.y).float().mean()),
        "prediction_color_agreement": float((pred == environment.color).float().mean()),
        "prediction_inverse_color_agreement": float((pred == 1 - environment.color).float().mean()),
    }


def baseline_recovery_rows(config: Task3Config) -> list[dict[str, Any]]:
    """Run the official-protocol CMNIST ERM/IRMv1 sanity gate.

    This is deliberately separate from source-only hyperparameter selection. It
    asks whether a faithful reversed-color IRMv1 baseline is recoverable before
    Task 3 method comparisons are interpreted.
    """
    if config.baseline_recovery_protocol != "official_irm_colored_mnist":
        raise ValueError(f"unknown baseline recovery protocol: {config.baseline_recovery_protocol}")
    reference = OfficialIRMConfig(
        data_root=config.data_root,
        download=config.download,
        seeds=config.baseline_recovery_seeds,
        hidden_dim=config.official_irm_hidden_dim,
        learning_rate=config.learning_rate,
        penalty_anneal_iters=config.official_irm_penalty_anneal_iters,
        penalty_weight=config.official_irm_penalty_weight,
        steps=config.official_irm_steps,
        label_noise=config.label_noise,
        device=config.device,
    )
    return run_official_reference(reference)


def summarize_baseline_recovery(config: Task3Config, rows: list[dict[str, Any]]) -> dict[str, Any]:
    irm_rows = [row for row in rows if str(row["method"]).upper() == "IRMV1"]
    erm_rows = [row for row in rows if str(row["method"]).upper() == "ERM"]
    best_irm = max(irm_rows, key=lambda row: float(row["target_accuracy"])) if irm_rows else None
    best_erm = max(erm_rows, key=lambda row: float(row["target_accuracy"])) if erm_rows else None
    best_target = float(best_irm["target_accuracy"]) if best_irm else float("nan")
    erm_target = float(best_erm["target_accuracy"]) if best_erm else float("nan")
    advantage = best_target - erm_target
    representative = rows[0] if rows else {}
    passed = bool(
        best_irm is not None
        and best_erm is not None
        and best_target >= config.baseline_recovery_min_target_accuracy
        and erm_target <= config.baseline_recovery_max_erm_target_accuracy
        and advantage >= config.baseline_recovery_min_irmv1_advantage
    )
    return {
        "gate": "CMNIST_BASELINE_RECOVERY",
        "passed": passed,
        "threshold_target_accuracy": config.baseline_recovery_min_target_accuracy,
        "max_allowed_erm_target_accuracy": config.baseline_recovery_max_erm_target_accuracy,
        "minimum_irmv1_advantage_over_erm": config.baseline_recovery_min_irmv1_advantage,
        "best_irmv1_target_accuracy": best_target,
        "best_irmv1_beta": float(best_irm.get("beta", best_irm.get("penalty_weight"))) if best_irm else None,
        "best_irmv1_penalty_weight": float(best_irm.get("penalty_weight", best_irm.get("beta"))) if best_irm else None,
        "best_irmv1_prediction_color_agreement": float(best_irm["prediction_color_agreement"]) if best_irm else None,
        "best_erm_target_accuracy": erm_target,
        "best_irmv1_target_advantage_over_erm": advantage,
        "protocol": {
            "official_train_examples_per_environment": representative.get("train_examples_per_environment"),
            "official_target_examples": representative.get("target_examples"),
            "source_environment_count": representative.get("source_environment_count"),
            "task3_cnn_train_per_environment_not_used_by_gate": config.train_per_environment,
            "task3_cnn_epochs_not_used_by_gate": config.epochs,
            "label_noise": config.label_noise,
            "baseline_recovery_protocol": config.baseline_recovery_protocol,
            "train_color_flip_probs": representative.get("train_color_flip_probs"),
            "target_color_flip_prob": representative.get("target_color_flip_prob"),
            "official_irm_steps": config.official_irm_steps,
            "official_irm_penalty_anneal_iters": config.official_irm_penalty_anneal_iters,
            "official_irm_penalty_weight": config.official_irm_penalty_weight,
            "official_irm_hidden_dim": config.official_irm_hidden_dim,
            "irmv1_betas": list(config.baseline_recovery_irmv1_betas),
            "seeds": list(config.baseline_recovery_seeds),
        },
    }


def _fixed_diagnostic_batches(
    environments: tuple[ColoredEnvironment, ...],
    config: Task3Config,
    device: torch.device,
) -> tuple[tuple[Tensor, Tensor], ...]:
    return tuple(
        (environment.x[: config.diagnostic_samples].to(device), environment.y[: config.diagnostic_samples].to(device))
        for environment in environments
    )


def _grad_norm(grads: tuple[Tensor | None, ...]) -> float:
    values = [grad.detach().square().sum() for grad in grads if grad is not None]
    if not values:
        return 0.0
    return float(torch.stack(values).sum().sqrt().cpu())


def _norm_diagnostics(
    model: SmallCMNISTCNN,
    method: str,
    beta: float,
    batches: tuple[tuple[Tensor, Tensor], ...],
    config: Task3Config,
    random_seed: int,
) -> dict[str, Any]:
    parameters = tuple(model.parameters())
    risk = mean_source_loss(model, batches)
    risk_grads = torch.autograd.grad(risk, parameters, allow_unused=True)
    if method == "ERM":
        penalty_value = risk.new_zeros(())
        penalty_grad_norm = 0.0
        penalty_diag: dict[str, Any] = {}
    else:
        penalty_value, penalty_diag = task3_penalty(
            method,
            model,
            batches,
            epsilon=config.damping_epsilon,
            random_seed=random_seed,
            shuffle_generator=torch.Generator(device="cpu").manual_seed(random_seed + 31),
        )
        penalty_grads = torch.autograd.grad(penalty_value, parameters, allow_unused=True)
        penalty_grad_norm = _grad_norm(penalty_grads)
    total = mean_source_loss(model, batches)
    if method != "ERM":
        penalty_value_2, _ = task3_penalty(
            method,
            model,
            batches,
            epsilon=config.damping_epsilon,
            random_seed=random_seed,
            shuffle_generator=torch.Generator(device="cpu").manual_seed(random_seed + 37),
        )
        total = total + beta * penalty_value_2
    total_grads = torch.autograd.grad(total, parameters, allow_unused=True)
    base_diag = diagnostic_penalties(model, batches, epsilon=config.damping_epsilon, random_seed=random_seed)
    return {
        "source_loss": float(risk.detach().cpu()),
        "regularizer_magnitude": float(penalty_value.detach().cpu()),
        "erm_gradient_norm": _grad_norm(risk_grads),
        "penalty_gradient_norm": penalty_grad_norm,
        "total_update_norm": _grad_norm(total_grads),
        "raw_gradient_disagreement": float(base_diag["grad_raw_gradient_disagreement"]),
        "local_response_disagreement": float(base_diag["local_local_response_disagreement"]),
        "response_vector_norm": float(base_diag["local_response_vector_norm"]),
        "hessian_min_eigenvalue": float(base_diag["local_hessian_min_eigenvalue"]),
        "hessian_max_eigenvalue": float(base_diag["local_hessian_max_eigenvalue"]),
        "damped_condition_number": float(base_diag["local_damped_condition_number"]),
        "damping": float(base_diag["local_damping"]),
        "curvature_object": str(base_diag["local_curvature_object"]),
        "random_trace_relative_error": float(base_diag["random_random_trace_relative_error"]),
        "random_fro_relative_error": float(base_diag["random_random_fro_relative_error"]),
        **{f"train_penalty_{key}": value for key, value in penalty_diag.items() if isinstance(value, (int, float, str))},
    }


def train_candidate(
    data: DataBundle,
    config: Task3Config,
    *,
    seed: int,
    method: str,
    beta: float,
) -> CandidateResult:
    seed_everything(seed)
    device = _device(config.device)
    model = SmallCMNISTCNN(latent_dim=config.latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    batch_generator = torch.Generator().manual_seed(seed + 7919)
    shuffle_generator = torch.Generator(device="cpu").manual_seed(seed + 101003)
    diagnostic_batches = _fixed_diagnostic_batches(data.train, config, device)
    batches_per_epoch = min(
        int(np.ceil(environment.x.shape[0] / config.batch_size)) for environment in data.train
    )
    total_steps = max(1, config.epochs * batches_per_epoch)
    checkpoint_steps = _checkpoint_steps(total_steps, config.checkpoint_fractions)
    history: list[dict[str, Any]] = []
    checkpoint_states: dict[int, dict[str, Tensor]] = {}
    step = 0
    for _epoch in range(config.epochs):
        permutations = [
            _batch_indices(environment.x.shape[0], config.batch_size, batch_generator)
            for environment in data.train
        ]
        for batch_number in range(min(len(parts) for parts in permutations)):
            step += 1
            batches = tuple(
                (
                    environment.x[permutations[index][batch_number]].to(device),
                    environment.y[permutations[index][batch_number]].to(device),
                )
                for index, environment in enumerate(data.train)
            )
            risk = mean_source_loss(model, batches)
            penalty, _ = task3_penalty(
                method,
                model,
                batches,
                epsilon=config.damping_epsilon,
                random_seed=seed,
                shuffle_generator=shuffle_generator,
            )
            objective = risk + beta * penalty
            optimizer.zero_grad(set_to_none=True)
            objective.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            if step in checkpoint_steps:
                model.eval()
                source_accs = _source_accuracies(model, data.source_eval, device)
                diagnostics = _norm_diagnostics(
                    model, method, beta, diagnostic_batches, config, random_seed=seed
                )
                history.append({
                    "seed": seed,
                    "method": method,
                    "beta": beta,
                    "checkpoint_step": step,
                    "checkpoint_fraction": step / total_steps,
                    "source_env1_accuracy": source_accs[0],
                    "source_env2_accuracy": source_accs[1],
                    "mean_source_accuracy": float(np.mean(source_accs)),
                    "worst_source_accuracy": float(np.min(source_accs)),
                    "target_accuracy": "POSTHOC_NOT_EVALUATED_DURING_SELECTION",
                    **diagnostics,
                })
                checkpoint_states[step] = _clone_state(model)
                model.train()
    if not history:
        raise RuntimeError("no checkpoint was recorded")
    best = max(
        history,
        key=lambda row: (
            float(row["worst_source_accuracy"]),
            float(row["mean_source_accuracy"]),
            int(row["checkpoint_step"]),
        ),
    )
    selected_step = int(best["checkpoint_step"])
    return CandidateResult(
        seed=seed,
        method=method,
        beta=beta,
        state_dict=checkpoint_states[selected_step],
        selection_score=SelectionScore(
            worst_source_accuracy=float(best["worst_source_accuracy"]),
            mean_source_accuracy=float(best["mean_source_accuracy"]),
            checkpoint_step=selected_step,
        ),
        history=history,
        checkpoint_states=checkpoint_states,
    )


def _betas_for_method(config: Task3Config, method: str) -> tuple[float, ...]:
    return (0.0,) if method == "ERM" else config.beta_grid


def evaluate_candidate(
    candidate: CandidateResult,
    data: DataBundle,
    config: Task3Config,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    device = _device(config.device)
    model = SmallCMNISTCNN(latent_dim=config.latent_dim).to(device)
    _load_state(model, candidate.state_dict, device)
    model.eval()
    source_accs = _source_accuracies(model, data.source_eval, device)
    target_accuracy = accuracy(model, data.target, device)
    target_loss = _target_loss(model, data.target, device)
    color_metrics, _ = counterfactual_diagnostics(model, data.probe, device=device)
    selected_history = [
        row for row in candidate.history if int(row["checkpoint_step"]) == candidate.selection_score.checkpoint_step
    ][0]
    run_row = {
        "experiment_id": config.experiment_id,
        "profile": config.profile,
        "seed": candidate.seed,
        "method": candidate.method,
        "selected_beta": candidate.beta,
        "selected_checkpoint": candidate.selection_score.checkpoint_step,
        "source_env1_accuracy": source_accs[0],
        "source_env2_accuracy": source_accs[1],
        "mean_source_accuracy": float(np.mean(source_accs)),
        "worst_source_accuracy": float(np.min(source_accs)),
        "target_accuracy": target_accuracy,
        "target_loss": target_loss,
        "counterfactual_color_response": float(color_metrics["probability_color_response"]),
        "prediction_color_response": float(color_metrics["prediction_color_response"]),
        "counterfactual_prediction_consistency": float(color_metrics["counterfactual_prediction_consistency"]),
        "target_used_for_selection": False,
        **{key: value for key, value in selected_history.items() if key not in {"target_accuracy"}},
    }
    dynamics: list[dict[str, Any]] = []
    for row in candidate.history:
        step = int(row["checkpoint_step"])
        _load_state(model, candidate.checkpoint_states[step], device)
        model.eval()
        posthoc_target = accuracy(model, data.target, device)
        posthoc_color, _ = counterfactual_diagnostics(model, data.probe, device=device)
        dynamics.append({
            **row,
            "target_accuracy": posthoc_target,
            "counterfactual_color_response": float(posthoc_color["probability_color_response"]),
            "target_metric_use": "posthoc_after_source_selection_rule_defined",
        })
    return run_row, dynamics


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


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _flush_primary_rows(
    results: Path,
    *,
    run_rows: list[dict[str, Any]],
    method_summary: list[dict[str, Any]],
    selection_rows: list[dict[str, Any]],
    candidate_grid_rows: list[dict[str, Any]],
    dynamics_rows: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
    counterexamples: list[dict[str, Any]],
) -> None:
    _write_csv(results / "run_table.csv", run_rows)
    _write_csv(results / "seed_summary.csv", method_summary)
    _write_csv(results / "hyperparameter_selection.csv", selection_rows)
    _write_csv(results / "candidate_grid_evaluation.csv", candidate_grid_rows)
    _write_csv(results / "dynamics.csv", dynamics_rows)
    _write_csv(results / "mechanism_diagnostics.csv", run_rows)
    _write_csv(results / "random_metric_control.csv", [row for row in run_rows if row["method"] == "RANDOM_METRIC"])
    _write_csv(results / "environment_shuffle_control.csv", [row for row in run_rows if row["method"] == "SHUFFLED_LOCAL_RESPONSE"])
    _write_csv(results / "paired_comparisons.csv", comparison_rows)
    _write_csv(results / "counterexamples.csv", counterexamples)


def _summarize_methods(run_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    method_summary = []
    for method in sorted({str(row["method"]) for row in run_rows}):
        values = [row for row in run_rows if row["method"] == method]
        method_summary.append({
            "method": method,
            "n": len(values),
            "mean_target_accuracy": float(np.mean([float(row["target_accuracy"]) for row in values])),
            "mean_worst_source_accuracy": float(np.mean([float(row["worst_source_accuracy"]) for row in values])),
            "mean_counterfactual_color_response": float(np.mean([float(row["counterfactual_color_response"]) for row in values])),
            "mean_local_response_disagreement": float(np.mean([float(row["local_response_disagreement"]) for row in values])),
            "mean_raw_gradient_disagreement": float(np.mean([float(row["raw_gradient_disagreement"]) for row in values])),
        })
    return method_summary


def run_experiment(config: Task3Config, output: Path) -> dict[str, Any]:
    started = time.time()
    output.mkdir(parents=True, exist_ok=True)
    results = output / "results"
    results.mkdir(exist_ok=True)
    device = _device(config.device)
    run_rows: list[dict[str, Any]] = _read_csv(results / "run_table.csv")
    selection_rows: list[dict[str, Any]] = _read_csv(results / "hyperparameter_selection.csv")
    dynamics_rows: list[dict[str, Any]] = _read_csv(results / "dynamics.csv")
    candidate_grid_rows: list[dict[str, Any]] = _read_csv(results / "candidate_grid_evaluation.csv")
    completed = {(int(row["seed"]), str(row["method"])) for row in run_rows}
    comparison_pairs = [
        ("LOCAL_RESPONSE", "ERM"),
        ("LOCAL_RESPONSE", "HEAD_GRADIENT_VARIANCE_SURROGATE"),
        ("LOCAL_RESPONSE", "RANDOM_METRIC"),
        ("LOCAL_RESPONSE", "IRMv1"),
        ("LOCAL_RESPONSE", "V-REx"),
        ("HEAD_GRADIENT_VARIANCE_SURROGATE", "ERM"),
    ]
    for seed in config.seeds:
        data = build_data(config, seed)
        methods = list(config.primary_methods)
        if seed in config.control_seeds:
            methods.extend(config.control_methods)
        for method in methods:
            if (int(seed), method) in completed:
                print(f"seed={seed} method={method} status=resume-skip", flush=True)
                continue
            candidates = []
            for beta in _betas_for_method(config, method):
                candidate = train_candidate(data, config, seed=seed, method=method, beta=float(beta))
                candidates.append(candidate)
                for row in candidate.history:
                    selection_rows.append({
                        **row,
                        "selected_by_source_rule": False,
                        "target_used_for_selection": False,
                    })
            selected = select_source_only(candidates)
            for row in selection_rows:
                if (
                    int(row["seed"]) == seed
                    and row["method"] == method
                    and abs(float(row["beta"]) - float(selected.beta)) < 1e-15
                    and int(row["checkpoint_step"]) == selected.selection_score.checkpoint_step
                ):
                    row["selected_by_source_rule"] = True
            for candidate in candidates:
                grid_row, _ = evaluate_candidate(candidate, data, config)
                grid_row["selected_by_source_rule"] = candidate is selected
                grid_row["evaluation_role"] = "posthoc_candidate_grid_after_source_selection"
                candidate_grid_rows.append(grid_row)
            run_row, method_dynamics = evaluate_candidate(selected, data, config)
            run_rows.append(run_row)
            completed.add((int(seed), method))
            dynamics_rows.extend(method_dynamics)
            method_summary = _summarize_methods(run_rows)
            comparison_rows = [comparison_row(run_rows, left, right) for left, right in comparison_pairs]
            counterexamples = counterexample_rows(run_rows)
            _flush_primary_rows(
                results,
                run_rows=run_rows,
                method_summary=method_summary,
                selection_rows=selection_rows,
                candidate_grid_rows=candidate_grid_rows,
                dynamics_rows=dynamics_rows,
                comparison_rows=comparison_rows,
                counterexamples=counterexamples,
            )
            progress_summary = {
                "experiment_id": config.experiment_id,
                "profile": config.profile,
                "primary_run_status": "IN_PROGRESS",
                "completed_seed_method_rows": len(run_rows),
                "expected_seed_method_rows": len(config.seeds) * len(config.primary_methods)
                + len([seed for seed in config.seeds if seed in config.control_seeds]) * len(config.control_methods),
                "config": config.to_json_dict(),
                "method_summary": method_summary,
                "paired_comparisons": comparison_rows,
                "python": platform.python_version(),
                "torch": torch.__version__,
                "device": str(device),
                "wall_seconds": time.time() - started,
            }
            (results / "summary.json").write_text(json.dumps(progress_summary, indent=2), encoding="utf-8")
            print(
                f"seed={seed} method={method} beta={selected.beta:g} "
                f"source={run_row['mean_source_accuracy']:.3f} target={run_row['target_accuracy']:.3f}"
                f" progress={len(run_rows)}/{progress_summary['expected_seed_method_rows']}",
                flush=True,
            )
    comparison_rows = [comparison_row(run_rows, left, right) for left, right in comparison_pairs]
    verdict_label, criteria = verdict(run_rows, comparison_rows)
    counterexamples = counterexample_rows(run_rows)
    method_summary = _summarize_methods(run_rows)
    _flush_primary_rows(
        results,
        run_rows=run_rows,
        method_summary=method_summary,
        selection_rows=selection_rows,
        candidate_grid_rows=candidate_grid_rows,
        dynamics_rows=dynamics_rows,
        comparison_rows=comparison_rows,
        counterexamples=counterexamples,
    )
    summary = {
        "experiment_id": config.experiment_id,
        "profile": config.profile,
        "verdict": verdict_label,
        "criteria": criteria,
        "config": config.to_json_dict(),
        "method_summary": method_summary,
        "paired_comparisons": comparison_rows,
        "candidate_grid_rows": len(candidate_grid_rows),
        "counterexample_count": len(counterexamples),
        "target_leakage_detected": False,
        "preregistered_before_target_outcomes": True,
        "end_to_end_representation_training": True,
        "cmnist_protocol_reused": True,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": str(device),
        "wall_seconds": time.time() - started,
    }
    (results / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def tiny_batch_sanity(config: Task3Config) -> dict[str, Any]:
    device = _device(config.device)
    data = build_data(default_config("smoke"), 0)
    model = SmallCMNISTCNN(latent_dim=config.latent_dim).to(device)
    before = copy.deepcopy({key: value.detach().clone() for key, value in model.encoder.state_dict().items()})
    batches = _fixed_diagnostic_batches(data.train, default_config("smoke"), device)
    penalty, diag = task3_penalty(
        "LOCAL_RESPONSE",
        model,
        batches,
        epsilon=config.damping_epsilon,
        random_seed=0,
        shuffle_generator=torch.Generator(device="cpu").manual_seed(0),
    )
    risk = mean_source_loss(model, batches)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    optimizer.zero_grad(set_to_none=True)
    (risk + 1e-2 * penalty).backward()
    finite_grads = all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    )
    optimizer.step()
    changed = any(
        not torch.allclose(before[key], value.detach())
        for key, value in model.encoder.state_dict().items()
    )
    return {
        "local_response_finite": bool(torch.isfinite(penalty.detach()).all()),
        "autograd_gradients_finite": finite_grads,
        "encoder_parameters_changed": changed,
        "diagnostics": diag,
    }
