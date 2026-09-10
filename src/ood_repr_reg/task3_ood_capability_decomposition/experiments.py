"""Coverage, separability, and selection experiments for frozen CMNIST encoders."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..task3_cmnist_counterfactual_audit.diagnostics import compute_counterfactual_diagnostics
from ..task3_cmnist_cpu_minimal.model import parameter_hash
from .artifacts import FeatureBundle
from .linear import (
    apply_projection,
    binary_accuracy_from_logits,
    classifier_accuracy,
    color_response_basis,
    fit_ridge_classifier,
    projection_matrix,
    safe_ratio,
    two_fold_ridge_accuracy,
)


RANKS = [0, 1, 2, 4, 8, 16, 32, 64]
RIDGE = 1e-3


@dataclass(frozen=True)
class CapabilityRows:
    coverage: list[dict[str, Any]]
    separability: list[dict[str, Any]]
    selection: list[dict[str, Any]]


def _labels(bundle: FeatureBundle) -> dict[str, Tensor]:
    return {
        "source0_noisy": bundle.data.source_envs[0].labels.detach().cpu().double().reshape(-1),
        "source1_noisy": bundle.data.source_envs[1].labels.detach().cpu().double().reshape(-1),
        "target_noisy": bundle.data.target_env.labels.detach().cpu().double().reshape(-1),
        "target_clean": bundle.probe.clean_labels.detach().cpu().double().reshape(-1),
    }


def _head_logits(bundle: FeatureBundle, features: Tensor) -> Tensor:
    weight = bundle.model.head.weight.detach().cpu().double().reshape(-1)
    bias = bundle.model.head.bias.detach().cpu().double().reshape(())
    return features.detach().cpu().double() @ weight + bias


def _linear_logits(head: nn.Linear, features: Tensor) -> Tensor:
    weight = head.weight.detach().cpu().double().reshape(-1)
    bias = head.bias.detach().cpu().double().reshape(())
    return features.detach().cpu().double() @ weight + bias


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _balanced_features(bundle: FeatureBundle) -> Tensor:
    return 0.5 * (bundle.red_features + bundle.green_features)


def _counterfactual_balanced_features(bundle: FeatureBundle) -> Tensor:
    return torch.cat((bundle.red_features, bundle.green_features), dim=0)


def _counterfactual_balanced_clean_labels(bundle: FeatureBundle) -> Tensor:
    clean = bundle.probe.clean_labels.detach().cpu().double().reshape(-1)
    return torch.cat((clean, clean), dim=0)


def _task_direction(features: Tensor, labels: Tensor) -> Tensor:
    y = labels.detach().cpu().double().reshape(-1)
    x = features.detach().cpu().double()
    return x[y == 1.0].mean(dim=0) - x[y == 0.0].mean(dim=0)


def _clean_balanced_accuracy_from_logits(red_logits: Tensor, green_logits: Tensor, clean_labels: Tensor) -> float:
    y = clean_labels.detach().cpu().double().reshape(-1)
    red = (red_logits.detach().cpu().double().reshape(-1) >= 0.0).double()
    green = (green_logits.detach().cpu().double().reshape(-1) >= 0.0).double()
    return float(0.5 * ((red == y).double().mean() + (green == y).double().mean()).item())


def run_feature_coverage(bundle: FeatureBundle) -> dict[str, Any]:
    labels = _labels(bundle)
    source_x = torch.cat((bundle.source0_features, bundle.source1_features), dim=0)
    source_y = torch.cat((labels["source0_noisy"], labels["source1_noisy"]), dim=0)
    source_probe = fit_ridge_classifier(source_x, source_y, ridge=RIDGE)
    balanced = _counterfactual_balanced_features(bundle)
    oracle = two_fold_ridge_accuracy(balanced, _counterfactual_balanced_clean_labels(bundle), ridge=RIDGE)
    original_target_acc = binary_accuracy_from_logits(_head_logits(bundle, bundle.target_features), labels["target_noisy"])
    return {
        "seed": bundle.record.seed,
        "encoder_method": bundle.record.method,
        "checkpoint_sha256": bundle.record.checkpoint_sha256,
        "source_only_probe": True,
        "oracle_clean_probe": True,
        "source_probe_source_env0_acc": classifier_accuracy(source_probe, bundle.source0_features, labels["source0_noisy"]),
        "source_probe_source_env1_acc": classifier_accuracy(source_probe, bundle.source1_features, labels["source1_noisy"]),
        "source_probe_source_mean_acc": _mean([
            classifier_accuracy(source_probe, bundle.source0_features, labels["source0_noisy"]),
            classifier_accuracy(source_probe, bundle.source1_features, labels["source1_noisy"]),
        ]),
        "source_probe_target_acc": classifier_accuracy(source_probe, bundle.target_features, labels["target_noisy"]),
        "oracle_clean_balanced_accuracy": oracle["accuracy"],
        "oracle_clean_fold0_accuracy": oracle["fold0_accuracy"],
        "oracle_clean_fold1_accuracy": oracle["fold1_accuracy"],
        "original_target_acc": original_target_acc,
        "coverage_release_vs_original": oracle["accuracy"] - original_target_acc,
        "ridge": RIDGE,
    }


def _projected_head_metrics(bundle: FeatureBundle, projection: Tensor) -> dict[str, float]:
    labels = _labels(bundle)
    s0 = apply_projection(bundle.source0_features, projection)
    s1 = apply_projection(bundle.source1_features, projection)
    target = apply_projection(bundle.target_features, projection)
    red = apply_projection(bundle.red_features, projection)
    green = apply_projection(bundle.green_features, projection)
    s0_acc = binary_accuracy_from_logits(_head_logits(bundle, s0), labels["source0_noisy"])
    s1_acc = binary_accuracy_from_logits(_head_logits(bundle, s1), labels["source1_noisy"])
    target_acc = binary_accuracy_from_logits(_head_logits(bundle, target), labels["target_noisy"])
    diagnostics = compute_counterfactual_diagnostics(
        red_z=red,
        green_z=green,
        red_logits=_head_logits(bundle, red),
        green_logits=_head_logits(bundle, green),
        head_weight=bundle.model.head.weight.detach().cpu().reshape(-1),
        clean_labels=labels["target_clean"],
    )
    return {
        "projected_source_env0_acc": s0_acc,
        "projected_source_env1_acc": s1_acc,
        "projected_source_mean_acc": _mean([s0_acc, s1_acc]),
        "projected_target_acc": target_acc,
        "scalar_logit_color_response": float(diagnostics["prediction_color_response"]),
        "probability_color_response": float(diagnostics["probability_color_response"]),
        "counterfactual_prediction_consistency": float(diagnostics["counterfactual_prediction_consistency"]),
        "counterfactual_prediction_flip_rate": float(diagnostics["counterfactual_prediction_flip_rate"]),
    }


def run_feature_separability(bundle: FeatureBundle, ranks: list[int] | None = None) -> list[dict[str, Any]]:
    labels = _labels(bundle)
    ranks = ranks or RANKS
    basis = color_response_basis(bundle.red_features, bundle.green_features)
    original_projection = projection_matrix(basis, 0)
    original = _projected_head_metrics(bundle, original_projection)
    rows: list[dict[str, Any]] = []
    rank0_clean_acc: float | None = None
    for rank in ranks:
        projection = projection_matrix(basis, rank)
        clipped_rank = min(max(int(rank), 0), bundle.red_features.shape[1])
        red = apply_projection(bundle.red_features, projection)
        green = apply_projection(bundle.green_features, projection)
        balanced = torch.cat((red, green), dim=0)
        metrics = _projected_head_metrics(bundle, projection)
        clean_oracle = two_fold_ridge_accuracy(balanced, torch.cat((labels["target_clean"], labels["target_clean"]), dim=0), ridge=RIDGE)
        if rank0_clean_acc is None:
            rank0_clean_acc = clean_oracle["accuracy"]
        color_features = torch.cat((red, green), dim=0)
        color_labels = torch.cat((torch.zeros(red.shape[0]), torch.ones(green.shape[0]))).double()
        color_probe = two_fold_ridge_accuracy(color_features, color_labels, ridge=RIDGE)
        rows.append({
            "seed": bundle.record.seed,
            "encoder_method": bundle.record.method,
            "rank_k": int(rank),
            "rank_clipped": int(clipped_rank),
            "projection_symmetric": bool(torch.allclose(projection, projection.T, atol=1e-8)),
            "projection_idempotent": bool(torch.allclose(projection @ projection, projection, atol=1e-8)),
            "original_target_acc": original["projected_target_acc"],
            "projected_target_acc": metrics["projected_target_acc"],
            "target_delta_vs_original": metrics["projected_target_acc"] - original["projected_target_acc"],
            "source_env0_acc": metrics["projected_source_env0_acc"],
            "source_env1_acc": metrics["projected_source_env1_acc"],
            "source_mean_acc": metrics["projected_source_mean_acc"],
            "scalar_logit_color_response": metrics["scalar_logit_color_response"],
            "scalar_logit_color_suppression_ratio": safe_ratio(
                metrics["scalar_logit_color_response"], original["scalar_logit_color_response"]
            ),
            "probability_color_response": metrics["probability_color_response"],
            "counterfactual_prediction_consistency": metrics["counterfactual_prediction_consistency"],
            "counterfactual_prediction_flip_rate": metrics["counterfactual_prediction_flip_rate"],
            "clean_oracle_coverage_acc": clean_oracle["accuracy"],
            "clean_oracle_delta_vs_rank0": clean_oracle["accuracy"] - rank0_clean_acc,
            "red_green_color_probe_acc": color_probe["accuracy"],
            "ridge": RIDGE,
        })
    return rows


def initialized_head(dim: int, seed: int) -> nn.Linear:
    torch.manual_seed(int(seed))
    head = nn.Linear(dim, 1)
    nn.init.xavier_uniform_(head.weight)
    nn.init.zeros_(head.bias)
    return head


def _l2_head(head: nn.Linear) -> Tensor:
    return head.weight.square().sum() + head.bias.square().sum()


def _irm_penalty(logits: Tensor, labels: Tensor) -> Tensor:
    scale = torch.tensor(1.0, dtype=logits.dtype, requires_grad=True)
    risk = F.binary_cross_entropy_with_logits(logits.reshape(-1, 1) * scale, labels.reshape(-1, 1).double())
    grad = torch.autograd.grad(risk, [scale], create_graph=True)[0]
    return grad.square().sum()


def train_source_head(
    bundle: FeatureBundle,
    *,
    method: str,
    initial_state: dict[str, Tensor],
    config: dict[str, Any],
) -> nn.Linear:
    if method not in {"HEAD_ERM", "HEAD_IRMv1"}:
        raise ValueError(f"unknown source head method: {method}")
    head = initialized_head(bundle.source0_features.shape[1], bundle.record.seed + 12345).double()
    head.load_state_dict(copy.deepcopy(initial_state))
    optimizer = torch.optim.Adam(head.parameters(), lr=float(config["training"]["learning_rate"]))
    l2 = float(config["training"]["l2_regularizer_weight"])
    anneal = int(config["irmv1"]["penalty_anneal_iters"])
    penalty_weight = float(config["irmv1"]["penalty_weight"])
    labels = _labels(bundle)
    for step in range(int(config["training"]["steps"])):
        idx0 = bundle.data.batch_schedule.indices[0][step]
        idx1 = bundle.data.batch_schedule.indices[1][step]
        logits0 = head(bundle.source0_features[idx0].double())
        logits1 = head(bundle.source1_features[idx1].double())
        y0 = labels["source0_noisy"][idx0].reshape(-1, 1)
        y1 = labels["source1_noisy"][idx1].reshape(-1, 1)
        risks = torch.stack([
            F.binary_cross_entropy_with_logits(logits0, y0),
            F.binary_cross_entropy_with_logits(logits1, y1),
        ])
        risk = risks.mean()
        objective = risk + l2 * _l2_head(head)
        if method == "HEAD_IRMv1":
            penalty = torch.stack([_irm_penalty(logits0, y0), _irm_penalty(logits1, y1)]).mean()
            applied = penalty_weight if step >= anneal else 1.0
            objective = risk + l2 * _l2_head(head) + applied * penalty
            if applied > 1.0:
                objective = objective / applied
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        optimizer.step()
    return head.eval()


def train_oracle_clean_head(
    bundle: FeatureBundle,
    *,
    initial_state: dict[str, Tensor],
    config: dict[str, Any],
) -> nn.Linear:
    head = initialized_head(bundle.source0_features.shape[1], bundle.record.seed + 12345).double()
    head.load_state_dict(copy.deepcopy(initial_state))
    optimizer = torch.optim.Adam(head.parameters(), lr=float(config["training"]["learning_rate"]))
    l2 = float(config["training"]["l2_regularizer_weight"])
    x = _counterfactual_balanced_features(bundle).double()
    y = _counterfactual_balanced_clean_labels(bundle).reshape(-1, 1)
    generator = torch.Generator().manual_seed(bundle.record.seed + 77731)
    indices = torch.randint(x.shape[0], (int(config["training"]["steps"]), int(config["training"]["batch_size_per_environment"])), generator=generator)
    for step in range(int(config["training"]["steps"])):
        idx = indices[step]
        logits = head(x[idx])
        loss = F.binary_cross_entropy_with_logits(logits, y[idx]) + l2 * _l2_head(head)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return head.eval()


def _head_selection_metrics(bundle: FeatureBundle, head: nn.Linear, *, head_method: str, oracle: bool, source_only: bool) -> dict[str, Any]:
    labels = _labels(bundle)
    basis = color_response_basis(bundle.red_features, bundle.green_features)
    task_dir = _task_direction(_balanced_features(bundle), labels["target_clean"])
    weight = head.weight.detach().cpu().double().reshape(-1)
    weight_norm = float(weight.norm().item())
    top1 = basis[:, :1]
    top8 = basis[:, : min(8, basis.shape[1])]
    color_top1 = float((top1.T @ weight).norm().item() / max(weight_norm, 1e-12))
    color_top8 = float((top8.T @ weight).norm().item() / max(weight_norm, 1e-12))
    task_alignment = float(abs(torch.dot(weight, task_dir)).item() / max(weight_norm * float(task_dir.norm().item()), 1e-12))
    red_logits = _linear_logits(head, bundle.red_features)
    green_logits = _linear_logits(head, bundle.green_features)
    source0_acc = binary_accuracy_from_logits(_linear_logits(head, bundle.source0_features), labels["source0_noisy"])
    source1_acc = binary_accuracy_from_logits(_linear_logits(head, bundle.source1_features), labels["source1_noisy"])
    target_acc = binary_accuracy_from_logits(_linear_logits(head, bundle.target_features), labels["target_noisy"])
    clean_balanced = _clean_balanced_accuracy_from_logits(red_logits, green_logits, labels["target_clean"])
    flip_rate = float(((red_logits.reshape(-1) >= 0.0) != (green_logits.reshape(-1) >= 0.0)).double().mean().item())
    return {
        "seed": bundle.record.seed,
        "encoder_method": bundle.record.method,
        "head_method": head_method,
        "source_only": bool(source_only),
        "oracle": bool(oracle),
        "source_env0_acc": source0_acc,
        "source_env1_acc": source1_acc,
        "source_mean_acc": _mean([source0_acc, source1_acc]),
        "target_acc": target_acc,
        "clean_balanced_accuracy": clean_balanced,
        "counterfactual_flip_rate": flip_rate,
        "counterfactual_consistency": 1.0 - flip_rate,
        "color_projection_ratio_top1": color_top1,
        "color_projection_ratio_top8": color_top8,
        "task_alignment_cosine": task_alignment,
    }


def run_feature_selection(bundle: FeatureBundle, config: dict[str, Any]) -> list[dict[str, Any]]:
    dim = bundle.source0_features.shape[1]
    initial = initialized_head(dim, bundle.record.seed + 12345)
    initial_state = {key: value.detach().cpu().double().clone() for key, value in initial.state_dict().items()}
    before = parameter_hash(bundle.model)
    rows = []
    for head_method in ("HEAD_ERM", "HEAD_IRMv1"):
        head = train_source_head(bundle, method=head_method, initial_state=initial_state, config=config)
        rows.append(_head_selection_metrics(bundle, head, head_method=head_method, oracle=False, source_only=True))
    oracle_head = train_oracle_clean_head(bundle, initial_state=initial_state, config=config)
    rows.append(_head_selection_metrics(bundle, oracle_head, head_method="ORACLE_CLEAN", oracle=True, source_only=False))
    after = parameter_hash(bundle.model)
    if before != after:
        raise ValueError("head-only selection training mutated frozen encoder/model parameters")
    for row in rows:
        row["initial_head_seed"] = bundle.record.seed + 12345
        row["frozen_encoder_parameter_hash_before"] = before
        row["frozen_encoder_parameter_hash_after"] = after
    return rows


def run_capability_experiments(bundles: list[FeatureBundle], *, config: dict[str, Any]) -> CapabilityRows:
    coverage: list[dict[str, Any]] = []
    separability: list[dict[str, Any]] = []
    selection: list[dict[str, Any]] = []
    for bundle in bundles:
        coverage.append(run_feature_coverage(bundle))
        separability.extend(run_feature_separability(bundle, RANKS))
        selection.extend(run_feature_selection(bundle, config))
    return CapabilityRows(coverage=coverage, separability=separability, selection=selection)
