from __future__ import annotations

import argparse
import json
import random
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, autograd
from torch.nn import functional as F

REPO_SRC = Path("/Users/sunlay/Documents/Codex/2026-09-13/zu/repo/src")
sys.path.insert(0, str(REPO_SRC))

from ood_repr_reg.task3_baseline_fidelity.official_irm import (
    OfficialEnvironment,
    OfficialIRMMLP,
    evaluate_official,
    make_official_environment,
    weight_norm_squared,
)
from source_only_response_optimizer import (
    clip_common_statistic_radial_update,
    clip_curvature_aligned_excess_transfer,
    cap_norm,
    make_correction,
    project,
    response_balanced_correction,
    response_radial_rate_balance_correction,
)


METHODS = ("erm", "irm_vanilla", "irm_centered", "vrex_centered", "coral_centered", "fishr_centered")
ARMS = METHODS + tuple(f"{method}_source_only_response" for method in METHODS if method != "erm") + tuple(f"{method}_rbi" for method in METHODS if method != "erm") + tuple(f"{method}_rrb" for method in METHODS if method != "erm") + tuple(f"{method}_rrn" for method in METHODS if method != "erm")


@dataclass(frozen=True)
class SourceEnvironment:
    """Training view containing only tensors available to source training."""

    images: Tensor
    labels: Tensor


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def read_mnist_source(root: Path, count: int = 50000) -> tuple[Tensor, Tensor, Path, Path]:
    raw = root / "MNIST" / "raw"
    image_path = raw / "train-images-idx3-ubyte"
    label_path = raw / "train-labels-idx1-ubyte"
    with image_path.open("rb") as stream:
        magic, image_count, rows, columns = struct.unpack(">IIII", stream.read(16))
    with label_path.open("rb") as stream:
        label_magic, label_count = struct.unpack(">II", stream.read(8))
    if (magic, rows, columns, label_magic) != (2051, 28, 28, 2049):
        raise ValueError("unexpected MNIST IDX headers")
    if image_count != label_count or count > image_count:
        raise ValueError("invalid MNIST source split")
    images = np.memmap(image_path, mode="r", dtype=np.uint8, offset=16, shape=(image_count, 28, 28))
    labels = np.memmap(label_path, mode="r", dtype=np.uint8, offset=8, shape=(label_count,))
    source_images = torch.from_numpy(np.array(images[:count], copy=True)).float().div_(255.0)
    source_digits = torch.from_numpy(np.array(labels[:count], copy=True)).long()
    return source_images, source_digits, image_path, label_path


def make_source_environment(
    images: Tensor,
    digits: Tensor,
    *,
    flip_probability: float,
    label_noise: float,
    role: str,
    generator: torch.Generator,
) -> SourceEnvironment:
    downsampled = images.reshape((-1, 28, 28))[:, ::2, ::2]
    labels = (digits < 5).float()
    labels = (labels - (torch.rand(len(labels), generator=generator) < label_noise).float()).abs()
    active_channel = (labels - (torch.rand(len(labels), generator=generator) < flip_probability).float()).abs()
    colored = torch.stack([downsampled, downsampled], dim=1)
    colored[torch.arange(len(colored)), (1 - active_channel).long(), :, :] = 0.0
    return SourceEnvironment(
        images=colored.float(), labels=labels[:, None].float(),
    )


def make_source(
    images: Tensor, digits: Tensor, seed: int, max_train_per_env: int
) -> tuple[tuple[SourceEnvironment, SourceEnvironment], Tensor]:
    order = torch.randperm(len(images), generator=torch.Generator().manual_seed(seed))
    generator = torch.Generator().manual_seed(seed + 314159)
    source_environments = (
        make_source_environment(
            images[order[::2]], digits[order[::2]], flip_probability=0.2,
            label_noise=0.25, role="train_env0", generator=generator,
        ),
        make_source_environment(
            images[order[1::2]], digits[order[1::2]], flip_probability=0.1,
            label_noise=0.25, role="train_env1", generator=generator,
        ),
    )
    environments = tuple(
        SourceEnvironment(
            images=env.images[:max_train_per_env] if max_train_per_env > 0 else env.images,
            labels=env.labels[:max_train_per_env] if max_train_per_env > 0 else env.labels,
        ) for env in source_environments
    )
    return environments, generator.get_state()  # type: ignore[return-value]


def flatten(values: tuple[Tensor | None, ...], parameters: tuple[Tensor, ...]) -> Tensor:
    return torch.cat([
        (value if value is not None else torch.zeros_like(parameter)).reshape(-1)
        for value, parameter in zip(values, parameters)
    ])


def unflatten(vector: Tensor, parameters: tuple[Tensor, ...]) -> list[Tensor]:
    result = []
    offset = 0
    for parameter in parameters:
        size = parameter.numel()
        result.append(vector[offset : offset + size].reshape_as(parameter))
        offset += size
    if offset != vector.numel():
        raise ValueError("vector size does not match parameter shapes")
    return result


def adam_delta(
    gradients: Tensor, parameters: tuple[Tensor, ...], optimizer: torch.optim.Optimizer, step: int
) -> Tensor:
    group = optimizer.param_groups[0]
    beta1, beta2 = group["betas"]
    pieces = []
    offset = 0
    for parameter in parameters:
        size = parameter.numel()
        gradient = gradients[offset : offset + size].reshape_as(parameter)
        state = optimizer.state.get(parameter, {})
        first = state.get("exp_avg", torch.zeros_like(parameter))
        second = state.get("exp_avg_sq", torch.zeros_like(parameter))
        first_next = beta1 * first + (1.0 - beta1) * gradient
        second_next = beta2 * second + (1.0 - beta2) * gradient.square()
        first_hat = first_next / (1.0 - beta1**step)
        second_hat = second_next / (1.0 - beta2**step)
        pieces.append((-group["lr"] * first_hat / (second_hat.sqrt() + group["eps"])).reshape(-1))
        offset += size
    return torch.cat(pieces).detach()


def source_statistics(
    method: str, model: OfficialIRMMLP, environments: tuple[SourceEnvironment, SourceEnvironment]
) -> tuple[Tensor, Tensor, list[Tensor]]:
    risks = []
    logits_list = []
    for environment in environments:
        logits = model(environment.images)
        logits_list.append(logits)
        risks.append(F.binary_cross_entropy_with_logits(logits, environment.labels))
    risk_vector = torch.stack(risks)
    if method == "erm":
        return risk_vector.mean(), risk_vector.sum() * 0.0, logits_list
    if method in {"irm_centered", "irm_vanilla"}:
        values = []
        for logits, environment in zip(logits_list, environments):
            scale = torch.ones((), requires_grad=True)
            scaled_risk = F.binary_cross_entropy_with_logits(logits * scale, environment.labels)
            values.append(autograd.grad(scaled_risk, scale, create_graph=True)[0])
        values_tensor = torch.stack(values)
        if method == "irm_vanilla":
            penalty = values_tensor.square().mean()
        else:
            penalty = (values_tensor - values_tensor.mean()).square().mean()
    elif method == "vrex_centered":
        penalty = (risk_vector - risk_vector.mean()).square().mean()
    elif method == "coral_centered":
        values = []
        for environment in environments:
            hidden = model._main[:4](environment.images.reshape(environment.images.shape[0], -1))
            centered = hidden - hidden.mean(dim=0, keepdim=True)
            values.append((centered.T @ centered / hidden.shape[0]).reshape(-1))
        values_tensor = torch.stack(values)
        penalty = (values_tensor - values_tensor.mean(dim=0, keepdim=True)).square().mean()
    elif method == "fishr_centered":
        values = []
        for logits, environment in zip(logits_list, environments):
            hidden = model._main[:4](environment.images.reshape(environment.images.shape[0], -1))
            residual = torch.sigmoid(logits) - environment.labels
            head_gradient = torch.cat((residual * hidden, residual), dim=1)
            values.append(head_gradient.var(dim=0, unbiased=False))
        values_tensor = torch.stack(values)
        penalty = (values_tensor - values_tensor.mean(dim=0, keepdim=True)).square().mean()
    else:
        raise ValueError(method)
    return risk_vector.mean(), penalty, logits_list


def statistic_vector(
    method: str, model: OfficialIRMMLP, environments: tuple[SourceEnvironment, SourceEnvironment], *, centered: bool = True
) -> Tensor:
    """Return the centered source statistic vector used by a method.

    This is an observation-space quantity. It contains no semantic feature
    labels and is used only to estimate relative radial response speed.
    """
    risks = []
    logits_list = []
    for environment in environments:
        logits = model(environment.images)
        logits_list.append(logits)
        risks.append(F.binary_cross_entropy_with_logits(logits, environment.labels))
    if method == "vrex_centered":
        values = torch.stack(risks)
    elif method in {"irm_centered", "irm_vanilla"}:
        values = []
        for logits, environment in zip(logits_list, environments):
            scale = torch.ones((), requires_grad=True)
            risk = F.binary_cross_entropy_with_logits(logits * scale, environment.labels)
            values.append(autograd.grad(risk, scale, create_graph=True)[0])
        values = torch.stack(values)
    elif method == "coral_centered":
        values = []
        for environment in environments:
            hidden = model._main[:4](environment.images.reshape(environment.images.shape[0], -1))
            centered = hidden - hidden.mean(dim=0, keepdim=True)
            values.append((centered.T @ centered / hidden.shape[0]).reshape(-1))
        values = torch.stack(values)
    elif method == "fishr_centered":
        values = []
        for logits, environment in zip(logits_list, environments):
            hidden = model._main[:4](environment.images.reshape(environment.images.shape[0], -1))
            residual = torch.sigmoid(logits) - environment.labels
            head_gradient = torch.cat((residual * hidden, residual), dim=1)
            values.append(head_gradient.var(dim=0, unbiased=False))
        values = torch.stack(values)
    else:
        values = torch.stack(risks)
    return values - values.mean(dim=0, keepdim=True) if centered else values


def statistic_radial_rate(
    statistic: Tensor, update: Tensor, parameters: tuple[Tensor, ...], *, retain_graph: bool
) -> float:
    energy = 0.5 * statistic.square().sum()
    statistic_gradient = gradient_at(energy, parameters, retain_graph=retain_graph, create_graph=False)
    denominator = statistic.square().sum().detach().clamp_min(1e-12)
    return float((statistic_gradient.detach() * update.detach()).sum() / denominator)


def normalize_feature_radial_update(
    weight: Tensor, update: Tensor, *, eps: float = 1e-8
) -> tuple[Tensor, Tensor, float, float, float]:
    row_norm_sq = weight.detach().square().sum(dim=1).clamp_min(eps)
    radial_rates = (weight.detach() * update.detach()).sum(dim=1) / row_norm_sq
    target_magnitude = torch.median(radial_rates.abs()).detach()
    target_rates = torch.sign(radial_rates) * target_magnitude
    normalized_update = update.detach().clone()
    radial_component = radial_rates[:, None] * weight.detach()
    normalized_update = normalized_update - radial_component + target_rates[:, None] * weight.detach()
    return (
        normalized_update,
        normalized_update - update.detach(),
        float(radial_rates.mean().detach()),
        float(radial_rates.std(unbiased=False).detach()),
        float(target_magnitude),
    )


def cap_feature_radial_excess(
    weight: Tensor,
    actual_update: Tensor,
    reference_update: Tensor,
    *,
    relative_margin: float = 0.0,
    eps: float = 1e-8,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    row_norm_sq = weight.detach().square().sum(dim=1).clamp_min(eps)
    actual_rates = (weight.detach() * actual_update.detach()).sum(dim=1) / row_norm_sq
    reference_rates = (weight.detach() * reference_update.detach()).sum(dim=1) / row_norm_sq
    reference_speed = torch.quantile(reference_rates.abs(), 0.5).detach()
    allowed = torch.full_like(actual_rates, reference_speed * (1.0 + relative_margin))
    target_rates = actual_rates.sign() * torch.minimum(actual_rates.abs(), allowed)
    correction = (target_rates - actual_rates)[:, None] * weight.detach()
    return correction.detach(), actual_rates.detach(), reference_rates.detach(), target_rates.detach()


def clip_cross_curvature_update(
    cross_update: Tensor, risk_update_norm: Tensor, budget_ratio: float, eps: float = 1e-8
) -> tuple[Tensor, Tensor]:
    budget = budget_ratio * risk_update_norm.detach().clamp_min(eps)
    cross_norm = cross_update.detach().norm()
    if float(cross_norm) <= float(budget):
        return cross_update.detach(), torch.zeros_like(cross_update)
    clipped = cross_update.detach() * (budget / cross_norm.clamp_min(eps))
    return clipped, clipped - cross_update.detach()


def gradient_at(
    scalar: Tensor, parameters: tuple[Tensor, ...], *, retain_graph: bool, create_graph: bool
) -> Tensor:
    return flatten(autograd.grad(scalar, parameters, retain_graph=retain_graph, create_graph=create_graph, allow_unused=True), parameters)


def contrast_basis(
    model: OfficialIRMMLP, environments: tuple[SourceEnvironment, SourceEnvironment], parameters: tuple[Tensor, ...]
) -> Tensor:
    gradients = []
    for environment in environments:
        risk = F.binary_cross_entropy_with_logits(model(environment.images), environment.labels)
        gradients.append(gradient_at(risk, parameters, retain_graph=True, create_graph=False))
    contrast = torch.stack(gradients) - torch.stack(gradients).mean(dim=0, keepdim=True)
    if contrast.norm() < 1e-12:
        return torch.zeros((contrast.shape[1], 0), dtype=contrast.dtype)
    basis, _ = torch.linalg.qr(contrast.T, mode="reduced")
    return basis.detach()


def hvp(scalar: Tensor, parameters: tuple[Tensor, ...], direction: Tensor) -> Tensor:
    gradient = gradient_at(scalar, parameters, retain_graph=True, create_graph=True)
    return gradient_at((gradient * direction).sum(), parameters, retain_graph=True, create_graph=False)


@torch.no_grad()
def evaluate_source(model: OfficialIRMMLP, environment: SourceEnvironment) -> dict[str, float]:
    logits = model(environment.images)
    labels = environment.labels
    predictions = (logits > 0.0).float()
    return {
        "nll": float(F.binary_cross_entropy_with_logits(logits, labels)),
        "accuracy": float((predictions == labels).float().mean()),
    }


def stable_ratio(numerator: Tensor, denominator: Tensor) -> float:
    numerator_value = float(numerator.detach())
    denominator_value = float(denominator.detach())
    if denominator_value < 1e-12:
        return float("nan") if numerator_value > 1e-12 else 0.0
    return numerator_value / denominator_value


def save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8")
    temporary.replace(path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    source_images, source_digits, _, _ = read_mnist_source(args.data_root)
    traces: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    source_eval: list[dict[str, Any]] = []
    snapshot_dir = args.snapshot_dir
    checkpoint_dir = args.checkpoint_dir
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    payload: dict[str, Any] = {
        "experiment": "official_cmnist_source_only_response_validation",
        "status": "source_training",
        "feature_leakage_policy": "The optimizer receives only source images, source labels, source risks, source penalties, source gradients, and source HVPs. No target rows, target metrics, color tensors, semantic decomposition, or oracle probe is available in this runner.",
        "measurement_policy": "This runner stores raw first-layer snapshots only. Invariant/shortcut decomposition is performed by a separate posthoc analyzer after training.",
        "target_policy": "Target and colorless silver evaluation are performed by a separate evaluator after all checkpoints are frozen.",
        "theory_validation": {
            "direct_residual": "The source-contrast projection of the penalty gradient and of its virtual optimizer displacement; it is allowed to be nonzero.",
            "cross_response": "The source-contrast projection of the source-risk Hessian acting on the penalty-only virtual displacement.",
            "dominance_ratio": "cross_curvature_response_gradient_norm / (direct_residual_gradient_norm + 1e-12); both terms are in gradient units and are source-observable near-blind diagnostics, not oracle shortcut labels.",
            "causal_control": "Every arm logs the same source-only response quantities. Only response arms apply the correction.",
        },
        "algorithm": {
            "name": "Source-only Response Radial-Rate Normalization",
            "rule": "RRN clips source-observable curvature-aligned transfer and caps excess first-layer radial speed relative to an unscaled source-risk reference; optional penalty-scale adaptation is disabled by default.",
            "transfer_budget": "The curvature-aligned transfer cap is method-specific because the source statistics have different gradient scales: V-REx uses a stricter source-only cap than IRMv1.",
            "scope": "Both algorithms are source-only optimization probes; neither identifies a spurious feature or uses target information.",
        },
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "warmup_protocol": {
            "steps": args.erm_warmup_steps,
            "definition": "All arms use the same ERM objective (source risk plus L2) for zero-based steps below erm_warmup_steps. Method penalties and response corrections start at zero-based step erm_warmup_steps.",
        },
        "training_trace": traces, "summaries": summaries,
        "source_evaluation": source_eval,
    }

    for seed in args.seeds:
        seed_everything(seed)
        environments, _ = make_source(source_images, source_digits, seed, args.max_train_per_env)
        torch.manual_seed(seed)
        initial = OfficialIRMMLP(args.hidden_dim)
        initial_state = {key: value.detach().clone() for key, value in initial.state_dict().items()}
        warmup_reference: Tensor | None = None
        warmup_reference_arm: str | None = None
        for arm in args.arms:
            if arm.endswith("_source_only_response"):
                method = arm.removesuffix("_source_only_response")
            elif arm.endswith("_rbi"):
                method = arm.removesuffix("_rbi")
            elif arm.endswith("_rrb"):
                method = arm.removesuffix("_rrb")
            elif arm.endswith("_rrn"):
                method = arm.removesuffix("_rrn")
            else:
                method = arm
            response_arm = arm.endswith("_source_only_response") or arm.endswith("_rbi")
            rbi_arm = arm.endswith("_rbi")
            rrb_arm = arm.endswith("_rrb")
            rrn_arm = arm.endswith("_rrn")
            adaptive_penalty_scale = 1.0
            adaptive_rate_ema = None
            rrn_transfer_budget_ratio = (
                args.rrn_vrex_transfer_ratio
                if method == "vrex_centered"
                else args.rrn_irm_transfer_ratio
                if method in {"irm_vanilla", "irm_centered"}
                else args.radial_budget_ratio
            )
            rrn_correction_budget_ratio = (
                args.rrn_vrex_correction_ratio
                if method == "vrex_centered"
                else args.rrn_irm_correction_ratio
                if method in {"irm_vanilla", "irm_centered"}
                else args.radial_budget_ratio
            )
            rrn_correction_gain = (
                args.rrn_vrex_correction_gain
                if method == "vrex_centered"
                else args.rrn_irm_correction_gain
                if method in {"irm_vanilla", "irm_centered"}
                else 1.0
            )
            model = OfficialIRMMLP(args.hidden_dim)
            model.load_state_dict(initial_state)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
            parameters = tuple(model.parameters())
            response_updates = 0
            correction_norms = []
            shield_norms = []
            last_diagnostic_ratio = None
            last_diagnostic_step = None
            warmup_max_abs_diff = None
            for step in range(args.steps):
                is_vrex = method == "vrex_centered"
                if is_vrex and not args.common_penalty_schedule and step == args.vrex_penalty_anneal_iters:
                    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
                in_erm_warmup = step < args.erm_warmup_steps and (
                    not is_vrex or args.common_penalty_schedule
                )
                active_method = "erm" if in_erm_warmup else method
                risk, penalty, _ = source_statistics(active_method, model, environments)
                erm_objective = risk + args.l2 * weight_norm_squared(model)
                if in_erm_warmup or method == "erm":
                    objective = erm_objective
                    applied_penalty_weight = 0.0
                    penalty_coefficient = 0.0
                    anneal_scale = 1.0
                    active_rrn_scale = 1.0
                elif is_vrex and not args.common_penalty_schedule:
                    applied_penalty_weight = (
                        1.0 if step < args.vrex_penalty_anneal_iters else args.vrex_lambda
                    )
                    penalty_coefficient = applied_penalty_weight
                    anneal_scale = 1.0
                    active_rrn_scale = adaptive_penalty_scale if rrn_arm and args.rrn_adaptive_scale else 1.0
                    objective = erm_objective + penalty_coefficient * active_rrn_scale * penalty
                elif args.common_penalty_schedule:
                    applied_penalty_weight = args.penalty_weight
                    penalty_coefficient = args.penalty_weight
                    anneal_scale = 1.0
                    active_rrn_scale = adaptive_penalty_scale if rrn_arm and args.rrn_adaptive_scale else 1.0
                    objective = erm_objective + penalty_coefficient * active_rrn_scale * penalty
                else:
                    applied_penalty_weight = args.penalty_weight
                    penalty_coefficient = args.penalty_weight
                    anneal_scale = args.penalty_weight
                    active_rrn_scale = adaptive_penalty_scale if rrn_arm and args.rrn_adaptive_scale else 1.0
                    objective = (
                        erm_objective
                        + penalty_coefficient * active_rrn_scale * penalty
                    ) / anneal_scale
                diagnostic_step = (
                    not in_erm_warmup
                    and method != "erm"
                    and (step - args.erm_warmup_steps) % args.response_interval == 0
                )
                rrn_feature_step = rrn_arm and not in_erm_warmup and method != "erm"
                gradient_step = diagnostic_step or rrn_feature_step
                basis = contrast_basis(model, environments, parameters) if diagnostic_step else torch.zeros((sum(p.numel() for p in parameters), 0))
                risk_gradient = None
                penalty_gradient = None
                penalty_delta = torch.zeros(sum(p.numel() for p in parameters))
                risk_delta = torch.zeros_like(penalty_delta)
                h_cross = torch.zeros_like(penalty_delta)
                response_basis = torch.zeros((sum(p.numel() for p in parameters), 0))
                direct_gradient = torch.zeros_like(penalty_delta)
                direct_update = torch.zeros_like(penalty_delta)
                cross_update = torch.zeros_like(penalty_delta)
                rrn_cross_transfer = torch.zeros_like(penalty_delta)
                radial_risk_rates = torch.zeros(0)
                radial_penalty_rates = torch.zeros(0)
                radial_coordinate_norm = torch.zeros(())
                radial_risk_response_norm = torch.zeros(())
                radial_penalty_response_norm = torch.zeros(())
                statistic_risk_radial_rate = None
                statistic_penalty_radial_rate = None
                common_gradient = None
                risk_only_optimizer_update = None
                source_risk_reference_update = None
                penalty_reference_update = None
                if gradient_step:
                    risk_gradient = gradient_at(risk + args.l2 * weight_norm_squared(model), parameters, retain_graph=True, create_graph=False)
                    penalty_gradient = gradient_at(penalty, parameters, retain_graph=True, create_graph=False)
                    risk_component_gradient = risk_gradient / anneal_scale
                    penalty_component_gradient = penalty_gradient * (penalty_coefficient * active_rrn_scale / anneal_scale)
                    penalty_delta = -args.learning_rate * penalty_component_gradient
                    risk_delta = -args.learning_rate * risk_component_gradient
                    risk_only_optimizer_update = adam_delta(
                        risk_component_gradient, parameters, optimizer, step + 1
                    )
                    source_risk_reference_update = adam_delta(
                        risk_gradient, parameters, optimizer, step + 1
                    )
                    penalty_reference_update = adam_delta(
                        penalty_gradient * active_rrn_scale, parameters, optimizer, step + 1
                    )
                    if diagnostic_step:
                        observed_statistic = statistic_vector(method, model, environments)
                        if rrn_arm and method in {"irm_vanilla", "vrex_centered"}:
                            common_gradient = gradient_at(
                                statistic_vector(method, model, environments, centered=False).mean(),
                                parameters,
                                retain_graph=True,
                                create_graph=False,
                            )
                        statistic_risk_radial_rate = statistic_radial_rate(
                            observed_statistic, source_risk_reference_update, parameters, retain_graph=True
                        )
                        statistic_penalty_radial_rate = statistic_radial_rate(
                            observed_statistic, penalty_reference_update, parameters, retain_graph=True
                        )
                        direct_gradient = project(penalty_component_gradient, basis)
                        direct_update = project(penalty_delta, basis)
                        h_action = hvp(risk, parameters, penalty_component_gradient)
                        h_cross = project(h_action, basis)
                        if rrn_arm:
                            rrn_cross_transfer = -args.learning_rate * h_cross
                        if rbi_arm and basis.shape[1]:
                            response_basis = torch.stack(
                                [hvp(risk, parameters, basis[:, column]) for column in range(basis.shape[1])],
                                dim=1,
                            )
                        if rrb_arm and basis.shape[1]:
                            h_risk_action = hvp(risk, parameters, risk_gradient)
                            response_coordinate = basis.T @ penalty_gradient
                            risk_response = basis.T @ h_risk_action
                            penalty_response = basis.T @ h_action
                            radial_coordinate_norm = response_coordinate.norm()
                            radial_risk_response_norm = risk_response.norm()
                            radial_penalty_response_norm = penalty_response.norm()
                        cross_update = args.learning_rate * h_cross
                        if response_arm:
                            response_updates += 1
                before_optimizer = torch.cat([parameter.detach().reshape(-1) for parameter in parameters])
                optimizer.zero_grad(set_to_none=True)
                objective.backward()
                optimizer.step()
                after_optimizer = torch.cat([parameter.detach().reshape(-1) for parameter in parameters])
                correction = torch.zeros_like(penalty_delta)
                shield = torch.zeros_like(penalty_delta)
                feature_radial_correction = torch.zeros_like(penalty_delta)
                cross_transfer_correction = torch.zeros_like(penalty_delta)
                cross_transfer_norm = None
                clipped_cross_transfer_norm = None
                curvature_aligned_response = None
                common_statistic_residual_response = None
                common_statistic_clipped_response = None
                feature_radial_rate_mean = None
                feature_radial_rate_std = None
                feature_radial_rate_target = None
                feature_radial_rate_post_mean = None
                feature_radial_rate_post_std = None
                feature_radial_rate_post_max_excess = None
                if rrn_arm and diagnostic_step and penalty_gradient is not None and risk_gradient is not None:
                    cross_transfer_correction, cross_transfer, clipped_cross_transfer, curvature_aligned_response = (
                        clip_curvature_aligned_excess_transfer(
                            actual_update=after_optimizer - before_optimizer,
                            risk_update=risk_only_optimizer_update,
                            curvature_direction=-h_cross,
                            basis=basis,
                            budget_ratio=rrn_transfer_budget_ratio,
                            budget_update=after_optimizer - before_optimizer,
                            correction_budget_ratio=rrn_correction_budget_ratio,
                            correction_gain=rrn_correction_gain,
                            eps=args.radial_rate_eps,
                        )
                    )
                    cross_transfer_norm = float(cross_transfer.norm())
                    clipped_cross_transfer_norm = float(clipped_cross_transfer.norm())
                    for parameter, delta in zip(parameters, unflatten(cross_transfer_correction, parameters)):
                        parameter.data.add_(delta)
                    if method in {"irm_vanilla", "vrex_centered"} and common_gradient is not None and risk_only_optimizer_update is not None:
                        after_cross = torch.cat([parameter.detach().reshape(-1) for parameter in parameters])
                        common_correction, common_statistic_residual_response, common_statistic_clipped_response = (
                            clip_common_statistic_radial_update(
                                actual_update=after_cross - before_optimizer,
                                risk_update=risk_only_optimizer_update,
                                statistic_gradient=common_gradient,
                                budget_ratio=rrn_transfer_budget_ratio,
                                eps=args.radial_rate_eps,
                            )
                        )
                        for parameter, delta in zip(parameters, unflatten(common_correction, parameters)):
                            parameter.data.add_(delta)
                if rrn_feature_step and source_risk_reference_update is not None:
                    after_curvature = torch.cat([parameter.detach().reshape(-1) for parameter in parameters])
                    first_layer_size = parameters[0].numel()
                    feature_correction, feature_rates, reference_rates, feature_targets = cap_feature_radial_excess(
                        before_optimizer[:first_layer_size].reshape_as(parameters[0]),
                        (after_curvature[:first_layer_size] - before_optimizer[:first_layer_size]).reshape_as(parameters[0]),
                        source_risk_reference_update[:first_layer_size].reshape_as(parameters[0]),
                        relative_margin=args.rrn_feature_rate_margin,
                    )
                    feature_correction = cap_norm(
                        feature_correction,
                        rrn_correction_budget_ratio * (after_curvature - before_optimizer).norm(),
                    )
                    feature_radial_correction[:first_layer_size] = feature_correction.reshape(-1)
                    feature_radial_rate_mean = float(feature_rates.mean())
                    feature_radial_rate_std = float(feature_rates.std(unbiased=False))
                    feature_radial_rate_target = float(feature_targets.abs().mean())
                    for parameter, delta in zip(parameters, unflatten(feature_radial_correction, parameters)):
                        parameter.data.add_(delta)
                if response_arm and diagnostic_step and basis.shape[1]:
                    if rrb_arm:
                        correction, radial_risk_rates, radial_penalty_rates = response_radial_rate_balance_correction(
                            basis=basis,
                            response_coordinate=response_coordinate,
                            risk_response=risk_response,
                            penalty_response=penalty_response,
                            learning_rate=args.learning_rate,
                            radial_margin=args.radial_margin,
                            coordinate_eps=args.radial_coordinate_eps,
                            budget_ratio=args.radial_budget_ratio,
                            risk_update_norm=risk_delta.norm(),
                        )
                    elif rbi_arm:
                        correction = response_balanced_correction(
                            basis=basis,
                            cross_gradient=h_cross,
                            response_basis=response_basis,
                            risk_gradient=risk_gradient,
                            penalty_gradient=penalty_gradient,
                            ridge=args.rbi_ridge,
                            budget_ratio=args.rbi_budget_ratio,
                            descent_tolerance=args.rbi_descent_tolerance,
                        )
                    else:
                        response = make_correction(
                            direct_gradient=direct_gradient,
                            penalty_update=direct_update,
                            cross_gradient=h_cross,
                            learning_rate=args.learning_rate,
                            risk_update_norm=risk_delta.norm(),
                            curvature_kappa=args.curvature_kappa,
                            direct_budget_ratio=args.direct_budget_ratio,
                            max_correction_ratio=args.max_correction_ratio,
                            max_shield_ratio=args.max_shield_ratio,
                            response_shield=args.response_shield,
                        )
                        correction = response.correction
                        shield = response.shield
                    for parameter, delta in zip(parameters, unflatten(correction, parameters)):
                        parameter.data.add_(delta)
                    for parameter, delta in zip(parameters, unflatten(shield, parameters)):
                        parameter.data.add_(delta)
                    after = torch.cat([parameter.detach().reshape(-1) for parameter in parameters])
                    applied_response = after - after_optimizer
                    correction_norms.append(float(correction.norm()))
                    shield_norms.append(float(shield.norm()))
                if rrn_arm and diagnostic_step and statistic_risk_radial_rate is not None and statistic_penalty_radial_rate is not None:
                    raw_rate_ratio = abs(statistic_risk_radial_rate) / max(
                        abs(statistic_penalty_radial_rate), args.radial_rate_eps
                    )
                    if adaptive_rate_ema is None:
                        adaptive_rate_ema = raw_rate_ratio
                    else:
                        adaptive_rate_ema = (
                            args.radial_rate_ema * adaptive_rate_ema
                            + (1.0 - args.radial_rate_ema) * raw_rate_ratio
                        )
                    adaptive_penalty_scale = float(np.clip(
                        adaptive_rate_ema ** args.radial_rate_gain,
                        args.radial_rate_min_scale,
                        args.radial_rate_max_scale,
                    ))
                else:
                    applied_response = torch.zeros_like(penalty_delta)
                after_step = torch.cat([parameter.detach().reshape(-1) for parameter in parameters])
                actual_update = after_step - before_optimizer
                if rrn_feature_step and source_risk_reference_update is not None:
                    first_layer_size = parameters[0].numel()
                    first_layer_weight_before = before_optimizer[:first_layer_size].reshape_as(parameters[0])
                    first_layer_update_final = actual_update[:first_layer_size].reshape_as(parameters[0])
                    first_layer_reference_update = source_risk_reference_update[:first_layer_size].reshape_as(parameters[0])
                    row_norm_sq = first_layer_weight_before.detach().square().sum(dim=1).clamp_min(args.radial_rate_eps)
                    final_rates = (
                        first_layer_weight_before.detach() * first_layer_update_final.detach()
                    ).sum(dim=1) / row_norm_sq
                    reference_rates_final = (
                        first_layer_weight_before.detach() * first_layer_reference_update.detach()
                    ).sum(dim=1) / row_norm_sq
                    reference_speed_final = torch.quantile(reference_rates_final.abs(), 0.5)
                    allowed_speed_final = reference_speed_final * (1.0 + args.rrn_feature_rate_margin)
                    feature_radial_rate_post_mean = float(final_rates.mean())
                    feature_radial_rate_post_std = float(final_rates.std(unbiased=False))
                    feature_radial_rate_post_max_excess = float(
                        (final_rates.abs() - allowed_speed_final).clamp_min(0.0).max()
                    )
                if step == args.erm_warmup_steps - 1:
                    if warmup_reference is None:
                        warmup_reference = after_step.detach().cpu().clone()
                        warmup_reference_arm = arm
                    else:
                        warmup_max_abs_diff = float((after_step.detach().cpu() - warmup_reference).abs().max())
                row = {
                    "seed": seed, "arm": arm, "method": method, "step": step,
                    "training_phase": "erm_warmup" if in_erm_warmup else ("erm" if method == "erm" else "regularized"),
                    "applied_penalty_weight": applied_penalty_weight,
                    "penalty_coefficient": penalty_coefficient,
                    "anneal_scale": anneal_scale,
                    "active_rrn_scale": active_rrn_scale,
                    "warmup_reference_arm": warmup_reference_arm if step == args.erm_warmup_steps - 1 else None,
                    "warmup_model_max_abs_diff_from_reference": warmup_max_abs_diff if step == args.erm_warmup_steps - 1 else None,
                    "source_risk": float(risk.detach()), "penalty": float(penalty.detach()),
                    "diagnostic_step": diagnostic_step,
                    "direct_residual_gradient_norm": float(direct_gradient.norm()) if diagnostic_step else None,
                    "direct_residual_update_norm": float(direct_update.norm()) if diagnostic_step else None,
                    "cross_curvature_response_gradient_norm": float(h_cross.norm()) if diagnostic_step else None,
                    "cross_curvature_update_response_norm": float(cross_update.norm()) if diagnostic_step else None,
                    "cross_over_direct_gradient_ratio": stable_ratio(h_cross.norm(), direct_gradient.norm()) if diagnostic_step else None,
                    "cross_over_direct_update_ratio": stable_ratio(cross_update.norm(), direct_update.norm()) if diagnostic_step else None,
                    "cross_over_direct_ratio": stable_ratio(h_cross.norm(), direct_gradient.norm()) if diagnostic_step else None,
                    "penalty_only_virtual_update_norm": float(penalty_delta.norm()) if diagnostic_step else None,
                    "risk_only_virtual_update_norm": float(risk_delta.norm()) if diagnostic_step else None,
                    "optimizer_risk_only_update_norm": float(risk_only_optimizer_update.norm()) if diagnostic_step else None,
                    "source_risk_reference_update_norm": float(source_risk_reference_update.norm()) if diagnostic_step else None,
                    "penalty_reference_update_norm": float(penalty_reference_update.norm()) if diagnostic_step else None,
                    "response_correction_norm": float(correction.norm()) if diagnostic_step else None,
                    "feature_radial_correction_norm": float(feature_radial_correction.norm()) if rrn_arm and not in_erm_warmup else None,
                    "cross_transfer_correction_norm": float(cross_transfer_correction.norm()) if rrn_arm and diagnostic_step else None,
                    "cross_transfer_norm": cross_transfer_norm,
                    "clipped_cross_transfer_norm": clipped_cross_transfer_norm,
                    "curvature_aligned_response": float(curvature_aligned_response) if curvature_aligned_response is not None else None,
                    "common_statistic_residual_response": float(common_statistic_residual_response) if common_statistic_residual_response is not None else None,
                    "common_statistic_clipped_response": float(common_statistic_clipped_response) if common_statistic_clipped_response is not None else None,
                    "rrn_budget_ratio": args.radial_budget_ratio if rrn_arm else None,
                    "rrn_transfer_budget_ratio": rrn_transfer_budget_ratio if rrn_arm else None,
                    "rrn_correction_budget_ratio": rrn_correction_budget_ratio if rrn_arm else None,
                    "rrn_correction_gain": rrn_correction_gain if rrn_arm else None,
                    "feature_radial_rate_mean": feature_radial_rate_mean,
                    "feature_radial_rate_std": feature_radial_rate_std,
                    "feature_radial_rate_target": feature_radial_rate_target,
                    "feature_radial_rate_post_mean": feature_radial_rate_post_mean,
                    "feature_radial_rate_post_std": feature_radial_rate_post_std,
                    "feature_radial_rate_post_max_excess": feature_radial_rate_post_max_excess,
                    "response_shield_norm": float(shield.norm()) if diagnostic_step else None,
                    "radial_risk_rate_norm": float(radial_risk_rates.norm()) if diagnostic_step and rrb_arm else None,
                    "radial_penalty_rate_norm": float(radial_penalty_rates.norm()) if diagnostic_step and rrb_arm else None,
                    "radial_coordinate_norm": float(radial_coordinate_norm) if diagnostic_step and rrb_arm else None,
                    "radial_risk_response_norm": float(radial_risk_response_norm) if diagnostic_step and rrb_arm else None,
                    "radial_penalty_response_norm": float(radial_penalty_response_norm) if diagnostic_step and rrb_arm else None,
                    "observed_statistic_risk_radial_rate": statistic_risk_radial_rate,
                    "observed_statistic_penalty_radial_rate": statistic_penalty_radial_rate,
                    "observed_statistic_radial_gap": (
                        statistic_penalty_radial_rate - statistic_risk_radial_rate
                        if statistic_risk_radial_rate is not None and statistic_penalty_radial_rate is not None else None
                    ),
                    "adaptive_penalty_scale": adaptive_penalty_scale if rrn_arm else None,
                    "observed_statistic_rate_ratio": (
                        abs(statistic_risk_radial_rate) / max(abs(statistic_penalty_radial_rate), args.radial_rate_eps)
                        if rrn_arm and statistic_risk_radial_rate is not None and statistic_penalty_radial_rate is not None else None
                    ),
                    "actual_update_norm": float(actual_update.norm()),
                    "actual_response_update_norm": float(applied_response.norm()),
                    "actual_parameter_norm": float(torch.cat([p.detach().reshape(-1) for p in parameters]).norm()),
                }
                traces.append(row)
                if diagnostic_step:
                    last_diagnostic_ratio = row["cross_over_direct_ratio"]
                    last_diagnostic_step = step
                should_snapshot = (
                    step == args.erm_warmup_steps - 1
                    or (
                        step >= args.erm_warmup_steps
                        and (step - args.erm_warmup_steps) % args.trajectory_snapshot_interval == 0
                    )
                )
                if should_snapshot:
                    snapshot_path = snapshot_dir / f"seed{seed}__arm-{arm}__step{step:05d}.npz"
                    np.savez_compressed(
                        snapshot_path,
                        seed=np.int64(seed),
                        step=np.int64(step),
                        update_count=np.int64(step + 1),
                        training_phase=np.array("erm_warmup_end" if step < args.erm_warmup_steps else ("erm" if method == "erm" else "regularized")),
                        arm=np.array(arm),
                        first_layer_weight=model._main[0].weight.detach().cpu().numpy().astype(np.float32),
                        first_layer_bias=model._main[0].bias.detach().cpu().numpy().astype(np.float32),
                    )
                    row["weight_snapshot"] = str(snapshot_path)
                    if args.full_checkpoint_interval > 0 and (
                        step == args.erm_warmup_steps - 1
                        or step % args.full_checkpoint_interval == 0
                        or step + 1 == args.steps
                    ):
                        trajectory_checkpoint_dir = checkpoint_dir / "trajectory"
                        trajectory_checkpoint_dir.mkdir(parents=True, exist_ok=True)
                        trajectory_checkpoint_path = trajectory_checkpoint_dir / f"seed{seed}__arm-{arm}__step{step:05d}.pt"
                        torch.save(
                            {
                                "model_state": model.state_dict(),
                                "optimizer_state": optimizer.state_dict() if args.save_optimizer_state else None,
                                "seed": seed,
                                "arm": arm,
                                "step": step,
                                "hidden_dim": args.hidden_dim,
                            },
                            trajectory_checkpoint_path,
                        )
                        row["trajectory_checkpoint"] = str(trajectory_checkpoint_path)
                if (step + 1) % args.progress_every == 0 or step + 1 == args.steps:
                    if last_diagnostic_ratio is None:
                        response_text = "cross/direct=unmeasured"
                    else:
                        response_text = f"cross/direct@step{last_diagnostic_step}={last_diagnostic_ratio:.3g}"
                    print(f"seed={seed} arm={arm} step={step + 1}/{args.steps} risk={row['source_risk']:.4f} {response_text}", flush=True)
                if (step + 1) % args.save_every == 0 or step + 1 == args.steps:
                    payload["training_trace"] = traces; payload["summaries"] = summaries; payload["source_evaluation"] = source_eval; save(args.output, payload)
            checkpoint_path = checkpoint_dir / f"seed{seed}__arm-{arm}.pt"
            torch.save(
                {"model_state": model.state_dict(), "seed": seed, "arm": arm, "hidden_dim": args.hidden_dim},
                checkpoint_path,
            )
            evaluations = [evaluate_source(model, env) for env in environments]
            source_eval.append({"seed": seed, "arm": arm, "source_accuracy": float(np.mean([x["accuracy"] for x in evaluations])), "source_nll": float(np.mean([x["nll"] for x in evaluations]))})
            summaries.append({"seed": seed, "arm": arm, "checkpoint": str(checkpoint_path), "response_updates": response_updates, "mean_correction_norm": float(np.mean(correction_norms)) if correction_norms else 0.0, "mean_shield_norm": float(np.mean(shield_norms)) if shield_norms else 0.0})
    payload["status"] = "source_training_complete"; payload["elapsed_seconds"] = time.time() - started
    save(args.output, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--steps", type=int, default=501)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--l2", type=float, default=1e-3)
    parser.add_argument("--erm-warmup-steps", "--penalty-anneal-iters", dest="erm_warmup_steps", type=int, default=100)
    parser.add_argument("--penalty-weight", type=float, default=10000.0)
    parser.add_argument("--vrex-lambda", type=float, default=10.0)
    parser.add_argument("--vrex-penalty-anneal-iters", type=int, default=500)
    parser.add_argument("--common-penalty-schedule", action="store_true")
    parser.add_argument("--response-interval", type=int, default=10)
    parser.add_argument("--trajectory-snapshot-interval", type=int, default=1)
    parser.add_argument("--curvature-kappa", type=float, default=1.0)
    parser.add_argument("--response-shield", type=float, default=1.0)
    parser.add_argument("--direct-budget-ratio", type=float, default=0.25)
    parser.add_argument("--max-correction-ratio", type=float, default=0.25)
    parser.add_argument("--max-shield-ratio", type=float, default=0.25)
    parser.add_argument("--rbi-ridge", type=float, default=1e-4)
    parser.add_argument("--rbi-budget-ratio", type=float, default=1.0)
    parser.add_argument("--rbi-descent-tolerance", type=float, default=0.05)
    parser.add_argument("--radial-margin", type=float, default=0.05)
    parser.add_argument("--radial-coordinate-eps", type=float, default=1e-8)
    parser.add_argument("--radial-budget-ratio", type=float, default=1.0)
    parser.add_argument("--radial-rate-ema", type=float, default=0.9)
    parser.add_argument("--radial-rate-gain", type=float, default=1.0)
    parser.add_argument("--radial-rate-min-scale", type=float, default=0.25)
    parser.add_argument("--radial-rate-max-scale", type=float, default=4.0)
    parser.add_argument("--radial-rate-eps", type=float, default=1e-8)
    parser.add_argument("--rrn-vrex-transfer-ratio", type=float, default=0.10)
    parser.add_argument("--rrn-irm-transfer-ratio", type=float, default=0.25)
    parser.add_argument("--rrn-vrex-correction-ratio", type=float, default=1.50)
    parser.add_argument("--rrn-irm-correction-ratio", type=float, default=0.50)
    parser.add_argument("--rrn-vrex-correction-gain", type=float, default=2.0)
    parser.add_argument("--rrn-irm-correction-gain", type=float, default=1.0)
    parser.add_argument("--rrn-feature-rate-margin", type=float, default=0.0)
    parser.add_argument("--rrn-adaptive-scale", action="store_true")
    parser.add_argument("--max-train-per-env", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--full-checkpoint-interval", type=int, default=0)
    parser.add_argument("--save-optimizer-state", action="store_true")
    args = parser.parse_args()
    result = run(args)
    print(f"wrote {args.output}; status={result['status']}; elapsed={result['elapsed_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
