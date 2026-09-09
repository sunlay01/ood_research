"""Official-protocol Colored MNIST baseline for IRMv1 recovery audits.

This module mirrors the experimental protocol from Facebook Research's
InvariantRiskMinimization Colored MNIST script without importing its source
file wholesale.  The reference is used only as a baseline-recovery gate for
Task 3; the Task 3 candidate methods keep their own isolated trainer.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn, optim, autograd
from torch.nn import functional as F

from ..cmnist_feature_probe import load_mnist_tensors, seed_everything
from .curvature import inverse_via_eigh, matched_random_metric


@dataclass(frozen=True)
class OfficialIRMConfig:
    data_root: str = "data"
    download: bool = True
    seeds: tuple[int, ...] = (0,)
    hidden_dim: int = 256
    l2_regularizer_weight: float = 1e-3
    learning_rate: float = 1e-3
    penalty_anneal_iters: int = 100
    penalty_weight: float = 10000.0
    steps: int = 501
    label_noise: float = 0.25
    train_color_flip_probs: tuple[float, float] = (0.2, 0.1)
    target_color_flip_prob: float = 0.9
    device: str = "cpu"
    task3_methods: tuple[str, ...] = (
        "ERM",
        "IRMv1",
        "UNPRECONDITIONED_GRAD_ALIGN",
        "LOCAL_RESPONSE",
    )
    task3_penalty_sample_per_env: int = 512
    task3_response_penalty_weights: tuple[float, ...] = (1e-3, 1e-2, 1e-1, 1.0, 10.0)
    task3_irm_penalty_weights: tuple[float, ...] = (10000.0,)
    curvature_epsilon: float = 1e-2


@dataclass(frozen=True)
class OfficialEnvironment:
    images: Tensor
    labels: Tensor
    colors: Tensor
    color_flip_prob: float
    role: str


class OfficialIRMMLP(nn.Module):
    def __init__(self, hidden_dim: int = 256) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(2 * 14 * 14, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
        )
        self.head = nn.Linear(hidden_dim, 1)
        for module in [*self.encoder, self.head]:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def encode(self, x: Tensor) -> Tensor:
        return self.encoder(x.view(x.shape[0], 2 * 14 * 14))

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.encode(x))


def _xor(a: Tensor, b: Tensor) -> Tensor:
    return (a - b).abs()


def _bernoulli(probability: float, size: int, generator: torch.Generator) -> Tensor:
    return (torch.rand(size, generator=generator) < probability).float()


def _make_environment(
    images: Tensor,
    digits: Tensor,
    *,
    color_flip_prob: float,
    label_noise: float,
    role: str,
    generator: torch.Generator,
) -> OfficialEnvironment:
    gray = images[:, 0, ::2, ::2]
    labels = (digits < 5).float()
    labels = _xor(labels, _bernoulli(label_noise, len(labels), generator))
    colors = _xor(labels, _bernoulli(color_flip_prob, len(labels), generator))
    colored = torch.stack([gray, gray], dim=1)
    colored[torch.arange(len(colored)), (1 - colors).long(), :, :] = 0.0
    return OfficialEnvironment(
        images=colored.float(),
        labels=labels[:, None].float(),
        colors=colors.float(),
        color_flip_prob=color_flip_prob,
        role=role,
    )


def build_official_environments(config: OfficialIRMConfig, seed: int) -> tuple[OfficialEnvironment, OfficialEnvironment, OfficialEnvironment]:
    root = Path(config.data_root)
    gray, digits = load_mnist_tensors(root, train=True, download=config.download)
    generator = torch.Generator().manual_seed(seed + 12345)
    env0 = _make_environment(
        gray[:25000],
        digits[:25000],
        color_flip_prob=config.train_color_flip_probs[0],
        label_noise=config.label_noise,
        role="train_env0",
        generator=generator,
    )
    env1 = _make_environment(
        gray[25000:50000],
        digits[25000:50000],
        color_flip_prob=config.train_color_flip_probs[1],
        label_noise=config.label_noise,
        role="train_env1",
        generator=generator,
    )
    target = _make_environment(
        gray[50000:],
        digits[50000:],
        color_flip_prob=config.target_color_flip_prob,
        label_noise=config.label_noise,
        role="reversed_target",
        generator=generator,
    )
    return env0, env1, target


def _mean_nll(logits: Tensor, labels: Tensor) -> Tensor:
    return F.binary_cross_entropy_with_logits(logits, labels)


def _mean_accuracy(logits: Tensor, labels: Tensor) -> Tensor:
    preds = (logits > 0.0).float()
    return ((preds - labels).abs() < 1e-2).float().mean()


def _irm_penalty(logits: Tensor, labels: Tensor) -> Tensor:
    scale = torch.tensor(1.0, device=logits.device, requires_grad=True)
    loss = _mean_nll(logits * scale, labels)
    grad = autograd.grad(loss, [scale], create_graph=True)[0]
    return grad.square().sum()


def _head_gradient_vector(loss: Tensor, model: OfficialIRMMLP) -> Tensor:
    grads = torch.autograd.grad(
        loss,
        tuple(model.head.parameters()),
        create_graph=True,
        retain_graph=True,
        allow_unused=False,
    )
    return torch.cat([grad.reshape(-1) for grad in grads])


def _binary_head_hessian(features: tuple[Tensor, ...], logits: tuple[Tensor, ...]) -> Tensor:
    with torch.no_grad():
        z = torch.cat([item.detach() for item in features], dim=0).double()
        out = torch.cat([item.detach() for item in logits], dim=0).double().reshape(-1)
        probs = torch.sigmoid(out)
        coeff = probs * (1.0 - probs) / max(out.numel(), 1)
        hidden_dim = z.shape[1]
        hessian = torch.zeros((hidden_dim + 1, hidden_dim + 1), dtype=z.dtype, device=z.device)
        hessian[:hidden_dim, :hidden_dim] = z.T @ (z * coeff[:, None])
        cross = (z * coeff[:, None]).sum(dim=0)
        hessian[:hidden_dim, hidden_dim] = cross
        hessian[hidden_dim, :hidden_dim] = cross
        hessian[hidden_dim, hidden_dim] = coeff.sum()
        return (hessian + hessian.T) / 2.0


def _damped_inverse(hessian: Tensor, epsilon: float) -> tuple[Tensor, Tensor, Tensor, float, float]:
    sym = (hessian.detach().double() + hessian.detach().double().T) / 2.0
    dim = sym.shape[0]
    trace_scale = float(torch.trace(sym).abs().item() / max(dim, 1))
    damping = epsilon * max(trace_scale, 1e-8)
    matrix = sym + damping * torch.eye(dim, dtype=sym.dtype, device=sym.device)
    eigenvalues = torch.linalg.eigvalsh((matrix + matrix.T) / 2.0)
    if float(eigenvalues.min()) <= 0.0:
        matrix = matrix + (-float(eigenvalues.min()) + max(damping, 1e-8)) * torch.eye(
            dim, dtype=sym.dtype, device=sym.device
        )
        eigenvalues = torch.linalg.eigvalsh((matrix + matrix.T) / 2.0)
    inverse = inverse_via_eigh(matrix).detach()
    condition = float(eigenvalues.max().item() / max(eigenvalues.min().item(), 1e-12))
    return matrix.detach(), inverse, eigenvalues.detach(), float(damping), condition


def _official_source_forward(
    model: OfficialIRMMLP,
    train_envs: tuple[OfficialEnvironment, OfficialEnvironment],
    device: torch.device,
) -> tuple[tuple[Tensor, ...], tuple[Tensor, ...], tuple[Tensor, ...]]:
    losses: list[Tensor] = []
    features: list[Tensor] = []
    logits: list[Tensor] = []
    for env in train_envs:
        z = model.encode(env.images.to(device))
        out = model.head(z)
        features.append(z)
        logits.append(out)
        losses.append(_mean_nll(out, env.labels.to(device)))
    return tuple(losses), tuple(features), tuple(logits)


def _slice_envs(
    train_envs: tuple[OfficialEnvironment, OfficialEnvironment], max_per_env: int
) -> tuple[OfficialEnvironment, OfficialEnvironment]:
    result = []
    for env in train_envs:
        stop = min(max_per_env, env.images.shape[0])
        result.append(
            OfficialEnvironment(
                images=env.images[:stop],
                labels=env.labels[:stop],
                colors=env.colors[:stop],
                color_flip_prob=env.color_flip_prob,
                role=f"{env.role}_penalty_subset",
            )
        )
    return tuple(result)  # type: ignore[return-value]


def official_response_penalty(
    method: str,
    model: OfficialIRMMLP,
    train_envs: tuple[OfficialEnvironment, OfficialEnvironment],
    *,
    device: torch.device,
    epsilon: float,
    random_seed: int = 0,
) -> tuple[Tensor, dict[str, float | str]]:
    losses, features, logits = _official_source_forward(model, train_envs, device)
    centered = torch.stack([_head_gradient_vector(loss, model) for loss in losses], dim=0)
    centered = centered - centered.mean(dim=0, keepdim=True)
    raw = centered.square().sum(dim=1).mean()
    key = method.upper()
    if key == "UNPRECONDITIONED_GRAD_ALIGN":
        response_norm = centered.square().sum(dim=1).mean().sqrt()
        return raw, {
            "response_penalty_metric": "identity",
            "raw_gradient_disagreement": float(raw.detach().cpu()),
            "local_response_disagreement": float("nan"),
            "response_vector_norm": float(response_norm.detach().cpu()),
            "hessian_min_eigenvalue": float("nan"),
            "hessian_max_eigenvalue": float("nan"),
            "damped_condition_number": float("nan"),
            "damping": float("nan"),
        }

    hessian = _binary_head_hessian(features, logits)
    matrix, inverse, eigenvalues, damping, condition = _damped_inverse(hessian, epsilon)
    if key == "LOCAL_RESPONSE":
        inv = inverse.to(dtype=centered.dtype, device=centered.device)
        penalty = torch.einsum("bi,ij,bj->", centered, inv, centered) / centered.shape[0]
        metric_kind = "real_inverse_hessian"
    elif key == "RANDOM_METRIC":
        random_metric = matched_random_metric(matrix, random_seed)
        inv = inverse_via_eigh(random_metric).detach().to(dtype=centered.dtype, device=centered.device)
        penalty = torch.einsum("bi,ij,bj->", centered, inv, centered) / centered.shape[0]
        metric_kind = "random_matched_spd"
    else:
        raise ValueError(f"unknown response penalty method: {method}")
    response_vectors = centered @ inverse.to(dtype=centered.dtype, device=centered.device).T
    return penalty, {
        "response_penalty_metric": metric_kind,
        "raw_gradient_disagreement": float(raw.detach().cpu()),
        "local_response_disagreement": float(
            torch.einsum(
                "bi,ij,bj->",
                centered,
                inverse.to(dtype=centered.dtype, device=centered.device),
                centered,
            ).detach().cpu()
            / centered.shape[0]
        ),
        "response_vector_norm": float(response_vectors.square().sum(dim=1).mean().sqrt().detach().cpu()),
        "hessian_min_eigenvalue": float(eigenvalues.min().cpu()),
        "hessian_max_eigenvalue": float(eigenvalues.max().cpu()),
        "damped_condition_number": condition,
        "damping": damping,
    }


def _weight_norm(model: nn.Module) -> Tensor:
    total = next(model.parameters()).new_zeros(())
    for parameter in model.parameters():
        total = total + parameter.norm().pow(2)
    return total


@torch.no_grad()
def _eval_model(model: nn.Module, environment: OfficialEnvironment, device: torch.device) -> dict[str, float]:
    logits = model(environment.images.to(device))
    labels = environment.labels.to(device)
    preds = (logits > 0.0).float().cpu().reshape(-1)
    labels_cpu = environment.labels.cpu().reshape(-1)
    colors_cpu = environment.colors.cpu().reshape(-1)
    return {
        "nll": float(_mean_nll(logits, labels).detach().cpu()),
        "accuracy": float(_mean_accuracy(logits, labels).detach().cpu()),
        "color_label_accuracy": float((colors_cpu == labels_cpu).float().mean()),
        "inverse_color_label_accuracy": float(((1 - colors_cpu) == labels_cpu).float().mean()),
        "prediction_color_agreement": float((preds == colors_cpu).float().mean()),
        "prediction_inverse_color_agreement": float((preds == 1 - colors_cpu).float().mean()),
    }


def train_official_reference(
    train_envs: tuple[OfficialEnvironment, OfficialEnvironment],
    target_env: OfficialEnvironment,
    *,
    method: str,
    seed: int,
    config: OfficialIRMConfig,
) -> dict[str, Any]:
    seed_everything(seed)
    device = torch.device(config.device)
    model = OfficialIRMMLP(config.hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    method_key = method.upper()

    for step in range(config.steps):
        env_metrics = []
        for env in train_envs:
            logits = model(env.images.to(device))
            labels = env.labels.to(device)
            env_metrics.append({
                "nll": _mean_nll(logits, labels),
                "acc": _mean_accuracy(logits, labels),
                "penalty": _irm_penalty(logits, labels),
            })
        train_nll = torch.stack([item["nll"] for item in env_metrics]).mean()
        train_acc = torch.stack([item["acc"] for item in env_metrics]).mean()
        train_penalty = torch.stack([item["penalty"] for item in env_metrics]).mean()
        loss = train_nll + config.l2_regularizer_weight * _weight_norm(model)
        if method_key == "IRMV1":
            penalty_weight = config.penalty_weight if step >= config.penalty_anneal_iters else 1.0
            loss = loss + penalty_weight * train_penalty
            if penalty_weight > 1.0:
                loss = loss / penalty_weight
        elif method_key != "ERM":
            raise ValueError(f"unknown official reference method: {method}")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_eval = [_eval_model(model, env, device) for env in train_envs]
    target_eval = _eval_model(model, target_env, device)
    return {
        "seed": seed,
        "method": method_key,
        "steps": config.steps,
        "hidden_dim": config.hidden_dim,
        "l2_regularizer_weight": config.l2_regularizer_weight,
        "learning_rate": config.learning_rate,
        "penalty_anneal_iters": config.penalty_anneal_iters,
        "penalty_weight": config.penalty_weight if method_key == "IRMV1" else 0.0,
        "label_noise": config.label_noise,
        "train_color_flip_probs": list(config.train_color_flip_probs),
        "target_color_flip_prob": config.target_color_flip_prob,
        "train_examples_per_environment": [int(env.images.shape[0]) for env in train_envs],
        "target_examples": int(target_env.images.shape[0]),
        "source_environment_count": len(train_envs),
        "train_accuracy": float(sum(item["accuracy"] for item in train_eval) / len(train_eval)),
        "target_accuracy": target_eval["accuracy"],
        "target_nll": target_eval["nll"],
        "target_color_label_accuracy": target_eval["color_label_accuracy"],
        "target_inverse_color_label_accuracy": target_eval["inverse_color_label_accuracy"],
        "prediction_color_agreement": target_eval["prediction_color_agreement"],
        "prediction_inverse_color_agreement": target_eval["prediction_inverse_color_agreement"],
        "final_train_nll": float(train_nll.detach().cpu()),
        "final_train_accuracy_step_metric": float(train_acc.detach().cpu()),
        "final_train_penalty": float(train_penalty.detach().cpu()),
        "reference_protocol": "facebookresearch_irm_colored_mnist_reimplementation",
    }


def run_official_reference(config: OfficialIRMConfig) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in config.seeds:
        env0, env1, target = build_official_environments(config, int(seed))
        for method in ("ERM", "IRMv1"):
            rows.append(
                train_official_reference((env0, env1), target, method=method, seed=int(seed), config=config)
            )
    return rows


def train_official_task3_method(
    train_envs: tuple[OfficialEnvironment, OfficialEnvironment],
    target_env: OfficialEnvironment,
    *,
    method: str,
    seed: int,
    config: OfficialIRMConfig,
    penalty_weight: float | None = None,
) -> dict[str, Any]:
    seed_everything(seed)
    device = torch.device(config.device)
    model = OfficialIRMMLP(config.hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    method_key = method.upper()
    configured_weight = 0.0 if method_key == "ERM" else float(
        config.penalty_weight if penalty_weight is None else penalty_weight
    )
    penalty_envs = _slice_envs(train_envs, config.task3_penalty_sample_per_env)
    final_diag: dict[str, float | str] = {}

    for step in range(config.steps):
        env_metrics = []
        for env in train_envs:
            logits = model(env.images.to(device))
            labels = env.labels.to(device)
            env_metrics.append({
                "nll": _mean_nll(logits, labels),
                "acc": _mean_accuracy(logits, labels),
                "penalty": _irm_penalty(logits, labels),
            })
        train_nll = torch.stack([item["nll"] for item in env_metrics]).mean()
        train_acc = torch.stack([item["acc"] for item in env_metrics]).mean()
        train_penalty = next(model.parameters()).new_zeros(())
        final_diag = {}
        if method_key == "IRMV1":
            train_penalty = torch.stack([item["penalty"] for item in env_metrics]).mean()
            final_diag = {"response_penalty_metric": "official_scalar_scale"}
        elif method_key in {"UNPRECONDITIONED_GRAD_ALIGN", "LOCAL_RESPONSE", "RANDOM_METRIC"}:
            train_penalty, final_diag = official_response_penalty(
                method_key,
                model,
                penalty_envs,
                device=device,
                epsilon=config.curvature_epsilon,
                random_seed=seed,
            )
        elif method_key != "ERM":
            raise ValueError(f"unknown official Task 3 method: {method}")

        loss = train_nll + config.l2_regularizer_weight * _weight_norm(model)
        step_penalty_weight = 0.0
        rescaled_after_anneal = False
        if method_key == "IRMV1":
            step_penalty_weight = configured_weight if step >= config.penalty_anneal_iters else 1.0
            loss = loss + step_penalty_weight * train_penalty
            if step_penalty_weight > 1.0:
                loss = loss / step_penalty_weight
                rescaled_after_anneal = True
        elif method_key != "ERM":
            step_penalty_weight = configured_weight
            loss = loss + step_penalty_weight * train_penalty
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_eval = [_eval_model(model, env, device) for env in train_envs]
    target_eval = _eval_model(model, target_env, device)
    with torch.enable_grad():
        diagnostics = {}
        for diag_method in ("UNPRECONDITIONED_GRAD_ALIGN", "LOCAL_RESPONSE"):
            _, diag = official_response_penalty(
                diag_method,
                model,
                penalty_envs,
                device=device,
                epsilon=config.curvature_epsilon,
                random_seed=seed,
            )
            diagnostics.update({f"diagnostic_{diag_method.lower()}_{key}": value for key, value in diag.items()})
    return {
        "seed": seed,
        "method": method_key,
        "steps": config.steps,
        "hidden_dim": config.hidden_dim,
        "penalty_sample_per_env": config.task3_penalty_sample_per_env,
        "l2_regularizer_weight": config.l2_regularizer_weight,
        "learning_rate": config.learning_rate,
        "penalty_anneal_iters": config.penalty_anneal_iters,
        "penalty_weight": configured_weight,
        "final_step_penalty_weight": step_penalty_weight,
        "official_irm_loss_rescale_applied": bool(method_key == "IRMV1" and rescaled_after_anneal),
        "response_method_uses_irm_loss_rescale": False,
        "label_noise": config.label_noise,
        "train_color_flip_probs": list(config.train_color_flip_probs),
        "target_color_flip_prob": config.target_color_flip_prob,
        "train_examples_per_environment": [int(env.images.shape[0]) for env in train_envs],
        "target_examples": int(target_env.images.shape[0]),
        "source_environment_count": len(train_envs),
        "train_accuracy": float(sum(item["accuracy"] for item in train_eval) / len(train_eval)),
        "target_accuracy": target_eval["accuracy"],
        "target_nll": target_eval["nll"],
        "target_color_label_accuracy": target_eval["color_label_accuracy"],
        "target_inverse_color_label_accuracy": target_eval["inverse_color_label_accuracy"],
        "prediction_color_agreement": target_eval["prediction_color_agreement"],
        "prediction_inverse_color_agreement": target_eval["prediction_inverse_color_agreement"],
        "final_train_nll": float(train_nll.detach().cpu()),
        "final_train_accuracy_step_metric": float(train_acc.detach().cpu()),
        "final_train_penalty": float(train_penalty.detach().cpu()),
        **{key: value for key, value in final_diag.items() if isinstance(value, (float, int, str))},
        **diagnostics,
        "reference_protocol": "official_colored_mnist_task3_same_protocol_pilot",
    }


def run_official_task3_comparison(config: OfficialIRMConfig) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in config.seeds:
        env0, env1, target = build_official_environments(config, int(seed))
        for method in config.task3_methods:
            method_key = method.upper()
            if method_key == "ERM":
                weights = (0.0,)
            elif method_key == "IRMV1":
                weights = config.task3_irm_penalty_weights
            else:
                weights = config.task3_response_penalty_weights
            group_start = len(rows)
            for weight in weights:
                rows.append(
                    train_official_task3_method(
                        (env0, env1),
                        target,
                        method=method,
                        seed=int(seed),
                        config=config,
                        penalty_weight=float(weight),
                    )
                )
            group = rows[group_start:]
            selected = max(
                group,
                key=lambda row: (float(row["train_accuracy"]), -abs(float(row["penalty_weight"]))),
            )
            for row in group:
                row["selected_by_source_rule"] = row is selected
                row["selection_rule"] = "max_train_accuracy_then_smaller_abs_penalty_weight_no_target"
    return rows


def official_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def is_selected(row: dict[str, Any]) -> bool:
        value = row.get("selected_by_source_rule", True)
        if isinstance(value, str):
            return value.strip().lower() == "true"
        return bool(value)

    selected_rows = [row for row in rows if is_selected(row)]
    methods = sorted({str(row["method"]) for row in selected_rows})
    method_summary = []
    for method in methods:
        values = [row for row in selected_rows if row["method"] == method]
        method_summary.append({
            "method": method,
            "n": len(values),
            "mean_selected_penalty_weight": float(np.mean([float(row["penalty_weight"]) for row in values])),
            "mean_target_accuracy": float(np.mean([float(row["target_accuracy"]) for row in values])),
            "mean_train_accuracy": float(np.mean([float(row["train_accuracy"]) for row in values])),
            "mean_prediction_color_agreement": float(np.mean([float(row["prediction_color_agreement"]) for row in values])),
        })
    by_seed_method = {(int(row["seed"]), str(row["method"])): row for row in selected_rows}
    comparisons = []
    for left, right in [("LOCAL_RESPONSE", "IRMV1"), ("LOCAL_RESPONSE", "UNPRECONDITIONED_GRAD_ALIGN"), ("LOCAL_RESPONSE", "ERM"), ("IRMV1", "ERM")]:
        diffs = []
        for seed in sorted({seed for seed, _method in by_seed_method}):
            if (seed, left) in by_seed_method and (seed, right) in by_seed_method:
                diffs.append(float(by_seed_method[(seed, left)]["target_accuracy"]) - float(by_seed_method[(seed, right)]["target_accuracy"]))
        if diffs:
            comparisons.append({
                "left_method": left,
                "right_method": right,
                "n_pairs": len(diffs),
                "mean_difference": float(np.mean(diffs)),
                "seed_wins": int(sum(value > 0 for value in diffs)),
            })
    return {
        "method_summary": method_summary,
        "paired_comparisons": comparisons,
        "selected_rows": len(selected_rows),
        "candidate_rows": len(rows),
    }
