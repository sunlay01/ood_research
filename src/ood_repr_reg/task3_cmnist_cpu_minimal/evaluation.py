"""Evaluation-only utilities for the CPU-minimal ColoredMNIST probe."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .data import ColoredEnvironment
from .methods import grad_response_penalty, local_response_penalty


@torch.no_grad()
def evaluate_environment(
    model: nn.Module,
    env: ColoredEnvironment,
    *,
    device: torch.device | str = "cpu",
    batch_size: int = 4096,
) -> dict[str, float]:
    model.eval()
    logits_parts: list[Tensor] = []
    for start in range(0, env.images.shape[0], batch_size):
        stop = min(start + batch_size, env.images.shape[0])
        logits_parts.append(model(env.images[start:stop].to(device)).detach().cpu())
    logits = torch.cat(logits_parts, dim=0)
    labels = env.labels.cpu()
    preds = (logits > 0.0).float().reshape(-1)
    colors = env.colors.cpu().reshape(-1)
    return {
        "loss": float(F.binary_cross_entropy_with_logits(logits, labels).item()),
        "accuracy": float((preds[:, None] == labels).float().mean().item()),
        "prediction_color_agreement": float((preds == colors).float().mean().item()),
    }


def evaluate_checkpoint(
    model: nn.Module,
    source_envs: tuple[ColoredEnvironment, ColoredEnvironment],
    target_env: ColoredEnvironment,
    *,
    device: torch.device | str = "cpu",
) -> dict[str, float]:
    source0 = evaluate_environment(model, source_envs[0], device=device)
    source1 = evaluate_environment(model, source_envs[1], device=device)
    target = evaluate_environment(model, target_env, device=device)
    return {
        "source_env0_acc": source0["accuracy"],
        "source_env1_acc": source1["accuracy"],
        "source_mean_acc": (source0["accuracy"] + source1["accuracy"]) / 2.0,
        "target_acc": target["accuracy"],
        "target_loss": target["loss"],
        "prediction_color_agreement": target["prediction_color_agreement"],
    }


def response_diagnostics(
    model: nn.Module,
    batches: tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]],
    *,
    damping_epsilon: float,
) -> dict[str, float]:
    model.train()
    grad = grad_response_penalty(model, batches)
    local = local_response_penalty(model, batches, damping_epsilon=damping_epsilon)
    return {
        "raw_grad_disagreement": float(grad.penalty.detach().cpu()),
        "local_response_disagreement": float(local.penalty.detach().cpu()),
        "local_response_min_eigenvalue": float(local.min_eigenvalue or float("nan")),
        "local_response_condition_number": float(local.condition_number or float("nan")),
    }


def checkpoint_row(
    *,
    seed: int,
    method: str,
    step: int,
    eval_metrics: dict[str, float],
    diagnostics: dict[str, float],
    loss: float,
) -> dict[str, Any]:
    return {
        "seed": int(seed),
        "method": method,
        "step": int(step),
        "source_env0_acc": eval_metrics["source_env0_acc"],
        "source_env1_acc": eval_metrics["source_env1_acc"],
        "target_acc": eval_metrics["target_acc"],
        "prediction_color_agreement": eval_metrics["prediction_color_agreement"],
        "raw_grad_disagreement": diagnostics["raw_grad_disagreement"],
        "local_response_disagreement": diagnostics["local_response_disagreement"],
        "loss": float(loss),
    }
