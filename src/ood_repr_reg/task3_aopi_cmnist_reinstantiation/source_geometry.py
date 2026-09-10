"""Frozen-encoder feature extraction and exact 65-dimensional A/O geometry."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ..task3_cmnist_cpu_minimal.data import ColoredEnvironment
from ..task3_cmnist_cpu_minimal.model import CPUColoredMNISTMLP, parameter_hash
from .learner_response import HEAD_DIMENSION, feature_tuple, head_logits, risk_loss, source_state
from .world_tangents import TangentWorlds


HESSIAN_DAMPING_RELATIVE = 1e-6
RETAIN_RELATIVE = 1e-10


@dataclass(frozen=True)
class FrozenEncoder:
    model: CPUColoredMNISTMLP
    encoder_hash: str


@dataclass(frozen=True)
class Whitening:
    hessian: Tensor
    effective_hessian: Tensor
    root: Tensor
    inverse_root: Tensor
    eigenvalues: Tensor
    retained: Tensor
    damping: float
    min_eigenvalue: float
    max_eigenvalue: float
    condition_number: float
    identity_error: float


def encoder_parameter_hash(model: CPUColoredMNISTMLP) -> str:
    import hashlib

    hasher = hashlib.sha256()
    with torch.no_grad():
        for parameter in model.encoder.parameters():
            hasher.update(parameter.detach().cpu().numpy().tobytes())
    return hasher.hexdigest()


def freeze_encoder(model: CPUColoredMNISTMLP) -> FrozenEncoder:
    model.eval()
    for parameter in model.encoder.parameters():
        parameter.requires_grad_(False)
    return FrozenEncoder(model=model, encoder_hash=encoder_parameter_hash(model))


@torch.no_grad()
def encode_environment(encoder: FrozenEncoder, environment: ColoredEnvironment, *, batch_size: int = 4096) -> Tensor:
    parts: list[Tensor] = []
    for start in range(0, int(environment.images.shape[0]), batch_size):
        parts.append(encoder.model.encode(environment.images[start:start + batch_size]).detach().cpu().double())
    output = torch.cat(parts, dim=0)
    if encoder_parameter_hash(encoder.model) != encoder.encoder_hash:
        raise RuntimeError("frozen encoder parameters changed during feature extraction")
    return output


def source_features(encoder: FrozenEncoder, worlds: TangentWorlds) -> tuple[Tensor, Tensor]:
    return feature_tuple(encode_environment(encoder, item) for item in worlds.source_envs)


def source_labels(worlds: TangentWorlds) -> tuple[Tensor, Tensor]:
    return tuple(item.labels.detach().cpu().double() for item in worlds.source_envs)  # type: ignore[return-value]


def evaluation_gradient(encoder: FrozenEncoder, worlds: TangentWorlds, weights: Tensor) -> Tensor:
    features = feature_tuple(encode_environment(encoder, item) for item in worlds.evaluation_envs)
    labels = tuple(item.labels.detach().cpu().double() for item in worlds.evaluation_envs)
    point = weights.detach().cpu().double().clone().requires_grad_(True)
    loss = torch.stack([risk_loss(x, y, point) for x, y in zip(features, labels)]).mean()
    return torch.autograd.grad(loss, point)[0].detach().cpu().double()


def source_risk_hessian(features: tuple[Tensor, Tensor], weights: Tensor) -> Tensor:
    terms: list[Tensor] = []
    for item in features:
        logits = head_logits(item, weights).reshape(-1)
        augmented = torch.cat((item.double(), torch.ones((item.shape[0], 1), dtype=torch.double)), dim=1)
        coefficient = torch.sigmoid(logits) * (1.0 - torch.sigmoid(logits)) / max(int(item.shape[0]), 1)
        terms.append(augmented.T @ (augmented * coefficient[:, None]))
    hessian = torch.stack(terms).mean(dim=0)
    return ((hessian + hessian.T) / 2.0).detach().cpu().double()


def whiten_source_hessian(hessian: Tensor) -> Whitening:
    symmetric = ((hessian.double() + hessian.double().T) / 2.0).detach().cpu()
    values, vectors = torch.linalg.eigh(symmetric)
    max_value = max(float(values.max().item()), 1e-8)
    retained = values >= RETAIN_RELATIVE * max_value
    damping = HESSIAN_DAMPING_RELATIVE * max(float(torch.trace(symmetric).item() / HEAD_DIMENSION), 1e-8)
    effective_values = values + damping
    if bool((effective_values <= 0.0).any()):
        raise ValueError("source risk Hessian is not positive after fixed damping")
    root = vectors @ torch.diag(effective_values.sqrt()) @ vectors.T
    inverse_root = vectors @ torch.diag(effective_values.rsqrt()) @ vectors.T
    effective = root @ root
    identity = inverse_root @ effective @ inverse_root
    error = float((identity - torch.eye(HEAD_DIMENSION, dtype=torch.double)).norm().item() / HEAD_DIMENSION)
    return Whitening(
        hessian=symmetric,
        effective_hessian=effective,
        root=root,
        inverse_root=inverse_root,
        eigenvalues=effective_values,
        retained=retained,
        damping=damping,
        min_eigenvalue=float(effective_values.min().item()),
        max_eigenvalue=float(effective_values.max().item()),
        condition_number=float((effective_values.max() / effective_values.min()).item()),
        identity_error=error,
    )


def response_A(
    encoder: FrozenEncoder,
    plus: TangentWorlds,
    minus: TangentWorlds,
    weights: Tensor,
    inverse_root: Tensor,
    epsilon: float,
) -> Tensor:
    derivative = (evaluation_gradient(encoder, plus, weights) - evaluation_gradient(encoder, minus, weights)) / (2.0 * float(epsilon))
    return inverse_root @ derivative


def observation_O(
    plus_features: tuple[Tensor, Tensor],
    plus_labels: tuple[Tensor, Tensor],
    minus_features: tuple[Tensor, Tensor],
    minus_labels: tuple[Tensor, Tensor],
    weights: Tensor,
    epsilon: float,
) -> Tensor:
    plus_state = source_state(plus_features, plus_labels, weights, create_graph=False)
    minus_state = source_state(minus_features, minus_labels, weights, create_graph=False)
    return ((plus_state - minus_state) / (2.0 * float(epsilon))).detach().cpu().double()


def normalized_gram(columns: Tensor) -> Tensor:
    gram = columns.T @ columns
    norm = gram.norm()
    return gram / norm if float(norm) > 0.0 else gram


def geometry_summary(columns: Tensor) -> dict[str, float | int]:
    singular = torch.linalg.svdvals(columns)
    threshold = max(float(singular.max().item()) if singular.numel() else 0.0, 1e-12) * 1e-8
    return {
        "response_rank": int((singular > threshold).sum().item()),
        "response_norm": float(columns.norm().item()),
        "response_singular_max": float(singular.max().item()) if singular.numel() else 0.0,
        "response_singular_min": float(singular.min().item()) if singular.numel() else 0.0,
    }
