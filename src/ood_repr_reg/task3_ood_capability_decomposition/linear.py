"""Small deterministic linear utilities for capability diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import math

import torch
from torch import Tensor


@dataclass(frozen=True)
class RidgeClassifier:
    weight: Tensor
    bias: Tensor
    threshold: float = 0.5

    def scores(self, features: Tensor) -> Tensor:
        return features.detach().cpu().double() @ self.weight + self.bias

    def predictions(self, features: Tensor) -> Tensor:
        return (self.scores(features) >= float(self.threshold)).double()


def _vector_labels(labels: Tensor) -> Tensor:
    return labels.detach().cpu().double().reshape(-1)


def fit_ridge_classifier(features: Tensor, labels: Tensor, *, ridge: float = 1e-3) -> RidgeClassifier:
    """Fit a closed-form ridge regressor used as a deterministic binary classifier."""
    x = features.detach().cpu().double()
    y = _vector_labels(labels)
    if x.ndim != 2 or x.shape[0] != y.numel():
        raise ValueError("features must be [N,D] and labels must have N entries")
    ones = torch.ones((x.shape[0], 1), dtype=x.dtype)
    design = torch.cat((x, ones), dim=1)
    regularizer = torch.eye(design.shape[1], dtype=x.dtype) * float(ridge)
    regularizer[-1, -1] = 0.0
    solution = torch.linalg.solve(design.T @ design + regularizer, design.T @ y)
    return RidgeClassifier(weight=solution[:-1].contiguous(), bias=solution[-1].clone())


def classifier_accuracy(classifier: RidgeClassifier, features: Tensor, labels: Tensor) -> float:
    y = _vector_labels(labels)
    return float((classifier.predictions(features) == y).double().mean().item())


def two_fold_ridge_accuracy(features: Tensor, labels: Tensor, *, ridge: float = 1e-3) -> dict[str, float]:
    x = features.detach().cpu().double()
    y = _vector_labels(labels)
    indices = torch.arange(x.shape[0])
    fold0_eval = indices % 2 == 0
    fold1_eval = ~fold0_eval
    if int(fold0_eval.sum().item()) == 0 or int(fold1_eval.sum().item()) == 0:
        raise ValueError("two-fold ridge accuracy requires at least two examples")
    clf0 = fit_ridge_classifier(x[fold1_eval], y[fold1_eval], ridge=ridge)
    clf1 = fit_ridge_classifier(x[fold0_eval], y[fold0_eval], ridge=ridge)
    pred = torch.empty_like(y)
    pred[fold0_eval] = clf0.predictions(x[fold0_eval])
    pred[fold1_eval] = clf1.predictions(x[fold1_eval])
    return {
        "accuracy": float((pred == y).double().mean().item()),
        "fold0_accuracy": float((pred[fold0_eval] == y[fold0_eval]).double().mean().item()),
        "fold1_accuracy": float((pred[fold1_eval] == y[fold1_eval]).double().mean().item()),
    }


def color_response_basis(red_features: Tensor, green_features: Tensor) -> Tensor:
    delta = green_features.detach().cpu().double() - red_features.detach().cpu().double()
    if delta.ndim != 2:
        raise ValueError("counterfactual feature deltas must be [N,D]")
    moment = delta.T @ delta / max(int(delta.shape[0]), 1)
    eigenvalues, eigenvectors = torch.linalg.eigh((moment + moment.T) / 2.0)
    order = torch.argsort(eigenvalues, descending=True)
    return eigenvectors[:, order].contiguous()


def projection_matrix(basis: Tensor, rank: int) -> Tensor:
    b = basis.detach().cpu().double()
    if b.ndim != 2 or b.shape[0] != b.shape[1]:
        raise ValueError("basis must be square [D,D]")
    dim = b.shape[0]
    clipped = max(0, min(int(rank), int(dim)))
    identity = torch.eye(dim, dtype=b.dtype)
    if clipped == 0:
        return identity
    u = b[:, :clipped]
    projection = identity - u @ u.T
    return ((projection + projection.T) / 2.0).contiguous()


def apply_projection(features: Tensor, projection: Tensor) -> Tensor:
    return features.detach().cpu().double() @ projection.detach().cpu().double()


def binary_accuracy_from_logits(logits: Tensor, labels: Tensor) -> float:
    preds = (logits.detach().cpu().double().reshape(-1) >= 0.0).double()
    y = _vector_labels(labels)
    return float((preds == y).double().mean().item())


def safe_ratio(numerator: float, denominator: float, *, tol: float = 1e-12) -> float:
    if not math.isfinite(numerator) or not math.isfinite(denominator) or abs(denominator) <= tol:
        return float("nan")
    return numerator / denominator
