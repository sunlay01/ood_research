"""Finite source-perturbation validation for frozen-encoder head responses."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from .learner_response import HeadReference, refine_head


@dataclass(frozen=True)
class FiniteResponse:
    predicted: Tensor
    actual: Tensor
    cosine_similarity: float
    relative_norm_error: float
    relative_vector_error: float
    finite: bool


def finite_response(
    method: str,
    plus_features: tuple[Tensor, Tensor],
    plus_labels: tuple[Tensor, Tensor],
    minus_features: tuple[Tensor, Tensor],
    minus_labels: tuple[Tensor, Tensor],
    reference: HeadReference,
    source_root: Tensor,
    predicted: Tensor,
    epsilon: float,
) -> FiniteResponse:
    plus = refine_head(method, plus_features, plus_labels, reference.weights)
    minus = refine_head(method, minus_features, minus_labels, reference.weights)
    actual = source_root @ ((plus.weights - minus.weights) / (2.0 * float(epsilon)))
    prediction = predicted.detach().cpu().double()
    actual = actual.detach().cpu().double()
    denom = max(float(prediction.norm().item() * actual.norm().item()), 1e-12)
    cosine = float((prediction @ actual).item() / denom)
    norm_error = float(abs(prediction.norm().item() - actual.norm().item()) / max(actual.norm().item(), 1e-12))
    vector_error = float((prediction - actual).norm().item() / max(actual.norm().item(), 1e-12))
    finite = bool(torch.isfinite(prediction).all() and torch.isfinite(actual).all() and math.isfinite(cosine) and math.isfinite(vector_error))
    return FiniteResponse(prediction, actual, cosine, norm_error, vector_error, finite)


def sign_agreement(task_response: Tensor, response: FiniteResponse) -> bool:
    expected = float(task_response.detach().cpu().double() @ response.predicted)
    observed = float(task_response.detach().cpu().double() @ response.actual)
    if abs(expected) <= 1e-12 or abs(observed) <= 1e-12:
        return False
    return (expected > 0.0) == (observed > 0.0)
