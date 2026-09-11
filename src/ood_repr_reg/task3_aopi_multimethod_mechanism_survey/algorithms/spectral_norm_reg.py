"""Spectral norm regularization, distinct from spectral normalization."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ...task3_cmnist_cpu_minimal.methods import ObjectiveParts, source_environment_losses, weight_norm_squared
from ..smooth_world5 import SmoothWorld5
from .base import BaseAlgorithm, smooth_source_risk_vector


def linear_weights(model: nn.Module) -> tuple[Tensor, ...]:
    return tuple(module.weight for module in model.modules() if isinstance(module, nn.Linear))


def spectral_norm_regularizer(model: nn.Module) -> Tensor:
    return sum(torch.linalg.matrix_norm(weight, ord=2).square() for weight in linear_weights(model))


class SpectralNormRegAlgorithm(BaseAlgorithm):
    name = "SPECTRAL_NORM_REG"
    formula_id = "CMNIST_SPECTRAL_NORM_REG_SUM_SIGMA1_SQUARED_V1"
    reference_id = "YOSHIDA_MIYATO_2017_SPECTRAL_NORM_REGULARIZATION"
    variant_id = "SNR_LAMBDA0P001_ALL_LINEAR_WEIGHTS"
    admission_role = "SPECTRAL_GEOMETRY"

    @property
    def penalty_weight(self) -> float:
        return float(self.config["spectral_norm_reg"]["lambda"])

    def objective(self, model: nn.Module, batches: tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]], *, step: int) -> ObjectiveParts:
        losses = torch.stack(source_environment_losses(model, batches))
        risk = losses.mean()
        l2 = weight_norm_squared(model)
        penalty = spectral_norm_regularizer(model)
        return ObjectiveParts(risk + self.l2_weight * l2 + self.penalty_weight * penalty, risk, l2, penalty, self.penalty_weight, False)

    def smooth_objective(self, model: nn.Module, worlds: SmoothWorld5, indices: tuple[Tensor, Tensor], delta: Tensor, *, step: int) -> Tensor:
        risk = smooth_source_risk_vector(model, worlds, indices, delta).mean()
        return risk + self.l2_weight * self.l2_norm(model) + self.penalty_weight * spectral_norm_regularizer(model)
