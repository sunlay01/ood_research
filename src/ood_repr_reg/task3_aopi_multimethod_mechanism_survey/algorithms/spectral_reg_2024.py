"""Lewandowski et al. 2024 spectral regularization."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ...task3_cmnist_cpu_minimal.methods import ObjectiveParts, source_environment_losses, weight_norm_squared
from ..smooth_world5 import SmoothWorld5
from .base import BaseAlgorithm, smooth_source_risk_vector


def spectral_reg_2024_penalty(model: nn.Module, *, exponent: int = 2) -> Tensor:
    total = next(model.parameters()).new_zeros(())
    for module in model.modules():
        if isinstance(module, nn.Linear):
            sigma = torch.linalg.matrix_norm(module.weight, ord=2)
            total = total + (sigma.pow(exponent) - 1.0).square()
            if module.bias is not None:
                total = total + module.bias.norm().pow(2 * exponent)
    return total


class SpectralReg2024Algorithm(BaseAlgorithm):
    name = "SPECTRAL_REG_2024"
    formula_id = "CMNIST_SPECTRAL_REG_2024_SIGMA_TARGET_ONE_V1"
    reference_id = "LEWANDOWSKI_2024_LEARNING_CONTINUALLY_BY_SPECTRAL_REGULARIZATION"
    variant_id = "SR2024_LAMBDA0P001_K2_ALL_LINEAR"
    admission_role = "SPECTRAL_GEOMETRY"

    @property
    def penalty_weight(self) -> float:
        return float(self.config["spectral_reg_2024"]["lambda"])

    @property
    def exponent(self) -> int:
        return int(self.config["spectral_reg_2024"]["exponent"])

    def objective(self, model: nn.Module, batches: tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]], *, step: int) -> ObjectiveParts:
        losses = torch.stack(source_environment_losses(model, batches))
        risk = losses.mean()
        l2 = weight_norm_squared(model)
        penalty = spectral_reg_2024_penalty(model, exponent=self.exponent)
        return ObjectiveParts(risk + self.l2_weight * l2 + self.penalty_weight * penalty, risk, l2, penalty, self.penalty_weight, False)

    def smooth_objective(self, model: nn.Module, worlds: SmoothWorld5, indices: tuple[Tensor, Tensor], delta: Tensor, *, step: int) -> Tensor:
        risk = smooth_source_risk_vector(model, worlds, indices, delta).mean()
        return risk + self.l2_weight * self.l2_norm(model) + self.penalty_weight * spectral_reg_2024_penalty(model, exponent=self.exponent)
