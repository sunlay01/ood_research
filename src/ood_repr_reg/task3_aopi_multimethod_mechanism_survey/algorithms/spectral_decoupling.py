"""Spectral Decoupling (logit penalty) for ColoredMNIST."""
from __future__ import annotations
import torch
from torch import Tensor, nn
from ...task3_cmnist_cpu_minimal.methods import ObjectiveParts, source_environment_losses, weight_norm_squared
from ..smooth_world5 import SmoothWorld5
from .base import BaseAlgorithm, smooth_source_risk_vector

def logit_l2_penalty(model: nn.Module, batches) -> Tensor:
    return torch.stack([model(x).square().mean() for x, _ in batches]).mean()

class SpectralDecouplingAlgorithm(BaseAlgorithm):
    name = "SPECTRAL_DECOUPLING"
    formula_id = "CMNIST_SPECTRAL_DECOUPLING_LOGIT_L2_V1"
    reference_id = "PEZESHKI_2021_GRADIENT_STARVATION"
    variant_id = "SD_LAMBDA0P1_LOGIT_L2"
    admission_role = "FEATURE_LEARNING"
    @property
    def penalty_weight(self) -> float:
        return float(self.config.get("spectral_decoupling", {}).get("lambda", 0.1))
    def objective(self, model, batches, *, step):
        losses = torch.stack(source_environment_losses(model, batches)); risk = losses.mean()
        l2 = weight_norm_squared(model); penalty = logit_l2_penalty(model, batches)
        return ObjectiveParts(risk + self.l2_weight*l2 + self.penalty_weight*penalty, risk, l2, penalty, self.penalty_weight, False)
    def smooth_objective(self, model, worlds: SmoothWorld5, indices, delta, *, step):
        return smooth_source_risk_vector(model, worlds, indices, delta).mean() + self.l2_weight*self.l2_norm(model)
