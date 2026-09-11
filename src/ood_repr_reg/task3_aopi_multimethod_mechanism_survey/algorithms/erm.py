"""ERM algorithm definition for the multi-method CMNIST survey."""

from __future__ import annotations

from torch import Tensor, nn

from ...task3_cmnist_cpu_minimal.methods import ObjectiveParts, method_objective as canonical_method_objective
from ..smooth_world5 import SmoothWorld5
from .base import BaseAlgorithm, smooth_source_risk_vector


class ERMAlgorithm(BaseAlgorithm):
    name = "ERM"
    formula_id = "CPU_MINIMAL_ERM_V1"

    def objective(
        self,
        model: nn.Module,
        batches: tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]],
        *,
        step: int,
    ) -> ObjectiveParts:
        return canonical_method_objective(self.name, model, batches, step=step, config=self.config)

    def smooth_objective(
        self,
        model: nn.Module,
        worlds: SmoothWorld5,
        indices: tuple[Tensor, Tensor],
        delta: Tensor,
        *,
        step: int,
    ) -> Tensor:
        risk = smooth_source_risk_vector(model, worlds, indices, delta).mean()
        return risk + self.l2_weight * self.l2_norm(model)
