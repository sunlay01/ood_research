"""IRMv1 algorithm definition for the multi-method CMNIST survey."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ...task3_cmnist_cpu_minimal.methods import ObjectiveParts, method_objective as canonical_method_objective
from ..smooth_world5 import SmoothWorld5, environment_parameters, outcome_weight
from .base import BaseAlgorithm, smooth_source_risk_vector


class IRMv1Algorithm(BaseAlgorithm):
    name = "IRMv1"
    formula_id = "CPU_MINIMAL_IRMV1_V1"

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
        risk_vector = smooth_source_risk_vector(model, worlds, indices, delta)
        risk = risk_vector.mean()
        penalties = []
        for environment, (pool, index) in enumerate(zip(worlds.source, indices)):
            p, q = environment_parameters(delta, environment=environment, evaluation=False)
            scale = torch.ones((), device=next(model.parameters()).device, requires_grad=True)
            weighted = []
            for images, labels, label_flip, color_flip in pool.outcome_batches(index):
                loss = F.binary_cross_entropy_with_logits(model(images) * scale, labels.float())
                weighted.append(outcome_weight(p, q, label_flip, color_flip).to(loss) * loss)
            penalty_gradient = torch.autograd.grad(torch.stack(weighted).sum(), scale, create_graph=True)[0]
            penalties.append(penalty_gradient.square())
        applied = float(
            self.config["irmv1"]["penalty_weight"]
            if step >= int(self.config["irmv1"]["penalty_anneal_iters"])
            else 1.0
        )
        objective = risk + self.l2_weight * self.l2_norm(model) + applied * torch.stack(penalties).mean()
        return objective / applied if applied > 1.0 else objective
