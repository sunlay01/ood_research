"""Explicitly deferred algorithms with unresolved common-harness fidelity."""

from __future__ import annotations

from torch import Tensor, nn

from ...task3_cmnist_cpu_minimal.methods import ObjectiveParts
from ..smooth_world5 import SmoothWorld5
from .base import BaseAlgorithm


class DeferredAlgorithm(BaseAlgorithm):
    admits_to_training = False
    admits_to_pi = False
    admission_role = "FIDELITY_UNRESOLVED"
    deferred_reason = "METHOD_FIDELITY_UNRESOLVED"

    def objective(self, model: nn.Module, batches: tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]], *, step: int) -> ObjectiveParts:
        raise RuntimeError(f"{self.name} is deferred: {self.deferred_reason}")

    def smooth_objective(self, model: nn.Module, worlds: SmoothWorld5, indices: tuple[Tensor, Tensor], delta: Tensor, *, step: int) -> Tensor:
        raise RuntimeError(f"{self.name} is deferred: {self.deferred_reason}")
