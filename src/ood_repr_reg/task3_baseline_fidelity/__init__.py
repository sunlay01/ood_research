"""Task 3 baseline-fidelity recovery gate.

This package is intentionally separate from the Task 3 local-response method.
It only checks whether ERM, IRMv1, IGA, and Fish baselines are implemented and
reported with faithful upstream identities before any Task 3 scientific verdict
is interpreted.
"""

from .official_irm import OfficialIRMConfig, run_official_irm_reimplementation
from .ports import (
    BaselineMLP,
    fish_outer_update,
    head_gradient_variance_surrogate_notice,
    iga_objective,
    irmv1_objective,
)

__all__ = [
    "BaselineMLP",
    "OfficialIRMConfig",
    "fish_outer_update",
    "head_gradient_variance_surrogate_notice",
    "iga_objective",
    "irmv1_objective",
    "run_official_irm_reimplementation",
]
