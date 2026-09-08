"""Explicit 3D exposure/identifiability counterexamples."""

from __future__ import annotations

import numpy as np

from .round3r_3b_benchmark import ModuleEnvironment, environment_state, source_design, source_optimum
from .round3r_3d_exposure import response_operator, source_design_span, exposed_response_basis
from .round3r_3d_identifiability import hidden_emergent_world_pair


def source_identical_target_different(base: ModuleEnvironment | None = None) -> dict[str, object]:
    base = ModuleEnvironment(n_noise=4) if base is None else base
    environments = source_design(base, relation_exposed=True)
    return hidden_emergent_world_pair(base, environments)


def design_unexposed_but_source_visible(base: ModuleEnvironment | None = None) -> dict[str, object]:
    base = ModuleEnvironment(n_noise=4) if base is None else base
    environments = source_design(base, relation_exposed=False)
    optimum, source = source_optimum(environments)
    operator = response_operator(source, optimum)
    design = source_design_span(tuple(environment_state(environment) for environment in environments))
    total_environment = environment_state(base.updated(u_gamma=0.4))
    response = response_operator(source, optimum, total_environment)
    return {"design_exposure_rank": int(exposed_response_basis(design["contrasts"], operator)["rank"]),
            "target_response_norm": float(np.linalg.norm(response)),
            "target_response": response,
            "description": "a target-relevant direction is not in the chosen source contrast design"}


__all__ = ["source_identical_target_different", "design_unexposed_but_source_visible"]
