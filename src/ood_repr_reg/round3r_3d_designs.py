"""Pre-registered source-design ladder for 3D."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .round3r_3b_benchmark import ModuleEnvironment, environment_state, source_design
from .round3r_3d_exposure import exposed_response_basis, source_design_span
from .round3r_3d_state import task_state


@dataclass(frozen=True)
class SourceDesign:
    name: str
    environments: tuple[ModuleEnvironment, ...]
    note: str


def _relation_pair(base: ModuleEnvironment, index: int) -> tuple[ModuleEnvironment, ModuleEnvironment]:
    plus = list(base.shortcut_rhos)
    minus = list(base.shortcut_rhos)
    plus[index] += 0.12
    minus[index] -= 0.12
    return base.updated(shortcut_rhos=tuple(plus)), base.updated(shortcut_rhos=tuple(minus))


def source_design_ladder(base: ModuleEnvironment) -> tuple[SourceDesign, ...]:
    """Return source designs in a fixed order, with no target inputs."""
    pair1 = _relation_pair(base, 0)
    pair2 = _relation_pair(base, 1 if len(base.shortcut_rhos) > 1 else 0)
    null_noise = base.updated(noise_variances=np.asarray(base.noise_variances, dtype=float) + np.array([0.7] + [0.0] * (base.n_noise - 1)))
    emergent = base.updated(u_gamma=0.45)
    return (
        SourceDesign("single_source", (base,), "single source environment"),
        SourceDesign("one_relation_contrast", (base, *pair1), "one independent relation contrast"),
        SourceDesign("two_shortcut_contrasts", (base, *pair1, *pair2), "two independent shortcut contrasts"),
        SourceDesign("duplicate_source", (base, base, *pair1), "duplicate source row"),
        SourceDesign("risk_null_nuisance", (base, *pair1, null_noise), "source variation in an independent noise variance"),
        SourceDesign("new_target_relevant_contrast", (base, *pair1, emergent), "new U-Y relevant contrast"),
        SourceDesign("redundant_shortcut_copies", (base, *pair1, pair1[0], pair1[1]), "repeated shortcut contrast copies"),
        SourceDesign("independent_nuisance_dimensions", (base, *pair1, null_noise), "additional nuisance-state diversity"),
    )


def design_diagnostics(design: SourceDesign, source_optimum: np.ndarray,
                       total_response_basis: np.ndarray, response_map: np.ndarray,
                       tolerance: float = 1e-9) -> dict[str, object]:
    states = tuple(environment_state(environment) for environment in design.environments)
    state_span = source_design_span(states, 0, tolerance)
    exposure = exposed_response_basis(state_span["contrasts"], response_map, tolerance)
    total_basis = np.asarray(total_response_basis, dtype=float)
    singular_values = np.linalg.svd(np.asarray(exposure["responses"], dtype=float), compute_uv=False)
    if singular_values.size:
        positive = singular_values[singular_values > tolerance * singular_values[0]]
        condition = float(positive[0] / positive[-1]) if positive.size else float("inf")
    else:
        condition = float("inf")
    return {
        "name": design.name,
        "note": design.note,
        "n_source": len(states),
        "ambient_state_rank": int(np.linalg.matrix_rank(np.column_stack([task_state(state) for state in states]), tol=tolerance)),
        "source_contrast_rank": int(state_span["rank"]),
        "exposed_response_rank": int(exposure["rank"]),
        "total_response_rank": int(np.asarray(total_basis).shape[1]),
        "unexposed_quotient_dimension": int(np.asarray(total_basis).shape[1] - exposure["rank"]),
        "singular_values": singular_values,
        "condition_number": condition,
        "tolerance_profile": {"relative": tolerance, "rank_rule": "singular > relative * max singular"},
        "source_optimum_dimension": int(np.asarray(source_optimum).size),
    }


__all__ = ["SourceDesign", "source_design_ladder", "design_diagnostics"]
