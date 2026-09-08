"""Build the common family-relative task/source geometry."""

from __future__ import annotations

import numpy as np

from ..round3r_3b_benchmark import environment_state, source_optimum
from ..round3r_3d_exposure import response_operator
from ..round3r_3d_state import state_difference, task_state
from .base import Environment, EnvironmentFamily, FamilyTaskGeometry, TangentSpec

Array = np.ndarray


def source_observation_operator(family: EnvironmentFamily,
                                environments: tuple[Environment, ...],
                                spec: TangentSpec | None = None,
                                *, step: float | None = None,
                                reference: Environment | None = None) -> Array:
    reference = family.reference_environment() if reference is None else reference
    tangent = family.tangent_spec(reference) if spec is None else spec
    h = tangent.finite_difference_step if step is None else float(step)
    if h <= 0.0:
        raise ValueError("step must be positive")
    blocks: list[Array] = []
    for environment in environments:
        if tangent.dimension == 0:
            blocks.append(np.zeros((task_state(environment_state(environment)).size, 0)))
            continue
        columns = []
        for direction in tangent.directions:
            plus = task_state(environment_state(family.perturb(
                environment, direction, h, role="source")))
            minus = task_state(environment_state(family.perturb(
                environment, direction, -h, role="source")))
            columns.append((plus - minus) / (2.0 * h))
        blocks.append(np.column_stack(columns))
    return np.vstack(blocks) if blocks else np.zeros((0, tangent.dimension))


def response_operator_for_family(
    family: EnvironmentFamily,
    source_state: object,
    source_optimum: Array,
    spec: TangentSpec | None = None,
    *,
    step: float | None = None,
    reference: Environment | None = None,
) -> Array:
    reference = family.reference_environment() if reference is None else reference
    tangent = family.tangent_spec(reference) if spec is None else spec
    h = tangent.finite_difference_step if step is None else float(step)
    if h <= 0.0:
        raise ValueError("step must be positive")
    columns = []
    for direction in tangent.directions:
        plus = response_operator(
            source_state, source_optimum,
            environment_state(family.perturb(reference, direction, h, role="target")),
        )
        minus = response_operator(
            source_state, source_optimum,
            environment_state(family.perturb(reference, direction, -h, role="target")),
        )
        columns.append((plus - minus) / (2.0 * h))
    return np.column_stack(columns) if columns else np.zeros((source_optimum.size, 0))


def build_task_geometry(
    family: EnvironmentFamily,
    reference: Environment | None = None,
    source_environments: tuple[Environment, ...] | None = None,
    *,
    step: float | None = None,
) -> FamilyTaskGeometry:
    reference = family.reference_environment() if reference is None else reference
    spec = family.tangent_spec(reference)
    environments = family.source_environments(reference) if source_environments is None else tuple(source_environments)
    optimum, source_state = source_optimum(environments)
    observation = source_observation_operator(
        family, environments, spec, step=step, reference=reference,
    )
    response = response_operator_for_family(
        family, source_state, optimum, spec, step=step, reference=reference,
    )
    return FamilyTaskGeometry(
        family=family, reference=reference, spec=spec,
        source_environments=environments, source_state=source_state,
        source_optimum=optimum, observation=observation, response=response,
    )


def family_response_factorization_residual(geometry: FamilyTaskGeometry,
                                           direction: str | None = None) -> float:
    """Check the source-state response operator on one declared family direction."""
    direction = geometry.spec.directions[0] if direction is None else direction
    geometry.spec.index(direction)
    target = environment_state(geometry.family.perturb(
        geometry.reference, direction, 1.0, role="target"))
    direct = response_operator(geometry.source_state, geometry.source_optimum, target)
    delta = state_difference(target, geometry.source_state)
    operator = response_operator(geometry.source_state, geometry.source_optimum)
    return float(np.linalg.norm(direct - operator @ delta))


__all__ = [
    "source_observation_operator", "response_operator_for_family",
    "build_task_geometry", "family_response_factorization_residual",
]
