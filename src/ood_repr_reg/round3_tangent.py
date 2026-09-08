"""Mechanism tangent and risk-response calculations for round three."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .round3_mechanisms import (
    Array,
    MechanismDirection,
    StructuralMechanism,
    add_direction,
    directional_moment_matrix,
    risk,
    risk_response,
)


@dataclass(frozen=True)
class TangentResponse:
    name: str
    analytic: float
    finite_difference: float
    absolute_error: float
    moment_direction: Array


def tangent_response(
    beta: Array,
    w_c: Array,
    w_a: Array,
    mechanism: StructuralMechanism,
    direction: MechanismDirection,
    step: float = 1e-6,
) -> TangentResponse:
    analytic = risk_response(beta, w_c, w_a, mechanism, direction)
    plus = risk(beta, w_c, w_a, add_direction(mechanism, direction, step))
    minus = risk(beta, w_c, w_a, add_direction(mechanism, direction, -step))
    finite_difference = (plus - minus) / (2.0 * step)
    return TangentResponse(direction.name, analytic, finite_difference, abs(analytic - finite_difference), directional_moment_matrix(mechanism, direction))


def tangent_operator(mechanism: StructuralMechanism, directions: tuple[MechanismDirection, ...]) -> Array:
    """Stack vec(DM[d]) columns; rank exposes mechanism-response overlap."""
    if not directions:
        raise ValueError("at least one mechanism direction is required")
    matrices = [directional_moment_matrix(mechanism, direction).reshape(-1) for direction in directions]
    return np.stack(matrices, axis=1)


def response_profile(
    beta: Array,
    w_c: Array,
    w_a: Array,
    mechanism: StructuralMechanism,
    directions: dict[str, MechanismDirection],
) -> dict[str, float]:
    return {name: risk_response(beta, w_c, w_a, mechanism, direction) for name, direction in directions.items()}


def dual_sensitivity(
    beta: Array,
    w_c: Array,
    w_a: Array,
    mechanism: StructuralMechanism,
    directions: tuple[MechanismDirection, ...],
) -> float:
    values = [abs(risk_response(beta, w_c, w_a, mechanism, direction)) for direction in directions]
    return float(max(values, default=0.0))
