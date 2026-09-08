"""Central finite-difference checks for the round 3B order audit."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from .round3_3b_derivatives import (
    Array,
    OrderDirection,
    OrderEnvironment,
    model_response_from_liftings,
    risk_at,
    risk_derivative,
    risk_statistic_lift_derivative,
    moment_statistics,
    svec_symmetric,
)


def central_difference_samples(values: dict[int, Array | float], order: int, h: float) -> Array:
    """Apply a symmetric stencil to samples indexed by multiples of ``h``.

    The third-order stencil is
    ``[f(-2h) - 2 f(-h) + 2 f(h) - f(2h)] / (-2h**3)``;
    written this way its sign is explicit and the test polynomial ``t**3``
    returns ``6``.
    """
    if h <= 0.0:
        raise ValueError("h must be positive")
    if order == 1:
        required = (-1, 1)
        if not all(index in values for index in required):
            raise ValueError("first derivative requires -h and +h")
        return (np.asarray(values[1]) - np.asarray(values[-1])) / (2.0 * h)
    if order == 2:
        required = (-1, 0, 1)
        if not all(index in values for index in required):
            raise ValueError("second derivative requires -h, 0, and +h")
        return (np.asarray(values[1]) - 2.0 * np.asarray(values[0]) + np.asarray(values[-1])) / h**2
    if order == 3:
        required = (-2, -1, 1, 2)
        if not all(index in values for index in required):
            raise ValueError("third derivative requires -2h, -h, +h, and +2h")
        return (
            np.asarray(values[2])
            - 2.0 * np.asarray(values[1])
            + 2.0 * np.asarray(values[-1])
            - np.asarray(values[-2])
        ) / (2.0 * h**3)
    if order == 4:
        required = (-2, -1, 0, 1, 2)
        if not all(index in values for index in required):
            raise ValueError("fourth derivative requires -2h, -h, 0, +h, and +2h")
        return (
            np.asarray(values[-2])
            - 4.0 * np.asarray(values[-1])
            + 6.0 * np.asarray(values[0])
            - 4.0 * np.asarray(values[1])
            + np.asarray(values[2])
        ) / h**4
    raise ValueError("supported finite-difference orders are 1, 2, 3, and 4")


def _risk_samples(weights: Array, base: OrderEnvironment, direction: OrderDirection, h: float, order: int) -> dict[int, float]:
    offsets = {-1, 1} if order == 1 else {-1, 0, 1} if order == 2 else {-2, -1, 1, 2} if order == 3 else {-2, -1, 0, 1, 2}
    return {offset: risk_at(weights, base.at(direction, offset * h)) for offset in offsets}


def finite_difference_risk_derivative(
    weights: Array,
    base: OrderEnvironment,
    direction: OrderDirection,
    h: float,
    order: int,
) -> float:
    """Estimate one directional risk derivative from exact population risk."""
    return float(central_difference_samples(_risk_samples(weights, base, direction, h, order), order, h))


def response_matrix_at(
    weights: Iterable[Array],
    base: OrderEnvironment,
    directions: Iterable[OrderDirection],
    t: float,
) -> Array:
    """Return model-by-direction finite risk responses at path time ``t``."""
    weights = tuple(np.asarray(value, dtype=float) for value in weights)
    directions = tuple(directions)
    phi = np.asarray([np.concatenate((svec_symmetric(np.outer(model, model)), -2.0 * model, [1.0])) for model in weights])
    base_stats = moment_statistics(base)
    base_lift = np.concatenate((svec_symmetric(base_stats[0]), base_stats[1], [base_stats[2]]))
    columns = []
    for direction in directions:
        point = base.at(direction, t)
        stats = moment_statistics(point)
        point_lift = np.concatenate((svec_symmetric(stats[0]), stats[1], [stats[2]]))
        columns.append(phi @ (point_lift - base_lift))
    return np.column_stack(columns) if columns else np.zeros((len(weights), 0), dtype=float)


def finite_difference_response_matrix(
    weights: Iterable[Array],
    base: OrderEnvironment,
    directions: Iterable[OrderDirection],
    h: float,
    order: int,
) -> Array:
    """Return an ``n_models x n_directions`` central-difference matrix."""
    weights = tuple(np.asarray(value, dtype=float) for value in weights)
    directions = tuple(directions)
    if not directions:
        return np.zeros((len(weights), 0), dtype=float)
    response = {offset: response_matrix_at(weights, base, directions, offset * h) for offset in {-2, -1, 0, 1, 2}}
    return central_difference_samples(response, order, h)


def finite_response_at_one(
    weights: Iterable[Array],
    base: OrderEnvironment,
    directions: Iterable[OrderDirection],
) -> Array:
    """Return finite path responses ``R(1)-R(0)`` for a direction set."""
    return response_matrix_at(weights, base, directions, 1.0)


def analytic_response_matrix(
    weights: Iterable[Array],
    base: OrderEnvironment,
    directions: Iterable[OrderDirection],
    order: int,
) -> Array:
    """Return the analytic model-by-direction derivative matrix."""
    weights = tuple(np.asarray(value, dtype=float) for value in weights)
    directions = tuple(directions)
    if not directions:
        return np.zeros((len(weights), 0), dtype=float)
    liftings = np.asarray([risk_statistic_lift_derivative(base, direction, order) for direction in directions], dtype=float)
    return model_response_from_liftings(weights, liftings)


def validation_error(analytic: Array, estimate: Array, epsilon: float = 1e-14) -> float:
    analytic = np.asarray(analytic, dtype=float)
    estimate = np.asarray(estimate, dtype=float)
    scale = np.linalg.norm(analytic)
    error = np.linalg.norm(estimate - analytic)
    # A relative error is undefined for an analytically zero derivative.  In
    # that case report the absolute numerical residue instead of amplifying
    # round-off by dividing by epsilon.
    return float(error if scale <= epsilon else error / scale)


def step_size_convergence(
    weights: Iterable[Array],
    base: OrderEnvironment,
    directions: Iterable[OrderDirection],
    order: int,
    h_values: Iterable[float],
    epsilon: float = 1e-14,
) -> list[dict[str, float]]:
    """Compare ``D_h`` with ``D_{h/2}`` at adjacent requested scales."""
    values = tuple(float(h) for h in h_values)
    if any(h <= 0.0 for h in values):
        raise ValueError("h_values must be positive")
    estimates = {
        h: finite_difference_response_matrix(weights, base, directions, h, order)
        for h in values
    }
    rows: list[dict[str, float]] = []
    for larger, smaller in zip(values[1:], values[:-1]):
        numerator = np.linalg.norm(estimates[larger] - estimates[smaller])
        denominator = np.linalg.norm(estimates[smaller]) + epsilon
        rows.append({"order": order, "h": larger, "h_half": smaller, "relative_change": float(numerator / denominator)})
    return rows
