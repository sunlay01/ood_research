"""Residual-coupling identities for the relevance benchmark."""

from __future__ import annotations

import numpy as np

from .round3r_3a_benchmark import Array, RelevanceEnvironment
from .round3r_3a_source_geometry import PopulationMoments, feature_residual_coupling, risk_difference_gradient


def target_residual_coupling(
    optimum: Array,
    source: PopulationMoments,
    target: PopulationMoments,
) -> Array:
    """Return ``E_T[X (Y - X'w*)]``."""
    del source  # The source normal equation is used by the identity, not here.
    return feature_residual_coupling(optimum, target)


def gradient_from_residual(
    optimum: Array,
    source: PopulationMoments,
    target: PopulationMoments,
) -> Array:
    """Compute ``g_s = -2 E_T[X r*]``."""
    return -2.0 * target_residual_coupling(optimum, source, target)


def analytic_gradient(
    optimum: Array,
    source: PopulationMoments,
    target: PopulationMoments,
) -> Array:
    return risk_difference_gradient(optimum, source, target)


def residual_identity_error(
    optimum: Array,
    source: PopulationMoments,
    target: PopulationMoments,
) -> float:
    return float(np.linalg.norm(analytic_gradient(optimum, source, target) - gradient_from_residual(optimum, source, target)))
