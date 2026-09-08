"""Induced regularizer costs and failure diagnostics for round two."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .intervention_linear import GaussianEnvironment, LinearGaussianSCM, scalar_relation_design

Array = np.ndarray


@dataclass(frozen=True)
class InducedCost:
    method: str
    value: float
    status: str
    direct_descent: bool
    notes: str


def l2_induced_cost(q: Array, beta: Array, tolerance: float = 1e-10) -> InducedCost:
    """Closed form L2 cost for a rank-one quadratic state Q=q q.T."""
    q = np.asarray(q, dtype=float)
    beta = np.asarray(beta, dtype=float)
    if q.ndim == 2:
        if q.shape[0] != q.shape[1]:
            raise ValueError("Q must be square")
        eigenvalues, eigenvectors = np.linalg.eigh(q)
        if np.count_nonzero(eigenvalues > tolerance) != 1 or np.min(eigenvalues) < -tolerance:
            raise ValueError("Q must be positive semidefinite rank one")
        q = eigenvectors[:, int(np.argmax(eigenvalues))] * np.sqrt(max(eigenvalues.max(), 0.0))
    if q.ndim != 1 or beta.ndim != 1 or q.size < beta.size:
        raise ValueError("q and beta must be matching vectors")
    if np.linalg.norm(np.outer(q, q), ord="fro") <= tolerance:
        value = float(beta @ beta)
    else:
        value = float(q @ q + beta @ beta - 2.0 * abs(q[: beta.size] @ beta))
    return InducedCost("l2", value, "exact", False, "nonselective parameter magnitude")


def irmv1_direct_induced_cost(
    q: Array,
    beta: float,
    relations: Array,
    nuisance_variance: float,
) -> InducedCost:
    """Exact scalar-scale IRMv1 infimum over the two rank-one signs."""
    q = np.asarray(q, dtype=float)
    relations = np.asarray(relations, dtype=float)
    if q.shape != (2,) or relations.ndim != 1 or relations.size == 0 or nuisance_variance < 0:
        raise ValueError("direct IRMv1 inputs have incompatible shapes")
    values = []
    for sign in (-1.0, 1.0):
        w = np.array([beta, 0.0]) + sign * q
        derivatives = []
        for relation in relations:
            sigma = np.array([[1.0, relation], [relation, relation * relation + nuisance_variance]])
            cross = np.array([beta, relation * beta])
            derivatives.append(2.0 * (w @ sigma @ w - w @ cross))
        values.append(float(np.mean(np.square(derivatives))))
    return InducedCost("irmv1", min(values), "exact_fiber_infimum", False, "direct predictor sign fiber")


def coral_induced_cost(
    risk_state: Array,
    source_covariances: tuple[Array, ...],
    gauge: float | None = None,
) -> InducedCost:
    """Analyze CORAL's representation scaling fiber.

    For any realization with nonzero CORAL value, scaling B by c and the head
    by 1/c preserves the predictor while scaling CORAL by c^4.  ``scale`` is
    an optional fixed gauge used only for a diagnostic finite representative.
    """
    state = np.asarray(risk_state, dtype=float)
    if state.ndim not in (1, 2):
        raise ValueError("risk_state must be a vector or matrix")
    if len(source_covariances) < 2:
        raise ValueError("at least two source covariances are required")
    covariances = [np.asarray(covariance, dtype=float) for covariance in source_covariances]
    if any(covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1] for covariance in covariances):
        raise ValueError("source covariances must be square")
    base = float(sum(np.linalg.norm(covariances[i] - covariances[j], ord="fro") ** 2 for i in range(len(covariances)) for j in range(i + 1, len(covariances))))
    if gauge is None:
        return InducedCost("coral", 0.0, "degenerate_infimum", False, "representation rescaling drives c^4 CORAL cost to zero")
    if gauge <= 0:
        raise ValueError("gauge must be positive")
    return InducedCost("coral", float(gauge**4 * base), "conditional", False, "covariance-only representation cost under fixed scale")


def coral_scaling_law(base_penalty: float, scale: float) -> float:
    """Return CORAL's exact c^4 scaling for a fixed representation fiber."""
    if base_penalty < 0 or scale <= 0:
        raise ValueError("base_penalty must be nonnegative and scale positive")
    return float(scale**4 * base_penalty)


def irmv1_induced_cost(
    scm: LinearGaussianSCM,
    environments: tuple[GaussianEnvironment, ...],
    w: Array,
) -> InducedCost:
    """Return the exact scalar-scale IRMv1 value for a predictor realization."""
    from .intervention_linear import irmv1_scale_penalty

    return InducedCost("irmv1", irmv1_scale_penalty(scm, environments, w), "conditional", False, "radial environment response; quotient fiber not yet minimized")


def irm_relation_operator(environments: tuple[GaussianEnvironment, ...]) -> Array:
    """Vandermonde operator for the scalar relation response quotient."""
    return scalar_relation_design(np.array([environment.relation[0, 0] for environment in environments]))


def classify_failure(
    *,
    source_unidentified: bool,
    regularizer_blind: bool,
    destructive: bool,
    parameterization_degenerate: bool = False,
) -> str:
    """Use explicit precedence while retaining multiple flags upstream."""
    if parameterization_degenerate:
        return "quotient-external/parameterization-degeneracy"
    if destructive:
        return "destructive"
    if regularizer_blind:
        return "regularizer-blind"
    if source_unidentified:
        return "source-unidentified"
    return "none-detected"
