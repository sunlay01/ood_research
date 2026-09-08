"""Optimal source-only recovery geometry for round 3E.

The primary object is the pair of finite-dimensional linear maps
``O: U -> Y`` and ``A: U -> R``.  All norms are Euclidean.  The numerical
implementation uses an orthonormal basis of ``ker(O)`` rather than forming a
dense projector as its main path.
"""

from __future__ import annotations

import numpy as np

Array = np.ndarray


def _matrix(value: Array, *, name: str) -> Array:
    result = np.asarray(value, dtype=float)
    if result.ndim != 2:
        raise ValueError(f"{name} must be a matrix")
    return result


def _check_pair(response: Array, observation: Array) -> tuple[Array, Array]:
    a = _matrix(response, name="response")
    o = _matrix(observation, name="observation")
    if a.shape[1] != o.shape[1]:
        raise ValueError("response and observation must share the world dimension")
    return a, o


def _rank(singular: Array, tolerance: float) -> int:
    if singular.size == 0 or singular[0] <= 0.0:
        return 0
    return int(np.sum(singular > tolerance * singular[0]))


def nullspace_basis(observation: Array, tolerance: float = 1e-10) -> Array:
    """Return an orthonormal basis ``N`` for ``ker(observation)``."""
    o = _matrix(observation, name="observation")
    _, singular, vh = np.linalg.svd(o, full_matrices=True)
    rank = _rank(singular, tolerance)
    return vh[rank:].T.copy()


def orthogonal_projector_from_basis(null_basis: Array) -> Array:
    """Return ``N N.T`` for an orthonormal or numerically orthonormal basis."""
    n = np.asarray(null_basis, dtype=float)
    if n.ndim != 2:
        raise ValueError("null_basis must be a matrix")
    return n @ n.T


def response_rank(response: Array, tolerance: float = 1e-10) -> int:
    singular = np.linalg.svd(_matrix(response, name="response"), compute_uv=False)
    return _rank(singular, tolerance)


def operator_norm(matrix: Array) -> float:
    """Euclidean operator norm of a finite matrix."""
    singular = np.linalg.svd(_matrix(matrix, name="matrix"), compute_uv=False)
    return float(singular[0]) if singular.size else 0.0


def irreducible_response_operator(response: Array, observation: Array,
                                  tolerance: float = 1e-10) -> Array:
    """Return ``A P_ker(O)`` using the null-space basis path."""
    a, o = _check_pair(response, observation)
    return a @ orthogonal_projector_from_basis(nullspace_basis(o, tolerance))


def minimax_recovery_error(response: Array, observation: Array,
                           tolerance: float = 1e-10) -> float:
    """Compute the exact finite-dimensional minimax value ``||A P_ker(O)||``."""
    a, o = _check_pair(response, observation)
    n = nullspace_basis(o, tolerance)
    if n.shape[1] == 0:
        return 0.0
    singular = np.linalg.svd(a @ n, compute_uv=False)
    return float(singular[0]) if singular.size else 0.0


def normalized_nonidentifiability(response: Array, observation: Array,
                                  tolerance: float = 1e-10) -> float:
    """Return ``alpha / ||A||``; zero response maps are assigned zero."""
    a = _matrix(response, name="response")
    denominator = np.linalg.svd(a, compute_uv=False)
    scale = float(denominator[0]) if denominator.size else 0.0
    return minimax_recovery_error(a, observation, tolerance) / scale if scale > 0 else 0.0


def pseudoinverse_recoverer(response: Array, observation: Array) -> Array:
    """Return the matrix of ``Phi*(y) = A O^dagger y``."""
    a, o = _check_pair(response, observation)
    return a @ np.linalg.pinv(o)


def pseudoinverse_residual(response: Array, observation: Array, world: Array) -> Array:
    """Evaluate ``Au - Phi*(Ou)`` for a world vector ``u``."""
    a, o = _check_pair(response, observation)
    u = np.asarray(world, dtype=float)
    if u.ndim != 1 or u.size != o.shape[1]:
        raise ValueError("world has incompatible shape")
    return a @ u - pseudoinverse_recoverer(a, o) @ (o @ u)


def recoverable_operator(response: Array, observation: Array) -> Array:
    """Return ``A P_(ker O)^perp = A O^dagger O``."""
    a, o = _check_pair(response, observation)
    return a @ np.linalg.pinv(o) @ o


def coordinate_transformed_pair(response: Array, observation: Array,
                                source_change: Array | None = None,
                                world_isometry: Array | None = None,
                                response_isometry: Array | None = None) -> tuple[Array, Array]:
    """Apply consistent optional coordinate changes to a pair ``(A, O)``.

    ``source_change`` is an invertible recoding of observations.  The other
    two maps are required to be isometries when supplied; the function checks
    this numerically because the recovery metric is part of the theorem.
    """
    a, o = _check_pair(response, observation)
    if source_change is not None:
        l = _matrix(source_change, name="source_change")
        if l.shape != (o.shape[0], o.shape[0]) or abs(np.linalg.det(l)) <= 1e-12:
            raise ValueError("source_change must be invertible on source coordinates")
        o = l @ o
    if world_isometry is not None:
        u = _matrix(world_isometry, name="world_isometry")
        if u.shape != (a.shape[1], a.shape[1]) or not np.allclose(u.T @ u, np.eye(u.shape[0]), atol=1e-8):
            raise ValueError("world_isometry must be square and orthogonal")
        a, o = a @ u, o @ u
    if response_isometry is not None:
        v = _matrix(response_isometry, name="response_isometry")
        if v.shape != (a.shape[0], a.shape[0]) or not np.allclose(v.T @ v, np.eye(v.shape[0]), atol=1e-8):
            raise ValueError("response_isometry must be square and orthogonal")
        a = v @ a
    return a, o


def ambiguity_diameter(response: Array, observation: Array,
                       tolerance: float = 1e-10) -> float:
    """Return the source-equivalent unit-ball diameter ``2 alpha``."""
    return 2.0 * minimax_recovery_error(response, observation, tolerance)


def factorization_diagnostic(response: Array, observation: Array,
                             tolerance: float = 1e-10) -> dict[str, object]:
    """Report the zero-error/kernel/factorization equivalence numerically."""
    a, o = _check_pair(response, observation)
    n = nullspace_basis(o, tolerance)
    alpha = minimax_recovery_error(a, o, tolerance)
    factor = pseudoinverse_recoverer(a, o)
    residual = float(np.linalg.norm(a - factor @ o, ord="fro"))
    kernel_action = float(np.linalg.norm(a @ n, ord="fro")) if n.shape[1] else 0.0
    scale = max(1.0, float(np.linalg.norm(a, ord="fro")))
    return {
        "alpha": alpha,
        "kernel_dimension": int(n.shape[1]),
        "kernel_action_frobenius": kernel_action,
        "factorization_residual": residual,
        "zero_error": bool(alpha <= tolerance * scale),
        "kernel_inclusion": bool(kernel_action <= tolerance * scale),
        "factorization_exists_numerically": bool(residual <= tolerance * scale),
        "rank_observation": int(np.linalg.matrix_rank(o, tol=tolerance)),
        "rank_response": response_rank(a, tolerance),
        "tolerance": float(tolerance),
    }


def structural_ambiguity_dimension(response: Array, observation: Array,
                                   tolerance: float = 1e-10) -> int:
    """Dimension of ``ker(O)/(ker(O) intersection ker(A))``."""
    a, o = _check_pair(response, observation)
    n = nullspace_basis(o, tolerance)
    if n.shape[1] == 0:
        return 0
    visible_rank = response_rank(a @ n, tolerance)
    return visible_rank


def operator_summary(response: Array, observation: Array,
                    tolerance: float = 1e-10) -> dict[str, object]:
    a, o = _check_pair(response, observation)
    factor = factorization_diagnostic(a, o, tolerance)
    alpha = float(factor["alpha"])
    norm_a = float(np.linalg.svd(a, compute_uv=False)[0]) if a.size else 0.0
    return {
        "dim_U": int(o.shape[1]),
        "source_dimension": int(o.shape[0]),
        "response_dimension": int(a.shape[0]),
        "rank_O": int(factor["rank_observation"]),
        "kernel_dimension": int(factor["kernel_dimension"]),
        "rank_A": int(factor["rank_response"]),
        "structural_ambiguity_dimension": int(structural_ambiguity_dimension(a, o, tolerance)),
        "alpha": alpha,
        "normalized_alpha": alpha / norm_a if norm_a > 0 else 0.0,
        "ambiguity_diameter": 2.0 * alpha,
        "pseudoinverse_attainment_error": float(np.linalg.norm(
            a @ (np.eye(o.shape[1]) - np.linalg.pinv(o) @ o)
            - irreducible_response_operator(a, o, tolerance), ord="fro")),
        "factorization": factor,
        "tolerance": float(tolerance),
    }


__all__ = [
    "nullspace_basis", "orthogonal_projector_from_basis", "response_rank",
    "operator_norm",
    "irreducible_response_operator", "minimax_recovery_error",
    "normalized_nonidentifiability", "pseudoinverse_recoverer",
    "pseudoinverse_residual", "ambiguity_diameter", "factorization_diagnostic",
    "recoverable_operator", "coordinate_transformed_pair",
    "structural_ambiguity_dimension", "operator_summary",
]
