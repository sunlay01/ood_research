"""Post-hoc intersection of frozen 3D exposure with 3C control geometry."""

from __future__ import annotations

import numpy as np

from .round3r_3b_geometry import GeometryTolerance, matrix_rank, subspace_basis
from .round3r_3c_benchmark import make_benchmark
from .round3r_3c_regularizers import regularizer_state


def posthoc_regularizer_intersection(exposed_basis: np.ndarray, benchmark=None,
                                     tolerance: float = 1e-9) -> list[dict[str, object]]:
    """Load 3C only after exposure basis is supplied and frozen."""
    benchmark = make_benchmark() if benchmark is None else benchmark
    exposed = subspace_basis(np.asarray(exposed_basis, dtype=float))
    rows = []
    for method in ("l2", "coral", "irmv1", "vrex"):
        state = regularizer_state(method, benchmark)
        hessian = benchmark.hessian
        values, vectors = np.linalg.eigh((hessian + hessian.T) / 2.0)
        root = vectors @ np.diag(1.0 / np.sqrt(values)) @ vectors.T
        k = root @ state.hessian @ root
        image = k @ exposed
        controlled = matrix_rank(image, tolerance=GeometryTolerance(rank_relative=tolerance)) if image.shape[1] else 0
        rank = int(exposed.shape[1])
        complement = np.eye(rank) - exposed.T @ exposed if rank else np.zeros((0, 0))
        rows.append({
            "method": method.upper(),
            "exposed_response_dimension": rank,
            "exposed_controlled_dimension": int(controlled),
            "exposed_blind_dimension": int(rank - controlled),
            "unexposed_controlled_diagnostic": float(np.linalg.norm(k @ (np.eye(k.shape[0]) - exposed @ exposed.T))),
            "unexposed_blind_diagnostic": float(np.linalg.norm((np.eye(k.shape[0]) - exposed @ exposed.T) @ k @ (np.eye(k.shape[0]) - exposed @ exposed.T))),
            "constraint_object": state.constraint_object,
            "posthoc_only": True,
            "tolerance": tolerance,
            "exposed_basis_orthogonality": float(np.linalg.norm(complement)),
        })
    return rows


__all__ = ["posthoc_regularizer_intersection"]
