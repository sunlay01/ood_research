"""Finite-dimensional source identifiability diagnostics."""

from __future__ import annotations

import numpy as np

from .round3r_3b_benchmark import MomentState, ModuleEnvironment, environment_state, source_optimum
from .round3r_3d_exposure import response_operator


def _nullspace(matrix: np.ndarray, tolerance: float = 1e-9) -> np.ndarray:
    value = np.asarray(matrix, dtype=float)
    _, singular, vh = np.linalg.svd(value, full_matrices=True)
    rank = int(np.sum(singular > tolerance * singular[0])) if singular.size and singular[0] > 0 else 0
    return vh[rank:].T


def structural_identifiability(observation: np.ndarray, target_map: np.ndarray,
                               tolerance: float = 1e-9) -> dict[str, object]:
    """Compute ker(O), ker(A) overlap and factorization criterion."""
    observation = np.asarray(observation, dtype=float)
    target_map = np.asarray(target_map, dtype=float)
    if observation.shape[1] != target_map.shape[1]:
        raise ValueError("world maps must have the same domain dimension")
    kernel_o = _nullspace(observation, tolerance)
    kernel_a = _nullspace(target_map, tolerance)
    intersection = _nullspace(np.vstack((observation, target_map)), tolerance)
    d_struct = int(kernel_o.shape[1] - intersection.shape[1])
    residual = float(np.linalg.norm(target_map @ kernel_o)) if kernel_o.shape[1] else 0.0
    inclusion = residual <= tolerance * max(1.0, np.linalg.norm(target_map))
    # A factorization A = Atilde O exists exactly when ker(O) is included in ker(A).
    a_tilde = target_map @ np.linalg.pinv(observation) if observation.size else np.zeros((target_map.shape[0], observation.shape[0]))
    factor_residual = float(np.linalg.norm(target_map - a_tilde @ observation))
    return {
        "kernel_observation_dimension": int(kernel_o.shape[1]),
        "kernel_target_dimension": int(kernel_a.shape[1]),
        "kernel_intersection_dimension": int(intersection.shape[1]),
        "structural_ambiguity_dimension": d_struct,
        "kernel_inclusion": bool(inclusion),
        "factorization_residual": factor_residual,
        "factorization_exists_numerically": bool(factor_residual <= tolerance * max(1.0, np.linalg.norm(target_map))),
        "observation_rank": int(np.linalg.matrix_rank(observation, tol=tolerance)),
        "target_map_rank": int(np.linalg.matrix_rank(target_map, tol=tolerance)),
        "tolerance": tolerance,
    }


def hidden_emergent_world_pair(base: ModuleEnvironment, source_environments: tuple[ModuleEnvironment, ...],
                               magnitude: float = 0.45) -> dict[str, object]:
    """Construct exact source-identical worlds with opposite target U coupling."""
    source_states = tuple(environment_state(environment) for environment in source_environments)
    optimum, source = source_optimum(source_environments)
    plus = environment_state(base.updated(u_gamma=abs(magnitude)))
    minus = environment_state(base.updated(u_gamma=-abs(magnitude)))
    q_plus = response_operator(source, optimum, plus)
    q_minus = response_operator(source, optimum, minus)
    source_equal = all(np.array_equal(left.second, right.second) and np.array_equal(left.xy, right.xy) and left.y2 == right.y2 for left, right in zip(source_states, source_states))
    return {
        "source_states_identical_exactly": bool(source_equal),
        "world_plus_source_states": source_states,
        "world_minus_source_states": source_states,
        "source_state_count": len(source_states),
        "world_plus_target_u_gamma": abs(magnitude),
        "world_minus_target_u_gamma": -abs(magnitude),
        "q_plus": q_plus,
        "q_minus": q_minus,
        "q_different": bool(not np.allclose(q_plus, q_minus, atol=1e-12, rtol=1e-12)),
        "q_difference_norm": float(np.linalg.norm(q_plus - q_minus)),
        "source_optimum": optimum,
        "source_state": source,
    }


def exact_world_pair(base: ModuleEnvironment, source_environments: tuple[ModuleEnvironment, ...],
                     magnitude: float = 0.45) -> dict[str, object]:
    return hidden_emergent_world_pair(base, source_environments, magnitude)


__all__ = ["structural_identifiability", "hidden_emergent_world_pair", "exact_world_pair"]
