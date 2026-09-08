"""Small abstract counterexamples for the 3E recovery theorem."""

from __future__ import annotations

import numpy as np

from .round3r_3e_recovery import minimax_recovery_error, operator_summary


def _record(name: str, a: np.ndarray, o: np.ndarray) -> dict[str, object]:
    result = operator_summary(a, o)
    result.update({"name": name, "response": a, "observation": o})
    return result


def counterexamples() -> dict[str, object]:
    # Same one-dimensional structural ambiguity, different response magnitude.
    same_dimension = {
        "small": _record("same_d_struct_small", np.array([[0.0, 0.2]]), np.array([[1.0, 0.0]])),
        "large": _record("same_d_struct_large", np.array([[0.0, 1.0]]), np.array([[1.0, 0.0]])),
    }
    # Different ambiguity dimensions, equal alpha.
    different_dimension = {
        "one": _record("d_struct_one", np.array([[0.0, 1.0, 0.0]]), np.array([[1.0, 0.0, 0.0]])),
        "two": _record("d_struct_two", np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]), np.array([[1.0, 0.0, 0.0]])),
    }
    # Same observation rank, different null-space orientation relative to A.
    same_rank = {
        "observe_sensitive": _record("same_rank_sensitive_observed", np.diag([10.0, 1.0]), np.array([[1.0, 0.0]])),
        "observe_weak": _record("same_rank_weak_observed", np.diag([10.0, 1.0]), np.array([[0.0, 1.0]])),
    }
    # Extra information can be redundant for the worst direction.
    unchanged = {
        "before": _record("extra_info_before", np.diag([1.0, 2.0, 5.0]), np.array([[1.0, 0.0, 0.0]])),
        "after": _record("extra_info_after", np.diag([1.0, 2.0, 5.0]), np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])),
    }
    aligned = {
        "before": _record("aligned_before", np.diag([10.0, 1.0]), np.zeros((0, 2))),
        "after": _record("aligned_after", np.diag([10.0, 1.0]), np.array([[1.0, 0.0]])),
    }
    return {
        "same_structural_dimension_different_alpha": same_dimension,
        "different_structural_dimension_same_alpha": different_dimension,
        "same_observation_rank_different_alpha": same_rank,
        "more_information_same_alpha": unchanged,
        "one_aligned_observation_large_drop": aligned,
        "injective_endpoint": _record("injective", np.array([[1.0, 2.0]]), np.eye(2)),
        "no_information_endpoint": _record("no_information", np.diag([3.0, 2.0]), np.zeros((0, 2))),
        "zero_response_endpoint": _record("zero_response", np.zeros((1, 2)), np.array([[1.0, 0.0]])),
    }


__all__ = ["counterexamples"]
