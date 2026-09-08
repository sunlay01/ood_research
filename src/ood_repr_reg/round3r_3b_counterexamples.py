"""Response-level identifiability counterexamples."""

from __future__ import annotations

import numpy as np


def identical_image_counterexample() -> dict[str, object]:
    first = np.array([[1.0, 0.0], [0.0, 0.0]])
    second = np.array([[2.0, 0.0], [0.0, 0.0]])
    return {"u1": first, "u2": second, "same_image": True, "intersection_dimension": 1,
            "response_only_identifiable": False, "label_used_in_construction": True}


def partial_overlap_counterexample() -> dict[str, object]:
    first = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    second = np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]])
    return {"u1": first, "u2": second, "same_image": False, "intersection_dimension": 1,
            "response_only_unique": False, "label_used_in_construction": True}


def direct_sum_decomposition(contributions: tuple[np.ndarray, ...]) -> np.ndarray:
    if not contributions:
        raise ValueError("at least one contribution is required")
    return np.sum(np.stack(contributions), axis=0)
