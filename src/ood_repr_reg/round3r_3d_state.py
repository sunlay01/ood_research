"""Observable task-complete moment states for the 3D exposure analysis.

The state contains exactly the moments needed by a linear population square
risk.  It deliberately does not contain target risk, mechanism labels, or
regularizer metadata.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .round3r_3b_benchmark import MomentState

Array = np.ndarray


def svec_symmetric(matrix: Array) -> Array:
    """Frobenius-isometric vectorization of a symmetric matrix."""
    value = np.asarray(matrix, dtype=float)
    if value.ndim != 2 or value.shape[0] != value.shape[1]:
        raise ValueError("matrix must be square")
    if not np.allclose(value, value.T, atol=1e-10, rtol=1e-10):
        raise ValueError("matrix must be symmetric")
    entries = []
    for row in range(value.shape[0]):
        entries.append(value[row, row])
        for col in range(row):
            entries.append(np.sqrt(2.0) * value[row, col])
    return np.asarray(entries, dtype=float)


def _unpack_state(state: MomentState | tuple[Array, Array, float] | dict[str, object]) -> tuple[Array, Array, float]:
    if isinstance(state, MomentState):
        return np.asarray(state.second, dtype=float), np.asarray(state.xy, dtype=float), float(state.y2)
    if isinstance(state, dict):
        return np.asarray(state["second"], dtype=float), np.asarray(state["xy"], dtype=float), float(state.get("y2", 1.0))
    if len(state) != 3:
        raise ValueError("state tuple must contain second, xy, y2")
    return np.asarray(state[0], dtype=float), np.asarray(state[1], dtype=float), float(state[2])


@dataclass(frozen=True)
class TaskState:
    """The observable state psi=(svec(M), m, c)."""

    matrix: Array
    cross: Array
    y2: float

    def vector(self) -> Array:
        return np.concatenate((svec_symmetric(self.matrix), self.cross, np.asarray([self.y2])))


def task_state(state: MomentState | tuple[Array, Array, float] | dict[str, object]) -> Array:
    """Return the task-complete observable state as a flat vector."""
    matrix, cross, y2 = _unpack_state(state)
    return np.concatenate((svec_symmetric(matrix), cross, np.asarray([y2])))


def state_difference(target: MomentState | Array, reference: MomentState | Array) -> Array:
    """Return psi(target)-psi(reference)."""
    target_vector = target if isinstance(target, np.ndarray) and target.ndim == 1 else task_state(target)
    reference_vector = reference if isinstance(reference, np.ndarray) and reference.ndim == 1 else task_state(reference)
    return np.asarray(target_vector, dtype=float) - np.asarray(reference_vector, dtype=float)


def state_components(state: MomentState | Array) -> tuple[Array, Array, float]:
    """Decode a state only when its matrix dimension can be inferred."""
    if isinstance(state, MomentState):
        return _unpack_state(state)
    value = np.asarray(state, dtype=float)
    # d(d+1)/2 + d + 1 = len(value), with d the feature dimension.
    length = value.size
    for dimension in range(1, length + 1):
        if dimension * (dimension + 1) // 2 + dimension + 1 == length:
            matrix = np.zeros((dimension, dimension), dtype=float)
            index = 0
            for row in range(dimension):
                matrix[row, row] = value[index]
                index += 1
                for col in range(row):
                    matrix[row, col] = matrix[col, row] = value[index] / np.sqrt(2.0)
                    index += 1
            return matrix, value[index:index + dimension], float(value[-1])
    raise ValueError("cannot infer feature dimension from state length")


def transform_moment_state(state: MomentState, transform: Array) -> MomentState:
    """Transform coordinates by ``X' = T X`` while preserving ``Y``."""
    matrix = np.asarray(transform, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] != state.second.shape[0]:
        raise ValueError("transform has incompatible shape")
    if abs(np.linalg.det(matrix)) <= 1e-12:
        raise ValueError("transform must be invertible")
    return MomentState(matrix @ state.second @ matrix.T, matrix @ state.xy, state.y2)


__all__ = ["TaskState", "svec_symmetric", "task_state", "state_difference", "state_components", "transform_moment_state"]
