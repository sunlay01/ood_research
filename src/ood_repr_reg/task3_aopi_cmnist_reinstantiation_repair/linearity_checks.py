"""Numerical checks for the declared three-dimensional tangent construction."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .smooth_world import DERIVED_DIRECTIONS


LINEARITY_TOLERANCE = 0.02


@dataclass(frozen=True)
class LinearityResult:
    a_rank: int
    o_rank: int
    common_a_remainder: float
    antisymmetric_a_remainder: float
    common_o_remainder: float
    antisymmetric_o_remainder: float
    passed: bool


def numerical_rank(matrix: Tensor) -> int:
    singular = torch.linalg.svdvals(matrix)
    threshold = max(float(singular.max().item()) if singular.numel() else 0.0, 1e-12) * 1e-8
    return int((singular > threshold).sum().item())


def _remainder(observed: Tensor, expected: Tensor) -> float:
    return float((observed - expected).norm().item() / max(float(observed.norm().item()), 1e-12))


def check_linearity(A: Tensor, O: Tensor, *, actual_common_A: Tensor, actual_anti_A: Tensor, actual_common_O: Tensor, actual_anti_O: Tensor) -> LinearityResult:
    common = DERIVED_DIRECTIONS["common_source_color"]
    anti = DERIVED_DIRECTIONS["antisymmetric_source_color"]
    expected_common_a = (A[:, 0] + A[:, 1]) / (2.0 ** 0.5)
    expected_anti_a = (A[:, 0] - A[:, 1]) / (2.0 ** 0.5)
    expected_common_o = (O[:, 0] + O[:, 1]) / (2.0 ** 0.5)
    expected_anti_o = (O[:, 0] - O[:, 1]) / (2.0 ** 0.5)
    values = (
        _remainder(actual_common_A, expected_common_a),
        _remainder(actual_anti_A, expected_anti_a),
        _remainder(actual_common_O, expected_common_o),
        _remainder(actual_anti_O, expected_anti_o),
    )
    a_rank, o_rank = numerical_rank(A), numerical_rank(O)
    return LinearityResult(a_rank, o_rank, *values, passed=a_rank <= 3 and o_rank <= 3 and all(value <= LINEARITY_TOLERANCE for value in values))
