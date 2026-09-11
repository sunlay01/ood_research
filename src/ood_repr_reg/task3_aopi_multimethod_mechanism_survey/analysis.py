"""Numerical gates and descriptive summaries for the multi-method survey."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch
from torch import Tensor

from .smooth_world5 import all_direction_vectors, BASIS, OPAQUE_IDS, SEMANTIC_NAMES


def numerical_rank(matrix: Tensor, tolerance: float = 1e-8) -> int:
    values = torch.linalg.svdvals(matrix.double())
    if not values.numel():
        return 0
    return int((values > max(float(values.max()) * tolerance, 1e-12)).sum())


def _linear_rows(matrix: Tensor) -> list[dict[str, Any]]:
    primary = matrix[:, :5]
    rows = []
    for index, direction in enumerate(all_direction_vectors()[5:], start=5):
        actual = matrix @ direction
        predicted = primary @ direction
        remainder = float((actual - predicted).norm().item())
        denominator = max(float(actual.norm().item()), 1e-12)
        rows.append({
            "opaque_direction_id": OPAQUE_IDS[index],
            "semantic_name": SEMANTIC_NAMES[index],
            "relative_remainder": remainder / denominator,
            "passed": remainder / denominator <= 0.02,
        })
    return rows


def derived_direction_checks(A: Tensor, O: Tensor) -> list[dict[str, Any]]:
    if A.shape[1] != 5 or O.shape[1] != 5:
        raise ValueError("A and O must use the five primary basis columns")
    a_rows = _linear_rows(A)
    o_rows = _linear_rows(O)
    return [
        {
            "opaque_direction_id": left["opaque_direction_id"],
            "semantic_name": left["semantic_name"],
            "A_relative_remainder": left["relative_remainder"],
            "O_relative_remainder": right["relative_remainder"],
            "A_pass": left["passed"],
            "O_pass": right["passed"],
            "passed": bool(left["passed"] and right["passed"]),
        }
        for left, right in zip(a_rows, o_rows)
    ]


def geometry_rows(seed: int, method: str, A: Tensor, O: Tensor) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if A.shape[1] != 5 or O.shape[1] != 5:
        raise ValueError("geometry requires five primary directions")
    a_scale = float(torch.linalg.vector_norm(A, dim=0).mean().item())
    o_scale = float(torch.linalg.vector_norm(O, dim=0).mean().item())
    directions = all_direction_vectors()
    a_rows, o_rows = [], []
    for index, direction in enumerate(directions):
        a_value = A @ direction
        o_value = O @ direction
        a_rows.append({
            "seed": seed, "method": method, "opaque_direction_id": OPAQUE_IDS[index],
            "basis_index": index if index < 5 else -1,
            "A_norm": float(a_value.norm()), "A_normalized_norm": float(a_value.norm()) / max(a_scale, 1e-12),
            "A_rank_primary": numerical_rank(A), "finite": bool(torch.isfinite(a_value).all()),
        })
        o_rows.append({
            "seed": seed, "method": method, "opaque_direction_id": OPAQUE_IDS[index],
            "basis_index": index if index < 5 else -1,
            "O_norm": float(o_value.norm()), "O_normalized_norm": float(o_value.norm()) / max(o_scale, 1e-12),
            "O_rank_primary": numerical_rank(O), "finite": bool(torch.isfinite(o_value).all()),
        })
    return a_rows, o_rows


def world_gate(A: Tensor, O: Tensor, *, atol: float = 1e-8) -> dict[str, Any]:
    checks = derived_direction_checks(A, O)
    oe3 = float(O[:, 2].norm())
    oe5 = float(O[:, 4].norm())
    rank_a = numerical_rank(A)
    rank_o = numerical_rank(O)
    passed = rank_a <= 5 and rank_o <= 3 and oe3 <= atol and oe5 <= atol and all(row["passed"] for row in checks)
    return {
        "rank_A": rank_a, "rank_O": rank_o, "Oe3_norm": oe3, "Oe5_norm": oe5,
        "derived_direction_checks": checks, "passed": bool(passed),
    }


def paired_method_summary(performance_rows: list[dict[str, Any]], response_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str], dict[str, Any]] = {}
    for row in performance_rows:
        grouped[(int(row["seed"]), str(row["method"]))] = row
    response: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in response_rows:
        response[(int(row["seed"]), str(row["method"]))].append(row)
    seeds = sorted({key[0] for key in grouped})
    methods = sorted({key[1] for key in grouped})
    rows = []
    for seed in seeds:
        for method in methods:
            item = grouped[(seed, method)]
            values = response[(seed, method)]
            rows.append({
                "seed": seed, "method": method,
                "source_mean_acc": item["source_mean_acc"], "target_acc": item["target_acc"],
                "target_color_agreement": item["prediction_color_agreement"],
                "functional_source_response_K20_mean": sum(float(x["source_bank_response_norm"]) for x in values if int(x["K"]) == 20) / max(sum(int(x["K"]) == 20 for x in values), 1),
                "functional_counterfactual_response_K20_mean": sum(float(x["counterfactual_bank_response_norm"]) for x in values if int(x["K"]) == 20) / max(sum(int(x["K"]) == 20 for x in values), 1),
            })
    return rows


def final_verdict(*, valid: bool, complete: bool, grouping_stable: bool) -> str:
    if not valid:
        return "ALGORITHM-PANEL-EXPANSION-INVALID"
    if not complete or not grouping_stable:
        return "ALGORITHM-PANEL-EXPANSION-PARTIAL"
    return "ALGORITHM-PANEL-EXPANSION-PARTIAL"
