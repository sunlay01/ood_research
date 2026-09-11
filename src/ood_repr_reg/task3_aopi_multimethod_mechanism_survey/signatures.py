"""Opaque, normalized numerical signatures for blind direction grouping."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

import numpy as np


def method_code_map(methods: Iterable[str]) -> dict[str, str]:
    ordered: list[str] = []
    for method in methods:
        name = str(method)
        if name not in ordered:
            ordered.append(name)
    return {method: f"m{index:03d}" for index, method in enumerate(ordered)}


def _rms(values: Iterable[float]) -> float:
    values = tuple(float(value) for value in values)
    return float(np.sqrt(np.mean(np.square(values)))) if values else 0.0


def method_internal_scales(rows: list[dict[str, Any]]) -> dict[tuple[str, int], float]:
    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in rows:
        if int(row["basis_index"]) in (0, 1, 3):
            grouped[(str(row["method"]), int(row["K"]))].append(float(row["source_bank_response_norm"]))
    return {key: _rms(values) for key, values in grouped.items()}


def normalized_response_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scales = method_internal_scales(rows)
    result = []
    for row in rows:
        scale = scales.get((str(row["method"]), int(row["K"])), 0.0)
        denominator = scale + 1e-12
        result.append({
            "seed": row["seed"], "method": row["method"], "opaque_direction_id": row["opaque_direction_id"], "K": row["K"],
            "normalization_scale": scale,
            "normalized_source_response": float(row["source_bank_response_norm"]) / denominator,
            "normalized_counterfactual_response": float(row["counterfactual_bank_response_norm"]) / denominator,
            "normalized_clean_task_response": float(row["clean_task_bank_response_norm"]) / denominator,
        })
    return result


def mechanism_signature_rows(a_rows: list[dict[str, Any]], o_rows: list[dict[str, Any]], normalized_rows: list[dict[str, Any]], *, methods: Iterable[str] | None = None) -> list[dict[str, Any]]:
    by_direction: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    method_list = list(methods) if methods is not None else [str(row["method"]) for row in [*a_rows, *o_rows, *normalized_rows] if "method" in row]
    method_codes = method_code_map(method_list)
    for row in a_rows:
        by_direction[str(row["opaque_direction_id"])][f"a_{method_codes.get(str(row['method']), 'm_unknown')}"].append(float(row["A_normalized_norm"]))
    for row in o_rows:
        code = method_codes.get(str(row.get("method", "")), "m_unknown")
        by_direction[str(row["opaque_direction_id"])][f"o_{code}"].append(float(row["O_normalized_norm"]))
    for row in normalized_rows:
        key = f"r_{method_codes.get(str(row['method']), 'm_unknown')}_K{row['K']}"
        by_direction[str(row["opaque_direction_id"])][key].append(float(row["normalized_source_response"]))
    keys = sorted({key for values in by_direction.values() for key in values})
    return [
        {"opaque_direction_id": direction, **{key: float(np.mean(values.get(key, [0.0]))) for key in keys}}
        for direction, values in sorted(by_direction.items())
    ]
