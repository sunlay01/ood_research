"""Preregistered gates for the frozen-encoder A/O/Pi audit."""

from __future__ import annotations

import math
from collections import defaultdict
from statistics import mean
from typing import Any


ALLOWED_VERDICTS = {
    "AOPI-REINSTANTIATION-PASS",
    "AOPI-REINSTANTIATION-PARTIAL",
    "AOPI-REINSTANTIATION-FAIL",
    "AOPI-AUDIT-INVALID",
}


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def geometry_gate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["seed"]), str(row["method"]), str(row["tangent"]))].append(row)
    checks: list[bool] = []
    for values in grouped.values():
        if len(values) != 2:
            checks.append(False)
            continue
        values = sorted(values, key=lambda item: float(item["epsilon"]))
        left, right = values
        checks.append(
            bool(left["finite"]) and bool(right["finite"])
            and float(left["gram_disagreement_with_epsilon_pair"]) <= 0.15
            and int(left["response_rank"]) == int(right["response_rank"])
            and float(left["reparameterization_gram_disagreement"]) <= 0.05
        )
    per_seed: dict[int, bool] = {}
    for seed in sorted({int(row["seed"]) for row in rows}):
        relevant = [item for key, item in grouped.items() if key[0] == seed]
        per_seed[seed] = bool(relevant) and all(
            bool(pair[0]["finite"]) and bool(pair[-1]["finite"])
            and float(pair[0]["gram_disagreement_with_epsilon_pair"]) <= 0.15
            and int(pair[0]["response_rank"]) == int(pair[-1]["response_rank"])
            and float(pair[0]["reparameterization_gram_disagreement"]) <= 0.05
            for pair in relevant
        )
    return {"passed_seed_count": sum(per_seed.values()), "per_seed": per_seed, "pass": sum(per_seed.values()) >= 4}


def finite_prediction_gate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if float(row["epsilon"]) == 0.01 and row["tangent"] in {"source_env0_color", "source_env1_color", "shared_label_noise"}:
            by_method[str(row["method"])].append(row)
    per_method: dict[str, dict[str, Any]] = {}
    for method in ("ERM", "IRMv1"):
        values = by_method[method]
        passed = sum(
            bool(row["finite"])
            and float(row["cosine_similarity"]) >= 0.90
            and float(row["relative_vector_error"]) <= 0.25
            and bool(row["task_inner_product_sign_agreement"])
            for row in values
        )
        per_method[method] = {"passed_rows": passed, "total_rows": len(values), "pass": len(values) == 15 and passed >= 12}
    return {"per_method": per_method, "pass": all(item["pass"] for item in per_method.values())}


def discrimination_gate(rows: list[dict[str, Any]], *, geometry_pass: bool, finite_pass: bool) -> dict[str, Any]:
    grouped: dict[tuple[int, str], list[float]] = defaultdict(list)
    for row in rows:
        if float(row["epsilon"]) == 0.01:
            grouped[(int(row["seed"]), str(row["method"]))].append(float(row["mismatch_norm"]))
    paired: list[dict[str, Any]] = []
    for seed in range(10, 15):
        erm, irm = grouped[(seed, "ERM")], grouped[(seed, "IRMv1")]
        if not erm or not irm:
            continue
        er, ir = float(math.sqrt(sum(value * value for value in erm))), float(math.sqrt(sum(value * value for value in irm)))
        paired.append({"seed": seed, "erm_primary_mismatch": er, "irmv1_primary_mismatch": ir, "paired_difference": er - ir, "irmv1_lower": ir < er})
    wins = sum(bool(row["irmv1_lower"]) for row in paired)
    effect = mean([float(row["paired_difference"]) for row in paired]) if paired else float("nan")
    if geometry_pass and finite_pass and len(paired) == 5 and wins >= 4 and effect > 0.0:
        label = "DISCRIMINATIVE"
    elif math.isfinite(effect) and effect > 0.0:
        label = "WEAKLY-DISCRIMINATIVE"
    else:
        label = "NONDISCRIMINATIVE"
    return {"label": label, "paired_rows": paired, "irmv1_lower_seed_count": wins, "mean_paired_improvement": effect, "pass": label == "DISCRIMINATIVE"}


def overall_verdict(*, valid: bool, geometry: dict[str, Any], finite: dict[str, Any], discrimination: dict[str, Any]) -> str:
    if not valid:
        return "AOPI-AUDIT-INVALID"
    if geometry["pass"] and finite["pass"] and discrimination["pass"]:
        return "AOPI-REINSTANTIATION-PASS"
    if geometry["pass"] or finite["pass"] or discrimination["label"] == "WEAKLY-DISCRIMINATIVE":
        return "AOPI-REINSTANTIATION-PARTIAL"
    return "AOPI-REINSTANTIATION-FAIL"
