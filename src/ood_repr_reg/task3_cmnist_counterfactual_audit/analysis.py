"""Paired ERM/IRMv1 analysis for counterfactual CMNIST diagnostics."""

from __future__ import annotations

import math
from statistics import mean, stdev
from typing import Any


ALLOWED_VERDICTS = {
    "REPRESENTATION-DOMINANT",
    "HEAD-USAGE-DOMINANT",
    "MIXED-DECOMPOSITION",
    "DESCRIPTIVE-INCONCLUSIVE",
    "AUDIT-INVALID",
}

PAIRED_METRICS = [
    "final_target_acc",
    "latent_color_response",
    "task_signal",
    "task_color_overlap",
    "prediction_color_response",
    "probability_color_response",
    "task_head_margin",
    "balanced_clean_accuracy",
    "counterfactual_prediction_consistency",
    "counterfactual_prediction_flip_rate",
]

REPRESENTATION_METRICS = [
    "latent_color_response",
    "task_signal",
    "task_color_overlap",
    "balanced_clean_accuracy",
]

HEAD_USAGE_METRICS = [
    "prediction_color_response",
    "probability_color_response",
    "task_head_margin",
    "counterfactual_prediction_consistency",
    "counterfactual_prediction_flip_rate",
]


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _mean(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return mean(finite) if finite else float("nan")


def _std(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return stdev(finite) if len(finite) > 1 else 0.0


def _ratio(numerator: float, denominator: float, *, tol: float = 1e-12) -> float:
    if not math.isfinite(numerator) or not math.isfinite(denominator) or abs(denominator) <= tol:
        return float("nan")
    return numerator / denominator


def build_paired_effects(rows: list[dict[str, Any]], *, seeds: list[int]) -> list[dict[str, Any]]:
    by_key = {(int(row["seed"]), str(row["method"])): row for row in rows}
    paired: list[dict[str, Any]] = []
    for seed in seeds:
        erm = by_key.get((int(seed), "ERM"))
        irm = by_key.get((int(seed), "IRMv1"))
        if erm is None or irm is None:
            raise ValueError(f"missing ERM/IRMv1 pair for seed {seed}")
        row: dict[str, Any] = {"seed": int(seed)}
        for metric in PAIRED_METRICS:
            erm_value = _to_float(erm.get(metric))
            irm_value = _to_float(irm.get(metric))
            row[f"erm_{metric}"] = erm_value
            row[f"irmv1_{metric}"] = irm_value
            row[f"delta_{metric}"] = irm_value - erm_value
            row[f"ratio_{metric}"] = _ratio(irm_value, erm_value)
        paired.append(row)
    return paired


def _metric_summary(paired: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    deltas = [_to_float(row[f"delta_{metric}"]) for row in paired]
    ratios = [_to_float(row[f"ratio_{metric}"]) for row in paired]
    return {
        "erm_mean": _mean([_to_float(row[f"erm_{metric}"]) for row in paired]),
        "irmv1_mean": _mean([_to_float(row[f"irmv1_{metric}"]) for row in paired]),
        "delta_mean": _mean(deltas),
        "delta_std": _std(deltas),
        "ratio_mean": _mean(ratios),
        "paired_deltas": deltas,
        "paired_ratios": ratios,
    }


def _consistent_count(values: list[float], predicate) -> int:
    return sum(1 for value in values if math.isfinite(value) and predicate(value))


def classify_decomposition(paired: list[dict[str, Any]], *, valid: bool) -> str:
    if not valid:
        return "AUDIT-INVALID"
    if len(paired) != 5:
        return "AUDIT-INVALID"

    summaries = {metric: _metric_summary(paired, metric) for metric in PAIRED_METRICS}
    representation_flags = {
        "task_signal_higher": (
            summaries["task_signal"]["ratio_mean"] >= 1.25
            and _consistent_count(summaries["task_signal"]["paired_ratios"], lambda value: value >= 1.0) >= 4
        ),
        "balanced_clean_accuracy_higher": (
            summaries["balanced_clean_accuracy"]["delta_mean"] >= 0.05
            and _consistent_count(summaries["balanced_clean_accuracy"]["paired_deltas"], lambda value: value > 0.0) >= 4
        ),
        "latent_color_response_lower": (
            summaries["latent_color_response"]["ratio_mean"] <= 0.80
            and _consistent_count(summaries["latent_color_response"]["paired_ratios"], lambda value: value <= 1.0) >= 4
        ),
    }
    head_flags = {
        "prediction_color_response_lower": (
            summaries["prediction_color_response"]["ratio_mean"] <= 0.80
            and _consistent_count(summaries["prediction_color_response"]["paired_ratios"], lambda value: value <= 1.0) >= 4
        ),
        "probability_color_response_lower": (
            summaries["probability_color_response"]["ratio_mean"] <= 0.80
            and _consistent_count(summaries["probability_color_response"]["paired_ratios"], lambda value: value <= 1.0) >= 4
        ),
        "counterfactual_consistency_higher": (
            summaries["counterfactual_prediction_consistency"]["delta_mean"] >= 0.05
            and _consistent_count(summaries["counterfactual_prediction_consistency"]["paired_deltas"], lambda value: value > 0.0) >= 4
        ),
        "counterfactual_flip_rate_lower": (
            summaries["counterfactual_prediction_flip_rate"]["delta_mean"] <= -0.05
            and _consistent_count(summaries["counterfactual_prediction_flip_rate"]["paired_deltas"], lambda value: value < 0.0) >= 4
        ),
    }
    representation = any(representation_flags.values())
    head = any(head_flags.values())
    if representation and head:
        return "MIXED-DECOMPOSITION"
    if representation:
        return "REPRESENTATION-DOMINANT"
    if head:
        return "HEAD-USAGE-DOMINANT"
    return "DESCRIPTIVE-INCONCLUSIVE"


def summarize_audit(
    *,
    diagnostic_rows: list[dict[str, Any]],
    paired_rows: list[dict[str, Any]],
    valid: bool,
    checkpoint_source: str,
    seeds: list[int],
) -> dict[str, Any]:
    verdict = classify_decomposition(paired_rows, valid=valid)
    if verdict not in ALLOWED_VERDICTS:
        raise ValueError(f"invalid verdict: {verdict}")
    summaries = {metric: _metric_summary(paired_rows, metric) for metric in PAIRED_METRICS}
    return {
        "task_id": "TASK3-CMNIST-COUNTERFACTUAL-DIAGNOSTIC-PORT",
        "valid": bool(valid),
        "checkpoint_source": checkpoint_source,
        "seeds": [int(seed) for seed in seeds],
        "methods": ["ERM", "IRMv1"],
        "official_accuracy_gap": summaries["final_target_acc"],
        "representation_content": {metric: summaries[metric] for metric in REPRESENTATION_METRICS},
        "head_usage": {metric: summaries[metric] for metric in HEAD_USAGE_METRICS},
        "paired_effects": summaries,
        "row_count": len(diagnostic_rows),
        "verdict": verdict,
        "interpretation_ceiling": "descriptive diagnostic only",
    }
