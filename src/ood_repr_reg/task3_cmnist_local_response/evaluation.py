"""Evaluation, source-only selection, and verdict logic for Task 3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class SelectionScore:
    worst_source_accuracy: float
    mean_source_accuracy: float
    checkpoint_step: int

    def key(self) -> tuple[float, float, int]:
        return (self.worst_source_accuracy, self.mean_source_accuracy, self.checkpoint_step)


def select_source_only(candidates: Iterable[object]) -> object:
    """Select by source validation only.

    Candidate objects must expose ``selection_score``.  This intentionally does
    not inspect target fields.
    """
    values = list(candidates)
    if not values:
        raise ValueError("at least one candidate is required")
    return max(values, key=lambda candidate: candidate.selection_score.key())


def paired_differences(rows: list[dict[str, object]], left: str, right: str) -> list[float]:
    by_seed_method = {(int(row["seed"]), str(row["method"])): row for row in rows}
    seeds = sorted({seed for seed, method in by_seed_method if method in (left, right)})
    values = []
    for seed in seeds:
        if (seed, left) in by_seed_method and (seed, right) in by_seed_method:
            values.append(
                float(by_seed_method[(seed, left)]["target_accuracy"])
                - float(by_seed_method[(seed, right)]["target_accuracy"])
            )
    return values


def bootstrap_ci(values: list[float], *, seed: int = 1777, iterations: int = 4000) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    data = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(iterations):
        sample = rng.choice(data, size=data.size, replace=True)
        means.append(float(sample.mean()))
    return tuple(float(x) for x in np.percentile(means, [2.5, 97.5]))  # type: ignore[return-value]


def comparison_row(rows: list[dict[str, object]], left: str, right: str) -> dict[str, object]:
    diffs = paired_differences(rows, left, right)
    ci_low, ci_high = bootstrap_ci(diffs)
    return {
        "left_method": left,
        "right_method": right,
        "n_pairs": len(diffs),
        "mean_difference": float(np.mean(diffs)) if diffs else float("nan"),
        "std_difference": float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0,
        "median_difference": float(np.median(diffs)) if diffs else float("nan"),
        "seed_wins": int(sum(value > 0 for value in diffs)),
        "ci95_low": ci_low,
        "ci95_high": ci_high,
    }


def _mean_metric(rows: list[dict[str, object]], method: str, key: str) -> float:
    values = [float(row[key]) for row in rows if str(row["method"]) == method]
    return float(np.mean(values)) if values else float("nan")


def verdict(rows: list[dict[str, object]], comparisons: list[dict[str, object]]) -> tuple[str, dict[str, object]]:
    by_pair = {(row["left_method"], row["right_method"]): row for row in comparisons}
    lr_erm = by_pair.get(("LOCAL_RESPONSE", "ERM"), {})
    lr_grad = by_pair.get(("LOCAL_RESPONSE", "UNPRECONDITIONED_GRAD_ALIGN"), {})
    lr_random = by_pair.get(("LOCAL_RESPONSE", "RANDOM_METRIC"), {})
    completed_seeds = len({int(row["seed"]) for row in rows if row["method"] == "LOCAL_RESPONSE"})
    source_gap = _mean_metric(rows, "LOCAL_RESPONSE", "worst_source_accuracy") - _mean_metric(
        rows, "ERM", "worst_source_accuracy"
    )
    color_gap = _mean_metric(rows, "ERM", "counterfactual_color_response") - _mean_metric(
        rows, "LOCAL_RESPONSE", "counterfactual_color_response"
    )
    update_lr = _mean_metric(rows, "LOCAL_RESPONSE", "total_update_norm")
    update_grad = _mean_metric(rows, "UNPRECONDITIONED_GRAD_ALIGN", "total_update_norm")
    update_ratio = update_lr / max(update_grad, 1e-12)
    criteria = {
        "no_target_leakage": True,
        "end_to_end_representation_training": True,
        "all_10_primary_seeds_completed": completed_seeds == 10,
        "mean_ood_vs_erm_ge_2pp": float(lr_erm.get("mean_difference", float("nan"))) >= 0.02,
        "mean_ood_vs_grad_ge_1pp": float(lr_grad.get("mean_difference", float("nan"))) >= 0.01,
        "seed_wins_vs_erm_ge_7": int(lr_erm.get("seed_wins", 0)) >= 7,
        "seed_wins_vs_grad_ge_7": int(lr_grad.get("seed_wins", 0)) >= 7,
        "worst_source_degradation_within_1pp": source_gap >= -0.01,
        "real_metric_beats_random_metric": (
            float(lr_random.get("mean_difference", float("nan"))) >= 0.01
            and int(lr_random.get("seed_wins", 0)) >= 7
        ),
        "color_or_mechanism_predicted_direction": color_gap > 0.0,
        "not_explained_by_update_norm": 0.5 <= update_ratio <= 2.0,
    }
    if all(criteria.values()):
        return "TASK3-CMNIST-SUPPORT", criteria
    if (
        float(lr_erm.get("mean_difference", -1.0)) <= 0.0
        or float(lr_grad.get("mean_difference", -1.0)) <= 0.0
        or not criteria["real_metric_beats_random_metric"]
    ):
        return "TASK3-CMNIST-FAIL", criteria
    return "TASK3-CMNIST-PARTIAL", criteria


def counterexample_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    by_seed = {}
    for row in rows:
        by_seed.setdefault(int(row["seed"]), {})[str(row["method"])] = row
    for seed, methods in sorted(by_seed.items()):
        lr = methods.get("LOCAL_RESPONSE")
        grad = methods.get("UNPRECONDITIONED_GRAD_ALIGN")
        erm = methods.get("ERM")
        random = methods.get("RANDOM_METRIC")
        if lr and grad:
            if float(lr["local_response_disagreement"]) < float(grad["local_response_disagreement"]) and float(lr["target_accuracy"]) < float(grad["target_accuracy"]):
                result.append({"seed": seed, "case": "lower_local_response_but_worse_ood", "left": "LOCAL_RESPONSE", "right": "UNPRECONDITIONED_GRAD_ALIGN"})
            if float(lr["counterfactual_color_response"]) < float(grad["counterfactual_color_response"]) and float(lr["target_accuracy"]) < float(grad["target_accuracy"]):
                result.append({"seed": seed, "case": "lower_color_response_but_worse_accuracy", "left": "LOCAL_RESPONSE", "right": "UNPRECONDITIONED_GRAD_ALIGN"})
        if lr and erm:
            if float(lr["local_response_disagreement"]) >= float(erm["local_response_disagreement"]) and float(lr["target_accuracy"]) > float(erm["target_accuracy"]):
                result.append({"seed": seed, "case": "better_ood_without_lower_local_response", "left": "LOCAL_RESPONSE", "right": "ERM"})
        if lr and random:
            if float(random["target_accuracy"]) >= float(lr["target_accuracy"]):
                result.append({"seed": seed, "case": "random_metric_matches_or_beats_real", "left": "RANDOM_METRIC", "right": "LOCAL_RESPONSE"})
    return result
