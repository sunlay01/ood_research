"""Conservative summary logic for first-round OOD capability decomposition."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any


ALLOWED_VERDICTS = {
    "FIRST-ROUND-CAPABILITY-ISOLATED",
    "FIRST-ROUND-CAPABILITY-PARTIAL",
    "FIRST-ROUND-CAPABILITY-INCONCLUSIVE",
    "FIRST-ROUND-AUDIT-INVALID",
}

ALLOWED_DOMINANT_BOTTLENECKS = {
    "coverage",
    "separability",
    "selection",
    "mixed",
    "unresolved",
    "invalid",
}


def _float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if math.isfinite(result) else float("nan")


def _mean(values: list[float]) -> float:
    clean = [value for value in values if math.isfinite(value)]
    return sum(clean) / len(clean) if clean else float("nan")


def _median(values: list[float]) -> float:
    clean = sorted(value for value in values if math.isfinite(value))
    if not clean:
        return float("nan")
    mid = len(clean) // 2
    if len(clean) % 2:
        return clean[mid]
    return 0.5 * (clean[mid - 1] + clean[mid])


def _sample_std(values: list[float]) -> float:
    clean = [value for value in values if math.isfinite(value)]
    if len(clean) < 2:
        return 0.0 if len(clean) == 1 else float("nan")
    mu = _mean(clean)
    return math.sqrt(sum((value - mu) ** 2 for value in clean) / (len(clean) - 1))


def _lookup(rows: list[dict[str, Any]], *, key_fields: tuple[str, ...]) -> dict[tuple[Any, ...], dict[str, Any]]:
    result = {}
    for row in rows:
        key = tuple(row[field] for field in key_fields)
        result[key] = row
    return result


def _best_by(rows: list[dict[str, Any]], metric: str) -> dict[tuple[int, str], dict[str, Any]]:
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["seed"]), str(row["encoder_method"]))].append(row)
    best = {}
    for key, group in grouped.items():
        best[key] = max(group, key=lambda item: _float(item.get(metric)))
    return best


def paired_capability_rows(
    coverage: list[dict[str, Any]],
    separability: list[dict[str, Any]],
    selection: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build per-seed ERM-vs-IRMv1 capability deltas without adding causal claims."""
    coverage_by_key = _lookup(coverage, key_fields=("seed", "encoder_method"))
    best_sep = _best_by(separability, "projected_target_acc")
    selection_by_key = _lookup(selection, key_fields=("seed", "encoder_method", "head_method"))
    seeds = sorted({int(row["seed"]) for row in coverage})
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        erm = coverage_by_key.get((seed, "ERM"))
        irm = coverage_by_key.get((seed, "IRMv1"))
        if erm is None or irm is None:
            continue
        erm_oracle = selection_by_key.get((seed, "ERM", "ORACLE_CLEAN"))
        irm_oracle = selection_by_key.get((seed, "IRMv1", "ORACLE_CLEAN"))
        erm_head_irm = selection_by_key.get((seed, "ERM", "HEAD_IRMv1"))
        irm_head_irm = selection_by_key.get((seed, "IRMv1", "HEAD_IRMv1"))
        erm_sep = best_sep.get((seed, "ERM"))
        irm_sep = best_sep.get((seed, "IRMv1"))
        erm_target = _float(erm["original_target_acc"])
        irm_target = _float(irm["original_target_acc"])
        row = {
            "seed": seed,
            "erm_target_acc": erm_target,
            "irmv1_target_acc": irm_target,
            "irmv1_minus_erm_target_acc": irm_target - erm_target,
            "erm_oracle_clean_coverage_acc": _float(erm["oracle_clean_balanced_accuracy"]),
            "irmv1_oracle_clean_coverage_acc": _float(irm["oracle_clean_balanced_accuracy"]),
            "coverage_oracle_gap_erm": _float(erm["oracle_clean_balanced_accuracy"]) - erm_target,
            "coverage_oracle_gap_irmv1": _float(irm["oracle_clean_balanced_accuracy"]) - irm_target,
            "coverage_oracle_gap_delta_erm_minus_irmv1": (_float(erm["oracle_clean_balanced_accuracy"]) - erm_target) - (_float(irm["oracle_clean_balanced_accuracy"]) - irm_target),
            "best_separability_rank_erm": int(erm_sep["rank_k"]) if erm_sep else -1,
            "best_separability_rank_irmv1": int(irm_sep["rank_k"]) if irm_sep else -1,
            "separability_best_gain_erm": _float(erm_sep["projected_target_acc"]) - erm_target if erm_sep else float("nan"),
            "separability_best_gain_irmv1": _float(irm_sep["projected_target_acc"]) - irm_target if irm_sep else float("nan"),
            "selection_oracle_gap_erm": _float(erm_oracle["target_acc"]) - erm_target if erm_oracle else float("nan"),
            "selection_oracle_gap_irmv1": _float(irm_oracle["target_acc"]) - irm_target if irm_oracle else float("nan"),
            "source_head_irmv1_gap_on_erm_encoder": _float(erm_head_irm["target_acc"]) - erm_target if erm_head_irm else float("nan"),
            "source_head_irmv1_gap_on_irmv1_encoder": _float(irm_head_irm["target_acc"]) - irm_target if irm_head_irm else float("nan"),
        }
        row["coverage_isolation_vote"] = bool(row["coverage_oracle_gap_delta_erm_minus_irmv1"] >= 0.20)
        row["separability_isolation_vote"] = bool(row["separability_best_gain_erm"] >= 0.20)
        row["selection_isolation_vote"] = bool(row["selection_oracle_gap_erm"] >= 0.20)
        rows.append(row)
    return rows


def summarize_capabilities(
    coverage: list[dict[str, Any]],
    separability: list[dict[str, Any]],
    selection: list[dict[str, Any]],
    *,
    valid: bool = True,
) -> dict[str, Any]:
    paired = paired_capability_rows(coverage, separability, selection)
    coverage_votes = sum(1 for row in paired if row["coverage_isolation_vote"])
    separability_votes = sum(1 for row in paired if row["separability_isolation_vote"])
    selection_votes = sum(1 for row in paired if row["selection_isolation_vote"])
    vote_counts = {
        "coverage": coverage_votes,
        "separability": separability_votes,
        "selection": selection_votes,
    }
    isolated_candidates = [name for name, count in vote_counts.items() if count >= 4]
    if not valid or len(paired) != 5 or len(coverage) != 10 or len(selection) != 30:
        verdict = "FIRST-ROUND-AUDIT-INVALID"
        bottleneck = "invalid"
    elif len(isolated_candidates) == 1:
        verdict = "FIRST-ROUND-CAPABILITY-ISOLATED"
        bottleneck = isolated_candidates[0]
    elif len(isolated_candidates) > 1:
        verdict = "FIRST-ROUND-CAPABILITY-PARTIAL"
        bottleneck = "mixed"
    elif max(vote_counts.values()) >= 2:
        verdict = "FIRST-ROUND-CAPABILITY-PARTIAL"
        bottleneck = "mixed"
    else:
        verdict = "FIRST-ROUND-CAPABILITY-INCONCLUSIVE"
        bottleneck = "unresolved"

    target_gaps = [row["irmv1_minus_erm_target_acc"] for row in paired]
    coverage_erm = [row["coverage_oracle_gap_erm"] for row in paired]
    sep_erm = [row["separability_best_gain_erm"] for row in paired]
    sel_erm = [row["selection_oracle_gap_erm"] for row in paired]
    return {
        "verdict": verdict,
        "dominant_bottleneck": bottleneck,
        "valid": bool(valid),
        "row_counts": {
            "coverage": len(coverage),
            "separability": len(separability),
            "selection": len(selection),
            "paired": len(paired),
        },
        "vote_counts": vote_counts,
        "isolated_candidates": isolated_candidates,
        "target_gap_mean": _mean(target_gaps),
        "target_gap_median": _median(target_gaps),
        "target_gap_std": _sample_std(target_gaps),
        "coverage_oracle_gap_erm_mean": _mean(coverage_erm),
        "separability_best_gain_erm_mean": _mean(sep_erm),
        "selection_oracle_gap_erm_mean": _mean(sel_erm),
        "interpretation_ceiling": "diagnostic first-round A/B/C only; no causal, source-identifiability, algorithm, or theory-validation claim",
    }
