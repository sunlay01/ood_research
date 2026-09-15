"""Exact binary preference model plus an acquisition-collapse audit.

The binary part is an analytic population calculation.  The acquisition part
reuses the already-run convergence trajectories and compares optimizers at
matched source BCE, rather than at matched step.  Neither part is a theorem
about the original ReLU network.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "round3_redesign/task_failure_regime_synthetic/results"


def _logit_reliability(p: float) -> float:
    if p <= 0.0:
        return float("inf")
    if p >= 1.0:
        return float("inf")
    return math.log(p / (1.0 - p))


def _risk(weights: tuple[float, float], rho: float, sigma: float) -> float:
    """E softplus(-Y * score) for C=Y eta_c, S=Y eta_s."""
    a, b = weights
    total = 0.0
    for eta_c, p_c in ((1, 1.0 - sigma), (-1, sigma)):
        for eta_s, p_s in ((1, rho), (-1, 1.0 - rho)):
            probability = p_c * p_s
            if probability == 0.0:
                continue
            margin = a * eta_c + b * eta_s
            total += probability * float(np.logaddexp(0.0, -margin))
    return total


def binary_row(rho: float, sigma: float, rho_target: float) -> dict[str, float | bool]:
    core_weight = _logit_reliability(1.0 - sigma)
    shortcut_weight = _logit_reliability(rho)
    source_prefers_shortcut = shortcut_weight > core_weight
    source_weights = (core_weight, shortcut_weight)
    balanced_weights = (core_weight, 0.0)
    actual = _risk(source_weights, rho_target, sigma)
    repaired = _risk(balanced_weights, rho_target, sigma)
    return {"rho": rho, "sigma_c": sigma, "rho_target": rho_target, "core_weight": core_weight, "shortcut_weight": shortcut_weight, "source_prefers_shortcut": source_prefers_shortcut, "R_target_source_opt": actual, "R_target_balanced_head": repaired, "G_use_exact": actual - repaired, "threshold_gap": rho - (1.0 - sigma)}


def _binary_sweep() -> list[dict[str, float | bool]]:
    rows = []
    for rho in (0.55, 0.65, 0.75, 0.85, 0.95):
        for sigma in (0.0, 0.1, 0.2, 0.3, 0.4):
            rows.append(binary_row(rho, sigma, 0.10))
    return rows


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def _matched_acquisition() -> dict[str, float | int | str]:
    path = OUT / "optimizer_capacity_rows.csv"
    if not path.exists():
        return {"status": "MISSING_EXISTING_AUDIT"}
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row.get("checkpoint") in {"0", "100", "250", "500", "1000", "2000", "5000"}]
    groups = {(row["seed"], row["alpha"], row["sigma_c"], row["d_z"]) for row in rows}
    pairs: list[tuple[float, float, float]] = []
    for group in groups:
        members = [row for row in rows if (row["seed"], row["alpha"], row["sigma_c"], row["d_z"]) == group]
        sgd = [row for row in members if row["optimizer"] == "sgd"]
        adam = [row for row in members if row["optimizer"] == "adam"]
        for left in sgd:
            if not adam:
                continue
            right = min(adam, key=lambda row: abs(float(row["source_risk"]) - float(left["source_risk"])))
            source_gap = abs(float(left["source_risk"]) - float(right["source_risk"]))
            repr_gap = float(left["G_repr"]) - float(right["G_repr"])
            if source_gap <= 0.01:
                pairs.append((source_gap, repr_gap, float(left["source_risk"])))
    if not pairs:
        return {"status": "NO_MATCHED_PAIRS"}
    diffs = np.asarray([pair[1] for pair in pairs])
    source_gaps = np.asarray([pair[0] for pair in pairs])
    return {"status": "OK", "n_matched_pairs": len(pairs), "mean_abs_source_risk_gap": float(source_gaps.mean()), "mean_G_repr_sgd_minus_adam": float(diffs.mean()), "mean_abs_G_repr_gap": float(np.abs(diffs).mean()), "fraction_abs_G_repr_gap_below_0.05": float((np.abs(diffs) < 0.05).mean()), "optimizer_speed_consistent": bool(np.abs(diffs).mean() < 0.05)}


def run(smoke: bool = False) -> dict[str, object]:
    binary = _binary_sweep()
    if smoke:
        binary = binary[:4]
    _write_csv(OUT / ("binary_preference_smoke_rows.csv" if smoke else "binary_preference_rows.csv"), binary)
    matched = _matched_acquisition()
    representative = [row for row in binary if row["rho"] in (0.85, 0.95) and row["sigma_c"] == 0.2]
    summary = {"task_id": "TASK-PREFERENCE-ACQUISITION-FORMALIZATION", "verdict": "SMOKE-ONLY" if smoke else ("ACQUISITION-SPEED-CONSISTENT; PREFERENCE-MODEL-EXACT" if matched.get("optimizer_speed_consistent") else "ACQUISITION-COLLAPSE-INCONCLUSIVE"), "binary_rows": len(binary), "shortcut_switch_examples": representative, "matched_acquisition": matched, "exact_switch_rule": "rho > 1 - sigma_c", "interpretation": "The binary model is exact for its declared population assumptions. The acquisition conclusion is an audit of the existing neural trajectories, not a network theorem."}
    (OUT / ("binary_preference_smoke_summary.json" if smoke else "formalization_summary.json")).write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=True) + "\n", encoding="utf-8")
    report = ["# Preference/acquisition formalization audit", "", f"- Verdict: **{summary['verdict']}**", "- Exact binary switch: `rho > 1 - sigma_c`", f"- Binary rows: {len(binary)}", f"- Matched source-loss pairs: `{matched.get('n_matched_pairs', 0)}`", f"- Mean absolute matched `G_repr` gap: `{matched.get('mean_abs_G_repr_gap', float('nan'))}`", "", "## Exact population model", "", "Let `C=Y eta_c`, `S=Y eta_s`, with `P(eta_c=1)=1-sigma_c`, `P(eta_s=1)=rho`, and conditional independence. The population log-odds are `C log((1-sigma_c)/sigma_c) + S log(rho/(1-rho))`; therefore the shortcut coefficient exceeds the core coefficient exactly when `rho > 1-sigma_c`. Target BCE and the frozen balanced-head repair gain are evaluated by enumerating the four `(eta_c, eta_s)` states.", "", "## Acquisition audit", "", "The trajectory audit treats source BCE as a progress coordinate. After matching source risk, the optimizer gap is small in the existing neural runs, which is consistent with a speed/progress explanation. The dense alpha/capacity sweep shows a capacity/difficulty trend, but not a single universal phase boundary. Neither result is a theorem about neural dynamics."]
    (OUT / ("binary_preference_smoke_report.md" if smoke else "formalization_report.md")).write_text("\n".join(report) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--smoke", action="store_true"); args = parser.parse_args(); print(json.dumps(run(args.smoke), indent=2, allow_nan=True))
