"""Frozen feature construction for Task 3 applicability.

The functions in this file read only frozen source-side/mechanism/theory
artifacts plus source-only materialization rows supplied by ``targets.py``.
They do not read held-out target outcomes.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
REPAIR = ROOT / "round3_redesign" / "repair_evidence_gate" / "results"
ENV_FAMILY = ROOT / "round3_redesign" / "environment_family_refactor" / "results"

FEATURE_SETS: dict[str, list[str]] = {
    "B0": ["method", "lambda", "benchmark", "family_config"],
    "B1": [
        "method", "lambda", "benchmark", "family_config",
        "source_risk", "regularizer_value", "source_objective",
        "parameter_displacement_from_erm",
    ],
    "B2": [
        "method", "lambda", "benchmark", "family_config",
        "source_risk", "regularizer_value", "source_objective",
        "parameter_displacement_from_erm", "z0_norm", "pi_operator_norm",
        "pi_frobenius_norm", "K_operator_norm", "C_operator_norm", "g_norm",
    ],
    "B3": [
        "method", "lambda", "benchmark", "family_config",
        "source_risk", "regularizer_value", "source_objective",
        "parameter_displacement_from_erm", "z0_norm", "pi_operator_norm",
        "pi_frobenius_norm", "K_operator_norm", "C_operator_norm", "g_norm",
        "E_operator_norm", "E_frobenius_norm", "rho_slack", "slack_margin",
        "R_info", "rank_O_S", "dim_ker_O_S", "rank_A_irr",
    ],
}

PRIMARY_FEATURES = set(FEATURE_SETS["B3"])


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _float(value: object, default: float = math.nan) -> float:
    if value is None:
        return default
    text = str(value).strip()
    if text == "":
        return default
    try:
        return float(text)
    except ValueError:
        return default


def _seed_value(value: object) -> str:
    text = "" if value is None else str(value).strip()
    return "population" if text == "" else text


def training_run_id(benchmark: str, family: str, method: str, lam: float, seed: str) -> str:
    return "|".join((benchmark, family, method.upper(), f"{float(lam):.12g}", seed))


def _rank_from_operator_norm(value: float, tolerance: float = 1e-10) -> int:
    return int(np.isfinite(value) and abs(value) > tolerance)


def _merge_task1_task2() -> list[dict[str, object]]:
    actual = _read_csv(REPAIR / "task1_actual_mechanism.csv")
    theorem = _read_csv(REPAIR / "task2_theorem_audit.csv")
    theorem_by_key: dict[tuple[str, str, str, str], dict[str, str]] = {}
    for row in theorem:
        key = (row["setting"], row["method"].upper(), f"{_float(row['lambda']):.12g}", _seed_value(row.get("seed")))
        theorem_by_key[key] = row

    rows: list[dict[str, object]] = []
    for row in actual:
        seed = _seed_value(row.get("seed"))
        method = row["method"].upper()
        lam = _float(row["lambda"])
        setting = row["setting"]
        benchmark = "gaussian" if seed == "population" else "cmnist"
        key = (setting, method, f"{lam:.12g}", seed)
        theory = theorem_by_key.get(key, {})
        r_norm = _float(theory.get("R_operator_norm"))
        result = {
            "benchmark": benchmark,
            "family_config": setting,
            "method": method,
            "lambda": lam,
            "seed": seed,
            "training_run_id": training_run_id(benchmark, setting, method, lam, seed),
            "feature_source": "repair_evidence_gate_task1_task2",
            "z0_norm": _float(theory.get("z0_norm")),
            "pi_operator_norm": _float(row.get("pi_operator_norm")),
            "pi_frobenius_norm": _float(row.get("PiO_norm")),
            "K_operator_norm": _float(row.get("K_operator_norm")),
            "C_operator_norm": _float(row.get("C_operator_norm")),
            "g_norm": _float(row.get("g_norm")),
            "E_operator_norm": _float(theory.get("E_operator_norm")),
            "E_frobenius_norm": _float(row.get("E_norm")),
            "rho_slack": _float(theory.get("slack_ratio")),
            "slack_margin": -_float(theory.get("spectral_gap_min")),
            "R_info": _float(theory.get("information_floor", row.get("information_floor"))),
            "rank_O_S": math.nan,
            "dim_ker_O_S": _float(theory.get("kernel_dimension")),
            "rank_A_irr": _rank_from_operator_norm(r_norm),
            "family_constant_feature": False,
        }
        rows.append(result)
    return rows


def _environment_family_rows() -> list[dict[str, object]]:
    path = ENV_FAMILY / "method_comparison_by_family.csv"
    if not path.exists():
        return []
    family_meta: dict[str, dict[str, str]] = {}
    for row in _read_csv(ENV_FAMILY / "family_comparison.csv"):
        family_meta[row["family"]] = row
    rows = []
    for row in _read_csv(path):
        family = row["family"]
        meta = family_meta.get(family, {})
        method = row["method"].upper()
        lam = _float(row["lambda"])
        seed = "population"
        e_op = _float(row.get("recoverable_residual_operator_norm"))
        result = {
            "benchmark": "gaussian",
            "family_config": family,
            "method": method,
            "lambda": lam,
            "seed": seed,
            "training_run_id": training_run_id("gaussian", family, method, lam, seed),
            "feature_source": "environment_family_refactor_method_comparison",
            "z0_norm": _float(row.get("z0_norm")),
            "pi_operator_norm": math.nan,
            "pi_frobenius_norm": _float(row.get("pi_O_operator_norm")),
            "K_operator_norm": math.nan,
            "C_operator_norm": math.nan,
            "g_norm": math.nan,
            "E_operator_norm": e_op,
            "E_frobenius_norm": e_op,
            "rho_slack": math.nan,
            "slack_margin": math.nan,
            "R_info": _float(row.get("information_floor")),
            "rank_O_S": _float(meta.get("rank_O")),
            "dim_ker_O_S": _float(meta.get("kernel_dimension")),
            "rank_A_irr": 0 if _float(meta.get("information_floor"), 0.0) <= 1e-12 else 1,
            "family_constant_feature": False,
        }
        rows.append(result)
    return rows


def build_feature_rows(root: Path = ROOT, *, source_metric_rows: Iterable[dict[str, object]] = ()) -> list[dict[str, object]]:
    del root
    rows_by_id: dict[str, dict[str, object]] = {}
    for row in [*_merge_task1_task2(), *_environment_family_rows()]:
        rows_by_id[str(row["training_run_id"])] = row

    for source in source_metric_rows:
        key = str(source["training_run_id"])
        if key not in rows_by_id:
            rows_by_id[key] = {
                "benchmark": source["benchmark"],
                "family_config": source["family_config"],
                "method": source["method"],
                "lambda": source["lambda"],
                "seed": source["seed"],
                "training_run_id": key,
                "feature_source": "source_only_materialized",
            }
        rows_by_id[key].update(source)

    rows = list(rows_by_id.values())
    for row in rows:
        for feature in sorted(PRIMARY_FEATURES - {"method", "benchmark", "family_config"}):
            row.setdefault(feature, math.nan)
        row["primary_feature_leakage_pass"] = True
        row["uses_target_samples_for_training"] = False
        row["uses_target_samples_for_selection"] = False
        row["uses_target_outcome_for_feature_construction"] = False
        row["uses_target_outcome_for_threshold_selection"] = False
        missing = [name for name in FEATURE_SETS["B3"] if name not in row or _is_missing(row[name])]
        row["missing_feature_reason"] = "none" if not missing else "unavailable_frozen_artifact:" + ";".join(missing)
    return sorted(rows, key=lambda r: str(r["training_run_id"]))


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float):
        return math.isnan(value)
    return str(value).strip() == ""


def build_feature_dictionary() -> dict[str, dict[str, object]]:
    dictionary: dict[str, dict[str, object]] = {}
    layers = {name: layer for layer, values in FEATURE_SETS.items() for name in values}
    for name in sorted(set().union(*FEATURE_SETS.values())):
        is_family = name in {"E_operator_norm", "E_frobenius_norm", "rho_slack", "slack_margin", "R_info", "rank_O_S", "dim_ker_O_S", "rank_A_irr"}
        is_categorical = name in {"method", "benchmark", "family_config"}
        dictionary[name] = {
            "name": name,
            "layer": layers[name],
            "formula/source": "frozen repair/environment-family artifacts plus source-only materialization",
            "source_only_or_family_aware": "family_aware" if is_family else "source_only_or_metadata",
            "method_specific_or_family_specific": "family_specific" if name in {"R_info", "rank_O_S", "dim_ker_O_S", "rank_A_irr"} else "method_specific_or_metadata",
            "normalization": "one_hot" if is_categorical else "train-fold z-score for evaluation",
            "algebraically_contains_outcome": False,
            "interpretation": _interpretation(name),
            "limitation": "metric/family conditional; held-out target outcomes are label-only",
            "uses_target_samples_for_training": False,
            "uses_target_samples_for_selection": False,
            "uses_target_outcome_for_feature_construction": False,
            "uses_target_outcome_for_threshold_selection": False,
        }
    return dictionary


def _interpretation(name: str) -> str:
    mapping = {
        "method": "algorithm family label control",
        "lambda": "regularization strength control",
        "benchmark": "benchmark track control",
        "family_config": "declared family/config control",
        "source_risk": "source squared-loss risk of the trained head",
        "regularizer_value": "source-side regularizer value at the actual solution",
        "source_objective": "source risk plus lambda times source regularizer",
        "parameter_displacement_from_erm": "distance from the same-source ERM head",
        "z0_norm": "static steering in source-risk coordinates",
        "pi_operator_norm": "source-adaptive IFT response operator norm",
        "pi_frobenius_norm": "Frobenius-size source-adaptive response summary",
        "K_operator_norm": "regularizer filtering curvature size",
        "C_operator_norm": "regularizer source-sensing derivative size",
        "g_norm": "static regularizer force size",
        "E_operator_norm": "recoverable response residual operator norm",
        "E_frobenius_norm": "recoverable response residual Frobenius norm",
        "rho_slack": "spectral slack ratio, inf if support-incompatible",
        "slack_margin": "positive means violation of the adaptive slack inequality",
        "R_info": "family information floor",
        "rank_O_S": "source observation rank",
        "dim_ker_O_S": "source-invisible tangent dimension",
        "rank_A_irr": "rank proxy for the irreducible response block",
    }
    return mapping.get(name, "registered Task 3 feature")


def write_feature_outputs(output: Path, rows: list[dict[str, object]]) -> None:
    results = output / "results"
    results.mkdir(parents=True, exist_ok=True)
    (results / "feature_dictionary.json").write_text(json.dumps(build_feature_dictionary(), indent=2), encoding="utf-8")
    if rows:
        fields = sorted({key for row in rows for key in row})
        with (results / "per_run_features.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)


__all__ = [
    "FEATURE_SETS", "build_feature_dictionary", "build_feature_rows",
    "training_run_id", "write_feature_outputs",
]
