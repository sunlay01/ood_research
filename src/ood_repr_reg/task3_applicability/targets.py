"""Legal held-out target generation for Task 3.

The source materialization step computes source-only quantities and stores the
objects needed later for target evaluation.  Held-out target outcomes are not
computed until after preregistration and feature output have been written.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..cmnist_feature_probe import deterministic_subset, load_mnist_tensors, train_model
from ..cmnist_geometry_bridge import (
    CMNISTFamily,
    RepresentationBank,
    head_from_state,
    moment_state,
    source_state_stack,
    target_state,
)
from ..environment_family.legacy_gaussian import legacy_gaussian_family
from ..environment_family.source_induced import build_source_induced_family
from ..round3r_3b_benchmark import MomentState, environment_state, risk
from ..round3r_3c_affine import (
    _solve_regularized,
    regularizer_value_gradient,
    source_objective,
    source_task_state_matrix,
)
from ..round3r_3e_c_benchmarks import (
    LegalWorld,
    family_world,
    primary_hidden_u_world,
    primary_u_exposed_world,
)
from ..round3r_3e_world_tangent import perturb_environment
from ..run_cmnist_feature_probe import _build_data
from .features import training_run_id

Array = np.ndarray
ROOT = Path(__file__).resolve().parents[3]


@dataclass
class MaterializedRun:
    benchmark: str
    family_config: str
    method: str
    lam: float
    seed: str
    training_run_id: str
    weights: Array
    erm_weights: Array
    source_risk: float
    regularizer_value: float
    source_objective: float
    parameter_displacement_from_erm: float
    world: LegalWorld | None = None
    family: Any | None = None
    bank: RepresentationBank | None = None

    def source_feature_row(self) -> dict[str, object]:
        return {
            "benchmark": self.benchmark,
            "family_config": self.family_config,
            "method": self.method,
            "lambda": self.lam,
            "seed": self.seed,
            "training_run_id": self.training_run_id,
            "source_risk": self.source_risk,
            "regularizer_value": self.regularizer_value,
            "source_objective": self.source_objective,
            "parameter_displacement_from_erm": self.parameter_displacement_from_erm,
        }


def build_design(*, smoke: bool = False) -> dict[str, object]:
    cmnist_seeds = [0] if smoke else [0, 1, 2, 3, 4]
    return {
        "task_id": "TASK3-APPLICABILITY",
        "baseline_commit": "c45c071",
        "preregistration_barrier": "written_before_feature_and_target_outcome_generation",
        "smoke": smoke,
        "methods": ["L2", "IRMV1", "VREX"],
        "primary_erm_representation": {"method": "L2", "lambda": 0.0},
        "lambda_grid": {
            "default": [0.0, 0.001, 0.01, 0.1],
            "environment_family_frozen_comparison": [0.0, 0.01, 0.1],
        },
        "benchmarks": {
            "gaussian": [
                "hidden_u_hurts",
                "u_exposed_complete_information",
                "legacy_gaussian_mechanism",
                "source_induced",
            ],
            "cmnist": [
                "declared_source_target_coupled_correlation",
                "mechanism_defined_hidden",
                "mechanism_defined_exposed",
                "irrelevant_source_diversity",
            ],
        },
        "cmnist_seeds": cmnist_seeds,
        "target_radius_grid": {
            "gaussian": {"local": 0.25, "medium": 0.5, "larger_but_legal": 1.0},
            "cmnist": {"near_source": 0.02, "moderate": 0.05, "stronger_legal_shift": 0.08},
        },
        "feature_sets": {
            "B0": "method/lambda/benchmark/family metadata baseline",
            "B1": "B0 plus conventional source-only source risk/objective/regularizer/displacement",
            "B2": "B1 plus actual-solution mechanism summaries",
            "B3": "B2 plus family-aware E/slack/information summaries",
        },
        "outcomes": [
            "target_squared_loss_risk",
            "cmnist_target_accuracy",
            "delta_target_risk_vs_erm",
            "win_loss_vs_erm",
            "target_ranking",
            "finite_set_worst_case_risk",
        ],
        "cv_grouping": {
            "cmnist": "leave_one_seed_out",
            "gaussian": "leave_one_family_config_out_with_target_direction_audit",
            "never_split_key": "training_run_id",
        },
        "primary_metrics": {
            "ranking": "Kendall tau",
            "win_loss": "grouped out-of-sample AUC / balanced accuracy / Brier",
            "risk_prediction": "grouped out-of-sample R2 / MAE / rank correlation",
        },
        "null_controls": [
            "permuted_theory_features_within_strata",
            "random_feature_block_same_size_as_B3_increment",
            "method_lambda_only_explanation",
            "within_method_and_family_centered_audits",
        ],
        "verdict_thresholds": {
            "support": "B3 improves over B1 on both Gaussian and CMNIST in at least two primary tests, beats null medians, survives method/lambda conditioning, and has >=60% stable fold direction.",
            "partial": "Non-leaky evidence exists but is one-benchmark, one-test, weak, unstable, or substantially confounded.",
            "fail": "Leaky/tautological analysis, B3 not better than B1/nulls, unstable ranking, or only method/lambda/family constants explain results.",
        },
        "target_outcome_use": "offline_labels_only",
        "no_new_algorithms_or_families": True,
    }


def _grid(default_zero_for_all_methods: bool = False) -> list[tuple[str, float]]:
    if default_zero_for_all_methods:
        return [(m, lam) for m in ("L2", "IRMV1", "VREX") for lam in (0.0, 0.01, 0.1)]
    return [("L2", 0.0)] + [(m, lam) for m in ("L2", "IRMV1", "VREX") for lam in (0.001, 0.01, 0.1)]


def gaussian_worlds() -> dict[str, LegalWorld]:
    return {
        "hidden_u_hurts": primary_hidden_u_world(),
        "u_exposed_complete_information": primary_u_exposed_world(),
        "legacy_gaussian_mechanism": family_world(legacy_gaussian_family()),
        "source_induced": family_world(build_source_induced_family()),
    }


def cmnist_families() -> dict[str, CMNISTFamily]:
    return {
        "declared_source_target_coupled_correlation": CMNISTFamily(
            "declared_source_target_coupled_correlation",
            directions=("rho_source_1", "rho_source_2"),
        ),
        "mechanism_defined_hidden": CMNISTFamily("mechanism_defined_hidden"),
        "mechanism_defined_exposed": CMNISTFamily("mechanism_defined_exposed", hidden_exposed=True),
        "irrelevant_source_diversity": CMNISTFamily(
            "irrelevant_source_diversity",
            directions=("rho_source_1", "rho_source_2", "rho_hidden", "brightness_nuisance"),
        ),
    }


def _gaussian_source_metrics(world: LegalWorld, method: str, lam: float) -> MaterializedRun:
    weights, valid, status = _solve_regularized(world.benchmark, method.lower(), lam)
    if not valid:
        raise RuntimeError(status)
    erm, ok, _ = _solve_regularized(world.benchmark, "l2", 0.0)
    if not ok:
        raise RuntimeError("ERM source solve failed")
    vector = source_task_state_matrix(world.benchmark)
    p = world.benchmark.optimum.size
    source_risk_value = risk(weights, world.benchmark.source)
    regularizer_value = regularizer_value_gradient(method.lower(), weights, vector, p)[0]
    objective = source_objective(weights, vector, p, method.lower(), lam)
    seed = "population"
    run_id = training_run_id("gaussian", world.name, method, lam, seed)
    return MaterializedRun(
        "gaussian", world.name, method.upper(), float(lam), seed, run_id,
        np.asarray(weights), np.asarray(erm), source_risk_value, regularizer_value,
        objective, float(np.linalg.norm(weights - erm)), world=world,
    )


def _cmnist_regularizer_value(weights: Array, state: Array, dimension: int, method: str) -> float:
    width = dimension * (dimension + 1) // 2 + dimension + 1
    risks = []
    gradients = []
    del gradients
    from ..cmnist_geometry_bridge import decode_svec

    blocks = []
    for start in range(0, state.size, width):
        block = state[start:start + width]
        nmat = dimension * (dimension + 1) // 2
        blocks.append((decode_svec(block, dimension), block[nmat:nmat + dimension], block[-1]))
    for matrix, cross, constant in blocks:
        risks.append(float(weights @ matrix @ weights - 2.0 * weights @ cross + constant))
    values = np.asarray(risks)
    key = method.lower()
    if key == "l2":
        return 0.5 * float(weights @ weights)
    if key == "vrex":
        centered = values - values.mean()
        return float(np.mean(centered ** 2))
    if key == "irmv1":
        radial = np.asarray([2.0 * (weights @ matrix @ weights - weights @ cross) for matrix, cross, _ in blocks])
        return float(np.mean(radial ** 2))
    return 0.0


def _cmnist_source_risk(weights: Array, state: Array, dimension: int) -> float:
    width = dimension * (dimension + 1) // 2 + dimension + 1
    from ..cmnist_geometry_bridge import decode_svec

    values = []
    for start in range(0, state.size, width):
        block = state[start:start + width]
        nmat = dimension * (dimension + 1) // 2
        matrix = decode_svec(block, dimension)
        cross = block[nmat:nmat + dimension]
        values.append(float(weights @ matrix @ weights - 2.0 * weights @ cross + block[-1]))
    return float(np.mean(values))


def _cmnist_run(bank: RepresentationBank, family: CMNISTFamily, method: str, lam: float, seed: int) -> MaterializedRun:
    state = source_state_stack(bank, family, np.zeros(family.dimension))
    p = bank.dimension
    weights = head_from_state(state, p, method=method.lower(), lam=lam)
    erm = head_from_state(state, p, method="erm", lam=0.0)
    source_risk_value = _cmnist_source_risk(weights, state, p)
    regularizer_value = _cmnist_regularizer_value(weights, state, p, method)
    objective = source_risk_value + float(lam) * regularizer_value
    seed_text = str(seed)
    run_id = training_run_id("cmnist", family.name, method, lam, seed_text)
    return MaterializedRun(
        "cmnist", family.name, method.upper(), float(lam), seed_text, run_id,
        np.asarray(weights), np.asarray(erm), source_risk_value, regularizer_value,
        objective, float(np.linalg.norm(weights - erm)), family=family, bank=bank,
    )


def materialize_training_runs(design: dict[str, object], root: Path = ROOT) -> list[MaterializedRun]:
    runs: list[MaterializedRun] = []
    worlds = gaussian_worlds()
    for name, world in worlds.items():
        family_grid = _grid(default_zero_for_all_methods=name in {"legacy_gaussian_mechanism", "source_induced"})
        for method, lam in family_grid:
            runs.append(_gaussian_source_metrics(world, method, lam))

    config = json.loads((root / "configs" / "cmnist_vis_001_main.json").read_text())
    import torch
    from ..cmnist_geometry_bridge import feature_bank

    device = torch.device(config.get("device", "cpu"))
    gray_all, digit_all = load_mnist_tensors(root / config["data_root"], train=True, download=config["download"])
    bank_size = int(config.get("bridge_bank_size", 1200))
    if bool(design.get("smoke")):
        bank_size = min(bank_size, 256)
    gray, digit = deterministic_subset(gray_all, digit_all, n=bank_size, seed=3711, offset=0)
    epochs = min(int(config["epochs"]), 2) if bool(design.get("smoke")) else int(config["epochs"])
    families = cmnist_families()
    for seed in [int(value) for value in design["cmnist_seeds"]]:
        train, _, _, _ = _build_data(config, seed)
        model, _ = train_model(
            train, method="erm", strength=0.0, latent_dim=int(config["latent_dim"]),
            epochs=epochs, batch_size=int(config["batch_size"]),
            learning_rate=float(config["learning_rate"]), seed=seed, device=device,
        )
        bank = feature_bank(model, gray, digit, device=device)
        for family in families.values():
            for method, lam in _grid():
                runs.append(_cmnist_run(bank, family, method, lam, seed))
    return runs


def _target_risk_gaussian(run: MaterializedRun, target: MomentState) -> float:
    return risk(run.weights, target)


def _gaussian_target_rows(run: MaterializedRun, design: dict[str, object]) -> list[dict[str, object]]:
    assert run.world is not None
    world = run.world
    rows: list[dict[str, object]] = []
    family = getattr(world, "family", None)
    if family is not None:
        spec = family.tangent_spec(world.base)
        direction_names = spec.directions
        def make_env(direction: str, signed: float):
            return family.perturb(world.base, direction, signed, role="target")
    else:
        direction_names = world.spec.directions
        def make_env(direction: str, signed: float):
            return perturb_environment(world.base, world.spec, direction, signed, expose_u_at_source=True)
    for radius_name, radius in dict(design["target_radius_grid"]["gaussian"]).items():
        for direction in direction_names:
            for sign in (-1.0, 1.0):
                try:
                    environment = make_env(direction, sign * float(radius))
                    state = environment_state(environment)
                    target_id = f"{run.family_config}:{radius_name}:{direction}:{sign:+.0f}"
                    rows.append({
                        "training_run_id": run.training_run_id,
                        "benchmark": "gaussian",
                        "family_config": run.family_config,
                        "method": run.method,
                        "lambda": run.lam,
                        "seed": run.seed,
                        "target_id": target_id,
                        "target_group": direction,
                        "radius_regime": radius_name,
                        "target_radius": float(radius),
                        "target_direction": direction,
                        "target_sign": int(sign),
                        "target_squared_loss_risk": _target_risk_gaussian(run, state),
                        "target_accuracy": math.nan,
                        "target_outcome_generation": "legal_gaussian_environment_family",
                    })
                except Exception as exc:
                    rows.append({
                        "training_run_id": run.training_run_id,
                        "benchmark": "gaussian",
                        "family_config": run.family_config,
                        "method": run.method,
                        "lambda": run.lam,
                        "seed": run.seed,
                        "target_id": f"{run.family_config}:{radius_name}:{direction}:{sign:+.0f}",
                        "target_group": direction,
                        "radius_regime": radius_name,
                        "target_radius": float(radius),
                        "target_direction": direction,
                        "target_sign": int(sign),
                        "target_squared_loss_risk": math.nan,
                        "target_accuracy": math.nan,
                        "invalid_target_reason": repr(exc),
                    })
    return rows


def _cmnist_accuracy(bank: RepresentationBank, weights: Array, rho: float) -> float:
    labels = bank.labels.astype(float)
    xr = np.column_stack((np.ones(len(labels)), bank.red))
    xg = np.column_stack((np.ones(len(labels)), bank.green))
    pred_r = (xr @ weights >= 0.5).astype(float)
    pred_g = (xg @ weights >= 0.5).astype(float)
    red_probability = np.where(labels < 0.5, rho, 1.0 - rho)
    correct_r = (pred_r == labels).astype(float)
    correct_g = (pred_g == labels).astype(float)
    return float(np.mean(red_probability * correct_r + (1.0 - red_probability) * correct_g))


def _cmnist_target_rows(run: MaterializedRun, design: dict[str, object]) -> list[dict[str, object]]:
    assert run.bank is not None and run.family is not None
    family = run.family
    rows: list[dict[str, object]] = []
    zero = np.zeros(family.dimension)
    base_rho = family.target_rho_at(zero)
    for radius_name, radius in dict(design["target_radius_grid"]["cmnist"]).items():
        for index, direction_name in enumerate(family.directions):
            for sign in (-1.0, 1.0):
                theta = np.zeros(family.dimension)
                theta[index] = sign * float(radius)
                target_id = f"{family.name}:{radius_name}:{direction_name}:{sign:+.0f}"
                try:
                    rho = family.target_rho_at(theta)
                    if abs(rho - base_rho) <= 1e-12:
                        raise ValueError("coordinate has no finite target generator in this family")
                    matrix, cross, constant = target_state(run.bank, family, theta)
                    target = MomentState(matrix, cross, constant)
                    rows.append({
                        "training_run_id": run.training_run_id,
                        "benchmark": "cmnist",
                        "family_config": family.name,
                        "method": run.method,
                        "lambda": run.lam,
                        "seed": run.seed,
                        "target_id": target_id,
                        "target_group": direction_name,
                        "radius_regime": radius_name,
                        "target_radius": float(radius),
                        "target_direction": direction_name,
                        "target_sign": int(sign),
                        "target_correlation": rho,
                        "target_squared_loss_risk": risk(run.weights, target),
                        "target_accuracy": _cmnist_accuracy(run.bank, run.weights, rho),
                        "target_outcome_generation": "legal_cmnist_family_parameterization",
                    })
                except Exception as exc:
                    rows.append({
                        "training_run_id": run.training_run_id,
                        "benchmark": "cmnist",
                        "family_config": family.name,
                        "method": run.method,
                        "lambda": run.lam,
                        "seed": run.seed,
                        "target_id": target_id,
                        "target_group": direction_name,
                        "radius_regime": radius_name,
                        "target_radius": float(radius),
                        "target_direction": direction_name,
                        "target_sign": int(sign),
                        "target_correlation": math.nan,
                        "target_squared_loss_risk": math.nan,
                        "target_accuracy": math.nan,
                        "invalid_target_reason": repr(exc),
                    })
    return rows


def _annotate_relative_outcomes(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    baseline: dict[tuple[str, str, str, str], float] = {}
    for row in rows:
        if row["method"] == "L2" and abs(float(row["lambda"])) <= 1e-15 and np.isfinite(float(row["target_squared_loss_risk"])):
            key = (row["benchmark"], row["family_config"], row["seed"], row["target_id"])
            baseline[key] = float(row["target_squared_loss_risk"])
    worst_by_run: dict[str, float] = {}
    for row in rows:
        value = float(row["target_squared_loss_risk"])
        if np.isfinite(value):
            worst_by_run[row["training_run_id"]] = max(value, worst_by_run.get(row["training_run_id"], -math.inf))
    for row in rows:
        key = (row["benchmark"], row["family_config"], row["seed"], row["target_id"])
        base = baseline.get(key, math.nan)
        risk_value = float(row["target_squared_loss_risk"])
        delta = risk_value - base if np.isfinite(risk_value) and np.isfinite(base) else math.nan
        row["delta_target_risk_vs_erm"] = delta
        row["win_loss_vs_erm"] = bool(delta < 0.0) if np.isfinite(delta) and not (row["method"] == "L2" and abs(float(row["lambda"])) <= 1e-15) else False
        row["finite_set_worst_case_risk"] = worst_by_run.get(row["training_run_id"], math.nan)
    groups: dict[tuple[str, str, str, str], list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault((row["benchmark"], row["family_config"], row["seed"], row["target_id"]), []).append(row)
    for group_rows in groups.values():
        valid = [row for row in group_rows if np.isfinite(float(row["target_squared_loss_risk"]))]
        ordered = sorted(valid, key=lambda row: (float(row["target_squared_loss_risk"]), row["method"], float(row["lambda"])))
        for rank, row in enumerate(ordered, start=1):
            row["target_rank"] = rank
        for row in group_rows:
            row.setdefault("target_rank", math.nan)
    return rows


def generate_target_outcomes(runs: list[MaterializedRun], design: dict[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for run in runs:
        if run.benchmark == "gaussian":
            rows.extend(_gaussian_target_rows(run, design))
        elif run.benchmark == "cmnist":
            rows.extend(_cmnist_target_rows(run, design))
        else:
            raise ValueError(f"unknown benchmark: {run.benchmark}")
    return _annotate_relative_outcomes(rows)


__all__ = [
    "MaterializedRun", "build_design", "cmnist_families", "gaussian_worlds",
    "generate_target_outcomes", "materialize_training_runs",
]
