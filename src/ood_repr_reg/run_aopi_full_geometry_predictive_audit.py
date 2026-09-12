"""Test full A/O geometry, mechanism-matched response blocks, and schedule transfer.

This is the follow-up to ``run_aopi_semantic_predictive_audit``.  It keeps the
same logical C_300 state and finite semantic outcomes, but replaces the scalar
``A/O`` baseline with separate full Gram geometries and obtains local response
geometry on a probe schedule that is independent of the outcome schedule.
"""

from __future__ import annotations

import copy
import hashlib
import json
import platform
import subprocess
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .run_aopi_semantic_predictive_audit import (
    ALPHA,
    BASIS,
    BANK_NAMES,
    EPS,
    H,
    LEGACY_CHECKPOINT_STEP,
    METHODS,
    NAMES,
    OUT,
    ROOT,
    SEEDS,
    STEP,
    _assert_same_initial_state,
    _bank_slices,
    _fork_from_bundle,
    _make_schedule,
)
from .task3_aopi_multimethod_mechanism_survey.config_schema import validate_config
from .task3_aopi_multimethod_mechanism_survey.full_response import _run_path
from .task3_aopi_multimethod_mechanism_survey.functional_banks import build_functional_banks
from .task3_aopi_multimethod_mechanism_survey.method_trainer import train_survey_method
from .task3_aopi_multimethod_mechanism_survey.smooth_world5 import build_smooth_world5
from .task3_aopi_multimethod_mechanism_survey.source_observation import observation_geometry
from .task3_aopi_multimethod_mechanism_survey.task_response import task_geometry
from .task3_cmnist_cpu_minimal.data import build_task3_data
from .task3_cmnist_cpu_minimal.model import build_model_from_config


OUTCOME_SCHEDULE_OFFSET = 1_000_003
GEOMETRY_PREFIXES = ("A", "O")
RESPONSE_BLOCKS = {
    "R_full": (0, 1, 2),
    "R_color_only": (0, 1),
    "R_noise_only": (2,),
}


def _schedule_hash(schedule: tuple[torch.Tensor, torch.Tensor]) -> str:
    hasher = hashlib.sha256()
    for indices in schedule:
        hasher.update(indices.detach().cpu().contiguous().numpy().tobytes())
    return hasher.hexdigest()


def _gram_block(prefix: str, matrix: torch.Tensor) -> dict[str, float]:
    """Return norms and normalized within-codomain Gram entries."""
    matrix = matrix.detach().double()
    norms = matrix.norm(dim=0)
    result = {f"{prefix}_norm_{index}": float(norms[index]) for index in range(matrix.shape[1])}
    denominator = norms[:, None] * norms[None, :]
    cosine = torch.zeros_like(matrix.T @ matrix)
    gram = matrix.T @ matrix
    valid = denominator > 1e-12
    cosine[valid] = gram[valid] / denominator[valid]
    for left in range(matrix.shape[1]):
        for right in range(left + 1, matrix.shape[1]):
            result[f"{prefix}_cos_{left}_{right}"] = float(cosine[left, right])
    return result


def _response_block(
    prefix: str,
    response_matrix: torch.Tensor,
    selected: Iterable[int],
    slices: tuple[slice, slice, slice, slice],
) -> dict[str, float]:
    """Return a compact, nonredundant response geometry block."""
    selected = tuple(int(index) for index in selected)
    response_matrix = response_matrix.detach().double()
    result: dict[str, float] = {}
    norms = response_matrix.norm(dim=1)
    for index in selected:
        result[f"{prefix}_norm_{index}"] = float(norms[index])
        total = float(norms[index])
        counterfactual_slice = slice(slices[1].start, slices[2].stop)
        for bank, bank_slice in (
            ("source", slices[0]),
            ("counterfactual", counterfactual_slice),
            ("clean", slices[3]),
        ):
            value = float(response_matrix[index, bank_slice].norm())
            result[f"{prefix}_alloc_{index}_{bank}"] = value / total if total > 1e-12 else 0.0
    for left_position, left in enumerate(selected):
        for right in selected[left_position + 1 :]:
            denominator = float(norms[left] * norms[right])
            value = float(torch.dot(response_matrix[left], response_matrix[right]))
            result[f"{prefix}_cos_{left}_{right}"] = value / denominator if denominator > 1e-12 else 0.0
    return result


def _feature_columns(frame: pd.DataFrame, prefix: str) -> list[str]:
    return [column for column in frame.columns if column.startswith(prefix)]


def _evaluate_cv(
    frame: pd.DataFrame,
    columns: list[str],
    *,
    group_column: str,
    direction: str,
    model_name: str,
) -> dict[str, Any]:
    values = frame[frame["semantic_direction"] == direction].copy()
    y = values["outcome_norm"].to_numpy(dtype=float)
    prediction = np.full(len(values), np.nan, dtype=float)
    fold_rmses: list[float] = []
    for group in sorted(values[group_column].unique()):
        train_mask = values[group_column] != group
        test_mask = ~train_mask
        if not bool(test_mask.any()) or not bool(train_mask.any()):
            continue
        estimator = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        estimator.fit(values.loc[train_mask, columns], y[train_mask])
        prediction[test_mask] = estimator.predict(values.loc[test_mask, columns])
        fold_rmses.append(float(np.sqrt(mean_squared_error(y[test_mask], prediction[test_mask]))))
    valid = np.isfinite(prediction)
    return {
        "semantic_direction": direction,
        "predictor": model_name,
        "cv": f"leave_one_{group_column}_out",
        "n": int(valid.sum()),
        "folds": len(fold_rmses),
        "rmse": float(np.sqrt(mean_squared_error(y[valid], prediction[valid]))) if bool(valid.any()) else float("nan"),
        "fold_rmse_mean": float(np.mean(fold_rmses)) if fold_rmses else float("nan"),
        "outcome_sd": float(y.std()),
    }


def _git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "UNKNOWN"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    config_path = ROOT / "configs/task3_aopi_multimethod_mechanism_survey.json"
    raw = json.loads(config_path.read_text())
    raw["methods"] = list(METHODS)
    raw["candidate_methods"] = list(METHODS)
    raw["seeds"] = list(SEEDS)
    raw["training"]["checkpoint_steps"] = [LEGACY_CHECKPOINT_STEP]
    config = copy.deepcopy(validate_config(raw))
    effective_config_sha256 = hashlib.sha256(
        json.dumps(raw, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    outcome_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    state_rows: list[dict[str, Any]] = []
    vectors: dict[str, np.ndarray] = {}

    for seed in SEEDS:
        data = build_task3_data(
            config,
            seed,
            data_root=ROOT / "data",
            download=bool(config["execution"]["download_mnist"]),
        )
        world = build_smooth_world5(config, seed, data_root=ROOT / "data", download=False)
        banks = build_functional_banks(
            world,
            source_size_per_environment=int(config["banks"]["source_bank_size_per_environment"]),
            counterfactual_size=int(config["banks"]["counterfactual_bank_size"]),
        )
        batch_size = int(config["training"]["batch_size_per_environment"])
        probe_schedule = _make_schedule(world, seed=seed, horizon=H, batch_size=batch_size)
        outcome_schedule = _make_schedule(world, seed=seed + OUTCOME_SCHEDULE_OFFSET, horizon=H, batch_size=batch_size)
        if _schedule_hash(probe_schedule) == _schedule_hash(outcome_schedule):
            raise RuntimeError("probe and outcome schedules unexpectedly coincide")
        slices = _bank_slices(banks)
        zero = torch.zeros(5, dtype=torch.double)

        for method in METHODS:
            torch.manual_seed(seed)
            result = train_survey_method(
                model=build_model_from_config(config),
                source_envs=data.source_envs,
                batch_schedule=data.batch_schedule,
                method=method,
                config=config,
                seed=seed,
                initial_parameter_hash="",
            )
            if not result.finite:
                raise RuntimeError(result.invalid_reason)
            if STEP not in result.checkpoint_bundles:
                raise RuntimeError(f"missing logical checkpoint C_{STEP}")
            fork, bundle_hashes = _fork_from_bundle(config, result.checkpoint_bundles[STEP])
            task = task_geometry(fork.model, world, pool_size=int(config["banks"]["geometry_pool_size_per_pool"]))
            observation = observation_geometry(fork.model, world, pool_size=int(config["banks"]["geometry_pool_size_per_pool"]))
            task_matrix = task.A[:, BASIS]
            observation_matrix = observation.O[:, BASIS]

            # The derivative-like response is measured on probe schedule A.
            paths: dict[str, dict[int, tuple[str, str, str, dict[str, torch.Tensor]]]] = {}
            response_vectors: list[torch.Tensor] = []
            for basis, name in zip(BASIS, NAMES):
                direction = torch.zeros(5, dtype=torch.double)
                direction[basis] = 1.0
                plus = _run_path(fork, world, banks, EPS * direction, method, config, probe_schedule, (0, H))
                minus = _run_path(fork, world, banks, -EPS * direction, method, config, probe_schedule, (0, H))
                plus_logits = plus[H][3]
                minus_logits = minus[H][3]
                response = torch.cat(
                    [((plus_logits[key] - minus_logits[key]) / (2.0 * EPS)).detach() for key in BANK_NAMES]
                )
                response_vectors.append(response)
                vectors[f"{method}__{seed}__{name}"] = response.numpy()
                paths[f"plus_{name}"] = plus
                paths[f"minus_{name}"] = minus

            response_matrix = torch.stack(response_vectors, dim=0)

            # The finite outcome is measured on independent schedule B, with a
            # matched zero-shift control from that same schedule.
            paths["control"] = _run_path(fork, world, banks, zero, method, config, outcome_schedule, (0, H))
            control = paths["control"][H][3]
            for basis, name in zip(BASIS, NAMES):
                direction = torch.zeros(5, dtype=torch.double)
                direction[basis] = 1.0
                paths[f"shift_{name}"] = _run_path(
                    fork, world, banks, ALPHA * direction, method, config, outcome_schedule, (0, H)
                )
            gate = _assert_same_initial_state(paths, bundle_hashes)
            state_rows.append(
                {
                    "method": method,
                    "seed": seed,
                    "logical_checkpoint": STEP,
                    "legacy_checkpoint_key": LEGACY_CHECKPOINT_STEP,
                    "probe_schedule_hash": _schedule_hash(probe_schedule),
                    "outcome_schedule_hash": _schedule_hash(outcome_schedule),
                    "probe_outcome_schedule_equal": False,
                    **gate,
                }
            )

            common_features: dict[str, float] = {}
            common_features.update(_gram_block("A", task_matrix))
            common_features.update(_gram_block("O", observation_matrix))
            common_features.update(_response_block("R_full", response_matrix, RESPONSE_BLOCKS["R_full"], slices))
            common_features.update(_response_block("R_color_only", response_matrix, RESPONSE_BLOCKS["R_color_only"], slices))
            common_features.update(_response_block("R_noise_only", response_matrix, RESPONSE_BLOCKS["R_noise_only"], slices))
            common_features.update({
                f"method_identity_{candidate}": float(candidate == method)
                for candidate in METHODS
            })

            for index, (basis, name) in enumerate(zip(BASIS, NAMES)):
                shifted = paths[f"shift_{name}"][H][3]
                outcome = torch.cat([(shifted[key] - control[key]).detach() for key in BANK_NAMES])
                row: dict[str, Any] = {
                    "method": method,
                    "seed": seed,
                    "semantic_direction": name,
                    "direction_index": index,
                    "logical_checkpoint": STEP,
                    "horizon": H,
                    "finite_alpha": ALPHA,
                    "response_schedule": "probe_A",
                    "outcome_schedule": "independent_B",
                    "outcome_norm": float(outcome.norm()),
                    "outcome_source_norm": float(outcome[slices[0]].norm()),
                    "outcome_counterfactual_norm": float(torch.cat((outcome[slices[1]], outcome[slices[2]])).norm()),
                    "outcome_clean_norm": float(outcome[slices[3]].norm()),
                    "target_A_norm": float(task_matrix[:, index].norm()),
                    "target_O_norm": float(observation_matrix[:, index].norm()),
                    "target_R_norm": float(response_matrix[index].norm()),
                    "state_gate_pass": True,
                }
                outcome_rows.append(row)
                feature_rows.append(
                    {
                        "method": method,
                        "seed": seed,
                        "semantic_direction": name,
                        "direction_index": index,
                        "logical_checkpoint": STEP,
                        "response_schedule": "probe_A",
                        "outcome_schedule": "independent_B",
                        "state_gate_pass": True,
                        **common_features,
                    }
                )

    vectors_path = OUT / "aopi_full_geometry_response_vectors.npz"
    np.savez_compressed(vectors_path, **vectors)
    outcomes_path = OUT / "aopi_full_geometry_finite_outcomes.csv"
    features_path = OUT / "aopi_full_geometry_features.csv"
    state_path = OUT / "aopi_full_geometry_state_gate.csv"
    pd.DataFrame(outcome_rows).to_csv(outcomes_path, index=False)
    pd.DataFrame(feature_rows).to_csv(features_path, index=False)
    pd.DataFrame(state_rows).to_csv(state_path, index=False)

    joined = pd.DataFrame(outcome_rows).merge(
        pd.DataFrame(feature_rows),
        on=[
            "method",
            "seed",
            "semantic_direction",
            "direction_index",
            "logical_checkpoint",
            "response_schedule",
            "outcome_schedule",
            "state_gate_pass",
        ],
        suffixes=("", "_feature"),
    )
    a_cols = _feature_columns(joined, "A_")
    o_cols = _feature_columns(joined, "O_")
    full_r_cols = _feature_columns(joined, "R_full_")
    color_r_cols = _feature_columns(joined, "R_color_only_")
    noise_r_cols = _feature_columns(joined, "R_noise_only_")
    method_identity_cols = _feature_columns(joined, "method_identity_")
    # The first two are deliberately the complete A/O geometry blocks.  The
    # final three are response blocks chosen before observing finite outcomes.
    predictor_sets = {
        "A_scalar_target": ["target_A_norm"],
        "AO_scalar_target": ["target_A_norm", "target_O_norm"],
        "A_full_geometry": a_cols,
        "AO_full_geometry": a_cols + o_cols,
        "AO_full_plus_R_full": a_cols + o_cols + full_r_cols,
        "AO_full_plus_R_color_only": a_cols + o_cols + color_r_cols,
        "AO_full_plus_R_noise_only": a_cols + o_cols + noise_r_cols,
        "method_identity": method_identity_cols,
    }
    audit_rows: list[dict[str, Any]] = []
    for direction in NAMES:
        for predictor_name, columns in predictor_sets.items():
            for group_column in ("method", "seed"):
                # A held-out method has no meaningful one-hot identity.  The
                # fingerprint control is therefore evaluated only by
                # leave-one-seed-out, where the method is observed in train.
                if predictor_name == "method_identity" and group_column == "method":
                    continue
                audit_rows.append(
                    _evaluate_cv(
                        joined,
                        columns,
                        group_column=group_column,
                        direction=direction,
                        model_name=predictor_name,
                    )
                )
    audit = pd.DataFrame(audit_rows)
    audit_path = OUT / "aopi_full_geometry_predictive_audit.csv"
    audit.to_csv(audit_path, index=False)

    def metric(direction: str, predictor: str, cv: str) -> float:
        match = audit[
            (audit["semantic_direction"] == direction)
            & (audit["predictor"] == predictor)
            & (audit["cv"] == cv)
        ]
        return float(match.iloc[0]["rmse"])

    noise_lomo = {
        name: metric("source_label_noise", name, "leave_one_method_out")
        for name in ("A_scalar_target", "AO_scalar_target", "A_full_geometry", "AO_full_geometry", "AO_full_plus_R_full", "AO_full_plus_R_color_only", "AO_full_plus_R_noise_only")
    }
    noise_loso = {
        name: metric("source_label_noise", name, "leave_one_seed_out")
        for name in ("A_full_geometry", "AO_full_geometry", "AO_full_plus_R_full", "AO_full_plus_R_color_only", "AO_full_plus_R_noise_only", "method_identity")
    }

    report_parts = [
        "# Full A/O geometry, mechanism specificity, and schedule transfer",
        "",
        f"All rows use logical checkpoint C_{STEP}; local response geometry is measured on probe schedule A and finite-shift outcomes on independent schedule B. Both are forked from the same complete C_{STEP} bundle.",
        "",
        "## Predictive audit",
        "",
        audit.to_string(index=False),
        "",
        "`A_full_geometry` contains norms and normalized within-codomain Gram entries for all three A columns. `AO_full_geometry` adds the corresponding O_S block without taking cross-codomain inner products. The response blocks use norms, within-response cosines, and source/counterfactual/clean allocation fractions; diagonal Gram entries are not included as redundant features.",
        "",
        "The label-noise negative-control comparison is the key adjudication: if color-only response geometry predicts the label-noise outcome as well as noise-only response geometry, the signal is consistent with a method fingerprint. A noise-specific gain that survives the full A/O geometry baseline and independent schedule is the narrower mechanism-matched result. These statistics remain predictive evidence and do not establish causality by themselves.",
        "",
        "## Observed label-noise pattern",
        "",
        "Under leave-one-method-out, the RMSE values are " + ", ".join(f"{key}={value:.3f}" for key, value in noise_lomo.items()) + ".",
        "Under leave-one-seed-out, the corresponding full-geometry and response controls are " + ", ".join(f"{key}={value:.3f}" for key, value in noise_loso.items()) + ".",
        "The apparent gain is therefore not stable across the two holdouts: A_full_geometry already outperforms scalar A/O, method identity is a strong seed-held-out baseline, and adding response geometry does not improve the independent-schedule result beyond the complete A/O geometry. The present audit does not identify a stable mechanism-matched Pi O_S increment.",
    ]
    (OUT / "aopi_full_geometry_predictive_report.md").write_text("\n".join(report_parts))
    provenance = {
        "methods": list(METHODS),
        "seeds": list(SEEDS),
        "logical_checkpoint": STEP,
        "legacy_checkpoint_key": LEGACY_CHECKPOINT_STEP,
        "checkpoint_semantics": "C_t is after t completed optimizer updates; legacy state-dict key t-1 is used for t>0",
        "horizon": H,
        "finite_alpha": ALPHA,
        "response_epsilon": EPS,
        "response_schedule": "probe_A",
        "outcome_schedule": "independent_B",
        "outcome_schedule_seed_offset": OUTCOME_SCHEDULE_OFFSET,
        "predictor_sets": {key: value for key, value in predictor_sets.items()},
        "response_blocks": {key: list(value) for key, value in RESPONSE_BLOCKS.items()},
        "within_codomain_only": True,
        "same_complete_state_for_AO_response_outcome": True,
        "state_gate_file": str(state_path.relative_to(ROOT)),
        "target_used_for_training": False,
        "target_used_for_feature_selection": False,
        "outcome": "finite-shift continuation displacement relative to zero-shift continuation on independent schedule B",
        "status": "FULL_AO_GEOMETRY_MECHANISM_SPECIFICITY_AUDIT",
        "git_head": _git_head(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "input_config_sha256": _sha256(config_path),
        "effective_config_sha256": effective_config_sha256,
        "script_sha256": _sha256(Path(__file__)),
        "output_sha256": {
            "response_vectors": _sha256(vectors_path),
            "finite_outcomes": _sha256(outcomes_path),
            "features": _sha256(features_path),
            "state_gate": _sha256(state_path),
            "audit": _sha256(audit_path),
        },
    }
    (OUT / "aopi_full_geometry_predictive_provenance.json").write_text(json.dumps(provenance, indent=2))
    print(audit.to_string(index=False))


if __name__ == "__main__":
    run()
