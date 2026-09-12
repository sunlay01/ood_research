"""Audit whether local response geometry predicts finite semantic behavior.

The response and the finite continuation must be evaluated from exactly the
same complete training state.  In particular, ``STEP`` denotes the logical
state C_STEP (after STEP completed optimizer updates), while the legacy
trainer checkpoint key for that state is STEP - 1 because its loop is indexed
from zero.
"""

from __future__ import annotations

import copy
import hashlib
import json
import platform
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

from .task3_aopi_multimethod_mechanism_survey.config_schema import validate_config
from .task3_aopi_multimethod_mechanism_survey.full_response import _run_path
from .task3_aopi_multimethod_mechanism_survey.functional_banks import build_functional_banks
from .task3_aopi_multimethod_mechanism_survey.method_trainer import (
    optimizer_state_hash,
    train_survey_method,
)
from .task3_aopi_multimethod_mechanism_survey.smooth_world5 import build_smooth_world5
from .task3_aopi_multimethod_mechanism_survey.source_observation import observation_geometry
from .task3_aopi_multimethod_mechanism_survey.task_response import task_geometry
from .task3_cmnist_cpu_minimal.data import build_task3_data
from .task3_cmnist_cpu_minimal.model import build_model_from_config, parameter_hash


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "round3_redesign/semantic_mechanism_bridge"
CFG = ROOT / "configs/task3_aopi_multimethod_mechanism_survey.json"
METHODS = ("ERM", "CORAL", "IRMv1", "VREX", "FISHR")
SEEDS = (10, 11, 12, 13, 14)
STEP = 300
LEGACY_CHECKPOINT_STEP = STEP - 1
H = 20
EPS = 0.01
ALPHA = 0.1
BASIS = (0, 1, 3)
NAMES = ("source_env0_color", "source_env1_color", "source_label_noise")
BANK_NAMES = ("source", "counterfactual_red", "counterfactual_green", "clean_task")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        return "UNKNOWN"


def _make_schedule(world: Any, *, seed: int, horizon: int, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(int(seed) + 271828)
    return (
        torch.randint(len(world.source[0].digits), (horizon, batch_size), generator=generator),
        torch.randint(len(world.source[1].digits), (horizon, batch_size), generator=generator),
    )


def _fork_from_bundle(cfg: dict[str, Any], bundle: dict[str, Any]) -> tuple[SimpleNamespace, dict[str, str]]:
    """Materialize a continuation fork and verify its declared bundle hashes."""
    if int(bundle.get("checkpoint_step", -1)) != STEP:
        raise RuntimeError("checkpoint bundle does not identify logical C_300")
    if int(bundle.get("continuation_step", -1)) != STEP:
        raise RuntimeError("checkpoint bundle continuation step is not 300")

    model = build_model_from_config(cfg)
    model.load_state_dict(copy.deepcopy(bundle["model_state"]))
    model.cpu()

    saved_groups = bundle["optimizer_state"].get("param_groups", [])
    saved_lr = (
        float(saved_groups[0].get("lr", cfg["training"]["learning_rate"]))
        if saved_groups
        else float(cfg["training"]["learning_rate"])
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=saved_lr)
    optimizer.load_state_dict(copy.deepcopy(bundle["optimizer_state"]))
    algorithm_state = bundle["algorithm_state"].clone()

    hashes = {
        "parameter": parameter_hash(model),
        "optimizer": optimizer_state_hash(optimizer.state_dict()),
        "algorithm": algorithm_state.hash(),
    }
    declared = {
        "parameter": str(bundle.get("model_parameter_hash", hashes["parameter"])),
        "optimizer": str(bundle.get("optimizer_state_hash", hashes["optimizer"])),
        "algorithm": str(bundle.get("algorithm_state_hash", hashes["algorithm"])),
    }
    if hashes != declared:
        raise RuntimeError(f"checkpoint bundle hash mismatch: actual={hashes} declared={declared}")

    fork = SimpleNamespace(
        model=model,
        optimizer_state=optimizer.state_dict(),
        algorithm_state=algorithm_state,
        continuation_step=STEP,
    )
    return fork, hashes


def _assert_same_initial_state(
    paths: dict[str, dict[int, tuple[str, str, str, dict[str, torch.Tensor]]]],
    expected: dict[str, str],
) -> dict[str, Any]:
    """Hard gate: every response/outcome fork must start at the same C_300."""
    if not paths or any(0 not in path for path in paths.values()):
        raise RuntimeError("continuation path did not expose an initial snapshot")
    reference_label = next(iter(paths))
    reference = paths[reference_label][0]
    reference_hashes = {
        "parameter": reference[0],
        "optimizer": reference[1],
        "algorithm": reference[2],
    }
    if reference_hashes != expected:
        raise RuntimeError(
            f"reference continuation hash differs from bundle: {reference_hashes} vs {expected}"
        )

    mismatches: list[str] = []
    bank_equal = True
    for label, path in paths.items():
        snapshot = path[0]
        actual = {"parameter": snapshot[0], "optimizer": snapshot[1], "algorithm": snapshot[2]}
        if actual != expected:
            mismatches.append(f"{label}: {actual}")
        for bank_name in BANK_NAMES:
            if not torch.equal(snapshot[3][bank_name], reference[3][bank_name]):
                bank_equal = False
                mismatches.append(f"{label}:bank={bank_name}")
    if mismatches or not bank_equal:
        raise RuntimeError("INVALID-AO-STATE-MISMATCH: " + "; ".join(mismatches))
    return {
        "state_gate_pass": True,
        "parameter_hash": expected["parameter"],
        "optimizer_hash": expected["optimizer"],
        "algorithm_hash": expected["algorithm"],
        "bank_initial_logits_equal": bool(bank_equal),
    }


def _flat_bank_response(response: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat([response[name] for name in BANK_NAMES])


def _bank_slices(banks: Any) -> tuple[slice, slice, slice, slice]:
    n_source = len(banks.source)
    n_red = len(banks.counterfactual_red)
    n_green = len(banks.counterfactual_green)
    n_clean = len(banks.clean_task)
    return (
        slice(0, n_source),
        slice(n_source, n_source + n_red),
        slice(n_source + n_red, n_source + n_red + n_green),
        slice(n_source + n_red + n_green, n_source + n_red + n_green + n_clean),
    )


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    raw = json.loads(CFG.read_text())
    raw["methods"] = list(METHODS)
    raw["candidate_methods"] = list(METHODS)
    raw["seeds"] = list(SEEDS)
    # The trainer's legacy key is the loop index.  Key 299 is the state after
    # updates 0..299, i.e. the logical checkpoint C_300.
    raw["training"]["checkpoint_steps"] = [LEGACY_CHECKPOINT_STEP]
    cfg = copy.deepcopy(validate_config(raw))
    effective_config_sha256 = hashlib.sha256(
        json.dumps(raw, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    state_gate_rows: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {}

    for seed in SEEDS:
        data = build_task3_data(
            cfg,
            seed,
            data_root=ROOT / "data",
            download=bool(cfg["execution"]["download_mnist"]),
        )
        world = build_smooth_world5(cfg, seed, data_root=ROOT / "data", download=False)
        banks = build_functional_banks(
            world,
            source_size_per_environment=int(cfg["banks"]["source_bank_size_per_environment"]),
            counterfactual_size=int(cfg["banks"]["counterfactual_bank_size"]),
        )
        schedule = _make_schedule(
            world,
            seed=seed,
            horizon=H,
            batch_size=int(cfg["training"]["batch_size_per_environment"]),
        )
        zero = torch.zeros(5, dtype=torch.double)
        slices = _bank_slices(banks)

        for method in METHODS:
            torch.manual_seed(seed)
            model = build_model_from_config(cfg)
            result = train_survey_method(
                model=model,
                source_envs=data.source_envs,
                batch_schedule=data.batch_schedule,
                method=method,
                config=cfg,
                seed=seed,
                initial_parameter_hash="",
            )
            if not result.finite:
                raise RuntimeError(result.invalid_reason)
            if STEP not in result.checkpoint_bundles:
                raise RuntimeError(
                    f"missing logical checkpoint bundle C_{STEP}; available={sorted(result.checkpoint_bundles)}"
                )

            bundle = result.checkpoint_bundles[STEP]
            fork, bundle_hashes = _fork_from_bundle(cfg, bundle)
            task = task_geometry(
                fork.model,
                world,
                pool_size=int(cfg["banks"]["geometry_pool_size_per_pool"]),
            )
            observation = observation_geometry(
                fork.model,
                world,
                pool_size=int(cfg["banks"]["geometry_pool_size_per_pool"]),
            )

            paths: dict[str, dict[int, tuple[str, str, str, dict[str, torch.Tensor]]]] = {
                "control": _run_path(fork, world, banks, zero, method, cfg, schedule, (0, H)),
            }
            control = paths["control"][H][3]
            responses: list[torch.Tensor] = []
            for basis, name in zip(BASIS, NAMES):
                direction = torch.zeros(5, dtype=torch.double)
                direction[basis] = 1.0
                paths["plus_" + name] = _run_path(
                    fork, world, banks, EPS * direction, method, cfg, schedule, (0, H)
                )
                paths["minus_" + name] = _run_path(
                    fork, world, banks, -EPS * direction, method, cfg, schedule, (0, H)
                )
                plus = paths["plus_" + name][H][3]
                minus = paths["minus_" + name][H][3]
                response = {
                    key: ((plus[key] - minus[key]) / (2.0 * EPS)).detach()
                    for key in BANK_NAMES
                }
                flat = _flat_bank_response(response)
                responses.append(flat)
                arrays[f"{method}__{seed}__{name}"] = flat.numpy()

            # The finite-shift outcome is a separate fork from the same C_300;
            # it is not an algebraic transform of the local finite difference.
            for basis, name in zip(BASIS, NAMES):
                direction = torch.zeros(5, dtype=torch.double)
                direction[basis] = 1.0
                paths["shift_" + name] = _run_path(
                    fork, world, banks, ALPHA * direction, method, cfg, schedule, (0, H)
                )
            gate = _assert_same_initial_state(paths, bundle_hashes)
            state_gate_rows.append(
                {
                    "method": method,
                    "seed": seed,
                    "logical_checkpoint": STEP,
                    "legacy_checkpoint_key": LEGACY_CHECKPOINT_STEP,
                    **gate,
                }
            )

            response_matrix = torch.stack(responses, dim=0)
            gram = response_matrix @ response_matrix.T
            task_a = task.A[:, BASIS]
            obs_o = observation.O[:, BASIS]

            for j, (basis, name) in enumerate(zip(BASIS, NAMES)):
                shift = paths["shift_" + name][H][3]
                outcome = _flat_bank_response(
                    {key: (shift[key] - control[key]).detach() for key in BANK_NAMES}
                )
                row: dict[str, Any] = {
                    "method": method,
                    "seed": seed,
                    "semantic_direction": name,
                    "horizon": H,
                    "finite_alpha": ALPHA,
                    "logical_checkpoint": STEP,
                    "legacy_checkpoint_key": LEGACY_CHECKPOINT_STEP,
                    "state_gate_pass": True,
                    "outcome_norm": float(outcome.norm()),
                    "outcome_source_norm": float(outcome[slices[0]].norm()),
                    "outcome_counterfactual_norm": float(torch.cat((outcome[slices[1]], outcome[slices[2]])).norm()),
                    "outcome_clean_norm": float(outcome[slices[3]].norm()),
                    "A_norm": float(task_a[:, j].norm()),
                    "O_norm": float(obs_o[:, j].norm()),
                    "R_norm": float(response_matrix[j].norm()),
                    "A_rank": task.A.shape[1],
                    "O_rank": observation.rank,
                }
                for q in range(3):
                    row[f"G_{j}_{q}"] = float(gram[j, q])
                rows.append(row)

            for j, name in enumerate(NAMES):
                feature_rows.append(
                    {
                        "method": method,
                        "seed": seed,
                        "semantic_direction": name,
                        "logical_checkpoint": STEP,
                        "state_gate_pass": True,
                        "A_norm": float(task_a[:, j].norm()),
                        "O_norm": float(obs_o[:, j].norm()),
                        "R_norm": float(response_matrix[j].norm()),
                        "G_diag": float(gram[j, j]),
                        "G_cross_01": float(gram[0, 1]),
                        "G_cross_03": float(gram[0, 2]),
                        "G_cross_13": float(gram[1, 2]),
                        "R_source_norm": float(response_matrix[j, slices[0]].norm()),
                        "R_counterfactual_norm": float(torch.cat((response_matrix[j, slices[1]], response_matrix[j, slices[2]])).norm()),
                        "R_clean_norm": float(response_matrix[j, slices[3]].norm()),
                    }
                )

    vector_path = OUT / "aopi_semantic_response_vectors.npz"
    np.savez_compressed(vector_path, **arrays)
    outcome_path = OUT / "aopi_semantic_finite_outcomes.csv"
    feature_path = OUT / "aopi_semantic_predictor_features.csv"
    state_path = OUT / "aopi_semantic_state_gate.csv"
    pd.DataFrame(rows).to_csv(outcome_path, index=False)
    pd.DataFrame(feature_rows).to_csv(feature_path, index=False)
    pd.DataFrame(state_gate_rows).to_csv(state_path, index=False)

    outcome_frame = pd.DataFrame(rows)
    feature_frame = pd.DataFrame(feature_rows)
    audit: list[dict[str, Any]] = []
    for direction in NAMES:
        joined = outcome_frame[outcome_frame.semantic_direction == direction].merge(
            feature_frame,
            on=["method", "seed", "semantic_direction", "logical_checkpoint", "state_gate_pass"],
            suffixes=("", "_feat"),
        )
        y = joined.outcome_norm.values
        predictors = (
            ("A_only", ["A_norm"]),
            ("AO", ["A_norm", "O_norm"]),
            (
                "AO_R",
                [
                    "A_norm",
                    "O_norm",
                    "R_norm",
                    "G_diag",
                    "G_cross_01",
                    "G_cross_03",
                    "G_cross_13",
                    "R_source_norm",
                    "R_counterfactual_norm",
                    "R_clean_norm",
                ],
            ),
        )
        for label, columns in predictors:
            prediction = np.zeros(len(joined))
            for method in joined.method.unique():
                train_mask = joined.method != method
                test_mask = ~train_mask
                if int(train_mask.sum()) == 0:
                    continue
                estimator = Ridge(alpha=1e-8).fit(joined.loc[train_mask, columns], y[train_mask])
                prediction[test_mask] = estimator.predict(joined.loc[test_mask, columns])
            audit.append(
                {
                    "semantic_direction": direction,
                    "model": label,
                    "n": len(joined),
                    "methods": joined.method.nunique(),
                    "lo_method_rmse": float(np.sqrt(mean_squared_error(y, prediction))),
                    "outcome_sd": float(y.std()),
                }
            )

    audit_frame = pd.DataFrame(audit)
    audit_path = OUT / "aopi_semantic_incremental_predictive_audit.csv"
    audit_frame.to_csv(audit_path, index=False)
    report = [
        "# A/O-conditioned semantic predictive audit",
        "",
        f"Five methods and five seeds; logical checkpoint C_{STEP} (after updates 0..{STEP - 1}), legacy trainer checkpoint key {LEGACY_CHECKPOINT_STEP}; finite shift alpha={ALPHA}, continuation horizon {H}.",
        "All A/O geometry, local response forks, and finite-shift outcome forks were materialized from the same complete model/optimizer/algorithm-state bundle. The initial model, optimizer, algorithm-state, and bank-logit hashes passed the hard state gate for every method/seed.",
        "",
        "Leave-one-method-out results:",
        "",
        audit_frame.to_string(index=False),
        "",
        "The full response model is evidence beyond A/O only if its leave-one-method-out RMSE improves over AO without using target outcomes. This audit remains a predictive response test; it does not by itself establish a causal semantic mechanism.",
    ]
    (OUT / "aopi_semantic_predictive_report.md").write_text("\n".join(report))
    provenance = {
        "methods": list(METHODS),
        "seeds": list(SEEDS),
        "logical_checkpoint": STEP,
        "legacy_checkpoint_key": LEGACY_CHECKPOINT_STEP,
        "checkpoint_semantics": "C_t is after t completed optimizer updates; legacy state-dict key t-1 is used for t>0",
        "horizon": H,
        "finite_alpha": ALPHA,
        "predictors": ["A_norm", "O_norm", "full_response_norms", "response_Gram", "bank_allocation"],
        "outcome": "finite-shift continuation displacement relative to zero-shift continuation",
        "same_complete_state_for_AO_response_outcome": True,
        "state_gate_file": str(state_path.relative_to(ROOT)),
        "target_used_for_training": False,
        "target_used_for_feature_selection": False,
        "birm_boundary": "separate static head-only artifact",
        "status": "STATE_MATCHED_PREDICTIVE_AUDIT",
        "git_head": _git_head(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "input_config_sha256": _sha256(CFG),
        "effective_config_sha256": effective_config_sha256,
        "effective_training_checkpoint_steps": [LEGACY_CHECKPOINT_STEP],
        "script_sha256": _sha256(Path(__file__)),
        "output_sha256": {
            "response_vectors": _sha256(vector_path),
            "finite_outcomes": _sha256(outcome_path),
            "predictor_features": _sha256(feature_path),
            "state_gate": _sha256(state_path),
            "audit": _sha256(audit_path),
        },
    }
    (OUT / "aopi_semantic_predictive_provenance.json").write_text(json.dumps(provenance, indent=2))
    print(audit_frame.to_string(index=False))


if __name__ == "__main__":
    run()
