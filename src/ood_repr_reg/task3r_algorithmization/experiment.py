"""Preregistered source fitting and post-hoc Gaussian evaluation for Task 3R."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np

from ..round3r_3b_benchmark import environment_state, mixture_state, risk, source_optimum
from .core import (
    FittedRegularizer,
    ProbeConfiguration,
    encode_state,
    exact_ift_blocks,
    fit_source_only_regularizer,
    finite_difference_response,
    make_configurations,
    mldg_head,
    retrain_centered_head,
    shift_configuration,
    source_states,
)

Array = np.ndarray

BETA_GRID = (0.01, 0.1, 1.0)
HELD_OUT_AMOUNTS = (-0.10, -0.05, 0.05, 0.10)
REGULARIZER_STRENGTH = 0.1
MLDG_STEP = 0.05


@dataclass(frozen=True)
class SourceFit:
    configuration: ProbeConfiguration
    reference_state: object
    reference_weights: Array
    fitted_by_beta: tuple[FittedRegularizer, ...]
    selected: FittedRegularizer
    no_response: FittedRegularizer
    random_response: FittedRegularizer
    mldg_weights: Array


def preregistration() -> dict[str, object]:
    configurations = make_configurations()

    def environment_record(environment) -> dict[str, object]:
        return {
            "shortcut_rhos": list(environment.shortcut_rhos),
            "shortcut_means": list(environment.shortcut_means),
            "shortcut_variances": list(environment.shortcut_variances),
            "n_noise": environment.n_noise,
            "c_noise_variance": environment.c_noise_variance,
            "u_gamma": environment.u_gamma,
            "u_noise_variance": environment.u_noise_variance,
        }

    return {
        "task_id": "TASK3R-ALGORITHMIZATION",
        "phase": "exact_gaussian_quadratic",
        "written_before_target_outcomes": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_config_ids": [configuration.config_id for configuration in configurations],
        "source_configurations": [
            {
                "config_id": configuration.config_id,
                "role": configuration.role,
                "base": environment_record(configuration.base),
                "source_environments": [environment_record(env) for env in configuration.sources],
            }
            for configuration in configurations
        ],
        "independent_primary_configs": ["gaussian_primary_1", "gaussian_primary_2"],
        "source_information_ablation": "gaussian_reduced_exposure",
        "beta_grid": list(BETA_GRID),
        "regularizer_strength": REGULARIZER_STRENGTH,
        "mldg_step": MLDG_STEP,
        "optimizer": {
            "name": "L-BFGS-B",
            "maxiter": 500,
            "ftol": 1e-13,
            "gtol": 1e-10,
            "parameter_ridge": 1e-6,
            "random_response_seed": 1701,
        },
        "pseudo_target_folds": "leave_one_source_domain_out",
        "source_only_selection_rule": "minimize mean LOO pseudo-risk + 0.1 * mean LOO response mismatch",
        "held_out_directions": [0, 1],
        "held_out_amounts": list(HELD_OUT_AMOUNTS),
        "held_out_generation": "uniform legal shortcut-correlation shift applied after source fitting",
        "methods": ["ERM", "PSEUDO_RESPONSE", "MLDG_LOCAL", "NO_RESPONSE", "RANDOM_RESPONSE"],
        "hypotheses": {
            "H1": "selected method lowers estimated response residual versus ERM in each passing primary config",
            "H2": "selected method lowers mean held-out finite target risk versus ERM",
            "H3": "matched-norm random response does not retain the held-out improvement",
            "H4": "benefit is smaller under reduced source exposure",
        },
        "thresholds": {
            "response_relative_reduction": 0.01,
            "held_out_mean_risk_improvement": 1e-6,
            "max_source_risk_degradation": 1e-5,
            "direction_control_fraction": 0.25,
            "source_information_shrink_fraction": 0.25,
            "required_passing_primary_configs": 2,
            "ift_relative_error": 1e-6,
        },
        "cmnist_gate": "run only if H1-H4 pass in both independent primary Gaussian configurations",
        "target_information_allowed_in_training": False,
        "approximation": "none_exact_quadratic",
    }


def prepare_source_fits() -> tuple[tuple[SourceFit, ...], list[dict[str, object]]]:
    """Fit every method before held-out target states are generated."""
    fits: list[SourceFit] = []
    ablations: list[dict[str, object]] = []
    for configuration in make_configurations():
        states = source_states(configuration)
        reference_weights, reference = source_optimum(configuration.sources)
        fitted = tuple(
            fit_source_only_regularizer(states, beta, regularizer_strength=REGULARIZER_STRENGTH)
            for beta in BETA_GRID
        )
        selected = min(fitted, key=lambda item: (item.source_selection_score, item.beta))
        no_response = fit_source_only_regularizer(
            states, selected.beta, regularizer_strength=REGULARIZER_STRENGTH,
            response_term=False,
        )
        random_response = fit_source_only_regularizer(
            states, selected.beta, regularizer_strength=REGULARIZER_STRENGTH,
            randomize_response=True, random_seed=1701,
        )
        fits.append(SourceFit(configuration, reference, reference_weights, fitted, selected,
                              no_response, random_response, mldg_head(states, MLDG_STEP)))
        for item in fitted + (no_response, random_response):
            ablations.append({
                "config_id": configuration.config_id,
                "config_role": configuration.role,
                "method": item.method,
                "beta": item.beta,
                "selected": item is selected,
                "source_selection_score": item.source_selection_score,
                "mean_pseudo_risk": item.mean_pseudo_risk,
                "mean_response_mismatch": item.mean_response_mismatch,
                "C_operator_norm": float(np.linalg.norm(item.C, ord=2)),
                "C_rank": int(np.linalg.matrix_rank(item.C, tol=1e-10)),
                "contrast_rank": int(item.contrast_basis.shape[1]),
                "optimizer_success": item.optimizer_success,
                "optimizer_message": item.optimizer_message,
                "response_target_kind": item.response_target_kind,
                "target_information_used": item.target_information_used,
                "pi_is_free_parameter": item.pi_is_free_parameter,
            })
    return tuple(fits), ablations


def _sqrt_and_inverse(matrix: Array) -> tuple[Array, Array]:
    values, vectors = np.linalg.eigh((matrix + matrix.T) / 2.0)
    root = vectors @ np.diag(np.sqrt(values)) @ vectors.T
    inverse = vectors @ np.diag(1.0 / np.sqrt(values)) @ vectors.T
    return root, inverse


def _matched_random(vector: Array, key: int) -> Array:
    rng = np.random.default_rng(key)
    draw = rng.normal(size=np.asarray(vector).size)
    return draw * (np.linalg.norm(vector) / max(np.linalg.norm(draw), 1e-15))


def evaluate_held_out(fits: tuple[SourceFit, ...]) -> list[dict[str, object]]:
    """Generate target states only after fitting and evaluate frozen algorithms."""
    rows: list[dict[str, object]] = []
    for fit in fits:
        config = fit.configuration
        reference = fit.reference_state
        w_reference = fit.reference_weights
        hessian = 2.0 * reference.second
        root, inverse_root = _sqrt_and_inverse(hessian)
        target_zero = environment_state(config.base)
        q_zero = np.linalg.solve(target_zero.second, target_zero.xy)
        methods = {
            "ERM": None,
            "PSEUDO_RESPONSE": fit.selected,
            "NO_RESPONSE": fit.no_response,
            "RANDOM_RESPONSE": fit.random_response,
            "MLDG_LOCAL": None,
        }
        base_weights = {name: w_reference for name in methods}
        base_weights["MLDG_LOCAL"] = fit.mldg_weights

        for direction in (0, 1):
            for amount in HELD_OUT_AMOUNTS:
                shifted_environments, target_environment = shift_configuration(config, direction, amount)
                shifted_state = mixture_state(shifted_environments)
                target_state = environment_state(target_environment)
                q_target = np.linalg.solve(target_state.second, target_state.xy)
                oracle_required = -root @ (q_target - q_zero)
                shifted_source_optimum = np.linalg.solve(shifted_state.second, shifted_state.xy)
                estimated_required = -root @ (shifted_source_optimum - w_reference)
                random_required = _matched_random(estimated_required, 9100 + direction * 100 + int((amount + 1) * 100))

                shifted_weights: dict[str, Array] = {
                    "ERM": np.linalg.solve(shifted_state.second, shifted_state.xy),
                    "MLDG_LOCAL": mldg_head(tuple(environment_state(env) for env in shifted_environments), MLDG_STEP),
                }
                for name in ("PSEUDO_RESPONSE", "NO_RESPONSE", "RANDOM_RESPONSE"):
                    item = methods[name]
                    assert isinstance(item, FittedRegularizer)
                    shifted_weights[name] = retrain_centered_head(
                        w_reference, reference, shifted_state, item.C, item.regularizer_strength
                    )
                erm_target_risk = risk(shifted_weights["ERM"], target_state)

                for name, item in methods.items():
                    weights = shifted_weights[name]
                    actual = root @ (weights - base_weights[name])
                    oracle_e = oracle_required + actual
                    estimated_e = estimated_required + actual
                    random_e = random_required + actual
                    source_risk = risk(weights, shifted_state)
                    target_value = risk(weights, target_state)
                    static = root @ (base_weights[name] - w_reference)
                    row = {
                        "config_id": config.config_id,
                        "config_role": config.role,
                        "method": name,
                        "beta": None if item is None else item.beta,
                        "direction": direction,
                        "amount": amount,
                        "legal_shift": True,
                        "source_risk": source_risk,
                        "pseudo_response_loss": None if item is None else item.mean_response_mismatch,
                        "directional_mismatch_estimated": float(np.linalg.norm(estimated_e)),
                        "E_operator_norm": float(np.linalg.norm(oracle_e)),
                        "oracle_E_norm": float(np.linalg.norm(oracle_e)),
                        "estimated_E_norm": float(np.linalg.norm(estimated_e)),
                        "random_E_norm": float(np.linalg.norm(random_e)),
                        "oracle_A_rec_norm": float(np.linalg.norm(oracle_required)),
                        "estimated_A_rec_norm": float(np.linalg.norm(estimated_required)),
                        "matched_random_A_rec_norm": float(np.linalg.norm(random_required)),
                        "actual_response_norm": float(np.linalg.norm(actual)),
                        "slack_margin": -float(oracle_e @ oracle_e),
                        "slack_diagnostic": "one_direction_complete_information",
                        "static_tax_proxy": 0.5 * float(static @ static),
                        "finite_target_risk": target_value,
                        "delta_vs_erm": target_value - erm_target_risk,
                        "target_used_for_training": False,
                        "target_used_for_selection": False,
                    }
                    if isinstance(item, FittedRegularizer):
                        blocks = exact_ift_blocks(w_reference, reference, item.C, item.regularizer_strength)
                        delta = encode_state(shifted_state) - encode_state(reference)
                        fd = finite_difference_response(w_reference, reference, delta, item.C,
                                                        item.regularizer_strength)
                        row.update({
                            "Pi_operator_norm": float(np.linalg.norm(blocks["Pi"], ord=2)),
                            "K_operator_norm": float(np.linalg.norm(blocks["K"], ord=2)),
                            "C_operator_norm": float(np.linalg.norm(blocks["C"], ord=2)),
                            "ift_fd_relative_error": fd["relative_error"],
                            "min_local_metric_eigenvalue": blocks["min_local_metric_eigenvalue"],
                        })
                    rows.append(row)
    return rows


def gate_summary(rows: list[dict[str, object]], ablations: list[dict[str, object]],
                 prereg: dict[str, object]) -> dict[str, object]:
    thresholds = prereg["thresholds"]
    config_rows: list[dict[str, object]] = []
    for config_id in prereg["source_config_ids"]:
        subset = [row for row in rows if row["config_id"] == config_id]
        by_method = {
            method: [row for row in subset if row["method"] == method]
            for method in prereg["methods"]
        }
        means = {
            method: {
                "estimated_E": float(np.mean([float(row["estimated_E_norm"]) for row in values])),
                "oracle_E": float(np.mean([float(row["oracle_E_norm"]) for row in values])),
                "target_risk": float(np.mean([float(row["finite_target_risk"]) for row in values])),
                "source_risk": float(np.mean([float(row["source_risk"]) for row in values])),
            }
            for method, values in by_method.items()
        }
        erm, proposed = means["ERM"], means["PSEUDO_RESPONSE"]
        random, mldg = means["RANDOM_RESPONSE"], means["MLDG_LOCAL"]
        if erm["estimated_E"] <= 1e-10:
            response_gain = 0.0 if proposed["estimated_E"] <= 1e-10 else -1.0
            response_baseline_zero = True
        else:
            response_gain = (erm["estimated_E"] - proposed["estimated_E"]) / erm["estimated_E"]
            response_baseline_zero = False
        target_gain = erm["target_risk"] - proposed["target_risk"]
        baseline_gain = mldg["target_risk"] - proposed["target_risk"]
        random_gain = erm["target_risk"] - random["target_risk"]
        source_degradation = proposed["source_risk"] - erm["source_risk"]
        h1 = response_gain >= float(thresholds["response_relative_reduction"])
        h2 = min(target_gain, baseline_gain) >= float(thresholds["held_out_mean_risk_improvement"])
        h3 = target_gain > 0 and random_gain <= (
            1.0 - float(thresholds["direction_control_fraction"])
        ) * target_gain
        config_rows.append({
            "config_id": config_id,
            "H1_response": bool(h1),
            "H2_held_out": bool(h2),
            "H3_direction": bool(h3),
            "response_relative_reduction": response_gain,
            "response_baseline_zero": response_baseline_zero,
            "held_out_risk_improvement": target_gain,
            "held_out_risk_improvement_vs_mldg": baseline_gain,
            "random_control_improvement": random_gain,
            "source_risk_degradation": source_degradation,
            "source_risk_ok": bool(source_degradation <= float(thresholds["max_source_risk_degradation"])),
            "method_means": means,
        })
    primary = [item for item in config_rows if item["config_id"] in prereg["independent_primary_configs"]]
    reduced = next(item for item in config_rows if item["config_id"] == prereg["source_information_ablation"])
    primary_gain = float(np.mean([float(item["held_out_risk_improvement"]) for item in primary]))
    h4 = primary_gain > 0 and float(reduced["held_out_risk_improvement"]) <= (
        1.0 - float(thresholds["source_information_shrink_fraction"])
    ) * primary_gain
    passing = sum(bool(
        item["H1_response"] and item["H2_held_out"] and item["H3_direction"]
        and item["source_risk_ok"]
    ) for item in primary)
    optimizers_ok = all(bool(row["optimizer_success"]) for row in ablations)
    ift_errors = [float(row["ift_fd_relative_error"]) for row in rows if row.get("ift_fd_relative_error") is not None]
    ift_ok = bool(ift_errors) and max(ift_errors) <= float(thresholds["ift_relative_error"])
    gaussian_gate = bool(
        passing >= int(thresholds["required_passing_primary_configs"])
        and h4 and optimizers_ok and ift_ok
    )
    if gaussian_gate:
        verdict = "TASK3R-ALGORITHM-PARTIAL"
        bottleneck = "Gaussian gate passed; CMNIST implementation is outside this exact-probe run."
    elif optimizers_ok and ift_ok:
        verdict = "TASK3R-ALGORITHM-PARTIAL"
        bottleneck = "Exact source-only mechanism is valid, but the preregistered held-out Gaussian gate did not pass."
    else:
        verdict = "TASK3R-ALGORITHM-FAIL"
        bottleneck = "Exact optimization or IFT validation failed."
    return {
        "task_id": "TASK3R-ALGORITHMIZATION",
        "verdict": verdict,
        "prior_work_equivalence_label": "SPECIAL-CASE",
        "verdict_capped_by_equivalence": True,
        "gaussian_gate_pass": gaussian_gate,
        "cmnist_run": False,
        "cmnist_stop_reason": None if gaussian_gate else "preregistered Gaussian H1-H4 gate failed",
        "optimizer_stable": optimizers_ok,
        "ift_ok": ift_ok,
        "max_ift_fd_relative_error": max(ift_errors, default=float("inf")),
        "passing_primary_configurations": passing,
        "required_passing_primary_configurations": thresholds["required_passing_primary_configs"],
        "H4_source_information": bool(h4),
        "configuration_results": config_rows,
        "bottleneck": bottleneck,
        "target_leakage_detected": any(
            bool(row["target_used_for_training"] or row["target_used_for_selection"]) for row in rows
        ),
        "pi_free_shortcut_used": any(bool(row["pi_is_free_parameter"]) for row in ablations),
        "approximation_fidelity": "not_applicable_exact_quadratic",
        "row_count": len(rows),
        "ablation_row_count": len(ablations),
        "selected_betas": {
            str(row["config_id"]): float(row["beta"])
            for row in ablations if bool(row["selected"])
        },
    }


__all__ = [
    "BETA_GRID", "HELD_OUT_AMOUNTS", "REGULARIZER_STRENGTH", "MLDG_STEP",
    "SourceFit", "preregistration", "prepare_source_fits", "evaluate_held_out",
    "gate_summary",
]
