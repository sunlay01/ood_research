from __future__ import annotations

import inspect
import json

import numpy as np
import pytest

from ood_repr_reg.round3r_3b_benchmark import environment_state, source_optimum
from ood_repr_reg.task3r_algorithmization.core import (
    encode_state,
    exact_directional_response,
    exact_ift_blocks,
    finite_difference_response,
    fit_source_only_regularizer,
    leave_one_out_folds,
    make_configurations,
    matched_random_response_targets,
    shift_configuration,
    source_only_response_target,
    source_states,
)
from ood_repr_reg.task3r_algorithmization.experiment import (
    evaluate_held_out,
    gate_summary,
    prepare_source_fits,
    preregistration,
)


def test_preregistration_freezes_source_only_protocol() -> None:
    design = preregistration()
    assert design["written_before_target_outcomes"]
    assert design["target_information_allowed_in_training"] is False
    assert design["beta_grid"] == [0.01, 0.1, 1.0]
    assert len(design["independent_primary_configs"]) == 2


def test_loo_estimator_is_exact_source_optimum_displacement() -> None:
    states = source_states(make_configurations()[0])
    for fold in leave_one_out_folds(states):
        expected = source_only_response_target(
            fold.reference, fold.pseudo_target, fold.reference_weights
        )
        assert np.allclose(fold.required_response, expected, atol=1e-12)
        assert fold.delta.shape == encode_state(fold.reference).shape


def test_exact_ift_response_matches_retraining_difference() -> None:
    configuration = make_configurations()[0]
    states = source_states(configuration)
    fit = fit_source_only_regularizer(states, beta=0.1)
    weights, reference = source_optimum(configuration.sources)
    delta = leave_one_out_folds(states)[0].delta
    audit = finite_difference_response(weights, reference, delta, fit.C,
                                       fit.regularizer_strength)
    assert audit["relative_error"] < 1e-7
    blocks = exact_ift_blocks(weights, reference, fit.C, fit.regularizer_strength)
    direct = exact_directional_response(weights, reference, delta, fit.C,
                                        fit.regularizer_strength)["response_z"]
    assert np.allclose(blocks["Pi"] @ delta, direct, atol=1e-10)


def test_source_only_api_has_no_target_or_free_pi_input() -> None:
    parameters = inspect.signature(fit_source_only_regularizer).parameters
    assert "target" not in parameters
    assert "A" not in parameters
    assert "Pi" not in parameters
    fit = fit_source_only_regularizer(source_states(make_configurations()[0]), 0.1)
    assert fit.target_information_used is False
    assert fit.pi_is_free_parameter is False


def test_random_direction_is_matched_norm_and_intervenes_on_fitted_C() -> None:
    states = source_states(make_configurations()[0])
    exact = fit_source_only_regularizer(states, 0.1)
    random = fit_source_only_regularizer(states, 0.1, randomize_response=True,
                                         random_seed=1701)
    folds = leave_one_out_folds(states)
    random_targets = matched_random_response_targets(folds, 1701)
    assert all(np.isclose(np.linalg.norm(source.required_response), np.linalg.norm(random_target))
               for source, random_target in zip(folds, random_targets))
    assert random.response_target_kind == "matched_norm_random_source_response"
    assert not np.allclose(exact.C, random.C)
    assert exact.C.shape == random.C.shape
    assert all(np.linalg.norm(fold.required_response) >= 0 for fold in folds)


def test_legal_held_out_shifts_do_not_modify_source_configuration() -> None:
    configuration = make_configurations()[0]
    original = tuple(environment.shortcut_rhos for environment in configuration.sources)
    shifted, target = shift_configuration(configuration, 1, 0.1)
    assert tuple(environment.shortcut_rhos for environment in configuration.sources) == original
    assert all(np.isclose(after.shortcut_rhos[1] - before.shortcut_rhos[1], 0.1)
               for before, after in zip(configuration.sources, shifted))
    assert np.isclose(target.shortcut_rhos[1] - configuration.base.shortcut_rhos[1], 0.1)
    with pytest.raises(ValueError):
        shift_configuration(configuration, 0, 0.5)


def test_oracle_estimated_and_random_diagnostics_are_separate() -> None:
    fits, _ = prepare_source_fits()
    rows = evaluate_held_out(fits[:1])
    required = {
        "oracle_A_rec_norm", "estimated_A_rec_norm", "matched_random_A_rec_norm",
        "oracle_E_norm", "estimated_E_norm", "random_E_norm",
    }
    assert required.issubset(rows[0])
    assert all(not row["target_used_for_training"] for row in rows)
    assert all(not row["target_used_for_selection"] for row in rows)
    assert any(abs(row["oracle_E_norm"] - row["estimated_E_norm"]) > 1e-10 for row in rows)


def test_erm_exactly_cancels_finite_source_optimum_response() -> None:
    fits, _ = prepare_source_fits()
    rows = evaluate_held_out(fits[:1])
    erm = [row for row in rows if row["method"] == "ERM"]
    assert max(float(row["estimated_E_norm"]) for row in erm) < 1e-12


def test_verdict_obeys_preregistered_gate() -> None:
    design = preregistration()
    fits, ablations = prepare_source_fits()
    rows = evaluate_held_out(fits)
    summary = gate_summary(rows, ablations, design)
    assert summary["passing_primary_configurations"] < 2
    assert summary["gaussian_gate_pass"] is False
    assert summary["cmnist_run"] is False
    assert summary["verdict"] == "TASK3R-ALGORITHM-PARTIAL"
    assert not summary["target_leakage_detected"]
    assert not summary["pi_free_shortcut_used"]
