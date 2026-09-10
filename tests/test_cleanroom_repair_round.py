from __future__ import annotations

import inspect

import numpy as np
import torch

from ood_repr_reg.cleanroom_rerun_1_3d import repair_round
from ood_repr_reg.cmnist_geometry_bridge import CMNISTFamily, RepresentationBank, head_from_state, ift_matrices, moment_state, source_observation, source_state_stack


def test_source_only_input_builder_does_not_pull_target_into_source_state() -> None:
    source = inspect.getsource(repair_round.build_source_only_bridge_context)
    source_side_block = source.split("state = source_state_stack", 1)[1].split("response = _target_response_operator", 1)[0]
    assert "target_bank" not in source_side_block
    assert "source_state_stack_fast(source_bank" in source
    assert "source_observation_analytic(source_bank" in source
    builder = inspect.getsource(repair_round.source_only_cmnist_input)
    assert "target_bank" not in builder


def test_counterfactual_feature_extraction_uses_supplied_envs_only() -> None:
    source = inspect.getsource(repair_round.extract_counterfactual_features)
    assert "target_env" not in source
    assert "data." not in source


def test_common_projection_is_source_only_and_not_fixed_8d() -> None:
    rng = np.random.default_rng(3)
    labels = rng.integers(0, 2, size=200).astype(float)
    source_features = {
        "ERM": repair_round.CounterfactualFeatures(rng.normal(size=(200, 12)), rng.normal(size=(200, 12)), labels),
        "IRMv1": repair_round.CounterfactualFeatures(rng.normal(size=(200, 12)), rng.normal(size=(200, 12)), labels),
    }
    projection = repair_round.fit_common_source_projection(source_features, (0.8, 0.9), tolerance=1e-10)
    assert projection.source_only_fit
    assert projection.common_to_erm_and_irmv1
    assert not projection.target_used_for_fit
    assert projection.dimension != 8


def test_projected_source_metric_positive() -> None:
    rng = np.random.default_rng(5)
    labels = rng.integers(0, 2, size=120).astype(float)
    source_features = {
        "ERM": repair_round.CounterfactualFeatures(rng.normal(size=(120, 10)), rng.normal(size=(120, 10)), labels),
        "IRMv1": repair_round.CounterfactualFeatures(rng.normal(size=(120, 10)), rng.normal(size=(120, 10)), labels),
    }
    projection = repair_round.fit_common_source_projection(source_features, (0.8, 0.9), tolerance=1e-10)
    for features in source_features.values():
        bank = RepresentationBank((features.red - projection.center) @ projection.basis, (features.green - projection.center) @ projection.basis, labels)
        matrix = 2.0 * np.mean([moment_state(bank, rho)[0] for rho in (0.8, 0.9)], axis=0)
        assert np.linalg.eigvalsh((matrix + matrix.T) / 2.0).min() > 0.0


def test_fast_moment_and_source_state_match_original() -> None:
    rng = np.random.default_rng(7)
    labels = rng.integers(0, 2, size=40).astype(float)
    bank = RepresentationBank(
        red=rng.normal(size=(40, 5)),
        green=rng.normal(size=(40, 5)),
        labels=labels,
    )
    family = CMNISTFamily("fast_equivalence", source_rhos=(0.8, 0.9), target_rho=0.1, directions=("rho_source_1", "rho_source_2"))
    for rho in (0.1, 0.8, 0.9):
        original = moment_state(bank, rho)
        fast = repair_round.moment_state_fast(bank, rho)
        for left, right in zip(original, fast, strict=True):
            np.testing.assert_allclose(left, right, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(
        repair_round.source_state_stack_fast(bank, family),
        source_state_stack(bank, family, np.zeros(family.dimension)),
        atol=1e-10,
        rtol=1e-10,
    )


def test_direction_summary_tracks_helps_and_hurts() -> None:
    rows = [
        {"setting": "s", "method": "IRMV1", "lambda": "0.1", "E00_operator_norm": "2.0", "ECK_operator_norm": "1.0", "representation_method": "ERM"},
        {"setting": "s", "method": "IRMV1", "lambda": "0.1", "E00_operator_norm": "2.0", "ECK_operator_norm": "1.5", "representation_method": "ERM"},
        {"setting": "s", "method": "L2", "lambda": "0.1", "E00_operator_norm": "1.0", "ECK_operator_norm": "1.2", "representation_method": "IRMv1"},
    ]
    summary = repair_round.summarize_counterfactual_rows(rows, include_representation=True)
    keyed = {(row["method"], row["representation_method"]): row for row in summary}
    assert keyed[("IRMV1", "ERM")]["direction"] == "helps"
    assert keyed[("L2", "IRMv1")]["direction"] == "hurts"


def test_help_diff_classifier_requires_family_counterpart() -> None:
    old = {"setting": "mechanism_defined_hidden", "method": "VREX", "lambda": 0.1, "mean_E_change": -1.0, "count": 5}
    status, direction, new_change, ratio = repair_round.classify_help_diff(old, [])
    assert status == "OLD_HELP_NOT_REPRODUCED_NO_REPAIRED_FAMILY_COUNTERPART"
    assert direction == "no_family_counterpart"
    assert new_change is None
    assert ratio is None


def test_help_diff_classifier_marks_large_attenuation() -> None:
    old = {"setting": "declared_source_target_coupled_correlation", "method": "VREX", "lambda": 0.1, "mean_E_change": -1.0, "count": 5}
    candidates = [{"representation_method": "ERM", "direction": "helps", "mean_E_change": -0.05}]
    status, direction, new_change, ratio = repair_round.classify_help_diff(old, candidates)
    assert status == "HELP_DIRECTION_PERSISTED_BUT_ATTENUATED_BELOW_10PCT"
    assert direction == "ERM:helps"
    assert new_change == -0.05
    assert ratio == 0.05


def test_repair_final_verdict_is_conservative() -> None:
    assert "RERUN-INCONCLUSIVE" in inspect.getsource(repair_round.write_reports)


def test_no_literal_target_pca_bridge_remains_in_repair_source() -> None:
    source = inspect.getsource(repair_round)
    assert "projection_dimension = 8" not in source
    assert "projection_dim=8" not in source
    assert "env = data.target_env" not in source
    assert "evaluate_input(" not in source


def test_source_tangent_ift_matches_full_jacobian_projection_on_small_bank() -> None:
    rng = np.random.default_rng(11)
    labels = rng.integers(0, 2, size=30).astype(float)
    bank = RepresentationBank(
        red=rng.normal(size=(30, 3)),
        green=rng.normal(size=(30, 3)),
        labels=labels,
    )
    family = CMNISTFamily("small_source_only", source_rhos=(0.8, 0.9), target_rho=0.1, directions=("rho_source_1", "rho_source_2"))
    state = source_state_stack(bank, family, np.zeros(family.dimension))
    observation = source_observation(bank, family, step=1e-4)
    np.testing.assert_allclose(repair_round.source_observation_analytic(bank, family), observation, atol=1e-8, rtol=1e-8)
    dimension = bank.dimension
    for method, lam in (("l2", 0.1), ("irmv1", 0.001), ("vrex", 0.001)):
        weights = head_from_state(state, dimension, method=method, lam=lam)
        full_dw, full_dy = ift_matrices(state, dimension, method, lam, weights)
        tangent_dw, tangent_dtheta = repair_round.source_tangent_ift_matrices(state, observation, dimension, method, lam, weights)
        np.testing.assert_allclose(tangent_dw, full_dw, atol=1e-7, rtol=1e-7)
        np.testing.assert_allclose(tangent_dtheta, full_dy @ observation, atol=1e-7, rtol=1e-7)


def test_torch_available_for_repair_feature_path() -> None:
    assert torch.__version__
