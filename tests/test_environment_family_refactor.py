from __future__ import annotations

import numpy as np

from ood_repr_reg.environment_family import (
    build_source_induced_family,
    build_task_geometry,
    family_coordinate_audit,
    family_response_factorization_residual,
    legacy_gaussian_family,
    mechanism_family,
)
from ood_repr_reg.environment_family.invariance_audit import failure_classification, source_span_projector
from ood_repr_reg.environment_family.metrics import coordinate_transport, metric_operator_summary
from ood_repr_reg.round3r_3b_benchmark import environment_state
from ood_repr_reg.round3r_3c_affine import main_benchmark
from ood_repr_reg.round3r_3e_world_tangent import coupled_primary_geometry, u_exposed_observation
from ood_repr_reg.run_round3r_environment_family import family_comparison, legacy_regression
from ood_repr_reg.round3r_3d_state import task_state


def test_all_families_have_legal_reference_tangent_and_positive_metric():
    for family in (legacy_gaussian_family(), mechanism_family(), build_source_induced_family()):
        reference = family.reference_environment()
        spec = family.tangent_spec(reference)
        assert len(spec.directions) == spec.dimension == len(spec.scales)
        assert spec.metric.shape == (spec.dimension, spec.dimension)
        if spec.dimension:
            assert np.linalg.eigvalsh(spec.metric).min() > 0.0
        for direction in spec.directions:
            source = family.perturb(reference, direction, 1e-4, role="source")
            target = family.perturb(reference, direction, 1e-4, role="target")
            assert environment_state(source).second.shape == environment_state(reference).second.shape
            assert environment_state(target).second.shape == environment_state(reference).second.shape


def test_legacy_geometry_is_exactly_the_frozen_eight_dimension_pair():
    family_geometry = build_task_geometry(legacy_gaussian_family())
    frozen = coupled_primary_geometry()
    assert family_geometry.observation.shape == (275, 8)
    assert family_geometry.response.shape == (9, 8)
    assert np.linalg.matrix_rank(family_geometry.observation, tol=1e-9) == 7
    assert np.array_equal(family_geometry.observation, frozen.observation)
    assert np.array_equal(family_geometry.response, frozen.response)
    floor = metric_operator_summary(family_geometry.response, family_geometry.observation, tolerance=1e-9)["information_floor"]
    exposed = metric_operator_summary(family_geometry.response, u_exposed_observation(frozen), tolerance=1e-9)["information_floor"]
    assert np.isclose(floor, 0.05118145108608892, atol=1e-12)
    assert np.isclose(exposed, 0.0, atol=1e-12)
    assert family_response_factorization_residual(family_geometry) < 1e-10


def test_source_induced_basis_is_source_only_and_has_separate_realization_ranks():
    family = build_source_induced_family()
    audit = family.metadata()
    assert family.state_span_rank == 4
    assert family.realizable_rank == 4
    assert audit["target_risk_used"] is False
    assert audit["response_operator_used"] is False
    assert audit["regularizer_geometry_used"] is False
    assert audit["semantic_labels_used"] is False
    assert all(residual < 1e-7 for residual in family.realization_residuals)
    geometry = build_task_geometry(family)
    assert geometry.spec.source_defined is True
    assert "U_emergent" not in geometry.spec.directions


def test_source_reference_duplicate_and_independent_variation_audits():
    family = legacy_gaussian_family()
    environments = family.source_environments()
    states = np.stack([task_state(environment_state(environment)) for environment in environments])
    assert np.linalg.norm(source_span_projector(states, 0) - source_span_projector(states, len(states) - 1)) < 1e-8
    duplicate = build_source_induced_family(environments + (environments[0],))
    assert duplicate.state_span_rank == build_source_induced_family(environments).state_span_rank
    base = family.base
    mean_shift = base.updated(shortcut_means=(base.shortcut_means[0] + 0.2, base.shortcut_means[1]))
    independent = build_source_induced_family(environments + (mean_shift,))
    assert independent.state_span_rank > duplicate.state_span_rank
    with np.testing.assert_raises(ValueError):
        build_source_induced_family(environments, reference_index=len(environments))
    with np.testing.assert_raises(ValueError):
        build_source_induced_family(environments, base=family.base.updated(u_gamma=0.1))


def test_family_metric_transport_and_coordinate_audits():
    family = legacy_gaussian_family()
    geometry = build_task_geometry(family)
    rng = np.random.default_rng(123)
    recoding = rng.normal(size=(8, 8)) + 3.0 * np.eye(8)
    response, observation, metric = coordinate_transport(
        geometry.response, geometry.observation, geometry.spec.metric, recoding,
    )
    original = metric_operator_summary(geometry.response, geometry.observation, geometry.spec.metric)
    transformed = metric_operator_summary(response, observation, metric)
    assert np.isclose(original["alpha"], transformed["alpha"], atol=1e-10)
    assert family_coordinate_audit(family)["invariant"]
    assert family_coordinate_audit(build_source_induced_family())["invariant"]


def test_family_geometry_drives_3c_b_without_semantic_inputs():
    source_family = build_source_induced_family()
    result = main_benchmark(family=source_family, lambdas=(0.0, 1e-2))
    assert result["geometry"].spec.source_defined is True
    assert result["geometry"].observation.shape[1] == source_family.realizable_rank
    assert result["target_risk_used"] is False
    assert result["semantic_labels_used"] is False
    assert result["cluster_labels_used"] is False
    assert result["regularizer_geometry_used_for_selection"] is False


def test_cross_family_comparison_and_legacy_snapshot_pass():
    geometry_rows, method_rows = family_comparison()
    assert {row["family"] for row in geometry_rows} == {"legacy_gaussian_mechanism", "source_induced"}
    assert len(method_rows) == 18
    assert legacy_regression()["pass"]


def test_concrete_method_table_contains_full_affine_objects_for_both_families():
    _, method_rows = family_comparison()
    assert {row["family"] for row in method_rows} == {
        "legacy_gaussian_mechanism", "source_induced",
    }
    for row in method_rows:
        assert row["valid"] is True
        assert np.asarray(row["z0"]).shape == (9,)
        expected_width = 8 if row["family"] == "legacy_gaussian_mechanism" else 4
        assert np.asarray(row["pi_O"]).shape == (9, expected_width)
        assert np.asarray(row["E"]).shape == (9, expected_width)
        assert np.isfinite(float(row["affine_regret"]))


def test_failure_taxonomy_distinguishes_omission_information_and_algorithm_failure():
    assert failure_classification(in_family=False, source_visible=False, response_correct=False) == "family_misspecification_or_omission"
    assert failure_classification(in_family=True, source_visible=False, response_correct=False) == "information_failure"
    assert failure_classification(in_family=True, source_visible=True, response_correct=False) == "algorithm_failure"
    assert failure_classification(in_family=True, source_visible=True, response_correct=True) == "no_failure"
