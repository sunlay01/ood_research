import numpy as np

from ood_repr_reg.round3_response_matrix import build_response_matrices
from ood_repr_reg.round3_retry_discovery import bootstrap_column_stability, discover_shift_columns
from ood_repr_reg.round3_retry_validation import mechanism_enrichment
from ood_repr_reg.round3_shift_ensemble import base_environment, nonlinear_environment, shift_ensemble, source_environments
from ood_repr_reg.round3_trained_models import risk, train_models


def test_retry_trains_declared_objectives_and_records_metadata():
    base = base_environment()
    models = train_models(source_environments(base), seeds=(0, 1), lambdas=(0.1, 1.0))
    assert {model.method for model in models} == {"ERM", "L2", "IRMv1", "V-REx", "CORAL"}
    assert all(model.optimization_status for model in models)
    assert len({tuple(np.round(model.weights, 8)) for model in models}) > 1
    assert all(model.source_risk >= 0.0 for model in models)


def test_response_discovery_uses_shift_columns_and_is_label_free():
    base = base_environment()
    models = train_models(source_environments(base), seeds=(0, 1, 2), lambdas=(0.1,))
    probes = shift_ensemble(base, count=8, seed=3)
    matrices = build_response_matrices(models, base, probes)
    result = discover_shift_columns(matrices, "raw")
    assert matrices.raw.shape == (len(models), len(probes))
    assert result["n_shifts"] == len(probes)
    assert result["n_models"] == len(models)
    assert result["input_uses_mechanism_labels"] is False
    assert 0.0 <= bootstrap_column_stability(matrices, repeats=3) <= 1.0


def test_nonzero_mean_makes_mean_shift_response_nontrivial():
    base = base_environment()
    models = train_models(source_environments(base), seeds=(0,), lambdas=(0.1,))
    probes = tuple(p for p in shift_ensemble(base, count=8, seed=4) if p.kind == "core_mean")
    matrices = build_response_matrices(models, base, probes)
    assert np.any(np.abs(matrices.raw) > 1e-10)


def test_second_structural_family_is_evaluable():
    base = base_environment()
    nonlinear = nonlinear_environment(base)
    models = train_models(source_environments(base), seeds=(0,), lambdas=(0.1,))[:3]
    probes = shift_ensemble(nonlinear, count=2, seed=5)
    matrices = build_response_matrices(models, nonlinear, probes)
    assert np.isfinite(matrices.raw).all()
    assert risk(models[0].weights, nonlinear) >= 0.0


def test_posthoc_enrichment_has_no_effect_on_blind_matrix_shape():
    base = base_environment()
    models = train_models(source_environments(base), seeds=(0,), lambdas=(0.1,))
    probes = shift_ensemble(base, count=4, seed=6)
    matrices = build_response_matrices(models, base, probes)
    labels = np.zeros(len(probes), dtype=int)
    result = mechanism_enrichment(matrices, labels)
    assert result["labels_used_after_discovery"] is True
    assert matrices.raw.shape[1] == len(labels)

