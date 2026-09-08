import numpy as np

from ood_repr_reg.round3_intervention_validation import intervention_validation, regularizer_probe
from ood_repr_reg.round3_mechanisms import default_mechanism
from ood_repr_reg.round3_response_dataset import build_response_dataset


def test_intervention_validation_returns_all_declared_mechanisms():
    dataset = build_response_dataset(seeds=(0, 1))
    result = intervention_validation(dataset, default_mechanism())
    assert {"core", "nuisance", "relation", "core_variance", "nuisance_variance"} == set(result)


def test_regularizer_probe_is_diagnostic_only():
    dataset = build_response_dataset(seeds=(0,))
    rows = regularizer_probe(dataset.records)
    assert len(rows) == len(dataset.records)
    assert all(row["l2"] >= 0.0 for row in rows)
