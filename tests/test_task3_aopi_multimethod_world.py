import torch
import pytest

from ood_repr_reg.task3_aopi_multimethod_mechanism_survey.analysis import derived_direction_checks, world_gate
from ood_repr_reg.task3_aopi_multimethod_mechanism_survey.smooth_world5 import (
    BASE, BASIS, DERIVED_DIRECTIONS, outcome_weight, all_direction_vectors,
)


def test_base_world_and_outcome_weights_are_valid_probabilities():
    assert torch.equal(BASE, torch.tensor([0.2, 0.1, 0.9, 0.25, 0.25], dtype=torch.double))
    for p in (0.0, 0.2, 0.9, 1.0):
        for q in (0.0, 0.25, 1.0):
            weights = [float(outcome_weight(torch.tensor(p, dtype=torch.double), torch.tensor(q, dtype=torch.double), l, c)) for l in (0, 1) for c in (0, 1)]
            assert all(0.0 <= value <= 1.0 for value in weights)
            assert sum(weights) == pytest.approx(1.0)


def test_primary_basis_is_only_r5_basis_and_derived_directions_do_not_add_columns():
    assert BASIS.shape == (5, 5)
    assert all_direction_vectors().shape == (11, 5)
    assert DERIVED_DIRECTIONS.shape == (6, 5)
    assert torch.equal(BASIS, torch.eye(5, dtype=torch.double))


def test_derived_direction_linearity_has_zero_remainder_for_linear_maps():
    A = torch.arange(25, dtype=torch.double).reshape(5, 5)
    O = torch.arange(15, dtype=torch.double).reshape(3, 5)
    checks = derived_direction_checks(A, O)
    assert all(row["passed"] for row in checks)
    assert max(row["A_relative_remainder"] for row in checks) < 1e-12
    assert max(row["O_relative_remainder"] for row in checks) < 1e-12


def test_world_gate_rejects_rank_inflation_and_nonzero_target_only_observation():
    A = torch.eye(5, dtype=torch.double)
    O = torch.zeros(3, 5, dtype=torch.double)
    O[:, 2] = 1.0
    assert world_gate(A, O)["passed"] is False
