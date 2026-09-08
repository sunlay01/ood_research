import numpy as np

from ood_repr_reg.round3_counterexamples import mean_mechanism_nonidentifiability
from ood_repr_reg.round3_interactions import transport_accounting
from ood_repr_reg.round3_mechanisms import (
    add_direction,
    default_mechanism,
    interaction_terms,
    mechanism_directions,
    moment_matrix,
    risk,
    risk_state,
)
from ood_repr_reg.round3_tangent import tangent_response, tangent_operator


def test_structural_mean_map_and_moment_are_symmetric():
    matrix = moment_matrix(default_mechanism())
    assert matrix.shape == (2, 2)
    assert np.allclose(matrix, matrix.T)


def test_analytic_tangent_matches_finite_difference():
    mechanism = default_mechanism()
    direction = mechanism_directions()["relation"]
    response = tangent_response(np.array([1.0]), np.array([0.75]), np.array([0.3]), mechanism, direction)
    assert response.absolute_error < 1e-7


def test_interaction_terms_close_exact_transport():
    source = default_mechanism()
    target = add_direction(source, mechanism_directions()["relation"], 0.1)
    beta = np.array([1.0])
    w_c = np.array([0.75])
    w_a = np.array([0.3])
    terms = transport_accounting(beta, w_c, w_a, source, target)
    assert abs(terms["closure_error"]) < 1e-10
    assert np.isclose(terms["total"], terms["exact_transport"])


def test_tangent_operator_exposes_overlap_rank():
    directions = mechanism_directions()
    operator = tangent_operator(default_mechanism(), (directions["nuisance"], directions["nuisance_variance"]))
    assert operator.shape == (4, 2)
    assert np.linalg.matrix_rank(operator) >= 1


def test_structural_labels_can_be_nonidentifiable():
    result = mean_mechanism_nonidentifiability()
    assert result["moment_equal"]
