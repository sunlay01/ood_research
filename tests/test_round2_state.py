import numpy as np

from ood_repr_reg.round2_state import (
    ambiguity_support,
    recover_target_moment_from_sources,
    risk_visible_basis,
    source_observation_map,
    source_target_ambiguity,
    symmetric_devectorize,
    symmetric_vectorize,
)


def _moments():
    return (
        np.array([[1.0, 0.0], [0.0, 1.0]]),
        np.array([[1.0, 0.5], [0.5, 1.25]]),
        np.array([[1.0, -0.5], [-0.5, 1.25]]),
    )


def test_symmetric_vectorization_is_isometric_and_reversible():
    matrix = np.array([[2.0, 0.4], [0.4, 3.0]])
    vector = symmetric_vectorize(matrix)
    assert np.isclose(vector @ vector, np.sum(matrix * matrix))
    assert np.allclose(symmetric_devectorize(vector, 2), matrix)


def test_target_annihilator_is_invisible_to_all_target_moments():
    quotient = risk_visible_basis(_moments())
    annihilator = quotient.annihilator_basis
    assert annihilator.shape[1] == quotient.ambient_dimension - quotient.visible_basis.shape[1]
    if annihilator.shape[1]:
        assert np.allclose(quotient.target_vector_matrix.T @ annihilator, 0.0, atol=1e-10)


def test_source_target_ambiguity_removes_raw_kernel_that_target_cannot_see():
    sources = _moments()[:2]
    report = source_target_ambiguity(sources, _moments())
    assert report.target_rank == 3
    assert report.source_rank_in_quotient == 2
    assert report.ambiguity_dimension == 1
    assert report.source_kernel_dimension == 1


def test_target_in_source_span_is_exactly_recoverable():
    sources = _moments()[:2]
    target = 0.3 * sources[0] + 0.7 * sources[1]
    recovery = recover_target_moment_from_sources(sources, target)
    assert recovery.exact
    assert recovery.residual_norm < 1e-10


def test_shift_ball_support_function_uses_target_visible_projection():
    state = np.array([1.0, 2.0, 3.0])
    shift_space = np.eye(3)
    assert np.isclose(ambiguity_support(state, shift_space, 0.4), 0.4 * np.linalg.norm(state))


def test_centered_and_absolute_source_maps_are_distinct():
    absolute = source_observation_map(_moments())
    centered = source_observation_map(_moments(), centered=True)
    assert absolute.matrix.shape == (3, 3)
    assert centered.matrix.shape == (2, 3)

