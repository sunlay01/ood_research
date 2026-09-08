"""Minimal structural non-canonicity and hidden-composition constructions."""

from __future__ import annotations

import numpy as np

from .round3_mechanisms import StructuralMechanism, moment_matrix


def mean_mechanism_nonidentifiability() -> dict[str, object]:
    """Two structural explanations induce exactly the same observed law.

    The nuisance intercept b and the innovation mean mu_xi enter A only through
    b + mu_xi.  Without a zero-mean innovation convention, their labels cannot
    be recovered from the observed distribution.
    """
    common = dict(
        mu_c=np.array([0.0]),
        sigma_c=np.array([[1.0]]),
        gamma=np.array([[0.8]]),
        sigma_xi=np.array([[0.4]]),
    )
    first = StructuralMechanism(b=np.array([0.3]), mu_xi=np.array([0.0]), **common)
    second = StructuralMechanism(b=np.array([0.0]), mu_xi=np.array([0.3]), **common)
    return {
        "moment_equal": bool(np.allclose(moment_matrix(first), moment_matrix(second))),
        "first": first,
        "second": second,
        "conclusion": "structural labels are not identifiable without an innovation centering convention",
    }


def relation_variance_hidden_composition() -> dict[str, object]:
    """Different mechanisms can agree on a selected statistic but not all moments."""
    base = StructuralMechanism(
        np.array([0.0]), np.array([[1.0]]), np.array([[0.8]]), np.array([0.0]), np.array([0.0]), np.array([[0.4]])
    )
    relation_changed = StructuralMechanism(
        base.mu_c, base.sigma_c, np.array([[1.0]]), base.b, base.mu_xi, np.array([[0.04]])
    )
    nuisance_changed = StructuralMechanism(
        base.mu_c, base.sigma_c, base.gamma, base.b, base.mu_xi, np.array([[0.04 + 1.0 - 0.64]])
    )
    return {
        "relation_moment": moment_matrix(relation_changed),
        "nuisance_moment": moment_matrix(nuisance_changed),
        "same_aa_block": bool(np.isclose(moment_matrix(relation_changed)[1, 1], moment_matrix(nuisance_changed)[1, 1])),
        "full_moment_equal": bool(np.allclose(moment_matrix(relation_changed), moment_matrix(nuisance_changed))),
        "conclusion": "matching one marginal statistic does not identify the generating mechanism",
    }
