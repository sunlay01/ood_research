"""Minimal abstract counterexamples for the 3E-C theorems."""

from __future__ import annotations

import numpy as np

from .round3r_3e_c_optimality import full_affine_certificate
from .round3r_3e_joint_fixtures import same_b_k_different_pi


def counterexamples() -> dict[str, object]:
    # The first output coordinate is the information-hard direction.  The
    # second coordinate has slack, so a nonzero E can remain minimax-optimal.
    irreducible = np.diag([2.0, 0.0])
    recoverable = np.diag([0.0, 1.0])
    optimal_with_nonzero_e = full_affine_certificate(
        np.zeros(2), irreducible, recoverable, recoverable,
    )
    static_only = full_affine_certificate(
        np.array([0.5, 0.0]), irreducible, recoverable, np.zeros((2, 2)),
    )
    complete_info_failure = full_affine_certificate(
        np.zeros(2), np.zeros((2, 2)), np.eye(2), np.diag([0.5, 0.0]),
    )
    tight = full_affine_certificate(
        np.array([0.0, 0.5]), irreducible, recoverable, np.zeros((2, 2)),
    )
    strict = full_affine_certificate(
        np.array([0.5, 0.0]), irreducible, recoverable, np.zeros((2, 2)),
    )
    return {
        "nonzero_E_minimax_optimal": optimal_with_nonzero_e,
        "E_zero_nonzero_static_not_optimal": static_only,
        "complete_information_nonzero_E_failure": complete_info_failure,
        "static_tax_tight": tight,
        "static_tax_strict": strict,
        "same_b_k_different_pi": same_b_k_different_pi(),
    }


__all__ = ["counterexamples"]
