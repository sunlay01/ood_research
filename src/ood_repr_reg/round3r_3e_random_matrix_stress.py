"""Numerical audits for the finite-dimensional recovery theorem."""

from __future__ import annotations

import numpy as np

from .round3r_3e_recovery import (
    ambiguity_diameter,
    irreducible_response_operator,
    minimax_recovery_error,
    nullspace_basis,
    pseudoinverse_residual,
)


def random_matrix_stress(seed: int = 20260908, repeats: int = 32) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    rows = []
    max_formula_error = 0.0
    max_random_unit_pseudoinverse_slack = 0.0
    max_sampled_lower_bound_gap = 0.0
    max_estimator_lower_bound_violation = 0.0
    for index in range(repeats):
        d = int(rng.integers(2, 9))
        response_dim = int(rng.integers(1, 5))
        observation_dim = int(rng.integers(0, d + 1))
        a = rng.normal(size=(response_dim, d))
        o = rng.normal(size=(observation_dim, d)) if observation_dim else np.zeros((0, d))
        n = nullspace_basis(o)
        alpha = minimax_recovery_error(a, o)
        projector_value = float(np.linalg.svd(irreducible_response_operator(a, o), compute_uv=False)[0]) if alpha else 0.0
        formula_error = abs(alpha - projector_value)
        max_formula_error = max(max_formula_error, formula_error)
        unit = rng.normal(size=d)
        unit /= np.linalg.norm(unit)
        residual = np.linalg.norm(pseudoinverse_residual(a, o, unit))
        random_unit_slack = max(0.0, alpha - residual)
        max_random_unit_pseudoinverse_slack = max(max_random_unit_pseudoinverse_slack, random_unit_slack)
        # Every deterministic estimator must incur at least alpha on the pair
        # +/-v at y=0.  Use the leading right singular vector of A N.
        if n.shape[1]:
            _, _, vh = np.linalg.svd(a @ n, full_matrices=False)
            v = n @ vh[0]
            v /= np.linalg.norm(v)
            pair_gap = max(0.0, alpha - 0.5 * (np.linalg.norm(a @ v) + np.linalg.norm(a @ (-v))))
            max_sampled_lower_bound_gap = max(max_sampled_lower_bound_gap, pair_gap)
            # Audit the lower bound against a finite family of arbitrary
            # affine source-only rules.  The analytic theorem covers all
            # deterministic rules; this is only an implementation check.
            for _ in range(8):
                estimator = rng.normal(size=(response_dim, observation_dim))
                offset = rng.normal(size=response_dim)
                positive_error = np.linalg.norm(a @ v - (estimator @ (o @ v) + offset))
                negative_error = np.linalg.norm(a @ (-v) - (estimator @ (o @ (-v)) + offset))
                max_estimator_lower_bound_violation = max(
                    max_estimator_lower_bound_violation,
                    alpha - max(positive_error, negative_error),
                )
        rows.append({"index": index, "dim_U": d, "rank_O": int(np.linalg.matrix_rank(o)),
                     "kernel_dimension": int(n.shape[1]), "alpha": alpha,
                     "ambiguity_diameter": ambiguity_diameter(a, o)})
    return {
        "seed": seed,
        "repeats": repeats,
        "rows": rows,
        "max_projector_formula_error": max_formula_error,
        "max_random_unit_pseudoinverse_slack": max_random_unit_pseudoinverse_slack,
        "max_adversarial_pair_lower_bound_gap": max_sampled_lower_bound_gap,
        "max_sampled_estimator_lower_bound_violation": max_estimator_lower_bound_violation,
        "audit_pass": bool(
            max_formula_error < 1e-8
            and max_sampled_lower_bound_gap < 1e-8
            and max_estimator_lower_bound_violation < 1e-8
        ),
    }


__all__ = ["random_matrix_stress"]
