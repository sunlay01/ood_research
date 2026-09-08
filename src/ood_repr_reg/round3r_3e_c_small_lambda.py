"""Small-lambda diagnostics for the 3E-C affine policies."""

from __future__ import annotations

import numpy as np

from .round3r_3c_affine import exact_ift_affine, frozen_quadratic_audit
from .round3r_3e_c_benchmarks import primary_hidden_u_world
from .round3r_3e_c_optimality import full_affine_certificate

Array = np.ndarray


def small_lambda_diagnostics(
    lambdas: tuple[float, ...] = (1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2),
    methods: tuple[str, ...] = ("l2", "irmv1", "vrex"),
) -> dict[str, object]:
    """Compare frozen quadratic and exact IFT expansions near zero."""
    world = primary_hidden_u_world()
    rows: list[dict[str, object]] = []
    for method in methods:
        previous_pi = None
        for lam in lambdas:
            result = exact_ift_affine(world.benchmark, method, lam, world.observation)
            audit = frozen_quadratic_audit(world.benchmark, method, lam)
            frozen_b = np.asarray(audit["b"])
            frozen_z = np.asarray(audit["quadratic_z0"])
            z_scaled_error = float(np.linalg.norm(frozen_z + lam * frozen_b) / max(lam * lam, 1e-30))
            pi_slope = None if previous_pi is None else float(
                np.linalg.norm(result.pi - previous_pi) / max(lam, 1e-30)
            )
            row = {
                "method": method.upper(), "lambda": float(lam), "valid": bool(result.valid),
                "frozen_z_over_lambda_plus_b_norm": z_scaled_error,
                "exact_z_norm": float(np.linalg.norm(result.z0)),
                "pi_zero_reference_norm": None if previous_pi is None else float(np.linalg.norm(previous_pi)),
                "pi_increment_scaled_norm": pi_slope,
            }
            if result.valid:
                certificate = full_affine_certificate(
                    result.z0, world.response, world.observation, result.tangent,
                )
                row["total_excess"] = certificate["total_excess"]
                row["static_tax_lower_bound"] = certificate["static_tax_lower_bound"]
            else:
                row["total_excess"] = None
                row["static_tax_lower_bound"] = None
            rows.append(row)
            previous_pi = result.pi if result.valid else previous_pi
    return {
        "rows": rows,
        "frozen_quadratic_first_order": "z0 = -lambda*b + O(lambda^2)",
        "exact_ift_first_order": "Pi(lambda) = Pi(0) + lambda*Pi_dot + O(lambda^2)",
        "active_spectral_gap_assumed": False,
        "diagnostic_only": True,
    }


__all__ = ["small_lambda_diagnostics"]
