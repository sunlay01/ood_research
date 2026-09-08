"""Local tangent diagnostic for a smooth nonlinear source/response map."""

from __future__ import annotations

import numpy as np

from .round3r_3e_recovery import operator_summary


def nonlinear_local_stress() -> dict[str, object]:
    # z -> (z0, z1^2 + z2) is a source observable and
    # z -> (z0 + z1*z2, z2) is a response map.  These are Jacobians at z=0.
    point = np.zeros(3)
    observation_jacobian = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    response_jacobian = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    return {
        "status": "LOCAL-TANGENT-DIAGNOSTIC-ONLY",
        "reference_point": point,
        "observation_jacobian": observation_jacobian,
        "response_jacobian": response_jacobian,
        "recovery_summary": operator_summary(response_jacobian, observation_jacobian),
        "global_nonlinear_claim": False,
    }


__all__ = ["nonlinear_local_stress"]
