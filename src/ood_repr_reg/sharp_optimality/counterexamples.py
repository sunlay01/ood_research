"""Small exact fixtures separating spectral placement from raw residual size."""

from __future__ import annotations

import numpy as np

from .geometry import affine_policy_audit


def counterexamples() -> dict[str, dict[str, object]]:
    # O observes the second coordinate. R acts on the first output coordinate;
    # the second output is slack, so a nonzero E can still be minimax optimal.
    a = np.diag([2.0, 1.0])
    # The first world coordinate is observed.  The invisible response is the
    # second output, whose slack is strictly positive.
    o = np.array([[1.0, 0.0]])
    examples = {
        # ``adaptive`` below is Pi O, so E=A_rec+adaptive.  Its second column
        # must vanish, because the second world coordinate is source-invisible.
        "nonzero_E_within_slack": (np.zeros(2), a, o, np.array([[-1.5, 0.0], [0.0, 0.0]])),
        "small_E_in_zero_slack_direction": (np.zeros(2), a, o, np.array([[-2.0, 0.0], [0.01, 0.0]])),
        "static_steering_breaks_optimality": (np.array([0.2, 0.0]), a, o, np.array([[-1.5, 0.0], [0.0, 0.0]])),
        "same_norm_different_placement_good": (np.zeros(2), a, o, np.array([[-1.5, 0.0], [0.0, 0.0]])),
        "same_norm_different_placement_bad": (np.zeros(2), a, o, np.array([[-2.0, 0.0], [0.5, 0.0]])),
    }
    return {name: {"example": name, **affine_policy_audit(*values)} for name, values in examples.items()}


__all__ = ["counterexamples"]
