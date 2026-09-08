"""Held-out mixed-shift response validation."""

from __future__ import annotations

import numpy as np

from .round3r_3b_benchmark import MomentState, ShiftProbe, mixed_probe, risk_gradient
from .round3r_3b_geometry import mixed_shift_decomposition, whitened_responses


def mixed_shift_additivity(response_a: np.ndarray, response_b: np.ndarray, response_mixed: np.ndarray) -> dict[str, float | bool]:
    residual = np.asarray(response_mixed) - np.asarray(response_a) - np.asarray(response_b)
    scale = max(float(np.linalg.norm(response_mixed)), 1e-30)
    return {"residual_norm": float(np.linalg.norm(residual)), "relative_residual": float(np.linalg.norm(residual) / scale),
            "exact": bool(np.linalg.norm(residual) < 1e-9 * scale + 1e-12)}


def validate_mixed_shifts(probes: tuple[ShiftProbe, ...], source: MomentState, optimum: np.ndarray,
                          modules: dict[str, np.ndarray], names: tuple[tuple[str, ...], ...],
                          hessian: np.ndarray) -> list[dict[str, object]]:
    by_id = {p.shift_id: p for p in probes}
    pure_gradient = {p.shift_id: risk_gradient(optimum, source, p.target) for p in probes}
    rows = []
    for composition in names:
        mixed = mixed_probe(probes, composition, source)
        gradient = risk_gradient(optimum, source, mixed.target)
        expected = sum((pure_gradient[name] for name in composition), np.zeros_like(optimum))
        additivity = mixed_shift_additivity(
            sum((pure_gradient[name] for name in composition[:1]), np.zeros_like(optimum)),
            sum((pure_gradient[name] for name in composition[1:]), np.zeros_like(optimum)),
            gradient,
        )
        whitened = whitened_responses(hessian, gradient)[:, 0]
        reconstruction = mixed_shift_decomposition(whitened, modules)
        rows.append({"shift_id": mixed.shift_id, "components": composition, "additivity": additivity,
                     "reconstruction_residual": reconstruction["residual_norm"],
                     "joint_rank": reconstruction["joint_rank"], "unique_if_direct_sum": reconstruction["unique_if_direct_sum"]})
    del by_id
    return rows
