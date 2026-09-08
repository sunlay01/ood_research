"""Small nonlinear moment stress for the 3C local-action calculations."""

from __future__ import annotations

import numpy as np

from .round3r_3b_benchmark import ModuleEnvironment, nonlinear_environment_state, source_design, source_optimum


def run_nonlinear_stress() -> dict[str, object]:
    base = ModuleEnvironment(n_noise=4)
    states = tuple(nonlinear_environment_state(environment) for environment in source_design(base, True))
    source = states[0]
    optimum = np.linalg.solve(sum((state.second for state in states), np.zeros_like(source.second)) / len(states),
                              sum((state.xy for state in states), np.zeros_like(source.xy)) / len(states))
    hessian = 2.0 * sum((state.second for state in states), np.zeros_like(source.second)) / len(states)
    return {"status": "stress_only", "dimension": int(source.second.shape[0]),
            "source_hessian_min_eigenvalue": float(np.linalg.eigvalsh(hessian).min()),
            "optimum_norm": float(np.linalg.norm(optimum)),
            "moment_psd": bool(all(np.linalg.eigvalsh(state.second).min() > -1e-9 for state in states)),
            "deep_network_claim": False}
