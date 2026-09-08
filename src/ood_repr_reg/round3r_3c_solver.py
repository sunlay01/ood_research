"""Local and exact population regularized solvers for 3C."""

from __future__ import annotations

import numpy as np

from .round3r_3c_benchmark import ThreeCBenchmark, source_risk, target_risk
from .round3r_3c_regularizers import evaluate_regularizer
from .round3r_3c_geometry import local_delta

Array = np.ndarray


def local_solution(benchmark: ThreeCBenchmark, method: str, lam: float) -> dict[str, object]:
    _, gradient, hessian = evaluate_regularizer(method, benchmark.optimum, benchmark)
    from .round3r_3c_regularizers import RegularizerState

    state = RegularizerState(method.upper(), "", "", "", 0.0, gradient, hessian, "")
    delta, valid, minimum = local_delta(benchmark.hessian, state, lam)
    return {"weights": benchmark.optimum + delta, "delta": delta, "valid": valid, "minimum_eigenvalue": minimum}


def exact_solution(benchmark: ThreeCBenchmark, method: str, lam: float) -> dict[str, object]:
    from scipy.optimize import minimize

    if lam == 0.0 or method.lower() == "coral":
        weights = benchmark.optimum.copy()
        return {"weights": weights, "success": True, "status": "closed_form", "gradient_norm": float(np.linalg.norm(benchmark.hessian @ (weights - benchmark.optimum)))}

    def objective(weights: Array) -> float:
        source = source_risk(benchmark, weights)
        value = evaluate_regularizer(method, weights, benchmark)[0]
        return source + lam * value

    def gradient(weights: Array) -> Array:
        source_gradient = benchmark.hessian @ (weights - benchmark.optimum)
        return source_gradient + lam * evaluate_regularizer(method, weights, benchmark)[1]

    result = minimize(objective, benchmark.optimum, jac=gradient, method="BFGS",
                      options={"gtol": 1e-10, "maxiter": 2000})
    return {"weights": result.x, "success": bool(result.success or np.linalg.norm(gradient(result.x)) < 1e-7),
            "status": str(result.message), "gradient_norm": float(np.linalg.norm(gradient(result.x)))}


def lambda_grid() -> Array:
    return np.concatenate(([0.0], np.logspace(-6, 1, 15)))


def lambda_path(benchmark: ThreeCBenchmark, method: str, lambdas: Array | None = None) -> list[dict[str, object]]:
    values = lambda_grid() if lambdas is None else np.asarray(lambdas, dtype=float)
    rows = []
    for lam in values:
        local = local_solution(benchmark, method, float(lam))
        if not local["valid"]:
            rows.append({"method": method.upper(), "lambda": float(lam), "local_valid": False,
                         "metric_min_eigenvalue": local["minimum_eigenvalue"]})
            break
        exact = exact_solution(benchmark, method, float(lam))
        local_weights = np.asarray(local["weights"])
        exact_weights = np.asarray(exact["weights"])
        rows.append({
            "method": method.upper(), "lambda": float(lam), "local_valid": True,
            "metric_min_eigenvalue": float(local["minimum_eigenvalue"]),
            "local_displacement_norm": float(np.linalg.norm(local_weights - benchmark.optimum)),
            "exact_displacement_norm": float(np.linalg.norm(exact_weights - benchmark.optimum)),
            "local_exact_error": float(np.linalg.norm(local_weights - exact_weights)),
            "exact_source_risk": source_risk(benchmark, exact_weights),
            "exact_optimizer_success": bool(exact["success"]),
            "exact_optimizer_gradient_norm": exact["gradient_norm"],
        })
    return rows
