"""Exact source-adaptive affine response for the 3C-B rewrite.

The state variable is the stack of task-complete source moment states.  The
regularized optimum is differentiated in fixed source-whitened coordinates;
the older frozen-``w*`` quadratic action is returned separately as an audit.
No target risk, semantic label, cluster assignment, or exposure score is
needed by this module.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .round3r_3b_benchmark import MomentState, environment_state
from .round3r_3c_benchmark import ThreeCBenchmark
from .round3r_3c_geometry import symmetric_inverse_sqrt, whitened_action
from .round3r_3c_regularizers import evaluate_regularizer, regularizer_state
from .round3r_3e_world_tangent import coupled_primary_geometry
from .round3r_3d_state import task_state, state_components

Array = np.ndarray


def symmetric_sqrt(matrix: Array) -> Array:
    value = (np.asarray(matrix, dtype=float) + np.asarray(matrix, dtype=float).T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(value)
    if eigenvalues.size == 0 or eigenvalues.min() <= 0.0:
        raise ValueError("matrix must be positive definite")
    return eigenvectors @ np.diag(np.sqrt(eigenvalues)) @ eigenvectors.T


@dataclass(frozen=True)
class AffineIFTResult:
    method: str
    lam: float
    z0: Array
    weights: Array
    dz_f: Array
    dy_f: Array
    pi: Array
    tangent: Array
    frozen_b: Array
    frozen_k: Array
    metric_min_eigenvalue: float
    valid: bool
    solver_status: str
    fd_error: float | None


def source_task_state_matrix(benchmark: ThreeCBenchmark) -> Array:
    """Return ``Y`` whose columns are stacked source task states."""
    states = [environment_state(env) for env in benchmark.source_environments]
    return np.concatenate([task_state(state) for state in states])


def _state_shape(source_vector: Array, feature_dimension: int) -> tuple[int, int]:
    one = feature_dimension * (feature_dimension + 1) // 2 + feature_dimension + 1
    if source_vector.size % one:
        raise ValueError("stacked task state has incompatible feature dimension")
    return source_vector.size // one, one


def _states_from_vector(vector: Array, feature_dimension: int) -> tuple[MomentState, ...]:
    vector = np.asarray(vector, dtype=float)
    count, width = _state_shape(vector, feature_dimension)
    return tuple(
        MomentState(*state_components(vector[i * width:(i + 1) * width]))
        for i in range(count)
    )


def _source_state_value_gradient(weights: Array, source_vector: Array,
                                 feature_dimension: int) -> tuple[float, Array]:
    states = _states_from_vector(source_vector, feature_dimension)
    values = np.asarray([
        weights @ state.second @ weights - 2.0 * weights @ state.xy + state.y2
        for state in states
    ])
    gradients = np.asarray([2.0 * (state.second @ weights - state.xy) for state in states])
    return float(values.mean()), gradients.mean(axis=0)


def _mean_state(source_vector: Array, feature_dimension: int) -> MomentState:
    states = _states_from_vector(source_vector, feature_dimension)
    return MomentState(
        np.mean([state.second for state in states], axis=0),
        np.mean([state.xy for state in states], axis=0),
        float(np.mean([state.y2 for state in states])),
    )


def regularizer_value_gradient(method: str, weights: Array, source_vector: Array,
                               feature_dimension: int) -> tuple[float, Array]:
    """Evaluate a source regularizer from the task-complete state stack."""
    states = _states_from_vector(source_vector, feature_dimension)
    w = np.asarray(weights, dtype=float)
    risks = np.asarray([w @ s.second @ w - 2.0 * w @ s.xy + s.y2 for s in states])
    gradients = np.asarray([2.0 * (s.second @ w - s.xy) for s in states])
    key = method.lower()
    if key == "l2":
        return 0.5 * float(w @ w), w.copy()
    if key == "vrex":
        centered = risks - risks.mean()
        return float(np.mean(centered ** 2)), 2.0 * np.mean(centered[:, None] * gradients, axis=0)
    if key == "irmv1":
        radial = np.asarray([2.0 * (w @ s.second @ w - w @ s.xy) for s in states])
        radial_grad = np.asarray([4.0 * s.second @ w - 2.0 * s.xy for s in states])
        return float(np.mean(radial ** 2)), 2.0 * np.mean(radial[:, None] * radial_grad, axis=0)
    if key == "coral":
        return 0.0, np.zeros_like(w)
    raise ValueError(f"unknown regularizer: {method}")


def source_objective(weights: Array, source_vector: Array, feature_dimension: int,
                     method: str, lam: float) -> float:
    risk_value, _ = _source_state_value_gradient(weights, source_vector, feature_dimension)
    penalty, _ = regularizer_value_gradient(method, weights, source_vector, feature_dimension)
    return risk_value + lam * penalty


def source_objective_gradient(weights: Array, source_vector: Array, feature_dimension: int,
                              method: str, lam: float) -> Array:
    _, risk_grad = _source_state_value_gradient(weights, source_vector, feature_dimension)
    _, penalty_grad = regularizer_value_gradient(method, weights, source_vector, feature_dimension)
    return risk_grad + lam * penalty_grad


def _solve_regularized(benchmark: ThreeCBenchmark, method: str, lam: float,
                       source_vector: Array | None = None) -> tuple[Array, bool, str]:
    vector = source_task_state_matrix(benchmark) if source_vector is None else np.asarray(source_vector)
    p = benchmark.optimum.size
    mean_state = _mean_state(vector, p)
    if lam == 0.0:
        return np.linalg.solve(mean_state.second, mean_state.xy), True, "source_optimum"
    if method.lower() == "l2":
        metric = 2.0 * mean_state.second + lam * np.eye(p)
        return np.linalg.solve(metric, 2.0 * mean_state.xy), True, "closed_form_l2"
    try:
        from scipy.optimize import root
        result = root(
            lambda w: source_objective_gradient(w, vector, p, method, lam),
            benchmark.optimum.copy(), method="hybr", tol=1e-11,
        )
        grad_norm = np.linalg.norm(source_objective_gradient(result.x, vector, p, method, lam))
        if not result.success and grad_norm >= 1e-8:
            raise RuntimeError(str(result.message))
        return result.x, bool(grad_norm < 1e-8), str(result.message)
    except Exception as exc:  # pragma: no cover - only used without scipy
        return benchmark.optimum.copy(), False, f"solver_error:{exc}"


def _torch_jacobians(method: str, lam: float, z0: Array, source_vector: Array,
                     benchmark: ThreeCBenchmark) -> tuple[Array, Array]:
    """Autodiff ``F`` in the exact population source objective."""
    import torch

    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        p = benchmark.optimum.size
        hroot = torch.tensor(symmetric_inverse_sqrt(benchmark.hessian))
        wstar = torch.tensor(benchmark.optimum)
        z = torch.tensor(z0, requires_grad=True)
        y = torch.tensor(source_vector, requires_grad=True)
        width = p * (p + 1) // 2 + p + 1

        def unpack(block):
            matrix = torch.zeros((p, p), dtype=block.dtype)
            index = 0
            for row in range(p):
                matrix[row, row] = block[index]
                index += 1
                for col in range(row):
                    matrix[row, col] = block[index] / np.sqrt(2.0)
                    matrix[col, row] = block[index] / np.sqrt(2.0)
                    index += 1
            return matrix, block[index:index + p], block[-1]

        blocks = [unpack(y[i:i + width]) for i in range(0, y.numel(), width)]
        w = wstar + hroot @ z
        risks = torch.stack([w @ m @ w - 2.0 * w @ xy + y2 for m, xy, y2 in blocks])
        gradients = torch.stack([2.0 * (m @ w - xy) for m, xy, _ in blocks])
        source_grad = gradients.mean(dim=0)
        if method.lower() == "l2":
            penalty_grad = w
        elif method.lower() == "vrex":
            centered = risks - risks.mean()
            penalty_grad = 2.0 * torch.mean(centered[:, None] * gradients, dim=0)
        elif method.lower() == "irmv1":
            radial = torch.stack([2.0 * (w @ m @ w - w @ xy) for m, xy, _ in blocks])
            radial_grad = torch.stack([4.0 * m @ w - 2.0 * xy for m, xy, _ in blocks])
            penalty_grad = 2.0 * torch.mean(radial[:, None] * radial_grad, dim=0)
        else:
            penalty_grad = torch.zeros_like(w)
        f = hroot @ (source_grad + lam * penalty_grad)
        dz = torch.stack([torch.autograd.grad(entry, z, retain_graph=True)[0] for entry in f])
        dy = torch.stack([torch.autograd.grad(entry, y, retain_graph=True)[0] for entry in f])
        return dz.detach().numpy(), dy.detach().numpy()
    finally:
        torch.set_default_dtype(old)


def _finite_jacobian(function, point: Array, step: float = 2e-5) -> Array:
    point = np.asarray(point, dtype=float)
    value = np.asarray(function(point))
    result = np.zeros((value.size, point.size))
    for index in range(point.size):
        plus, minus = point.copy(), point.copy()
        plus[index] += step
        minus[index] -= step
        result[:, index] = (np.asarray(function(plus)) - np.asarray(function(minus))) / (2.0 * step)
    return result


def exact_ift_affine(benchmark: ThreeCBenchmark, method: str, lam: float,
                     observation: Array | None = None) -> AffineIFTResult:
    """Compute exact population base, IFT Jacobians and affine world action."""
    y0 = source_task_state_matrix(benchmark)
    hroot = symmetric_inverse_sqrt(benchmark.hessian)
    root = symmetric_sqrt(benchmark.hessian)
    weights, success, status = _solve_regularized(benchmark, method, lam, y0)
    z0 = root @ (weights - benchmark.optimum)
    metric = benchmark.hessian.copy()
    try:
        dz, dy = _torch_jacobians(method, lam, z0, y0, benchmark)
    except Exception:
        # This is an audit fallback only; exact source objective remains the definition.
        dz = _finite_jacobian(lambda z: hroot @ source_objective_gradient(
            benchmark.optimum + hroot @ z, y0, benchmark.optimum.size, method, lam), z0)
        dy = _finite_jacobian(lambda y: hroot @ source_objective_gradient(
            weights, y, benchmark.optimum.size, method, lam), y0)
        status = status + ":finite_difference_jacobian"
    eigen = np.linalg.eigvalsh((dz + dz.T) / 2.0)
    min_eigen = float(eigen.min()) if eigen.size else 0.0
    valid = bool(success and np.all(np.isfinite(dz)) and min_eigen > 1e-9)
    pi = -np.linalg.solve(dz, dy) if valid else np.full((dz.shape[0], dy.shape[1]), np.nan)
    tangent = None if observation is None else pi @ observation
    state = regularizer_state(method, benchmark)
    b, k, _ = whitened_action(metric, state)
    return AffineIFTResult(method.upper(), float(lam), z0, weights, dz, dy,
                           pi, np.zeros((dz.shape[0], 0)) if tangent is None else tangent,
                           b, k, min_eigen, valid, status, None)


def central_difference_trained_action(benchmark: ThreeCBenchmark, method: str, lam: float,
                                      observation: Array, steps: tuple[float, ...] = (1e-4, 1e-5, 1e-6)) -> dict[str, object]:
    """Compare exact regularized optimizer differences with ``Pi O``."""
    result = exact_ift_affine(benchmark, method, lam, observation)
    y0 = source_task_state_matrix(benchmark)
    rows = []
    for step in steps:
        errors = []
        for index in range(observation.shape[1]):
            plus, minus = y0 + step * observation[:, index], y0 - step * observation[:, index]
            wp, okp, _ = _solve_regularized(benchmark, method, lam, plus)
            wm, okm, _ = _solve_regularized(benchmark, method, lam, minus)
            if not (okp and okm):
                errors.append(float("nan"))
                continue
            root = symmetric_inverse_sqrt(benchmark.hessian)
            fd = symmetric_sqrt(benchmark.hessian) @ (wp - wm) / (2.0 * step)
            errors.append(float(np.linalg.norm(fd - result.tangent[:, index]) / max(1.0, np.linalg.norm(result.tangent[:, index]))))
        rows.append({"step": step, "max_relative_error": float(np.nanmax(errors))})
    return {"method": method.upper(), "lambda": float(lam), "rows": rows,
            "max_error": float(max(row["max_relative_error"] for row in rows)),
            "pass": bool(result.valid and max(row["max_relative_error"] for row in rows) < 1e-5)}


def frozen_quadratic_audit(benchmark: ThreeCBenchmark, method: str, lam: float) -> dict[str, object]:
    state = regularizer_state(method, benchmark)
    b, k, _ = whitened_action(benchmark.hessian, state)
    metric = np.eye(b.size) + lam * k
    valid = bool(np.linalg.eigvalsh((metric + metric.T) / 2.0).min() > 1e-9)
    z = -lam * np.linalg.solve(metric, b) if valid else np.full_like(b, np.nan)
    return {"b": b, "K": k, "quadratic_z0": z,
            "quadratic_Pi": -np.linalg.solve(metric, np.eye(b.size)) if valid else np.full_like(metric, np.nan),
            "valid": valid, "source": "frozen_wstar_audit_only"}


def main_benchmark(methods: tuple[str, ...] = ("l2", "irmv1", "vrex"),
                   lambdas: tuple[float, ...] = (0.0, 1e-4, 1e-3, 1e-2, 1e-1),
                   *, family=None, family_geometry=None) -> dict[str, object]:
    benchmark = __import__("ood_repr_reg.round3r_3c_benchmark", fromlist=["make_benchmark"]).make_benchmark(family=family)
    if family_geometry is not None:
        geometry = family_geometry
    elif family is None:
        geometry = coupled_primary_geometry()
    else:
        from .environment_family.geometry import build_task_geometry

        geometry = build_task_geometry(family)
    rows = []
    audits = []
    for method in methods:
        for lam in lambdas:
            result = exact_ift_affine(benchmark, method, lam, geometry.observation)
            fd = central_difference_trained_action(benchmark, method, lam, geometry.observation)
            result = AffineIFTResult(result.method, result.lam, result.z0, result.weights, result.dz_f,
                                     result.dy_f, result.pi, result.tangent, result.frozen_b,
                                     result.frozen_k, result.metric_min_eigenvalue, result.valid,
                                     result.solver_status, fd["max_error"])
            rows.append({"method": result.method, "lambda": result.lam,
                         "norm_z0": float(np.linalg.norm(result.z0)),
                         "rank_K": int(np.linalg.matrix_rank(result.frozen_k, tol=1e-9)),
                         "kernel_dimension": int(result.frozen_k.shape[0] - np.linalg.matrix_rank(result.frozen_k, tol=1e-9)),
                         "min_local_metric_eigenvalue": result.metric_min_eigenvalue,
                         "pi_operator_norm": float(np.linalg.svd(result.pi, compute_uv=False)[0]) if result.valid else float("nan"),
                         "fd_error": result.fd_error, "status": "PASS" if result.valid and fd["pass"] else "INVALID_OR_AUDIT_FAIL"})
            audits.append({"method": result.method, "lambda": result.lam, "z0": result.z0,
                           "weights": result.weights,
                           "dz_f": result.dz_f, "dy_f": result.dy_f, "pi": result.pi,
                           "tangent": result.tangent, "quadratic_audit": frozen_quadratic_audit(benchmark, method, lam),
                           "fd_audit": fd})
    return {"rows": rows, "audits": audits, "benchmark": benchmark, "geometry": geometry,
            "target_risk_used": False, "semantic_labels_used": False, "cluster_labels_used": False,
            "regularizer_geometry_used_for_selection": False}


__all__ = [
    "AffineIFTResult", "source_task_state_matrix", "regularizer_value_gradient",
    "source_objective", "source_objective_gradient", "exact_ift_affine",
    "central_difference_trained_action", "frozen_quadratic_audit", "main_benchmark",
]
