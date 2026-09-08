"""Analytic, autodiff, and finite-difference derivative audits."""

from __future__ import annotations

import numpy as np

from .round3r_3c_benchmark import ThreeCBenchmark
from .round3r_3c_regularizers import evaluate_regularizer

Array = np.ndarray


def _torch_derivatives(method: str, point: Array, benchmark: ThreeCBenchmark) -> tuple[Array, Array]:
    import torch

    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        value = torch.tensor(point, requires_grad=True)
        moments = [
            (torch.tensor(environment.second), torch.tensor(environment.xy), float(environment.y2))
            for environment in _env_states(benchmark)
        ]

        def objective(w):
            risks = torch.stack([w @ second @ w - 2.0 * w @ xy + y2 for second, xy, y2 in moments])
            if method == "l2":
                return 0.5 * (w @ w)
            if method == "vrex":
                return torch.var(risks, unbiased=False)
            if method == "irmv1":
                derivatives = torch.stack([2.0 * (w @ second @ w - w @ xy) for second, xy, _ in moments])
                return torch.mean(derivatives * derivatives)
            if method == "coral":
                return torch.zeros((), dtype=w.dtype) + 0.0 * (w @ w)
            raise ValueError(method)

        gradient = torch.autograd.grad(objective(value), value, create_graph=True)[0]
        rows = []
        for entry in gradient:
            rows.append(torch.autograd.grad(entry, value, retain_graph=True)[0])
        return gradient.detach().numpy(), torch.stack(rows).detach().numpy()
    finally:
        torch.set_default_dtype(previous_dtype)


def _env_states(benchmark: ThreeCBenchmark):
    from .round3r_3c_regularizers import _env_moments

    return _env_moments(benchmark)


def _finite_gradient(function, point: Array, step: float = 1e-5) -> Array:
    result = np.zeros_like(point, dtype=float)
    for index in range(point.size):
        plus = point.copy(); plus[index] += step
        minus = point.copy(); minus[index] -= step
        result[index] = (function(plus) - function(minus)) / (2.0 * step)
    return result


def _finite_hessian(function, point: Array, step: float = 2e-4) -> Array:
    dimension = point.size
    result = np.zeros((dimension, dimension))
    for index in range(dimension):
        plus = point.copy(); plus[index] += step
        minus = point.copy(); minus[index] -= step
        result[:, index] = (_finite_gradient(function, plus, step) - _finite_gradient(function, minus, step)) / (2.0 * step)
    return (result + result.T) / 2.0


def audit_regularizer(method: str, point: Array, benchmark: ThreeCBenchmark) -> dict[str, object]:
    value, analytic_gradient, analytic_hessian = evaluate_regularizer(method, point, benchmark)
    autodiff_gradient, autodiff_hessian = _torch_derivatives(method.lower(), point, benchmark)
    function = lambda x: evaluate_regularizer(method, x, benchmark)[0]
    finite_gradient = _finite_gradient(function, point)
    finite_hessian = _finite_hessian(function, point)
    gradient_error = max(float(np.max(np.abs(analytic_gradient - autodiff_gradient))), float(np.max(np.abs(analytic_gradient - finite_gradient))))
    hessian_error = max(float(np.max(np.abs(analytic_hessian - autodiff_hessian))), float(np.max(np.abs(analytic_hessian - finite_hessian))))
    eig = np.linalg.eigvalsh((analytic_hessian + analytic_hessian.T) / 2.0)
    singular = np.linalg.svd(analytic_hessian, compute_uv=False)
    return {
        "method": method.upper(),
        "value": float(value),
        "gradient_max_error": gradient_error,
        "hessian_max_error": hessian_error,
        "hessian_symmetry_error": float(np.max(np.abs(analytic_hessian - analytic_hessian.T))),
        "min_eigenvalue": float(eig.min()),
        "max_eigenvalue": float(eig.max()),
        "numerical_rank": int(np.sum(singular > 1e-9 * max(singular[0], 1e-30))) if singular.size else 0,
        "condition_number": float(singular[0] / singular[-1]) if singular.size and singular[-1] > 1e-15 else float("inf"),
        "psd_or_indefinite": "PSD" if eig.min() >= -1e-9 else "indefinite",
        "audit_pass": bool(gradient_error < 1e-5 and hessian_error < 1e-3 and np.max(np.abs(analytic_hessian - analytic_hessian.T)) < 1e-10),
    }
