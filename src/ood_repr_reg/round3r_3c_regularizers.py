"""Analytic population regularizers for the 3C local-action audit."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .round3r_3c_benchmark import ThreeCBenchmark, environment_risks, source_gradient

Array = np.ndarray


@dataclass(frozen=True)
class RegularizerState:
    method: str
    constraint_object: str
    predictor_status: str
    gauge_status: str
    value: float
    gradient: Array
    hessian: Array
    notes: str


def _env_moments(benchmark: ThreeCBenchmark) -> tuple[Array, ...]:
    from .round3r_3b_benchmark import environment_state

    return tuple(environment_state(environment) for environment in benchmark.source_environments)


def l2_value_gradient_hessian(weights: Array) -> tuple[float, Array, Array]:
    w = np.asarray(weights, dtype=float)
    return 0.5 * float(w @ w), w.copy(), np.eye(w.size)


def vrex_value_gradient_hessian(weights: Array, benchmark: ThreeCBenchmark) -> tuple[float, Array, Array]:
    w = np.asarray(weights, dtype=float)
    moments = _env_moments(benchmark)
    values = environment_risks(w, moments)
    gradients = np.stack([2.0 * (moment.second @ w - moment.xy) for moment in moments])
    mean_value = float(values.mean())
    centered = values - mean_value
    value = float(np.mean(centered * centered))
    gradient = 2.0 * np.mean(centered[:, None] * gradients, axis=0)
    hessian = 2.0 * np.mean(
        np.einsum("ni,nj->nij", gradients, gradients)
        + 2.0 * centered[:, None, None] * np.stack([moment.second for moment in moments]),
        axis=0,
    )
    return value, gradient, (hessian + hessian.T) / 2.0


def irmv1_value_gradient_hessian(weights: Array, benchmark: ThreeCBenchmark) -> tuple[float, Array, Array]:
    w = np.asarray(weights, dtype=float)
    moments = _env_moments(benchmark)
    derivatives = np.asarray([2.0 * (w @ moment.second @ w - w @ moment.xy) for moment in moments])
    derivative_gradients = np.stack([4.0 * moment.second @ w - 2.0 * moment.xy for moment in moments])
    value = float(np.mean(derivatives * derivatives))
    gradient = 2.0 * np.mean(derivatives[:, None] * derivative_gradients, axis=0)
    hessian = 2.0 * np.mean(
        np.einsum("ni,nj->nij", derivative_gradients, derivative_gradients)
        + derivatives[:, None, None] * np.stack([4.0 * moment.second for moment in moments]),
        axis=0,
    )
    return value, gradient, (hessian + hessian.T) / 2.0


def coral_fixed_value_gradient_hessian(weights: Array, benchmark: ThreeCBenchmark) -> tuple[float, Array, Array]:
    del benchmark
    w = np.asarray(weights, dtype=float)
    return 0.0, np.zeros_like(w), np.zeros((w.size, w.size))


def evaluate_regularizer(method: str, weights: Array, benchmark: ThreeCBenchmark) -> tuple[float, Array, Array]:
    key = method.lower()
    if key == "l2":
        return l2_value_gradient_hessian(weights)
    if key == "vrex":
        return vrex_value_gradient_hessian(weights, benchmark)
    if key == "irmv1":
        return irmv1_value_gradient_hessian(weights, benchmark)
    if key == "coral":
        return coral_fixed_value_gradient_hessian(weights, benchmark)
    raise ValueError(f"unknown 3C method: {method}")


def regularizer_state(method: str, benchmark: ThreeCBenchmark) -> RegularizerState:
    value, gradient, hessian = evaluate_regularizer(method, benchmark.optimum, benchmark)
    metadata = {
        "l2": ("parameter/predictor norm", "predictor-intrinsic", "not-applicable", "global parameter geometry"),
        "vrex": ("environment risk variance", "predictor-intrinsic", "not-applicable", "source-environment risk dispersion"),
        "irmv1": ("environment-wise radial optimality proxy", "predictor-intrinsic", "not-applicable", "scalar classifier-scale response"),
        "coral": ("representation covariance differences", "representation-level fixed-gauge conditional", "gauge-dependent/induced-degenerate", "fixed representation gives zero predictor action"),
    }
    key = method.lower()
    object_name, predictor_status, gauge_status, notes = metadata[key]
    return RegularizerState(method.upper(), object_name, predictor_status, gauge_status,
                            value, gradient, hessian, notes)
