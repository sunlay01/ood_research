"""Source-whitened regularizer action and OOD response coverage."""

from __future__ import annotations

import numpy as np

from .round3r_3b_geometry import matrix_rank, principal_angles, subspace_basis
from .round3r_3c_benchmark import ThreeCBenchmark, target_shift_curvature
from .round3r_3c_regularizers import RegularizerState

Array = np.ndarray


def symmetric_inverse_sqrt(matrix: Array) -> Array:
    value = (np.asarray(matrix) + np.asarray(matrix).T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(value)
    if eigenvalues.size == 0 or eigenvalues.min() <= 0:
        raise ValueError("matrix must be positive definite")
    return eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues)) @ eigenvectors.T


def whitened_action(hessian: Array, state: RegularizerState) -> tuple[Array, Array, Array]:
    inverse_root = symmetric_inverse_sqrt(hessian)
    b = inverse_root @ state.gradient
    k = inverse_root @ state.hessian @ inverse_root
    return b, (k + k.T) / 2.0, inverse_root


def local_delta(hessian: Array, state: RegularizerState, lam: float) -> tuple[Array, bool, float]:
    metric = (hessian + lam * state.hessian + (hessian + lam * state.hessian).T) / 2.0
    minimum = float(np.linalg.eigvalsh(metric).min())
    if minimum <= 0:
        return np.full(hessian.shape[0], np.nan), False, minimum
    return -lam * np.linalg.solve(metric, state.gradient), True, minimum


def steering_score(q: Array, b: Array, k: Array, lam: float) -> float:
    metric = np.eye(q.size) + lam * k
    return float(-lam * q @ np.linalg.solve(metric, b))


def curvature_ratio(q: Array, k: Array, lam: float) -> float:
    denominator = float(q @ q)
    if denominator <= 1e-30:
        return float("nan")
    return float(q @ np.linalg.solve(np.eye(q.size) + lam * k, q) / denominator)


def infinitesimal_score(q: Array, k: Array) -> float:
    denominator = float(q @ q)
    return float(q @ k @ q / denominator) if denominator > 1e-30 else float("nan")


def blind_fraction(q: Array, k: Array, tolerance: float = 1e-9) -> float | None:
    eigenvalues, eigenvectors = np.linalg.eigh((k + k.T) / 2.0)
    if eigenvalues.size and eigenvalues.min() < -tolerance:
        return None
    denominator = float(q @ q)
    if denominator <= 1e-30:
        return float("nan")
    kernel = eigenvectors[:, np.abs(eigenvalues) <= tolerance * max(1.0, np.max(np.abs(eigenvalues), initial=0.0))]
    projection = kernel @ (kernel.T @ q) if kernel.shape[1] else np.zeros_like(q)
    return float(projection @ projection / denominator)


def response_space_coverage(q_responses: Array, state: RegularizerState,
                            hessian: Array, tolerance: float = 1e-9) -> dict[str, object]:
    b, k, inverse_root = whitened_action(hessian, state)
    relevant = np.asarray(q_responses, dtype=float)
    basis = subspace_basis(relevant)
    image = k @ basis
    k_rank = matrix_rank(image) if image.shape[1] else 0
    kernel_dimension = int(basis.shape[1] - k_rank)
    eig = np.linalg.eigvalsh(k)
    psd = bool(eig.min() >= -tolerance)
    range_basis = subspace_basis(k)
    angles = principal_angles(basis, range_basis) if range_basis.shape[1] else np.zeros(0)
    projector = basis @ basis.T if basis.shape[1] else np.zeros_like(k)
    denominator = float(np.linalg.norm(k, ord="fro") ** 2)
    outside = np.linalg.norm((np.eye(k.shape[0]) - projector) @ k, ord="fro") ** 2
    return {
        "response_rank": int(basis.shape[1]),
        "restricted_rank": int(k_rank),
        "relevant_kernel_dimension": kernel_dimension,
        "principal_angles": angles,
        "psd": psd,
        "min_eigenvalue": float(eig.min()),
        "max_eigenvalue": float(eig.max()),
        "nonselective_outside_fraction": float(outside / denominator) if denominator > 1e-30 else 0.0,
        "b": b,
        "k": k,
        "inverse_hessian_root": inverse_root,
    }


def response_metrics(benchmark: ThreeCBenchmark, state: RegularizerState,
                     lam: float, epsilon: float = 1e-4) -> list[dict[str, object]]:
    b, k, _ = whitened_action(benchmark.hessian, state)
    delta, valid, min_eigenvalue = local_delta(benchmark.hessian, state, lam)
    rows = []
    for index in benchmark.relevant_indices:
        q = benchmark.q_responses[:, index]
        probe = benchmark.probes[int(index)]
        ratio = curvature_ratio(q, k, lam) if valid else float("nan")
        steering = steering_score(q, b, k, lam) if valid else float("nan")
        blind = blind_fraction(q, k)
        shift_curvature = target_shift_curvature(benchmark, probe)
        shifted_gradient = benchmark.gradients[:, index] + shift_curvature @ delta if valid else np.full_like(q, np.nan)
        shifted_metric = benchmark.hessian + lam * state.hessian
        vulnerability = float(np.sqrt(2.0 * epsilon * max(shifted_gradient @ np.linalg.solve(shifted_metric, shifted_gradient), 0.0))) if valid else float("nan")
        rows.append({
            "method": state.method,
            "shift_index": int(index),
            "shift_id": probe.shift_id,
            "lambda": float(lam),
            "relevance_squared": float(q @ q),
            "control_score_c": infinitesimal_score(q, k),
            "curvature_ratio": ratio,
            "blind_fraction": blind,
            "steering": steering,
            "steering_class": "helpful" if steering < -1e-12 else "harmful" if steering > 1e-12 else "neutral",
            "local_vulnerability": vulnerability,
            "metric_min_eigenvalue": min_eigenvalue,
            "local_valid": valid,
            "exposure_posthoc": probe.mechanism,
            "intervention_family_posthoc": probe.intervention_family,
        })
    return rows
