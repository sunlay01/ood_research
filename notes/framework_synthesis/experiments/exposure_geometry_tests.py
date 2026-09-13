"""Minimal falsification tests for the source-exposure geometry probe."""

from __future__ import annotations

import numpy as np


def covariance(deltas: np.ndarray) -> np.ndarray:
    return deltas.T @ deltas / deltas.shape[0]


def projection_range(deltas: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    u, s, _ = np.linalg.svd(deltas.T, full_matrices=True)
    rank = int(np.sum(s > tol))
    return u[:, :rank] @ u[:, :rank].T


def pseudoinverse_quadratic(deltas: np.ndarray, ell: np.ndarray) -> float:
    d = deltas.T
    r = d.T @ ell
    k = d.T @ d
    return float(r @ np.linalg.pinv(k) @ r)


def support_bound(c_s: np.ndarray, ell: np.ndarray, rho: float, kappa: float) -> tuple[float, float, float]:
    eigvals, eigvecs = np.linalg.eigh(c_s)
    positive = eigvals > 1e-10
    c_half_norm = float(np.sqrt(np.sum(eigvals[positive] * (eigvecs[:, positive].T @ ell) ** 2)))
    p_kernel = eigvecs[:, ~positive] @ eigvecs[:, ~positive].T if np.any(~positive) else np.zeros_like(c_s)
    kernel_norm = float(np.linalg.norm(p_kernel @ ell))
    return rho * c_half_norm + kappa * kernel_norm, c_half_norm, kernel_norm


def check_close(name: str, actual: float, expected: float, atol: float = 1e-9) -> None:
    if not np.isclose(actual, expected, atol=atol, rtol=1e-9):
        raise AssertionError(f"{name}: {actual} != {expected}")
    print(f"PASS {name}: {actual:.12g}")


def amplitude_test() -> None:
    base = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    ell = np.array([1.0, 2.0]) / np.sqrt(5.0)
    c0 = covariance(base)
    variance0 = float(ell @ c0 @ ell)
    q0 = pseudoinverse_quadratic(base, ell)
    for eps in (0.1, 2.5):
        scaled = eps * base
        c = covariance(scaled)
        variance = float(ell @ c @ ell)
        q = pseudoinverse_quadratic(scaled, ell)
        check_close(f"amplitude variance eps={eps}", variance, eps**2 * variance0)
        check_close(f"amplitude pseudoinverse eps={eps}", q, q0)
    print(f"INFO amplitude baseline variance={variance0:.12g}, projection norm squared={q0:.12g}")


def rank_test() -> None:
    ell = np.array([1.0, 1.0]) / np.sqrt(2.0)
    rank1 = np.array([[1.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [-1.0, 0.0]])
    rank2 = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    c1, c2 = covariance(rank1), covariance(rank2)
    v1, v2 = float(ell @ c1 @ ell), float(ell @ c2 @ ell)
    check_close("rank test matched V-REx penalty", v1, v2)
    if np.linalg.matrix_rank(c1) != 1 or np.linalg.matrix_rank(c2) != 2:
        raise AssertionError("rank construction did not produce ranks 1 and 2")
    print("PASS rank test ranks: rank(C1)=1, rank(C2)=2")
    b1, exp1, ker1 = support_bound(c1, ell, rho=1.0, kappa=0.25)
    b2, exp2, ker2 = support_bound(c2, ell, rho=1.0, kappa=0.25)
    check_close("rank1 exposed sensitivity", exp1, 1.0 / np.sqrt(2.0))
    check_close("rank1 unexposed sensitivity", ker1, 1.0 / np.sqrt(2.0))
    check_close("rank2 exposed sensitivity", exp2, 1.0 / np.sqrt(2.0))
    check_close("rank2 unexposed sensitivity", ker2, 0.0)
    if not b1 > b2:
        raise AssertionError("nullspace-aware support did not separate rank-1 and rank-2 systems")
    print(f"PASS rank-aware support: rank1={b1:.12g}, rank2={b2:.12g}")


def projection_identity_test() -> None:
    deltas = np.array([[1.0, 2.0, 0.0], [-1.0, -2.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
    ell = np.array([2.0, -1.0, 3.0])
    q = pseudoinverse_quadratic(deltas, ell)
    p = projection_range(deltas)
    check_close("pseudoinverse projection identity", q, float(np.linalg.norm(p @ ell) ** 2))


if __name__ == "__main__":
    projection_identity_test()
    amplitude_test()
    rank_test()
    print("ALL TESTS PASSED")
