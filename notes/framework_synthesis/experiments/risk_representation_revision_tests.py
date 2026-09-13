"""Deterministic Stage 11R metric/gauge interface audit."""

from __future__ import annotations

import numpy as np

TOL = 1e-10


def check_close(name: str, actual, expected) -> None:
    if not np.allclose(actual, expected, atol=TOL, rtol=TOL):
        raise AssertionError(f"{name}: {actual} != {expected}")
    print(f"PASS {name}")


def source_operator(shifts: np.ndarray) -> np.ndarray:
    return shifts.T @ shifts / len(shifts)


def metric_kernel_projection(g: np.ndarray, D: np.ndarray, G: np.ndarray) -> np.ndarray:
    """Projection onto ker(D) in the dual metric G^{-1}."""
    _, _, vh = np.linalg.svd(D)
    rank = np.linalg.matrix_rank(D)
    Z = vh[rank:].T
    if Z.shape[1] == 0:
        return np.zeros_like(g)
    M = np.linalg.inv(G)
    return Z @ np.linalg.solve(Z.T @ M @ Z, Z.T @ M @ g)


def pairing_and_operator_test() -> None:
    shifts = np.array([[1.0, 2.0], [-2.0, 0.5], [0.5, -1.0]])
    g = np.array([0.7, -1.3])
    A = source_operator(shifts)
    check_close("exposure energy identity", g @ A @ g, np.mean((shifts @ g) ** 2))
    print("PASS A: V* -> V finite-rank exposure operator semantics")


def covariance_and_support_invariance_test() -> None:
    shifts = np.array([[1.0, 2.0], [-2.0, 0.5], [0.5, -1.0]])
    g = np.array([0.7, -1.3])
    T = np.array([[2.0, 0.3], [0.4, 1.5]])
    A = source_operator(shifts)
    shifts_p = shifts @ T.T
    gp = np.linalg.inv(T).T @ g
    Ap = source_operator(shifts_p)
    check_close("A' = T A T^T", Ap, T @ A @ T.T)
    check_close("pairing invariance", gp @ (T @ shifts[0]), g @ shifts[0])
    check_close("S_A invariance", gp @ Ap @ gp, g @ A @ g)

    physical = np.array([[1.0, 1.0], [-1.0, 0.5], [0.2, -1.2]])
    physical_p = physical @ T.T
    support = np.max(physical @ g)
    support_p = np.max(physical_p @ gp)
    check_close("exact support invariance", support_p, support)


def annihilator_quotient_test() -> None:
    shifts = np.array([[1.0, 0.0], [-1.0, 0.0]])
    D = shifts
    g = np.array([0.4, 1.0])
    h = np.array([0.0, 2.0])  # h in S°
    check_close("annihilator condition", D @ h, np.zeros(2))
    check_close("source restriction equivalence", D @ (g + h), D @ g)
    print("PASS [g] in V*/S° is the source-observable quotient")


def metric_gauge_test() -> None:
    shifts = np.array([[1.0, 0.0], [-1.0, 0.0]])
    D = shifts
    g = np.array([0.8, 1.7])
    G = np.array([[2.0, 0.4], [0.4, 1.5]])
    blind = metric_kernel_projection(g, D, G)
    blind_norm = np.sqrt(blind @ np.linalg.inv(G) @ blind)

    T = np.array([[1.0, 0.7], [0.2, 1.4]])
    gp = np.linalg.inv(T).T @ g
    Dp = D @ T.T
    Gp = np.linalg.inv(T).T @ G @ np.linalg.inv(T)
    blind_p = metric_kernel_projection(gp, Dp, Gp)
    blind_norm_p = np.sqrt(blind_p @ np.linalg.inv(Gp) @ blind_p)
    check_close("metric-aware blind norm invariance", blind_norm_p, blind_norm)

    # Raw Euclidean projection is not invariant under the same shear.
    raw = metric_kernel_projection(g, D, np.eye(2))
    raw_p = metric_kernel_projection(gp, Dp, np.eye(2))
    if np.isclose(np.linalg.norm(raw_p), np.linalg.norm(raw), atol=TOL):
        raise AssertionError("raw Euclidean blind norm should change under shear")
    print("PASS raw Euclidean N is metric-dependent")


def exact_support_vs_split_test() -> None:
    physical = np.array([[1.0, 1.0], [-1.0, -1.0]])
    g = np.array([1.0, -1.0])
    check_close("coupled support", np.max(physical @ g), 0.0)
    check_close("split relaxation", abs(g[0]) + abs(g[1]), 2.0)


if __name__ == "__main__":
    pairing_and_operator_test()
    covariance_and_support_invariance_test()
    annihilator_quotient_test()
    metric_gauge_test()
    exact_support_vs_split_test()
    print("ALL STAGE-11R TESTS PASSED")
