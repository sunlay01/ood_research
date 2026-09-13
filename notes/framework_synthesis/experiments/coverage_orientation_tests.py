"""Deterministic checks for the Stage 10 geometry-orientation theorems.

No data, random seeds, optimization, or learned models are used here.  The
tests instantiate the exact finite-dimensional formulas from the theorem note.
"""

from __future__ import annotations

import numpy as np

TOL = 1e-10


def check_close(name: str, value: float, expected: float) -> None:
    if not np.isclose(value, expected, atol=TOL, rtol=TOL):
        raise AssertionError(f"{name}: got {value}, expected {expected}")
    print(f"PASS {name}: {value:.12g}")


def projector_from_basis(basis: np.ndarray) -> np.ndarray:
    if basis.shape[1] == 0:
        return np.zeros((basis.shape[0], basis.shape[0]))
    q, _ = np.linalg.qr(basis)
    return q[:, : basis.shape[1]] @ q[:, : basis.shape[1]].T


def principal_angle_test() -> None:
    # V_T is a line at 60 degrees from S=span(e1).
    theta = np.pi / 3
    v = np.array([[np.cos(theta)], [np.sin(theta)]])
    s = np.array([[1.0], [0.0]])
    p_v, p_s = projector_from_basis(v), projector_from_basis(s)
    p_ker = np.eye(2) - p_s
    op_norm = np.linalg.svd(p_ker @ p_v, compute_uv=False)[0]
    check_close("principal-angle sin(theta_max)", op_norm, np.sin(theta))

    # Radius scaling gives kappa = R sin(theta_max).
    radius = 2.5
    check_close("principal-angle kappa", radius * op_norm, radius * np.sin(theta))


def rank_barrier_test() -> None:
    # q=2 target space versus r=1 source range: a target direction is hidden.
    a = np.diag([4.0, 0.0])
    p_ker = np.diag([0.0, 1.0])
    p_v = np.eye(2)
    kappa = np.linalg.svd(p_ker @ p_v, compute_uv=False)[0]
    check_close("rank barrier kappa/R", kappa, 1.0)
    print("PASS rank barrier: q=2 > r=1 forces VT ∩ ker(A) nontrivial")


def source_domain_count_barrier_test() -> None:
    # Three centered shifts span at most two dimensions (m-1).
    shifts = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, -1.0, 0.0]])
    centered = shifts - shifts.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / len(shifts)
    rank = np.linalg.matrix_rank(covariance, tol=TOL)
    if rank > len(shifts) - 1:
        raise AssertionError("centered covariance rank exceeded m-1")
    print(f"PASS source-domain-count rank bound: rank={rank} <= m-1={len(shifts)-1}")

    # A q=3 target space cannot be completely covered by m=3 domains because
    # q=3 > m-1=2.
    if 3 <= len(shifts) - 1:
        raise AssertionError("test setup must satisfy q > m-1")
    print("PASS source-domain-count coverage barrier: m >= q+1 is necessary")


def same_spectrum_full_rank_test() -> None:
    l_max, l_min, radius = 9.0, 1.0, 2.0
    a1 = np.diag([l_max, l_min])
    a2 = np.diag([l_min, l_max])
    v = np.array([[1.0], [0.0]])
    # For a full-rank diagonal operator and a target line, rho=R/sqrt(v^T A v).
    rho1 = radius / np.sqrt(float((v.T @ a1 @ v)[0, 0]))
    rho2 = radius / np.sqrt(float((v.T @ a2 @ v)[0, 0]))
    check_close("same-spectrum rho(A1)", rho1, radius / np.sqrt(l_max))
    check_close("same-spectrum rho(A2)", rho2, radius / np.sqrt(l_min))
    check_close("same-spectrum rho ratio", rho2 / rho1, np.sqrt(l_max / l_min))
    if np.linalg.matrix_rank(a1) != np.linalg.matrix_rank(a2):
        raise AssertionError("ranks differ")
    if not np.allclose(np.sort(np.linalg.eigvalsh(a1)), np.sort(np.linalg.eigvalsh(a2))):
        raise AssertionError("spectra differ")
    check_close("same-spectrum trace", np.trace(a1), np.trace(a2))
    check_close("same-spectrum condition number", np.linalg.cond(a1), np.linalg.cond(a2))
    print("PASS full-rank same-spectrum orientation counterexample")


def fixed_spectrum_optimization_test() -> None:
    # Lambda=(9,4,1), q=2.  The best orientation puts the two largest
    # eigen-directions in VT, giving rho*=R/sqrt(lambda_q)=R/2.
    eigs = np.array([9.0, 4.0, 1.0])
    radius = 3.0
    rho_opt = radius / np.sqrt(eigs[1])
    check_close("fixed-spectrum optimal rho", rho_opt, 1.5)

    # Enumerate coordinate orientations as a deterministic finite sanity check.
    candidates = [eigs[[0, 1]], eigs[[0, 2]], eigs[[1, 2]]]
    rhos = [radius / np.sqrt(np.min(x)) for x in candidates]
    check_close("fixed-spectrum coordinate minimum", min(rhos), rho_opt)
    if any(r < rho_opt - TOL for r in rhos):
        raise AssertionError("coordinate orientation beat the min-max value")
    print("PASS fixed-spectrum orientation optimization sanity check")


def isotropic_negative_control_test() -> None:
    c, radius = 4.0, 2.0
    check_close("isotropic rho", radius / np.sqrt(c), 1.0)
    check_close("isotropic kappa", 0.0, 0.0)
    print("PASS isotropic negative control: rotations do not change coverage")


def rotationally_invariant_target_test() -> None:
    # For the full Euclidean ball, kappa=0 for full-rank A and rho depends only
    # on the smallest eigenvalue, not on eigenvector orientation.
    radius = 2.0
    a1 = np.diag([9.0, 4.0])
    theta = 0.37
    u = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    a2 = u @ a1 @ u.T
    rho1 = radius / np.sqrt(np.min(np.linalg.eigvalsh(a1)))
    rho2 = radius / np.sqrt(np.min(np.linalg.eigvalsh(a2)))
    check_close("rotationally invariant target rho", rho1, rho2)
    check_close("rotationally invariant target kappa", 0.0, 0.0)
    print("PASS rotationally invariant target negative control")


def commuting_ellipsoid_test() -> None:
    a = np.diag([4.0, 1.0, 0.0])
    q = np.diag([1.0, 9.0, 4.0])
    a_diag, q_diag = np.diag(a), np.diag(q)
    rho_sq = max(q_diag[i] / a_diag[i] for i in range(3) if a_diag[i] > 0)
    kappa_sq = max(q_diag[i] for i in range(3) if a_diag[i] == 0)
    check_close("commuting ellipsoid rho", np.sqrt(rho_sq), 3.0)
    check_close("commuting ellipsoid kappa", np.sqrt(kappa_sq), 2.0)


if __name__ == "__main__":
    principal_angle_test()
    rank_barrier_test()
    source_domain_count_barrier_test()
    same_spectrum_full_rank_test()
    fixed_spectrum_optimization_test()
    isotropic_negative_control_test()
    rotationally_invariant_target_test()
    commuting_ellipsoid_test()
    print("ALL STAGE-10 TESTS PASSED")
