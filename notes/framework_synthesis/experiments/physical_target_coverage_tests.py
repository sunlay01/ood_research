"""Deterministic Stage-9 tests for physical target coverage calibration."""

from __future__ import annotations

import numpy as np


TOL = 1e-9


def psd_sqrt(a: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh(a)
    return (vectors * np.sqrt(np.clip(values, 0.0, None))) @ vectors.T


def psd_pinv_sqrt(a: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh(a)
    inv = np.zeros_like(values)
    positive = values > TOL
    inv[positive] = 1.0 / np.sqrt(values[positive])
    return (vectors * inv) @ vectors.T


def projector_range(a: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh(a)
    return (vectors[:, values > TOL]) @ vectors[:, values > TOL].T


def projector_kernel(a: np.ndarray) -> np.ndarray:
    return np.eye(a.shape[0]) - projector_range(a)


def support_product(a: np.ndarray, g: np.ndarray, rho: float, kappa: float) -> float:
    exposed = float(rho * np.linalg.norm(psd_sqrt(a) @ g))
    hidden = float(kappa * np.linalg.norm(projector_kernel(a) @ g))
    return exposed + hidden


def physical_radii(a: np.ndarray, points: np.ndarray) -> tuple[float, float]:
    """Radii for a finite compact family whose rows are physical shifts."""
    pinv_sqrt = psd_pinv_sqrt(a)
    p_range = projector_range(a)
    p_kernel = projector_kernel(a)
    rho = max(float(np.linalg.norm(pinv_sqrt @ p_range @ delta)) for delta in points)
    kappa = max(float(np.linalg.norm(p_kernel @ delta)) for delta in points)
    return rho, kappa


def check_close(name: str, actual: float, expected: float, atol: float = TOL) -> None:
    if not np.isclose(actual, expected, atol=atol, rtol=1e-9):
        raise AssertionError(f"{name}: {actual} != {expected}")
    print(f"PASS {name}: {actual:.12g}")


def product_support_test() -> None:
    a = np.diag([4.0, 0.0])
    g = np.array([3.0, -2.0])
    rho, kappa = 1.5, 0.75
    p_range = projector_range(a)
    p_kernel = projector_kernel(a)
    range_norm = np.linalg.norm(psd_sqrt(a) @ g)
    kernel_norm = np.linalg.norm(p_kernel @ g)
    delta_range = rho * (a @ g) / range_norm
    delta_kernel = kappa * (p_kernel @ g) / kernel_norm
    delta = delta_range + delta_kernel
    check_close("product support identity", float(g @ delta), support_product(a, g, rho, kappa))
    check_close("range constraint at maximizer", np.linalg.norm(psd_pinv_sqrt(a) @ p_range @ delta), rho)
    check_close("kernel constraint at maximizer", np.linalg.norm(p_kernel @ delta), kappa)


def scale_calibration_test() -> None:
    a = np.diag([4.0, 1.0, 0.0])
    # Fixed physical ellipsoid represented by its boundary axes and origin.
    points = np.array([[2.0, 0.0, 0.0], [-2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 3.0]])
    rho, kappa = physical_radii(a, points)
    exposed = np.linalg.norm(psd_sqrt(a) @ np.array([1.0, 1.0, 1.0]))
    for eps in (0.2, 3.0):
        ae = (eps**2) * a
        rho_e, kappa_e = physical_radii(ae, points)
        check_close(f"scale rho eps={eps}", rho_e, rho / abs(eps))
        check_close(f"scale kappa eps={eps}", kappa_e, kappa)
        exposed_e = np.linalg.norm(psd_sqrt(ae) @ np.array([1.0, 1.0, 1.0]))
        check_close(f"fixed-family exposed bound eps={eps}", rho_e * exposed_e, rho * exposed)


def subspace_ball_formula_test() -> None:
    a = np.diag([4.0, 1.0, 0.0])
    # V is span((1,0,1)/sqrt(2), (0,1,0)); radius R=2.
    basis = np.array([[1.0 / np.sqrt(2.0), 0.0], [0.0, 1.0], [1.0 / np.sqrt(2.0), 0.0]])
    p_v = basis @ basis.T
    r = 2.0
    # Restrict the ambient operators to V_T coordinates before taking norms.
    b_rho = psd_pinv_sqrt(a) @ projector_range(a) @ basis
    b_kappa = projector_kernel(a) @ basis
    formula_rho = r * np.linalg.svd(b_rho, compute_uv=False)[0]
    formula_kappa = r * np.linalg.svd(b_kappa, compute_uv=False)[0]
    # The operator norms are attained by the top right singular vectors.
    _, _, vh = np.linalg.svd(b_rho, full_matrices=False)
    delta_rho = r * basis @ vh[0]
    _, _, vh_k = np.linalg.svd(b_kappa, full_matrices=False)
    delta_kappa = r * basis @ vh_k[0]
    actual_rho = np.linalg.norm(psd_pinv_sqrt(a) @ projector_range(a) @ delta_rho)
    actual_kappa = np.linalg.norm(projector_kernel(a) @ delta_kappa)
    check_close("subspace-ball rho formula", actual_rho, formula_rho)
    check_close("subspace-ball kappa formula", actual_kappa, formula_kappa)


def ellipsoid_formula_test() -> None:
    a = np.diag([4.0, 1.0, 0.0])
    q = np.diag([9.0, 1.0, 4.0])
    q_half = psd_sqrt(q)
    formula_rho = np.linalg.svd(psd_pinv_sqrt(a) @ projector_range(a) @ q_half, compute_uv=False)[0]
    formula_kappa = np.linalg.svd(projector_kernel(a) @ q_half, compute_uv=False)[0]
    check_close("ellipsoid rho formula", formula_rho, 1.5)
    check_close("ellipsoid kappa formula", formula_kappa, 2.0)


def orientation_test() -> None:
    q = np.diag([4.0, 1.0])
    q_half = psd_sqrt(q)
    a1 = np.diag([1.0, 0.0])
    a2 = np.diag([0.0, 1.0])
    rho1, kappa1 = np.linalg.svd(psd_pinv_sqrt(a1) @ projector_range(a1) @ q_half, compute_uv=False)[0], np.linalg.svd(projector_kernel(a1) @ q_half, compute_uv=False)[0]
    rho2, kappa2 = np.linalg.svd(psd_pinv_sqrt(a2) @ projector_range(a2) @ q_half, compute_uv=False)[0], np.linalg.svd(projector_kernel(a2) @ q_half, compute_uv=False)[0]
    check_close("orientation rho(A1)", rho1, 2.0)
    check_close("orientation rho(A2)", rho2, 1.0)
    check_close("orientation kappa(A1)", kappa1, 1.0)
    check_close("orientation kappa(A2)", kappa2, 2.0)
    if np.isclose(rho1, rho2) or np.isclose(kappa1, kappa2):
        raise AssertionError("orientation did not change coverage")
    print("PASS orientation: A1 and A2 have identical spectrum but different coverage")


def loose_outer_bound_test() -> None:
    a = np.diag([1.0, 0.0])
    delta = np.array([1.0, 1.0])
    points = np.array([delta, -delta])
    g = np.array([1.0, -1.0])
    rho, kappa = physical_radii(a, points)
    physical_support = max(float(g @ point) for point in points)
    outer = support_product(a, g, rho, kappa)
    check_close("loose outer physical support", physical_support, 0.0)
    check_close("loose outer bound", outer, 2.0)
    if not outer > physical_support:
        raise AssertionError("outer approximation was not strictly loose")
    print("PASS loose outer ratio: infinite because physical support is zero")


def ridge_cutoff_test() -> None:
    a = np.diag([1.0, 0.0])
    points = np.array([[0.0, 1.0], [0.0, -1.0]])
    rho, kappa = physical_radii(a, points)
    check_close("ideal hidden radius", rho, 0.0)
    check_close("ideal kernel radius", kappa, 1.0)
    lam = 0.01
    ridge = a + lam * np.eye(2)
    rho_ridge, kappa_ridge = physical_radii(ridge, points)
    check_close("ridge kernel radius", kappa_ridge, 0.0)
    check_close("ridge price for hidden direction", rho_ridge, 10.0)
    cutoff = np.diag([1.0, 0.0])
    rho_cut, kappa_cut = physical_radii(cutoff, points)
    check_close("cutoff preserves hidden radius", kappa_cut, 1.0)
    check_close("cutoff exposed radius", rho_cut, 0.0)


def counterexample_summary() -> None:
    # F1: U=R^2 and A=I gives rho=+infinity.
    print("INFO F1 unbounded family: U=R^2, A=I => rho_A=+infinity")
    # F2/F5 are represented by the rank-deficient and high-trace examples above.
    print("INFO F2 nullspace-dominated case: A=diag(1,0), target shift e2, g=e2")
    print("INFO F5 diversity-without-coverage: A=diag(100,100,0), target shifts along e3")


if __name__ == "__main__":
    product_support_test()
    scale_calibration_test()
    subspace_ball_formula_test()
    ellipsoid_formula_test()
    orientation_test()
    loose_outer_bound_test()
    ridge_cutoff_test()
    counterexample_summary()
    print("ALL STAGE-9 TESTS PASSED")
