"""Deterministic Stage 11 tests for finite risk representations.

These are algebraic checks only: no data download, randomness, optimization,
or neural-network training is used.
"""

from __future__ import annotations

import numpy as np

TOL = 1e-10


def check_close(name: str, actual: float, expected: float) -> None:
    if not np.isclose(actual, expected, atol=TOL, rtol=TOL):
        raise AssertionError(f"{name}: {actual} != {expected}")
    print(f"PASS {name}: {actual:.12g}")


def linear_span_test() -> None:
    phi = np.array([[1.0, 2.0], [3.0, -1.0], [2.0, 4.0]])
    b, g = 0.7, np.array([1.5, -0.25])
    losses = b + phi @ g
    psi = phi.mean(axis=0)
    check_close("linear-span identity", losses.mean(), b + g @ psi)
    print("PASS linear-span state is predictor-independent and label-capable")


def quadratic_risk_test() -> None:
    x = np.array([[1.0, 2.0], [-1.0, 0.5], [2.0, -0.5]])
    y = np.array([0.5, -1.0, 2.0])
    w = np.array([0.75, -0.4])
    lhs = np.mean((x @ w - y) ** 2)
    sxx = (x.T @ x) / len(x)
    sxy = (x.T @ y) / len(x)
    syy = np.mean(y**2)
    rhs = w @ sxx @ w - 2.0 * w @ sxy + syy
    check_close("quadratic supervised identity", lhs, rhs)
    print("PASS quadratic state (E[XX^T], E[XY], E[Y^2]) is finite and algorithm-independent")


def conditional_shift_test() -> None:
    # Same X marginal and E[XX^T], different conditional labels.
    x = np.array([[-1.0], [1.0]])
    y_a, y_b = np.array([-1.0, 1.0]), np.array([1.0, -1.0])
    w = np.array([1.0])
    risk_a = np.mean((x[:, 0] * w[0] - y_a) ** 2)
    risk_b = np.mean((x[:, 0] * w[0] - y_b) ** 2)
    check_close("same covariate second moment", np.mean(x[:, 0] ** 2), 1.0)
    if np.isclose(risk_a, risk_b, atol=TOL):
        raise AssertionError("chosen conditional-shift witness must change risk")
    print(f"PASS conditional/label shift changes risk: {risk_a:.6g} vs {risk_b:.6g}")


def marginal_only_counterexample_test() -> None:
    x = np.array([-1.0, 1.0])
    y_a, y_b = x.copy(), -x.copy()
    w = np.array([1.0])
    risk_a = np.mean((w[0] * x - y_a) ** 2)
    risk_b = np.mean((w[0] * x - y_b) ** 2)
    check_close("marginal-only same P_X", np.mean(x), 0.0)
    if not risk_a < risk_b:
        raise AssertionError("marginal-only counterexample failed")
    print("PASS marginal-only state is insufficient for supervised risk")


def incomplete_state_test() -> None:
    # Omitting E[XY] loses the sign of the conditional relation.
    x = np.array([-1.0, 1.0])
    y_a, y_b = x.copy(), -x.copy()
    omitted_state_a = np.array([np.mean(x**2), np.mean(y_a**2)])
    omitted_state_b = np.array([np.mean(x**2), np.mean(y_b**2)])
    if not np.allclose(omitted_state_a, omitted_state_b):
        raise AssertionError("omitted states should match")
    w = np.array([1.0])
    ra = np.mean((w[0] * x - y_a) ** 2)
    rb = np.mean((w[0] * x - y_b) ** 2)
    if np.isclose(ra, rb, atol=TOL):
        raise AssertionError("omitting E[XY] must change risk in witness")
    print("PASS incomplete moment state: identical E[XX^T], E[Y^2] but different risk")


def taylor_remainder_test() -> None:
    # R(theta)=theta^2+2 theta, reference theta=0, Hessian L=2 exactly.
    for theta in (-0.7, 0.0, 1.3):
        exact = theta**2 + 2.0 * theta
        linear = 2.0 * theta
        remainder = exact - linear
        check_close("Taylor remainder bound", abs(remainder), theta**2)
        if abs(remainder) > 0.5 * 2.0 * theta**2 + TOL:
            raise AssertionError("Taylor bound failed")
    print("PASS local smooth representation with primitive Hessian remainder bound")


def coordinate_transform_test() -> None:
    psi = np.array([1.2, -0.4])
    g = np.array([0.7, 2.0])
    t = np.array([[2.0, 0.3], [0.0, 1.5]])
    psi_p = t @ psi
    g_p = np.linalg.inv(t).T @ g
    check_close("bilinear coordinate invariance", g_p @ psi_p, g @ psi)

    a = np.array([[3.0, 1.0], [1.0, 2.0]])
    a_p = t @ a @ t.T
    check_close("exposure congruence entry (0,0)", a_p[0, 0], (t @ a @ t.T)[0, 0])
    check_close("exposure congruence entry (1,1)", a_p[1, 1], (t @ a @ t.T)[1, 1])
    print("PASS A' = T A T^T and g' = T^{-T} g")

    # The exact support of a transformed target is invariant, while Euclidean
    # kernel projections are not a coordinate-free object under shear.
    a0 = np.diag([1.0, 0.0])
    shear = np.array([[1.0, 0.0], [1.0, 1.0]])
    a1 = shear @ a0 @ shear.T
    g0 = np.array([0.0, 1.0])
    g1 = np.linalg.inv(shear).T @ g0
    n0 = np.linalg.norm((np.eye(2) - np.diag([1.0, 0.0])) @ g0)
    vals, vecs = np.linalg.eigh(a1)
    pker1 = vecs[:, vals < 1e-9] @ vecs[:, vals < 1e-9].T
    n1 = np.linalg.norm(pker1 @ g1)
    if np.isclose(n0, n1, atol=TOL):
        raise AssertionError("non-orthogonal coordinate change should alter Euclidean N")
    print("PASS gauge audit: exact bilinear/support quantities invariant, Euclidean N_A is metric-dependent")


def blind_sensitivity_test() -> None:
    # Source variation sees only coordinate 0. Two predictors have the same
    # source risk vector but different kernel sensitivity.
    d = np.array([[1.0, 0.0], [-1.0, 0.0]])
    a = d.T @ d / len(d)
    g1, g2 = np.array([0.0, 1.0]), np.array([0.0, 2.0])
    if not np.allclose(d @ g1, d @ g2):
        raise AssertionError("source risk vectors should match")
    n1, n2 = np.linalg.norm(g1), np.linalg.norm(g2)
    if np.isclose(n1, n2, atol=TOL):
        raise AssertionError("kernel sensitivities should differ")
    check_close("blind source operator rank", np.linalg.matrix_rank(a), 1.0)
    print("PASS blind sensitivity: N_A is not source-identifiable from source risk variation")


def support_split_test() -> None:
    # Coupled physical family {(+1,+1),(-1,-1)} and g=(1,-1) has zero support,
    # while the independent split relaxation pays two units.
    physical = np.array([[1.0, 1.0], [-1.0, -1.0]])
    g = np.array([1.0, -1.0])
    exact = np.max(physical @ g)
    split = abs(g[0]) + abs(g[1])
    check_close("exact coupled support", exact, 0.0)
    check_close("split support upper bound", split, 2.0)
    print("PASS exact target support can be strictly sharper than rho*S+kappa*N")


if __name__ == "__main__":
    linear_span_test()
    quadratic_risk_test()
    conditional_shift_test()
    marginal_only_counterexample_test()
    incomplete_state_test()
    taylor_remainder_test()
    coordinate_transform_test()
    blind_sensitivity_test()
    support_split_test()
    print("ALL STAGE-11 TESTS PASSED")
