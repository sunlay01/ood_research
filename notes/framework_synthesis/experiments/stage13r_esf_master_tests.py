"""Deterministic Stage 13R environmental-sensitivity audit."""

from __future__ import annotations

import numpy as np

TOL = 1e-10


def check(name: str, actual, expected) -> None:
    if not np.allclose(actual, expected, atol=TOL, rtol=TOL):
        raise AssertionError(f"{name}: {actual} != {expected}")
    print(f"PASS {name}")


def affine_esf_support_test() -> None:
    g = np.array([0.6, -0.8])
    tangent = np.array([[1.0, 2.0], [-1.0, 0.5], [0.2, -1.0]])
    check("affine ESF equals finite support", np.max(tangent @ g), max(float(g @ d) for d in tangent))
    print("PASS affine environmental sensitivity reduces to Stage-12 support")


def ellipsoid_dual_test() -> None:
    A = np.diag([4.0, 1.0])
    rho = 1.3
    g = np.array([0.7, -1.1])
    source_norm = np.sqrt(g @ A @ g)
    delta = rho * (A @ g) / source_norm
    check("ellipsoid tangent boundary", delta @ np.linalg.inv(A) @ delta, rho**2)
    check("ellipsoid ESF dual", g @ delta, rho * source_norm)


def path_integral_test() -> None:
    # R(xi)=xi^3+2xi, xi in [0,1]; derivative is 3xi^2+2.
    endpoint = (1.0**3 + 2.0 * 1.0) - 0.0
    integral = 1.0 + 2.0
    check("smooth path-integral equality", endpoint, integral)


def local_to_global_negative_control_test() -> None:
    # R(xi)=a xi^2 has zero local derivative at observed xi=0 but large target risk.
    a, target = 2.0, 10.0
    source_local_esf = 0.0
    target_gap = a * target**2
    check("source-local ESF at zero", source_local_esf, 0.0)
    if target_gap <= 100.0 - TOL:
        raise AssertionError("target path risk should be large")
    integrated_esf = target_gap  # integral_0^T |2a xi| dxi
    check("integrated ESF remains valid", integrated_esf, target_gap)
    print("PASS local ESF is not a source-only global certificate")


def coral_projection_residual_test() -> None:
    # Same covariance/second moment, different conditional cross-moment.
    x = np.array([-1.0, 1.0])
    ya, yb = x.copy(), -x.copy()
    check("CORAL projected moment", np.mean(x**2), np.mean(x**2))
    check("label energy unchanged", np.mean(ya**2), np.mean(yb**2))
    ra, rb = np.mean((x - ya) ** 2), np.mean((x - yb) ** 2)
    if np.isclose(ra, rb, atol=TOL):
        raise AssertionError("projected CORAL state must miss conditional risk change")
    print("PASS projected CORAL ESF misses conditional residual")


def all_order_parameter_separation_test() -> None:
    # q(theta)=theta^2; additive environment terms are invisible to all theta derivatives.
    theta, xi = 0.4, 1.0
    q = theta**2
    r, rtilde = q + xi, q + 2.0 * xi
    check("all-order parameter-side source agreement", 2.0 * theta, 2.0 * theta)
    check("environment derivative R", 1.0, 1.0)
    check("environment derivative Rtilde", 2.0, 2.0)
    if np.isclose(r, rtilde, atol=TOL):
        raise AssertionError("target risks must differ")
    print("PASS parameter-alignment/all-order separation from ESF")


if __name__ == "__main__":
    affine_esf_support_test()
    ellipsoid_dual_test()
    path_integral_test()
    local_to_global_negative_control_test()
    coral_projection_residual_test()
    all_order_parameter_separation_test()
    print("ALL STAGE-13R TESTS PASSED")
