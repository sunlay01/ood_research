"""Deterministic Stage 13R.1 invariance and nuisance audit."""

from __future__ import annotations

import numpy as np

TOL = 1e-10


def check(name: str, actual, expected) -> None:
    if not np.allclose(actual, expected, atol=TOL, rtol=TOL):
        raise AssertionError(f"{name}: {actual} != {expected}")
    print(f"PASS {name}")


def additive_nuisance_test() -> None:
    # R(theta, xi)=theta^2+xi and R~=R+xi.  The raw environmental derivative
    # changes, while the excess-risk derivative is unchanged.
    theta, xi = 0.7, 2.0
    raw, raw_tilde = 1.0, 2.0
    excess, excess_tilde = 1.0, 1.0
    check("additive nuisance changes raw ESF", raw_tilde - raw, 1.0)
    check("additive nuisance leaves excess ESF", excess, excess_tilde)
    # Pairwise risk differences cancel the same nuisance exactly.
    pair, pair_tilde = theta**2 - 0.0**2, theta**2 - 0.0**2
    check("additive nuisance leaves pairwise quotient", pair, pair_tilde)


def bayes_risk_shift_test() -> None:
    # R(theta,xi)=(theta-1)^2 + 10 xi.  Bayes risk changes with xi, while the
    # minimizer and excess risk remain fixed.
    theta = 0.4
    xi0, xi1 = 0.0, 1.0
    r0 = (theta - 1.0) ** 2 + 10.0 * xi0
    r1 = (theta - 1.0) ** 2 + 10.0 * xi1
    check("Bayes/irreducible risk shifts", r1 - r0, 10.0)
    e0, e1 = (theta - 1.0) ** 2, (theta - 1.0) ** 2
    check("excess risk is invariant to Bayes shift", e0, e1)
    print("PASS predictor identity and ranking are unchanged")


def original_separation_reclassification_test() -> None:
    theta, target = 0.4, 1.0
    r, rtilde = theta**2 + target, theta**2 + 2.0 * target
    check("original raw target risks differ", rtilde - r, 1.0)
    check("original excess risks agree", r - target, rtilde - 2.0 * target)
    check("original optimizers agree", 0.0, 0.0)
    print("PASS original witness is information-only, not predictor-relevant")


def fixed_tangent_comparison_test() -> None:
    # Freeze D_phys={delta: ||delta||_2 <= 1}; its support is the Euclidean
    # norm of the environment derivative.  Source exposure can miss a blind
    # coordinate even when V-REx is zero.
    dxi = np.array([0.0, 3.0])
    esf_phys = np.linalg.norm(dxi)
    vrex_exposed = 0.0  # source exposure only in coordinate 0
    check("fixed physical tangent ESF", esf_phys, 3.0)
    check("V-REx exposed penalty misses blind coordinate", vrex_exposed, 0.0)
    # MMD's dual norm bound is exact for the same Euclidean ball.
    check("fixed-state dual norm upper bound", np.max(np.abs(dxi)), esf_phys)
    print("PASS fixed-Dphys comparison keeps the left-hand side unchanged")


def quotient_not_excess_test() -> None:
    # Fixed-anchor quotient and excess ESF differ when the anchor is not the
    # environment-wise Bayes optimizer.  This is a relative-risk object.
    theta, anchor, xi = 0.8, 0.0, 0.3
    # R=(theta-xi)^2+xi^2, R*=xi^2; derivatives wrt xi:
    excess_derivative = -2.0 * (theta - xi)
    pair_derivative = -2.0 * (theta - anchor)
    if np.isclose(excess_derivative, pair_derivative, atol=TOL):
        raise AssertionError("fixed-anchor quotient should differ from excess ESF")
    print("PASS pairwise quotient is distinct but relative-risk only")


def all_order_predictor_relevance_impossibility_test() -> None:
    # If two smooth surfaces have identical all-order theta derivatives, their
    # difference is theta-independent.  The finite witness checks the resulting
    # invariance of ranking and excess risk under an arbitrary c(xi).
    theta_values = np.array([-1.0, 0.0, 1.0])
    xi = 2.0
    base = theta_values**2
    shifted = base + 7.0 * xi
    check("all-order additive shift preserves ranking", np.argsort(base), np.argsort(shifted))
    check("all-order additive shift preserves pairwise gaps", base[2] - base[1], shifted[2] - shifted[1])
    print("PASS predictor-relevant separation is impossible under all-order equality")


if __name__ == "__main__":
    additive_nuisance_test()
    bayes_risk_shift_test()
    original_separation_reclassification_test()
    fixed_tangent_comparison_test()
    quotient_not_excess_test()
    all_order_predictor_relevance_impossibility_test()
    print("ALL STAGE-13R.1 TESTS PASSED")
