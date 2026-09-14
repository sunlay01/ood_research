"""Deterministic tests for the first master-functional candidate screen."""

from __future__ import annotations

import numpy as np

TOL = 1e-10


def check(name: str, actual, expected) -> None:
    if not np.allclose(actual, expected, atol=TOL, rtol=TOL):
        raise AssertionError(f"{name}: {actual} != {expected}")
    print(f"PASS {name}")


def quotient_sup_equals_pairwise_discrepancy() -> None:
    d = np.array([2.0, -1.0, 0.5, 4.0])
    dmax, dmin = d.max(), d.min()
    quotient = (dmax - dmin) / 2.0
    pairwise = max(abs(float(x - y)) for x in d for y in d)
    check("sup quotient radius", quotient, 2.5)
    check("pairwise discrepancy", pairwise, 5.0)
    check("exact quotient/discrepancy equivalence", quotient, pairwise / 2.0)


def additive_nuisance_quotient_invariance() -> None:
    landscape_delta = np.array([1.0, -2.0, 0.5])
    shifted = landscape_delta + 7.3
    q = (landscape_delta.max() - landscape_delta.min()) / 2.0
    q_shifted = (shifted.max() - shifted.min()) / 2.0
    check("risk-landscape quotient removes additive nuisance", q_shifted, q)


def l2_measure_dependence() -> None:
    delta = np.array([0.0, 2.0, 4.0])
    w1 = np.array([1 / 3, 1 / 3, 1 / 3])
    w2 = np.array([0.8, 0.1, 0.1])
    c1 = np.sum(w1 * delta)
    c2 = np.sum(w2 * delta)
    l2_1 = np.sqrt(np.sum(w1 * (delta - c1) ** 2))
    l2_2 = np.sqrt(np.sum(w2 * (delta - c2) ** 2))
    if np.isclose(l2_1, l2_2, atol=TOL):
        raise AssertionError("L2 quotient must depend on the declared predictor measure")
    print("PASS L2 quotient requires an external predictor measure")


def original_witness_is_nuisance_only() -> None:
    theta = np.array([-1.0, 0.0, 1.0])
    r = theta**2 + 1.0
    r_tilde = theta**2 + 2.0
    check("same predictor ranking", np.argsort(r), np.argsort(r_tilde))
    check("same pairwise landscape differences", r[2] - r[1], r_tilde[2] - r_tilde[1])


def optimizer_diameter_not_risk_certificate() -> None:
    # Same optimizer set, arbitrary additive target difficulty.
    theta = np.array([-1.0, 0.0, 1.0])
    r = theta**2
    r_harder = r + 1000.0
    check("same optimizer under difficulty shift", int(np.argmin(r)), int(np.argmin(r_harder)))
    check("optimizer diameter gives no absolute-risk information", r_harder.min() - r.min(), 1000.0)


if __name__ == "__main__":
    quotient_sup_equals_pairwise_discrepancy()
    additive_nuisance_quotient_invariance()
    l2_measure_dependence()
    original_witness_is_nuisance_only()
    optimizer_diameter_not_risk_certificate()
    print("ALL MASTER-FUNCTIONAL SEARCH TESTS PASSED")
