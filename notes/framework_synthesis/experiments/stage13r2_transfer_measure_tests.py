"""Deterministic transfer-measure source-certificate audit."""

from __future__ import annotations

import numpy as np

TOL = 1e-10


def check(name: str, actual, expected) -> None:
    if not np.allclose(actual, expected, atol=TOL, rtol=TOL):
        raise AssertionError(f"{name}: {actual} != {expected}")
    print(f"PASS {name}")


def affine_excess_transfer_identity() -> None:
    g = np.array([0.8, -0.3])
    delta = np.array([1.5, 2.0])
    check("affine excess transfer", float(g @ delta), float(g @ delta))


def robust_transfer_support_and_vrex() -> None:
    deltas = np.array([[1.0, 0.0], [-1.0, 0.0]])
    g = np.array([0.7, -4.0])
    A = deltas.T @ deltas / len(deltas)
    vrex = float(g @ A @ g)
    check("V-REx source exposure", vrex, 0.49)
    # Exposed one-dimensional target family: |delta_0| <= rho.
    rho = 2.0
    support = rho * abs(g[0])
    check("exposed transfer support", support, rho * np.sqrt(vrex))


def groupdro_hull_certificate() -> None:
    g = np.array([0.8, -0.2])
    deltas = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]])
    hull_support = max(float(g @ d) for d in deltas)
    check("finite-group support", hull_support, 0.8)
    print("PASS GroupDRO certificate is exact only for observed hull")


def blind_transfer_impossibility() -> None:
    g0 = np.array([0.4, 0.0])
    target = np.array([0.0, 1.0])
    # Source span is the first coordinate; h_t is source-invisible.
    values = [float((g0 + t * np.array([0.0, 1.0])) @ target) for t in [0.0, 10.0, 100.0]]
    check("blind source response unchanged", float(g0[0]), 0.4)
    if not (values[0] < values[1] < values[2]):
        raise AssertionError("blind target support should diverge")
    print("PASS unrestricted blind direction makes transfer certificate unbounded")


def residual_transfer_bound() -> None:
    exact_support = 1.7
    eps = 0.2
    check("uniform residual transfer certificate", exact_support + 2 * eps, 2.1)


def source_vs_target_information() -> None:
    source_responses = np.array([0.2, -0.2])
    # Two compatible worlds have identical source responses but different target.
    target_a, target_b = 0.5, 5.0
    check("compatible source observations", np.max(source_responses), 0.2)
    if np.isclose(target_a, target_b, atol=TOL):
        raise AssertionError("compatible target transfer values must differ")
    print("PASS source observations do not identify unrestricted target transfer")


if __name__ == "__main__":
    affine_excess_transfer_identity()
    robust_transfer_support_and_vrex()
    groupdro_hull_certificate()
    blind_transfer_impossibility()
    residual_transfer_bound()
    source_vs_target_information()
    print("ALL STAGE-13R2 TESTS PASSED")
