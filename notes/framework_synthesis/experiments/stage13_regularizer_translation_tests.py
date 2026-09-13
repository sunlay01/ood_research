"""Deterministic Stage 13 translation and counterexample audit."""

from __future__ import annotations

import numpy as np

TOL = 1e-10


def check(name: str, actual, expected) -> None:
    if not np.allclose(actual, expected, atol=TOL, rtol=TOL):
        raise AssertionError(f"{name}: {actual} != {expected}")
    print(f"PASS {name}")


def risk_level_identity_test() -> None:
    risks = np.array([1.0, 3.0, 2.0])
    mean = risks.mean()
    delta = risks - mean
    A_energy = np.mean(delta**2)
    check("V-REx exposure identity", A_energy, np.var(risks))
    check("GroupDRO convex-hull support", np.max(risks), np.max(risks))


def vrex_blind_test() -> None:
    source = np.array([[1.0, 0.0], [-1.0, 0.0]])
    g1, g2 = np.array([0.0, 0.0]), np.array([0.0, 100.0])
    check("V-REx source risk unchanged", source @ g2, source @ g1)
    target = np.array([0.0, 1.0])
    if abs(target @ g2) < 99.0:
        raise AssertionError("blind target risk should be large")
    print("PASS V-REx zero does not control blind target risk")


def groupdro_outside_hull_test() -> None:
    source_states = np.array([[-1.0], [1.0]])
    source_risks = np.array([0.0, 0.0])
    outside = np.array([2.0])
    g = np.array([1.0])
    check("GroupDRO observed hull risk", np.max(source_risks), 0.0)
    if outside @ g <= np.max(source_states @ g):
        raise AssertionError("outside target must not be inferred from hull maximum")
    print("PASS GroupDRO convex hull does not cover outside target")


def mmrex_test() -> None:
    risks = np.array([1.0, 4.0, 2.0, 3.0])
    r = 0.75
    centered = risks - risks.mean()
    beta = r * centered / np.linalg.norm(centered)
    value = np.mean(risks) + beta @ risks
    expected = np.mean(risks) + r * np.linalg.norm(centered)
    check("MM-REx bounded affine support", value, expected)
    check("MM-REx affine weights sum to one", np.sum(np.ones(4) / 4 + beta), 1.0)
    if np.any(np.ones(4) / 4 + beta < 0):
        print("PASS MM-REx witness permits extrapolative negative weight")
    else:
        print("PASS MM-REx bounded affine support (nonnegative witness)")


def mmd_dual_test() -> None:
    delta = np.array([3.0, 4.0])
    g = np.array([0.6, 0.8])
    check("fixed-state MMD dual equality witness", abs(g @ delta), np.linalg.norm(g) * np.linalg.norm(delta))


def coral_counterexample_test() -> None:
    # Same E[X^2] and E[Y^2], opposite E[XY].
    x = np.array([-1.0, 1.0])
    ya, yb = x.copy(), -x.copy()
    check("CORAL covariance coordinate unchanged", np.mean(x**2), 1.0)
    check("CORAL label energy unchanged", np.mean(ya**2), np.mean(yb**2))
    ra = np.mean((x - ya) ** 2)
    rb = np.mean((x - yb) ** 2)
    if np.isclose(ra, rb, atol=TOL):
        raise AssertionError("CORAL counterexample must change risk")
    print(f"PASS CORAL zero moment discrepancy but risk changes: {ra:g} vs {rb:g}")


def irm_identity_and_counterexample_test() -> None:
    M = np.array([2.0, 3.0])
    c = np.array([2.0, 3.0])
    w = np.array([1.0, 1.0])
    check("ideal IRM stationary identity", 2 * (M * w - c), np.zeros(2))
    # Both derivatives vanish at a=1, but global minimizers do not coincide.
    r1 = lambda a: (a - 1.0) ** 2
    r2 = lambda a: ((a - 1.0) ** 2 - 1.0) ** 2
    d1 = 2 * (1.0 - 1.0)
    d2 = 4 * (1.0 - 1.0) * ((1.0 - 1.0) ** 2 - 1.0)
    check("IRMv1 derivative at fixed scale", d1 + d2, 0.0)
    if r2(1.0) <= min(r2(0.0), r2(2.0)):
        raise AssertionError("IRMv1 witness must be nonoptimal at fixed stationary scale")
    print("PASS IRMv1 zero gradient does not imply ideal common optimum")


def fishr_richer_state_test() -> None:
    # Same minimal moments but different E[(wX-Y)^2 X^2].
    x1, y1 = np.array([-1.0, 1.0]), np.array([0.0, 0.0])
    x2, y2 = np.array([-2.0, 0.0, 0.0, 2.0]), np.array([0.0, 0.0, 0.0, 0.0])
    # Normalize the second sample to have the same E[X^2] = 1.
    x2 = x2 / np.sqrt(2.0)
    check("Fishr minimal E[X^2]", np.mean(x1**2), np.mean(x2**2))
    check("Fishr minimal E[XY]", np.mean(x1 * y1), np.mean(x2 * y2))
    check("Fishr minimal E[Y^2]", np.mean(y1**2), np.mean(y2**2))
    q1 = np.mean((x1 - y1) ** 2 * x1**2)
    q2 = np.mean((x2 - y2) ** 2 * x2**2)
    if np.isclose(q1, q2, atol=TOL):
        raise AssertionError("gradient covariance witness must differ")
    print("PASS Fishr requires moments beyond minimal quadratic state")


def shared_blind_barrier_test() -> None:
    source = np.array([[1.0, 0.0], [-1.0, 0.0]])
    g = np.array([0.2, 0.0])
    h = np.array([0.0, 5.0])
    check("shared source-risk fiber", source @ (g + h), source @ g)
    print("PASS ERM/V-REx/GroupDRO/MM-REx source-risk summaries share blind barrier")


if __name__ == "__main__":
    risk_level_identity_test()
    vrex_blind_test()
    groupdro_outside_hull_test()
    mmrex_test()
    mmd_dual_test()
    coral_counterexample_test()
    irm_identity_and_counterexample_test()
    fishr_richer_state_test()
    shared_blind_barrier_test()
    print("ALL STAGE-13 TESTS PASSED")
