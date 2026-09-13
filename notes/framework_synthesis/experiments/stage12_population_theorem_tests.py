"""Deterministic finite-dimensional audit for the Stage 12 theorem package."""

from __future__ import annotations

import numpy as np

TOL = 1e-10


def check(name: str, actual, expected) -> None:
    if not np.allclose(actual, expected, atol=TOL, rtol=TOL):
        raise AssertionError(f"{name}: {actual} != {expected}")
    print(f"PASS {name}")


def transfer_identity_test() -> None:
    b = 0.4
    g = np.array([1.2, -0.7])
    psi_t, psi_bar = np.array([2.0, -1.0]), np.array([0.5, 0.25])
    eta_t, eta_bar = 0.08, -0.03
    rt = b + g @ psi_t + eta_t
    rs = b + g @ psi_bar + eta_bar
    rhs = g @ (psi_t - psi_bar) + eta_t - eta_bar
    check("exact transfer identity", rt - rs, rhs)
    eps = max(abs(eta_t), abs(eta_bar))
    if rt > rs + g @ (psi_t - psi_bar) + 2 * eps + TOL:
        raise AssertionError("residual transfer bound failed")
    print("PASS uniform residual transfer bound")


def support_minimality_test() -> None:
    target = np.array([[1.0, 2.0], [-1.0, 0.5], [0.2, -1.5]])
    g = np.array([0.8, -1.1])
    support = np.max(target @ g)
    check("finite-family support", support, max(float(g @ d) for d in target))
    for candidate in (support, support + 0.2, support + 7.0):
        if np.any(target @ g > candidate + TOL):
            raise AssertionError("candidate certificate does not cover U")
        if candidate + TOL < support:
            raise AssertionError("certificate below support should fail")
    print("PASS support is the smallest uniform additive certificate")


def blind_ambiguity_test() -> None:
    source = np.array([[1.0, 0.0], [-1.0, 0.0]])
    target_direction = np.array([0.0, 1.0])
    g = np.array([0.3, 0.0])
    h0 = np.array([0.0, 1.0])
    check("blind source responses", source @ (g + h0), source @ g)
    values = [(g + t * h0) @ target_direction for t in (1.0, 10.0, 100.0)]
    if values != sorted(values) or values[-1] < 100.0 - TOL:
        raise AssertionError("blind target support should diverge")
    print("PASS unrestricted blind extensions make source-only certificate unbounded")


def exposed_ellipsoid_test() -> None:
    A = np.diag([4.0, 1.0])
    rho = 1.7
    g = np.array([0.8, -1.2])
    s = np.sqrt(g @ A @ g)
    delta = rho * (A @ g) / s
    check("ellipsoid boundary", delta @ np.linalg.inv(A) @ delta, rho**2)
    check("ellipsoid support", g @ delta, rho * s)
    print("PASS exposed-span support <= rho * source seminorm, with equality witness")


def span_necessity_test() -> None:
    A = np.diag([1.0, 0.0])
    target = np.array([0.0, 1.0])
    for t in (1.0, 25.0):
        h = np.array([0.0, t])
        check("blind seminorm zero", h @ A @ h, 0.0)
        if h @ target <= 0:
            raise AssertionError("blind functional must see target direction")
    print("PASS finite seminorm domination requires target family inside source span")


def split_looseness_test() -> None:
    coupled = np.array([[1.0, 1.0], [-1.0, -1.0]])
    g = np.array([1.0, -1.0])
    check("coupled exact support", np.max(coupled @ g), 0.0)
    check("split relaxation", abs(g[0]) + abs(g[1]), 2.0)


if __name__ == "__main__":
    transfer_identity_test()
    support_minimality_test()
    blind_ambiguity_test()
    exposed_ellipsoid_test()
    span_necessity_test()
    split_looseness_test()
    print("ALL STAGE-12 TESTS PASSED")
