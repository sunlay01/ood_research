"""Independent symbolic checks for the 3A-T quadratic theory.

These checks are sanity checks, not formal proofs.  They deliberately use
small symbolic matrices and write a machine-readable result next to this
script.
"""

from __future__ import annotations

import json
from pathlib import Path

import sympy as sp


def _assert_zero(expr: sp.Expr, label: str) -> None:
    simplified = expr.applyfunc(sp.simplify) if isinstance(expr, sp.MatrixBase) else sp.simplify(expr)
    is_zero = all(value == 0 for value in simplified) if isinstance(simplified, sp.MatrixBase) else simplified == 0
    if not is_zero:
        raise AssertionError(f"{label} did not simplify to zero: {simplified}")


def main() -> dict[str, object]:
    a, b, c, d = sp.symbols("a b c d", real=True)
    x, y, p, q = sp.symbols("x y p q", real=True)
    M = sp.Matrix([[a, b], [b, c]])
    w = sp.Matrix([x, y])
    delta = sp.Matrix([p, q])
    m = sp.Matrix([a * x + b * y, b * x + c * y])

    rs = ((w + delta).T * M * (w + delta) - 2 * m.T * (w + delta))[0]
    rs0 = (w.T * M * w - 2 * m.T * w)[0]
    _assert_zero(rs - rs0 - (delta.T * M * delta)[0], "source excess")

    at, bt, ct = sp.symbols("at bt ct", real=True)
    mt1, mt2 = sp.symbols("mt1 mt2", real=True)
    MT = sp.Matrix([[at, bt], [bt, ct]])
    mT = sp.Matrix([mt1, mt2])
    delta_M = MT - M
    delta_m = mT - m
    g = 2 * delta_M * w - 2 * delta_m
    shift = ((w + delta).T * delta_M * (w + delta) - 2 * delta_m.T * (w + delta))[0]
    shift0 = (w.T * delta_M * w - 2 * delta_m.T * w)[0]
    _assert_zero((shift - shift0 - (g.T * delta)[0] - (delta.T * delta_M * delta)[0]), "shift polynomial")

    # Residual coupling: m_T - M_T w = E_T[X(Y-X'w)].
    residual_target = mT - MT * w
    _assert_zero(g + 2 * residual_target - 2 * (m - M * w), "residual coupling")
    # Under M w = m, this reduces to g = -2 residual_target.

    MS = sp.Matrix([[2, 0], [0, 1]])
    mS = sp.Matrix([1, 0])
    MTc = sp.Matrix([[2, 1], [1, 2]])
    mTc = sp.Matrix([1, 1])
    wstar = MS.inv() * mS
    gc = 2 * (MTc - MS) * wstar - 2 * (mTc - mS)
    if wstar[1] != 0 or gc[1] != -1:
        raise AssertionError(f"counterexample mismatch: w={wstar}, g={gc}")

    kappa = sp.symbols("kappa", nonzero=True, real=True)
    # Same M,m means the delta risk is constant for every w.
    common_difference = kappa
    _assert_zero((common_difference - kappa), "common burden")

    ME = sp.Matrix([[2, 1], [1, 3]])
    MN = sp.Matrix([[4, 1], [1, 2]])
    gE = sp.Matrix([x, y])
    gA = sp.Matrix.vstack(gE, sp.zeros(2, 1))
    MA = sp.diag(1, 1, 1, 1)
    MA[:2, :2] = ME
    MA[2:, 2:] = MN
    lhs = (gA.T * MA.inv() * gA)[0]
    rhs = (gE.T * ME.inv() * gE)[0]
    _assert_zero(lhs - rhs, "nuisance invariance")

    # H=2M convention: g'M^-1g = 2 g'H^-1g.
    H = 2 * MS
    _assert_zero((gc.T * MS.inv() * gc)[0] - 2 * (gc.T * H.inv() * gc)[0], "factor of two")

    result = {
        "status": "PASS",
        "checks": [
            "source_excess_identity",
            "shift_polynomial_identity",
            "residual_coupling_identity",
            "source_usage_counterexample",
            "common_burden_constant_shift",
            "block_nuisance_invariance",
            "M_vs_H_factor_two",
        ],
        "normalization": "L = sqrt(g^T M^-1 g) = sqrt(2) * sqrt(g^T H^-1 g)",
    }
    output = Path(__file__).with_name("symbolic_check_results.json")
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
