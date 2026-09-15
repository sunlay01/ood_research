"""Generate the deterministic Phase-0 repairability validation report."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from ood_repr_reg.counterfactual_repairability import run_phase0_diagnostics  # noqa: E402


def _jsonable(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def main() -> int:
    output = ROOT / "round3_redesign" / "task3_counterfactual_repairability"
    results = output / "results"
    results.mkdir(parents=True, exist_ok=True)
    diagnostic = run_phase0_diagnostics(random_instances=100)
    (results / "phase0_diagnostics.json").write_text(
        json.dumps(_jsonable(diagnostic), indent=2) + "\n", encoding="utf-8"
    )
    checks = diagnostic["checks"]
    rows = "\n".join(
        f"| `{name}` | {'PASS' if passed else 'FAIL'} |" for name, passed in checks.items()
    )
    report = f"""# Phase 0 Validation Report

**Scope:** finite-dimensional centered adaptive repair only.  No target risk,
benchmark selection, semantic mechanism label, or static bias `z_0` is used.

The implemented quantity is

`R_adap(Pi) = 1/2 ||A + Pi O_S||_op^2`.

The controller class is a response-space linear class with an optional
operator/Frobenius budget.  Every class optimum retains its controller,
membership residual, solver trace, and spectral-slack certificate.

## Checks

| Check | Status |
| --- | --- |
{rows}

Overall Phase-0 status: **{'PASS' if diagnostic['pass'] else 'FAIL'}**.

## Toy values

| Quantity | Value |
| --- | ---: |
| Toy 1 `R_info` | `{diagnostic['separation']['R_info']:.12g}` |
| Toy 1 `R_head*` | `{diagnostic['separation']['R_head_star']:.12g}` |
| Toy 1 `R_full*` | `{diagnostic['separation']['R_full_star']:.12g}` |
| Toy 1 `R_actual` | `{diagnostic['separation']['R_actual']:.12g}` |
| Toy 1 value of representation-changing intervention | `{diagnostic['separation']['V_repr']:.12g}` |
| Toy 1 realization gap | `{diagnostic['separation']['Delta_real']:.12g}` |
| Toy 2 `R_info = R_class*` | `{diagnostic['information_limited']['R_info']:.12g}` |
| Toy 3 `||E||_op` | `{diagnostic['spectral_slack']['E_norm']:.12g}` |

Toy 3 has nonzero `E` and still attains the information floor.  This is an
explicit regression against restoring `E = 0` as a necessity.

## Independent checks

- `pytest -q tests/test_counterfactual_*.py` covers the controller interface,
  membership bridge, locality convention, nested classes, transport, and grid
  audit.
- `formalization/OODRelevance/RestrictedRepair.lean` was checked with the
  locally cached Lean/mathlib environment.  Its scope is the abstract class
  optimality and zero-observation identities; it does not formalize numerical
  optimization.
- Existing frozen tests remain a separate regression gate and are not folded
  into this Phase-0 scientific status.

## Interpretation boundary

This report establishes only that the proposed centered repairability objects
are internally consistent on declared finite-dimensional classes.  It does not
show that a class is a faithful model of any training algorithm, that a local
response predicts finite updates, or that response regret is target OOD risk.
Those are later gates and must not be inferred from this report.
"""
    (output / "phase0_validation_report.md").write_text(report, encoding="utf-8")
    print(json.dumps({"pass": diagnostic["pass"], "report": str(output / "phase0_validation_report.md")}, indent=2))
    return 0 if diagnostic["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
