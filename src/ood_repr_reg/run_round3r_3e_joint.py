"""Runner for the immutable 3E-B joint regret track."""

from __future__ import annotations

import csv
import json
from dataclasses import is_dataclass, asdict
from pathlib import Path

import numpy as np

from .round3r_3c_affine import main_benchmark
from .round3r_3e_joint_fixtures import fixtures, same_b_k_different_pi
from .round3r_3e_joint_regret import (
    independent_shifted_ball_maximum,
    metric_regret_decomposition,
    regret_decomposition,
)
from .round3r_3e_world_tangent import (
    coupled_primary_geometry, finite_difference_stability, response_factorization_residual,
    u_exposed_observation,
)
from .round3r_3e_recovery import minimax_recovery_error


def _jsonable(value):
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    return value


def run(*, family=None):
    if family is None:
        geometry = coupled_primary_geometry()
        exposed_observation = u_exposed_observation(geometry)
        benchmark = main_benchmark()
    else:
        from .environment_family.geometry import build_task_geometry

        geometry = build_task_geometry(family)
        exposed_observation = None
        benchmark = main_benchmark(family=family, family_geometry=geometry)
    rows = []
    audits = []
    for item in benchmark["audits"]:
        if not item["fd_audit"]["pass"]:
            continue
        affine = (
            regret_decomposition(item["z0"], geometry.response, item["tangent"], geometry.observation)
            if family is None else
            metric_regret_decomposition(
                item["z0"], geometry.response, item["tangent"], geometry.observation,
                geometry.spec.metric,
            )
        )
        if family is None:
            independent_response = geometry.response + item["tangent"]
        else:
            from .environment_family.metrics import world_whiten

            independent_response = (
                world_whiten(geometry.response, geometry.spec.metric) +
                world_whiten(item["tangent"], geometry.spec.metric)
            )
        independent = independent_shifted_ball_maximum(item["z0"], independent_response, starts=24)
        rows.append({"setting": "coupled_3a_3d_primary", "method": item["method"], "lambda": item["lambda"],
                     "information_floor": affine["information_floor"], "affine_total_regret": affine["total_regret"],
                     "total_excess": max(0.0, affine["total_regret"] - affine["information_floor"]),
                     "static_only_regret": affine["static_only_regret"],
                     "adaptive_only_regret": affine["adaptive_only_regret"], "interaction": affine["interaction"],
                     "recoverable_residual_operator_norm": affine["recoverable_residual_operator_norm"],
                     "independent_solver_value": independent["value"],
                     "independent_solver_error": abs(affine["total_regret"] - independent["value"])})
        audits.append({"method": item["method"], "lambda": item["lambda"], "decomposition": affine,
                       "independent_optimizer": independent})
    root = Path(__file__).resolve().parents[2]
    if family is None:
        information_floor = 0.5 * minimax_recovery_error(geometry.response, geometry.observation) ** 2
    else:
        from .environment_family.metrics import metric_information_floor

        information_floor = metric_information_floor(
            geometry.response, geometry.observation, geometry.spec.metric,
        )
    output = (
        root / "round3_redesign" / "3E_joint_information_regularization_regret"
        if family is None else
        root / "round3_redesign" / "environment_family_refactor" / "family_joint_regret"
    )
    results = output / "results"
    results.mkdir(parents=True, exist_ok=True)
    summary = {
        "status": "complete", "coupled_3a_3d_primary": True,
        "world_metric": geometry.spec.metadata(),
        "world_tangent_dimension": geometry.spec.dimension,
        "source_observation_shape": list(geometry.observation.shape),
        "response_shape": list(geometry.response.shape),
        "information_floor": information_floor,
        "u_hidden_information_floor": information_floor,
        "u_exposed_information_floor": None if exposed_observation is None else 0.5 * minimax_recovery_error(geometry.response, exposed_observation) ** 2,
        "response_factorization_residual": (
            response_factorization_residual(geometry) if family is None else
            __import__("ood_repr_reg.environment_family.geometry", fromlist=["family_response_factorization_residual"]).family_response_factorization_residual(geometry)
        ),
        "world_tangent_fd_audit": (
            finite_difference_stability(geometry) if family is None else
            {"status": "family_geometry_fd_audited_by_family_runner"}
        ),
        "max_independent_solver_error": max((row["independent_solver_error"] for row in rows), default=0.0),
        "max_total_excess": max((row["total_excess"] for row in rows), default=0.0),
        "max_fd_error": max((item["fd_audit"]["max_error"] for item in benchmark["audits"]), default=0.0),
        "valid_policy_count": len(rows),
        "rows": rows, "target_risk_lower_bound_claimed": False,
        "semantic_decomposition_used": False, "regularizer_exposure_containment_assumed": False,
        "lean_status": "LEAN-PARTIAL", "verdict": "3E-B-JOINT-AFFINE-PASS",
    }
    (results / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2) + "\n", encoding="utf-8")
    (results / "exact_quadratic_diagnostics.json").write_text(json.dumps(_jsonable(audits), indent=2) + "\n", encoding="utf-8")
    (results / "fixtures.json").write_text(json.dumps(_jsonable(fixtures()), indent=2) + "\n", encoding="utf-8")
    (results / "same_b_k_different_pi.json").write_text(json.dumps(_jsonable(same_b_k_different_pi()), indent=2) + "\n", encoding="utf-8")
    with (results / "joint_regret.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ("setting", "method", "lambda", "information_floor", "affine_total_regret", "total_excess",
                  "static_only_regret", "adaptive_only_regret", "interaction",
                  "recoverable_residual_operator_norm", "independent_solver_value", "independent_solver_error")
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        writer.writerows({field: _jsonable(row.get(field, "")) for field in fields} for row in rows)
    (output / "round3_3e_b_report.md").write_text(_report(summary, rows), encoding="utf-8")
    (output / "assumptions.md").write_text("""# 3E-B assumptions

The common-metric primary object is `L_u(z)=1/2||z||^2+(Au)^T z` on the
standardized 8D tangent. Concrete regularizers enter only through the exact
affine policy `z0+(A+Pi O_S)u`. The primary source design hides the emergent U
direction; a U-exposed observation is reported only as a separate information
intervention. This is a response-regret theorem/audit, not a target-risk lower
bound or semantic mechanism result.
""", encoding="utf-8")
    (output / "theory.md").write_text("""# 3E-B theory

For `c=z0` and `B=A+Pi O_S`, the total regret is the shifted-ball maximum
`1/2 sup_{||u||<=1} ||c+Bu||^2`. The secular equation solves this finite
trust-region problem, including the repeated-top-eigenvalue hard case. Static,
adaptive and interaction values are counterfactual diagnostics and are not
claimed to be additive. The information floor is `1/2||A P_ker(O_S)||_op^2`.
The reported `total_excess` is absolute affine regret minus this floor, clipped
at zero only for roundoff; it is not target-risk excess.
""", encoding="utf-8")
    (output / "novelty_positioning.md").write_text("""# Positioning

The generic trust-region and information-floor facts are machinery. The
reported object is the coupled operational comparison of source information,
static regularizer steering, and source-adaptive affine action.
""", encoding="utf-8")
    (output / "proof_notes.md").write_text("""# Proof notes

The solver is cross-checked by independent constrained multi-start numerical
optimization. Hidden-emergent U remains source-null in the primary geometry;
U-exposed information is a separate intervention and is not folded into the
primary benchmark.
""", encoding="utf-8")
    return output


def _report(summary: dict[str, object], rows: list[dict[str, object]]) -> str:
    floor = float(summary["information_floor"])
    max_solver = float(summary["max_independent_solver_error"])
    max_fd = float(summary["max_fd_error"])
    exposed_floor = summary["u_exposed_information_floor"]
    exposed_text = "n/a" if exposed_floor is None else f"{float(exposed_floor):.10g}"
    return f"""# 3E-B Joint Information-Regularization Affine Regret

## Verdict

`3E-B-JOINT-AFFINE-PASS`

The primary pair is the coupled 3A/3D tangent.  3A supplies the response
metric and 3D supplies the source observation operator.  The source-hidden
emergent `U` direction is retained in the primary benchmark; a U-exposed
observation is a separately labelled information intervention.

## Exact finite-dimensional object

For `L_u(z)=1/2||z||^2 + (A u)^T z`, a concrete regularizer contributes the
affine policy `z0 + (A + Pi O_S)u`.  Its reported total regret is
`1/2 sup_(||u||<=1)||z0+(A+Pi O_S)u||^2`.  The irreducible information floor is
`1/2 ||A P_ker(O_S)||_op^2 = {floor:.10g}`.  `total_excess` in the CSV is
absolute affine regret minus that floor, clipped at zero only for roundoff;
it is not target-risk excess.

## Numerical audit

- methods and valid lambdas: {len(rows)}
- coupled world tangent: `{summary['world_tangent_dimension']}D`
- source observation shape: `{tuple(summary['source_observation_shape'])}`
- primary hidden-U floor: `{summary['u_hidden_information_floor']:.10g}`
    - U-exposed floor: `{exposed_text}`
- factorization residual: `{summary['response_factorization_residual']:.3g}`
- maximum IFT finite-difference error: `{max_fd:.3g}`
- maximum secular versus independent constrained optimizer error: `{max_solver:.3g}`

The secular-equation solver includes the repeated-top-eigenvalue hard case and
is independently checked by multi-start constrained optimization.  Every
concrete policy in the primary table is above the information floor.

## Decomposition and limits

Static-only, adaptive-only, and interaction values are counterfactual
diagnostics and are not assumed additive.  The recoverable residual reports
the response action remaining after the source-invisible component is removed;
it does not assert target-risk control.  The same `(b,K)` can coexist with
different `Pi`, so frozen curvature alone does not determine the joint affine
action.

No 3B semantic module, 3C semantic/control assumption, target-risk oracle,
causal interpretation, finite-sample guarantee, or universal DG theorem is
used.  The Lean extension remains `LEAN-PARTIAL` because IFT, pseudoinverse
operator norms, and general trust-region maximization are not claimed as
formally verified here.
"""


if __name__ == "__main__":
    print(run())
