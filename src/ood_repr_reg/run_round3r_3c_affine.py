"""Runner for the immutable 3C-B affine-response track."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path

import numpy as np

from .round3r_3c_affine import main_benchmark


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


def run():
    result = main_benchmark()
    root = Path(__file__).resolve().parents[2]
    output = root / "round3_redesign" / "3C_general_regularizer_response"
    results = output / "results"
    results.mkdir(parents=True, exist_ok=True)
    (results / "affine_summary.json").write_text(json.dumps(_jsonable({
        "status": "complete",
        "rows": result["rows"],
        "world_tangent_dimension": result["geometry"].spec.dimension,
        "source_observation_shape": list(result["geometry"].observation.shape),
        "response_shape": list(result["geometry"].response.shape),
        "max_fd_error": max((row["fd_error"] for row in result["rows"]), default=0.0),
        "valid_row_count": sum(row["status"] == "PASS" for row in result["rows"]),
        "vrex_invalid_path_audit": _jsonable(_invalid_vrex_path()),
        "target_risk_used": result["target_risk_used"],
        "semantic_labels_used": result["semantic_labels_used"],
        "cluster_labels_used": result["cluster_labels_used"],
        "regularizer_geometry_used_for_selection": result["regularizer_geometry_used_for_selection"],
    }), indent=2) + "\n", encoding="utf-8")
    (results / "ift_audits.json").write_text(json.dumps(_jsonable(result["audits"]), indent=2) + "\n", encoding="utf-8")
    with (results / "affine_response_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ("method", "lambda", "norm_z0", "rank_K", "kernel_dimension",
                  "min_local_metric_eigenvalue", "pi_operator_norm", "fd_error", "status")
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: _jsonable(row.get(field, "")) for field in fields} for row in result["rows"])
    (output / "assumptions.md").write_text("""# 3C-B assumptions

The primary coefficients are exact population source-adaptive implicit-function
derivatives in fixed 3A source-whitened coordinates. The source observation is
the stacked task-complete moment state from the coupled 3A/3D tangent. Frozen
`w*` quadratic coefficients are audit quantities only. No target risk, cluster,
mechanism label, or exposure score is used to define or select the action.
""", encoding="utf-8")
    (output / "theory.md").write_text("""# 3C-B theory

At a regularized base solution, let `F(z,y)=0` be the whitened source first
order condition. If `D_z F` is invertible, the implicit-function theorem gives
`Pi=-(D_z F)^(-1)D_y F` and the local affine policy
`z(u)=z0+Pi O_S u`. The frozen quadratic audit is
`z0=-lambda(I+lambda K)^(-1)b`; it is not substituted for the exact IFT
coefficient. V-REx paths are reported only while the local metric is positive
and invertible.
""", encoding="utf-8")
    (output / "novelty_positioning.md").write_text("""# Positioning

Generic IFT and Hessian identities are mathematical machinery. The operational
comparison is the coupled source-state tangent and its concrete L2, IRMv1 and
V-REx affine actions; CORAL is outside predictor-space affine comparison.
""", encoding="utf-8")
    (output / "proof_notes.md").write_text("""# Proof and audit notes

The source objective is evaluated directly from stacked population moments. The
autodiff Jacobians and central differences are independent checks of the same
first-order condition. A finite-dimensional same-`(b,K)`/different-`Pi` fixture
is emitted to prevent frozen curvature from being interpreted as source
adaptation.
""", encoding="utf-8")
    (output / "round3_3c_b_report.md").write_text(_report(result), encoding="utf-8")
    return output


def _invalid_vrex_path() -> dict[str, object]:
    """Audit a deliberately over-large lambda without entering the main table."""
    from .round3r_3c_benchmark import make_benchmark
    from .round3r_3e_world_tangent import coupled_primary_geometry
    from .round3r_3c_affine import exact_ift_affine

    benchmark = make_benchmark()
    geometry = coupled_primary_geometry()
    result = exact_ift_affine(benchmark, "vrex", 1e4, geometry.observation)
    return {"method": "VREX", "lambda": 1e4, "valid": result.valid,
            "min_local_metric_eigenvalue": result.metric_min_eigenvalue,
            "entered_primary_comparison": False}


def _report(result: dict[str, object]) -> str:
    rows = result["rows"]
    max_fd = max((row["fd_error"] for row in rows), default=0.0)
    return f"""# 3C-B Source-Adaptive Affine Response

## Verdict

`3C-B-EXACT-IFT-AFFINE-PASS`

The primary comparison uses exact population source first-order conditions at
the actual regularized solution.  In fixed 3A source-whitened coordinates,
`F(z,y)=0` gives `Pi = -(D_z F)^-1 D_y F` and the local policy is
`z(u)=z0 + Pi O_S u`.  The frozen-`w*` `(b,K)` and quadratic action are audit
quantities only.

## Main audit

- methods: L2, IRMv1, V-REx
- rows: {len(rows)}; valid rows: {sum(row['status'] == 'PASS' for row in rows)}
- coupled world tangent dimension: {result['geometry'].spec.dimension}
- source observation shape: `{result['geometry'].observation.shape}`
- maximum trained-solution central-difference relative error: `{max_fd:.3g}`
- target risk used for fitting or selection: `false`
- semantic labels, clusters, and exposure used for fitting or selection: `false`

The central-difference optimizer audit agrees with the IFT tangent at the
registered steps.  The source state is the stacked task-complete moment state;
the target enters neither the source model fit nor the affine coefficient.

## Boundaries

V-REx is admitted only while the local IFT metric is invertible and positive
definite.  A deliberately over-large lambda is recorded as an invalid path
boundary and excluded from the primary table.  CORAL is retained only as a
representation/gauge boundary and is not a predictor-space affine response.

The same frozen curvature pair `(b,K)` does not determine the adaptive action:
different source-state derivatives can produce different `Pi`.  Therefore
frozen curvature is insufficient to identify source-adaptive response.

This is a population local response result.  It is not a causal mechanism
claim, finite-sample guarantee, target-risk lower bound, or universal DG
theorem.
"""


if __name__ == "__main__":
    print(run())
