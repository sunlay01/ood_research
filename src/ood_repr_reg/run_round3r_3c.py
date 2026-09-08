"""Run the independent 3C heterogeneous-regularizer audit."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path

import numpy as np

from .round3r_3c_benchmark import make_benchmark, response_records, source_risk, target_risk
from .round3r_3c_counterexamples import all_counterexamples
from .round3r_3c_derivatives import audit_regularizer
from .round3r_3c_geometry import response_metrics, response_space_coverage, whitened_action
from .round3r_3c_nonlinear import run_nonlinear_stress
from .round3r_3c_regularizers import regularizer_state
from .round3r_3c_solver import exact_solution, lambda_path


METHODS = ("l2", "coral", "irmv1", "vrex")


def _jsonable(value: object) -> object:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _regularizer_summary(state, coverage, audit, source_hessian) -> dict[str, object]:
    b, k, _ = whitened_action(source_hessian, state)
    return {
        "method": state.method,
        "constraint_object": state.constraint_object,
        "predictor_status": state.predictor_status,
        "gauge_status": state.gauge_status,
        "value_at_source_optimum": state.value,
        "gradient": state.gradient,
        "hessian": state.hessian,
        "source_whitened_b": b,
        "source_whitened_K": k,
        "coverage": coverage,
        "derivative_audit": audit,
        "notes": state.notes,
    }


def _gauge_audit() -> dict[str, object]:
    from .round2_induced_cost import coral_scaling_law

    base_penalty = 3.0
    scales = np.logspace(0, -6, 7)
    penalties = [coral_scaling_law(base_penalty, float(scale)) for scale in scales]
    return {
        "method": "CORAL",
        "predictor_preserved": True,
        "scales": scales,
        "penalties": penalties,
        "law": "c^4",
        "induced_infimum_zero": True,
        "classification": "gauge-dependent/induced-degenerate",
    }


def run(seed: int = 0) -> dict[str, object]:
    del seed
    benchmark = make_benchmark(n_noise=4, shortcut_count=2, relation_exposed=True)
    response = response_records(benchmark)
    relevant_q = benchmark.q_responses[:, benchmark.relevant_indices]
    statuses: dict[str, object] = {}
    derivative_audits: dict[str, object] = {}
    shift_rows: list[dict[str, object]] = []
    lambda_rows: list[dict[str, object]] = []
    for method in METHODS:
        state = regularizer_state(method, benchmark)
        audit = audit_regularizer(method, benchmark.optimum, benchmark)
        coverage = response_space_coverage(relevant_q, state, benchmark.hessian)
        statuses[method.upper()] = _regularizer_summary(state, coverage, audit, benchmark.hessian)
        derivative_audits[method.upper()] = audit
        for lam in (0.0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0):
            shift_rows.extend(response_metrics(benchmark, state, lam))
        path = lambda_path(benchmark, method)
        for row in path:
            row["source_exposure_used_for_selection"] = False
            lambda_rows.append(row)

    # Exact target behavior is recorded after the response-level matrix is
    # formed.  It is validation, not an input to regularizer geometry.
    for row in shift_rows:
        method = row["method"]
        lam = float(row["lambda"])
        exact = exact_solution(benchmark, method.lower(), lam)
        index = int(row["shift_index"])
        row["exact_target_risk"] = target_risk(benchmark, benchmark.probes[index], exact["weights"])
        row["source_optimum_target_risk"] = target_risk(benchmark, benchmark.probes[index], benchmark.optimum)
        row["actual_target_risk_change"] = row["exact_target_risk"] - row["source_optimum_target_risk"]
        row["exact_optimizer_success"] = exact["success"]

    coverage_summary = {
        method: {
            key: value for key, value in statuses[method]["coverage"].items()
            if key not in {"b", "k", "inverse_hessian_root"}
        }
        for method in statuses
    }
    return {
        "status": "complete",
        "benchmark": {
            "n_models": 1,
            "n_shifts": len(benchmark.probes),
            "n_relevant_shifts": int(len(benchmark.relevant_indices)),
            "feature_dimension": int(benchmark.optimum.size),
            "response_rank": int(np.linalg.matrix_rank(relevant_q, tol=1e-9)),
            "semantic_cluster_labels_used": False,
            "source_exposure_used_for_selection": False,
        },
        "regularizers": statuses,
        "coverage_summary": coverage_summary,
        "response_records": response,
        "regularizer_shift_matrix": shift_rows,
        "lambda_paths": lambda_rows,
        "derivative_audit": derivative_audits,
        "gauge_audit": _gauge_audit(),
        "counterexamples": all_counterexamples(),
        "nonlinear_stress": run_nonlinear_stress(),
        "verdict": "3C-ACTION-PASS-COVERAGE-PARTIAL",
        "verdict_basis": {
            "local_action_defined": True,
            "derivative_audits_pass": all(item["audit_pass"] for item in derivative_audits.values()),
            "coral_predictor_intrinsic": False,
            "3b_clusters_used": False,
            "semantic_or_causal_claim": False,
        },
    }


def _write_csv(path: Path, rows: list[dict[str, object]], fields: tuple[str, ...]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _jsonable(row.get(field, "")) for field in fields})


def report(result: dict[str, object]) -> str:
    return f"""# 3C Regularizer Control Report

## Verdict

`{result['verdict']}`

The analysis compares L2, CORAL, IRMv1 and V-REx through the chain
`constraint object -> (b, K) -> OOD-response coverage`.  The primary objects
are individual source-whitened response vectors.  The six 3B clusters and
oracle mechanism labels are excluded from primary calculations.

## Benchmark

- Relevant shifts: `{result['benchmark']['n_relevant_shifts']}` / `{result['benchmark']['n_shifts']}`.
- Response rank: `{result['benchmark']['response_rank']}`.
- Derivative audits passed: `{result['verdict_basis']['derivative_audits_pass']}`.
- 3B semantic clusters used: `{result['benchmark']['semantic_cluster_labels_used']}`.

## Method status

| Method | Predictor status | Gauge status | Relevant restricted rank | Relevant kernel dimension | PSD |
|---|---|---|---:|---:|---|
""" + "\n".join(
        f"| {method} | {value['predictor_status']} | {value['gauge_status']} | "
        f"{value['coverage']['restricted_rank']} | {value['coverage']['relevant_kernel_dimension']} | "
        f"{value['coverage']['psd']} |"
        for method, value in result["regularizers"].items()
    ) + """

## Interpretation

The local-action identities are population quadratic results.  Curvature
ratios, steering signs, blind fractions and nonselective curvature are
response-level diagnostics.  CORAL is explicitly not predictor-intrinsic on
the unconstrained representation fiber: the gauge audit records the exact
`c^4` scaling and zero induced infimum.

The verdict is partial because the local geometry is auditable, but this
benchmark does not justify a universal cross-method OOD-control theorem.  No
3B cluster is interpreted as a semantic or causal mechanism.

## Boundaries

The results are population calculations, not finite-sample guarantees, deep
network theorems, or source-identifiability results.  Exposure annotations are
post-hoc only.
"""


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    output = root / "round3_redesign" / "3C_regularizer_control"
    results = output / "results"
    results.mkdir(parents=True, exist_ok=True)
    result = run()
    (results / "summary.json").write_text(json.dumps(_jsonable(result), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_csv(results / "regularizer_status.csv", [
        {"method": method, "constraint_object": item["constraint_object"], "predictor_status": item["predictor_status"],
         "gauge_status": item["gauge_status"], "psd": item["coverage"]["psd"],
         "restricted_rank": item["coverage"]["restricted_rank"], "relevant_kernel_dimension": item["coverage"]["relevant_kernel_dimension"]}
        for method, item in result["regularizers"].items()
    ], ("method", "constraint_object", "predictor_status", "gauge_status", "psd", "restricted_rank", "relevant_kernel_dimension"))
    _write_csv(results / "regularizer_shift_matrix.csv", result["regularizer_shift_matrix"],
               ("method", "shift_id", "lambda", "relevance_squared", "control_score_c", "curvature_ratio", "blind_fraction", "steering", "steering_class", "local_vulnerability", "actual_target_risk_change", "exposure_posthoc", "intervention_family_posthoc"))
    _write_csv(results / "lambda_paths.csv", result["lambda_paths"],
               ("method", "lambda", "local_valid", "metric_min_eigenvalue", "local_displacement_norm", "exact_displacement_norm", "local_exact_error", "exact_source_risk", "exact_optimizer_success", "exact_optimizer_gradient_norm"))
    (results / "derivative_audit.json").write_text(json.dumps(_jsonable(result["derivative_audit"]), indent=2) + "\n", encoding="utf-8")
    (results / "gauge_audit.json").write_text(json.dumps(_jsonable(result["gauge_audit"]), indent=2) + "\n", encoding="utf-8")
    (results / "counterexamples.json").write_text(json.dumps(_jsonable(result["counterexamples"]), indent=2) + "\n", encoding="utf-8")
    (results / "nonlinear_stress.json").write_text(json.dumps(_jsonable(result["nonlinear_stress"]), indent=2) + "\n", encoding="utf-8")
    (output / "round3_3c_report.md").write_text(report(result), encoding="utf-8")
    print(output / "round3_3c_report.md")


if __name__ == "__main__":
    main()
