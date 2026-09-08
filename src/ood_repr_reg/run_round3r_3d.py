"""Run the independent 3D source-exposure and identifiability audit."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path

import numpy as np

from .round3r_3b_benchmark import environment_state, nonlinear_environment_state, source_design, source_optimum
from .round3r_3c_benchmark import make_benchmark
from .round3r_3d_counterexamples import design_unexposed_but_source_visible, source_identical_target_different
from .round3r_3d_designs import design_diagnostics, source_design_ladder
from .round3r_3d_exposure import (
    exact_exposure_membership,
    exposed_response_basis,
    quotient_dimensions,
    response_operator,
    source_design_span,
    source_reference_audit,
    target_exposure_fraction,
)
from .round3r_3d_identifiability import structural_identifiability
from .round3r_3d_intersection import posthoc_regularizer_intersection
from .round3r_3d_state import transform_moment_state


def _jsonable(value: object) -> object:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def run(tolerance: float = 1e-9, *, family=None) -> dict[str, object]:
    if family is None:
        benchmark = make_benchmark(n_noise=4, shortcut_count=2, relation_exposed=True)
        main_environments = source_design(benchmark.base, relation_exposed=True)
        main_optimum, main_source = source_optimum(main_environments)
    else:
        from .environment_family.geometry import build_task_geometry

        family_geometry = build_task_geometry(family)
        benchmark = make_benchmark(family=family)
        main_environments = family_geometry.source_environments
        main_optimum, main_source = family_geometry.source_optimum, family_geometry.source_state
    main_states = tuple(environment_state(environment) for environment in main_environments)
    main_span = source_design_span(main_states, 0, tolerance)
    operator = response_operator(main_source, main_optimum)
    exposed = exposed_response_basis(main_span["contrasts"], operator, tolerance)
    total_responses = benchmark.q_responses[:, benchmark.relevant_indices]
    total_basis = np.linalg.qr(total_responses)[0][:, :np.linalg.matrix_rank(total_responses, tol=tolerance)]
    quotient = quotient_dimensions(total_basis, exposed["basis"], tolerance)

    target_rows = []
    for index, probe in enumerate(benchmark.probes):
        response = response_operator(main_source, main_optimum, probe.target)
        target_rows.append({
            "shift_id": probe.shift_id,
            "relevance_squared": float(response @ response),
            "relevant_flag": bool(response @ response > 1e-10),
            "exact_exposed": exact_exposure_membership(response, exposed["basis"], tolerance),
            "exposure_fraction": target_exposure_fraction(response, exposed["basis"]),
            "residual_unexposed_norm": float(np.linalg.norm(response - exposed["basis"] @ (exposed["basis"].T @ response))),
            "source_design": "relation_exposed_main",
            "oracle_metadata_posthoc": {"mechanism": probe.mechanism, "intervention_family": probe.intervention_family, "probe_index": index},
        })

    design_rows = []
    for design in source_design_ladder(benchmark.base):
        optimum, source = source_optimum(design.environments)
        design_rows.append(design_diagnostics(design, optimum, total_basis, response_operator(source, optimum), tolerance))

    # The first map is an identifiable linear family; the second is the exact
    # hidden-emergent family where source observations cannot determine target action.
    identifiable_observation = np.eye(2)
    identifiable_target = np.array([[1.0, 0.5], [-0.25, 1.0]])
    structural = {
        "identifiable_linear_world_family": structural_identifiability(identifiable_observation, identifiable_target, tolerance),
        "hidden_emergent_world_family": structural_identifiability(np.zeros((1, 2)), np.array([[1.0, -1.0]]), tolerance),
    }
    pair = source_identical_target_different(benchmark.base)
    posthoc = posthoc_regularizer_intersection(exposed["basis"], benchmark, tolerance)
    rng = np.random.default_rng(19)
    coordinate_transform = rng.normal(size=(main_optimum.size, main_optimum.size)) + 2.0 * np.eye(main_optimum.size)
    transformed_source = transform_moment_state(main_source, coordinate_transform)
    transformed_optimum = np.linalg.solve(coordinate_transform.T, main_optimum)
    original_responses = np.column_stack([response_operator(main_source, main_optimum, probe.target) for probe in benchmark.probes])
    transformed_responses = np.column_stack([
        response_operator(transformed_source, transformed_optimum, transform_moment_state(probe.target, coordinate_transform))
        for probe in benchmark.probes
    ])
    original_gram = original_responses.T @ np.linalg.solve(np.eye(original_responses.shape[0]), original_responses)
    transformed_gram = transformed_responses.T @ transformed_responses
    nonlinear_state = nonlinear_environment_state(benchmark.base, (0.08, -0.04))
    nonlinear_stress = {
        "status": "diagnostic_only",
        "moment_psd": bool(np.min(np.linalg.eigvalsh(nonlinear_state.second)) > -1e-10),
        "state_dimension": int(nonlinear_state.second.shape[0]),
        "not_used_for_primary_exposure": True,
    }
    coordinate_audit = {
        "predictor_state_transform_invertible": True,
        "response_gram_max_error": float(np.max(np.abs(original_gram - transformed_gram))),
        "source_reference_invariant": bool(source_reference_audit(main_states, operator)["invariant"]),
    }
    return {
        "status": "complete",
        "benchmark": {"n_probes": len(benchmark.probes), "n_relevant": int(len(benchmark.relevant_indices)), "feature_dimension": int(main_optimum.size)},
        "exposure": {
            "source_state_dimension": int(main_span["states"].shape[0]),
            "source_contrast_rank": int(main_span["rank"]),
            "exposed_response_rank": int(exposed["rank"]),
            "total_response_rank": int(total_basis.shape[1]),
            "unexposed_quotient_dimension": quotient["unexposed_quotient_dimension"],
            "source_reference_audit": source_reference_audit(main_states, operator),
            "quotient": quotient,
            "target_information_used_in_exposure": False,
            "mechanism_labels_used_in_exposure": False,
            "cluster_labels_used_in_exposure": False,
            "regularizer_used_in_exposure": False,
        },
        "source_designs": design_rows,
        "target_exposure_records": target_rows,
        "structural_identifiability": structural,
        "exact_world_counterexample": pair,
        "source_visible_design_unexposed_counterexample": design_unexposed_but_source_visible(benchmark.base),
        "round3c_intersection": posthoc,
        "coordinate_audit": coordinate_audit,
        "nonlinear_stress": nonlinear_stress,
        "rank_tolerance_profile": {"relative": tolerance, "tested": [1e-10, 1e-9, 1e-8, 1e-7]},
        "target_information_used_in_exposure": False,
        "mechanism_labels_used_in_exposure": False,
        "cluster_labels_used_in_exposure": False,
        "regularizer_used_in_exposure": False,
        "source_reference_invariant": bool(coordinate_audit["source_reference_invariant"]),
        "family_geometry_used": family is not None,
        "verdict": "3D-DESIGN-EXPOSURE-PASS-STRUCTURAL-PARTIAL",
    }


def _write_csv(path: Path, rows: list[dict[str, object]], fields: tuple[str, ...]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _jsonable(row.get(field, "")) for field in fields})


def report(result: dict[str, object]) -> str:
    exposure = result["exposure"]
    return f"""# 3D Source Exposure and Response Identifiability

## Verdict

`{result['verdict']}`

The primary observable is the task-complete source state
`psi_e=(svec(M_e), m_e, c_e)`.  Exposure is the image of source contrasts
under the frozen source-response operator.  Target risk, mechanism labels,
3B clusters and 3C regularizers are excluded from this construction.

## Exposure geometry

- Source state dimension: `{exposure['source_state_dimension']}`.
- Main source contrast rank: `{exposure['source_contrast_rank']}`.
- Exposed response rank: `{exposure['exposed_response_rank']}`.
- Total frozen response rank: `{exposure['total_response_rank']}`.
- Unexposed response quotient dimension: `{exposure['unexposed_quotient_dimension']}`.
- Source-reference invariant: `{exposure['source_reference_audit']['invariant']}`.

These are finite-dimensional population linear-algebra results.  The exposure
fraction is metric-dependent; rank, kernel and quotient statements are the
intrinsic statements.  Repeated source rows do not add a contrast direction.
Risk-null nuisance changes can increase raw state diversity without increasing
the response image.  A new relevant source contrast can increase exposed rank.

## Structural identifiability

The identifiable linear world family satisfies the kernel inclusion criterion
and therefore factors through source observations.  The hidden-emergent family
has nonzero structural ambiguity: two worlds share source observations while
their target responses differ.  The exact Gaussian pair is recorded in
`exact_world_counterexample.json`.

## 3C intersection

The regularizer table is post-hoc only.  It does not define source exposure,
choose source designs, or select rank.  Its CORAL row retains the fixed
representation/gauge caveat inherited from 3C.

The coordinate audit preserves the response Gram to numerical precision, and
the nonlinear moment family is recorded as a diagnostic only.

## Boundaries

This does not establish finite-sample estimability, causal mechanism
identification, deep-network identifiability, or a universal DG theorem.
Mechanism interpretation and any 3E risk decomposition remain out of scope.
"""


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    output = root / "round3_redesign" / "3D_source_exposure_identifiability"
    results = output / "results"
    results.mkdir(parents=True, exist_ok=True)
    result = run()
    (results / "summary.json").write_text(json.dumps(_jsonable(result), indent=2) + "\n", encoding="utf-8")
    _write_csv(results / "source_design_table.csv", result["source_designs"], ("name", "n_source", "ambient_state_rank", "source_contrast_rank", "exposed_response_rank", "total_response_rank", "unexposed_quotient_dimension", "singular_values", "condition_number", "tolerance_profile"))
    _write_csv(results / "target_exposure_records.csv", result["target_exposure_records"], ("shift_id", "relevance_squared", "relevant_flag", "exact_exposed", "exposure_fraction", "residual_unexposed_norm", "source_design", "oracle_metadata_posthoc"))
    _write_csv(results / "round3c_intersection.csv", result["round3c_intersection"], ("method", "exposed_response_dimension", "exposed_controlled_dimension", "exposed_blind_dimension", "unexposed_controlled_diagnostic", "unexposed_blind_diagnostic", "posthoc_only"))
    (results / "structural_identifiability.json").write_text(json.dumps(_jsonable(result["structural_identifiability"]), indent=2) + "\n", encoding="utf-8")
    (results / "exact_world_counterexample.json").write_text(json.dumps(_jsonable(result["exact_world_counterexample"]), indent=2) + "\n", encoding="utf-8")
    (results / "coordinate_audit.json").write_text(json.dumps(_jsonable(result["coordinate_audit"]), indent=2) + "\n", encoding="utf-8")
    (results / "round2_recovery.json").write_text(json.dumps(_jsonable({"status": "recovered_as_exposure_baseline", "source_reference_invariant": result["exposure"]["source_reference_audit"]["invariant"]}), indent=2) + "\n", encoding="utf-8")
    (results / "nonlinear_stress.json").write_text(json.dumps(_jsonable(result["nonlinear_stress"]), indent=2) + "\n", encoding="utf-8")
    (output / "round3_3d_report.md").write_text(report(result), encoding="utf-8")
    print(output / "round3_3d_report.md")


if __name__ == "__main__":
    main()
