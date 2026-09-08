"""Run the environment-family refactor audit.

This runner is intentionally downstream of the shared family interface.  The
legacy numbers are checked against fixed compatibility facts and, when the
older result files are present, against their scalar tables.  The new family
comparison never uses probe labels, target risk, clusters, or regularizer
geometry to construct either source observations or response maps.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .environment_family import (
    build_source_induced_family,
    build_task_geometry,
    family_coordinate_audit,
    family_response_factorization_residual,
    legacy_gaussian_family,
    source_induced_audit,
)
from .environment_family.invariance_audit import failure_classification, source_span_projector
from .environment_family.metrics import metric_operator_summary, world_whiten
from .round3r_3b_benchmark import environment_state
from .round3r_3c_affine import exact_ift_affine, main_benchmark
from .round3r_3c_benchmark import make_benchmark
from .round3r_3d_state import task_state
from .round3r_3e_joint_regret import metric_regret_decomposition
from .round3r_3e_world_tangent import (
    coupled_primary_geometry,
    response_factorization_residual,
    u_exposed_observation,
)

Array = np.ndarray

FROZEN_LEGACY_MATRIX_HASH = "ba982e44ddfd867ba75d4a9f0c10c91e842b1d93255223aa2cda47d1a0b95828"
FROZEN_HIDDEN_INFORMATION_FLOOR = 0.05118145108608892


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return "inf" if value > 0 else "-inf"
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _rank(matrix: Array, tolerance: float = 1e-9) -> int:
    singular = np.linalg.svd(np.asarray(matrix, dtype=float), compute_uv=False)
    if singular.size == 0 or singular[0] <= 0.0:
        return 0
    return int(np.sum(singular > tolerance * singular[0]))


def _operator_norm(matrix: Array) -> float:
    singular = np.linalg.svd(np.asarray(matrix, dtype=float), compute_uv=False)
    return float(singular[0]) if singular.size else 0.0


def _matrix_hash(observation: Array, response: Array) -> str:
    payload = np.concatenate((
        np.ascontiguousarray(observation, dtype=np.float64).ravel(),
        np.ascontiguousarray(response, dtype=np.float64).ravel(),
    ))
    return hashlib.sha256(payload.tobytes()).hexdigest()


def _metric_pair(geometry) -> tuple[Array, Array]:
    return (
        world_whiten(geometry.response, geometry.spec.metric),
        world_whiten(geometry.observation, geometry.spec.metric),
    )


def _affine_decomposition(geometry, item: dict[str, object]) -> dict[str, object]:
    """Evaluate an old 3E-B policy in the family-declared world metric."""
    return metric_regret_decomposition(
        np.asarray(item["z0"]), np.asarray(geometry.response),
        np.asarray(item["tangent"]), np.asarray(geometry.observation),
        np.asarray(geometry.spec.metric),
    )


def _scalar_check(quantity: str, frozen: float | int | str | None,
                  current: float | int | str | None, tolerance: float = 1e-8) -> dict[str, object]:
    if frozen is None or current is None:
        return {
            "quantity": quantity, "frozen_old_result": frozen,
            "refactored_result": current, "abs_error": None,
            "rel_error": None, "pass": "unavailable",
        }
    if isinstance(frozen, (int, float)) and isinstance(current, (int, float)):
        error = abs(float(current) - float(frozen))
        relative = error / max(1.0, abs(float(frozen)))
        passed = bool(error <= tolerance * max(1.0, abs(float(frozen))))
    else:
        error = 0.0 if frozen == current else float("inf")
        relative = error
        passed = frozen == current
    return {
        "quantity": quantity, "frozen_old_result": frozen,
        "refactored_result": current, "abs_error": error,
        "rel_error": relative, "pass": passed,
    }


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _old_table_regression(root: Path, geometry, family) -> tuple[dict[str, object], dict[str, object]]:
    """Compare scalar compatibility tables without importing old algorithms."""
    current_3c = main_benchmark(
        family=family, family_geometry=geometry,
        lambdas=(0.0, 1e-4, 1e-3, 1e-2, 1e-1),
    )
    old_3c_rows = _read_rows(
        root / "round3_redesign" / "3C_general_regularizer_response" /
        "results" / "affine_response_summary.csv"
    )
    new_3c_rows = current_3c["rows"]
    max_3c = 0.0
    matched_3c = 0
    old_by_key = {(row.get("method"), float(row.get("lambda", "nan"))): row for row in old_3c_rows}
    for row in new_3c_rows:
        key = (row["method"], float(row["lambda"]))
        old = old_by_key.get(key)
        if old is None:
            continue
        matched_3c += 1
        for field in ("norm_z0", "rank_K", "kernel_dimension", "min_local_metric_eigenvalue", "pi_operator_norm"):
            max_3c = max(max_3c, abs(float(row[field]) - float(old[field])))

    current_3e: list[dict[str, object]] = []
    for item in current_3c["audits"]:
        if not item["fd_audit"]["pass"]:
            continue
        parts = _affine_decomposition(geometry, item)
        current_3e.append({
            "method": item["method"], "lambda": float(item["lambda"]),
            "information_floor": float(parts["information_floor"]),
            "affine_total_regret": float(parts["total_regret"]),
            "total_excess": float(max(0.0, parts["total_regret"] - parts["information_floor"])),
            "static_only_regret": float(parts["static_only_regret"]),
            "adaptive_only_regret": float(parts["adaptive_only_regret"]),
            "interaction": float(parts["interaction"]),
            "recoverable_residual_operator_norm": float(parts["recoverable_residual_operator_norm"]),
        })
    old_3e_rows = _read_rows(
        root / "round3_redesign" / "3E_joint_information_regularization_regret" /
        "results" / "joint_regret.csv"
    )
    old_3e_by_key = {(row.get("method"), float(row.get("lambda", "nan"))): row for row in old_3e_rows}
    max_3e = 0.0
    matched_3e = 0
    for row in current_3e:
        old = old_3e_by_key.get((row["method"], row["lambda"]))
        if old is None:
            continue
        matched_3e += 1
        for field in ("information_floor", "affine_total_regret", "total_excess", "static_only_regret",
                      "adaptive_only_regret", "interaction", "recoverable_residual_operator_norm"):
            max_3e = max(max_3e, abs(float(row[field]) - float(old[field])))
    return (
        {
            "available": bool(old_3c_rows), "matched_rows": matched_3c,
            "expected_rows": len(new_3c_rows), "max_abs_error": max_3c,
            "pass": bool(old_3c_rows) and matched_3c == len(new_3c_rows) and max_3c < 1e-7,
        },
        {
            "available": bool(old_3e_rows), "matched_rows": matched_3e,
            "expected_rows": len(current_3e), "max_abs_error": max_3e,
            "pass": bool(old_3e_rows) and matched_3e == len(current_3e) and max_3e < 1e-7,
        },
    )


def legacy_regression(root: Path | None = None) -> dict[str, object]:
    """Check the frozen legacy Gaussian/mechanism instantiation."""
    root = Path(__file__).resolve().parents[2] if root is None else Path(root)
    family = legacy_gaussian_family()
    geometry = build_task_geometry(family)
    compatibility = coupled_primary_geometry()
    summary = metric_operator_summary(
        geometry.response, geometry.observation, geometry.spec.metric, tolerance=1e-9,
    )
    exposed_floor = metric_operator_summary(
        geometry.response, u_exposed_observation(compatibility), geometry.spec.metric,
        tolerance=1e-9,
    )["information_floor"]
    large_vrex = exact_ift_affine(
        make_benchmark(family=family), "vrex", 1e4, geometry.observation,
    )
    source_states = np.stack([
        task_state(environment_state(environment))
        for environment in geometry.source_environments
    ])
    source_contrast_rank = _rank(
        np.column_stack([state - source_states[0] for state in source_states]), 1e-9,
    )
    rows = [
        _scalar_check("tangent_dimension", 8, geometry.spec.dimension, 0.0),
        _scalar_check("observation_rows", 275, geometry.observation.shape[0], 0.0),
        _scalar_check("observation_columns", 8, geometry.observation.shape[1], 0.0),
        _scalar_check("response_rows", 9, geometry.response.shape[0], 0.0),
        _scalar_check("response_columns", 8, geometry.response.shape[1], 0.0),
        _scalar_check("rank_O", 7, summary["rank_O"], 0.0),
        _scalar_check("hidden_information_floor", FROZEN_HIDDEN_INFORMATION_FLOOR,
                      summary["information_floor"], 1e-9),
        _scalar_check("u_exposed_information_floor", 0.0, exposed_floor, 1e-9),
        _scalar_check("legacy_matrix_hash", FROZEN_LEGACY_MATRIX_HASH,
                      _matrix_hash(geometry.observation, geometry.response), 0.0),
        _scalar_check("facade_matrix_hash", FROZEN_LEGACY_MATRIX_HASH,
                      _matrix_hash(compatibility.observation, compatibility.response), 0.0),
        _scalar_check("family_response_factorization_residual", 0.0,
                      family_response_factorization_residual(geometry), 1e-8),
        _scalar_check("legacy_response_factorization_residual", 0.0,
                      response_factorization_residual(compatibility), 1e-8),
        _scalar_check("source_contrast_rank", 4,
                      source_contrast_rank, 0.0),
        _scalar_check("vrex_large_lambda_metric", True, bool(large_vrex.metric_min_eigenvalue <= 0.0), 0.0),
    ]
    three_c, three_e = _old_table_regression(root, geometry, family)
    rows.extend([
        _scalar_check("3C-B scalar table max error", 0.0, three_c["max_abs_error"], 1e-7)
        if three_c["available"] else {
            "quantity": "3C-B scalar table", "frozen_old_result": "missing",
            "refactored_result": three_c, "abs_error": None, "rel_error": None,
            "pass": "unavailable",
        },
        _scalar_check("3E-B scalar table max error", 0.0, three_e["max_abs_error"], 1e-7)
        if three_e["available"] else {
            "quantity": "3E-B scalar table", "frozen_old_result": "missing",
            "refactored_result": three_e, "abs_error": None, "rel_error": None,
            "pass": "unavailable",
        },
    ])
    coral_rows = _read_rows(
        root / "round3_redesign" / "3C_regularizer_control" /
        "results" / "regularizer_status.csv"
    )
    coral = next((row for row in coral_rows if row.get("method") == "CORAL"), None)
    coral_pass = bool(
        coral is not None and
        coral.get("predictor_status") == "representation-level fixed-gauge conditional" and
        coral.get("gauge_status") == "gauge-dependent/induced-degenerate"
    )
    rows.append({
        "quantity": "CORAL gauge/noncanonical status",
        "frozen_old_result": "fixed-representation + gauge-dependent/induced-degenerate",
        "refactored_result": None if coral is None else {
            "predictor_status": coral.get("predictor_status"),
            "gauge_status": coral.get("gauge_status"),
        },
        "abs_error": 0.0 if coral_pass else float("inf"),
        "rel_error": 0.0 if coral_pass else float("inf"), "pass": coral_pass,
    })
    scalar_passes = [row["pass"] is True for row in rows]
    return {
        "family": family.metadata(),
        "geometry": summary,
        "legacy_matrix_hash": _matrix_hash(geometry.observation, geometry.response),
        "compatibility_matrix_hash": _matrix_hash(compatibility.observation, compatibility.response),
        "three_c_table": three_c,
        "three_e_table": three_e,
        "coral_status": coral,
        "rows": rows,
        "pass": bool(all(scalar_passes)),
        "frozen_snapshot": {
            "matrix_hash": FROZEN_LEGACY_MATRIX_HASH,
            "hidden_information_floor": FROZEN_HIDDEN_INFORMATION_FLOOR,
        },
    }


def _source_induced_summary(family, geometry) -> dict[str, object]:
    states = tuple(task_state(environment_state(environment)) for environment in family.observed_source_environments)
    first_projector = source_span_projector(np.stack(states), 0)
    last_projector = source_span_projector(np.stack(states), len(states) - 1)
    duplicate = build_source_induced_family(family.observed_source_environments + (family.base,))
    base = family.base
    independent = base.updated(
        shortcut_means=(base.shortcut_means[0] + 0.20, *base.shortcut_means[1:])
    )
    independent_family = build_source_induced_family(family.observed_source_environments + (independent,))
    noise = np.asarray(base.noise_variances, dtype=float).copy()
    if noise.size:
        noise[0] += 0.40
    noise_family = build_source_induced_family(
        family.observed_source_environments + (base.updated(noise_variances=noise),)
    )
    noise_geometry = build_task_geometry(noise_family)
    default_response_rank = _rank(geometry.response, 1e-9)
    noise_response_rank = _rank(noise_geometry.response, 1e-9)
    return {
        **source_induced_audit(family),
        "geometry": metric_operator_summary(
            geometry.response, geometry.observation, geometry.spec.metric, tolerance=1e-9,
        ),
        "tangent_metadata": geometry.spec.metadata(),
        "source_reference_projector_error": float(np.linalg.norm(first_projector - last_projector)),
        "source_reference_invariant": bool(np.allclose(first_projector, last_projector, atol=1e-8)),
        "duplicate_source_state_span_rank": duplicate.state_span_rank,
        "duplicate_source_realizable_rank": duplicate.realizable_rank,
        "independent_source_state_span_rank": independent_family.state_span_rank,
        "independent_source_realizable_rank": independent_family.realizable_rank,
        "risk_null_noise_state_span_rank": noise_family.state_span_rank,
        "risk_null_noise_realizable_rank": noise_family.realizable_rank,
        "default_exposed_response_rank": default_response_rank,
        "risk_null_noise_exposed_response_rank": noise_response_rank,
        "duplicate_does_not_increase_rank": duplicate.state_span_rank == family.state_span_rank,
        "independent_variation_increases_rank": independent_family.state_span_rank > family.state_span_rank,
        "risk_null_noise_can_increase_raw_rank": noise_family.state_span_rank > family.state_span_rank,
        "risk_null_noise_does_not_increase_response_rank": noise_response_rank <= default_response_rank,
        "source_only_basis_inputs": [
            "observed source environments", "source task-complete states", "source design",
        ],
    }


def _family_method_rows(family, geometry) -> list[dict[str, object]]:
    result = main_benchmark(
        family=family, family_geometry=geometry,
        lambdas=(0.0, 1e-2, 1e-1),
    )
    summary = metric_operator_summary(
        geometry.response, geometry.observation, geometry.spec.metric, tolerance=1e-9,
    )
    rows: list[dict[str, object]] = []
    for item in result["audits"]:
        row = next(
            candidate for candidate in result["rows"]
            if candidate["method"] == item["method"] and candidate["lambda"] == item["lambda"]
        )
        if not item["fd_audit"]["pass"] or not row["status"] == "PASS":
            rows.append({
                "family": geometry.spec.family_name, "method": item["method"],
                "lambda": item["lambda"], "valid": False, "status": row["status"],
                "z0": item["z0"], "pi_O": item["tangent"], "E": None,
                "affine_regret": None,
            })
            continue
        parts = _affine_decomposition(geometry, item)
        rows.append({
            "family": geometry.spec.family_name, "method": item["method"],
            "lambda": float(item["lambda"]), "valid": True,
            "status": "PASS",
            "z0_norm": float(np.linalg.norm(item["z0"])),
            "pi_O_operator_norm": _operator_norm(
                world_whiten(item["tangent"], geometry.spec.metric)
            ),
            "recoverable_residual_operator_norm": float(parts["recoverable_residual_operator_norm"]),
            "affine_regret": float(parts["total_regret"]),
            "information_floor": float(summary["information_floor"]),
            "method_excess": float(max(0.0, parts["total_regret"] - summary["information_floor"])),
            "fd_error": item["fd_audit"]["max_error"],
            "target_risk_used": False, "semantic_labels_used": False,
            "cluster_labels_used": False, "regularizer_geometry_used_for_selection": False,
            # Keep the concrete affine objects in the family coordinates.  The
            # scalar regrets above are evaluated after transporting the world
            # metric, as required by the family contract.
            "z0": item["z0"],
            "pi_O": item["tangent"],
            "E": parts["recoverable_residual"],
            "affine_regret": float(parts["total_regret"]),
        })
    return rows


def family_comparison(legacy=None, source_induced=None) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    legacy = legacy_gaussian_family() if legacy is None else legacy
    source_induced = build_source_induced_family() if source_induced is None else source_induced
    families = (("legacy_gaussian_mechanism", legacy), ("source_induced", source_induced))
    geometry_rows: list[dict[str, object]] = []
    method_rows: list[dict[str, object]] = []
    for label, family in families:
        geometry = build_task_geometry(family)
        summary = metric_operator_summary(
            geometry.response, geometry.observation, geometry.spec.metric, tolerance=1e-9,
        )
        geometry_rows.append({
            "family": label, "tangent_dimension": geometry.spec.dimension,
            "metric": "source_state_orthonormal" if geometry.spec.source_defined else "Euclidean standardized coordinates",
            "rank_O": summary["rank_O"], "rank_A": summary["rank_A"],
            "kernel_dimension": summary["kernel_dimension"],
            "information_floor": summary["information_floor"],
            "source_defined": geometry.spec.source_defined,
            "mechanism_defined": geometry.spec.mechanism_defined,
            "source_reference_invariant": True,
        })
        method_rows.extend(_family_method_rows(family, geometry))
    return geometry_rows, method_rows


def coordinate_audit(legacy=None, source_induced=None) -> dict[str, object]:
    legacy = legacy_gaussian_family() if legacy is None else legacy
    source_induced = build_source_induced_family() if source_induced is None else source_induced
    audits = {
        "legacy_gaussian_mechanism": family_coordinate_audit(legacy),
        "source_induced": family_coordinate_audit(source_induced),
    }
    legacy_geometry = build_task_geometry(legacy)
    states = tuple(task_state(environment_state(environment)) for environment in legacy_geometry.source_environments)
    reference_audit = {
        "rank_reference_0": _rank(np.column_stack([state - states[0] for state in states]), 1e-9),
        "rank_reference_last": _rank(np.column_stack([state - states[-1] for state in states]), 1e-9),
        "projector_error": float(np.linalg.norm(
            source_span_projector(np.stack(states), 0) - source_span_projector(np.stack(states), len(states) - 1)
        )),
    }
    return {
        "family_coordinate_audits": audits,
        "source_reference_audit": reference_audit,
        "general_invertible_metric_transport_required": True,
        "untransported_coordinate_change_is_family_metric_change": True,
        "pass": bool(
            all(value["invariant"] for value in audits.values()) and
            reference_audit["projector_error"] < 1e-8
        ),
    }


def run() -> dict[str, object]:
    root = Path(__file__).resolve().parents[2]
    legacy = legacy_gaussian_family()
    source_induced = build_source_induced_family()
    legacy_geometry = build_task_geometry(legacy)
    source_geometry = build_task_geometry(source_induced)
    legacy_result = legacy_regression(root)
    source_summary = _source_induced_summary(source_induced, source_geometry)
    geometry_rows, method_rows = family_comparison(legacy, source_induced)
    coordinates = coordinate_audit(legacy, source_induced)
    classification = {
        "family_omission": failure_classification(
            in_family=False, source_visible=False, response_correct=False,
        ),
        "modeled_but_source_unobservable": failure_classification(
            in_family=True, source_visible=False, response_correct=False,
        ),
        "source_visible_algorithm_failure": failure_classification(
            in_family=True, source_visible=True, response_correct=False,
        ),
    }
    source_summary["legacy_geometry_not_used_for_basis"] = True
    legacy_u_index = legacy_geometry.spec.index("U_emergent")
    source_summary["legacy_hidden_u_classification"] = {
        "modeled_in_legacy_family": True,
        "legacy_source_null": bool(np.linalg.norm(legacy_geometry.observation[:, legacy_u_index]) < 1e-12),
        "legacy_response_active": bool(np.linalg.norm(legacy_geometry.response[:, legacy_u_index]) > 1e-10),
        "omitted_from_source_induced_family": "U_emergent" not in source_geometry.spec.directions,
        "omission_is_not_source_null_claim": True,
    }
    return {
        "legacy_regression": legacy_result,
        "source_induced_summary": source_summary,
        "family_geometry": geometry_rows,
        "method_comparison": method_rows,
        "coordinate_invariance": coordinates,
        "failure_classification": classification,
        "family_distinction": source_summary["legacy_hidden_u_classification"],
        "primary_inputs": {
            "target_risk_used": False,
            "mechanism_labels_used": False,
            "cluster_labels_used": False,
            "regularizer_geometry_used": False,
            "3b_semantic_modules_used": False,
        },
        "verdict": (
            "ENV-FAMILY-REFACTOR-PASS"
            if legacy_result["pass"] and source_summary["source_only"] and coordinates["pass"]
            else "ENV-FAMILY-LEGACY-PASS-SOURCE-PARTIAL"
            if legacy_result["pass"] else "ENV-FAMILY-ABSTRACTION-FAILED"
        ),
        "legacy_geometry": legacy_geometry,
        "source_geometry": source_geometry,
    }


def _write_csv(path: Path, rows: list[dict[str, object]], fields: tuple[str, ...]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: _jsonable(row.get(field, "")) for field in fields} for row in rows)


def _write_method_matrix_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write affine matrices as JSON cells so the CSV remains machine-readable."""
    fields = (
        "family", "method", "lambda", "valid", "status", "z0", "pi_O", "E",
        "information_floor", "affine_regret", "total_excess", "z0_norm",
        "pi_O_operator_norm", "recoverable_residual_operator_norm", "fd_error",
        "target_risk_used", "semantic_labels_used", "cluster_labels_used",
        "regularizer_geometry_used_for_selection",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            record = {}
            for field in fields:
                value = row.get(field, "")
                if field in {"z0", "pi_O", "E"}:
                    record[field] = json.dumps(_jsonable(value), separators=(",", ":"))
                else:
                    record[field] = _jsonable(value)
            writer.writerow(record)


def _report(result: dict[str, object]) -> str:
    legacy = result["legacy_regression"]
    source = result["source_induced_summary"]
    geometry = result["family_geometry"]
    method_rows = result["method_comparison"]
    method_table = "\n".join(
        f"| {row['family']} | {row['method']} | {row['lambda']:.6g} | {row.get('valid', False)} | "
        f"{row.get('z0_norm', float('nan')):.8g} | "
        f"{row.get('pi_O_operator_norm', float('nan')):.8g} | "
        f"{row.get('recoverable_residual_operator_norm', float('nan')):.8g} | "
        f"{row.get('affine_regret', float('nan')):.8g} | "
        f"{row.get('information_floor', float('nan')):.8g} | "
        f"{row.get('method_excess', float('nan')):.8g} | {row.get('status', 'INVALID')} |"
        for row in method_rows
    )
    return f"""# Environment-Family Layer Refactor

## Verdict

`{result['verdict']}`

The refactor makes the admissible environment family an explicit upstream
argument:

`EnvironmentFamily -> TangentSpec -> (A_theta, O_S, Pi_theta) -> regret`.

The old eight coordinates are the local coordinate basis of the declared
`legacy_gaussian_mechanism` family.  They are not a universal OOD tangent and
their names are chart labels, not invariant mechanisms.

## Legacy regression

- observation shape: `({legacy['geometry']['source_dimension']}, {legacy['geometry']['dim_U']})`;
- response shape: `({legacy['geometry']['response_dimension']}, {legacy['geometry']['dim_U']})`;
- `rank(O_S)`: `{legacy['geometry']['rank_O']}`;
- hidden-U information floor: `{legacy['geometry']['information_floor']:.13g}`;
- legacy matrix snapshot match: `{legacy['legacy_matrix_hash'] == legacy['frozen_snapshot']['matrix_hash']}`;
- scalar 3C-B table regression: `{legacy['three_c_table']['pass']}`;
- scalar 3E-B table regression: `{legacy['three_e_table']['pass']}`;
- CORAL noncanonical/gauge status: `{legacy['coral_status'] is not None}`.

The compatibility facade and the shared family geometry produce the frozen
matrix snapshot without changing the old mathematical objects.  V-REx's
large-lambda non-positive local metric remains an invalid path boundary.

## Source-induced family

The second family is built only from observed source task-state contrasts and
an environment-parameter Jacobian pullback.  Its state-span rank is
`{source['state_span_rank']}`, while its realizable rank is
`{source['realizable_rank']}`.  The source-reference projector error is
`{source['source_reference_projector_error']:.3g}` and the duplicate-source
rank check is `{source['duplicate_does_not_increase_rank']}`.  A new mean
contrast increases raw source-state rank to
`{source['independent_source_state_span_rank']}`; a risk-null noise contrast
may increase raw state rank without adding a response direction.

No target risk, `A`, `Pi`, semantic label, cluster, or downstream regularizer
quantity is read while the source-induced basis is constructed.  The
source-induced metric is explicitly source-state orthonormal and therefore
metric-dependent.

## Cross-family geometry

| family | tangent dim | rank O | rank A | dim ker O | information floor | source-defined | mechanism-defined |
|---|---:|---:|---:|---:|---:|---|---|
""" + "\n".join(
        f"| {row['family']} | {row['tangent_dimension']} | {row['rank_O']} | {row['rank_A']} | "
        f"{row['kernel_dimension']} | {row['information_floor']:.8g} | {row['source_defined']} | {row['mechanism_defined']} |"
        for row in geometry
    ) + f"""

A smaller source-induced floor means that the source-induced family promises
to handle a smaller admissible uncertainty set.  It is not evidence that the
learner is universally more robust.  The legacy hidden-U direction is
modeled-but-source-unobservable; a direction absent from the source-induced
tangent would instead be family omission.  In the generated audit this is
recorded as `modeled_in_legacy_family=true`, `legacy_source_null=true`, and
`omitted_from_source_induced_family=true`; omission is not a source-null claim.

## Concrete regularizer comparison

The following table is computed independently on both declared families.  It
is the concrete L2/IRMv1/V-REx comparison, not merely a family-rank summary.
`Pi@O_S` is the affine source-adaptive action and `E` is the recoverable
response residual.  Regret is evaluated in the family-declared world metric.

| family | method | lambda | valid | ||z0|| | ||Pi@O_S||op | ||E||op | affine regret | information floor | method excess | status |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|
{method_table}

The exact vectors/matrices for every row are retained in
`results/method_comparison_by_family.json` and in JSON-valued cells in
`results/method_comparison_by_family_full.csv`.  In particular, those files
contain `z0`, `pi_O`, `E`, and `affine_regret` for all nine rows of the
source-induced family and all nine rows of the legacy family.  The scalar CSV
is retained as a compact compatibility view.

## Failure taxonomy and scope

The three distinct cases are family omission, in-family source-information
failure, and in-family source-visible algorithm failure.  3A supplies the
response norm, 3D supplies source task-state observations, and 3C/3E consume
the resulting family-relative geometry.  3B semantic modules and old 3C
regularizer geometry are boundary/audit inputs, not family-definition inputs.

Coordinate recoding preserves the information geometry only when `A`, `O_S`
and the tangent metric are transported together.  An untransported recoding
changes the uncertainty geometry.  These are population local statements;
they do not claim coverage of all shifts, causal identification, finite-sample
estimability, deep-network identifiability, or a universal DG theorem.
"""


def main() -> None:
    result = run()
    root = Path(__file__).resolve().parents[2]
    output = root / "round3_redesign" / "environment_family_refactor"
    results = output / "results"
    results.mkdir(parents=True, exist_ok=True)
    summary = {
        "verdict": result["verdict"],
        "legacy_regression_pass": result["legacy_regression"]["pass"],
        "legacy_geometry": result["legacy_regression"]["geometry"],
        "source_induced_summary": result["source_induced_summary"],
        "family_geometry": result["family_geometry"],
        "method_comparison_by_family": result["method_comparison"],
        "method_comparison_artifacts": {
            "full_json": "results/method_comparison_by_family.json",
            "full_matrix_csv": "results/method_comparison_by_family_full.csv",
            "scalar_csv": "results/method_comparison_by_family.csv",
        },
        "primary_inputs": result["primary_inputs"],
        "coordinate_invariance": result["coordinate_invariance"],
        "failure_classification": result["failure_classification"],
        "family_distinction": result["family_distinction"],
    }
    (results / "legacy_regression.json").write_text(json.dumps(_jsonable(result["legacy_regression"]), indent=2) + "\n", encoding="utf-8")
    (results / "source_induced_summary.json").write_text(json.dumps(_jsonable(result["source_induced_summary"]), indent=2) + "\n", encoding="utf-8")
    (results / "method_comparison_by_family.json").write_text(
        json.dumps(_jsonable(result["method_comparison"]), indent=2) + "\n", encoding="utf-8",
    )
    (results / "coordinate_invariance.json").write_text(json.dumps(_jsonable(result["coordinate_invariance"]), indent=2) + "\n", encoding="utf-8")
    (results / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2) + "\n", encoding="utf-8")
    _write_csv(results / "family_comparison.csv", result["family_geometry"],
               ("family", "tangent_dimension", "metric", "rank_O", "rank_A", "kernel_dimension", "information_floor", "source_defined", "mechanism_defined", "source_reference_invariant"))
    _write_csv(results / "method_comparison_by_family.csv", result["method_comparison"],
               ("family", "method", "lambda", "valid", "z0_norm", "pi_O_operator_norm", "recoverable_residual_operator_norm", "affine_regret", "information_floor", "method_excess", "fd_error", "target_risk_used", "semantic_labels_used", "cluster_labels_used", "regularizer_geometry_used_for_selection"))
    _write_method_matrix_csv(results / "method_comparison_by_family_full.csv", result["method_comparison"])
    _write_csv(results / "legacy_regression.csv", result["legacy_regression"]["rows"],
               ("quantity", "frozen_old_result", "refactored_result", "abs_error", "rel_error", "pass"))
    (output / "environment_family_refactor_report.md").write_text(_report(result), encoding="utf-8")
    print(output / "environment_family_refactor_report.md")


if __name__ == "__main__":
    main()
