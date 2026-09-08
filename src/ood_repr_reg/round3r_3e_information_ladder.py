"""Nested source-information designs for 3E."""

from __future__ import annotations

import numpy as np

from .round3r_3e_recovery import operator_summary
from .round3r_3e_world_tangent import CoupledTangentGeometry, source_observation_block

Array = np.ndarray


def information_ladder() -> tuple[dict[str, object], ...]:
    a = np.diag([1.0, 2.0, 5.0])
    observations = (
        ("no_information", np.zeros((0, 3))),
        ("first_observation", np.array([[1.0, 0.0, 0.0]])),
        ("duplicate_first_observation", np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])),
        ("second_observation", np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])),
        ("task_relevant_third_observation", np.eye(3)),
    )
    rows: list[dict[str, object]] = []
    previous = None
    for name, o in observations:
        summary = operator_summary(a, o)
        row = {"design": name, **{key: summary[key] for key in (
            "dim_U", "rank_O", "kernel_dimension", "rank_A",
            "structural_ambiguity_dimension", "alpha", "normalized_alpha",
            "ambiguity_diameter")}}
        row["change_from_previous"] = None if previous is None else float(summary["alpha"] - previous)
        rows.append(row)
        previous = float(summary["alpha"])
    return tuple(rows)


def monotonicity_audit(rows: tuple[dict[str, object], ...] | list[dict[str, object]],
                       tolerance: float = 1e-10) -> dict[str, object]:
    alpha = np.asarray([float(row["alpha"]) for row in rows])
    deltas = np.diff(alpha)
    return {"alpha": alpha, "nonincreasing": bool(np.all(deltas <= tolerance)),
            "strict_decreases": int(np.sum(deltas < -tolerance)), "tolerance": tolerance}


def coupled_information_ladder(geometry: CoupledTangentGeometry) -> tuple[dict[str, object], ...]:
    """A genuinely nested 3A/3D source-observation sequence with fixed A."""
    base = geometry.base
    spec = geometry.spec
    relation = list(base.shortcut_rhos)
    relation[0] += 0.12
    noise = np.asarray(base.noise_variances, dtype=float).copy()
    noise[0] += 0.70
    blocks = (
        ("base_source", source_observation_block((base,), spec, expose_u_at_source=False), 1,
         "one task-complete source observation"),
        ("duplicate_source", source_observation_block((base,), spec, expose_u_at_source=False), 1,
         "exact duplicate of the base source observation"),
        ("risk_null_noise_source", source_observation_block((base.updated(noise_variances=noise),), spec, expose_u_at_source=False), 1,
         "additional independent-noise source state"),
        ("relation_source", source_observation_block((base.updated(shortcut_rhos=tuple(relation)),), spec, expose_u_at_source=False), 1,
         "additional shortcut-relation source state"),
        ("u_exposed_source", source_observation_block((base.updated(u_gamma=0.45),), spec, expose_u_at_source=True), 1,
         "new source observation that exposes the previously hidden U tangent"),
    )
    stacked: list[Array] = []
    rows: list[dict[str, object]] = []
    previous = None
    source_count = 0
    for name, block, count, note in blocks:
        stacked.append(block)
        source_count += count
        summary = operator_summary(geometry.response, np.vstack(stacked))
        rows.append({
            "design": name,
            "note": note,
            "source_observation_count": source_count,
            "source_information_design": "stacked_task_complete_states",
            **{key: summary[key] for key in (
                "dim_U", "rank_O", "kernel_dimension", "rank_A",
                "structural_ambiguity_dimension", "alpha", "normalized_alpha",
                "ambiguity_diameter")},
            "change_from_previous": None if previous is None else float(summary["alpha"] - previous),
        })
        previous = float(summary["alpha"])
    full = operator_summary(geometry.response, np.eye(geometry.spec.dimension))
    rows.append({
        "design": "full_information_mathematical_endpoint",
        "note": "explicit identity observation; not a claim about an obtainable source design",
        "source_observation_count": None,
        "source_information_design": "mathematical_endpoint",
        **{key: full[key] for key in (
            "dim_U", "rank_O", "kernel_dimension", "rank_A",
            "structural_ambiguity_dimension", "alpha", "normalized_alpha",
            "ambiguity_diameter")},
        "change_from_previous": float(full["alpha"] - previous),
    })
    return tuple(rows)


def same_source_count_geometry(geometry: CoupledTangentGeometry) -> dict[str, object]:
    """Two two-source designs differing only in whether U is source-visible."""
    base_block = source_observation_block((geometry.base,), geometry.spec, expose_u_at_source=False)
    duplicate_block = source_observation_block((geometry.base,), geometry.spec, expose_u_at_source=False)
    u_block = source_observation_block((geometry.base.updated(u_gamma=0.45),), geometry.spec, expose_u_at_source=True)
    weak = operator_summary(geometry.response, np.vstack((base_block, duplicate_block)))
    aligned = operator_summary(geometry.response, np.vstack((base_block, u_block)))
    return {
        "source_count": 2,
        "duplicate_geometry": weak,
        "u_exposed_geometry": aligned,
        "same_source_count": True,
        "different_alpha": bool(not np.isclose(weak["alpha"], aligned["alpha"], atol=1e-10)),
        "aligned_observation_reduces_alpha": bool(aligned["alpha"] < weak["alpha"] - 1e-10),
    }


__all__ = [
    "information_ladder", "monotonicity_audit", "coupled_information_ladder",
    "same_source_count_geometry",
]
