"""Run the bounded population calculations for the second research round."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .intervention_linear import (
    GaussianEnvironment,
    LinearGaussianSCM,
    affine_erm,
    irmv1_scale_penalty,
    risk,
    scalar_irmv1_zero_candidates,
    source_risk,
)
from .round1_state import environment_state
from .round2_induced_cost import (
    coral_induced_cost,
    irm_relation_operator,
    irmv1_induced_cost,
    l2_induced_cost,
)
from .round2_state import (
    ambiguity_support,
    quotient_coordinates,
    recover_target_moment_from_sources,
    risk_visible_basis,
    source_observation_map,
    source_target_ambiguity,
    symmetric_vectorize,
)


def default_scm() -> LinearGaussianSCM:
    return LinearGaussianSCM(np.array([[1.0]]), np.array([1.0]), np.array([[0.25]]), 0.1)


def env(r: float, *, mean: float = 0.0, variance: float = 0.5, name: str = "env") -> GaussianEnvironment:
    return GaussianEnvironment(np.array([[r]]), np.array([mean]), np.array([[variance]]), name)


def _direct_covariance(environment: GaussianEnvironment) -> np.ndarray:
    relation = float(environment.relation[0, 0])
    return np.array([[1.0, relation], [relation, relation * relation + float(environment.sigma_a[0, 0])]])


def _jsonable(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    return value


def run() -> dict[str, object]:
    scm = default_scm()
    sources = tuple(env(r, name=f"source_{r}") for r in (-0.6, 0.1, 0.8))
    target_family = sources + (env(-1.0, mean=0.5, variance=0.9, name="target_unseen"),)
    source_moments = tuple(environment_state(scm, item) for item in sources)
    target_moments = tuple(environment_state(scm, item) for item in target_family)
    quotient = risk_visible_basis(target_moments)
    ambiguity = source_target_ambiguity(source_moments, target_moments)
    source_map = source_observation_map(source_moments)
    target_recovery = recover_target_moment_from_sources(source_moments, target_moments[-1])
    shift_space = np.stack([symmetric_vectorize(moment - source_moments[0]) for moment in target_moments[1:]])
    state_difference = shift_space[0].copy()

    two_sources = (env(0.7, variance=0.02, name="blind_plus"), env(-0.1, variance=0.02, name="blind_minus"))
    irm_blind = scalar_irmv1_zero_candidates(scm, two_sources)[-1]
    erm = affine_erm(scm, sources)
    direct_q = np.array([0.7 - 1.0, 0.3])
    direct_beta = np.array([1.0, 0.0])
    l2 = l2_induced_cost(direct_q, direct_beta)
    covariances = tuple(_direct_covariance(item) for item in sources)
    coral = coral_induced_cost(direct_q, covariances)
    irm = irmv1_induced_cost(scm, sources, np.array([0.0, 0.7, 0.3]))
    rows = [
        {"method": "L2", "induced_cost": l2.value, "status": l2.status, "direct_descent": l2.direct_descent},
        {"method": "CORAL", "induced_cost": coral.value, "status": coral.status, "direct_descent": coral.direct_descent},
        {"method": "IRMv1", "induced_cost": irm.value, "status": irm.status, "direct_descent": irm.direct_descent},
    ]
    return _jsonable({
        "ambiguity": {
            "target_visible_dimension": ambiguity.target_rank,
            "source_rank_in_quotient": ambiguity.source_rank_in_quotient,
            "ambiguity_dimension": ambiguity.ambiguity_dimension,
            "ambient_symmetric_dimension": quotient.ambient_dimension,
            "raw_source_kernel_dimension": ambiguity.source_kernel_dimension,
            "target_recovery_residual": target_recovery.residual_norm,
            "target_recovery_exact": target_recovery.exact,
            "support_example": ambiguity_support(state_difference, shift_space, 0.5),
        },
        "irmv1": {
            "two_source_penalty": irmv1_scale_penalty(scm, two_sources, irm_blind),
            "two_source_source_risk": source_risk(scm, two_sources, irm_blind),
            "two_source_target_risk": risk(scm, target_family[-1], irm_blind),
            "three_source_design_rank": int(np.linalg.matrix_rank(irm_relation_operator(sources))),
            "three_source_erm_penalty": irmv1_scale_penalty(scm, sources, erm),
        },
        "induced_costs": rows,
        "source_observation": {
            "absolute_rank": source_map.rank,
            "absolute_kernel_dimension": source_map.kernel_basis.shape[1],
            "target_coordinates": quotient_coordinates(target_moments[-1], quotient),
        },
    })


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    output_dir = root / "round2" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    result = run()
    (output_dir / "round2_results.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (output_dir / "ambiguity_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("metric", "value"))
        for key, value in result["ambiguity"].items():
            writer.writerow((key, value))
    with (output_dir / "induced_cost_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("method", "induced_cost", "status", "direct_descent"))
        writer.writeheader()
        writer.writerows(result["induced_costs"])
    print(output_dir / "round2_results.json")


if __name__ == "__main__":
    main()
