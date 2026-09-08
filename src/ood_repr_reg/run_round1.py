"""Run the bounded population sanity checks for round one."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .intervention_linear import (
    GaussianEnvironment,
    LinearGaussianSCM,
    affine_erm,
    irmv1_scale_penalty,
    ridge_solution,
    risk,
    scalar_irmv1_zero_candidates,
    source_risk,
)
from .round1_regularizers import coral_state_response, l2_state_response, regularizer_operator
from .round1_state import (
    component_transport,
    componentwise_bound,
    environment_state,
    exposure_identifiability,
    risk_pairing,
    risk_quotient_basis,
    risk_state,
    source_environment_moments,
    source_observation_operator,
)


def default_scm() -> LinearGaussianSCM:
    return LinearGaussianSCM(
        loading=np.array([[1.0]]),
        beta=np.array([1.0]),
        sigma_xi=np.array([[0.25]]),
        sigma_y=0.1,
    )


def environment(relation: float, *, mean: float = 0.0, variance: float = 0.5, name: str = "env") -> GaussianEnvironment:
    return GaussianEnvironment(np.array([[relation]]), np.array([mean]), np.array([[variance]]), name)


def _array_values(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, tuple):
        return [_array_values(item) for item in value]
    if isinstance(value, dict):
        return {key: _array_values(item) for key, item in value.items()}
    return value


def run() -> dict[str, object]:
    scm = default_scm()
    sources_two = (environment(0.7, variance=0.02, name="r_plus"), environment(-0.1, variance=0.02, name="r_minus"))
    sources_three = tuple(environment(value, name=f"r_{value}") for value in (-0.6, 0.1, 0.8))
    target_relation = environment(-1.0, mean=0.4, variance=0.9, name="target_compound")

    erm_two = affine_erm(scm, sources_two)
    erm_three = affine_erm(scm, sources_three)
    irm_blind = scalar_irmv1_zero_candidates(scm, sources_two)[-1]
    ridge = np.asarray(ridge_solution(scm, sources_three, 0.5))
    identity = np.eye(2)

    source_moments = source_environment_moments(scm, sources_three)
    target_moment = environment_state(scm, target_relation)
    quotient = risk_quotient_basis(source_moments + (target_moment,))
    erm_state = risk_state(scm, erm_three, sources_three[0])
    projected_quadratic = (quotient @ (quotient.T @ erm_state.quadratic.reshape(-1))).reshape(erm_state.quadratic.shape)
    projected = risk_pairing(projected_quadratic, target_moment)
    full = risk(scm, target_relation, erm_three) - scm.sigma_y**2
    source_operator = source_observation_operator(source_moments)
    exposure = exposure_identifiability(source_operator, (target_moment,))
    projectors = (np.diag([1.0, 0.0, 0.0, 0.0]), np.diag([0.0, 1.0, 0.0, 0.0]), np.diag([0.0, 0.0, 1.0, 0.0]), np.diag([0.0, 0.0, 0.0, 1.0]))
    diagonal, interactions = component_transport(erm_state.vector, target_moment - source_moments[0], projectors)

    coral = coral_state_response(scm, sources_three, identity)
    records = {
        "state": {
            "source_moment_rank": int(quotient.shape[1]),
            "ambient_state_dimension": int(erm_state.quadratic.size),
            "target_transport_pairing": float(projected - risk(scm, sources_three[0], erm_three) + scm.sigma_y**2),
            "full_target_excess_over_noise": float(full),
            "quotient_projection_residual": float(abs(projected - full)),
        },
        "exposure": {
            "source_observation_rank": exposure.source_rank,
            "source_observation_dimension": exposure.source_dimension,
            "source_kernel_dimension": exposure.source_kernel_dimension,
            "target_alignment": exposure.target_alignment,
            "target_blind_norm": exposure.target_blind_norm,
        },
        "methods": {
            "ERM_two_source": {
                "weights": erm_two,
                "source_risk": source_risk(scm, sources_two, erm_two),
                "target_risk": risk(scm, target_relation, erm_two),
                "irmv1_penalty": irmv1_scale_penalty(scm, sources_two, erm_two),
            },
            "IRMv1_two_source_blind": {
                "weights": irm_blind,
                "source_risk": source_risk(scm, sources_two, irm_blind),
                "target_risk": risk(scm, target_relation, irm_blind),
                "irmv1_penalty": irmv1_scale_penalty(scm, sources_two, irm_blind),
            },
            "ERM_three_source": {
                "weights": erm_three,
                "source_risk": source_risk(scm, sources_three, erm_three),
                "target_risk": risk(scm, target_relation, erm_three),
                "irmv1_penalty": irmv1_scale_penalty(scm, sources_three, erm_three),
                "irmv1_operator_rank": regularizer_operator("irmv1", scm, sources_three).rank,
            },
            "L2_ridge": {
                "weights": ridge,
                "source_risk": source_risk(scm, sources_three, ridge),
                "target_risk": risk(scm, target_relation, ridge),
                **l2_state_response(ridge),
            },
            "CORAL_identity_representation": {
                "source_covariance_penalty": coral["penalty"],
                "operator_rank": regularizer_operator("coral", scm, sources_three, identity).rank,
                "observed_object": coral["observed_object"],
                "blind_candidate": coral["blind_candidate"],
            },
        },
        "component_accounting": {
            "diagonal": diagonal,
            "interactions": interactions,
            "total": float(diagonal.sum() + interactions.sum()),
            "bound": componentwise_bound(erm_state.vector, target_moment - source_moments[0], projectors),
        },
    }
    return _array_values(records)


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    output = root / "round1" / "results" / "round1_results.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(run(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
