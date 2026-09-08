"""Recovery audit on the two abstract 3D world families."""

from __future__ import annotations

import numpy as np

from .round3r_3e_recovery import operator_summary
from .round3r_3b_benchmark import ModuleEnvironment, environment_state, source_design, source_optimum
from .round3r_3d_exposure import response_operator
from .round3r_3e_world_tangent import coupled_primary_geometry


def round3d_recovery() -> dict[str, object]:
    identifiable_o = np.eye(2)
    identifiable_a = np.array([[1.0, 0.5], [-0.25, 1.0]])
    hidden_o = np.zeros((1, 2))
    hidden_a = np.array([[1.0, -1.0]])
    result = {
        "identifiable_linear_world_family": operator_summary(identifiable_a, identifiable_o),
        "hidden_emergent_world_family": operator_summary(hidden_a, hidden_o),
    }

    # Independently reconstruct the finite Gaussian source-identical pair.
    base = ModuleEnvironment(n_noise=2)
    source_template = source_design(base, relation_exposed=True)
    # These are separate world records.  The source-side U loading is fixed at
    # zero in both, while only the target U--Y coupling has opposite signs.
    plus_source_worlds = tuple(environment.updated(u_gamma=0.0) for environment in source_template)
    minus_source_worlds = tuple(environment.updated(u_gamma=0.0) for environment in source_template)
    optimum, source = source_optimum(plus_source_worlds)
    plus_environment = base.updated(u_gamma=0.45)
    minus_environment = base.updated(u_gamma=-0.45)
    plus = response_operator(source, optimum, environment_state(plus_environment))
    minus = response_operator(source, optimum, environment_state(minus_environment))
    plus_source_states = tuple(environment_state(environment) for environment in plus_source_worlds)
    minus_source_states = tuple(environment_state(environment) for environment in minus_source_worlds)
    source_equal = all(
        np.array_equal(left.second, right.second)
        and np.array_equal(left.xy, right.xy)
        and left.y2 == right.y2
        for left, right in zip(plus_source_states, minus_source_states)
    )
    # The pair is embedded as +/- unit perturbations in a one-dimensional
    # hidden world coordinate, so its induced response map is exact and
    # extremal under that declared normalization.
    pair_difference = plus - minus
    pair_response = (pair_difference / 2.0)[:, None]
    pair_observation = np.zeros((0, 1))
    pair_summary = operator_summary(pair_response, pair_observation)
    pair_ratio = float(np.linalg.norm(pair_difference) / (2.0 * pair_summary["alpha"])) if pair_summary["alpha"] else 0.0
    result["source_identical_target_different_pair"] = {
        "source_states_identical_exactly": bool(source_equal),
        "source_state_equality_checked_between_distinct_world_records": True,
        "source_u_gamma_fixed_to_zero_in_both_worlds": True,
        "q_plus": plus,
        "q_minus": minus,
        "q_difference_norm": float(np.linalg.norm(pair_difference)),
        "pair_embedding": pair_summary,
        "pair_ratio_to_2alpha": pair_ratio,
        "pair_is_extremal_under_declared_embedding": bool(np.isclose(pair_ratio, 1.0)),
    }
    coupled = coupled_primary_geometry()
    u_index = coupled.spec.index("U_emergent")
    result["coupled_3a_3d_primary"] = {
        "world_metric": coupled.spec.metadata(),
        "summary": operator_summary(coupled.response, coupled.observation),
        "source_u_column_norm": float(np.linalg.norm(coupled.observation[:, u_index])),
        "response_u_column_norm": float(np.linalg.norm(coupled.response[:, u_index])),
        "u_source_null": bool(np.linalg.norm(coupled.observation[:, u_index]) < 1e-12),
        "u_response_active": bool(np.linalg.norm(coupled.response[:, u_index]) > 1e-10),
        "source_observation_uses_target_risk": False,
        "source_observation_uses_labels": False,
        "source_observation_uses_clusters": False,
        "source_observation_uses_regularizer_geometry": False,
    }
    return result


__all__ = ["round3d_recovery"]
