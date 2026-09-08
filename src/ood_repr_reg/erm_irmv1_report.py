"""Emit reproducible population diagnostics for the ERM--IRMv1 mechanism audit."""

from __future__ import annotations

import json

import numpy as np

from .intervention_linear import (
    GaussianEnvironment,
    LinearGaussianSCM,
    affine_erm,
    causal_oracle_risk,
    erm_stationarity_residual,
    irmv1_radial_response,
    irmv1_scale_penalty,
    nuisance_transport_bound,
    risk,
    risk_transport_components,
    scalar_irmv1_nuisance_squared_bound,
    scalar_relation_response,
    source_risk,
)


def _environment(relation: float, mean: float = 0.0, variance: float = 0.5) -> GaussianEnvironment:
    return GaussianEnvironment(
        relation=np.array([[relation]]),
        mean=np.array([mean]),
        sigma_a=np.array([[variance]]),
    )


def _three_relation_positive_control() -> dict[str, object]:
    scm = LinearGaussianSCM(
        loading=np.array([[1.0]]),
        beta=np.array([1.0]),
        sigma_xi=np.array([[0.25]]),
        sigma_y=0.1,
    )
    sources = tuple(_environment(relation) for relation in (-0.6, 0.1, 0.8))
    u_only = np.array([0.0, 1.0 / 1.25, 0.0])
    target = _environment(-1.4, mean=0.7, variance=1.2)
    response = scalar_relation_response(scm, sources, u_only)
    components = risk_transport_components(scm, sources[0], target, u_only)
    return {
        "source_relations": [-0.6, 0.1, 0.8],
        "relation_design_sigma_min": response.singular_value,
        "u_only_predictor": u_only.tolist(),
        "source_risk": source_risk(scm, sources, u_only),
        "zero_predictor_source_risk": source_risk(scm, sources, np.zeros(3)),
        "irmv1_penalty": irmv1_scale_penalty(scm, sources, u_only),
        "nuisance_squared_bound": scalar_irmv1_nuisance_squared_bound(scm, sources, u_only),
        "target_causal_excess": risk(scm, target, u_only) - causal_oracle_risk(scm),
        "transport_components": {
            "relation": components.relation,
            "mean": components.mean,
            "covariance": components.covariance,
            "total": components.total,
        },
    }


def _finite_penalty_bridge() -> dict[str, object]:
    scm = LinearGaussianSCM(
        loading=np.array([[1.0]]),
        beta=np.array([1.0]),
        sigma_xi=np.array([[0.25]]),
        sigma_y=0.1,
    )
    sources = tuple(_environment(relation) for relation in (-0.6, 0.1, 0.8))
    target = _environment(-1.1, mean=0.5, variance=1.0)
    w = np.array([0.0, 0.45, 0.35])
    squared_bound = scalar_irmv1_nuisance_squared_bound(scm, sources, w)
    transport_bound = nuisance_transport_bound(scm, sources[0], target, w, np.sqrt(squared_bound))
    actual_transport = risk(scm, target, w) - risk(scm, sources[0], w)
    return {
        "predictor": w.tolist(),
        "irmv1_penalty": irmv1_scale_penalty(scm, sources, w),
        "actual_nuisance_squared": float(w[-1] ** 2),
        "nuisance_squared_bound": squared_bound,
        "actual_transport": actual_transport,
        "absolute_transport_bound": transport_bound,
    }


def _erm_mixture_blindness() -> dict[str, object]:
    scm = LinearGaussianSCM(
        loading=np.array([[1.0]]),
        beta=np.array([1.0]),
        sigma_xi=np.array([[0.25]]),
        sigma_y=0.1,
    )
    sources = (_environment(0.8), _environment(0.3))
    erm = affine_erm(scm, sources)
    response = irmv1_radial_response(scm, sources, erm)
    return {
        "source_relations": [0.8, 0.3],
        "erm_predictor": erm.tolist(),
        "mixture_stationarity_residual": erm_stationarity_residual(scm, sources, erm),
        "irmv1_penalty_at_erm": response.penalty,
        "per_environment_scale_derivatives": response.derivatives.tolist(),
        "per_environment_tangential_gradient_norms": response.tangential_gradient_norms.tolist(),
    }


def _erm_symmetric_positive_control() -> dict[str, object]:
    scm = LinearGaussianSCM(
        loading=np.array([[1.0]]),
        beta=np.array([1.0]),
        sigma_xi=np.array([[0.25]]),
        sigma_y=0.1,
    )
    sources = (_environment(-0.8), _environment(0.8))
    target = _environment(-1.4, mean=0.7, variance=1.2)
    erm = affine_erm(scm, sources)
    return {
        "source_relations": [-0.8, 0.8],
        "erm_predictor": erm.tolist(),
        "effective_nuisance_coefficient": float(erm[-1]),
        "source_risk": source_risk(scm, sources, erm),
        "target_risk": risk(scm, target, erm),
        "target_transport": risk(scm, target, erm) - source_risk(scm, sources, erm),
    }


def report() -> dict[str, object]:
    return {
        "claim": "C010-ERM-IRMv1",
        "erm_mixture_blindness": _erm_mixture_blindness(),
        "erm_symmetric_positive_control": _erm_symmetric_positive_control(),
        "three_relation_positive_control": _three_relation_positive_control(),
        "finite_penalty_bridge": _finite_penalty_bridge(),
    }


def main() -> None:
    print(json.dumps(report(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
