"""Emit reproducible population diagnostics for the first intervention claims."""

from __future__ import annotations

import json

import numpy as np

from .intervention_linear import (
    GaussianEnvironment,
    LinearGaussianSCM,
    affine_erm,
    causal_oracle_risk,
    correlation_robust_risk,
    gradient_alignment_penalty,
    irmv1_scale_penalty,
    nuisance_observability,
    observational_oracle_risk,
    risk,
    scalar_irmv1_zero_candidates,
    source_risk,
)


def fixed_irmv1_counterexample_report() -> dict[str, object]:
    """The exact C002-IRM population counterexample used in the research ledger."""
    scm = LinearGaussianSCM(
        loading=np.array([[1.0]]),
        beta=np.array([1.0]),
        sigma_xi=np.array([[2.0]]),
        sigma_y=0.1,
    )
    sources = (
        GaussianEnvironment(np.array([[0.7]]), np.zeros(1), np.array([[0.02]]), "source_1"),
        GaussianEnvironment(np.array([[-0.1]]), np.zeros(1), np.array([[0.02]]), "source_2"),
    )
    target = GaussianEnvironment(np.array([[-1.0]]), np.zeros(1), np.array([[0.02]]), "target")
    erm = affine_erm(scm, sources)
    direct, adjusted = nuisance_observability(scm, sources)
    robust = correlation_robust_risk(
        scm,
        erm,
        relation_center=np.array([[0.0]]),
        relation_radius=1.0,
        sigma_a=np.array([[0.02]]),
    )
    return {
        "claim": "C002-IRM",
        "source_relations": [0.7, -0.1],
        "target_relation": -1.0,
        "population_erm": erm.tolist(),
        "source_observational_oracle_risk": observational_oracle_risk(scm, sources),
        "source_risk_at_erm": source_risk(scm, sources, erm),
        "irmv1_scale_penalty_at_erm": irmv1_scale_penalty(scm, sources, erm),
        "gradient_alignment_penalty_at_erm": gradient_alignment_penalty(scm, sources, erm),
        "target_risk_at_erm": risk(scm, target, erm),
        "causal_oracle_risk": causal_oracle_risk(scm),
        "target_causal_excess": risk(scm, target, erm) - causal_oracle_risk(scm),
        "correlation_ball_robust_risk": robust,
        "correlation_ball_robust_causal_excess": robust - causal_oracle_risk(scm),
        "direct_gradient_observability": direct,
        "task_adjusted_gradient_observability": adjusted,
        "all_scalar_irmv1_zeroes": [
            candidate.tolist() for candidate in scalar_irmv1_zero_candidates(scm, sources)
        ],
    }


def main() -> None:
    print(json.dumps(fixed_irmv1_counterexample_report(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
