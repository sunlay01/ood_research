"""Method-specific observations for the first-round state audit.

These functions deliberately expose what each objective measures.  They do
not infer a common semantic mechanism for IRMv1, CORAL, and L2.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .intervention_linear import (
    GaussianEnvironment,
    LinearGaussianSCM,
    feature_moments,
    irmv1_radial_response,
    irmv1_scale_penalty,
    scalar_relation_design,
    scalar_relation_response,
)
Array = np.ndarray


@dataclass(frozen=True)
class RegularizerObservation:
    method: str
    value: float
    observed_object: str
    controlled_candidate: str
    blind_candidate: str


@dataclass(frozen=True)
class ResponseOperator:
    """Linear response map for a method-specific state coordinate."""

    matrix: Array
    labels: tuple[str, ...]
    state_labels: tuple[str, ...]

    @property
    def rank(self) -> int:
        return int(np.linalg.matrix_rank(self.matrix))

    @property
    def kernel_dimension(self) -> int:
        return len(self.state_labels) - self.rank


def _covariance(scm: LinearGaussianSCM, environment: GaussianEnvironment) -> Array:
    second, _ = feature_moments(scm, environment)
    mean = second[1:, 0]
    return second[1:, 1:] - np.outer(mean, mean)


def irmv1_state_response(
    scm: LinearGaussianSCM,
    environments: tuple[GaussianEnvironment, ...],
    w: Array,
) -> dict[str, object]:
    """Return radial response, polynomial design when available, and blind notes."""
    response = irmv1_radial_response(scm, environments, w)
    result: dict[str, object] = {
        "penalty": response.penalty,
        "derivatives": response.derivatives,
        "radial_gradient_norms": response.radial_gradient_norms,
        "tangential_gradient_norms": response.tangential_gradient_norms,
        "observed_object": "per-environment radial risk response",
        "blind_candidate": "tangential gradient and response-polynomial nullspace",
    }
    if (
        scm.c_dim == 1
        and scm.u_dim == 1
        and len(environments) >= 1
        and all(environment.a_dim == 1 for environment in environments)
        and all(np.allclose(environment.mean, 0.0) for environment in environments)
        and all(np.allclose(environment.sigma_a, environments[0].sigma_a) for environment in environments)
        and np.isclose(w[0], 0.0)
    ):
        relations = np.array([environment.relation[0, 0] for environment in environments])
        result["relation_design"] = scalar_relation_design(relations)
        result["relation_response_coefficients"] = scalar_relation_response(scm, environments, w).coefficients
    return result


def coral_state_response(
    scm: LinearGaussianSCM,
    environments: tuple[GaussianEnvironment, ...],
    representation: Array | None = None,
) -> dict[str, object]:
    """Compute CORAL's pairwise centered covariance observations."""
    if len(environments) < 2:
        raise ValueError("CORAL requires at least two environments")
    input_dim = scm.u_dim + environments[0].a_dim
    if representation is None:
        representation = np.eye(input_dim)
    representation = np.asarray(representation, dtype=float)
    if representation.ndim != 2 or representation.shape[1] != input_dim:
        raise ValueError("representation has incompatible input dimension")
    covariances = [representation @ _covariance(scm, environment) @ representation.T for environment in environments]
    differences = []
    values = []
    for i in range(len(covariances)):
        for j in range(i + 1, len(covariances)):
            difference = covariances[i] - covariances[j]
            differences.append(difference)
            values.append(float(np.mean(difference * difference)))
    return {
        "penalty": float(np.mean(values)),
        "pairwise_values": np.asarray(values),
        "covariance_differences": tuple(differences),
        "observed_object": "representation covariance differences",
        "blind_candidate": "mean shift and conditional label response",
    }


def l2_state_response(w: Array) -> dict[str, object]:
    """Report global parameter magnitude and its nonselective blocks."""
    w = np.asarray(w, dtype=float)
    if w.ndim != 1 or w.size < 2:
        raise ValueError("w must be a non-empty affine coefficient vector")
    return {
        "penalty": float(w[1:] @ w[1:]),
        "task_observation_norm": float(w[1:-1] @ w[1:-1]) if w.size > 2 else 0.0,
        "nuisance_norm": float(w[-1] ** 2),
        "observed_object": "global coefficient norm",
        "blind_candidate": "task/nuisance distinction and shift direction",
    }


def regularizer_observation(
    method: str,
    scm: LinearGaussianSCM,
    environments: tuple[GaussianEnvironment, ...],
    w: Array,
    representation: Array | None = None,
) -> RegularizerObservation:
    if method == "irmv1":
        result = irmv1_state_response(scm, environments, w)
        return RegularizerObservation(method, float(result["penalty"]), str(result["observed_object"]), "relation radial response", str(result["blind_candidate"]))
    if method == "coral":
        result = coral_state_response(scm, environments, representation)
        return RegularizerObservation(method, float(result["penalty"]), str(result["observed_object"]), "covered covariance moments", str(result["blind_candidate"]))
    if method == "l2":
        result = l2_state_response(w)
        return RegularizerObservation(method, float(result["penalty"]), str(result["observed_object"]), "global norm under a shift budget", str(result["blind_candidate"]))
    raise ValueError(f"unsupported round-one method: {method}")


def regularizer_operator(
    method: str,
    scm: LinearGaussianSCM,
    environments: tuple[GaussianEnvironment, ...],
    representation: Array | None = None,
) -> ResponseOperator:
    """Return a linearized observation operator where the objective has one.

    IRMv1's scalar response is nonlinear in the predictor.  The scalar
    relation design is returned as its response-polynomial operator.  CORAL
    returns covariance-difference rows.  L2 has no shift-specific linear row;
    an empty operator makes that absence explicit.
    """
    if method == "irmv1":
        if scm.c_dim == 1 and scm.u_dim == 1 and all(environment.a_dim == 1 for environment in environments):
            design = scalar_relation_design([environment.relation[0, 0] for environment in environments])
            return ResponseOperator(design, tuple(f"source_env_{i}" for i in range(design.shape[0])), ("q0", "q1", "q2"))
        return ResponseOperator(np.empty((0, 0)), (), ())
    if method == "coral":
        result = coral_state_response(scm, environments, representation)
        differences = result["covariance_differences"]
        if not differences:
            return ResponseOperator(np.empty((0, 0)), (), ())
        dimension = differences[0].shape[0]
        return ResponseOperator(
            np.stack([difference.reshape(-1) for difference in differences]),
            tuple(f"coral_pair_{i}" for i in range(len(differences))),
            tuple(f"cov_{i}" for i in range(dimension * dimension)),
        )
    if method == "l2":
        return ResponseOperator(np.empty((0, 0)), (), ())
    raise ValueError(f"unsupported round-one method: {method}")
