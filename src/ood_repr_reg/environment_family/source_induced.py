"""Source-only environment family generated from observed task-state variation."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..round3r_3b_benchmark import ModuleEnvironment, environment_state, source_design
from ..round3r_3d_state import task_state
from .base import Environment, Role, TangentSpec

Array = np.ndarray


@dataclass(frozen=True)
class EnvironmentParameterization:
    """Continuous legal coordinates for a fixed-feature Gaussian environment."""

    shortcut_count: int
    noise_count: int

    @property
    def names(self) -> tuple[str, ...]:
        return (
            *(f"rho_{i+1}" for i in range(self.shortcut_count)),
            *(f"mean_{i+1}" for i in range(self.shortcut_count)),
            *(f"variance_{i+1}" for i in range(self.shortcut_count)),
            *(f"noise_mean_{i+1}" for i in range(self.noise_count)),
            *(f"noise_variance_{i+1}" for i in range(self.noise_count)),
            "c_noise_variance", "u_gamma", "u_noise_variance",
        )

    def vector(self, environment: ModuleEnvironment) -> Array:
        return np.asarray((
            *environment.shortcut_rhos, *environment.shortcut_means,
            *environment.shortcut_variances, *environment.noise_means,
            *environment.noise_variances, environment.c_noise_variance,
            environment.u_gamma, environment.u_noise_variance,
        ), dtype=float)

    def from_vector(self, reference: ModuleEnvironment, vector: Array) -> ModuleEnvironment:
        value = np.asarray(vector, dtype=float)
        expected = len(self.names)
        if value.shape != (expected,):
            raise ValueError("environment parameter vector has incompatible shape")
        if not np.all(np.isfinite(value)):
            raise ValueError("environment parameters must be finite")
        q, n = self.shortcut_count, self.noise_count
        index = 0
        rhos = tuple(value[index:index + q]); index += q
        means = tuple(value[index:index + q]); index += q
        variances = tuple(value[index:index + q]); index += q
        noise_means = value[index:index + n]; index += n
        noise_variances = value[index:index + n]; index += n
        c_noise_variance, u_gamma, u_noise_variance = value[index:index + 3]
        if min(*variances, *noise_variances, c_noise_variance, u_noise_variance) < 0.0:
            raise ValueError("parameter pullback left the Gaussian family")
        return reference.updated(
            shortcut_rhos=rhos, shortcut_means=means,
            shortcut_variances=variances, noise_means=noise_means,
            noise_variances=noise_variances,
            c_noise_variance=float(c_noise_variance), u_gamma=float(u_gamma),
            u_noise_variance=float(u_noise_variance),
        )


def _rank(singular: Array, tolerance: float) -> int:
    if singular.size == 0 or singular[0] <= 0.0:
        return 0
    return int(np.sum(singular > tolerance * singular[0]))


def source_state_contrasts(
    source_environments: tuple[ModuleEnvironment, ...], reference_index: int = 0,
) -> Array:
    """Return the column matrix of source task-state contrasts."""
    environments = tuple(source_environments)
    if not environments or not 0 <= reference_index < len(environments):
        raise ValueError("source environments and reference index are incompatible")
    states = np.stack([task_state(environment_state(environment)) for environment in environments])
    return (states - states[reference_index]).T


def source_state_basis(
    source_environments: tuple[ModuleEnvironment, ...], reference_index: int = 0,
    tolerance: float = 1e-10,
) -> tuple[Array, Array, int]:
    """Compute the stable SVD basis before legal parameter realization."""
    contrasts = source_state_contrasts(source_environments, reference_index)
    vectors, singular, _ = np.linalg.svd(contrasts, full_matrices=False)
    rank = _rank(singular, tolerance)
    return vectors[:, :rank], singular, rank


def _parameter_lower_bound_indices(parameters: EnvironmentParameterization) -> tuple[int, ...]:
    names = parameters.names
    constrained_names = (
        *(f"variance_{i + 1}" for i in range(parameters.shortcut_count)),
        *(f"noise_variance_{i + 1}" for i in range(parameters.noise_count)),
        "c_noise_variance", "u_noise_variance",
    )
    return tuple(names.index(name) for name in constrained_names)


def _parameter_step_radii(theta: Array, index: int,
                          lower_bound_indices: tuple[int, ...]) -> tuple[float, float]:
    if index in lower_bound_indices:
        return max(0.0, float(theta[index])), float("inf")
    return float("inf"), float("inf")


def parameter_jacobian(reference: ModuleEnvironment, parameters: EnvironmentParameterization,
                       step: float = 1e-6, *,
                       return_diagnostics: bool = False) -> Array | tuple[Array, list[dict[str, object]]]:
    if step <= 0.0:
        raise ValueError("step must be positive")
    theta = parameters.vector(reference)
    base = task_state(environment_state(reference))
    constrained = _parameter_lower_bound_indices(parameters)
    columns: list[Array] = []
    diagnostics: list[dict[str, object]] = []
    for index in range(theta.size):
        negative, positive = _parameter_step_radii(theta, index, constrained)
        h = float(step)
        direction = np.zeros_like(theta)
        if negative >= h and positive >= h:
            scheme = "central"
            direction[index] = h
            plus = task_state(environment_state(parameters.from_vector(reference, theta + direction)))
            minus = task_state(environment_state(parameters.from_vector(reference, theta - direction)))
            column = (plus - minus) / (2.0 * h)
        elif negative > 0.0 and positive > 0.0:
            h = min(h, negative, positive) * 0.5
            scheme = "central_shrunk"
            direction[index] = h
            plus = task_state(environment_state(parameters.from_vector(reference, theta + direction)))
            minus = task_state(environment_state(parameters.from_vector(reference, theta - direction)))
            column = (plus - minus) / (2.0 * h)
        elif positive > 0.0:
            h = min(h, positive / 2.0) * 0.5 if np.isfinite(positive) else h
            scheme = "forward_second_order"
            direction[index] = h
            first = task_state(environment_state(parameters.from_vector(reference, theta + direction)))
            second = task_state(environment_state(parameters.from_vector(reference, theta + 2.0 * direction)))
            column = (-3.0 * base + 4.0 * first - second) / (2.0 * h)
        elif negative > 0.0:
            h = min(h, negative / 2.0) * 0.5 if np.isfinite(negative) else h
            scheme = "backward_second_order"
            direction[index] = h
            first = task_state(environment_state(parameters.from_vector(reference, theta - direction)))
            second = task_state(environment_state(parameters.from_vector(reference, theta - 2.0 * direction)))
            column = (3.0 * base - 4.0 * first + second) / (2.0 * h)
        else:
            raise ValueError(f"no legal finite-difference step for parameter {parameters.names[index]!r}")
        columns.append(column)
        diagnostics.append({
            "parameter": parameters.names[index],
            "requested_step": float(step),
            "used_step": float(h),
            "scheme": scheme,
            "legal_plus": bool(positive >= h),
            "legal_minus": bool(negative >= h),
            "distance_to_boundary": {"negative": float(negative), "positive": float(positive)},
        })
    result = np.column_stack(columns) if columns else np.zeros((base.size, 0))
    return (result, diagnostics) if return_diagnostics else result


@dataclass(frozen=True)
class SourceInducedFamily:
    """Family whose basis is constructed only from observed source states."""

    base: ModuleEnvironment
    observed_source_environments: tuple[ModuleEnvironment, ...]
    mode_vectors: Array
    pullbacks: Array
    retained_mode_indices: tuple[int, ...]
    realized_state_gram: Array
    realization_residuals: tuple[float, ...]
    state_span_singular_values: Array
    state_span_rank: int
    realizable_rank: int
    unrealizable_modes: tuple[str, ...]
    parameterization: EnvironmentParameterization
    source_reference_index: int = 0
    realization_tolerance: float = 1e-8
    finite_difference_step: float = 1e-5
    _spec: TangentSpec | None = field(default=None, repr=False, compare=False)

    def reference_environment(self) -> ModuleEnvironment:
        return self.base

    def source_environments(self, reference: Environment | None = None) -> tuple[Environment, ...]:
        del reference
        return self.observed_source_environments

    def tangent_spec(self, reference: Environment | None = None) -> TangentSpec:
        if reference is not None and reference != self.base:
            raise ValueError("source-induced tangent is defined only at its source reference")
        if self._spec is None:
            directions = tuple(f"source_mode_{i}" for i in self.retained_mode_indices)
            self_spec = TangentSpec(
                directions=directions, scales=tuple(1.0 for _ in directions),
                metric=self.realized_state_gram,
                family_name="source_induced",
                coordinate_description=(
                    "legally realized source task-state contrast modes; metric is "
                    "the realized source-state Gram matrix"
                ),
                canonical_parameterization=False, source_defined=True,
                mechanism_defined=False,
                reference_metadata={
                    "state_span_rank": self.state_span_rank,
                    "realizable_rank": self.realizable_rank,
                    "unrealizable_modes": list(self.unrealizable_modes),
                    "retained_mode_indices": list(self.retained_mode_indices),
                    "metric_kind": "realized_source_state_gram",
                    "realized_metric_identity_error": float(np.linalg.norm(
                        self.realized_state_gram - np.eye(self.realizable_rank)
                    )),
                },
                finite_difference_step=self.finite_difference_step,
            )
            object.__setattr__(self, "_spec", self_spec)
        return self._spec

    def perturb(self, reference: Environment, coordinate: str, signed_step: float,
                *, role: Role) -> ModuleEnvironment:
        del role
        if (reference.dimension != self.base.dimension or reference.n_noise != self.base.n_noise
                or len(reference.shortcut_rhos) != len(self.base.shortcut_rhos)):
            raise ValueError("source-induced perturbation requires a compatible registered environment")
        # The chart is defined at ``base`` but the same legal parameter
        # displacement is applied to each registered source environment.  It
        # therefore preserves source-design offsets while differentiating the
        # stacked source observation.
        index = self.tangent_spec(self.base).index(coordinate)
        theta = self.parameterization.vector(reference)
        return self.parameterization.from_vector(
            reference, theta + float(signed_step) * self.pullbacks[:, index]
        )

    def legal_step_interval(self, reference: Environment, coordinate: str,
                            role: Role) -> tuple[float, float]:
        del role
        index = self.tangent_spec(self.base).index(coordinate)
        theta = self.parameterization.vector(reference)
        delta = np.asarray(self.pullbacks[:, index], dtype=float)
        names = self.parameterization.names
        constrained = _parameter_lower_bound_indices(self.parameterization)
        negative = positive = float("inf")
        for position in constrained:
            value, slope = float(theta[position]), float(delta[position])
            if slope > 0.0:
                negative = min(negative, value / slope)
            elif slope < 0.0:
                positive = min(positive, value / (-slope))
        return max(0.0, negative), max(0.0, positive)

    def metadata(self) -> dict[str, object]:
        return {
            "family_name": "source_induced",
            "source_defined": True,
            "mechanism_defined": False,
            "canonical_parameterization": False,
            "metric_kind": "realized_source_state_gram",
            "realized_metric": self.realized_state_gram.tolist(),
            "realized_metric_identity_error": float(np.linalg.norm(
                self.realized_state_gram - np.eye(self.realizable_rank)
            )),
            "retained_mode_indices": list(self.retained_mode_indices),
            "state_span_rank": self.state_span_rank,
            "source_contrast_rank": self.state_span_rank,
            "realizable_rank": self.realizable_rank,
            "unrealizable_modes": list(self.unrealizable_modes),
            "realization_residuals": list(self.realization_residuals),
            "source_reference_index": self.source_reference_index,
            "source_environment_count": len(self.observed_source_environments),
            "target_risk_used": False,
            "response_operator_used": False,
            "regularizer_geometry_used": False,
            "semantic_labels_used": False,
        }


def build_source_induced_family(
    source_environments: tuple[ModuleEnvironment, ...] | None = None,
    *,
    base: ModuleEnvironment | None = None,
    reference_index: int = 0,
    tolerance: float = 1e-10,
    realization_tolerance: float = 1e-8,
    jacobian_step: float = 1e-6,
) -> SourceInducedFamily:
    if source_environments is None:
        base = ModuleEnvironment(
            shortcut_rhos=(0.75, 0.57), shortcut_means=(0.18, 0.08),
            shortcut_variances=(0.49, 0.61), n_noise=4,
        ) if base is None else base
        source_environments = source_design(base, relation_exposed=True)
    environments = tuple(source_environments)
    if not environments:
        raise ValueError("source-induced family requires at least one source environment")
    if not 0 <= reference_index < len(environments):
        raise ValueError("reference_index is out of range")
    reference = environments[reference_index]
    if base is not None and base != reference:
        raise ValueError("base must equal source_environments[reference_index]; tangent transport is not implicit")
    base = reference
    if any(environment.dimension != base.dimension or environment.n_noise != base.n_noise
           or len(environment.shortcut_rhos) != len(base.shortcut_rhos)
           for environment in environments):
        raise ValueError("source environments must share the base feature configuration")
    modes, singular, rank = source_state_basis(environments, reference_index, tolerance)
    parameters = EnvironmentParameterization(len(base.shortcut_rhos), base.n_noise)
    jacobian = parameter_jacobian(base, parameters, jacobian_step)
    pullback_columns: list[Array] = []
    retained_indices: list[int] = []
    residuals: list[float] = []
    unrealizable: list[str] = []
    for index in range(rank):
        mode = modes[:, index]
        pullback = np.linalg.pinv(jacobian) @ mode
        residual = float(np.linalg.norm(jacobian @ pullback - mode))
        residuals.append(residual)
        if residual <= realization_tolerance * max(1.0, np.linalg.norm(mode)):
            pullback_columns.append(pullback)
            retained_indices.append(index)
        else:
            unrealizable.append(f"source_mode_{index}")
    pullbacks = np.column_stack(pullback_columns) if pullback_columns else np.zeros((jacobian.shape[1], 0))
    realized = jacobian @ pullbacks
    realized_gram = realized.T @ realized
    return SourceInducedFamily(
        base=base, observed_source_environments=environments,
        mode_vectors=modes, pullbacks=pullbacks,
        retained_mode_indices=tuple(retained_indices), realized_state_gram=realized_gram,
        realization_residuals=tuple(residuals), state_span_singular_values=singular,
        state_span_rank=rank, realizable_rank=len(pullback_columns),
        unrealizable_modes=tuple(unrealizable), parameterization=parameters,
        source_reference_index=reference_index,
        realization_tolerance=realization_tolerance, finite_difference_step=1e-5,
    )


def source_induced_audit(family: SourceInducedFamily) -> dict[str, object]:
    spec = family.tangent_spec()
    return {
        "family_name": spec.family_name,
        "state_span_rank": family.state_span_rank,
        "source_contrast_rank": family.state_span_rank,
        "realizable_rank": family.realizable_rank,
        "state_span_singular_values": family.state_span_singular_values,
        "realization_residuals": family.realization_residuals,
        "unrealizable_modes": family.unrealizable_modes,
        "retained_mode_indices": family.retained_mode_indices,
        "realized_metric": spec.metric,
        "realized_metric_identity_error": float(np.linalg.norm(spec.metric - np.eye(spec.dimension))),
        "source_only": True,
        "target_risk_used": False,
        "response_operator_used": False,
        "regularizer_geometry_used": False,
        "semantic_labels_used": False,
        "cluster_labels_used": False,
        "direction_names_are_mechanisms": False,
    }


__all__ = [
    "EnvironmentParameterization", "SourceInducedFamily", "parameter_jacobian",
    "source_state_contrasts", "source_state_basis", "build_source_induced_family",
    "source_induced_family", "source_induced_audit",
]


source_induced_family = build_source_induced_family
