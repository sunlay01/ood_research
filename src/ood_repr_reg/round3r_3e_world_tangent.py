"""Declared 3A/3D-coupled world tangent for the 3E recovery theorem.

The finite benchmark is deliberately a local information model, not a
mechanism identification claim.  Its Euclidean world metric is fixed by the
registered magnitude-one 3B interventions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .round3r_3b_benchmark import ModuleEnvironment, environment_state
from .round3r_3d_exposure import response_operator
from .environment_family.base import FamilyTaskGeometry, TangentSpec
from .environment_family.geometry import (
    build_task_geometry,
    response_operator_for_family,
    source_observation_operator,
)
from .environment_family.legacy_gaussian import LegacyGaussianFamily

Array = np.ndarray


@dataclass(frozen=True)
class WorldTangentSpec:
    """Standardized local coordinates and their physical magnitude-one scales."""

    directions: tuple[str, ...] = (
        "S1_relation", "S2_relation", "S1_mean", "S2_mean",
        "S1_variance", "S2_variance", "N1_variance", "U_emergent",
    )
    scales: tuple[float, ...] = (0.20, 0.20, 0.35, 0.35, 0.30, 0.30, 0.50, 0.75)
    finite_difference_step: float = 1e-5

    def __post_init__(self) -> None:
        if len(self.directions) != 8 or len(self.scales) != len(self.directions):
            raise ValueError("the declared 3E tangent must have eight scaled directions")
        if self.finite_difference_step <= 0.0 or any(scale <= 0.0 for scale in self.scales):
            raise ValueError("finite-difference step and tangent scales must be positive")

    @property
    def dimension(self) -> int:
        return len(self.directions)

    def index(self, direction: str) -> int:
        return self.directions.index(direction)

    def metadata(self) -> dict[str, object]:
        return {
            "world_metric": "Euclidean on standardized environment intervention coordinates",
            "directions": list(self.directions),
            "magnitude_one_scales": dict(zip(self.directions, self.scales, strict=True)),
            "source_metric": "frozen 3A H_S^{-1/2} response norm",
            "finite_difference_step": self.finite_difference_step,
            "metric_conditional": True,
            "canonical_mechanism_parameterization_claimed": False,
        }

    def as_tangent_spec(self) -> TangentSpec:
        """Return the shared family-layer representation of this legacy spec."""
        return TangentSpec(
            directions=self.directions,
            scales=self.scales,
            metric=np.eye(self.dimension),
            family_name="legacy_gaussian_mechanism",
            coordinate_description="Gaussian mechanism-family local relation/mean/variance chart",
            canonical_parameterization=False,
            source_defined=False,
            mechanism_defined=True,
            reference_metadata={
                "u_emergent_target_admissible": True,
                "u_emergent_source_default": False,
            },
            finite_difference_step=self.finite_difference_step,
        )

    def legacy_family(self, base: ModuleEnvironment | None = None,
                      *, expose_u_at_source: bool = False) -> LegacyGaussianFamily:
        """Return the family object behind this compatibility chart."""
        return LegacyGaussianFamily(
            base=default_environment() if base is None else base,
            expose_u_at_source=expose_u_at_source,
            _spec=self.as_tangent_spec(),
        )


@dataclass(frozen=True)
class CoupledTangentGeometry:
    """A frozen 3A response map paired with a 3D source observation map."""

    spec: WorldTangentSpec
    base: ModuleEnvironment
    source_environments: tuple[ModuleEnvironment, ...]
    source_state: object
    source_optimum: Array
    observation: Array
    response: Array
    family_geometry: FamilyTaskGeometry | None = None

    @property
    def family(self):
        """The declared family behind this compatibility object."""
        return None if self.family_geometry is None else self.family_geometry.family


def default_environment() -> ModuleEnvironment:
    """The frozen two-shortcut population benchmark used throughout 3B--3D."""
    return ModuleEnvironment(
        shortcut_rhos=(0.75, 0.57),
        shortcut_means=(0.18, 0.08),
        shortcut_variances=(0.49, 0.61),
        n_noise=4,
        u_gamma=0.0,
    )


def perturb_environment(environment: ModuleEnvironment, spec: WorldTangentSpec,
                        direction: str, signed_step: float,
                        *, expose_u_at_source: bool) -> ModuleEnvironment:
    """Perturb one declared coordinate while retaining source-design offsets."""
    family = LegacyGaussianFamily(
        base=environment,
        expose_u_at_source=expose_u_at_source,
        _spec=spec.as_tangent_spec(),
    )
    role = "source" if not expose_u_at_source else "target"
    return family.perturb(environment, direction, signed_step, role=role)


def source_observation_block(environments: tuple[ModuleEnvironment, ...], spec: WorldTangentSpec,
                             *, expose_u_at_source: bool = False,
                             step: float | None = None) -> Array:
    """Stack source-only task-state Jacobians for an observation block.

    No target state, risk, response label, cluster, or regularizer enters this
    function.  The U direction is deliberately zero unless the source design
    explicitly exposes it.
    """
    base = environments[0] if environments else default_environment()
    family = LegacyGaussianFamily(
        base=base,
        expose_u_at_source=expose_u_at_source,
        _spec=spec.as_tangent_spec(),
    )
    return source_observation_operator(
        family, tuple(environments), family.tangent_spec(), step=step,
    )


def frozen_response_jacobian(base: ModuleEnvironment, source_state, source_optimum: Array,
                             spec: WorldTangentSpec, *, step: float | None = None) -> Array:
    """Differentiate frozen-3A whitened target response in the same world basis."""
    family = LegacyGaussianFamily(base=base, _spec=spec.as_tangent_spec())
    return response_operator_for_family(
        family, source_state, source_optimum, family.tangent_spec(), step=step,
    )


def coupled_primary_geometry(spec: WorldTangentSpec | None = None, *, family=None):
    """Build the primary 3A/3D-coupled pair with source-hidden U emergence."""
    if family is not None:
        if spec is not None:
            raise ValueError("spec and family cannot both override the primary geometry")
        return build_task_geometry(family)
    spec = WorldTangentSpec() if spec is None else spec
    base = default_environment()
    family = LegacyGaussianFamily(base=base, _spec=spec.as_tangent_spec())
    geometry = build_task_geometry(family)
    return CoupledTangentGeometry(
        spec, geometry.reference, geometry.source_environments, geometry.source_state,
        geometry.source_optimum, geometry.observation, geometry.response, geometry,
    )


def build_family_geometry(family, *, reference=None, source_environments=None,
                          step: float | None = None):
    """Public family-first entry point used by refactored downstream tracks."""
    return build_task_geometry(
        family, reference=reference, source_environments=source_environments, step=step,
    )


def u_exposed_observation(geometry: CoupledTangentGeometry) -> Array:
    """Return the separately-labelled source design that observes ``U``.

    This is an information intervention, not a replacement for the primary
    hidden-emergent design.  The response map remains fixed while only the
    source observation block is changed.
    """
    return source_observation_block(
        geometry.source_environments, geometry.spec, expose_u_at_source=True,
    )


def response_factorization_residual(geometry: CoupledTangentGeometry,
                                    direction: str = "S1_relation") -> float:
    """Audit ``response_operator @ state_difference == whitened response``."""
    from .round3r_3d_state import state_difference

    index = geometry.spec.index(direction)
    scale = geometry.spec.scales[index]
    target = environment_state(perturb_environment(
        geometry.base, geometry.spec, direction, 1.0, expose_u_at_source=True,
    ))
    delta = state_difference(target, geometry.source_state)
    operator = response_operator(geometry.source_state, geometry.source_optimum)
    direct = response_operator(geometry.source_state, geometry.source_optimum, target)
    return float(np.linalg.norm(direct - operator @ delta))


def finite_difference_stability(geometry: CoupledTangentGeometry,
                                steps: tuple[float, ...] = (1e-4, 1e-5, 1e-6)) -> dict[str, object]:
    """Report central-difference stability against the declared primary step."""
    reference_o = geometry.observation
    reference_a = geometry.response
    rows = []
    for step in steps:
        observation = source_observation_block(
            geometry.source_environments, geometry.spec, expose_u_at_source=False, step=step,
        )
        response = frozen_response_jacobian(
            geometry.base, geometry.source_state, geometry.source_optimum, geometry.spec, step=step,
        )
        rows.append({
            "step": step,
            "source_observation_relative_error": float(np.linalg.norm(observation - reference_o) / max(1.0, np.linalg.norm(reference_o))),
            "response_relative_error": float(np.linalg.norm(response - reference_a) / max(1.0, np.linalg.norm(reference_a))),
        })
    return {
        "rows": rows,
        "source_only_observation_construction": True,
        "target_risk_used": False,
        "mechanism_labels_used": False,
        "cluster_labels_used": False,
        "regularizer_geometry_used": False,
        "pass": bool(all(
            row["source_observation_relative_error"] < 1e-7
            and row["response_relative_error"] < 1e-7
            for row in rows
        )),
    }


__all__ = [
    "WorldTangentSpec", "CoupledTangentGeometry", "default_environment",
    "perturb_environment", "source_observation_block", "frozen_response_jacobian",
    "coupled_primary_geometry", "u_exposed_observation",
    "build_family_geometry",
    "response_factorization_residual", "finite_difference_stability",
]
