"""Legacy Gaussian/mechanism environment family.

The eight intervention coordinates live here as one concrete family chart;
downstream geometry consumes only ``TangentSpec`` and ``EnvironmentFamily``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..round3r_3b_benchmark import ModuleEnvironment, source_design
from .base import Environment, Role, TangentSpec

LEGACY_DIRECTIONS = (
    "S1_relation", "S2_relation", "S1_mean", "S2_mean",
    "S1_variance", "S2_variance", "N1_variance", "U_emergent",
)
LEGACY_SCALES = (0.20, 0.20, 0.35, 0.35, 0.30, 0.30, 0.50, 0.75)


def default_legacy_environment() -> ModuleEnvironment:
    return ModuleEnvironment(
        shortcut_rhos=(0.75, 0.57),
        shortcut_means=(0.18, 0.08),
        shortcut_variances=(0.49, 0.61),
        n_noise=4,
        u_gamma=0.0,
    )


@dataclass(frozen=True)
class LegacyGaussianFamily:
    """The frozen Gaussian family used by the original coupled benchmark."""

    base: ModuleEnvironment = field(default_factory=default_legacy_environment)
    relation_exposed_source: bool = True
    expose_u_at_source: bool = False
    finite_difference_step: float = 1e-5
    _spec: TangentSpec | None = field(default=None, repr=False, compare=False)

    def reference_environment(self) -> ModuleEnvironment:
        return self.base

    def source_environments(self, reference: Environment | None = None) -> tuple[Environment, ...]:
        return source_design(self.base if reference is None else reference,
                             relation_exposed=self.relation_exposed_source)

    def tangent_spec(self, reference: Environment | None = None) -> TangentSpec:
        del reference
        if self._spec is not None:
            return self._spec
        return TangentSpec(
            directions=LEGACY_DIRECTIONS,
            scales=LEGACY_SCALES,
            metric=np.eye(len(LEGACY_DIRECTIONS)),
            family_name="legacy_gaussian_mechanism",
            coordinate_description="Gaussian mechanism-family local relation/mean/variance chart",
            canonical_parameterization=False,
            source_defined=False,
            mechanism_defined=True,
            reference_metadata={"u_emergent_target_admissible": True,
                                "u_emergent_source_default": self.expose_u_at_source},
            finite_difference_step=self.finite_difference_step,
        )

    def perturb(self, reference: Environment, coordinate: str, signed_step: float,
                *, role: Role) -> ModuleEnvironment:
        spec = self.tangent_spec(reference)
        scale = spec.scales[spec.index(coordinate)] * float(signed_step)
        value = reference
        if coordinate.startswith("S"):
            family, parameter = coordinate.split("_", maxsplit=1)
            index = int(family[1:]) - 1
            if index >= len(value.shortcut_rhos):
                raise ValueError(f"{coordinate} is unavailable for this shortcut family")
            if parameter == "relation":
                values = list(value.shortcut_rhos); values[index] += scale
                return value.updated(shortcut_rhos=tuple(values))
            if parameter == "mean":
                values = list(value.shortcut_means); values[index] += scale
                return value.updated(shortcut_means=tuple(values))
            if parameter == "variance":
                values = list(value.shortcut_variances); values[index] += scale
                if values[index] < 0.0:
                    raise ValueError("variance perturbation left the Gaussian family")
                return value.updated(shortcut_variances=tuple(values))
        if coordinate == "N1_variance":
            values = np.asarray(value.noise_variances, dtype=float).copy(); values[0] += scale
            if values[0] < 0.0:
                raise ValueError("noise variance perturbation left the Gaussian family")
            return value.updated(noise_variances=values)
        if coordinate == "U_emergent":
            active = role == "target" or self.expose_u_at_source
            return value.updated(u_gamma=value.u_gamma + scale) if active else value
        raise ValueError(f"unknown tangent direction: {coordinate}")

    def legal_step_interval(self, reference: Environment, coordinate: str,
                            role: Role) -> tuple[float, float]:
        """Return signed-coordinate radii before a variance reaches zero."""
        del role
        spec = self.tangent_spec(reference)
        scale = spec.scales[spec.index(coordinate)]
        if coordinate.startswith("S"):
            family, parameter = coordinate.split("_", maxsplit=1)
            index = int(family[1:]) - 1
            if parameter == "variance":
                value = float(reference.shortcut_variances[index])
                return value / scale, value / scale
            return float("inf"), float("inf")
        if coordinate == "N1_variance":
            value = float(reference.noise_variances[0])
            return value / scale, value / scale
        if coordinate == "U_emergent":
            return float("inf"), float("inf")
        raise ValueError(f"unknown tangent direction: {coordinate}")

    def metadata(self) -> dict[str, object]:
        return {
            "family_name": "legacy_gaussian_mechanism",
            "source_defined": False,
            "mechanism_defined": True,
            "canonical_parameterization": False,
            "relation_exposed_source": self.relation_exposed_source,
            "expose_u_at_source": self.expose_u_at_source,
            "reference_feature_dimension": self.base.dimension,
            "target_only_u_is_modeled": True,
            "source_observation_uses_target_risk": False,
        }


def legacy_gaussian_family(*, expose_u_at_source: bool = False,
                           relation_exposed_source: bool = True) -> LegacyGaussianFamily:
    return LegacyGaussianFamily(expose_u_at_source=expose_u_at_source,
                                relation_exposed_source=relation_exposed_source)


__all__ = [
    "LEGACY_DIRECTIONS", "LEGACY_SCALES", "LegacyGaussianFamily",
    "default_legacy_environment", "legacy_gaussian_family",
]
