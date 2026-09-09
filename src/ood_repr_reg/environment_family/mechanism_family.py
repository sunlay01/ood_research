"""Explicit mechanism-family metadata layered over the legacy Gaussian chart."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from .base import Environment, Role, TangentSpec
from .legacy_gaussian import LegacyGaussianFamily


@dataclass(frozen=True)
class MechanismFamily:
    """A lightweight mechanism-defined family wrapper.

    The wrapper changes declared metadata only; its population perturbations
    delegate to the legacy Gaussian implementation and therefore add no new
    semantic or causal claim.
    """

    delegate: LegacyGaussianFamily = field(default_factory=LegacyGaussianFamily)
    mutable_mechanisms: tuple[str, ...] = ("shortcut_1", "shortcut_2", "noise", "u_emergent")
    frozen_mechanisms: tuple[str, ...] = ("label", "core_predictive_feature")

    def reference_environment(self) -> Environment:
        return self.delegate.reference_environment()

    def source_environments(self, reference: Environment | None = None) -> tuple[Environment, ...]:
        return self.delegate.source_environments(reference)

    def tangent_spec(self, reference: Environment | None = None) -> TangentSpec:
        spec = self.delegate.tangent_spec(reference)
        return replace(spec, family_name="gaussian_mechanism_defined",
                       mechanism_defined=True,
                       reference_metadata={**spec.reference_metadata,
                                           "mutable_mechanisms": list(self.mutable_mechanisms),
                                           "frozen_mechanisms": list(self.frozen_mechanisms)})

    def perturb(self, reference: Environment, coordinate: str, signed_step: float,
                *, role: Role) -> Environment:
        return self.delegate.perturb(reference, coordinate, signed_step, role=role)

    def legal_step_interval(self, reference: Environment, coordinate: str,
                            role: Role) -> tuple[float, float]:
        return self.delegate.legal_step_interval(reference, coordinate, role=role)

    def metadata(self) -> dict[str, object]:
        return {
            **self.delegate.metadata(),
            "family_name": "gaussian_mechanism_defined",
            "mechanism_defined": True,
            "mutable_mechanisms": list(self.mutable_mechanisms),
            "frozen_mechanisms": list(self.frozen_mechanisms),
        }


def mechanism_family() -> MechanismFamily:
    return MechanismFamily()


__all__ = ["MechanismFamily", "mechanism_family"]
