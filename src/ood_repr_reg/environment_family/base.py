"""Base protocol and immutable metadata for admissible environment families."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol

import numpy as np

from ..round3r_3b_benchmark import ModuleEnvironment

Array = np.ndarray
Environment = ModuleEnvironment
Role = Literal["source", "target"]


@dataclass(frozen=True)
class TangentSpec:
    """A finite coordinate chart and metric for a family-local tangent."""

    directions: tuple[str, ...]
    scales: tuple[float, ...]
    metric: Array | None = None
    family_name: str = "unspecified"
    coordinate_description: str = "finite-dimensional local environment coordinates"
    canonical_parameterization: bool = False
    source_defined: bool = False
    mechanism_defined: bool = False
    reference_metadata: dict[str, object] = field(default_factory=dict)
    finite_difference_step: float = 1e-5

    def __post_init__(self) -> None:
        directions = tuple(str(item) for item in self.directions)
        scales = tuple(float(item) for item in self.scales)
        if len(directions) != len(scales):
            raise ValueError("directions and scales must be aligned")
        if len(set(directions)) != len(directions):
            raise ValueError("tangent directions must be unique within a family")
        if any(scale <= 0.0 or not np.isfinite(scale) for scale in scales):
            raise ValueError("tangent scales must be finite and positive")
        matrix = np.eye(len(directions)) if self.metric is None else np.asarray(self.metric, dtype=float)
        if matrix.shape != (len(directions), len(directions)):
            raise ValueError("tangent metric has incompatible shape")
        matrix = (matrix + matrix.T) / 2.0
        if not np.all(np.isfinite(matrix)):
            raise ValueError("tangent metric must be finite")
        eigenvalues = np.linalg.eigvalsh(matrix)
        if eigenvalues.size and eigenvalues.min() <= 0.0:
            raise ValueError("tangent metric must be positive definite")
        object.__setattr__(self, "directions", directions)
        object.__setattr__(self, "scales", scales)
        matrix = matrix.copy()
        matrix.setflags(write=False)
        object.__setattr__(self, "metric", matrix)
        if self.finite_difference_step <= 0.0:
            raise ValueError("finite_difference_step must be positive")

    @property
    def dimension(self) -> int:
        return len(self.directions)

    def index(self, direction: str) -> int:
        return self.directions.index(direction)

    def metadata(self) -> dict[str, object]:
        return {
            "family_name": self.family_name,
            "tangent_dimension": self.dimension,
            "directions": list(self.directions),
            "magnitude_one_scales": dict(zip(self.directions, self.scales, strict=True)),
            "metric": self.metric.tolist(),
            "coordinate_description": self.coordinate_description,
            "canonical_parameterization": self.canonical_parameterization,
            "source_defined": self.source_defined,
            "mechanism_defined": self.mechanism_defined,
            "reference_metadata": dict(self.reference_metadata),
            "finite_difference_step": self.finite_difference_step,
        }


class EnvironmentFamily(Protocol):
    """Protocol for a declared local family of admissible environments."""

    def reference_environment(self) -> Environment:
        ...

    def source_environments(self, reference: Environment | None = None) -> tuple[Environment, ...]:
        ...

    def tangent_spec(self, reference: Environment | None = None) -> TangentSpec:
        ...

    def perturb(
        self,
        reference: Environment,
        coordinate: str,
        signed_step: float,
        *,
        role: Role,
    ) -> Environment:
        ...

    def metadata(self) -> dict[str, object]:
        ...


@dataclass(frozen=True)
class FamilyTaskGeometry:
    """The common family-relative input consumed by 3A, 3C, 3D and 3E."""

    family: EnvironmentFamily
    reference: Environment
    spec: TangentSpec
    source_environments: tuple[Environment, ...]
    source_state: object
    source_optimum: Array
    observation: Array
    response: Array

    @property
    def metric(self) -> Array:
        return np.asarray(self.spec.metric, dtype=float)

    @property
    def base(self) -> Environment:
        """Compatibility name for the family reference environment."""
        return self.reference

    @property
    def source(self) -> object:
        """Compatibility name for the aggregate source moment state."""
        return self.source_state

    @property
    def optimum(self) -> Array:
        """Compatibility name for the source population optimum."""
        return self.source_optimum

    @property
    def source_observation_operator(self) -> Array:
        """Alias used by the 3D-facing interface."""
        return self.observation

    @property
    def response_operator(self) -> Array:
        """Alias used by the 3A-facing interface."""
        return self.response

    @property
    def O_S(self) -> Array:
        return self.observation

    @property
    def A(self) -> Array:
        return self.response

    def metadata(self) -> dict[str, object]:
        return {
            "family": self.family.metadata(),
            "tangent": self.spec.metadata(),
            "source_environment_count": len(self.source_environments),
            "observation_shape": list(self.observation.shape),
            "response_shape": list(self.response.shape),
        }


__all__ = ["Array", "Environment", "Role", "TangentSpec", "EnvironmentFamily", "FamilyTaskGeometry"]
