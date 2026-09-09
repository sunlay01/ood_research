"""Admissible environment-family abstractions for the coupled Round 3 geometry."""

from .base import EnvironmentFamily, FamilyTaskGeometry, TangentSpec
from .geometry import (
    build_task_geometry,
    family_response_factorization_residual,
    source_observation_operator,
    response_operator_for_family,
)
from .legacy_gaussian import (
    LEGACY_DIRECTIONS,
    LEGACY_SCALES,
    LegacyGaussianFamily,
    legacy_gaussian_family,
)
from .mechanism_family import MechanismFamily, mechanism_family
from .metrics import (
    coordinate_transport,
    metric_information_floor,
    metric_nullspace_basis,
    metric_operator_norm,
    metric_operator_summary,
    validate_metric,
    world_whiten,
)
from .source_induced import (
    SourceInducedFamily,
    build_source_induced_family,
    source_induced_family,
    source_state_basis,
    source_state_contrasts,
    source_induced_audit,
)
from .invariance_audit import failure_classification, family_coordinate_audit, source_span_projector
from .finite_difference import IllegalFiniteDifferenceStep, FiniteDifferenceDiagnostic, family_directional_derivative

__all__ = [
    "EnvironmentFamily", "FamilyTaskGeometry", "TangentSpec",
    "build_task_geometry", "family_response_factorization_residual",
    "source_observation_operator", "response_operator_for_family",
    "LEGACY_DIRECTIONS", "LEGACY_SCALES", "LegacyGaussianFamily", "legacy_gaussian_family",
    "MechanismFamily", "mechanism_family", "coordinate_transport", "metric_information_floor",
    "metric_operator_norm", "metric_operator_summary", "metric_nullspace_basis",
    "validate_metric", "world_whiten",
    "SourceInducedFamily", "build_source_induced_family", "source_induced_family",
    "source_state_basis", "source_state_contrasts", "source_induced_audit",
    "failure_classification", "family_coordinate_audit", "source_span_projector",
    "IllegalFiniteDifferenceStep", "FiniteDifferenceDiagnostic", "family_directional_derivative",
]
