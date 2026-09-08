"""Cross-domain error-certificate primitives.

The objects in this module describe candidate certificate inputs.  They do not
claim that any particular combination is a valid upper bound; that claim must
be established separately under explicit distributional assumptions.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt
from typing import Any


def _finite(value: float, name: str) -> float:
    value = float(value)
    if not isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return value


def positive_part(value: float) -> float:
    """Return the positive part of a scalar without changing its scale."""

    return max(_finite(value, "value"), 0.0)


def positive_degradation(target_risk: float, reference_target_risk: float) -> float:
    """Measure target-risk degradation relative to a fixed reference model.

    This is an evaluation quantity, not a training-time certificate input.
    """

    return positive_part(
        _finite(target_risk, "target_risk")
        - _finite(reference_target_risk, "reference_target_risk")
    )


def source_fit_excess(source_risk: float, erm_source_risk: float) -> float:
    """Return source-risk degradation relative to the source ERM reference."""

    return positive_part(
        _finite(source_risk, "source_risk")
        - _finite(erm_source_risk, "erm_source_risk")
    )


def metric_root(metric: float) -> float:
    """Return ``sqrt(metric)`` for a non-negative penalty value.

    Taking a root is only a reporting view.  It becomes part of a bound only
    after the relevant theorem fixes the penalty normalization.
    """

    metric = _finite(metric, "original_metric")
    if metric < 0.0:
        raise ValueError(f"original_metric must be non-negative, got {metric!r}")
    return sqrt(metric)


@dataclass(frozen=True)
class CrossDomainCertificate:
    """Inputs and optional evaluation labels for one algorithmic run.

    ``representation_covariate_shift``, ``head_mismatch``,
    ``source_domain_spread`` and ``complexity`` are diagnostics whose
    source-only status depends on the measurement protocol.  The two
    target-dependent decomposition terms and the target coverage residual are
    deliberately excluded from ``source_only_inputs``.
    """

    algorithm: str
    source_risk: float
    erm_source_risk: float
    original_metric: float
    target_risk: float | None = None
    erm_target_risk: float | None = None
    representation_covariate_shift: float | None = None
    conditional_label_shift: float | None = None
    representation_insufficiency: float | None = None
    head_mismatch: float | None = None
    source_domain_spread: float | None = None
    target_coverage_residual: float | None = None
    complexity: float | None = None

    def __post_init__(self) -> None:
        if not self.algorithm:
            raise ValueError("algorithm must be non-empty")
        _finite(self.source_risk, "source_risk")
        _finite(self.erm_source_risk, "erm_source_risk")
        metric_root(self.original_metric)
        for name in (
            "target_risk",
            "erm_target_risk",
            "representation_covariate_shift",
            "conditional_label_shift",
            "representation_insufficiency",
            "head_mismatch",
            "source_domain_spread",
            "target_coverage_residual",
            "complexity",
        ):
            value = getattr(self, name)
            if value is not None:
                _finite(value, name)

    @property
    def source_fit_excess(self) -> float:
        return source_fit_excess(self.source_risk, self.erm_source_risk)

    @property
    def target_degradation(self) -> float | None:
        if self.target_risk is None or self.erm_target_risk is None:
            return None
        return positive_degradation(self.target_risk, self.erm_target_risk)

    @property
    def omega_root(self) -> float:
        return metric_root(self.original_metric)

    def source_only_inputs(self) -> dict[str, float]:
        """Return only quantities available without target examples or labels."""

        values: dict[str, float] = {
            "original_metric": float(self.original_metric),
            "sqrt_original_metric": self.omega_root,
            "source_risk": float(self.source_risk),
            "erm_source_risk": float(self.erm_source_risk),
            "source_fit_excess": self.source_fit_excess,
        }
        optional_names = (
            "representation_covariate_shift",
            "head_mismatch",
            "source_domain_spread",
            "complexity",
        )
        for name in optional_names:
            value = getattr(self, name)
            if value is not None:
                values[name] = float(value)
        return values

    def metric_views(self) -> dict[str, dict[str, float]]:
        """Expose the three pre-registered certificate input views.

        The views are feature sets for comparison, not scalar upper bounds.
        ``conditional_label_shift``, ``representation_insufficiency`` and
        ``target_coverage_residual`` are intentionally excluded because they
        may depend on target responses, target support, or an unobservable
        coverage assumption.
        """

        source_inputs = self.source_only_inputs()
        return {
            "omega_only": {
                "original_metric": source_inputs["original_metric"],
                "sqrt_original_metric": source_inputs["sqrt_original_metric"],
            },
            "omega_plus_source_risk": {
                "original_metric": source_inputs["original_metric"],
                "sqrt_original_metric": source_inputs["sqrt_original_metric"],
                "source_risk": source_inputs["source_risk"],
                "source_fit_excess": source_inputs["source_fit_excess"],
            },
            "source_only_observable": source_inputs,
        }

    def as_record(self) -> dict[str, Any]:
        """Return a flat record suitable for a CSV or a dataframe."""

        record: dict[str, Any] = {
            "algorithm": self.algorithm,
            **self.source_only_inputs(),
            "conditional_label_shift": self.conditional_label_shift,
            "representation_insufficiency": self.representation_insufficiency,
            "target_degradation": self.target_degradation,
            "target_coverage_residual": self.target_coverage_residual,
        }
        return record
