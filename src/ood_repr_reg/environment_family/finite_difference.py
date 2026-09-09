"""Boundary-safe finite differences for declared environment families."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .base import Environment, EnvironmentFamily, Role

Array = np.ndarray


class IllegalFiniteDifferenceStep(ValueError):
    """Raised when no legal finite-difference stencil exists."""


@dataclass(frozen=True)
class FiniteDifferenceDiagnostic:
    direction: str
    requested_step: float
    used_step: float
    scheme: str
    legal_plus: bool
    legal_minus: bool
    distance_to_boundary: dict[str, float]

    def as_dict(self) -> dict[str, object]:
        return {
            "direction": self.direction,
            "requested_step": self.requested_step,
            "used_step": self.used_step,
            "scheme": self.scheme,
            "legal_plus": self.legal_plus,
            "legal_minus": self.legal_minus,
            "distance_to_boundary": dict(self.distance_to_boundary),
        }


def _interval(family: EnvironmentFamily, reference: Environment,
              direction: str, role: Role) -> tuple[float, float]:
    method = getattr(family, "legal_step_interval", None)
    if method is None:
        raise IllegalFiniteDifferenceStep(
            f"family {type(family).__name__} does not declare legal_step_interval"
        )
    negative, positive = method(reference, direction, role=role)
    negative, positive = float(negative), float(positive)
    if negative < 0.0 or positive < 0.0:
        raise ValueError("legal step radii must be nonnegative")
    return negative, positive


def family_directional_derivative(
    family: EnvironmentFamily,
    reference: Environment,
    direction: str,
    evaluate: Callable[[Environment], Array],
    requested_step: float,
    *,
    role: Role,
) -> tuple[Array, FiniteDifferenceDiagnostic]:
    """Evaluate a second-order stencil without sampling illegal environments.

    Central differences are retained whenever the requested step is legal on
    both sides.  At a one-sided boundary, the three-point one-sided stencil is
    used.  The small safety factor keeps the second point inside an open
    probability/variance domain when the reported radius is a boundary.
    """
    h_requested = float(requested_step)
    if h_requested <= 0.0 or not np.isfinite(h_requested):
        raise ValueError("requested_step must be finite and positive")
    negative, positive = _interval(family, reference, direction, role)
    safety = 0.5

    if negative >= h_requested and positive >= h_requested:
        h, scheme = h_requested, "central"
    elif negative > 0.0 and positive > 0.0:
        h = min(h_requested, negative, positive) * safety
        scheme = "central_shrunk"
    elif positive > 0.0:
        h = min(h_requested, positive / 2.0) * safety
        scheme = "forward_second_order"
    elif negative > 0.0:
        h = min(h_requested, negative / 2.0) * safety
        scheme = "backward_second_order"
    else:
        raise IllegalFiniteDifferenceStep(
            f"no legal finite-difference step for {direction!r} at the declared boundary"
        )

    if h <= 0.0 or not np.isfinite(h):
        raise IllegalFiniteDifferenceStep(f"no finite step available for {direction!r}")

    base = np.asarray(evaluate(reference), dtype=float)
    plus_legal = positive >= h
    minus_legal = negative >= h
    if scheme.startswith("central"):
        if not (plus_legal and minus_legal):
            raise IllegalFiniteDifferenceStep(f"central stencil became illegal for {direction!r}")
        plus = np.asarray(evaluate(family.perturb(reference, direction, h, role=role)), dtype=float)
        minus = np.asarray(evaluate(family.perturb(reference, direction, -h, role=role)), dtype=float)
        derivative = (plus - minus) / (2.0 * h)
    elif scheme == "forward_second_order":
        if positive < 2.0 * h:
            raise IllegalFiniteDifferenceStep(f"forward stencil became illegal for {direction!r}")
        first = np.asarray(evaluate(family.perturb(reference, direction, h, role=role)), dtype=float)
        second = np.asarray(evaluate(family.perturb(reference, direction, 2.0 * h, role=role)), dtype=float)
        derivative = (-3.0 * base + 4.0 * first - second) / (2.0 * h)
    else:
        if negative < 2.0 * h:
            raise IllegalFiniteDifferenceStep(f"backward stencil became illegal for {direction!r}")
        first = np.asarray(evaluate(family.perturb(reference, direction, -h, role=role)), dtype=float)
        second = np.asarray(evaluate(family.perturb(reference, direction, -2.0 * h, role=role)), dtype=float)
        derivative = (3.0 * base - 4.0 * first + second) / (2.0 * h)

    diagnostic = FiniteDifferenceDiagnostic(
        direction=direction,
        requested_step=h_requested,
        used_step=h,
        scheme=scheme,
        legal_plus=plus_legal,
        legal_minus=minus_legal,
        distance_to_boundary={"negative": negative, "positive": positive},
    )
    return derivative, diagnostic


__all__ = ["IllegalFiniteDifferenceStep", "FiniteDifferenceDiagnostic", "family_directional_derivative"]
