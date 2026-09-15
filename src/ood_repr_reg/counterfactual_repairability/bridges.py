"""Bridges between local parameter controllers and response-space classes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from .classes import ResponseControllerClass
from .optimize import adaptive_repair_value

Array = np.ndarray


def induced_response_controller(jacobian: Array, parameter_controller: Array) -> Array:
    """Return the induced ``Pi = J_B L`` response map.

    This function is intentionally separate from the source-objective IFT
    response.  An IFT response is already a map ``Y -> Z``; it must not be
    multiplied by a second parameter Jacobian.
    """
    jac = np.asarray(jacobian, dtype=float)
    controller = np.asarray(parameter_controller, dtype=float)
    if jac.ndim != 2 or controller.ndim != 2 or jac.shape[1] != controller.shape[0]:
        raise ValueError("jacobian and parameter controller dimensions are incompatible")
    return jac @ controller


def induced_response_basis(jacobian: Array, parameter_controller_basis: Iterable[Array]) -> tuple[Array, ...]:
    return tuple(induced_response_controller(jacobian, value) for value in parameter_controller_basis)


@dataclass(frozen=True)
class ActualControllerBridge:
    """Membership and reconstruction audit for an actual response map."""

    actual_response: Array
    controller_class: ResponseControllerClass
    audit: dict[str, Any]

    @property
    def valid(self) -> bool:
        return bool(self.audit["contains"])

    @property
    def membership_residual(self) -> float:
        return float(self.audit["membership_residual"])


def audit_actual_controller(actual_response: Array, controller_class: ResponseControllerClass,
                            *, tolerance: float = 1e-9,
                            jacobian: Array | None = None,
                            parameter_controller: Array | None = None) -> ActualControllerBridge:
    """Check that an actual ``Pi : Y -> Z`` belongs to the declared class.

    When a parameter Jacobian and a parameter controller are supplied, their
    induced map is also compared to ``actual_response``.  This catches the
    common coordinate mistake of treating an IFT response as if it were still
    a parameter-space update.
    """
    actual = np.asarray(actual_response, dtype=float)
    if actual.shape != controller_class.response_shape:
        raise ValueError("actual response has incompatible Y -> Z shape")
    audit = dict(controller_class.membership_audit(actual, tolerance=tolerance))
    audit["coordinate_convention"] = "Pi: Y -> Z"
    audit["tolerance"] = float(tolerance)
    if jacobian is not None or parameter_controller is not None:
        if jacobian is None or parameter_controller is None:
            raise ValueError("jacobian and parameter_controller must be supplied together")
        induced = induced_response_controller(jacobian, parameter_controller)
        if induced.shape != actual.shape:
            raise ValueError("induced response and actual response have incompatible shapes")
        residual = induced - actual
        audit["induced_response"] = induced
        audit["induced_response_residual"] = float(np.linalg.norm(residual, ord="fro"))
        audit["induced_response_relative_residual"] = audit["induced_response_residual"] / max(
            1.0, float(np.linalg.norm(actual, ord="fro"))
        )
        audit["induced_response_matches"] = bool(
            audit["induced_response_relative_residual"] <= tolerance
        )
    else:
        audit["induced_response"] = None
        audit["induced_response_residual"] = None
        audit["induced_response_relative_residual"] = None
        audit["induced_response_matches"] = None
    return ActualControllerBridge(actual, controller_class, audit)


def realization_gap(response: Array, observation: Array, actual_response: Array,
                    class_optimum: Any, controller_class: ResponseControllerClass,
                    *, tolerance: float = 1e-9) -> dict[str, Any]:
    """Compute ``R_actual - R_class_star`` only after membership validation."""
    bridge = audit_actual_controller(actual_response, controller_class, tolerance=tolerance)
    actual_value = adaptive_repair_value(response, observation, actual_response)
    result = {
        "membership": bridge.audit,
        "R_actual": float(actual_value),
        "R_class_star": float(class_optimum.R_class_star),
        "Delta_real": float(actual_value - class_optimum.R_class_star),
        "valid": bridge.valid,
    }
    if not bridge.valid:
        result["Delta_real"] = None
        result["invalid_reason"] = "actual_response_not_in_declared_controller_class"
    return result


__all__ = [
    "induced_response_controller", "induced_response_basis", "ActualControllerBridge",
    "audit_actual_controller", "realization_gap",
]
