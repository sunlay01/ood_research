"""Class-optimal centered adaptive repair.

The objective is the finite-dimensional quantity

``R_adap(Pi) = 1/2 * ||A + Pi O_S||_op^2``.

This module deliberately returns a certificate object instead of hiding the
optimization behind a scalar.  In particular, a realization gap is only
interpretable after the returned membership residual has been audited.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..round3r_3e_c_spectral import adaptive_minimax_certificate, decompose_response
from ..round3r_3e_recovery import minimax_recovery_error, operator_norm
from .classes import ResponseControllerClass

Array = np.ndarray


def _pair(response: Array, observation: Array) -> tuple[Array, Array]:
    a = np.asarray(response, dtype=float)
    o = np.asarray(observation, dtype=float)
    if a.ndim != 2 or o.ndim != 2 or a.shape[1] != o.shape[1]:
        raise ValueError("A and O_S must be matrices sharing the world dimension")
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(o)):
        raise ValueError("A and O_S must contain finite values")
    return a, o


def adaptive_repair_value(response: Array, observation: Array, controller: Array) -> float:
    """Evaluate the centered adaptive regret for one response controller."""
    a, o = _pair(response, observation)
    pi = np.asarray(controller, dtype=float)
    if pi.ndim != 2 or pi.shape != (a.shape[0], o.shape[0]):
        raise ValueError("controller must have shape (dim_Z, dim_Y)")
    return 0.5 * operator_norm(a + pi @ o) ** 2


@dataclass(frozen=True)
class RepairOptimizationResult:
    """Full result of a finite-dimensional class optimization."""

    R_info: float
    R_class_star: float
    Pi_star: Array
    control_gap: float
    slack_feasible: bool
    solver_certificate: dict[str, Any]
    coefficients: Array
    solver_status: str
    membership_residual: float
    spectral_certificate: dict[str, Any]

    @property
    def information_floor(self) -> float:
        return self.R_info

    @property
    def class_optimal_value(self) -> float:
        return self.R_class_star

    def as_dict(self) -> dict[str, Any]:
        return {
            "R_info": self.R_info,
            "R_class_star": self.R_class_star,
            "Pi_star": self.Pi_star,
            "control_gap": self.control_gap,
            "slack_feasible": self.slack_feasible,
            "solver_certificate": self.solver_certificate,
            "coefficients": self.coefficients,
            "solver_status": self.solver_status,
            "membership_residual": self.membership_residual,
            "spectral_certificate": self.spectral_certificate,
        }


def _bounded_candidate(controller_class: ResponseControllerClass, candidate: Array) -> Array:
    """Project a least-squares candidate onto a response-space budget."""
    value = np.asarray(candidate, dtype=float)
    if controller_class.budget_type == "none":
        return value
    size = controller_class.budget_value(value)
    if size <= controller_class.budget_radius or size == 0.0:
        return value
    return value * (controller_class.budget_radius / size)


def _scipy_optimize(response: Array, observation: Array,
                    controller_class: ResponseControllerClass,
                    starts: list[Array], tolerance: float,
                    maxiter: int) -> tuple[Array, dict[str, Any]]:
    try:
        from scipy.optimize import minimize
    except ImportError as exc:  # pragma: no cover - exercised only minimal installs
        raise RuntimeError("scipy is required for a nontrivial bounded controller class") from exc

    dimension = controller_class.dimension
    best_coefficients: Array | None = None
    best_value = float("inf")
    records: list[dict[str, Any]] = []

    def objective(coefficients: Array) -> float:
        return adaptive_repair_value(response, observation, controller_class.response_operator(coefficients))

    constraints = []
    if controller_class.budget_type != "none":
        constraints.append({
            "type": "ineq",
            "fun": lambda coefficients: controller_class.budget_residual(
                controller_class.response_operator(coefficients)
            ),
        })
    for start in starts:
        initial = np.asarray(start, dtype=float).reshape(dimension)
        result = minimize(
            objective,
            initial,
            method="SLSQP",
            constraints=constraints,
            options={"maxiter": int(maxiter), "ftol": max(tolerance * 0.1, 1e-13), "disp": False},
        )
        coefficients = np.asarray(result.x, dtype=float).reshape(dimension)
        value = objective(coefficients)
        feasible = controller_class.is_feasible(controller_class.response_operator(coefficients), tolerance=1e-7)
        records.append({
            "method": "SLSQP",
            "success": bool(result.success),
            "message": str(result.message),
            "iterations": int(getattr(result, "nit", 0)),
            "objective": float(value),
            "feasible": bool(feasible),
        })
        if feasible and value < best_value:
            best_value = float(value)
            best_coefficients = coefficients
    if best_coefficients is None:
        raise RuntimeError(f"controller optimization found no feasible solution: {records}")
    return best_coefficients, {"method": "SLSQP", "starts": records}


def optimize_controller(response: Array, observation: Array,
                        controller_class: ResponseControllerClass,
                        *, tolerance: float = 1e-9,
                        maxiter: int = 2000) -> RepairOptimizationResult:
    """Optimize centered adaptive repair over a declared controller class.

    The class is finite-dimensional and response-space based.  A projected
    least-squares recoverer is always evaluated as a candidate; SLSQP then
    audits the spectral-norm objective for restricted classes.  This keeps the
    exact full-information toy deterministic while still supporting bounded
    response classes.
    """
    a, o = _pair(response, observation)
    expected_shape = (a.shape[0], o.shape[0])
    if controller_class.response_shape != expected_shape:
        raise ValueError(
            f"controller class shape {controller_class.response_shape} does not match {expected_shape}"
        )
    pieces = decompose_response(a, o, tolerance)
    irreducible = np.asarray(pieces["A_irreducible"], dtype=float)
    recoverable = np.asarray(pieces["A_recoverable"], dtype=float)
    r_info = 0.5 * minimax_recovery_error(a, o, tolerance) ** 2

    # The Moore-Penrose response is the canonical unconstrained source-only
    # recoverer.  It is a useful exact candidate even when the declared class
    # can only approximate it.
    unconstrained = -a @ np.linalg.pinv(o)
    candidates: list[Array] = [np.zeros(expected_shape)]
    coefficients, projected, _ = controller_class.coefficients_for(unconstrained)
    projected = _bounded_candidate(controller_class, projected)
    projected_coefficients, _, _ = controller_class.coefficients_for(projected)
    candidates.append(projected_coefficients)
    starts = [np.zeros(controller_class.dimension), coefficients, projected_coefficients]

    if controller_class.dimension == 0:
        best_coefficients = np.empty(0, dtype=float)
        solver_certificate: dict[str, Any] = {"method": "zero-class", "starts": []}
    else:
        # A class can contain a candidate exactly, in which case no numerical
        # optimizer is needed to establish the floor.  We still record a
        # solver audit below for nontrivial classes.
        best_coefficients = min(
            (np.asarray(value, dtype=float) for value in starts),
            key=lambda value: adaptive_repair_value(a, o, controller_class.response_operator(value))
            if controller_class.is_feasible(controller_class.response_operator(value), tolerance=1e-7)
            else float("inf"),
        )
        solver_certificate = {"method": "candidate-plus-SLSQP", "candidate_count": len(starts)}
        try:
            numerical, audit = _scipy_optimize(a, o, controller_class, starts, tolerance, maxiter)
            numerical_value = adaptive_repair_value(a, o, controller_class.response_operator(numerical))
            best_value = adaptive_repair_value(a, o, controller_class.response_operator(best_coefficients))
            if numerical_value <= best_value + tolerance:
                best_coefficients = numerical
            solver_certificate.update(audit)
        except RuntimeError as exc:
            # Unbounded pure-numpy use remains valid for an exact candidate;
            # surface the fallback in the certificate instead of hiding it.
            solver_certificate.update({"fallback": str(exc)})

    # Numerical constrained solvers can end a few ulps outside a response
    # budget.  Radially project that final response back into the declared
    # functional class before producing certificates.
    pi_star = _bounded_candidate(
        controller_class, controller_class.response_operator(best_coefficients)
    )
    best_coefficients, _, _ = controller_class.coefficients_for(pi_star)
    class_value = adaptive_repair_value(a, o, pi_star)
    membership_tolerance = max(float(tolerance), 1e-8)
    membership = controller_class.membership_audit(pi_star, tolerance=membership_tolerance)
    e = recoverable + pi_star @ o
    spectral = adaptive_minimax_certificate(irreducible, e, tolerance)
    slack_feasible = bool(spectral["condition_holds"] and spectral["support_compatible"])
    scale = max(1.0, r_info, class_value)
    if class_value < r_info - 100.0 * tolerance * scale:
        solver_status = "INVALID-BELOW-INFORMATION-FLOOR"
    elif not bool(membership["contains"]):
        solver_status = "INVALID-MEMBERSHIP"
    else:
        solver_status = "SUCCESS"
    control_gap = max(0.0, float(class_value - r_info))
    solver_certificate.update({
        "information_floor_lower_bound": float(r_info),
        "objective_value": float(class_value),
        "objective_residual": float(np.linalg.norm(a + pi_star @ o, ord=2)),
        "rank_A_irreducible": int(np.linalg.matrix_rank(irreducible, tol=tolerance)),
        "rank_observation": int(np.linalg.matrix_rank(o, tol=tolerance)),
        "slack_psd_min": float(spectral["psd_min_gap"]),
        "slack_feasible": slack_feasible,
        "support_compatible": bool(spectral["support_compatible"]),
        "unconstrained_recoverer_membership_residual": controller_class.membership_residual(unconstrained),
        "candidate_value": float(adaptive_repair_value(a, o, controller_class.response_operator(starts[0]))),
        "class_dimension": controller_class.dimension,
        "response_shape": controller_class.response_shape,
        "tolerance": float(tolerance),
    })
    return RepairOptimizationResult(
        R_info=float(r_info),
        R_class_star=float(class_value),
        Pi_star=pi_star,
        control_gap=float(control_gap),
        slack_feasible=slack_feasible,
        solver_certificate=solver_certificate,
        coefficients=np.asarray(best_coefficients, dtype=float),
        solver_status=solver_status,
        membership_residual=float(membership["membership_residual"]),
        spectral_certificate=spectral,
    )


# Descriptive aliases used by the protocol and later bridge code.
class_optimal_repair = optimize_controller
optimize_repair = optimize_controller


__all__ = [
    "RepairOptimizationResult",
    "adaptive_repair_value",
    "optimize_controller",
    "class_optimal_repair",
    "optimize_repair",
]
