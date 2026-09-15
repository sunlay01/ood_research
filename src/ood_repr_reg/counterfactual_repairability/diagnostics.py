"""Small, deterministic Phase-0 diagnostics for repairability classes."""

from __future__ import annotations

from itertools import product
from typing import Any

import numpy as np

from ..round3r_3e_c_spectral import decompose_response
from .classes import ResponseControllerClass, standard_response_basis
from .optimize import RepairOptimizationResult, adaptive_repair_value, optimize_controller

Array = np.ndarray


def _result_summary(result: RepairOptimizationResult) -> dict[str, Any]:
    return {
        "R_info": result.R_info,
        "R_class_star": result.R_class_star,
        "control_gap": result.control_gap,
        "Pi_star": result.Pi_star,
        "solver_status": result.solver_status,
        "membership_residual": result.membership_residual,
        "slack_feasible": result.slack_feasible,
    }


def toy_head_full() -> dict[str, Any]:
    """Toy 1: head-only and full response reachability separate."""
    response = np.array([[1.0, 1.0]])
    observation = np.eye(2)
    head = ResponseControllerClass(
        [np.array([[1.0, 0.0]])], name="head-only",
    )
    full = ResponseControllerClass(standard_response_basis(1, 2), name="full")
    actual = np.array([[-0.5, -0.5]])
    head_result = optimize_controller(response, observation, head)
    full_result = optimize_controller(response, observation, full)
    actual_value = adaptive_repair_value(response, observation, actual)
    actual_membership = full.membership_audit(actual)
    return {
        "response": response,
        "observation": observation,
        "head_class": head,
        "full_class": full,
        "head": head_result,
        "full": full_result,
        "actual": actual,
        "R_actual": actual_value,
        "actual_membership": actual_membership,
        "V_repr": full_result.R_class_star * 0.0 + head_result.R_class_star - full_result.R_class_star,
        "Delta_real": actual_value - full_result.R_class_star,
        "summary": {
            "R_info": full_result.R_info,
            "R_head_star": head_result.R_class_star,
            "R_full_star": full_result.R_class_star,
            "R_actual": actual_value,
            "V_repr": head_result.R_class_star - full_result.R_class_star,
            "Delta_real": actual_value - full_result.R_class_star,
        },
    }


def toy_information_limited() -> dict[str, Any]:
    """Toy 2: no source observation leaves every controller at the floor."""
    response = np.array([[1.0, 1.0]])
    observation = np.zeros((2, 2))
    full = ResponseControllerClass(standard_response_basis(1, 2), name="full-zero-observation")
    result = optimize_controller(response, observation, full)
    return {"response": response, "observation": observation, "full": result,
            "summary": _result_summary(result)}


def toy_spectral_slack() -> dict[str, Any]:
    """Toy 3: nonzero recoverable ``E`` still attains the information floor."""
    response = np.array([[0.0, 1.0], [1.0, 0.0]])
    observation = np.array([[1.0, 0.0]])
    zero = ResponseControllerClass([], response_shape=(2, 1), name="zero-controller")
    result = optimize_controller(response, observation, zero)
    pieces = decompose_response(response, observation)
    e = np.asarray(pieces["A_recoverable"]) + result.Pi_star @ observation
    return {
        "response": response,
        "observation": observation,
        "zero": result,
        "E": e,
        "A_irreducible": pieces["A_irreducible"],
        "summary": {
            "R_info": result.R_info,
            "R_class_star": result.R_class_star,
            "E_norm": float(np.linalg.norm(e, ord=2)),
            "condition_holds": bool(result.spectral_certificate["condition_holds"]),
            "support_compatible": bool(result.spectral_certificate["support_compatible"]),
        },
    }


def random_nested_class(seed: int, *, response_dimension: int = 2,
                        source_dimension: int = 2, world_dimension: int = 3,
                        shared_dimension: int = 1, extra_dimension: int = 2) -> dict[str, Any]:
    """Generate a reproducible pair of nested response-space classes."""
    rng = np.random.default_rng(seed)
    response = rng.normal(size=(response_dimension, world_dimension))
    observation = rng.normal(size=(source_dimension, world_dimension))
    common = [rng.normal(size=(response_dimension, source_dimension)) for _ in range(shared_dimension)]
    extras = [rng.normal(size=(response_dimension, source_dimension)) for _ in range(extra_dimension)]
    first = ResponseControllerClass(common, name=f"nested-{seed}-small")
    second = ResponseControllerClass(common + extras, name=f"nested-{seed}-large")
    small = optimize_controller(response, observation, first)
    large = optimize_controller(response, observation, second)
    return {"seed": seed, "response": response, "observation": observation,
            "small_class": first, "large_class": second,
            "small": small, "large": large,
            "monotone": bool(large.R_class_star <= small.R_class_star + 1e-7)}


def coordinate_invariance_audit(seed: int = 0) -> dict[str, Any]:
    """Audit source-coordinate and response-isometry transport."""
    rng = np.random.default_rng(seed)
    response = rng.normal(size=(2, 3))
    observation = rng.normal(size=(2, 3))
    basis = [rng.normal(size=(2, 2)) for _ in range(3)]
    controller = ResponseControllerClass(basis, name="coordinate-base")
    base = optimize_controller(response, observation, controller)
    source_transform = rng.normal(size=(2, 2))
    while abs(np.linalg.det(source_transform)) < 0.2:
        source_transform = rng.normal(size=(2, 2))
    qr, _ = np.linalg.qr(rng.normal(size=(2, 2)))
    transformed_controller = controller.transported(
        source_transform=source_transform, response_transform=qr,
    )
    transformed = optimize_controller(
        qr @ response, source_transform @ observation, transformed_controller,
    )
    errors = {
        "R_info": abs(base.R_info - transformed.R_info),
        "R_class_star": abs(base.R_class_star - transformed.R_class_star),
        "control_gap": abs(base.control_gap - transformed.control_gap),
        "operator_residual": abs(
            np.linalg.norm(response + base.Pi_star @ observation, ord=2)
            - np.linalg.norm(qr @ response + transformed.Pi_star @ (source_transform @ observation), ord=2)
        ),
    }
    return {"base": base, "transformed": transformed,
            "source_transform": source_transform, "response_transform": qr,
            "errors": errors, "pass": bool(max(errors.values()) <= 1e-6)}


def grid_search_coefficients(response: Array, observation: Array,
                             controller_class: ResponseControllerClass,
                             *, radius: float | None = None, points: int = 31) -> dict[str, Any]:
    """Brute-force coefficient audit for classes of dimension at most three."""
    if controller_class.dimension > 3:
        raise ValueError("grid_search_coefficients is limited to at most three coefficients")
    if controller_class.dimension == 0:
        value = adaptive_repair_value(response, observation, controller_class.response_operator([]))
        return {"coefficients": np.empty(0), "value": value, "evaluated": 1}
    if radius is None:
        radius = controller_class.budget_radius if controller_class.budget_type != "none" else 2.0
    radius = float(radius)
    if radius <= 0.0:
        grid = np.zeros(1)
    else:
        grid = np.linspace(-radius, radius, int(points))
    best_coefficients = None
    best_value = float("inf")
    evaluated = 0
    for values in product(grid, repeat=controller_class.dimension):
        coefficients = np.asarray(values, dtype=float)
        operator = controller_class.response_operator(coefficients)
        if not controller_class.is_feasible(operator, tolerance=1e-7):
            continue
        evaluated += 1
        value = adaptive_repair_value(response, observation, operator)
        if value < best_value:
            best_value = value
            best_coefficients = coefficients
    if best_coefficients is None:
        raise RuntimeError("grid contained no feasible controller")
    return {"coefficients": best_coefficients, "value": float(best_value), "evaluated": evaluated,
            "radius": radius, "points": int(points)}


def solver_grid_audit(response: Array, observation: Array,
                      controller_class: ResponseControllerClass, *, points: int = 31,
                      tolerance: float = 0.05) -> dict[str, Any]:
    """Compare the optimizer with a dense tiny-class grid search."""
    result = optimize_controller(response, observation, controller_class)
    radius = controller_class.budget_radius if controller_class.budget_type != "none" else None
    if radius is None:
        coefficient_norm = float(np.linalg.norm(result.coefficients, ord=np.inf)) if result.coefficients.size else 0.0
        radius = max(2.0, 1.5 * coefficient_norm + 0.5)
    grid = grid_search_coefficients(response, observation, controller_class, radius=radius, points=points)
    difference = float(result.R_class_star - grid["value"])
    return {"optimizer": result, "grid": grid, "optimizer_minus_grid": difference,
            "pass": bool(abs(difference) <= tolerance * max(1.0, grid["value"]))}


def run_phase0_diagnostics(*, random_instances: int = 100, seed: int = 20260915) -> dict[str, Any]:
    """Run all deterministic Phase-0 checks and return JSON-friendly status."""
    separation = toy_head_full()
    information = toy_information_limited()
    spectral = toy_spectral_slack()
    rng = np.random.default_rng(seed)
    nested = [random_nested_class(int(value)) for value in rng.integers(0, 2**31 - 1, random_instances)]
    nested_pass = all(item["monotone"] for item in nested)
    coordinate = coordinate_invariance_audit(seed)
    solver_audit = solver_grid_audit(
        np.array([[1.0, 1.0]]), np.eye(2),
        ResponseControllerClass([np.array([[1.0, 0.0]])], name="grid-head"), points=101,
    )
    checks = {
        "toy_head_full": bool(
            abs(separation["head"].R_info) < 1e-10
            and abs(separation["head"].R_class_star - 0.5) < 1e-6
            and abs(separation["full"].R_class_star) < 1e-6
            and abs(separation["R_actual"] - 0.25) < 1e-6
        ),
        "toy_information_limited": bool(
            abs(information["full"].R_info - 1.0) < 1e-8
            and abs(information["full"].R_class_star - information["full"].R_info) < 1e-7
        ),
        "toy_spectral_slack": bool(
            spectral["summary"]["E_norm"] > 1e-8
            and abs(spectral["zero"].R_class_star - spectral["zero"].R_info) < 1e-7
            and spectral["summary"]["condition_holds"]
        ),
        "nested_monotonicity": bool(nested_pass),
        "coordinate_invariance": bool(coordinate["pass"]),
        "solver_grid": bool(solver_audit["pass"]),
    }
    return {
        "seed": seed,
        "random_instances": random_instances,
        "checks": checks,
        "pass": bool(all(checks.values())),
        "separation": {**separation["summary"], "head_solver": separation["head"].solver_status,
                        "full_solver": separation["full"].solver_status},
        "information_limited": information["summary"],
        "spectral_slack": spectral["summary"],
        "nested_monotonicity": {"pass": nested_pass, "count": len(nested)},
        "coordinate_invariance": {"pass": coordinate["pass"], "errors": coordinate["errors"]},
        "solver_grid": {"pass": solver_audit["pass"],
                         "optimizer_minus_grid": solver_audit["optimizer_minus_grid"]},
    }


__all__ = [
    "toy_head_full", "toy_information_limited", "toy_spectral_slack",
    "random_nested_class", "coordinate_invariance_audit", "grid_search_coefficients",
    "solver_grid_audit", "run_phase0_diagnostics",
]
