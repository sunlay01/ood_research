"""Counterfactual repairability / response controllability diagnostics."""

from .bridges import (
    ActualControllerBridge,
    audit_actual_controller,
    induced_response_basis,
    induced_response_controller,
    realization_gap,
)
from .classes import ResponseControllerClass, standard_response_basis
from .diagnostics import (
    coordinate_invariance_audit,
    grid_search_coefficients,
    random_nested_class,
    run_phase0_diagnostics,
    solver_grid_audit,
    toy_head_full,
    toy_information_limited,
    toy_spectral_slack,
)
from .optimize import (
    RepairOptimizationResult,
    adaptive_repair_value,
    class_optimal_repair,
    optimize_controller,
    optimize_repair,
)

__all__ = [
    "ResponseControllerClass", "standard_response_basis",
    "RepairOptimizationResult", "adaptive_repair_value", "optimize_controller",
    "class_optimal_repair", "optimize_repair", "ActualControllerBridge",
    "audit_actual_controller", "induced_response_basis", "induced_response_controller",
    "realization_gap", "toy_head_full", "toy_information_limited", "toy_spectral_slack",
    "random_nested_class", "coordinate_invariance_audit", "grid_search_coefficients",
    "solver_grid_audit", "run_phase0_diagnostics",
]
