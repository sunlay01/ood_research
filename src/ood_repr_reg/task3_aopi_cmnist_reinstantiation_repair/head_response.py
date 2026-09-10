"""Secondary frozen-encoder head-equilibrium diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .smooth_world import SmoothWorlds, environment_parameters
from .task_response import HEAD_DIMENSION, augmented_head, encoded_outcomes, expected_head_risk


IRM_WEIGHT = 10000.0
L2_WEIGHT = 0.001
DAMPING_GRID = (1e-10, 1e-8, 1e-6)


@dataclass(frozen=True)
class HeadReference:
    weights: Tensor
    initial_weights: Tensor
    objective: float
    gradient_norm: float
    displacement: float
    converged: bool


def _source_objective(method: str, outcomes, theta: Tensor, weights: Tensor) -> Tensor:
    risks: list[Tensor] = []
    penalties: list[Tensor] = []
    for environment, item in enumerate(outcomes):
        p, q = environment_parameters(theta, environment=environment, evaluation=False)
        risk = expected_head_risk(item, p, q, weights)
        risks.append(risk)
        scale = torch.ones((), dtype=torch.double, requires_grad=True)
        scaled = expected_head_risk(item, p, q, torch.cat((weights[:-1] * scale, weights[-1:].clone() * scale)))
        penalties.append(torch.autograd.grad(scaled, scale, create_graph=True)[0].square())
    risk = torch.stack(risks).mean()
    l2 = weights.square().sum()
    if method == "ERM":
        return risk + L2_WEIGHT * l2
    if method == "IRMv1":
        return (risk + L2_WEIGHT * l2 + IRM_WEIGHT * torch.stack(penalties).mean()) / IRM_WEIGHT
    raise ValueError(f"unsupported method: {method}")


def _outcomes(model: torch.nn.Module, worlds: SmoothWorlds):
    return tuple(encoded_outcomes(model, pool) for pool in worlds.source)


def h0_reference(model: torch.nn.Module, worlds: SmoothWorlds, method: str) -> HeadReference:
    outcomes = _outcomes(model, worlds)
    theta = worlds.base_theta.detach().clone()
    weights = augmented_head(model).requires_grad_(True)
    objective = _source_objective(method, outcomes, theta, weights)
    gradient = torch.autograd.grad(objective, weights)[0]
    return HeadReference(weights=weights.detach(), initial_weights=weights.detach(), objective=float(objective.detach()), gradient_norm=float(gradient.norm().detach()), displacement=0.0, converged=False)


def refine_h1(model: torch.nn.Module, worlds: SmoothWorlds, method: str) -> HeadReference:
    outcomes = _outcomes(model, worlds)
    theta = worlds.base_theta.detach().clone()
    initial = augmented_head(model)
    weights = torch.nn.Parameter(initial.clone())
    optimizer = torch.optim.LBFGS([weights], lr=1.0, max_iter=100, tolerance_grad=1e-9, tolerance_change=1e-12, line_search_fn="strong_wolfe")

    def closure() -> Tensor:
        optimizer.zero_grad(set_to_none=True)
        objective = _source_objective(method, outcomes, theta, weights)
        objective.backward()
        return objective

    optimizer.step(closure)
    objective = _source_objective(method, outcomes, theta, weights)
    gradient = torch.autograd.grad(objective, weights)[0]
    gradient_norm = float(gradient.detach().norm())
    return HeadReference(weights=weights.detach().cpu().double(), initial_weights=initial, objective=float(objective.detach()), gradient_norm=gradient_norm, displacement=float((weights.detach() - initial).norm()), converged=bool(torch.isfinite(objective.detach()) and gradient_norm <= 1e-4))


def _objective_hessian(method: str, outcomes, theta: Tensor, weights: Tensor) -> Tensor:
    point = weights.detach().clone().requires_grad_(True)
    result = torch.autograd.functional.hessian(lambda value: _source_objective(method, outcomes, theta, value), point)
    return ((result + result.T) / 2.0).detach().cpu().double()


def head_pi_diagnostics(model: torch.nn.Module, worlds: SmoothWorlds, method: str, reference: HeadReference, source_root: Tensor) -> list[dict[str, float | bool]]:
    outcomes = _outcomes(model, worlds)
    theta = worlds.base_theta.detach().clone().requires_grad_(True)
    hessian = _objective_hessian(method, outcomes, theta, reference.weights)
    forcing = torch.autograd.functional.jacobian(
        lambda value: torch.autograd.grad(_source_objective(method, outcomes, value, reference.weights.requires_grad_(True)), reference.weights, create_graph=True)[0],
        theta,
    ).detach().cpu().double()
    scale = max(float(torch.trace(hessian).abs() / HEAD_DIMENSION), 1e-8)
    rows: list[dict[str, float | bool]] = []
    for relative_damping in DAMPING_GRID:
        effective = hessian + relative_damping * scale * torch.eye(HEAD_DIMENSION, dtype=torch.double)
        singular = torch.linalg.svdvals(effective)
        inverse = torch.linalg.solve(effective, forcing)
        predicted = -source_root @ inverse
        rows.append({
            "relative_damping": relative_damping,
            "min_singular_value": float(singular.min()),
            "max_singular_value": float(singular.max()),
            "condition_number": float(singular.max() / singular.min()),
            "effective_rank": int((singular > singular.max() * 1e-8).sum()),
            "predicted_response_norm": float(predicted.norm()),
            "numerically_singular": bool(singular.min() <= singular.max() * 1e-12),
        })
    return rows


def finite_head_response(model: torch.nn.Module, worlds: SmoothWorlds, method: str, reference: HeadReference, source_root: Tensor, *, delta: float = 0.01) -> list[dict[str, float | bool | str]]:
    """Validate H1 IFT against smooth-source nearby equilibria for the three basis directions."""
    outcomes = _outcomes(model, worlds)
    theta0 = worlds.base_theta.detach().clone()
    hessian = _objective_hessian(method, outcomes, theta0, reference.weights)
    scale = max(float(torch.trace(hessian).abs() / HEAD_DIMENSION), 1e-8)
    effective = hessian + 1e-8 * scale * torch.eye(HEAD_DIMENSION, dtype=torch.double)
    forcing = torch.autograd.functional.jacobian(
        lambda value: torch.autograd.grad(_source_objective(method, outcomes, value, reference.weights.requires_grad_(True)), reference.weights, create_graph=True)[0],
        theta0.requires_grad_(True),
    ).detach().cpu().double()
    rows: list[dict[str, float | bool | str]] = []
    for index, name in enumerate(("source_env0_color", "source_env1_color", "shared_label_noise")):
        direction = torch.zeros(3, dtype=torch.double); direction[index] = 1.0
        prediction = -source_root @ torch.linalg.solve(effective, forcing @ direction)
        def solve(theta: Tensor) -> Tensor:
            point = torch.nn.Parameter(reference.weights.detach().clone())
            optimizer = torch.optim.LBFGS([point], lr=1.0, max_iter=100, tolerance_grad=1e-9, tolerance_change=1e-12, line_search_fn="strong_wolfe")
            def closure() -> Tensor:
                optimizer.zero_grad(set_to_none=True)
                loss = _source_objective(method, outcomes, theta, point)
                loss.backward()
                return loss
            optimizer.step(closure)
            return point.detach()
        plus, minus = solve(theta0 + delta * direction), solve(theta0 - delta * direction)
        actual = source_root @ ((plus - minus) / (2.0 * delta))
        denom = max(float(prediction.norm() * actual.norm()), 1e-12)
        rows.append({"tangent": name, "predicted_response_norm": float(prediction.norm()), "actual_response_norm": float(actual.norm()), "cosine_similarity": float(prediction @ actual / denom), "relative_vector_error": float((prediction - actual).norm() / max(float(actual.norm()), 1e-12)), "finite": bool(torch.isfinite(actual).all() and torch.isfinite(prediction).all())})
    return rows
