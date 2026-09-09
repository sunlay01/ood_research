"""Source-only adapters for the Gaussian and frozen-head CMNIST audits."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..cmnist_geometry_bridge import (
    CMNISTFamily,
    RepresentationBank,
    build_geometry,
    head_from_state,
    head_objective_gradient,
    ift_matrices,
    source_observation,
    source_state_stack,
)
from ..round3r_3c_affine import (
    _solve_regularized,
    _source_state_value_gradient,
    exact_ift_affine,
    regularizer_value_gradient,
    source_task_state_matrix,
    symmetric_inverse_sqrt,
    symmetric_sqrt,
)
from ..round3r_3c_benchmark import ThreeCBenchmark
from ..round3r_3e_c_benchmarks import LegalWorld

Array = np.ndarray


@dataclass(frozen=True)
class MechanismInput:
    setting: str
    method: str
    lam: float
    weights: Array
    common_base_weights: Array
    state: Array
    observation: Array
    observed_pi: Array
    observed_total_h: Array
    observed_total_b: Array
    response: Array
    response_adaptive: Array
    response_offset: Array
    response_transform: Array
    risk_gradient: object
    penalty_gradient: object
    solver: object
    source_dimension: int


def gaussian_input(world: LegalWorld, method: str, lam: float) -> MechanismInput:
    """Build raw-predictor-coordinate source-only mechanism quantities."""
    benchmark: ThreeCBenchmark = world.benchmark
    affine = exact_ift_affine(benchmark, method, lam, world.observation)
    if not affine.valid:
        raise ValueError(f"invalid exact IFT row: {affine.solver_status}")
    state = source_task_state_matrix(benchmark)
    p = benchmark.optimum.size
    h_inv_root = symmetric_inverse_sqrt(benchmark.hessian)
    h_root = symmetric_sqrt(benchmark.hessian)
    observed_pi = h_inv_root @ affine.pi
    observed_total_h = h_root @ affine.dz_f @ h_root
    observed_total_b = h_root @ affine.dy_f

    def risk_gradient(w: Array, y: Array) -> Array:
        return _source_state_value_gradient(w, y, p)[1]

    def penalty_gradient(w: Array, y: Array) -> Array:
        return regularizer_value_gradient(method, w, y, p)[1]

    def solver(local_lam: float, y: Array) -> Array:
        weights, valid, status = _solve_regularized(benchmark, method, local_lam, y)
        if not valid:
            raise ValueError(status)
        return weights

    return MechanismInput(
        setting=world.name, method=method.upper(), lam=float(lam), weights=affine.weights,
        common_base_weights=benchmark.optimum,
        state=state, observation=world.observation, observed_pi=observed_pi,
        observed_total_h=observed_total_h, observed_total_b=observed_total_b,
        response=world.response, response_adaptive=affine.tangent,
        response_offset=affine.z0, response_transform=h_root, risk_gradient=risk_gradient,
        penalty_gradient=penalty_gradient, solver=solver, source_dimension=p,
    )


def cmnist_input(bank: RepresentationBank, family: CMNISTFamily, method: str,
                 lam: float) -> MechanismInput:
    """Build the corresponding empirical frozen-head source-only quantities."""
    state = source_state_stack(bank, family, np.zeros(family.dimension))
    p = bank.dimension
    weights = head_from_state(state, p, method=method, lam=lam)
    erm = head_from_state(state, p, method="erm", lam=0.0)
    dw, dy = ift_matrices(state, p, method, lam, weights)
    eigen = np.linalg.eigvalsh((dw + dw.T) / 2.0)
    if eigen.size == 0 or eigen.min() <= 1e-10:
        raise ValueError("non-positive local head metric")
    observed_pi = -np.linalg.solve(dw, dy)
    # Source-head whitening is a source-only coordinate conversion.
    from ..cmnist_geometry_bridge import moment_state
    source_matrix = np.mean([moment_state(bank, rho)[0] for rho in family.source_rhos], axis=0)
    values, vectors = np.linalg.eigh(2.0 * source_matrix)
    h_root = vectors @ np.diag(np.sqrt(values)) @ vectors.T
    observation = source_observation(bank, family)

    def risk_gradient(w: Array, y: Array) -> Array:
        return head_objective_gradient(w, y, p, "erm", 0.0)

    def penalty_gradient(w: Array, y: Array) -> Array:
        return head_objective_gradient(w, y, p, method, 1.0) - risk_gradient(w, y)

    def solver(local_lam: float, y: Array) -> Array:
        return head_from_state(y, p, method=method, lam=local_lam)

    # Attach the target-side response only after source-side construction.
    geometry = build_geometry(bank, family)
    return MechanismInput(
        setting=family.name, method=method.upper(), lam=float(lam), weights=weights,
        common_base_weights=erm,
        state=state, observation=observation, observed_pi=observed_pi,
        observed_total_h=dw, observed_total_b=dy,
        response=np.asarray(geometry["A"]), response_adaptive=h_root @ observed_pi @ observation,
        response_offset=h_root @ (weights - erm), response_transform=h_root, risk_gradient=risk_gradient,
        penalty_gradient=penalty_gradient, solver=solver, source_dimension=p,
    )


__all__ = ["MechanismInput", "gaussian_input", "cmnist_input"]
