"""Frozen 3A/3B population objects adapted for the 3C audit."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .round3r_3b_benchmark import (
    ModuleEnvironment,
    MomentState,
    ShiftProbe,
    make_probe_set,
    risk,
    risk_gradient,
    source_design,
    source_optimum,
)

Array = np.ndarray


@dataclass(frozen=True)
class ThreeCBenchmark:
    base: ModuleEnvironment
    source_environments: tuple[ModuleEnvironment, ...]
    source: MomentState
    optimum: Array
    probes: tuple[ShiftProbe, ...]
    gradients: Array
    hessian: Array
    q_responses: Array
    relevant_indices: Array


def symmetric_inverse_sqrt(matrix: Array) -> Array:
    value = (np.asarray(matrix, dtype=float) + np.asarray(matrix, dtype=float).T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(value)
    if eigenvalues.size == 0 or eigenvalues.min() <= 0:
        raise ValueError("matrix must be positive definite")
    return eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues)) @ eigenvectors.T


def make_benchmark(n_noise: int = 4, shortcut_count: int = 2,
                   relation_exposed: bool = True, *, family=None) -> ThreeCBenchmark:
    """Build the legacy benchmark or a benchmark supplied by a family.

    The default path is intentionally unchanged.  A family supplies only the
    admissible reference/source geometry; probes and their post-hoc metadata
    remain the frozen population fixtures.
    """
    if family is None:
        base = ModuleEnvironment(
            shortcut_rhos=tuple(0.75 - 0.18 * i for i in range(shortcut_count)),
            shortcut_means=tuple(0.18 - 0.10 * i for i in range(shortcut_count)),
            shortcut_variances=tuple(0.49 + 0.12 * i for i in range(shortcut_count)),
            n_noise=n_noise,
            u_gamma=0.0,
        )
        environments = source_design(base, relation_exposed=relation_exposed)
    else:
        from .environment_family.geometry import build_task_geometry

        geometry = build_task_geometry(family)
        base = geometry.reference
        environments = geometry.source_environments
    optimum, source = source_optimum(environments)
    probes = make_probe_set(base, source)
    gradients = np.column_stack([risk_gradient(optimum, source, probe.target) for probe in probes])
    hessian = 2.0 * source.second
    inverse_root = symmetric_inverse_sqrt(hessian)
    q_responses = inverse_root @ gradients
    diagonal = np.sum(q_responses * q_responses, axis=0)
    scale = max(float(diagonal.max(initial=0.0)), 1e-30)
    relevant = np.flatnonzero(diagonal > 1e-10 * scale)
    return ThreeCBenchmark(base, environments, source, optimum, probes, gradients,
                           hessian, q_responses, relevant)


def source_risk(benchmark: ThreeCBenchmark, weights: Array) -> float:
    return float(np.mean([risk(weights, state) for state in _environment_states(benchmark)]))


def _environment_states(benchmark: ThreeCBenchmark) -> tuple[MomentState, ...]:
    from .round3r_3b_benchmark import environment_state

    return tuple(environment_state(environment) for environment in benchmark.source_environments)


def environment_risks(weights: Array, environments: tuple[MomentState, ...]) -> Array:
    return np.asarray([risk(weights, environment) for environment in environments], dtype=float)


def source_gradient(benchmark: ThreeCBenchmark, weights: Array) -> Array:
    return 2.0 * (benchmark.source.second @ np.asarray(weights) - benchmark.source.xy)


def target_shift_curvature(benchmark: ThreeCBenchmark, probe: ShiftProbe) -> Array:
    return 2.0 * (probe.target.second - benchmark.source.second)


def target_risk(benchmark: ThreeCBenchmark, probe: ShiftProbe, weights: Array) -> float:
    return risk(weights, probe.target)


def response_records(benchmark: ThreeCBenchmark) -> list[dict[str, object]]:
    rows = []
    for index, probe in enumerate(benchmark.probes):
        rows.append({
            "shift_index": index,
            "shift_id": probe.shift_id,
            "relevance_squared": float(benchmark.q_responses[:, index] @ benchmark.q_responses[:, index]),
            "relevance_norm": float(np.linalg.norm(benchmark.q_responses[:, index])),
            "is_relevant": bool(index in set(benchmark.relevant_indices.tolist())),
            "oracle_mechanism_posthoc": probe.mechanism,
            "intervention_family_posthoc": probe.intervention_family,
        })
    return rows
