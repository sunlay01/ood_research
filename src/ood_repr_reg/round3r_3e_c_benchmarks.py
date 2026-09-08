"""Legal environment-derived benchmarks for 3E-C."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .round3r_3b_benchmark import ModuleEnvironment, source_optimum
from .round3r_3c_benchmark import ThreeCBenchmark
from .round3r_3c_affine import exact_ift_affine
from .round3r_3e_joint_regret import regret_decomposition
from .round3r_3e_world_tangent import (
    WorldTangentSpec,
    default_environment,
    frozen_response_jacobian,
    source_observation_block,
    coupled_primary_geometry,
    u_exposed_observation,
)
from .round3r_3e_c_optimality import full_affine_certificate
from .round3r_3e_c_spectral import decompose_response

Array = np.ndarray


@dataclass(frozen=True)
class LegalWorld:
    name: str
    base: ModuleEnvironment
    source_environments: tuple[ModuleEnvironment, ...]
    benchmark: ThreeCBenchmark
    observation: Array
    response: Array
    spec: WorldTangentSpec


def family_world(family, name: str | None = None) -> LegalWorld:
    """Adapt a shared family geometry to the legacy 3E-C audit contract.

    The sharp 3E-C algebra only consumes ``(A, O_S, z0, Pi O_S)``.  This
    adapter therefore lets it run on source-induced tangents while preserving
    the old ``LegalWorld`` entry points.  The two default families use the
    Euclidean tangent metric; non-Euclidean callers should use the metric-aware
    3E-B path instead.
    """
    from .environment_family.geometry import build_task_geometry

    geometry = build_task_geometry(family)
    benchmark = _empty_benchmark(geometry.reference, geometry.source_environments)
    return LegalWorld(
        geometry.spec.family_name if name is None else name,
        geometry.reference, geometry.source_environments, benchmark,
        geometry.observation, geometry.response, geometry.spec,
    )


def family_evaluate_table(
    family, lambdas: tuple[float, ...] = (0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0),
    methods: tuple[str, ...] = ("l2", "irmv1", "vrex"),
) -> list[dict[str, object]]:
    """Run the post-hoc 3E-C table for a declared family geometry."""
    return evaluate_table(family_world(family), lambdas=lambdas, methods=methods)


def _empty_benchmark(base: ModuleEnvironment, environments: tuple[ModuleEnvironment, ...]) -> ThreeCBenchmark:
    optimum, source = source_optimum(environments)
    dimension = optimum.size
    hessian = 2.0 * source.second
    return ThreeCBenchmark(
        base, environments, source, optimum, (), np.zeros((dimension, 0)), hessian,
        np.zeros((dimension, 0)), np.zeros(0, dtype=int),
    )


def legal_world(name: str, base: ModuleEnvironment | None = None,
                source_environments: tuple[ModuleEnvironment, ...] | None = None,
                *, expose_u_at_source: bool = False,
                spec: WorldTangentSpec | None = None) -> LegalWorld:
    base = default_environment() if base is None else base
    spec = WorldTangentSpec() if spec is None else spec
    environments = tuple(source_environments or (base,))
    benchmark = _empty_benchmark(base, environments)
    observation = source_observation_block(environments, spec, expose_u_at_source=expose_u_at_source)
    response = frozen_response_jacobian(base, benchmark.source, benchmark.optimum, spec)
    return LegalWorld(name, base, environments, benchmark, observation, response, spec)


def primary_hidden_u_world() -> LegalWorld:
    geometry = coupled_primary_geometry()
    return LegalWorld(
        "hidden_u_hurts", geometry.base, geometry.source_environments,
        _empty_benchmark(geometry.base, geometry.source_environments),
        geometry.observation, geometry.response, geometry.spec,
    )


def primary_u_exposed_world() -> LegalWorld:
    geometry = coupled_primary_geometry()
    return LegalWorld(
        "u_exposed_complete_information", geometry.base, geometry.source_environments,
        _empty_benchmark(geometry.base, geometry.source_environments),
        u_exposed_observation(geometry), geometry.response, geometry.spec,
    )


def evaluate_world(world: LegalWorld, method: str, lam: float) -> dict[str, object]:
    """Evaluate a concrete exact-IFT policy without oracle metadata."""
    result = exact_ift_affine(world.benchmark, method, lam, world.observation)
    if not result.valid:
        return {"world": world.name, "method": method.upper(), "lambda": float(lam),
                "valid": False, "status": result.solver_status,
                "metric_min_eigenvalue": result.metric_min_eigenvalue}
    parts = decompose_response(world.response, world.observation)
    recoverable_residual = np.asarray(parts["A_recoverable"]) + result.tangent
    certificate = full_affine_certificate(
        result.z0, world.response, world.observation, recoverable_residual
    )
    adaptive_only = regret_decomposition(
        np.zeros_like(result.z0), world.response, result.tangent, world.observation,
    )
    return {
        "world": world.name, "method": method.upper(), "lambda": float(lam),
        "valid": True, "status": "PASS", "z0": result.z0, "pi": result.pi,
        "tangent": result.tangent, "E": recoverable_residual,
        "A_irreducible": certificate["A_irreducible"],
        "information_floor": certificate["information_floor"],
        "z0_norm": certificate["z0_norm"],
        "E_operator_norm": float(np.linalg.svd(recoverable_residual, compute_uv=False)[0]) if recoverable_residual.size else 0.0,
        "adaptive_regret": adaptive_only["total_regret"],
        "total_regret": certificate["total_regret"],
        "total_excess": certificate["total_excess"],
        "slack_ratio": certificate["slack_ratio"],
        "support_compatible": certificate["support_compatible"],
        "adaptive_condition_holds": certificate["condition_holds"],
        "recoverable_exact": certificate["recoverable_exact"],
        "complete_information": bool(
            certificate["alpha"] <= 1e-10 * max(1.0, np.linalg.norm(world.response, 2))
        ),
        "static_tax_lower_bound": certificate["static_tax_lower_bound"],
        "static_tax_holds": certificate["static_tax_holds"],
        "static_tightness_ratio": certificate["static_tightness_ratio"],
        "full_minimax": certificate["full_minimax_condition"],
        "metric_min_eigenvalue": result.metric_min_eigenvalue,
    }


def evaluate_table(world: LegalWorld, lambdas: tuple[float, ...] = (0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0),
                   methods: tuple[str, ...] = ("l2", "irmv1", "vrex")) -> list[dict[str, object]]:
    return [evaluate_world(world, method, lam) for method in methods for lam in lambdas]


def evaluate_hidden_u_row(method: str, lam: float) -> dict[str, object]:
    """Evaluate one policy row in the frozen hidden-U benchmark."""
    return evaluate_world(primary_hidden_u_world(), method, lam)


def evaluate_u_exposed_row(method: str, lam: float) -> dict[str, object]:
    """Evaluate one policy row in the separately-labelled U-exposed design."""
    return evaluate_world(primary_u_exposed_world(), method, lam)


def _candidate_sources(base: ModuleEnvironment) -> tuple[tuple[str, tuple[ModuleEnvironment, ...], bool], ...]:
    """Return the pre-registered legal source-design search family.

    The final three coordinates are source-environment offsets, not fitted
    matrices.  The first asymmetric U-exposed point is retained because it is
    the first lexicographic candidate found by the fixed grid audit.
    """
    relation = list(base.shortcut_rhos)
    relation[0] += 0.35
    reverse = list(base.shortcut_rhos)
    reverse[0] -= 0.35
    mean = list(base.shortcut_means)
    mean[0] += 0.55
    variance = list(base.shortcut_variances)
    variance[0] += 0.65
    candidates: list[tuple[str, tuple[ModuleEnvironment, ...], bool]] = [
        ("single_source", (base,), False),
        ("relation_plus", (base, base.updated(shortcut_rhos=tuple(relation))), False),
        ("relation_minus", (base, base.updated(shortcut_rhos=tuple(reverse))), False),
        ("mean_plus", (base, base.updated(shortcut_means=tuple(mean))), False),
        ("variance_plus", (base, base.updated(shortcut_variances=tuple(variance))), False),
    ]
    offsets = (
        (-0.80, -0.40, -0.30),
        (-0.80, -0.40, 0.30),
        (-0.50, 0.00, 0.30),
        (0.50, 0.40, 0.30),
        (0.80, 0.80, 0.80),
        (-0.50, 0.40, 0.80),
    )
    for index, (delta_rho, delta_mean, delta_variance) in enumerate(offsets):
        rhos = list(base.shortcut_rhos)
        means = list(base.shortcut_means)
        variances = list(base.shortcut_variances)
        rhos[0] += delta_rho
        means[0] += delta_mean
        variances[0] = max(0.03, variances[0] + delta_variance)
        environment = base.updated(
            shortcut_rhos=tuple(rhos), shortcut_means=tuple(means),
            shortcut_variances=tuple(variances),
        )
        candidates.append((f"grid_{index:02d}_u_exposed", (base, environment), True))
    return tuple(candidates)


def search_legal_helps_worlds(lambdas: tuple[float, ...] = (1e-3, 1e-2, 1e-1, 1.0),
                              base: ModuleEnvironment | None = None) -> dict[str, object]:
    """Search a fixed environment-derived family; never edits ``A/O/Pi``."""
    base = default_environment() if base is None else base
    trace: list[dict[str, object]] = []
    found: list[dict[str, object]] = []
    for name, environments, expose_u_at_source in _candidate_sources(base):
        try:
            world = legal_world(
                name, base, environments, expose_u_at_source=expose_u_at_source,
            )
            erm = evaluate_world(world, "l2", 0.0)
            if not erm.get("valid", False):
                continue
            for method in ("l2", "irmv1", "vrex"):
                for lam in lambdas:
                    candidate = evaluate_world(world, method, lam)
                    if not candidate.get("valid", False):
                        continue
                    row = {"world": name, "method": method.upper(), "lambda": lam,
                           "source_u_exposed": expose_u_at_source,
                           "erm_total_regret": erm["total_regret"],
                           "candidate_total_regret": candidate["total_regret"],
                           "improvement": erm["total_regret"] - candidate["total_regret"],
                           "erm_recoverable_norm": erm["E_operator_norm"],
                           "candidate": candidate}
                    trace.append({k: v for k, v in row.items() if k != "candidate"})
                    if row["improvement"] > 1e-8 and erm["E_operator_norm"] > 1e-8:
                        found.append(row)
        except (ValueError, np.linalg.LinAlgError) as exc:
            trace.append({"world": name, "status": f"invalid:{exc}"})
    return {
        "found": found,
        "selected": found[0] if found else None,
        "trace": trace,
        "candidate_count": len(trace),
        "search_family": "fixed legal ModuleEnvironment source designs",
        "target_oracle_used_for_selection": False,
    }


def verify_helps_hurts_world() -> dict[str, object]:
    """Verify both directions using only legal environment-derived worlds."""
    hidden_rows = evaluate_table(primary_hidden_u_world())
    erm = next(row for row in hidden_rows if row["method"] == "L2" and row["lambda"] == 0.0)
    hurts = [row for row in hidden_rows if row.get("valid", False) and row["lambda"] > 0.0
             and row["total_regret"] > erm["total_regret"] + 1e-10]
    helps = search_legal_helps_worlds()
    return {
        "hurts_found": bool(hurts),
        "hurts_count": len(hurts),
        "helps_found": helps["selected"] is not None,
        "helps_search": helps,
        "target_oracle_used_for_selection": False,
    }


__all__ = [
    "LegalWorld", "legal_world", "primary_hidden_u_world", "primary_u_exposed_world",
    "evaluate_world", "evaluate_table", "evaluate_hidden_u_row", "evaluate_u_exposed_row",
    "family_world", "family_evaluate_table", "search_legal_helps_worlds",
    "verify_helps_hurts_world",
]
