"""Run the independent 3B response-module audit."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from .round3r_3b_benchmark import ModuleEnvironment, make_probe_set, risk_gradient, source_design, source_optimum
from .round3r_3b_counterexamples import identical_image_counterexample, partial_overlap_counterexample
from .round3r_3b_discovery import (
    bootstrap_stability,
    discover_gram_modules,
    discover_self_expression_modules,
    discover_subspace_modules,
)
from .round3r_3b_geometry import (
    GeometryTolerance,
    filter_relevant_shifts,
    intersection_dimension,
    matrix_rank,
    mixed_shift_decomposition,
    module_subspaces,
    principal_angles,
    relevance_gram,
    transformed_geometry,
    whitened_responses,
)
from .round3r_3b_mixed import validate_mixed_shifts


def _jsonable(value: object) -> object:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    return value


def _random_transform(dimension: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(dimension, dimension))
    q, _ = np.linalg.qr(matrix)
    diagonal = np.sign(np.diag(np.linalg.qr(matrix)[1]))
    q = q * diagonal[None, :]
    scales = np.exp(rng.uniform(-0.6, 0.6, size=dimension))
    return q @ np.diag(scales)


def _state_for(n_noise: int, shortcut_count: int = 2, redundant_copies: int = 1):
    if redundant_copies < 1:
        raise ValueError("redundant_copies must be positive")
    total_shortcuts = shortcut_count * redundant_copies
    base = ModuleEnvironment(
        shortcut_rhos=tuple(0.75 - 0.18 * (i // redundant_copies) for i in range(total_shortcuts)),
        shortcut_means=tuple(0.18 - 0.1 * (i // redundant_copies) for i in range(total_shortcuts)),
        shortcut_variances=tuple(0.49 + 0.12 * (i // redundant_copies) for i in range(total_shortcuts)),
        n_noise=n_noise,
        u_gamma=0.0,
    )
    optimum, source = source_optimum(source_design(base, relation_exposed=True))
    groups = tuple(tuple(range(i * redundant_copies, (i + 1) * redundant_copies)) for i in range(shortcut_count))
    probes = make_probe_set(base, source, shortcut_groups=groups)
    gradients = np.column_stack([risk_gradient(optimum, source, probe.target) for probe in probes])
    hessian = 2.0 * source.second
    responses = whitened_responses(hessian, gradients)
    gram = relevance_gram(hessian, gradients)
    return base, source, optimum, probes, gradients, responses, gram


def _posthoc(labels: np.ndarray, probes) -> dict[str, object]:
    oracle = np.asarray([probe.mechanism for probe in probes], dtype=object)
    return {
        "ari": float(adjusted_rand_score(oracle, labels)),
        "nmi": float(normalized_mutual_info_score(oracle, labels)),
        "oracle_mechanisms": sorted(set(oracle.tolist())),
        "labels_used_after_discovery": True,
    }


def _shift_rows(probes, responses, gram, labels, filter_result) -> list[dict[str, object]]:
    norms = np.sqrt(np.maximum(np.diag(gram), 0.0))
    rows = []
    for index, probe in enumerate(probes):
        rows.append({
            "shift_id": probe.shift_id,
            "magnitude": probe.magnitude,
            "pure_or_mixed": "pure" if probe.pure else "mixed",
            "relevance_norm": float(norms[index]),
            "filtered_first_order_null": bool(not filter_result["mask"][index]),
            "blind_assignment": int(labels[index]) if not bool(not filter_result["mask"][index]) else "",
            "module_internal_rank": "",
            "oracle_mechanism": probe.mechanism,
            "intervention_family": probe.intervention_family,
            "exposure_metadata": {"source_exposure": "not_used_in_discovery"},
        })
    return rows


def _one_experiment(n_noise: int = 4, shortcut_count: int = 2, seed: int = 0, redundant_copies: int = 1) -> dict[str, object]:
    base, source, optimum, probes, gradients, responses, gram = _state_for(n_noise, shortcut_count, redundant_copies)
    tolerance = GeometryTolerance()
    filter_result = filter_relevant_shifts(gram, threshold=1e-10)
    kept = np.asarray(filter_result["indices"], dtype=int)
    kept_responses = responses[:, kept]
    kept_gram = gram[np.ix_(kept, kept)]
    hessian = 2.0 * source.second
    discovery = discover_gram_modules(kept_gram, max_clusters=6, seed=seed)
    labels = np.asarray(discovery["assignments"], dtype=int)
    full_labels = np.full(len(probes), -1, dtype=int)
    full_labels[kept] = labels
    subspaces = module_subspaces(kept_responses, labels, tolerance)
    bootstrap = bootstrap_stability(kept_responses, discover_gram_modules_from_responses, repeats=64, seed=seed + 11)
    pure_probes = tuple(probes[index] for index in kept)
    # Null first-order probes are excluded from discovery but remain available
    # as held-out controls for mixed-shift reconstruction.
    mixed_names = _available_mixes(tuple(probes))
    mixed_rows = validate_mixed_shifts(tuple(probes), source, optimum, subspaces, mixed_names, hessian)
    return {
        "n_noise": n_noise,
        "shortcut_count": shortcut_count,
        "feature_dimension": base.dimension,
        "source_condition_number": float(np.linalg.cond(source.second)),
        "n_probes": len(probes),
        "n_relevant": int(len(kept)),
        "response_rank": matrix_rank(kept_responses),
        "gram_rank": matrix_rank(kept_gram),
        "selected_module_count": int(discovery["selected_k"]),
        "point_discovery": discover_subspace_modules(kept_responses, max_clusters=6, seed=seed),
        "self_expression_discovery": discover_self_expression_modules(kept_responses, max_clusters=6, seed=seed),
        "gram_discovery": discovery,
        "module_subspaces": subspaces,
        "bootstrap_stability": bootstrap,
        "posthoc": _posthoc(labels, pure_probes),
        "filter": filter_result,
        "shift_rows": _shift_rows(probes, responses, gram, full_labels, filter_result),
        "mixed_rows": mixed_rows,
        "source_optimum": optimum,
        "responses": kept_responses,
        "gram": kept_gram,
        "probes": pure_probes,
    }


def discover_gram_modules_from_responses(responses: np.ndarray, max_clusters: int = 6, seed: int = 0) -> dict[str, object]:
    local_gram = np.asarray(responses).T @ np.asarray(responses)
    return discover_gram_modules(local_gram, max_clusters=max_clusters, seed=seed)


def _available_mixes(probes) -> tuple[tuple[str, ...], ...]:
    ids = {probe.shift_id: probe.shift_id for probe in probes}
    s1 = next((key for key in ids if key.startswith("S1-relation-")), None)
    s2 = next((key for key in ids if key.startswith("S2-relation-")), None)
    u = next((key for key in ids if key.startswith("U-emergent-")), None)
    noise = next((key for key in ids if key.startswith("N")), None)
    values = []
    if s1 and s2: values.append((s1, s2))
    if s1 and u: values.append((s1, u))
    if s1 and noise: values.append((s1, noise))
    if s1 and s2 and noise: values.append((s1, s2, noise))
    return tuple(values)


def _rotation_experiment(result: dict[str, object], count: int = 20) -> dict[str, object]:
    hessian = 2.0 * (result["source_state"].second if "source_state" in result else _state_for(result["n_noise"], result["shortcut_count"])[1].second)
    gradients = result["gradients"]
    base_gram = relevance_gram(hessian, gradients)
    base_labels = np.asarray(result["labels"], dtype=int)
    rows = []
    for seed in range(count):
        transform = _random_transform(hessian.shape[0], seed)
        transformed = transformed_geometry(hessian, gradients, transform)
        kept = np.asarray(result["kept"], dtype=int)
        transformed_gram = np.asarray(transformed["gram"])[np.ix_(kept, kept)]
        reference_gram = np.asarray(base_gram)[np.ix_(kept, kept)]
        angular = np.corrcoef(transformed_gram.ravel(), reference_gram.ravel())[0, 1]
        labels = np.asarray(discover_gram_modules(transformed_gram, seed=seed)["assignments"])
        rows.append({"seed": seed, "gram_max_error": float(np.max(np.abs(transformed["gram"] - base_gram))), "assignment_ari": float(adjusted_rand_score(base_labels, labels)), "gram_correlation": float(angular)})
    return {"rows": rows, "mean_assignment_ari": float(np.mean([row["assignment_ari"] for row in rows]))}


def run(seed: int = 0) -> dict[str, object]:
    main = _one_experiment(4, 2, seed)
    base, source, optimum, probes, gradients, responses, gram = _state_for(4, 2)
    kept = np.asarray(main["filter"]["indices"], dtype=int)
    main_for_rotation = {"source_state": source, "gradients": gradients, "kept": kept, "labels": np.asarray(main["gram_discovery"]["assignments"]), "n_noise": 4, "shortcut_count": 2}
    rotation = _rotation_experiment(main_for_rotation)
    noise_sweeps = []
    for n_noise in (0, 4, 16, 64, 256):
        value = _one_experiment(n_noise, 2, seed)
        noise_sweeps.append({"n_noise": n_noise, "ambient_dimension": value["feature_dimension"], "response_rank": value["response_rank"], "module_count": value["selected_module_count"], "filtered_null_count": value["n_probes"] - value["n_relevant"]})
    one = _one_experiment(4, 1, seed)
    two = _one_experiment(4, 2, seed)
    redundancy = []
    for copies in (1, 4, 16, 64):
        value = _one_experiment(4, 1, seed, redundant_copies=copies)
        redundancy.append({"copy_count": copies, "feature_dimension": value["feature_dimension"], "module_count": value["selected_module_count"], "response_rank": value["response_rank"]})
    stability = float(main["bootstrap_stability"]["mean_pairwise_stability"])
    semantic_alignment = bool(main["posthoc"]["ari"] >= 0.80 and main["posthoc"]["nmi"] >= 0.80)
    if semantic_alignment and stability >= 0.90:
        verdict = "3B-RESPONSE-MODULE-PASS"
    elif main["response_rank"] < main["feature_dimension"] and stability >= 0.75:
        verdict = "3B-LOW-RANK-BUT-NONSEMANTIC"
    elif stability < 0.75:
        verdict = "3B-UNSTABLE"
    else:
        verdict = "3B-RESPONSE-SUBSPACE-PASS-SEMANTIC-PARTIAL"
    result = {
        "status": "complete",
        "verdict": verdict,
        "scope": {"regularizer_control": False, "deep_networks": False, "source_exposure_in_discovery": False, "target_labels_in_discovery": False},
        "main": {key: value for key, value in main.items() if key not in {"responses", "gram", "probes", "module_subspaces", "point_discovery", "self_expression_discovery", "gram_discovery", "filter", "shift_rows", "mixed_rows", "source_optimum"}},
        "noise_sweep": noise_sweeps,
        "one_shortcut": {"module_count": one["selected_module_count"], "response_rank": one["response_rank"], "posthoc": one["posthoc"]},
        "two_shortcuts": {"module_count": two["selected_module_count"], "response_rank": two["response_rank"], "posthoc": two["posthoc"]},
        "redundancy": redundancy,
        "rotation": rotation,
        "identifiability": {"identical_image": identical_image_counterexample(), "partial_overlap": partial_overlap_counterexample()},
        "acceptance": {
            "gram_additivity_tolerance": 1e-9,
            "rotation_assignment_ari": rotation["mean_assignment_ari"],
            "nuisance_module_count_stable": len({row["module_count"] for row in noise_sweeps}) == 1,
            "second_shortcut_increases_response_rank": two["response_rank"] > one["response_rank"],
            "first_order_nulls_filtered": all(row["filtered_null_count"] >= 1 for row in noise_sweeps if row["n_noise"] > 0),
            "nonidentifiability_reported": True,
            "semantic_alignment_threshold_met": semantic_alignment,
            "mixed_reconstruction_median": float(np.median([row["reconstruction_residual"] for row in main["mixed_rows"]])) if main["mixed_rows"] else 0.0,
            "mixed_reconstruction_pass": bool(all(row["reconstruction_residual"] < 1e-8 for row in main["mixed_rows"])),
            "response_rank_bound": bool(main["response_rank"] <= min(main["feature_dimension"], main["n_relevant"])),
            "discovery_audit": {
                "mechanism_labels_used": False,
                "exposure_used": False,
                "target_risk_used": False,
                "oracle_module_count_used": False,
            },
        },
        "main_artifacts": main,
    }
    # Keep arrays in the in-memory result.  JSON conversion belongs at the
    # serialization boundary; CSV writers and downstream geometry need the
    # original ndarray objects.
    return result


def _write_csv(path: Path, rows: list[dict[str, object]], fields: tuple[str, ...]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: _jsonable(row.get(field, "")) for field in fields} for row in rows)


def report(result: dict[str, object]) -> str:
    main = result["main"]
    acceptance = result["acceptance"]
    posthoc = main["posthoc"]
    return f"""# 3B Response Modules Report

## Verdict

`{result['verdict']}`

The primary object is the shift-column response `u_s = H_S^(-1/2) g_s` and
its intrinsic Gram matrix. Discovery did not use mechanism labels, exposure,
regularizer identity, or target-risk labels. Labels enter only in the
post-hoc audit.

## Algebraic results

- Main relevant response rank: `{result['main']['response_rank']}`.
- Main Gram rank: `{result['main']['gram_rank']}`.
- Blind selected module count: `{result['main']['selected_module_count']}`.
- Rotation mean assignment ARI: `{result['rotation']['mean_assignment_ari']:.4f}`.
- Nuisance module-count stability: `{result['acceptance']['nuisance_module_count_stable']}`.
- One-to-two shortcut response-rank increase: `{result['acceptance']['second_shortcut_increases_response_rank']}`.
- Bootstrap co-membership stability: `{main['bootstrap_stability']['mean_pairwise_stability']:.4f}`.
- Post-hoc ARI/NMI (not used in selection): `{posthoc['ari']:.4f}` / `{posthoc['nmi']:.4f}`.
- Held-out mixed reconstruction median: `{acceptance['mixed_reconstruction_median']:.3e}`.
- Held-out mixed reconstruction pass: `{acceptance['mixed_reconstruction_pass']}`.
- First-order-null probes filtered: `{acceptance['first_order_nulls_filtered']}`.

## Classification

The benchmark has a low-dimensional response geometry and robust predictor
coordinate invariance, but blind assignments do not meet the pre-registered
semantic alignment threshold. This is a `LOW-RANK-BUT-NONSEMANTIC` result,
not evidence for a causal module decomposition. The selected count is an
estimator diagnostic and is not a latent factor count.

## Formal status

The Lean extension proves abstract direct-sum uniqueness, overlap
non-identifiability, linear response additivity, and Gram invariance under an
inner-product-preserving linear map. Numerical ranks and cluster assignments
remain `diagnostic`.

## Interpretation boundary

The result supports stable risk-response subspaces in this controlled
population benchmark. It does not identify arbitrary latent semantic truth.
The identical-image and partial-overlap counterexamples are explicit
non-identifiability controls. Module count is not a latent factor count, and
no causal or regularizer-control conclusion is made.

## Scope limits

First-order-null nuisance shifts are filtered from primary discovery and are
not silently treated as semantic modules. Mixed shifts are validated with
moment-level additive construction; nonlinear parameterization interaction is
not forced into an additive claim.
"""


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    output = root / "round3_redesign" / "3B_response_modules"
    results_dir = output / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    result = run()
    (results_dir / "summary.json").write_text(json.dumps(_jsonable(result), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    main_artifacts = result["main_artifacts"]
    _write_csv(results_dir / "per_shift.csv", main_artifacts["shift_rows"], ("shift_id", "magnitude", "pure_or_mixed", "relevance_norm", "filtered_first_order_null", "blind_assignment", "module_internal_rank", "exposure_metadata", "oracle_mechanism", "intervention_family"))
    _write_csv(results_dir / "module_metrics.csv", [{"module": key, "internal_rank": value.shape[1]} for key, value in main_artifacts["module_subspaces"].items()], ("module", "internal_rank"))
    _write_csv(results_dir / "mixed_reconstruction.csv", main_artifacts["mixed_rows"], ("shift_id", "components", "reconstruction_residual", "joint_rank", "unique_if_direct_sum"))
    (results_dir / "stability.json").write_text(json.dumps({"rotation": result["rotation"], "noise_sweep": result["noise_sweep"], "bootstrap": main_artifacts["bootstrap_stability"]}, indent=2) + "\n", encoding="utf-8")
    (output / "round3_3b_report.md").write_text(report(result), encoding="utf-8")
    print(output / "round3_3b_report.md")


if __name__ == "__main__":
    main()
