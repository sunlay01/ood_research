"""Blind shift-column module discovery."""

from __future__ import annotations

import numpy as np

from .round3r_3b_geometry import GeometryTolerance, matrix_rank, module_subspaces, subspace_basis

Array = np.ndarray


def normalized_responses(responses: Array) -> Array:
    value = np.asarray(responses, dtype=float)
    norms = np.linalg.norm(value, axis=0)
    return value / np.maximum(norms[None, :], 1e-30)


def _features(responses: Array, sign_invariant: bool) -> Array:
    value = normalized_responses(responses).T
    return np.abs(value) if sign_invariant else value


def _choose_k(points: Array, max_clusters: int, seed: int) -> tuple[int, Array, list[dict[str, float]]]:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    import warnings
    from sklearn.exceptions import ConvergenceWarning

    n = len(points)
    if n < 3:
        return 1, np.zeros(n, dtype=int), []
    scores: list[dict[str, float]] = []
    candidates: list[tuple[int, float, Array]] = []
    max_distinct = np.unique(points, axis=0).shape[0]
    if max_distinct < 2:
        return 1, np.zeros(n, dtype=int), scores
    for k in range(2, min(max_clusters, n - 1, max_distinct) + 1):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model = KMeans(n_clusters=k, random_state=seed, n_init=20).fit(points)
        if len(np.unique(model.labels_)) < 2:
            continue
        score = float(silhouette_score(points, model.labels_))
        scores.append({"k": k, "silhouette": score})
        candidates.append((k, score, model.labels_))
    if not candidates:
        return 1, np.zeros(n, dtype=int), scores
    best = max(candidates, key=lambda item: (item[1], -item[0]))
    return best[0], best[2], scores


def discover_point_modules(responses: Array, max_clusters: int = 6, seed: int = 0, sign_invariant: bool = True) -> dict[str, object]:
    points = _features(responses, sign_invariant)
    selected_k, labels, scores = _choose_k(points, max_clusters, seed)
    return {"method": "point-clustering", "assignments": labels, "selected_k": selected_k,
            "candidate_scores": scores, "sign_invariant": sign_invariant,
            "uses_mechanism_labels": False, "uses_exposure": False, "uses_target_risk": False}


def discover_subspace_modules(responses: Array, max_clusters: int = 6, seed: int = 0) -> dict[str, object]:
    """A deterministic local-PCA/subspace proxy.

    It clusters sign-invariant normalized response coordinates, then reports
    each cluster's subspace rank. The rank/subspace reports are primary; point
    labels are only a reproducible baseline for this small benchmark.
    """
    result = discover_point_modules(responses, max_clusters, seed, sign_invariant=True)
    result["method"] = "local-pca-subspace"
    result["subspaces"] = module_subspaces(responses, result["assignments"])
    result["internal_ranks"] = {key: int(value.shape[1]) for key, value in result["subspaces"].items()}
    return result


def discover_gram_modules(gram: Array, max_clusters: int = 6, seed: int = 0) -> dict[str, object]:
    value = np.asarray(gram, dtype=float)
    diagonal = np.sqrt(np.maximum(np.diag(value), 1e-30))
    correlation = value / diagonal[:, None] / diagonal[None, :]
    result = discover_point_modules(correlation, max_clusters, seed, sign_invariant=True)
    result["method"] = "relevance-gram-spectral-proxy"
    return result


def discover_self_expression_modules(responses: Array, max_clusters: int = 6, seed: int = 0) -> dict[str, object]:
    """A label-free affinity baseline for subspace discovery.

    The absolute normalized response Gram is used as a self-expression-style
    affinity.  This is intentionally a diagnostic baseline, not a sparse
    optimization theorem: the benchmark is too small to justify claiming a
    particular sparse subspace-clustering estimator.
    """
    from sklearn.cluster import SpectralClustering
    import warnings

    points = _features(responses, sign_invariant=False)
    selected_k, _, scores = _choose_k(points, max_clusters, seed)
    if selected_k <= 1:
        labels = np.zeros(points.shape[0], dtype=int)
    else:
        affinity = np.abs(points @ points.T)
        np.fill_diagonal(affinity, 1.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            labels = SpectralClustering(
                n_clusters=selected_k, affinity="precomputed", random_state=seed,
                assign_labels="cluster_qr",
            ).fit_predict(affinity)
    return {
        "method": "self-expression-affinity",
        "assignments": labels,
        "selected_k": selected_k,
        "candidate_scores": scores,
        "uses_mechanism_labels": False,
        "uses_exposure": False,
        "uses_target_risk": False,
    }


def bootstrap_stability(responses: Array, discover=discover_subspace_modules, repeats: int = 64, seed: int = 19) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    reference = discover(responses, seed=seed)
    ref = np.asarray(reference["assignments"], dtype=int)
    agreements = []
    for _ in range(repeats):
        rows = rng.integers(0, responses.shape[0], responses.shape[0])
        sampled = responses[rows]
        current = discover(sampled, seed=int(rng.integers(0, 1_000_000)))
        labels = np.asarray(current["assignments"], dtype=int)
        if labels.size != ref.size:
            continue
        agreements.append(float(np.mean((ref[:, None] == ref[None, :]) == (labels[:, None] == labels[None, :]))))
    return {"mean_pairwise_stability": float(np.mean(agreements)) if agreements else 0.0,
            "std_pairwise_stability": float(np.std(agreements)) if agreements else 0.0,
            "repeats": repeats, "uses_mechanism_labels": False}
