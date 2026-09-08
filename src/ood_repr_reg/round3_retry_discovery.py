"""Blind shift-column discovery for the round-three retry.

This module receives only response matrices.  It intentionally has no access
to mechanism, method, seed, or regularization labels.
"""

from __future__ import annotations

import numpy as np

from .round3_response_matrix import ResponseMatrices, column_signatures

Array = np.ndarray


def effective_rank(matrix: Array, tolerance: float = 1e-8) -> dict[str, object]:
    values = np.asarray(matrix, dtype=float)
    centered = values - values.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, full_matrices=False, compute_uv=False)
    explained = singular**2 / max(float(np.sum(singular**2)), 1e-30)
    rank = int(np.sum(singular > tolerance * singular[0])) if singular.size and singular[0] > 0 else 0
    return {"singular_values": singular, "explained": explained, "effective_rank": rank}


def _kmeans(points: Array, k: int, iterations: int = 100) -> Array:
    points = np.asarray(points, dtype=float)
    if not 1 <= k <= len(points):
        raise ValueError("invalid cluster count")
    indices = np.linspace(0, len(points) - 1, k, dtype=int)
    centers = points[indices].copy()
    labels = np.full(len(points), -1, dtype=int)
    for _ in range(iterations):
        distances = np.sum((points[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        new_labels = np.argmin(distances, axis=1)
        if np.array_equal(labels, new_labels):
            break
        labels = new_labels
        for cluster in range(k):
            if np.any(labels == cluster):
                centers[cluster] = points[labels == cluster].mean(axis=0)
    return labels


def _hierarchical(points: Array, k: int) -> Array:
    try:
        from scipy.cluster.hierarchy import fcluster, linkage
    except ImportError:  # pragma: no cover
        return np.full(len(points), -1, dtype=int)
    return fcluster(linkage(points, method="average", metric="euclidean"), t=k, criterion="maxclust") - 1


def _silhouette(points: Array, labels: Array) -> float:
    """Small dependency-free silhouette implementation for reproducibility."""
    labels = np.asarray(labels, dtype=int)
    if len(np.unique(labels)) < 2:
        return 0.0
    distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    values = []
    for index in range(len(points)):
        same = labels == labels[index]
        same[index] = False
        a = float(np.mean(distances[index, same])) if np.any(same) else 0.0
        between = [float(np.mean(distances[index, labels == other])) for other in np.unique(labels) if other != labels[index]]
        b = min(between) if between else 0.0
        values.append((b - a) / max(a, b, 1e-30))
    return float(np.mean(values))


def discover_shift_columns(matrices: ResponseMatrices, normalized: str = "raw", max_clusters: int = 6) -> dict[str, object]:
    points = column_signatures(matrices, normalized)
    centered = points - points.mean(axis=0, keepdims=True)
    rank = effective_rank(points)
    candidates = []
    for k in range(2, min(max_clusters, len(points) - 1) + 1):
        labels = _kmeans(centered, k)
        candidates.append((k, _silhouette(centered, labels), labels))
    best_k, best_score, best_labels = max(candidates, key=lambda item: item[1]) if candidates else (1, 0.0, np.zeros(len(points), dtype=int))
    return {
        "normalized": normalized,
        "n_shifts": int(points.shape[0]),
        "n_models": int(points.shape[1]),
        "effective_rank": rank["effective_rank"],
        "singular_values": rank["singular_values"],
        "explained": rank["explained"],
        "selected_k": int(best_k),
        "silhouette": float(best_score),
        "kmeans_labels": best_labels,
        "hierarchical_labels": _hierarchical(centered, best_k),
        "candidate_scores": [{"k": k, "silhouette": score} for k, score, _ in candidates],
        "input_uses_mechanism_labels": False,
        "input_uses_regularizer_labels": False,
        "input_uses_target_risk": False,
    }


def bootstrap_column_stability(matrices: ResponseMatrices, normalized: str = "raw", repeats: int = 32, seed: int = 19) -> float:
    """Resample model rows and compare shift co-membership, keeping columns fixed."""
    rng = np.random.default_rng(seed)
    base = column_signatures(matrices, normalized)
    reference = discover_shift_columns(matrices, normalized)
    ref = np.asarray(reference["kmeans_labels"])
    agreements = []
    for _ in range(repeats):
        row_indices = rng.integers(0, base.shape[1], base.shape[1])
        sampled = base[:, row_indices]
        centered = sampled - sampled.mean(axis=0, keepdims=True)
        labels = _kmeans(centered, int(reference["selected_k"]))
        ref_pairs = ref[:, None] == ref[None, :]
        new_pairs = labels[:, None] == labels[None, :]
        agreements.append(float(np.mean(ref_pairs == new_pairs)))
    return float(np.mean(agreements)) if agreements else 0.0


def row_diagnostics(matrices: ResponseMatrices, normalized: str = "raw") -> dict[str, object]:
    """Secondary model-side diagnostic; not used for the primary verdict."""
    rows = getattr(matrices, normalized)
    return effective_rank(rows.T)
