"""Unsupervised response-space diagnostics with labels held out."""

from __future__ import annotations

import numpy as np

from .round3_response_dataset import ResponseDataset, blind_features


def effective_rank(features: np.ndarray, tolerance: float = 1e-10) -> dict[str, object]:
    matrix = np.asarray(features, dtype=float)
    singular = np.linalg.svd(matrix - matrix.mean(axis=0), compute_uv=False)
    if singular.size == 0 or singular[0] == 0:
        return {"singular_values": singular, "effective_rank": 0, "explained": np.zeros(0)}
    explained = singular * singular / np.sum(singular * singular)
    return {"singular_values": singular, "effective_rank": int(np.sum(singular > tolerance * singular[0])), "explained": explained}


def _kmeans(features: np.ndarray, k: int, iterations: int = 50) -> np.ndarray:
    if k < 1 or k > features.shape[0]:
        raise ValueError("invalid k")
    centers = features[np.linspace(0, features.shape[0] - 1, k, dtype=int)].copy()
    labels = np.zeros(features.shape[0], dtype=int)
    for _ in range(iterations):
        distances = ((features[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_labels = distances.argmin(axis=1)
        if np.array_equal(labels, new_labels):
            break
        labels = new_labels
        for cluster in range(k):
            members = features[labels == cluster]
            if len(members):
                centers[cluster] = members.mean(axis=0)
    return labels


def clustering_diagnostics(dataset: ResponseDataset, k: int = 3) -> dict[str, object]:
    features = blind_features(dataset)
    result = effective_rank(features)
    labels = _kmeans(features, k)
    result["kmeans_labels"] = labels
    result["cluster_sizes"] = [int(np.sum(labels == cluster)) for cluster in range(k)]
    try:
        from scipy.cluster.hierarchy import fcluster, linkage

        linkage_matrix = linkage(features, method="average", metric="euclidean")
        result["hierarchical_labels"] = fcluster(linkage_matrix, t=k, criterion="maxclust") - 1
        result["hierarchical_available"] = True
    except ImportError:
        result["hierarchical_labels"] = np.full(features.shape[0], -1, dtype=int)
        result["hierarchical_available"] = False
    return result


def bootstrap_stability(dataset: ResponseDataset, k: int = 3, repeats: int = 20, seed: int = 0) -> float:
    features = blind_features(dataset)
    rng = np.random.default_rng(seed)
    reference = _kmeans(features, k)
    agreements = []
    for _ in range(repeats):
        indices = rng.integers(0, features.shape[0], features.shape[0])
        labels = _kmeans(features[indices], k)
        agreements.append(float(np.mean(labels == reference[indices])))
    return float(np.mean(agreements))
