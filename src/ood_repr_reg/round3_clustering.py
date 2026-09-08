"""Stable summaries for blind response-space clustering."""

from __future__ import annotations

import numpy as np


def cluster_enrichment(labels: np.ndarray, categories: tuple[str, ...], n_clusters: int) -> dict[str, dict[str, int]]:
    labels = np.asarray(labels, dtype=int)
    if labels.size != len(categories):
        raise ValueError("labels and categories must match")
    return {
        str(cluster): {
            category: int(np.sum((labels == cluster) & (np.asarray(categories) == category)))
            for category in sorted(set(categories))
        }
        for cluster in range(n_clusters)
    }


def compare_clusterings(first: np.ndarray, second: np.ndarray) -> float:
    """Permutation-free agreement proxy based on pairwise co-membership."""
    first = np.asarray(first)
    second = np.asarray(second)
    if first.shape != second.shape:
        raise ValueError("clusterings must have matching shapes")
    a = first[:, None] == first[None, :]
    b = second[:, None] == second[None, :]
    return float(np.mean(a == b))
