"""Semantic-free hierarchical grouping of opaque response signatures."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import cdist


def standardize(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("grouping requires a two-dimensional signature matrix")
    if not np.isfinite(values).all():
        raise ValueError("grouping signatures must be finite")
    scale = values.std(axis=0, ddof=0)
    return (values - values.mean(axis=0)) / np.where(scale > 1e-12, scale, 1.0)


def silhouette_score(points: np.ndarray, labels: np.ndarray) -> float:
    distances = cdist(points, points, metric="euclidean")
    values = []
    for index in range(len(points)):
        own = labels == labels[index]
        own[index] = False
        if not own.any():
            values.append(0.0)
            continue
        within = distances[index, own].mean()
        between = min(distances[index, labels == label].mean() for label in np.unique(labels) if label != labels[index])
        values.append((between - within) / max(within, between, 1e-12))
    return float(np.mean(values))


def adjusted_rand_index(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first)
    second = np.asarray(second)
    if first.shape != second.shape:
        raise ValueError("ARI inputs must have equal shape")
    def choose_two(value: int) -> float:
        return value * (value - 1) / 2.0
    total = choose_two(len(first))
    contingency = np.zeros((len(np.unique(first)), len(np.unique(second))), dtype=int)
    a = {label: index for index, label in enumerate(np.unique(first))}
    b = {label: index for index, label in enumerate(np.unique(second))}
    for left, right in zip(first, second):
        contingency[a[left], b[right]] += 1
    sum_rows = contingency.sum(axis=1)
    sum_cols = contingency.sum(axis=0)
    numerator = sum(choose_two(int(value)) for value in contingency.flat) - (sum(choose_two(int(value)) for value in sum_rows) * sum(choose_two(int(value)) for value in sum_cols) / max(total, 1.0))
    denominator = 0.5 * (sum(choose_two(int(value)) for value in sum_rows) + sum(choose_two(int(value)) for value in sum_cols)) - (sum(choose_two(int(value)) for value in sum_rows) * sum(choose_two(int(value)) for value in sum_cols) / max(total, 1.0))
    return 1.0 if abs(denominator) < 1e-12 else float(numerator / denominator)


def _labels(points: np.ndarray, k: int) -> np.ndarray:
    return fcluster(linkage(points, method="average", metric="euclidean"), t=k, criterion="maxclust")


def blind_group(signatures: list[dict[str, Any]], *, repeats: int = 200, seed: int = 20260910, candidate_k: tuple[int, ...] = (2, 3, 4), silhouette_threshold: float = 0.25, ari_threshold: float = 0.75, seed_replicates: list[list[dict[str, Any]]] | None = None) -> dict[str, Any]:
    if not signatures:
        raise ValueError("no signatures to group")
    ids = [str(row["opaque_direction_id"]) for row in signatures]
    if len(ids) != len(set(ids)):
        raise ValueError("opaque direction IDs must be unique")
    keys = sorted(key for key in signatures[0] if key != "opaque_direction_id")
    if not keys or any(set(row) != {"opaque_direction_id", *keys} for row in signatures):
        raise ValueError("signature rows must have identical numeric columns")
    matrix = standardize(np.asarray([[float(row[key]) for key in keys] for row in signatures], dtype=float))
    choices = []
    for k in sorted(candidate_k):
        if 1 < k < len(matrix):
            labels = _labels(matrix, k)
            choices.append((silhouette_score(matrix, labels), k, labels))
    if not choices:
        raise ValueError("not enough opaque signatures for candidate clusters")
    selected_score, selected_k, labels = sorted(choices, key=lambda item: (-item[0], item[1]))[0]
    rng = np.random.default_rng(seed)
    stability = []
    if seed_replicates:
        replicate_matrices = []
        for replicate in seed_replicates:
            if [str(row["opaque_direction_id"]) for row in replicate] != ids:
                raise ValueError("seed replicate IDs must match base signature IDs")
            replicate_matrices.append(standardize(np.asarray([[float(row[key]) for key in keys] for row in replicate], dtype=float)))
        for _ in range(int(repeats)):
            selected = rng.integers(0, len(replicate_matrices), size=len(replicate_matrices))
            sampled = np.mean([replicate_matrices[index] for index in selected], axis=0)
            stability.append(adjusted_rand_index(labels, _labels(sampled, selected_k)))
    else:
        for _ in range(int(repeats)):
            noise = rng.normal(0.0, 1e-6, size=matrix.shape)
            stability.append(adjusted_rand_index(labels, _labels(matrix + noise, selected_k)))
    mean_ari = float(np.mean(stability))
    silhouette = float(selected_score)
    frozen = silhouette >= silhouette_threshold and mean_ari >= ari_threshold
    return {
        "opaque_ids": ids, "standardized_matrix": matrix, "feature_keys": keys,
        "selected_k": int(selected_k), "labels": labels, "silhouette": silhouette,
        "bootstrap_mean_ari": mean_ari, "stable": bool(frozen),
        "status": "STAGE4-PASS" if frozen else "STAGE4-NO-STABLE-GROUPING",
        "grouping_method": "average_linkage_euclidean_column_zscore",
        "grouping_hyperparameters": {"candidate_k": list(candidate_k), "repeats": repeats, "seed": seed, "silhouette_threshold": silhouette_threshold, "ari_threshold": ari_threshold, "seed_resampling": bool(seed_replicates)},
    }
