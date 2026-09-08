"""Post-hoc validation of blind shift clusters."""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from .round3_response_matrix import ResponseMatrices


def mechanism_enrichment(matrices: ResponseMatrices, labels: np.ndarray) -> dict[str, object]:
    labels = np.asarray(labels, dtype=int)
    kinds = np.asarray([probe.kind for probe in matrices.probes], dtype=object)
    if labels.shape != kinds.shape:
        raise ValueError("cluster labels and probes must have matching shapes")
    table: dict[str, dict[str, int]] = defaultdict(dict)
    for cluster in sorted(set(labels.tolist())):
        for kind in sorted(set(kinds.tolist())):
            table[str(cluster)][kind] = int(np.sum((labels == cluster) & (kinds == kind)))
    purities = {
        cluster: max(counts.values()) / max(sum(counts.values()), 1)
        for cluster, counts in table.items()
    }
    weighted = sum(max(counts.values()) for counts in table.values()) / max(len(labels), 1)
    return {
        "contingency": dict(table),
        "cluster_purity": purities,
        "weighted_cluster_purity": float(weighted),
        "n_clusters": len(set(labels.tolist())),
        "labels_used_after_discovery": True,
    }


def intervention_validation(matrices: ResponseMatrices, labels: np.ndarray) -> dict[str, object]:
    """Measure whether clusters predict held-out mechanism response profiles."""
    labels = np.asarray(labels, dtype=int)
    raw = matrices.raw
    kinds = np.asarray([probe.kind for probe in matrices.probes], dtype=object)
    profiles: dict[str, dict[str, float]] = {}
    for cluster in sorted(set(labels.tolist())):
        profiles[str(cluster)] = {}
        for kind in sorted(set(kinds.tolist())):
            mask = (labels == cluster) & (kinds == kind)
            profiles[str(cluster)][kind] = float(np.mean(np.abs(raw[:, mask]))) if np.any(mask) else 0.0
    # A nearest-cluster profile classifier is deliberately descriptive only.
    cluster_profiles = {cluster: np.mean(np.abs(raw[:, labels == int(cluster)]), axis=1) for cluster in set(labels.tolist())}
    assignments = []
    for index in range(raw.shape[1]):
        distances = {cluster: float(np.linalg.norm(np.abs(raw[:, index]) - profile)) for cluster, profile in cluster_profiles.items()}
        assignments.append(min(distances, key=distances.get))
    agreement = float(np.mean(np.asarray(assignments) == labels)) if assignments else 0.0
    pair_groups: dict[str, list[int]] = defaultdict(list)
    for index, probe in enumerate(matrices.probes):
        if probe.pair_id is not None:
            pair_groups[probe.pair_id].append(index)
    pair_consistency = [labels[indexes[0]] == labels[indexes[1]] for indexes in pair_groups.values() if len(indexes) == 2]
    return {
        "cluster_mechanism_profiles": profiles,
        "nearest_cluster_self_assignment": agreement,
        "pair_cluster_consistency": float(np.mean(pair_consistency)) if pair_consistency else 0.0,
        "uses_labels_post_discovery": True,
    }


def hidden_composition_pairs(matrices: ResponseMatrices, limit: int = 10) -> list[dict[str, object]]:
    values = matrices.raw.T
    kinds = [probe.kind for probe in matrices.probes]
    distances = []
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            if kinds[i] == kinds[j]:
                continue
            distances.append((float(np.linalg.norm(values[i] - values[j])), i, j))
    result = []
    for distance, i, j in sorted(distances)[:limit]:
        result.append({"distance": distance, "left": matrices.shift_ids[i], "right": matrices.shift_ids[j], "left_kind": kinds[i], "right_kind": kinds[j]})
    return result
