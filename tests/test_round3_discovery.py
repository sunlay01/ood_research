import numpy as np

from ood_repr_reg.round3_clustering import compare_clusterings
from ood_repr_reg.round3_discovery import bootstrap_stability, clustering_diagnostics, effective_rank
from ood_repr_reg.round3_response_dataset import blind_features, build_response_dataset


def test_response_dataset_blind_features_exclude_metadata():
    dataset = build_response_dataset(seeds=(0, 1))
    features = blind_features(dataset)
    assert features.shape[0] == len(dataset.records)
    assert features.shape[1] == dataset.features.shape[1]
    assert "IRMv1" not in str(features)


def test_discovery_reports_rank_and_multiple_clusterers():
    dataset = build_response_dataset(seeds=(0, 1, 2))
    result = clustering_diagnostics(dataset, k=3)
    assert "effective_rank" in result
    assert len(result["kmeans_labels"]) == len(dataset.records)
    assert len(result["hierarchical_labels"]) == len(dataset.records)


def test_bootstrap_stability_is_bounded():
    dataset = build_response_dataset(seeds=(0, 1, 2))
    value = bootstrap_stability(dataset, k=3, repeats=3)
    assert 0.0 <= value <= 1.0


def test_pairwise_cluster_agreement_is_exact_for_same_labels():
    labels = np.array([0, 1, 0, 2])
    assert compare_clusterings(labels, labels) == 1.0
