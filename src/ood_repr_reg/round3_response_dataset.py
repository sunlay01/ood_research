"""Build a controlled population risk-response dataset for blind discovery."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .round3_mechanisms import (
    Array,
    StructuralMechanism,
    add_direction,
    default_mechanism,
    mechanism_directions,
    risk,
)


@dataclass(frozen=True)
class ResponseRecord:
    model_id: str
    regularizer: str
    seed: int
    w_c: Array
    w_a: Array
    feature: Array
    shift_names: tuple[str, ...]
    hidden_labels: tuple[str, ...]


@dataclass(frozen=True)
class ResponseDataset:
    features: Array
    records: tuple[ResponseRecord, ...]
    shift_names: tuple[str, ...]


def _model_library(seed: int) -> tuple[tuple[str, str, Array, Array], ...]:
    rng = np.random.default_rng(seed)
    noise = float(rng.normal(0.0, 0.025))
    return (
        (f"erm-{seed}", "ERM", np.array([0.75 + noise]), np.array([0.30 + noise])),
        (f"l2-{seed}", "L2", np.array([0.82 + noise]), np.array([0.12 + noise / 2.0])),
        (f"irm-{seed}", "IRMv1", np.array([0.98 + noise / 2.0]), np.array([0.03 + noise / 3.0])),
        (f"coral-{seed}", "CORAL", np.array([0.90 + noise / 2.0]), np.array([0.16 + noise / 2.0])),
    )


def build_response_dataset(
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
    shift_scale: float = 0.15,
) -> ResponseDataset:
    base = default_mechanism()
    directions = mechanism_directions()
    probes: list[tuple[str, StructuralMechanism, str]] = []
    for name, direction in directions.items():
        probes.append((name, add_direction(base, direction, shift_scale), name))
    mixed = add_direction(base, directions["relation"], shift_scale)
    mixed = add_direction(mixed, directions["nuisance_variance"], shift_scale)
    probes.append(("mixed_relation_nuisance_variance", mixed, "mixed"))
    hidden = add_direction(base, directions["nuisance"], shift_scale)
    hidden = add_direction(hidden, directions["relation"], -shift_scale / 2.0)
    probes.append(("hidden_composition", hidden, "hidden_composition"))

    records: list[ResponseRecord] = []
    for seed in seeds:
        for model_id, regularizer, w_c, w_a in _model_library(seed):
            values = [risk(np.array([1.0]), w_c, w_a, target) - risk(np.array([1.0]), w_c, w_a, base) for _, target, _ in probes]
            curvature = []
            for name, target, _ in probes[: len(directions)]:
                opposite = add_direction(base, directions[name], -shift_scale)
                plus = risk(np.array([1.0]), w_c, w_a, target)
                minus = risk(np.array([1.0]), w_c, w_a, opposite)
                curvature.append(plus + minus - 2.0 * risk(np.array([1.0]), w_c, w_a, base))
            feature = np.asarray(values + curvature, dtype=float)
            records.append(ResponseRecord(model_id, regularizer, seed, w_c, w_a, feature, tuple(name for name, _, _ in probes), tuple(label for _, _, label in probes)))
    return ResponseDataset(np.stack([record.feature for record in records]), tuple(records), records[0].shift_names)


def blind_features(dataset: ResponseDataset, normalize: bool = True) -> Array:
    features = np.asarray(dataset.features, dtype=float).copy()
    if normalize:
        mean = features.mean(axis=0)
        scale = features.std(axis=0)
        scale[scale < 1e-12] = 1.0
        features = (features - mean) / scale
    return features
