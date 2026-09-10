"""Input artifact loading and checkpoint verification."""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from ..task3_cmnist_counterfactual_audit.probe import ColorCounterfactualProbe, build_counterfactual_probe
from ..task3_cmnist_cpu_minimal.data import ColoredMNISTData, build_task3_data
from ..task3_cmnist_cpu_minimal.model import CPUColoredMNISTMLP, build_model_from_config, linear_layer_count, parameter_hash


@dataclass(frozen=True)
class CheckpointRecord:
    seed: int
    method: str
    path: Path
    checkpoint_sha256: str
    parameter_hash: str
    config_sha256: str
    git_commit: str


@dataclass(frozen=True)
class FeatureBundle:
    record: CheckpointRecord
    model: CPUColoredMNISTMLP
    data: ColoredMNISTData
    probe: ColorCounterfactualProbe
    source0_features: Tensor
    source1_features: Tensor
    target_features: Tensor
    red_features: Tensor
    green_features: Tensor
    parameter_hash_before: str


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_checkpoint_manifest(path: Path, *, seeds: list[int], methods: list[str], root: Path | None = None) -> list[CheckpointRecord]:
    path = path.resolve()
    root = root.resolve() if root is not None else path.parents[3]
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected: list[CheckpointRecord] = []
    for row in rows:
        seed = int(row["seed"])
        method = row["method"]
        if seed in seeds and method in methods:
            selected.append(
                CheckpointRecord(
                    seed=seed,
                    method=method,
                    path=root / row["path"],
                    checkpoint_sha256=row["checkpoint_sha256"],
                    parameter_hash=row["parameter_hash"],
                    config_sha256=row["config_sha256"],
                    git_commit=row["git_commit"],
                )
            )
    expected = {(seed, method) for seed in seeds for method in methods}
    observed = {(row.seed, row.method) for row in selected}
    missing = sorted(expected.difference(observed))
    if missing:
        raise ValueError(f"missing checkpoint manifest entries: {missing}")
    return sorted(selected, key=lambda item: (item.seed, item.method))


def load_verified_model(record: CheckpointRecord, config: dict[str, Any], *, config_sha256: str) -> CPUColoredMNISTMLP:
    if record.config_sha256 != config_sha256:
        raise ValueError(f"config SHA mismatch for {record.seed}/{record.method}")
    if sha256_file(record.path) != record.checkpoint_sha256:
        raise ValueError(f"checkpoint SHA mismatch for {record.path}")
    checkpoint = torch.load(record.path, map_location="cpu")
    if checkpoint.get("seed") != record.seed or checkpoint.get("method") != record.method:
        raise ValueError(f"checkpoint metadata mismatch for {record.path}")
    if checkpoint.get("parameter_hash") != record.parameter_hash:
        raise ValueError(f"checkpoint parameter hash metadata mismatch for {record.path}")
    model = build_model_from_config(config)
    model.load_state_dict(checkpoint["state_dict"])
    if linear_layer_count(model) != 3:
        raise ValueError("loaded checkpoint does not match 392->64->64->1 model identity")
    if parameter_hash(model) != record.parameter_hash:
        raise ValueError(f"loaded parameter hash mismatch for {record.path}")
    model.eval()
    return model


@torch.no_grad()
def encode_features(model: CPUColoredMNISTMLP, features: Tensor, *, batch_size: int = 4096) -> Tensor:
    model.eval()
    parts: list[Tensor] = []
    for start in range(0, features.shape[0], batch_size):
        stop = min(start + batch_size, features.shape[0])
        parts.append(model.encode(features[start:stop]).detach().cpu().double())
    return torch.cat(parts, dim=0)


def build_feature_bundle(
    record: CheckpointRecord,
    config: dict[str, Any],
    *,
    config_sha256: str,
    data_root: Path | str = "data",
    download: bool = False,
) -> FeatureBundle:
    model = load_verified_model(record, config, config_sha256=config_sha256)
    data = build_task3_data(config, record.seed, data_root=data_root, download=download)
    probe = build_counterfactual_probe(data.target_env)
    before = parameter_hash(model)
    bundle = FeatureBundle(
        record=record,
        model=model,
        data=data,
        probe=probe,
        source0_features=encode_features(model, data.source_envs[0].images),
        source1_features=encode_features(model, data.source_envs[1].images),
        target_features=encode_features(model, data.target_env.images),
        red_features=encode_features(model, probe.red),
        green_features=encode_features(model, probe.green),
        parameter_hash_before=before,
    )
    if parameter_hash(model) != before:
        raise ValueError("feature encoding mutated model parameters")
    return bundle


def load_verified_checkpoint_bundles(
    *,
    manifest_path: Path,
    config: dict[str, Any],
    config_sha256: str,
    seeds: list[int],
    methods: list[str],
    data_root: Path | str = "data",
    download: bool = False,
) -> list[FeatureBundle]:
    records = load_checkpoint_manifest(manifest_path, seeds=seeds, methods=methods)
    return [
        build_feature_bundle(
            record,
            config,
            config_sha256=config_sha256,
            data_root=data_root,
            download=download,
        )
        for record in records
    ]
