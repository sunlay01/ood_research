"""Run the exploratory regularizer sweep and write checkpoint-level metrics."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import random
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from .regularizers import (
    REGULARIZERS,
    LinearRepresentation,
    mean_source_risk,
    regularizer_value,
)
from .synthetic import Batch, DatasetBundle, make_dataset


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _mean_mse(model: LinearRepresentation, batches: tuple[Batch, ...]) -> float:
    values = [torch.mean((model.predict(batch) - batch.y) ** 2) for batch in batches]
    return float(torch.stack(values).mean())


def _ridge_weights(z: Tensor, y: Tensor, ridge: float) -> Tensor:
    ones = torch.ones((z.shape[0], 1), dtype=z.dtype, device=z.device)
    design = torch.cat((z, ones), dim=1)
    identity = torch.eye(design.shape[1], dtype=z.dtype, device=z.device)
    identity[-1, -1] = 0.0
    return torch.linalg.solve(design.T @ design + ridge * identity, design.T @ y)


def _ridge_predict(z: Tensor, weights: Tensor) -> Tensor:
    ones = torch.ones((z.shape[0], 1), dtype=z.dtype, device=z.device)
    return torch.cat((z, ones), dim=1) @ weights


def _target_task_errors(
    model: LinearRepresentation, bundle: DatasetBundle, ridge: float
) -> tuple[float, float]:
    support_z = torch.cat([model.encode(batch.x) for batch in bundle.target_task_support])
    support_y = torch.cat([batch.y for batch in bundle.target_task_support])
    weights = _ridge_weights(support_z, support_y, ridge)

    def evaluate(batches: tuple[Batch, ...]) -> float:
        losses = []
        for batch in batches:
            prediction = _ridge_predict(model.encode(batch.x), weights)
            losses.append(torch.mean((prediction - batch.y) ** 2))
        return float(torch.stack(losses).mean())

    return evaluate(bundle.target_task_source_eval), evaluate(bundle.joint_eval)


def _probe_r2(
    train_z: Tensor, train_y: Tensor, eval_z: Tensor, eval_y: Tensor, ridge: float
) -> float:
    weights = _ridge_weights(train_z, train_y, ridge)
    prediction = _ridge_predict(eval_z, weights)
    residual = (eval_y - prediction).square().sum(dim=0)
    centered = eval_y - eval_y.mean(dim=0, keepdim=True)
    total = centered.square().sum(dim=0).clamp_min(1e-12)
    return float((1.0 - residual / total).mean())


def _domain_probe_accuracy(
    model: LinearRepresentation,
    train: tuple[Batch, ...],
    evaluation: tuple[Batch, ...],
    ridge: float,
) -> float:
    train_z = torch.cat([model.encode(batch.x) for batch in train])
    train_env = torch.cat(
        [torch.full((batch.x.shape[0],), -1.0 if batch.env == 0 else 1.0) for batch in train]
    )
    weights = _ridge_weights(train_z, train_env, ridge)
    eval_z = torch.cat([model.encode(batch.x) for batch in evaluation])
    eval_env = torch.cat(
        [
            torch.full((batch.x.shape[0],), -1.0 if batch.env == 0 else 1.0)
            for batch in evaluation
        ]
    )
    prediction = _ridge_predict(eval_z, weights)
    return float((torch.where(prediction >= 0, 1.0, -1.0) == eval_env).float().mean())


def _covariance(z: Tensor) -> Tensor:
    centered = z - z.mean(dim=0, keepdim=True)
    return centered.T @ centered / max(z.shape[0] - 1, 1)


def _effective_rank(z: Tensor) -> tuple[float, float]:
    eigenvalues = torch.linalg.eigvalsh(_covariance(z)).clamp_min(0.0)
    trace = eigenvalues.sum()
    if float(trace) <= 1e-12:
        return 0.0, 0.0
    probabilities = (eigenvalues / trace).clamp_min(1e-12)
    rank = torch.exp(-(probabilities * probabilities.log()).sum())
    return float(rank), float(trace)


def _representation_metrics(
    model: LinearRepresentation, bundle: DatasetBundle, ridge: float
) -> dict[str, float]:
    train_z = torch.cat([model.encode(batch.x) for batch in bundle.train])
    train_x = torch.cat([batch.x for batch in bundle.train])
    eval_z = torch.cat([model.encode(batch.x) for batch in bundle.source_eval])
    eval_x = torch.cat([batch.x for batch in bundle.source_eval])
    effective_rank, latent_variance = _effective_rank(eval_z)

    by_env = []
    for env in (0, 1):
        by_env.append(
            torch.cat([model.encode(batch.x) for batch in bundle.source_eval if batch.env == env])
        )
    mean_gap = torch.linalg.vector_norm(by_env[0].mean(0) - by_env[1].mean(0))
    covariance_gap = torch.linalg.matrix_norm(_covariance(by_env[0]) - _covariance(by_env[1]))
    encoder = model.encoder.weight

    return {
        "core_probe_r2": _probe_r2(train_z, train_x[:, :2], eval_z, eval_x[:, :2], ridge),
        "spurious_probe_r2": _probe_r2(
            train_z, train_x[:, 2:3], eval_z, eval_x[:, 2:3], ridge
        ),
        "domain_probe_accuracy": _domain_probe_accuracy(
            model, bundle.train, bundle.source_eval, ridge
        ),
        "latent_mean_gap": float(mean_gap),
        "latent_covariance_gap": float(covariance_gap),
        "effective_rank": effective_rank,
        "latent_variance": latent_variance,
        "encoder_core_norm": float(torch.linalg.matrix_norm(encoder[:, :2])),
        "encoder_spurious_norm": float(torch.linalg.vector_norm(encoder[:, 2])),
        "encoder_domain_norm": float(torch.linalg.vector_norm(encoder[:, 3])),
        "encoder_nuisance_norm": float(torch.linalg.vector_norm(encoder[:, 4])),
        "source_head_norm": float(torch.linalg.matrix_norm(model.heads)),
    }


def _all_diagnostic_penalties(
    model: LinearRepresentation, bundle: DatasetBundle, mmd_max_samples: int
) -> dict[str, float]:
    return {
        f"diagnostic_{name}": float(
            regularizer_value(
                name, model, bundle.train, mmd_max_samples=mmd_max_samples
            )
        )
        for name in REGULARIZERS
    }


def _gradient_norm(values: tuple[Tensor | None, ...]) -> Tensor:
    parts = [value.flatten() for value in values if value is not None]
    if not parts:
        return torch.tensor(0.0)
    return torch.linalg.vector_norm(torch.cat(parts))


def _calibration_multiplier(
    model: LinearRepresentation,
    bundle: DatasetBundle,
    method: str,
    mmd_max_samples: int,
) -> float:
    parameters = tuple(model.parameters())
    risk = mean_source_risk(model, bundle.train)
    penalty = regularizer_value(
        method, model, bundle.train, mmd_max_samples=mmd_max_samples
    )
    risk_gradients = torch.autograd.grad(risk, parameters, retain_graph=True, allow_unused=True)
    penalty_gradients = torch.autograd.grad(penalty, parameters, allow_unused=True)
    risk_norm = _gradient_norm(risk_gradients)
    penalty_norm = _gradient_norm(penalty_gradients)
    ratio = risk_norm / penalty_norm.clamp_min(1e-12)
    return float(ratio.clamp(1e-4, 1e4))


def _checkpoint_metrics(
    *,
    model: LinearRepresentation,
    bundle: DatasetBundle,
    method: str,
    strength: float,
    seed: int,
    step: int,
    calibration: float,
    ridge: float,
    mmd_max_samples: int,
) -> dict[str, Any]:
    model.eval()
    with torch.no_grad():
        cross_task_mse, joint_mse = _target_task_errors(model, bundle, ridge)
        row: dict[str, Any] = {
            "method": method,
            "lambda": strength,
            "seed": seed,
            "step": step,
            "calibration_multiplier": calibration,
            "train_source_mse": float(mean_source_risk(model, bundle.train)),
            "source_mse": _mean_mse(model, bundle.source_eval),
            "cross_domain_mse": _mean_mse(model, bundle.target_domain_eval),
            "cross_task_mse": cross_task_mse,
            "joint_mse": joint_mse,
        }
        row.update(_representation_metrics(model, bundle, ridge))
        row.update(_all_diagnostic_penalties(model, bundle, mmd_max_samples))
        row["own_regularizer"] = 0.0 if method == "erm" else row[f"diagnostic_{method}"]
    model.train()
    return row


def _run_one(
    *, config: dict[str, Any], method: str, strength: float, seed: int
) -> list[dict[str, Any]]:
    _seed_everything(seed)
    bundle = make_dataset(
        seed=seed,
        n_train=int(config["n_train"]),
        n_eval=int(config["n_eval"]),
        n_probe=int(config["n_probe"]),
    )
    _seed_everything(seed + 10_000)
    model = LinearRepresentation(
        input_dim=5, latent_dim=int(config["latent_dim"]), n_tasks=2
    )
    calibration = (
        0.0
        if method == "erm"
        else _calibration_multiplier(
            model, bundle, method, int(config["mmd_max_samples"])
        )
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]))
    checkpoint_every = int(config["checkpoint_every"])
    total_steps = int(config["steps"])
    rows: list[dict[str, Any]] = []

    for step in range(total_steps + 1):
        if step % checkpoint_every == 0 or step == total_steps:
            rows.append(
                _checkpoint_metrics(
                    model=model,
                    bundle=bundle,
                    method=method,
                    strength=strength,
                    seed=seed,
                    step=step,
                    calibration=calibration,
                    ridge=float(config["ridge"]),
                    mmd_max_samples=int(config["mmd_max_samples"]),
                )
            )
        if step == total_steps:
            break
        optimizer.zero_grad(set_to_none=True)
        risk = mean_source_risk(model, bundle.train)
        if method == "erm":
            objective = risk
        else:
            penalty = regularizer_value(
                method,
                model,
                bundle.train,
                mmd_max_samples=int(config["mmd_max_samples"]),
            )
            objective = risk + strength * calibration * penalty
        if not bool(torch.isfinite(objective)):
            raise FloatingPointError(
                "Non-finite objective for "
                f"method={method}, lambda={strength}, seed={seed}, step={step}"
            )
        objective.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["gradient_clip"]))
        optimizer.step()
    return rows


def run(config_path: Path, output_dir: Path) -> None:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    unknown = set(config["methods"]) - {"erm", *REGULARIZERS}
    if unknown:
        raise ValueError(f"Unknown methods in config: {sorted(unknown)}")
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, output_dir / "config.json")
    started = time.perf_counter()
    all_rows: list[dict[str, Any]] = []

    for seed in config["seeds"]:
        for method in config["methods"]:
            strengths = [0.0] if method == "erm" else config["lambdas"]
            for strength in strengths:
                print(f"run method={method:10s} lambda={strength:g} seed={seed}", flush=True)
                all_rows.extend(
                    _run_one(
                        config=config,
                        method=str(method),
                        strength=float(strength),
                        seed=int(seed),
                    )
                )

    trajectory_path = output_dir / "trajectory.csv"
    with trajectory_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)

    metadata = {
        "experiment_id": config["experiment_id"],
        "config_path": str(config_path.resolve()),
        "output_dir": str(output_dir.resolve()),
        "rows": len(all_rows),
        "wall_time_seconds": time.perf_counter() - started,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "device": config["device"],
    }
    (output_dir / "environment.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.config, args.output)


if __name__ == "__main__":
    main()
