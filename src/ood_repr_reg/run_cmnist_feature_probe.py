"""CLI for CMNIST-VIS-001."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from .cmnist_feature_probe import (
    CounterfactualProbe,
    ColoredEnvironment,
    counterfactual_diagnostics,
    deterministic_subset,
    load_mnist_tensors,
    make_counterfactual_probe,
    make_environment,
    mean_accuracy,
    source_diagnostic_penalties,
    train_model,
)


def _device(name: str) -> torch.device:
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    return torch.device(name)


def _build_data(config: dict, seed: int) -> tuple[
    tuple[ColoredEnvironment, ...],
    tuple[ColoredEnvironment, ...],
    ColoredEnvironment,
    CounterfactualProbe,
]:
    root = Path(config["data_root"])
    train_gray, train_digit = load_mnist_tensors(root, train=True, download=config["download"])
    test_gray, test_digit = load_mnist_tensors(root, train=False, download=config["download"])

    n_train = int(config["train_per_environment"])
    train_environments = []
    label_noise = float(config.get("label_noise", 0.0))
    for env, correlation in enumerate(config["source_correlations"]):
        gray, digit = deterministic_subset(
            train_gray, train_digit, n=n_train, seed=seed + 101, offset=env * n_train
        )
        train_environments.append(
            make_environment(
                gray,
                digit,
                correlation=correlation,
                env=env,
                seed=seed + env * 17,
                label_noise=label_noise,
            )
        )

    n_source = int(config["source_eval_per_environment"])
    source_environments = []
    offset = 0
    for env, correlation in enumerate(config["source_correlations"]):
        gray, digit = deterministic_subset(
            test_gray, test_digit, n=n_source, seed=seed + 211, offset=offset
        )
        offset += n_source
        source_environments.append(
            make_environment(
                gray,
                digit,
                correlation=correlation,
                env=env,
                seed=seed + 1000 + env * 17,
                label_noise=label_noise,
            )
        )

    target_gray, target_digit = deterministic_subset(
        test_gray,
        test_digit,
        n=int(config["target_eval"]),
        seed=seed + 211,
        offset=offset,
    )
    offset += int(config["target_eval"])
    target = make_environment(
        target_gray,
        target_digit,
        correlation=float(config["target_correlation"]),
        env=len(source_environments),
        seed=seed + 2000,
        label_noise=label_noise,
    )
    probe_gray, probe_digit = deterministic_subset(
        test_gray,
        test_digit,
        n=int(config["probe_size"]),
        seed=seed + 211,
        offset=offset,
    )
    return (
        tuple(train_environments),
        tuple(source_environments),
        target,
        make_counterfactual_probe(probe_gray, probe_digit),
    )


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    columns = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _label(method: str, strength: float) -> str:
    return method.upper() if method == "erm" else f"{method.upper()} {strength:g}"


def _plot_paths(path: Path, records: list[dict[str, object]]) -> None:
    selected = [record for record in records if record["seed"] == 0]
    columns = 3
    rows = int(np.ceil(len(selected) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(12, 3.7 * rows), squeeze=False)
    for axis, record in zip(axes.flat, selected):
        values = record["paths"]
        red = values["red"][:80]
        green = values["green"][:80]
        y = values["y"][:80]
        for index in range(red.shape[0]):
            color = "#2563eb" if y[index] == 0 else "#d97706"
            axis.plot(
                [red[index, 0], green[index, 0]],
                [red[index, 1], green[index, 1]],
                color=color,
                alpha=0.18,
                linewidth=0.7,
            )
        axis.scatter(
            red[:, 0], red[:, 1], marker="o", s=10, c="#dc2626", alpha=0.55, label="red"
        )
        axis.scatter(
            green[:, 0],
            green[:, 1],
            marker="^",
            s=10,
            c="#16a34a",
            alpha=0.55,
            label="green",
        )
        axis.axhline(0.0, color="#d1d5db", linewidth=0.7)
        axis.axvline(0.0, color="#d1d5db", linewidth=0.7)
        axis.set_title(_label(record["method"], record["strength"]), fontsize=10)
        axis.set_xlabel("task probe coordinate")
        axis.set_ylabel("color probe coordinate")
    for axis in axes.flat[len(selected) :]:
        axis.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper right")
    figure.suptitle("Counterfactual feature paths: same digit, red to green", fontsize=13)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_summary(path: Path, rows: list[dict[str, object]]) -> None:
    seed_zero = [row for row in rows if row["seed"] == 0]
    labels = [_label(str(row["method"]), float(row["strength"])) for row in seed_zero]
    metrics = (
        ("target_accuracy", "Sign-flip accuracy", False),
        ("balanced_accuracy", "Balanced accuracy", False),
        ("prediction_color_response", "Head-used color response", True),
        ("latent_color_response", "Whitened latent color response", True),
        ("task_signal", "Balanced task signal", True),
        ("task_color_overlap", "Task/color overlap", False),
    )
    figure, axes = plt.subplots(2, 3, figsize=(14, 7.5))
    palette = ["#374151"] + [
        "#2563eb",
        "#60a5fa",
        "#d97706",
        "#f59e0b",
        "#059669",
        "#34d399",
    ]
    for axis, (key, title, log_scale) in zip(axes.flat, metrics):
        values = np.asarray([float(row[key]) for row in seed_zero])
        axis.bar(np.arange(len(values)), values, color=palette[: len(values)])
        axis.set_title(title)
        axis.set_xticks(np.arange(len(values)), labels, rotation=42, ha="right", fontsize=8)
        if log_scale and np.all(values > 0):
            axis.set_yscale("log")
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle("CMNIST feature-response audit (seed 0)", fontsize=14)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_examples(
    path: Path,
    probe: CounterfactualProbe,
    records: list[dict[str, object]],
) -> None:
    selected = [record for record in records if record["seed"] == 0]
    n = min(8, probe.y.shape[0])
    figure, axes = plt.subplots(
        n,
        2 + len(selected),
        figsize=(2.0 * (2 + len(selected)), 1.65 * n),
    )
    for index in range(n):
        axes[index, 0].imshow(probe.red[index].permute(1, 2, 0).numpy())
        axes[index, 1].imshow(probe.green[index].permute(1, 2, 0).numpy())
        axes[index, 0].set_ylabel(f"digit {int(probe.digit[index])}\ny={int(probe.y[index])}")
        for column in (0, 1):
            axes[index, column].set_xticks([])
            axes[index, column].set_yticks([])
        for column, record in enumerate(selected, start=2):
            values = record["paths"]
            red_probability = float(values["red_probability"][index])
            green_probability = float(values["green_probability"][index])
            axes[index, column].bar(
                [0, 1],
                [red_probability, green_probability],
                color=["#dc2626", "#16a34a"],
            )
            axes[index, column].set_ylim(0, 1)
            axes[index, column].set_xticks([0, 1], ["R", "G"])
            if column > 2:
                axes[index, column].set_yticks([])
    axes[0, 0].set_title("red input")
    axes[0, 1].set_title("green input")
    for column, record in enumerate(selected, start=2):
        axes[0, column].set_title(_label(record["method"], record["strength"]), fontsize=9)
    figure.suptitle("Counterfactual predictions: P(y=1) for identical shape", fontsize=13)
    figure.tight_layout(rect=(0, 0, 1, 0.98))
    figure.savefig(path, dpi=180)
    plt.close(figure)


def run(config_path: Path, output: Path) -> list[dict[str, object]]:
    config = json.loads(config_path.read_text())
    output.mkdir(parents=True, exist_ok=True)
    (output / "paths").mkdir(exist_ok=True)
    (output / "checkpoints").mkdir(exist_ok=True)
    (output / "config.json").write_text(json.dumps(config, indent=2))
    device = _device(config["device"])
    started = time.time()
    metric_rows: list[dict[str, object]] = []
    path_records: list[dict[str, object]] = []
    example_probe = None

    for seed in config["seeds"]:
        train, source_eval, target, probe = _build_data(config, int(seed))
        if example_probe is None:
            example_probe = probe
        for method, strengths in config["methods"].items():
            for strength in strengths:
                model, train_metrics = train_model(
                    train,
                    method=method,
                    strength=float(strength),
                    latent_dim=int(config["latent_dim"]),
                    epochs=int(config["epochs"]),
                    batch_size=int(config["batch_size"]),
                    learning_rate=float(config["learning_rate"]),
                    seed=int(seed),
                    device=device,
                )
                diagnostics, paths = counterfactual_diagnostics(model, probe, device=device)
                source_penalties = source_diagnostic_penalties(
                    model,
                    train,
                    max_samples=int(config.get("diagnostic_samples", 512)),
                    device=device,
                )
                row = {
                    "experiment_id": config["experiment_id"],
                    "seed": int(seed),
                    "method": method,
                    "strength": float(strength),
                    **train_metrics,
                    "source_accuracy": mean_accuracy(model, source_eval, device),
                    "target_accuracy": mean_accuracy(model, (target,), device),
                    **source_penalties,
                    **diagnostics,
                }
                metric_rows.append(row)
                path_records.append(
                    {
                        "seed": int(seed),
                        "method": method,
                        "strength": float(strength),
                        "paths": paths,
                    }
                )
                np.savez_compressed(
                    output / "paths" / f"seed{seed}-{method}-l{float(strength):g}.npz",
                    **paths,
                )
                torch.save(
                    model.state_dict(),
                    output
                    / "checkpoints"
                    / f"seed{seed}-{method}-l{float(strength):g}.pt",
                )
                print(
                    f"seed={seed} method={method} lambda={float(strength):g} "
                    f"source={row['source_accuracy']:.3f} target={row['target_accuracy']:.3f} "
                    f"pred_color={row['prediction_color_response']:.4g}"
                )

    _write_csv(output / "metrics.csv", metric_rows)
    _plot_paths(output / "counterfactual_paths.png", path_records)
    _plot_summary(output / "feature_response_summary.png", metric_rows)
    if example_probe is not None:
        _plot_examples(output / "counterfactual_examples.png", example_probe, path_records)
    environment = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "device": str(device),
        "wall_seconds": time.time() - started,
    }
    (output / "environment.json").write_text(json.dumps(environment, indent=2))
    return metric_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    run(arguments.config, arguments.output)


if __name__ == "__main__":
    main()
