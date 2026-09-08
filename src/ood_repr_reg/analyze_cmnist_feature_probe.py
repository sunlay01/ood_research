"""Paired multi-seed analysis for CMNIST-VIS-001."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OWN_PENALTY = {
    "l2": "diagnostic_l2",
    "irmv1": "diagnostic_irmv1",
    "coral": "diagnostic_coral",
}


def paired_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    erm = metrics[metrics["method"] == "erm"].set_index("seed")
    rows = []
    for row in metrics[metrics["method"] != "erm"].itertuples(index=False):
        baseline = erm.loc[row.seed]
        own_penalty = OWN_PENALTY[row.method]
        rows.append(
            {
                "seed": row.seed,
                "method": row.method,
                "strength": row.strength,
                "own_penalty_ratio": getattr(row, own_penalty) / baseline[own_penalty],
                "source_accuracy_delta": row.source_accuracy - baseline.source_accuracy,
                "target_accuracy_delta": row.target_accuracy - baseline.target_accuracy,
                "balanced_accuracy_delta": row.balanced_accuracy
                - baseline.balanced_accuracy,
                "latent_color_ratio": row.latent_color_response
                / baseline.latent_color_response,
                "prediction_color_ratio": row.prediction_color_response
                / baseline.prediction_color_response,
                "probability_color_ratio": row.probability_color_response
                / baseline.probability_color_response,
                "task_signal_ratio": row.task_signal / baseline.task_signal,
                "consistency_delta": row.counterfactual_prediction_consistency
                - baseline.counterfactual_prediction_consistency,
            }
        )
    return pd.DataFrame(rows)


def aggregate_metrics(paired: pd.DataFrame) -> pd.DataFrame:
    value_columns = [column for column in paired.columns if column not in {"seed", "method", "strength"}]
    grouped = paired.groupby(["method", "strength"])[value_columns].agg(["mean", "std"])
    grouped.columns = [f"{metric}_{statistic}" for metric, statistic in grouped.columns]
    return grouped.reset_index()


def _labels(summary: pd.DataFrame) -> list[str]:
    return [f"{row.method.upper()} {row.strength:g}" for row in summary.itertuples()]


def plot_paired_effects(summary: pd.DataFrame, path: Path) -> None:
    panels = (
        ("own_penalty_ratio", "Own source penalty / ERM", 1.0, False),
        ("target_accuracy_delta", "Sign-flip accuracy change", 0.0, True),
        ("balanced_accuracy_delta", "Balanced accuracy change", 0.0, True),
        ("latent_color_ratio", "Latent color response / ERM", 1.0, False),
        ("prediction_color_ratio", "Head-used color response / ERM", 1.0, False),
        ("task_signal_ratio", "Balanced task signal / ERM", 1.0, False),
    )
    labels = _labels(summary)
    colors = ["#2563eb", "#60a5fa", "#d97706", "#f59e0b", "#059669", "#34d399"]
    figure, axes = plt.subplots(2, 3, figsize=(14, 8))
    for axis, (metric, title, reference, percentage) in zip(axes.flat, panels):
        means = summary[f"{metric}_mean"].to_numpy()
        errors = summary[f"{metric}_std"].fillna(0.0).to_numpy()
        if percentage:
            means = 100.0 * means
            errors = 100.0 * errors
            reference = 100.0 * reference
        positions = np.arange(len(summary))
        axis.bar(positions, means, yerr=errors, capsize=3, color=colors[: len(summary)])
        axis.axhline(reference, color="#111827", linewidth=1.0, linestyle="--")
        axis.set_title(title)
        axis.set_xticks(positions, labels, rotation=40, ha="right", fontsize=8)
        axis.grid(axis="y", alpha=0.2)
        if percentage:
            axis.set_ylabel("percentage points")
    figure.suptitle("CMNIST regularization effects relative to same-seed ERM", fontsize=14)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(path, dpi=180)
    plt.close(figure)


def run(input_path: Path, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = pd.read_csv(input_path)
    paired = paired_metrics(metrics)
    summary = aggregate_metrics(paired)
    paired.to_csv(output_dir / "paired_metrics.csv", index=False)
    summary.to_csv(output_dir / "aggregate_summary.csv", index=False)
    plot_paired_effects(summary, output_dir / "paired_effects.png")
    return paired, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    run(arguments.input, arguments.output_dir)


if __name__ == "__main__":
    main()
