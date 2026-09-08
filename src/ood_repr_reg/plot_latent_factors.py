"""Plot the discovered latent response factors and regularization paths."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHOD_ORDER = ["l1", "l2", "irmv1", "mmd", "coral", "grad_align", "hess_align"]
METHOD_COLORS = {
    "l1": "#D55E00",
    "l2": "#E69F00",
    "irmv1": "#0072B2",
    "mmd": "#009E73",
    "coral": "#56B4E9",
    "grad_align": "#CC79A7",
    "hess_align": "#333333",
}
CLUSTER_COLORS = {1: "#0072B2", 2: "#D55E00", 3: "#009E73"}
FEATURE_LABELS = {
    "delta_source_mse": "source MSE",
    "delta_core_probe_r2": "core probe R2",
    "delta_spurious_probe_r2": "spurious probe R2",
    "delta_domain_probe_accuracy": "domain probe acc.",
    "delta_latent_mean_gap": "latent mean gap",
    "delta_latent_covariance_gap": "latent covariance gap",
    "delta_effective_rank": "effective rank",
    "delta_latent_variance": "latent variance",
    "delta_encoder_core_norm": "core encoder norm",
    "delta_encoder_spurious_norm": "spurious encoder norm",
    "delta_encoder_domain_norm": "domain encoder norm",
    "delta_encoder_nuisance_norm": "nuisance encoder norm",
    "delta_source_head_norm": "source head norm",
}


def _load(input_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scores = pd.read_csv(input_dir / "discovered_factor_scores.csv")
    loadings = pd.read_csv(input_dir / "discovered_factor_loadings.csv")
    profiles = pd.read_csv(input_dir / "discovered_cluster_profiles.csv")
    return scores, loadings, profiles


def _style_axes(axis: plt.Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(True, alpha=0.18, linewidth=0.6)
    axis.set_axisbelow(True)


def _plot_factor_map(scores: pd.DataFrame, loadings: pd.DataFrame, profiles: pd.DataFrame, path: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(15, 11), constrained_layout=True)
    figure.suptitle("Data-driven latent response factors", fontsize=16, fontweight="bold")

    axis = axes[0, 0]
    for cluster, group in scores.groupby("cluster"):
        axis.scatter(
            group["PC1"],
            group["PC2"],
            s=34,
            alpha=0.78,
            color=CLUSTER_COLORS.get(int(cluster), "#777777"),
            label=f"cluster {int(cluster)}",
            edgecolor="white",
            linewidth=0.35,
        )
    axis.axhline(0.0, color="#999999", linewidth=0.6)
    axis.axvline(0.0, color="#999999", linewidth=0.6)
    axis.set_xlabel("PC1: dominant overall response direction")
    axis.set_ylabel("PC2: domain displacement direction")
    axis.set_title("Run map (ERM-paired, target-free discovery)")
    axis.legend(frameon=False, fontsize=9)
    _style_axes(axis)

    axis = axes[0, 1]
    loading_matrix = loadings.set_index("feature")[["PC1", "PC2", "PC3", "PC4"]]
    image = axis.imshow(loading_matrix.to_numpy(), cmap="coolwarm", vmin=-0.6, vmax=0.6, aspect="auto")
    axis.set_xticks(range(4), ["PC1", "PC2", "PC3", "PC4"])
    axis.set_yticks(range(len(loading_matrix)), [FEATURE_LABELS.get(x, x) for x in loading_matrix.index], fontsize=8)
    axis.set_title("PCA loadings")
    for row in range(loading_matrix.shape[0]):
        for column in range(loading_matrix.shape[1]):
            axis.text(column, row, f"{loading_matrix.iloc[row, column]:.2f}", ha="center", va="center", fontsize=7)
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="loading")

    axis = axes[1, 0]
    profile_matrix = profiles.pivot(index="cluster", columns="feature", values="mean_standardized_delta")
    profile_matrix = profile_matrix.reindex(columns=loading_matrix.index)
    image = axis.imshow(profile_matrix.to_numpy(), cmap="coolwarm", vmin=-2.5, vmax=2.5, aspect="auto")
    axis.set_xticks(range(len(profile_matrix.columns)), [FEATURE_LABELS.get(x, x) for x in profile_matrix.columns], rotation=55, ha="right", fontsize=8)
    axis.set_yticks(range(len(profile_matrix.index)), [f"cluster {int(x)}" for x in profile_matrix.index])
    axis.set_title("Cluster profiles: mean standardized ERM delta")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="mean z-scored delta")

    axis = axes[1, 1]
    for method in METHOD_ORDER:
        group = scores[scores["method"] == method]
        axis.scatter(
            group["PC1"],
            group["target_signed_gap_vs_erm"],
            s=30,
            alpha=0.68,
            color=METHOD_COLORS[method],
            label=method,
            edgecolor="white",
            linewidth=0.3,
        )
    axis.axhline(0.0, color="#444444", linewidth=0.8)
    axis.set_xlabel("PC1 score")
    axis.set_ylabel("target signed gap vs ERM")
    axis.set_title("External outcome check (post-hoc only)")
    axis.legend(frameon=False, fontsize=8, ncol=2)
    _style_axes(axis)

    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_paths(scores: pd.DataFrame, path: Path) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True, constrained_layout=True)
    figure.suptitle("How regularization moves the discovered factors", fontsize=16, fontweight="bold")
    for axis, factor, title in zip(
        axes,
        ["PC1", "PC2"],
        ["PC1: overall representation response", "PC2: domain displacement response"],
    ):
        for method in METHOD_ORDER:
            group = scores[scores["method"] == method]
            summary = group.groupby("lambda")[factor].agg(["mean", "std"]).reset_index()
            axis.errorbar(
                summary["lambda"],
                summary["mean"],
                yerr=summary["std"].fillna(0.0),
                marker="o",
                markersize=4,
                linewidth=1.6,
                capsize=2,
                color=METHOD_COLORS[method],
                label=method,
            )
        axis.axhline(0.0, color="#444444", linewidth=0.8)
        axis.set_ylabel(f"{factor} score")
        axis.set_title(title, loc="left", fontsize=11)
        _style_axes(axis)
    axes[-1].set_xlabel("regularization strength lambda")
    axes[0].legend(frameon=False, ncol=4, fontsize=9)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scores, loadings, profiles = _load(args.input_dir)
    _plot_factor_map(scores, loadings, profiles, args.output_dir / "discovered_factor_map.png")
    _plot_paths(scores, args.output_dir / "discovered_factor_paths.png")
    print(f"wrote plots to {args.output_dir}")


if __name__ == "__main__":
    main()
