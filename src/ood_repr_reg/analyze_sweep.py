"""Aggregate the exploratory sweep and create compact diagnostic plots."""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ERROR_METRICS = ["source_mse", "cross_domain_mse", "cross_task_mse", "joint_mse"]
LATENT_METRICS = [
    "core_probe_r2",
    "spurious_probe_r2",
    "domain_probe_accuracy",
    "latent_mean_gap",
    "latent_covariance_gap",
    "effective_rank",
    "latent_variance",
]


def _final_rows(frame: pd.DataFrame) -> pd.DataFrame:
    group_keys = ["method", "lambda", "seed"]
    final_index = frame.groupby(group_keys)["step"].idxmax()
    return frame.loc[final_index].sort_values(group_keys).reset_index(drop=True)


def _write_correlations(final: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    rows = []
    for method, method_frame in final.groupby("method"):
        if method == "erm" or method_frame["own_regularizer"].nunique() < 3:
            continue
        for metric in ERROR_METRICS + LATENT_METRICS:
            correlation, p_value = spearmanr(
                method_frame["own_regularizer"], method_frame[metric]
            )
            rows.append(
                {
                    "method": method,
                    "metric": metric,
                    "spearman_rho": correlation,
                    "nominal_p_value": p_value,
                    "n": len(method_frame),
                    "interpretation": "exploratory_only",
                }
            )
    correlations = pd.DataFrame(rows)
    correlations.to_csv(output_dir / "final_correlations.csv", index=False)
    return correlations


def _exact_sign_flip_p_value(differences: np.ndarray) -> float:
    observed = abs(float(np.mean(differences)))
    permuted = []
    for signs in itertools.product((-1.0, 1.0), repeat=len(differences)):
        permuted.append(abs(float(np.mean(differences * np.asarray(signs)))))
    return float(np.mean(np.asarray(permuted) >= observed - 1e-12))


def _benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    order = np.argsort(p_values)
    ranked = p_values[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1].clip(0.0, 1.0)
    result = np.empty_like(adjusted)
    result[order] = adjusted
    return result


def _write_paired_effects(final: pd.DataFrame, output_dir: Path) -> None:
    erm = final[final["method"] == "erm"].set_index("seed")
    rows = []
    for (method, strength), method_frame in final[final["method"] != "erm"].groupby(
        ["method", "lambda"]
    ):
        indexed = method_frame.set_index("seed")
        shared_seeds = indexed.index.intersection(erm.index)
        for metric in ERROR_METRICS + LATENT_METRICS:
            differences = (
                indexed.loc[shared_seeds, metric].to_numpy()
                - erm.loc[shared_seeds, metric].to_numpy()
            )
            rows.append(
                {
                    "method": method,
                    "lambda": strength,
                    "metric": metric,
                    "mean_paired_delta_vs_erm": float(differences.mean()),
                    "std_paired_delta": (
                        float(differences.std(ddof=1)) if len(differences) > 1 else 0.0
                    ),
                    "positive_seed_count": int(np.sum(differences > 0.0)),
                    "negative_seed_count": int(np.sum(differences < 0.0)),
                    "zero_seed_count": int(np.sum(differences == 0.0)),
                    "n_seeds": len(shared_seeds),
                    "exact_two_sided_sign_flip_p": _exact_sign_flip_p_value(differences),
                }
            )
    effects = pd.DataFrame(rows)
    effects["bh_q_across_all_reported_tests"] = _benjamini_hochberg(
        effects["exact_two_sided_sign_flip_p"].to_numpy()
    )
    effects["interpretation"] = "exploratory_only"
    effects.to_csv(output_dir / "paired_effects.csv", index=False)


def _write_regularizer_reduction(frame: pd.DataFrame, output_dir: Path) -> None:
    keys = ["method", "lambda", "seed"]
    initial = frame.loc[frame.groupby(keys)["step"].idxmin(), keys + ["own_regularizer"]]
    final = frame.loc[frame.groupby(keys)["step"].idxmax(), keys + ["own_regularizer"]]
    initial = initial.rename(columns={"own_regularizer": "initial_regularizer"})
    final = final.rename(columns={"own_regularizer": "final_regularizer"})
    reduction = initial.merge(final, on=keys)
    reduction = reduction[reduction["method"] != "erm"].copy()
    reduction["final_to_initial_ratio"] = (
        reduction["final_regularizer"] / reduction["initial_regularizer"]
    )
    reduction.to_csv(output_dir / "regularizer_reduction.csv", index=False)


def _write_bound_diagnostics(final: pd.DataFrame, output_dir: Path) -> None:
    rows = []
    comparison_pairs = {
        "cross_domain_mse": "source_mse",
        "cross_task_mse": "source_mse",
        "joint_mse": "cross_task_mse",
    }
    for method, method_frame in final.groupby("method"):
        if method == "erm":
            continue
        omega_root = np.sqrt(
            np.clip(method_frame["own_regularizer"].to_numpy(), 0.0, None)
        )
        for target_metric, base_metric in comparison_pairs.items():
            gaps = (
                method_frame[target_metric].to_numpy()
                - method_frame[base_metric].to_numpy()
            )
            valid = omega_root > 1e-10
            required = np.maximum(gaps[valid], 0.0) / omega_root[valid]
            rows.append(
                {
                    "method": method,
                    "target_metric": target_metric,
                    "base_metric": base_metric,
                    "max_required_c_for_source_plus_c_sqrt_omega": (
                        float(required.max()) if required.size else np.nan
                    ),
                    "median_required_c": float(np.median(required)) if required.size else np.nan,
                    "near_zero_omega_positive_gap_count": int(
                        np.sum((~valid) & (gaps > 0.0))
                    ),
                    "note": (
                        "descriptive finite-sample diagnostic, not a proven bound; "
                        "the cross-task source_mse anchor compares different tasks"
                    ),
                }
            )
    pd.DataFrame(rows).to_csv(output_dir / "bound_diagnostics.csv", index=False)


def _plot_error_paths(summary: pd.DataFrame, output_dir: Path) -> None:
    methods = [method for method in summary["method"].unique() if method != "erm"]
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for axis, metric in zip(axes.flatten(), ERROR_METRICS, strict=True):
        for method in methods:
            method_frame = summary[summary["method"] == method]
            axis.plot(
                method_frame["lambda"],
                method_frame[f"{metric}_mean"],
                marker="o",
                label=method,
            )
        erm_value = summary.loc[summary["method"] == "erm", f"{metric}_mean"]
        if not erm_value.empty:
            axis.axhline(
                float(erm_value.iloc[0]),
                color="black",
                linestyle="--",
                linewidth=1,
            )
        axis.set_xscale("log")
        axis.set_xlabel("regularization strength")
        axis.set_ylabel(metric)
        axis.set_title(metric)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=4)
    figure.savefig(output_dir / "error_paths.png", dpi=180)
    plt.close(figure)


def _plot_omega_error(final: pd.DataFrame, output_dir: Path) -> None:
    methods = [method for method in final["method"].unique() if method != "erm"]
    figure, axes = plt.subplots(2, 4, figsize=(15, 7), constrained_layout=True)
    for axis, method in zip(axes.flatten(), methods, strict=False):
        method_frame = final[final["method"] == method]
        scatter = axis.scatter(
            method_frame["own_regularizer"],
            method_frame["joint_mse"],
            c=method_frame["lambda"],
            cmap="viridis",
            s=35,
        )
        axis.set_xscale("symlog", linthresh=1e-8)
        axis.set_title(method)
        axis.set_xlabel("final own regularizer")
        axis.set_ylabel("joint MSE")
        figure.colorbar(scatter, ax=axis, label="lambda")
    for axis in axes.flatten()[len(methods) :]:
        axis.axis("off")
    figure.savefig(output_dir / "omega_vs_joint_error.png", dpi=180)
    plt.close(figure)


def analyze(input_dir: Path) -> None:
    frame = pd.read_csv(input_dir / "trajectory.csv")
    final = _final_rows(frame)
    metrics = ERROR_METRICS + LATENT_METRICS + ["own_regularizer"]
    summary = (
        final.groupby(["method", "lambda"])[metrics]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        column if isinstance(column, str) else "_".join(part for part in column if part)
        for column in summary.columns
    ]
    summary.to_csv(input_dir / "final_summary.csv", index=False)
    _write_correlations(final, input_dir)
    _write_paired_effects(final, input_dir)
    _write_regularizer_reduction(frame, input_dir)
    _write_bound_diagnostics(final, input_dir)
    _plot_error_paths(summary, input_dir)
    _plot_omega_error(final, input_dir)
    print(f"analyzed {len(final)} final runs from {len(frame)} checkpoint rows")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.input)


if __name__ == "__main__":
    main()
