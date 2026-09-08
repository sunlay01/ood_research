"""Aggregate the exploratory sweep and create compact diagnostic plots."""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .certificates import CrossDomainCertificate


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
LATENT_ORACLE_METRICS = [
    "source_latent_oracle_mse",
    "target_latent_oracle_mse",
    "source_raw_oracle_mse",
    "target_raw_oracle_mse",
    "source_head_mismatch_mse",
    "target_head_mismatch_mse",
    "source_representation_gap_mse",
    "target_representation_gap_mse",
    "latent_oracle_cross_domain_gap",
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
        for metric in ERROR_METRICS + LATENT_METRICS + LATENT_ORACLE_METRICS:
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
    erm = _erm_by_seed(final)
    rows = []
    for (method, strength), method_frame in final[final["method"] != "erm"].groupby(
        ["method", "lambda"]
    ):
        indexed = method_frame.set_index("seed")
        shared_seeds = indexed.index.intersection(erm.index)
        for metric in ERROR_METRICS + LATENT_METRICS + LATENT_ORACLE_METRICS:
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


def _erm_by_seed(final: pd.DataFrame) -> pd.DataFrame:
    """Return the unique ERM reference row for every non-ERM seed."""

    erm_rows = final[final["method"] == "erm"]
    if erm_rows.empty or erm_rows["seed"].duplicated().any():
        raise ValueError("expected exactly one final ERM row per seed")
    non_erm_seeds = set(final.loc[final["method"] != "erm", "seed"])
    erm_seeds = set(erm_rows["seed"])
    missing = non_erm_seeds - erm_seeds
    if missing:
        raise ValueError(f"missing final ERM rows for seeds: {sorted(missing)}")
    return erm_rows.set_index("seed")


def _write_error_certificates(final: pd.DataFrame, output_dir: Path) -> None:
    """Write certificate inputs and target degradation as separate columns.

    The target columns are evaluation labels.  They are never included in a
    source-only certificate view and must not be used for model selection.
    """

    erm = _erm_by_seed(final)
    rows = []
    for (method, strength), method_frame in final[final["method"] != "erm"].groupby(
        ["method", "lambda"]
    ):
        for _, row in method_frame.iterrows():
            seed = row["seed"]
            if seed not in erm.index:
                continue
            erm_row = erm.loc[seed]
            certificate = CrossDomainCertificate(
                algorithm=str(method),
                source_risk=float(row["source_mse"]),
                erm_source_risk=float(erm_row["source_mse"]),
                original_metric=float(row["own_regularizer"]),
                target_risk=float(row["cross_domain_mse"]),
                erm_target_risk=float(erm_row["cross_domain_mse"]),
                representation_covariate_shift=float(row["latent_covariance_gap"]),
                head_mismatch=float(row[f"diagnostic_{method}"]),
                source_domain_spread=float(row["latent_mean_gap"]),
                complexity=float(row["source_head_norm"]),
            )
            for view, values in certificate.metric_views().items():
                rows.append(
                    {
                        "method": method,
                        "lambda": strength,
                        "seed": seed,
                        "certificate_view": view,
                        "target_risk": certificate.target_risk,
                        "erm_target_risk": certificate.erm_target_risk,
                        "positive_target_degradation": certificate.target_degradation,
                        "interpretation": "descriptive_evaluation_only",
                        **values,
                    }
                )
    pd.DataFrame(rows).to_csv(output_dir / "cross_domain_certificates.csv", index=False)


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


def _write_erm_paired_latent_decomposition(final: pd.DataFrame, output_dir: Path) -> None:
    """Write the algorithm-vs-ERM latent-space error accounting.

    These are cross-fitted finite-sample diagnostics.  The latent oracle is a
    linear ridge probe on the learned Z, so its gap from the raw oracle is not
    a population representation theorem and may include finite-sample and
    model-class effects.  It is used only to account for the observed
    algorithm-vs-ERM target-risk difference.
    """
    erm = _erm_by_seed(final)
    rows = []
    for _, row in final[final["method"] != "erm"].iterrows():
        seed = row["seed"]
        if seed not in erm.index:
            continue
        reference = erm.loc[seed]
        values: dict[str, float | str] = {
            "method": row["method"],
            "lambda": row["lambda"],
            "seed": seed,
            "source_mse": row["source_mse"],
            "target_domain_mse": row["cross_domain_mse"],
            "source_excess_vs_erm": row["source_mse"] - reference["source_mse"],
            "target_signed_gap_vs_erm": row["cross_domain_mse"]
            - reference["cross_domain_mse"],
            "target_positive_degradation_vs_erm": max(
                row["cross_domain_mse"] - reference["cross_domain_mse"], 0.0
            ),
            "source_target_gap": row["cross_domain_mse"] - row["source_mse"],
        }
        for metric in LATENT_ORACLE_METRICS:
            values[metric] = float(row[metric])
            values[f"{metric}_delta_vs_erm"] = float(row[metric] - reference[metric])
        values["target_gap_accounting_residual"] = float(
            values["target_signed_gap_vs_erm"]
            - values["target_latent_oracle_mse_delta_vs_erm"]
            - values["target_head_mismatch_mse_delta_vs_erm"]
        )
        values["source_gap_accounting_residual"] = float(
            values["source_excess_vs_erm"]
            - values["source_latent_oracle_mse_delta_vs_erm"]
            - values["source_head_mismatch_mse_delta_vs_erm"]
        )
        rows.append(values)
    pd.DataFrame(rows).to_csv(
        output_dir / "erm_paired_latent_decomposition.csv", index=False
    )


def _write_bound_diagnostics(final: pd.DataFrame, output_dir: Path) -> None:
    rows = []
    erm = _erm_by_seed(final)
    for method, method_frame in final[final["method"] != "erm"].groupby("method"):
        required_omega = []
        required_omega_source = []
        near_zero_count = 0
        for _, row in method_frame.iterrows():
            seed = row["seed"]
            if seed not in erm.index:
                continue
            erm_row = erm.loc[seed]
            degradation = max(
                float(row["cross_domain_mse"] - erm_row["cross_domain_mse"]), 0.0
            )
            source_excess = max(
                float(row["source_mse"] - erm_row["source_mse"]), 0.0
            )
            omega_root = float(np.sqrt(max(row["own_regularizer"], 0.0)))
            if omega_root <= 1e-10:
                if degradation > 0.0:
                    near_zero_count += 1
                continue
            required_omega.append(degradation / omega_root)
            required_omega_source.append(max(degradation - source_excess, 0.0) / omega_root)
        rows.append(
            {
                "method": method,
                "target_metric": "positive_target_degradation",
                "base_metric": "erm_cross_domain_mse",
                "max_required_c_for_omega_only": (
                    float(max(required_omega)) if required_omega else np.nan
                ),
                "median_required_c_for_omega_only": (
                    float(np.median(required_omega)) if required_omega else np.nan
                ),
                "max_required_c_for_source_excess_plus_omega": (
                    float(max(required_omega_source)) if required_omega_source else np.nan
                ),
                "median_required_c_for_source_excess_plus_omega": (
                    float(np.median(required_omega_source))
                    if required_omega_source
                    else np.nan
                ),
                "near_zero_omega_positive_gap_count": near_zero_count,
                "note": (
                    "descriptive finite-sample stress test, not a proven bound; "
                    "coverage and target risk remain unavailable to the source-only view"
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
    metrics = ERROR_METRICS + LATENT_METRICS + LATENT_ORACLE_METRICS + ["own_regularizer"]
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
    _write_error_certificates(final, input_dir)
    _write_regularizer_reduction(frame, input_dir)
    _write_erm_paired_latent_decomposition(final, input_dir)
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
