"""Discover latent response factors from an ERM-paired training trajectory.

This module deliberately separates discovery from interpretation.  PCA and
clustering only see source/latent observations.  Target risk is joined after
the discovery step as an external outcome, so it cannot leak into the latent
factor construction or cluster selection.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist, pdist
from scipy.stats import spearmanr


DISCOVERY_FEATURES = [
    "source_mse",
    "core_probe_r2",
    "spurious_probe_r2",
    "domain_probe_accuracy",
    "latent_mean_gap",
    "latent_covariance_gap",
    "effective_rank",
    "latent_variance",
    "encoder_core_norm",
    "encoder_spurious_norm",
    "encoder_domain_norm",
    "encoder_nuisance_norm",
    "source_head_norm",
]

TARGET_COLUMNS = {
    "cross_domain_mse",
    "cross_task_mse",
    "joint_mse",
    "target_latent_oracle_mse",
    "target_raw_oracle_mse",
    "target_head_mismatch_mse",
    "target_representation_gap_mse",
    "latent_oracle_cross_domain_gap",
}

FEATURE_SUBSETS = {
    "all_observations": DISCOVERY_FEATURES,
    "latent_only": [
        "core_probe_r2",
        "spurious_probe_r2",
        "domain_probe_accuracy",
        "latent_mean_gap",
        "latent_covariance_gap",
        "effective_rank",
        "latent_variance",
        "encoder_core_norm",
        "encoder_spurious_norm",
        "encoder_domain_norm",
        "encoder_nuisance_norm",
    ],
    "source_and_probes": [
        "source_mse",
        "core_probe_r2",
        "spurious_probe_r2",
        "domain_probe_accuracy",
        "source_head_norm",
    ],
    "without_source_mse": [
        feature for feature in DISCOVERY_FEATURES if feature != "source_mse"
    ],
    "without_encoder_norms": [
        feature
        for feature in DISCOVERY_FEATURES
        if not feature.startswith("encoder_")
    ],
}


def final_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Select the last checkpoint for every method/strength/seed run."""

    keys = ["method", "lambda", "seed"]
    indices = frame.groupby(keys, sort=False)["step"].idxmax()
    return frame.loc[indices].sort_values(keys).reset_index(drop=True)


def erm_paired_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return non-ERM rows with every discovery field paired to same-seed ERM."""

    final = final_rows(frame)
    erm = final[final["method"] == "erm"].set_index("seed")
    if erm.empty or erm.index.duplicated().any():
        raise ValueError("expected exactly one final ERM row per seed")

    runs = final[final["method"] != "erm"].copy()
    if runs.empty:
        raise ValueError("trajectory contains no non-ERM runs")
    missing = sorted(set(runs["seed"]) - set(erm.index))
    if missing:
        raise ValueError(f"missing ERM reference for seeds: {missing}")

    for feature in DISCOVERY_FEATURES:
        runs[f"delta_{feature}"] = runs.apply(
            lambda row: float(row[feature] - erm.loc[row["seed"], feature]), axis=1
        )
    runs["target_signed_gap_vs_erm"] = runs.apply(
        lambda row: float(
            row["cross_domain_mse"] - erm.loc[row["seed"], "cross_domain_mse"]
        ),
        axis=1,
    )
    runs["target_positive_degradation_vs_erm"] = runs[
        "target_signed_gap_vs_erm"
    ].clip(lower=0.0)
    runs["source_excess_vs_erm"] = runs.apply(
        lambda row: float(row["source_mse"] - erm.loc[row["seed"], "source_mse"]),
        axis=1,
    )
    return runs


def _standardize(
    matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    means = matrix.mean(axis=0)
    scales = matrix.std(axis=0)
    keep = scales > 1e-12
    if not np.any(keep):
        raise ValueError("all discovery features are constant")
    return (matrix[:, keep] - means[keep]) / scales[keep], means, scales, keep


def _orient_components(components: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fix PCA sign using the largest loading for reproducible reports."""

    components = components.copy()
    scores = scores.copy()
    for index in range(components.shape[0]):
        pivot = int(np.argmax(np.abs(components[index])))
        if components[index, pivot] < 0:
            components[index] *= -1.0
            scores[:, index] *= -1.0
    return components, scores


def pca(matrix: np.ndarray, max_components: int = 4) -> dict[str, np.ndarray]:
    standardized, means, scales, keep = _standardize(matrix)
    _, singular_values, vt = np.linalg.svd(standardized, full_matrices=False)
    rank = min(max_components, vt.shape[0], standardized.shape[0])
    components = vt[:rank]
    scores = standardized @ components.T
    components, scores = _orient_components(components, scores)
    eigenvalues = singular_values**2 / max(standardized.shape[0] - 1, 1)
    explained = eigenvalues / eigenvalues.sum()
    return {
        "standardized": standardized,
        "means": means,
        "scales": scales,
        "keep": keep,
        "components": components,
        "scores": scores,
        "explained_variance_ratio": explained[:rank],
    }


def silhouette_score(matrix: np.ndarray, labels: np.ndarray) -> float:
    """Compute the mean Euclidean silhouette without sklearn."""

    distances = cdist(matrix, matrix, metric="euclidean")
    values = []
    for index in range(len(matrix)):
        own = labels == labels[index]
        own[index] = False
        if not np.any(own):
            values.append(0.0)
            continue
        within = distances[index, own].mean()
        between = np.inf
        for label in np.unique(labels):
            if label == labels[index]:
                continue
            between = min(between, distances[index, labels == label].mean())
        values.append((between - within) / max(within, between, 1e-12))
    return float(np.mean(values))


def choose_clusters(matrix: np.ndarray, min_k: int = 2, max_k: int = 6) -> dict[str, object]:
    """Choose k by silhouette, with the smallest k as a deterministic tie-break."""

    upper = min(max_k, len(matrix) - 1)
    if upper < min_k:
        raise ValueError("not enough runs for cluster selection")
    tree = linkage(pdist(matrix), method="ward")
    rows = []
    for k in range(min_k, upper + 1):
        labels = fcluster(tree, t=k, criterion="maxclust")
        rows.append({"k": k, "silhouette": silhouette_score(matrix, labels)})
    scores = pd.DataFrame(rows)
    best = scores.sort_values(["silhouette", "k"], ascending=[False, True]).iloc[0]
    k = int(best["k"])
    labels = fcluster(tree, t=k, criterion="maxclust")
    return {"scores": scores, "labels": labels, "selected_k": k}


def adjusted_rand_index(first: np.ndarray, second: np.ndarray) -> float:
    """Calculate ARI for cluster labels without a sklearn dependency."""

    first_values = np.unique(first)
    second_values = np.unique(second)
    contingency = np.zeros((len(first_values), len(second_values)), dtype=np.int64)
    for i, first_value in enumerate(first_values):
        for j, second_value in enumerate(second_values):
            contingency[i, j] = np.sum((first == first_value) & (second == second_value))

    def choose_two(values: np.ndarray) -> float:
        return float(np.sum(values * (values - 1) / 2.0))

    pairs = choose_two(contingency)
    row_pairs = choose_two(contingency.sum(axis=1))
    col_pairs = choose_two(contingency.sum(axis=0))
    total_pairs = choose_two(np.array([len(first)]))
    expected = row_pairs * col_pairs / total_pairs if total_pairs else 0.0
    maximum = 0.5 * (row_pairs + col_pairs)
    if maximum == expected:
        return 1.0
    return float((pairs - expected) / (maximum - expected))


def factor_match_score(first: np.ndarray, second: np.ndarray) -> float:
    """Match PCA columns and return mean absolute score correlation."""

    count = min(first.shape[1], second.shape[1])
    correlations = np.zeros((count, count))
    for i in range(count):
        for j in range(count):
            correlations[i, j] = abs(np.corrcoef(first[:, i], second[:, j])[0, 1])
    rows, columns = linear_sum_assignment(-correlations)
    return float(correlations[rows, columns].mean())


def discover(matrix: np.ndarray, feature_names: list[str], selected_k: int | None = None) -> dict[str, object]:
    """Run PCA and Ward clustering on one feature view."""

    result = pca(matrix)
    cluster_input = result["standardized"]
    if selected_k is None:
        cluster = choose_clusters(cluster_input)
    else:
        tree = linkage(pdist(cluster_input), method="ward")
        labels = fcluster(tree, t=selected_k, criterion="maxclust")
        cluster = {
            "scores": pd.DataFrame(
                [{"k": selected_k, "silhouette": silhouette_score(cluster_input, labels)}]
            ),
            "labels": labels,
            "selected_k": selected_k,
        }
    used_feature_names = [
        name for name, keep in zip(feature_names, result["keep"]) if keep
    ]
    loadings = pd.DataFrame(
        result["components"].T,
        index=used_feature_names,
        columns=[f"PC{i + 1}" for i in range(result["components"].shape[0])],
    ).reset_index(names="feature")
    loadings["max_abs_loading"] = loadings.drop(columns="feature").abs().max(axis=1)
    return {"pca": result, "cluster": cluster, "loadings": loadings}


def _feature_names_for_subset(subset: list[str]) -> list[str]:
    return [f"delta_{feature}" for feature in subset]


def _external_correlations(scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for factor in [column for column in scores if column.startswith("PC")]:
        for outcome in ["target_signed_gap_vs_erm", "target_positive_degradation_vs_erm"]:
            rho, p_value = spearmanr(scores[factor], scores[outcome])
            rows.append(
                {
                    "factor": factor,
                    "outcome": outcome,
                    "spearman_rho": float(rho),
                    "nominal_p_value": float(p_value),
                    "role": "external_outcome_only",
                }
            )
    return pd.DataFrame(rows)


def _factor_labels(loadings: pd.DataFrame, n: int = 3) -> dict[str, str]:
    labels = {}
    for factor in [column for column in loadings if column.startswith("PC")]:
        top = loadings.loc[loadings[factor].abs().nlargest(n).index, "feature"]
        labels[factor] = " / ".join(top.tolist())
    return labels


def _markdown_table(frame: pd.DataFrame, digits: int = 3) -> str:
    """Render a small dependency-free Markdown table for the report."""

    def format_value(value: object) -> str:
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.{digits}f}"
        return str(value)

    columns = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---:" if frame[column].dtype.kind in "biufc" else "---" for column in frame.columns) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(format_value(value) for value in row) + " |")
    return "\n".join(lines)


def _stability_analysis(
    paired: pd.DataFrame,
    full: dict[str, object],
    full_features: list[str],
) -> pd.DataFrame:
    base_labels = np.asarray(full["cluster"]["labels"])
    base_scores = np.asarray(full["pca"]["scores"])
    rows = []
    for name, subset in FEATURE_SUBSETS.items():
        feature_names = _feature_names_for_subset(subset)
        current = discover(paired[feature_names].to_numpy(float), feature_names, selected_k=int(full["cluster"]["selected_k"]))
        labels = np.asarray(current["cluster"]["labels"])
        rows.append(
            {
                "stability_view": f"feature_subset:{name}",
                "n_runs": len(paired),
                "adjusted_rand_index": adjusted_rand_index(base_labels, labels),
                "mean_abs_factor_score_correlation": factor_match_score(
                    base_scores, np.asarray(current["pca"]["scores"])
                ),
                "selected_k": int(full["cluster"]["selected_k"]),
            }
        )

    for seed in sorted(paired["seed"].unique()):
        mask = paired["seed"].to_numpy() != seed
        current = discover(
            paired.loc[mask, _feature_names_for_subset(full_features)].to_numpy(float),
            _feature_names_for_subset(full_features),
            selected_k=int(full["cluster"]["selected_k"]),
        )
        base_mask = paired["seed"].to_numpy() != seed
        rows.append(
            {
                "stability_view": f"leave_one_seed_out:{seed}",
                "n_runs": int(base_mask.sum()),
                "adjusted_rand_index": adjusted_rand_index(
                    base_labels[base_mask], np.asarray(current["cluster"]["labels"])
                ),
                "mean_abs_factor_score_correlation": factor_match_score(
                    base_scores[base_mask], np.asarray(current["pca"]["scores"])
                ),
                "selected_k": int(full["cluster"]["selected_k"]),
            }
        )
    return pd.DataFrame(rows)


def write_outputs(trajectory_path: Path, output_dir: Path) -> dict[str, object]:
    frame = pd.read_csv(trajectory_path)
    missing = sorted(set(DISCOVERY_FEATURES + list(TARGET_COLUMNS)) - set(frame.columns))
    if missing:
        raise ValueError(f"trajectory is missing required columns: {missing}")
    paired = erm_paired_frame(frame)
    output_dir.mkdir(parents=True, exist_ok=True)
    full_features = DISCOVERY_FEATURES
    full_names = _feature_names_for_subset(full_features)
    full = discover(paired[full_names].to_numpy(float), full_names)
    pca_result = full["pca"]

    scores = paired[["method", "lambda", "seed"]].copy()
    for index in range(pca_result["scores"].shape[1]):
        scores[f"PC{index + 1}"] = pca_result["scores"][:, index]
    scores["cluster"] = full["cluster"]["labels"]
    scores["target_signed_gap_vs_erm"] = paired["target_signed_gap_vs_erm"].to_numpy()
    scores["target_positive_degradation_vs_erm"] = paired[
        "target_positive_degradation_vs_erm"
    ].to_numpy()
    scores["source_excess_vs_erm"] = paired["source_excess_vs_erm"].to_numpy()
    scores.to_csv(output_dir / "discovered_factor_scores.csv", index=False)

    loadings = full["loadings"].copy()
    for index, value in enumerate(pca_result["explained_variance_ratio"]):
        loadings[f"{index + 1}_explained_variance_ratio"] = value
    loadings.to_csv(output_dir / "discovered_factor_loadings.csv", index=False)

    silhouette = full["cluster"]["scores"].copy()
    silhouette["selected"] = silhouette["k"] == int(full["cluster"]["selected_k"])
    silhouette.to_csv(output_dir / "discovered_cluster_selection.csv", index=False)

    cluster_rows = paired[["method", "lambda", "seed"]].copy()
    cluster_rows["cluster"] = full["cluster"]["labels"]
    cluster_rows["target_signed_gap_vs_erm"] = paired["target_signed_gap_vs_erm"].to_numpy()
    cluster_rows["target_positive_degradation_vs_erm"] = paired[
        "target_positive_degradation_vs_erm"
    ].to_numpy()
    cluster_rows["source_excess_vs_erm"] = paired["source_excess_vs_erm"].to_numpy()
    cluster_rows.to_csv(output_dir / "discovered_clusters.csv", index=False)

    summary = (
        cluster_rows.groupby("cluster")
        .agg(
            n_runs=("cluster", "size"),
            mean_target_signed_gap=("target_signed_gap_vs_erm", "mean"),
            std_target_signed_gap=("target_signed_gap_vs_erm", "std"),
            mean_positive_degradation=("target_positive_degradation_vs_erm", "mean"),
            mean_source_excess=("source_excess_vs_erm", "mean"),
            methods=("method", lambda values: ", ".join(sorted(set(values)))),
        )
        .reset_index()
    )
    summary.to_csv(output_dir / "discovered_cluster_summary.csv", index=False)

    profile_frame = pd.DataFrame(
        pca_result["standardized"], columns=full["loadings"]["feature"]
    )
    profile_frame["cluster"] = full["cluster"]["labels"]
    profiles = (
        profile_frame.groupby("cluster")
        .mean(numeric_only=True)
        .reset_index()
        .melt(id_vars="cluster", var_name="feature", value_name="mean_standardized_delta")
    )
    profiles.to_csv(output_dir / "discovered_cluster_profiles.csv", index=False)

    correlations = _external_correlations(scores)
    correlations.to_csv(output_dir / "discovered_factor_external_correlations.csv", index=False)

    stability = _stability_analysis(paired, full, full_features)
    stability.to_csv(output_dir / "discovered_stability.csv", index=False)

    labels = _factor_labels(loadings)
    report = _render_report(
        trajectory_path,
        paired,
        pca_result,
        full,
        loadings,
        summary,
        correlations,
        stability,
        profiles,
        labels,
    )
    (output_dir / "discovered_factor_report.md").write_text(report, encoding="utf-8")
    return {
        "n_runs": len(paired),
        "selected_k": int(full["cluster"]["selected_k"]),
        "explained_variance": pca_result["explained_variance_ratio"],
        "stability": stability,
    }


def _render_report(
    trajectory_path: Path,
    paired: pd.DataFrame,
    pca_result: dict[str, np.ndarray],
    full: dict[str, object],
    loadings: pd.DataFrame,
    summary: pd.DataFrame,
    correlations: pd.DataFrame,
    stability: pd.DataFrame,
    profiles: pd.DataFrame,
    labels: dict[str, str],
) -> str:
    explained = ", ".join(
        f"PC{i + 1}={value:.3f}" for i, value in enumerate(pca_result["explained_variance_ratio"])
    )
    lines = [
        "# Data-driven latent response discovery",
        "",
        "> Status: exploratory discovery; the factors are not an exact error decomposition and their names are descriptive only.",
        "",
        "## Protocol",
        "",
        f"- Input: `{trajectory_path}`; final checkpoint per `(method, lambda, seed)`.",
        f"- Runs: `{len(paired)}` non-ERM runs, each paired to the ERM row with the same seed.",
        "- Discovery matrix: ERM-paired deltas of source risk, probes, latent geometry, rank/variance, encoder block norms, and source head norm.",
        "- Excluded from discovery: target errors, target oracle quantities, target gaps, regularizer values, and algorithm labels.",
        "- External validation only: target signed gap and positive degradation are joined after PCA/clustering.",
        "",
        "## PCA",
        "",
        f"Explained variance: `{explained}`.",
        "",
        "Candidate labels are generated from the largest absolute loadings, not assigned before fitting:",
        "",
    ]
    for factor, label in labels.items():
        lines.append(f"- `{factor}`: `{label}`")
    lines += [
        "",
        "These labels describe response directions. They must not be read as proof of information loss, invariance, compression, or shortcut suppression.",
        "",
        "## Cluster selection",
        "",
        f"Ward clustering uses the standardized discovery matrix. Selected `k={full['cluster']['selected_k']}` by mean silhouette.",
        "",
        "| k | silhouette | selected |",
        "|---:|---:|:---:|",
    ]
    for row in full["cluster"]["scores"].itertuples(index=False):
        lines.append(f"| {int(row.k)} | {row.silhouette:.3f} | {'yes' if int(row.k) == int(full['cluster']['selected_k']) else 'no'} |")
    lines += ["", "Cluster summaries (target gap is external and descriptive):", "", _markdown_table(summary), ""]
    profile_preview = (
        profiles.assign(
            absolute_mean_standardized_delta=profiles["mean_standardized_delta"].abs()
        )
        .sort_values(
            ["cluster", "absolute_mean_standardized_delta"], ascending=[True, False]
        )
        .groupby("cluster")
        .head(4)
        .drop(columns="absolute_mean_standardized_delta")
    )
    lines += [
        "The following cluster profiles are the mean standardized ERM-paired deltas; they are the primary mechanism description, not target-derived labels:",
        "",
        _markdown_table(profile_preview),
        "",
    ]
    lines += [
        "## External outcome association",
        "",
        "Spearman values below are exploratory and are not independent-sample significance claims because runs share data, seeds, and training trajectories.",
        "",
        _markdown_table(correlations),
        "",
        "## Stability",
        "",
        "ARI compares cluster assignments to the full feature view. Factor score correlation matches PCA columns by absolute correlation, accounting for sign and ordering ambiguity.",
        "",
        _markdown_table(stability),
        "",
        "## Interpretation boundary",
        "",
        "The result supports a data-driven candidate representation of the observed response surface. A theory crosswalk may be performed only after this step, using the factor loadings and failure cases as evidence. Existing Bayes-projection, transport, nested-oracle, latent-subspace, and approximation/estimation/optimization/shift decompositions are interpretation candidates, not inputs to this discovery run.",
        "",
        "The current experiment is not enough to claim that a factor controls target degradation. The next falsifiable step is to repeat the same discovery protocol across held-out source domains and shift regimes, then test whether the same response directions and cluster outcomes recur.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = write_outputs(args.trajectory, args.output_dir)
    explained = ", ".join(f"{value:.3f}" for value in result["explained_variance"])
    print(
        f"discovered {result['n_runs']} runs; selected k={result['selected_k']}; "
        f"PCA explained variance={explained}"
    )


if __name__ == "__main__":
    main()
