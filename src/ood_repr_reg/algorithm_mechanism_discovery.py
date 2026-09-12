"""Data-first discovery of local-geometry mechanism families.

This module deliberately keeps discovery features separate from target-domain
outcomes.  Target accuracy is loaded only by the external predictive and pair
audits, never by feature construction, scaling, or clustering.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, silhouette_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


FULL_METHODS = ("ERM", "IRMv1", "VREX", "FISHR")
REP_METHODS = ("IRMv1", "Full-BIRM (Official)", "LoRA-BIRM (Official)")
META_FULL = {"method", "seed", "target_acc", "target_loss"}


@dataclass(frozen=True)
class DiscoveryTables:
    full_features: pd.DataFrame
    rep_features: pd.DataFrame
    clusters: pd.DataFrame
    stability: dict
    prediction_audit: pd.DataFrame
    pairs: pd.DataFrame


def _entropy(values: np.ndarray) -> float:
    x = np.clip(np.asarray(values, dtype=float), 0.0, None)
    total = float(x.sum())
    if total <= 0:
        return 0.0
    p = x / total
    p = p[p > 0]
    return float(-(p * np.log(p)).sum() / max(np.log(len(p)), 1e-12))


def _safe_ratio(a: float, b: float) -> float:
    return float(a / max(abs(b), 1e-12))


def build_fullnetwork_features(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build one source-side feature row per (method, seed).

    The first return value contains only discovery features and metadata.  The
    second return value contains target outcomes for post-hoc audits.
    """
    base = root / "round3_redesign" / "task3_aopi_four_methods_rerun" / "results"
    perf = pd.read_csv(base / "method_performance.csv")
    nr = pd.read_csv(base / "normalized_response.csv")
    pi = pd.read_csv(base / "pi_full.csv")
    geom_a = pd.read_csv(base / "geometry_A.csv")
    geom_o = pd.read_csv(base / "geometry_O.csv")

    rows: list[dict[str, float | int | str]] = []
    outcomes: list[dict[str, float | int | str]] = []
    for (method, seed), group in perf.groupby(["method", "seed"], sort=True):
        if str(method) not in FULL_METHODS:
            continue
        method = str(method); seed = int(seed)
        g = nr[(nr.method == method) & (nr.seed == seed) & (nr.K == 20)]
        p = pi[(pi.method == method) & (pi.seed == seed) & (pi.K == 20)]
        a = geom_a[(geom_a.method == method) & (geom_a.seed == seed)]
        o = geom_o[(geom_o.method == method) & (geom_o.seed == seed)]
        source = g["normalized_source_response"].to_numpy(float)
        cf = g["normalized_counterfactual_response"].to_numpy(float)
        clean = g["normalized_clean_task_response"].to_numpy(float)
        weights = np.abs(source)
        rows.append({
            "method": method, "seed": seed,
            "response_source_scale": float(np.linalg.norm(source)),
            "response_cf_scale": float(np.linalg.norm(cf)),
            "response_clean_scale": float(np.linalg.norm(clean)),
            "response_cf_source_ratio": _safe_ratio(np.linalg.norm(cf), np.linalg.norm(source)),
            "response_clean_source_ratio": _safe_ratio(np.linalg.norm(clean), np.linalg.norm(source)),
            "direction_entropy": _entropy(weights),
            "direction_top1_mass": float(np.max(weights) / max(weights.sum(), 1e-12)),
            "direction_top3_mass": float(np.sort(weights)[-3:].sum() / max(weights.sum(), 1e-12)),
            "direction_effective_rank": float(np.exp(_entropy(weights) * np.log(max(len(weights), 1)))),
            "pi_source_bank_scale": float(p["source_bank_response_norm"].mean()) if len(p) else np.nan,
            "pi_cf_bank_scale": float(p["counterfactual_bank_response_norm"].mean()) if len(p) else np.nan,
            "pi_cf_source_ratio": _safe_ratio(float(p["counterfactual_bank_response_norm"].mean()), float(p["source_bank_response_norm"].mean())) if len(p) else np.nan,
            "A_normalized_scale": float(a["A_normalized_norm"].mean()),
            "O_normalized_scale": float(o["O_normalized_norm"].mean()),
            "A_O_scale_ratio": _safe_ratio(float(a["A_normalized_norm"].mean()), float(o["O_normalized_norm"].mean())),
            "A_rank": float(a["A_rank_primary"].mean()),
            "O_rank": float(o["O_rank_primary"].mean()),
        })
        outcomes.append({
            "method": method, "seed": seed,
            "source_mean_acc": float(group["source_mean_acc"].iloc[0]),
            "target_acc": float(group["target_acc"].iloc[0]),
            "target_loss": float(group["target_loss"].iloc[0]),
        })
    return pd.DataFrame(rows), pd.DataFrame(outcomes)


def build_rephead_features(root: Path) -> pd.DataFrame:
    """Build source-side representation/head features at checkpoint 501."""
    base = root / "round3_redesign" / "birm_cmnist_checkpoint_rerun"
    d = pd.read_csv(base / "checkpoint_geometry.csv")
    d = d[d.method.isin(REP_METHODS)].copy()
    d = d.sort_values(["method", "seed", "step"])
    keep = [
        "color_delta_norm", "conditional_color_delta_norm", "spurious_delta_norm",
        "between_trace", "env_within_trace", "trace_ratio", "within_effective_rank",
        "global_effective_rank", "global_stable_rank", "head_color_delta_cos2",
        "head_env_subspace_cos2", "head_interaction_subspace_cos2", "head_spurious_delta_cos2",
        "interaction_to_between_ratio", "interaction_to_within_ratio",
        "spurious_to_label_norm_ratio", "label_color_cos2", "label_env_interaction_trace",
        "global_eig_top1_ratio", "global_eig_top5_ratio", "within_eig_top1_ratio",
        "within_eig_top5_ratio",
    ]
    rows = []
    for (method, seed), g in d.groupby(["method", "seed"], sort=True):
        final = g[g.step == g.step.max()].iloc[0]
        previous = g[g.step < final.step].iloc[-1] if (g.step < final.step).any() else None
        row: dict[str, float | int | str] = {"method": str(method), "seed": int(seed), "checkpoint_step": int(final.step)}
        for col in keep:
            value = float(final[col]) if pd.notna(final[col]) else np.nan
            row[col] = value
            if previous is not None:
                prev = float(previous[col]) if pd.notna(previous[col]) else np.nan
                row[f"delta_{col}"] = value - prev if np.isfinite(value) and np.isfinite(prev) else np.nan
        rows.append(row)
    out = pd.DataFrame(rows)
    numeric = [c for c in out.columns if c not in {"method", "seed", "checkpoint_step"}]
    out[numeric] = out[numeric].replace([np.inf, -np.inf], np.nan)
    return out


def _feature_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if c not in {"method", "seed", "checkpoint_step"} and pd.api.types.is_numeric_dtype(frame[c])]


def discover_numeric_families(frame: pd.DataFrame, *, prefix: str) -> tuple[pd.DataFrame, dict]:
    """Cluster source-side features and bootstrap assignment stability."""
    out = frame.copy()
    cols = _feature_columns(out)
    x = out[cols].replace([np.inf, -np.inf], np.nan).fillna(out[cols].median(numeric_only=True)).fillna(0.0).to_numpy(float)
    x = StandardScaler().fit_transform(x)
    candidate_scores = []
    models = {}
    for k in range(2, min(4, len(out) - 1) + 1):
        labels = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(x)
        score = float(silhouette_score(x, labels)) if len(set(labels)) > 1 else -1.0
        candidate_scores.append({"k": k, "silhouette": score})
        models[k] = labels
    best = max(candidate_scores, key=lambda z: (z["silhouette"], -z["k"])) if candidate_scores else {"k": 1, "silhouette": 0.0}
    k = int(best["k"])
    out[f"{prefix}_cluster"] = models.get(k, np.zeros(len(out), dtype=int))

    rng = np.random.default_rng(8128)
    ari_values = []
    reference = out[f"{prefix}_cluster"].to_numpy(int)
    for _ in range(200):
        mask = rng.random(len(out)) > 0.1
        if mask.sum() < max(k + 1, 4):
            continue
        boot = KMeans(n_clusters=k, n_init=10, random_state=int(rng.integers(1_000_000))).fit_predict(x[mask])
        # Compare pairwise co-membership, invariant to label permutation.
        ref_pairs = reference[mask][:, None] == reference[mask][None, :]
        boot_pairs = boot[:, None] == boot[None, :]
        ari_values.append(float((ref_pairs == boot_pairs).mean()))
    stability = {
        "prefix": prefix, "feature_columns": cols, "selected_k": k,
        "candidate_scores": candidate_scores,
        "bootstrap_pair_agreement_mean": float(np.mean(ari_values)) if ari_values else 0.0,
        "bootstrap_pair_agreement_std": float(np.std(ari_values)) if ari_values else 0.0,
        "n_rows": int(len(out)),
    }
    return out, stability


def evaluate_help_hurt_prediction(features: pd.DataFrame, outcomes: pd.DataFrame) -> pd.DataFrame:
    """External audit; target enters only here to define post-hoc labels."""
    d = features.merge(outcomes, on=["method", "seed"], how="inner")
    erm = d[d.method == "ERM"].set_index("seed")["target_acc"]
    d["target_delta_vs_erm"] = [float(row.target_acc - erm.get(row.seed, np.nan)) for row in d.itertuples()]
    d["outcome_label"] = np.where(d.target_delta_vs_erm > 0.05, "help", np.where(d.target_delta_vs_erm < -0.05, "hurt", "neutral"))
    rows = []
    feat_cols = _feature_columns(features)
    for name, cols in {
        "method_identity": [],
        "source_risk": ["source_mean_acc"],
        "numeric_geometry": feat_cols,
        "identity_plus_geometry": feat_cols,
    }.items():
        preds = []; truth = []; fold_rows = []
        for seed in sorted(d.seed.unique()):
            train = d[d.seed != seed]; test = d[d.seed == seed]
            if name == "method_identity":
                enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
                xtr = enc.fit_transform(train[["method"]]); xte = enc.transform(test[["method"]])
            elif name == "identity_plus_geometry":
                enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
                ident_tr = enc.fit_transform(train[["method"]]); ident_te = enc.transform(test[["method"]])
                scaler = StandardScaler(); geom_tr = scaler.fit_transform(train[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)); geom_te = scaler.transform(test[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0))
                xtr = np.hstack([ident_tr, geom_tr]); xte = np.hstack([ident_te, geom_te])
            else:
                use = cols
                scaler = StandardScaler(); xtr = scaler.fit_transform(train[use].replace([np.inf, -np.inf], np.nan).fillna(0.0)); xte = scaler.transform(test[use].replace([np.inf, -np.inf], np.nan).fillna(0.0))
            clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=0)
            clf.fit(xtr, train.outcome_label)
            pred = clf.predict(xte)
            preds.extend(pred); truth.extend(test.outcome_label)
            fold_rows.append({"seed": int(seed), "n_test": int(len(test)), "test_accuracy": float((pred == test.outcome_label).mean())})
        rows.append({"model": name, "n": len(truth), "accuracy": float(np.mean(np.asarray(preds) == np.asarray(truth))), "balanced_accuracy": float(balanced_accuracy_score(truth, preds)), "folds": json.dumps(fold_rows, sort_keys=True)})
    return pd.DataFrame(rows)


def find_counterexample_pairs(features: pd.DataFrame, outcomes: pd.DataFrame) -> pd.DataFrame:
    d = features.merge(outcomes, on=["method", "seed"], how="inner")
    cols = _feature_columns(features)
    x = StandardScaler().fit_transform(d[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0))
    rows = []
    for i in range(len(d)):
        for j in range(i + 1, len(d)):
            same_method = d.iloc[i].method == d.iloc[j].method
            dist = float(np.linalg.norm(x[i] - x[j]))
            delta = abs(float(d.iloc[i].target_acc - d.iloc[j].target_acc))
            if same_method and dist < np.quantile([np.linalg.norm(x[a] - x[b]) for a in range(len(d)) for b in range(a + 1, len(d))], 0.25) and delta > 0.01:
                kind = "same_method_close_geometry_target_gap"
            elif (not same_method) and delta < 0.02 and dist > np.quantile([np.linalg.norm(x[a] - x[b]) for a in range(len(d)) for b in range(a + 1, len(d))], 0.75):
                kind = "different_method_close_target_geometry_gap"
            else:
                continue
            rows.append({"pair_type": kind, "method_a": d.iloc[i].method, "seed_a": int(d.iloc[i].seed), "target_a": float(d.iloc[i].target_acc), "method_b": d.iloc[j].method, "seed_b": int(d.iloc[j].seed), "target_b": float(d.iloc[j].target_acc), "feature_distance": dist, "target_gap": delta})
    return pd.DataFrame(rows)

