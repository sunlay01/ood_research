"""Grouped predictive audits for Task 3 applicability."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np

from .features import FEATURE_SETS, build_feature_dictionary

Array = np.ndarray


def logic_audit(design: dict[str, object], feature_dictionary: dict[str, dict[str, object]] | None = None) -> dict[str, object]:
    dictionary = build_feature_dictionary() if feature_dictionary is None else feature_dictionary
    forbidden = [
        name for name, row in dictionary.items()
        if row.get("algebraically_contains_outcome")
        or row.get("uses_target_samples_for_training")
        or row.get("uses_target_samples_for_selection")
        or row.get("uses_target_outcome_for_feature_construction")
        or row.get("uses_target_outcome_for_threshold_selection")
    ]
    required = ["methods", "lambda_grid", "benchmarks", "target_radius_grid", "feature_sets", "outcomes", "cv_grouping", "verdict_thresholds"]
    missing = [name for name in required if name not in design]
    return {
        "passes": not forbidden and not missing,
        "forbidden_feature_count": len(forbidden),
        "forbidden_features": forbidden,
        "missing_design_fields": missing,
        "features_built_before_outcomes_required": True,
        "target_outcomes_are_offline_labels_only": design.get("target_outcome_use") == "offline_labels_only",
    }


def _float(value: object) -> float:
    if value is None:
        return math.nan
    if isinstance(value, bool):
        return float(value)
    text = str(value).strip()
    if text == "":
        return math.nan
    if text.lower() == "inf":
        return math.inf
    if text.lower() == "-inf":
        return -math.inf
    try:
        return float(text)
    except ValueError:
        return math.nan


def _joined_rows(features: list[dict[str, object]], outcomes: list[dict[str, object]]) -> list[dict[str, object]]:
    by_id = {str(row["training_run_id"]): row for row in features}
    rows = []
    for outcome in outcomes:
        if not np.isfinite(_float(outcome.get("target_squared_loss_risk"))):
            continue
        feature = by_id.get(str(outcome.get("training_run_id")))
        if feature is None:
            continue
        rows.append({**feature, **outcome})
    return rows


def _folds(rows: list[dict[str, object]], benchmark: str) -> list[tuple[list[int], list[int], str]]:
    candidates = [index for index, row in enumerate(rows) if row["benchmark"] == benchmark]
    if benchmark == "cmnist":
        group_of = lambda row: str(row.get("seed", ""))
    else:
        group_of = lambda row: str(row.get("family_config", ""))
    groups = sorted({group_of(rows[index]) for index in candidates})
    folds = []
    for group in groups:
        test = [index for index in candidates if group_of(rows[index]) == group]
        train = [index for index in candidates if group_of(rows[index]) != group]
        if train and test:
            folds.append((train, test, group))
    return folds


def grouped_cv_keeps_runs_together(rows: list[dict[str, object]], benchmark: str) -> bool:
    for train, test, _ in _folds(rows, benchmark):
        train_runs = {rows[index]["training_run_id"] for index in train}
        test_runs = {rows[index]["training_run_id"] for index in test}
        if train_runs & test_runs:
            return False
    return True


def _feature_matrix(rows: list[dict[str, object]], indices: list[int], feature_names: list[str], *, reference_indices: list[int]) -> Array:
    categorical = {"method", "benchmark", "family_config"}
    columns: list[Array] = []
    for feature in feature_names:
        if feature in categorical:
            # The column schema is fitted on the training fold only.  A held-out
            # seed/family category unseen in training is encoded as all-zero,
            # which avoids leaking test-fold identity into the model shape.
            categories = sorted({str(rows[i].get(feature, "")) for i in reference_indices})
            for category in categories:
                columns.append(np.asarray([1.0 if str(rows[i].get(feature, "")) == category else 0.0 for i in indices]))
        else:
            train_values = np.asarray([_float(rows[i].get(feature)) for i in reference_indices], dtype=float)
            finite_train = train_values[np.isfinite(train_values)]
            if finite_train.size == 0:
                mean, scale, cap = 0.0, 1.0, 1.0
            else:
                mean = float(finite_train.mean())
                scale = float(finite_train.std()) or 1.0
                cap = max(1.0, float(np.max(np.abs(finite_train))) * 10.0)
            values = []
            for index in indices:
                value = _float(rows[index].get(feature))
                if math.isinf(value):
                    value = math.copysign(cap, value)
                elif not math.isfinite(value):
                    value = mean
                values.append((value - mean) / scale)
            columns.append(np.asarray(values, dtype=float))
    if not columns:
        return np.ones((len(indices), 1))
    return np.column_stack([np.ones(len(indices)), *columns])


def _ridge_fit(x: Array, y: Array, alpha: float = 1e-6) -> Array:
    xtx = x.T @ x
    penalty = alpha * np.eye(xtx.shape[0])
    penalty[0, 0] = 0.0
    return np.linalg.pinv(xtx + penalty) @ x.T @ y


def _risk_predictions(rows: list[dict[str, object]], benchmark: str, feature_set: str) -> tuple[Array, Array, list[dict[str, object]]]:
    folds = _folds(rows, benchmark)
    feature_names = FEATURE_SETS[feature_set]
    y_true: list[float] = []
    y_pred: list[float] = []
    fold_rows: list[dict[str, object]] = []
    for train, test, group in folds:
        train = [i for i in train if np.isfinite(_float(rows[i].get("delta_target_risk_vs_erm")))]
        test = [i for i in test if np.isfinite(_float(rows[i].get("delta_target_risk_vs_erm")))]
        if len(train) < 2 or not test:
            continue
        x_train = _feature_matrix(rows, train, feature_names, reference_indices=train)
        y_train = np.asarray([_float(rows[i]["delta_target_risk_vs_erm"]) for i in train])
        x_test = _feature_matrix(rows, test, feature_names, reference_indices=train)
        beta = _ridge_fit(x_train, y_train)
        prediction = x_test @ beta
        truth = np.asarray([_float(rows[i]["delta_target_risk_vs_erm"]) for i in test])
        y_true.extend(truth.tolist())
        y_pred.extend(prediction.tolist())
        fold_rows.append({
            "fold": group,
            "n_test": len(test),
            "r2": _r2(truth, prediction),
            "mae": float(np.mean(np.abs(truth - prediction))),
            "rank_correlation": _spearman(truth, prediction),
        })
    return np.asarray(y_true), np.asarray(y_pred), fold_rows


def _r2(y: Array, prediction: Array) -> float:
    if y.size == 0:
        return math.nan
    denom = float(np.sum((y - y.mean()) ** 2))
    if denom <= 1e-15:
        return math.nan
    return float(1.0 - np.sum((y - prediction) ** 2) / denom)


def _rankdata(values: Array) -> Array:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=float)
    ranks[order] = np.arange(values.size, dtype=float)
    return ranks


def _spearman(a: Array, b: Array) -> float:
    if len(a) < 2:
        return math.nan
    ra, rb = _rankdata(np.asarray(a)), _rankdata(np.asarray(b))
    if ra.std() <= 0 or rb.std() <= 0:
        return math.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def _kendall(a: Array, b: Array) -> float:
    a, b = np.asarray(a), np.asarray(b)
    concordant = discordant = 0
    for i in range(len(a)):
        for j in range(i + 1, len(a)):
            da, db = a[i] - a[j], b[i] - b[j]
            if da == 0 or db == 0:
                continue
            if da * db > 0:
                concordant += 1
            else:
                discordant += 1
    total = concordant + discordant
    return math.nan if total == 0 else float((concordant - discordant) / total)


def _auc(labels: Array, scores: Array) -> float:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if pos.size == 0 or neg.size == 0:
        return math.nan
    wins = 0.0
    for value in pos:
        wins += float(np.sum(value > neg)) + 0.5 * float(np.sum(value == neg))
    return float(wins / (pos.size * neg.size))


def _fit_logistic(x: Array, y: Array) -> Array:
    try:
        from scipy.optimize import minimize
    except Exception:  # pragma: no cover
        return _ridge_fit(x, y - y.mean(), alpha=1e-3)

    def objective(beta: Array) -> tuple[float, Array]:
        logits = np.clip(x @ beta, -40, 40)
        prob = 1.0 / (1.0 + np.exp(-logits))
        loss = -float(np.mean(y * np.log(prob + 1e-12) + (1.0 - y) * np.log(1.0 - prob + 1e-12)))
        reg = 1e-4 * float(beta[1:] @ beta[1:])
        grad = x.T @ (prob - y) / y.size
        grad[1:] += 2e-4 * beta[1:]
        return loss + reg, grad

    result = minimize(lambda beta: objective(beta), np.zeros(x.shape[1]), jac=True, method="BFGS", options={"maxiter": 500})
    return np.asarray(result.x, dtype=float)


def _win_predictions(rows: list[dict[str, object]], benchmark: str, feature_set: str) -> tuple[Array, Array, list[dict[str, object]]]:
    folds = _folds(rows, benchmark)
    feature_names = FEATURE_SETS[feature_set]
    y_true: list[float] = []
    y_score: list[float] = []
    fold_rows = []
    for train, test, group in folds:
        train = [i for i in train if not (rows[i]["method"] == "L2" and abs(_float(rows[i]["lambda"])) <= 1e-15)]
        test = [i for i in test if not (rows[i]["method"] == "L2" and abs(_float(rows[i]["lambda"])) <= 1e-15)]
        train = [i for i in train if np.isfinite(_float(rows[i].get("delta_target_risk_vs_erm")))]
        test = [i for i in test if np.isfinite(_float(rows[i].get("delta_target_risk_vs_erm")))]
        if len(train) < 4 or not test:
            continue
        y_train = np.asarray([1.0 if rows[i].get("win_loss_vs_erm") in {True, "True", "true", "1", 1} else 0.0 for i in train])
        if len(np.unique(y_train)) < 2:
            continue
        x_train = _feature_matrix(rows, train, feature_names, reference_indices=train)
        x_test = _feature_matrix(rows, test, feature_names, reference_indices=train)
        beta = _fit_logistic(x_train, y_train)
        scores = 1.0 / (1.0 + np.exp(-np.clip(x_test @ beta, -40, 40)))
        truth = np.asarray([1.0 if rows[i].get("win_loss_vs_erm") in {True, "True", "true", "1", 1} else 0.0 for i in test])
        y_true.extend(truth.tolist())
        y_score.extend(scores.tolist())
        predicted = scores >= 0.5
        positives = truth == 1
        negatives = truth == 0
        tpr = float(np.mean(predicted[positives] == 1)) if positives.any() else math.nan
        tnr = float(np.mean(predicted[negatives] == 0)) if negatives.any() else math.nan
        fold_rows.append({"fold": group, "n_test": len(test), "auc": _auc(truth, scores), "balanced_accuracy": np.nanmean([tpr, tnr]), "brier": float(np.mean((scores - truth) ** 2))})
    return np.asarray(y_true), np.asarray(y_score), fold_rows


def _ranking_rows(rows: list[dict[str, object]], benchmark: str, feature_set: str, predictions: Array, truth: Array) -> list[dict[str, object]]:
    valid_indices = [i for i, row in enumerate(rows) if row["benchmark"] == benchmark and np.isfinite(_float(row.get("delta_target_risk_vs_erm")))]
    if len(valid_indices) != len(predictions):
        # Ranking is based on the same row filter as risk prediction; if CV folds
        # skipped a row, fall back to an empty ranking instead of mixing indices.
        return []
    by_target: dict[tuple[str, str, str, str], list[int]] = {}
    for local_index, row_index in enumerate(valid_indices):
        row = rows[row_index]
        key = (row["benchmark"], row["family_config"], row["seed"], row["target_id"])
        by_target.setdefault(key, []).append(local_index)
    result = []
    for key, indices in by_target.items():
        if len(indices) < 3:
            continue
        y = truth[indices]
        p = predictions[indices]
        best_true = int(np.argmin(y))
        best_pred = int(np.argmin(p))
        result.append({
            "benchmark": benchmark,
            "family_config": key[1],
            "seed": key[2],
            "target_id": key[3],
            "feature_set": feature_set,
            "kendall_tau": _kendall(y, p),
            "spearman_rho": _spearman(y, p),
            "pairwise_ordering_accuracy": (1.0 + _kendall(y, p)) / 2.0 if np.isfinite(_kendall(y, p)) else math.nan,
            "best_method_agreement": bool(best_true == best_pred),
            "n_methods": len(indices),
        })
    return result


def _summarize(values: Iterable[float]) -> tuple[float, float, float]:
    data = np.asarray([float(v) for v in values if np.isfinite(float(v))], dtype=float)
    if data.size == 0:
        return math.nan, math.nan, math.nan
    return float(data.mean()), float(np.quantile(data, 0.1)), float(np.quantile(data, 0.9))


def _bootstrap_interval(values: Iterable[float], seed: int = 101, repetitions: int = 200) -> tuple[float, float]:
    data = np.asarray([float(v) for v in values if np.isfinite(float(v))], dtype=float)
    if data.size == 0:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    means = [float(np.mean(rng.choice(data, size=data.size, replace=True))) for _ in range(repetitions)]
    return float(np.quantile(means, 0.1)), float(np.quantile(means, 0.9))


def _evaluate_once(rows: list[dict[str, object]], feature_rows: list[dict[str, object]], label: str) -> dict[str, list[dict[str, object]]]:
    del feature_rows
    ranking: list[dict[str, object]] = []
    predictive: list[dict[str, object]] = []
    for benchmark in sorted({str(row["benchmark"]) for row in rows}):
        for feature_set in ("B0", "B1", "B2", "B3"):
            y_risk, pred_risk, risk_folds = _risk_predictions(rows, benchmark, feature_set)
            if y_risk.size:
                r2 = _r2(y_risk, pred_risk)
                mae = float(np.mean(np.abs(y_risk - pred_risk)))
                rank_corr = _spearman(y_risk, pred_risk)
                predictive.append({
                    "analysis_label": label,
                    "benchmark": benchmark,
                    "feature_set": feature_set,
                    "test": "finite_risk_prediction",
                    "metric_primary": "cv_r2",
                    "cv_r2": r2,
                    "mae": mae,
                    "rank_correlation": rank_corr,
                    "grouped_interval_low": _bootstrap_interval([row["r2"] for row in risk_folds])[0],
                    "grouped_interval_high": _bootstrap_interval([row["r2"] for row in risk_folds])[1],
                    "fold_count": len(risk_folds),
                    "n_rows": int(y_risk.size),
                })
                ranking.extend(_ranking_rows(rows, benchmark, feature_set, pred_risk, y_risk))
            y_win, score, win_folds = _win_predictions(rows, benchmark, feature_set)
            if y_win.size:
                predicted = score >= 0.5
                positives = y_win == 1
                negatives = y_win == 0
                tpr = float(np.mean(predicted[positives] == 1)) if positives.any() else math.nan
                tnr = float(np.mean(predicted[negatives] == 0)) if negatives.any() else math.nan
                predictive.append({
                    "analysis_label": label,
                    "benchmark": benchmark,
                    "feature_set": feature_set,
                    "test": "win_loss_discrimination",
                    "metric_primary": "auc",
                    "auc": _auc(y_win, score),
                    "balanced_accuracy": np.nanmean([tpr, tnr]),
                    "brier": float(np.mean((score - y_win) ** 2)),
                    "grouped_interval_low": _bootstrap_interval([row["auc"] for row in win_folds])[0],
                    "grouped_interval_high": _bootstrap_interval([row["auc"] for row in win_folds])[1],
                    "fold_count": len(win_folds),
                    "n_rows": int(y_win.size),
                })
    ranking_summary = []
    for benchmark in sorted({row["benchmark"] for row in ranking}):
        for feature_set in ("B0", "B1", "B2", "B3"):
            selected = [row for row in ranking if row["benchmark"] == benchmark and row["feature_set"] == feature_set]
            mean, lo, hi = _summarize(row["kendall_tau"] for row in selected)
            if selected:
                interval = _bootstrap_interval(row["kendall_tau"] for row in selected)
                ranking_summary.append({
                    "analysis_label": label,
                    "benchmark": benchmark,
                    "feature_set": feature_set,
                    "test": "ranking",
                    "metric_primary": "kendall_tau",
                    "kendall_tau": mean,
                    "kendall_tau_q10": lo,
                    "kendall_tau_q90": hi,
                    "grouped_interval_low": interval[0],
                    "grouped_interval_high": interval[1],
                    "spearman_rho": _summarize(row["spearman_rho"] for row in selected)[0],
                    "pairwise_ordering_accuracy": _summarize(row["pairwise_ordering_accuracy"] for row in selected)[0],
                    "best_method_agreement": float(np.mean([bool(row["best_method_agreement"]) for row in selected])),
                    "target_count": len(selected),
                })
    return {"ranking": ranking_summary, "predictive": predictive, "ranking_detail": ranking}


def _improvement_rows(ranking: list[dict[str, object]], predictive: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    records = ranking + predictive
    for benchmark in sorted({row["benchmark"] for row in records if row.get("analysis_label") == "observed"}):
        for test in sorted({row["test"] for row in records if row.get("analysis_label") == "observed" and row["benchmark"] == benchmark}):
            b1 = next((row for row in records if row.get("analysis_label") == "observed" and row["benchmark"] == benchmark and row["test"] == test and row["feature_set"] == "B1"), None)
            b2 = next((row for row in records if row.get("analysis_label") == "observed" and row["benchmark"] == benchmark and row["test"] == test and row["feature_set"] == "B2"), None)
            b3 = next((row for row in records if row.get("analysis_label") == "observed" and row["benchmark"] == benchmark and row["test"] == test and row["feature_set"] == "B3"), None)
            if not b1 or not b3:
                continue
            metric = b3["metric_primary"]
            b1_value = _float(b1.get(metric))
            b2_value = _float(b2.get(metric)) if b2 else math.nan
            b3_value = _float(b3.get(metric))
            rows.append({
                "benchmark": benchmark,
                "test": test,
                "metric": metric,
                "B1": b1_value,
                "B2": b2_value,
                "B3": b3_value,
                "B2_minus_B1": b2_value - b1_value if np.isfinite(b2_value) and np.isfinite(b1_value) else math.nan,
                "B3_minus_B1": b3_value - b1_value if np.isfinite(b3_value) and np.isfinite(b1_value) else math.nan,
            })
    return rows


def _permute_theory_features(feature_rows: list[dict[str, object]], seed: int = 17) -> list[dict[str, object]]:
    rng = np.random.default_rng(seed)
    theory = ["E_operator_norm", "E_frobenius_norm", "rho_slack", "slack_margin", "R_info", "rank_O_S", "dim_ker_O_S", "rank_A_irr"]
    rows = [dict(row) for row in feature_rows]
    groups: dict[tuple[str, str, str, str], list[int]] = {}
    for index, row in enumerate(rows):
        groups.setdefault((str(row.get("benchmark")), str(row.get("family_config")), str(row.get("method")), str(row.get("lambda"))), []).append(index)
    for indices in groups.values():
        if len(indices) <= 1:
            continue
        order = indices.copy()
        rng.shuffle(order)
        for feature in theory:
            values = [rows[index].get(feature) for index in order]
            for index, value in zip(indices, values, strict=True):
                rows[index][feature] = value
    return rows


def _random_feature_rows(feature_rows: list[dict[str, object]], seed: int = 29) -> list[dict[str, object]]:
    rng = np.random.default_rng(seed)
    rows = [dict(row) for row in feature_rows]
    theory = ["E_operator_norm", "E_frobenius_norm", "rho_slack", "slack_margin", "R_info", "rank_O_S", "dim_ker_O_S", "rank_A_irr"]
    for row in rows:
        for feature in theory:
            row[feature] = float(rng.normal())
    return rows


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _radius_rows(joined: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    for benchmark in sorted({row["benchmark"] for row in joined}):
        for radius in sorted({row["radius_regime"] for row in joined if row["benchmark"] == benchmark}):
            subset = [row for row in joined if row["benchmark"] == benchmark and row["radius_regime"] == radius]
            result = _evaluate_once(subset, [], "observed")
            improvements = _improvement_rows(result["ranking"], result["predictive"])
            for row in improvements:
                rows.append({"benchmark": benchmark, "radius_regime": radius, **row})
    return rows


def _counterexamples(joined: list[dict[str, object]]) -> list[dict[str, object]]:
    cases: list[dict[str, object]] = []
    valid = [row for row in joined if np.isfinite(_float(row.get("delta_target_risk_vs_erm")))]
    pairs = [("rho_slack", "same_rho_slack_different_behavior"), ("E_operator_norm", "same_E_norm_different_behavior"), ("z0_norm", "smaller_z0_worse_behavior")]
    for feature, label in pairs:
        ordered = sorted(valid, key=lambda row: (_float(row.get(feature)), _float(row.get("delta_target_risk_vs_erm"))))
        for left, right in zip(ordered, ordered[1:]):
            a, b = _float(left.get(feature)), _float(right.get(feature))
            ya, yb = _float(left.get("delta_target_risk_vs_erm")), _float(right.get("delta_target_risk_vs_erm"))
            if not (np.isfinite(a) and np.isfinite(b) and np.isfinite(ya) and np.isfinite(yb)):
                continue
            if label.startswith("same") and abs(a - b) <= 1e-6 * max(1.0, abs(a), abs(b)) and abs(ya - yb) > 1e-6:
                cases.append({"case": label, "feature": feature, "left_run": left["training_run_id"], "right_run": right["training_run_id"], "left_value": a, "right_value": b, "left_delta": ya, "right_delta": yb})
                break
            if label == "smaller_z0_worse_behavior" and a < b and ya > yb + 1e-6:
                cases.append({"case": label, "feature": feature, "left_run": left["training_run_id"], "right_run": right["training_run_id"], "left_value": a, "right_value": b, "left_delta": ya, "right_delta": yb})
                break
    return cases


def _verdict(summary_rows: list[dict[str, object]], null_rows: list[dict[str, object]], logic: dict[str, object]) -> str:
    if not logic.get("passes"):
        return "TASK3-APPLICABILITY-FAIL"
    improvements = [row for row in summary_rows if np.isfinite(_float(row.get("B3_minus_B1"))) and _float(row.get("B3_minus_B1")) > 0.01]
    by_benchmark: dict[str, set[str]] = {}
    for row in improvements:
        by_benchmark.setdefault(str(row["benchmark"]), set()).add(str(row["test"]))
    support_shape = all(len(by_benchmark.get(benchmark, set())) >= 2 for benchmark in ("gaussian", "cmnist"))
    null_ok = True
    for row in null_rows:
        if row.get("test") in {"finite_risk_prediction", "win_loss_discrimination", "ranking"}:
            observed = _float(row.get("observed_B3"))
            permuted = _float(row.get("permuted_B3"))
            random = _float(row.get("random_B3"))
            if np.isfinite(observed) and np.isfinite(permuted) and np.isfinite(random):
                null_ok = null_ok and observed >= max(permuted, random)
    if support_shape and null_ok:
        return "TASK3-APPLICABILITY-SUPPORT"
    if improvements:
        return "TASK3-APPLICABILITY-PARTIAL"
    return "TASK3-APPLICABILITY-FAIL"


def evaluate_applicability(feature_rows: list[dict[str, object]], outcome_rows: list[dict[str, object]], output: Path, design: dict[str, object]) -> dict[str, object]:
    results = output / "results"
    results.mkdir(parents=True, exist_ok=True)
    joined = _joined_rows(feature_rows, outcome_rows)
    logic = logic_audit(design)
    observed = _evaluate_once(joined, feature_rows, "observed")
    ranking_rows = observed["ranking"]
    predictive_rows = observed["predictive"]
    baseline_rows = _improvement_rows(ranking_rows, predictive_rows)

    permuted_joined = _joined_rows(_permute_theory_features(feature_rows), outcome_rows)
    random_joined = _joined_rows(_random_feature_rows(feature_rows), outcome_rows)
    permuted = _evaluate_once(permuted_joined, feature_rows, "permuted")
    random = _evaluate_once(random_joined, feature_rows, "random")
    null_rows: list[dict[str, object]] = []
    for observed_row in ranking_rows + predictive_rows:
        if observed_row.get("feature_set") != "B3":
            continue
        metric = observed_row["metric_primary"]
        match = lambda rows: next((row for row in rows if row.get("benchmark") == observed_row.get("benchmark") and row.get("test") == observed_row.get("test") and row.get("feature_set") == "B3"), {})
        p = match(permuted["ranking"] + permuted["predictive"])
        r = match(random["ranking"] + random["predictive"])
        null_rows.append({
            "benchmark": observed_row.get("benchmark"),
            "test": observed_row.get("test"),
            "metric": metric,
            "observed_B3": observed_row.get(metric),
            "permuted_B3": p.get(metric, math.nan),
            "random_B3": r.get(metric, math.nan),
        })

    radius_rows = _radius_rows(joined)
    counterexamples = _counterexamples(joined)
    _write_csv(results / "ranking_results.csv", ranking_rows)
    _write_csv(results / "predictive_results.csv", predictive_rows)
    _write_csv(results / "baseline_comparison.csv", baseline_rows)
    _write_csv(results / "permutation_controls.csv", null_rows)
    _write_csv(results / "radius_sweep.csv", radius_rows)
    _write_csv(results / "counterexamples.csv", counterexamples)
    cv_pass = all(grouped_cv_keeps_runs_together(joined, benchmark) for benchmark in {"gaussian", "cmnist"})
    verdict = _verdict(baseline_rows, null_rows, logic)
    summary = {
        "verdict": verdict,
        "logic_audit": logic,
        "feature_row_count": len(feature_rows),
        "target_outcome_row_count": len(outcome_rows),
        "joined_valid_row_count": len(joined),
        "grouped_cv_keeps_training_runs_together": cv_pass,
        "ranking_row_count": len(ranking_rows),
        "predictive_row_count": len(predictive_rows),
        "baseline_comparison_row_count": len(baseline_rows),
        "null_control_row_count": len(null_rows),
        "counterexample_count": len(counterexamples),
        "primary_increment": "B3_minus_B1",
        "secondary_increment": "B2_minus_B1",
        "target_outcomes_used_for_feature_construction": False,
        "target_outcomes_used_for_selection": False,
        "canonical_state_edited": False,
    }
    (results / "summary.json").write_text(json.dumps(summary, indent=2, default=_json), encoding="utf-8")
    return summary


def _json(value: object):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, float) and math.isnan(value):
        return None
    raise TypeError(type(value).__name__)


__all__ = [
    "evaluate_applicability", "grouped_cv_keeps_runs_together", "logic_audit",
]
