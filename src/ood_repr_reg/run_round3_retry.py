"""Execute the third-round empirical retry and write an auditable report."""

from __future__ import annotations

import csv
import dataclasses
import json
from pathlib import Path

import numpy as np

from .round3_response_matrix import build_response_matrices, source_only_metadata
from .round3_retry_discovery import bootstrap_column_stability, discover_shift_columns, row_diagnostics
from .round3_retry_validation import hidden_composition_pairs, intervention_validation, mechanism_enrichment
from .round3_shift_ensemble import base_environment, nonlinear_environment, shift_ensemble, source_environments
from .round3_trained_models import TrainedModel, train_models


def _jsonable(value: object) -> object:
    if dataclasses.is_dataclass(value):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _model_summary(models: tuple[TrainedModel, ...]) -> list[dict[str, object]]:
    return [
        {
            "model_id": model.model_id,
            "method": model.method,
            "seed": model.seed,
            "lambda": model.lambda_,
            "source_risk": model.source_risk,
            "penalty": model.penalty,
            "optimization_status": model.optimization_status,
            "representation_dim": model.representation_dim,
            "weights": model.weights,
        }
        for model in models
    ]


def _pairwise_agreement(first: object, second: object) -> float:
    a = np.asarray(first, dtype=int)
    b = np.asarray(second, dtype=int)
    return float(np.mean((a[:, None] == a[None, :]) == (b[:, None] == b[None, :])))


def _method_response_summary(models: tuple[TrainedModel, ...], raw: np.ndarray, probes: tuple) -> dict[str, object]:
    kinds = sorted({probe.kind for probe in probes})
    summary: dict[str, object] = {}
    for method in sorted({model.method for model in models}):
        indices = [index for index, model in enumerate(models) if model.method == method]
        summary[method] = {
            "n_models": len(indices),
            "mean_source_risk": float(np.mean([models[index].source_risk for index in indices])),
            "mean_abs_response": float(np.mean(np.abs(raw[indices]))),
            "mean_abs_response_by_kind": {
                kind: float(np.mean(np.abs(raw[indices][:, [i for i, probe in enumerate(probes) if probe.kind == kind]])))
                for kind in kinds
            },
            "mean_penalty": float(np.mean([models[index].penalty for index in indices])),
        }
    erm = summary.get("ERM")
    if isinstance(erm, dict):
        for method, values in summary.items():
            if method == "ERM":
                continue
            values["mean_abs_response_ratio_to_erm"] = values["mean_abs_response"] / max(erm["mean_abs_response"], 1e-30)
            values["response_ratio_by_kind_to_erm"] = {
                kind: values["mean_abs_response_by_kind"][kind] / max(erm["mean_abs_response_by_kind"][kind], 1e-30)
                for kind in kinds
            }
    return summary


def _write_matrix_csv(path: Path, matrix: np.ndarray, ids: tuple[str, ...]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("id", *[f"model_{index}" for index in range(matrix.shape[1])]))
        rows = matrix.T if matrix.shape[0] != len(ids) else matrix
        writer.writerows((identifier, *row.tolist()) for identifier, row in zip(ids, rows))


def _write_figures(output: Path, result: dict[str, object]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - figures are optional research artifacts
        return
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    raw = result["discovery"]["raw"]
    singular = np.asarray(raw["singular_values"], dtype=float)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.arange(1, len(singular) + 1), singular, marker="o", linewidth=1.2)
    ax.set(xlabel="singular-value index", ylabel="singular value", title="Shift-response spectrum")
    fig.tight_layout()
    fig.savefig(figures / "shift_response_spectrum.png", dpi=160)
    plt.close(fig)

    matrix = np.asarray(result["response"]["raw"], dtype=float)
    labels = np.asarray(raw["kmeans_labels"], dtype=int)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(np.arange(matrix.shape[1]), np.linalg.norm(matrix, axis=0), c=labels, s=9, cmap="tab10")
    ax.set(xlabel="shift index", ylabel="model-response norm", title="Shift-column response magnitude")
    fig.tight_layout()
    fig.savefig(figures / "shift_column_response_magnitude.png", dpi=160)
    plt.close(fig)


def run(
    seed_count: int = 20,
    shift_pair_count: int = 150,
    lambdas: tuple[float, ...] | None = None,
    run_stress: bool = True,
) -> dict[str, object]:
    base = base_environment()
    sources = source_environments(base)
    strengths = lambdas or tuple(np.logspace(-3, 1, 8))
    models = train_models(sources, seeds=range(seed_count), lambdas=strengths)
    probes = shift_ensemble(base, count=shift_pair_count)
    matrices = build_response_matrices(models, base, probes)

    raw_discovery = discover_shift_columns(matrices, "raw")
    model_norm_discovery = discover_shift_columns(matrices, "normalized_by_model")
    shift_norm_discovery = discover_shift_columns(matrices, "normalized_by_shift")
    primary_labels = np.asarray(raw_discovery["kmeans_labels"], dtype=int)
    discovery = {
        "raw": raw_discovery,
        "normalized_by_model": model_norm_discovery,
        "normalized_by_shift": shift_norm_discovery,
        "primary_labels": primary_labels,
        "bootstrap_stability_raw": bootstrap_column_stability(matrices, "raw"),
        "bootstrap_stability_model_normalized": bootstrap_column_stability(matrices, "normalized_by_model"),
        "row_diagnostic": row_diagnostics(matrices, "raw"),
    }
    validation = {
        "mechanism_enrichment": mechanism_enrichment(matrices, primary_labels),
        "intervention": intervention_validation(matrices, primary_labels),
        "hidden_composition_pairs": hidden_composition_pairs(matrices),
    }
    validation["cross_normalization_pairwise_agreement"] = {
        "raw_vs_model_normalized": _pairwise_agreement(raw_discovery["kmeans_labels"], model_norm_discovery["kmeans_labels"]),
        "raw_vs_shift_normalized": _pairwise_agreement(raw_discovery["kmeans_labels"], shift_norm_discovery["kmeans_labels"]),
    }

    stress = None
    if run_stress:
        nonlinear_base = nonlinear_environment(base)
        nonlinear_probes = shift_ensemble(nonlinear_base, count=min(30, shift_pair_count), seed=20260907)
        stress_models = models[: min(40, len(models))]
        stress_matrices = build_response_matrices(stress_models, nonlinear_base, nonlinear_probes)
        stress_discovery = discover_shift_columns(stress_matrices, "raw")
        stress = {
            "family": nonlinear_base.family,
            "n_models": len(stress_models),
            "n_shifts": len(nonlinear_probes),
            "effective_rank": stress_discovery["effective_rank"],
            "selected_k": stress_discovery["selected_k"],
            "silhouette": stress_discovery["silhouette"],
        }

    result = {
        "status": "retry-complete",
        "verdict": "PENDING-POSTHOC-INTERPRETATION",
        "experiment": {
            "core_dim": base.c_dim,
            "nuisance_dim": base.a_dim,
            "n_sources": len(sources),
            "n_models": len(models),
            "n_methods": 5,
            "seed_count": seed_count,
            "lambda_count": len(strengths),
            "n_shift_pairs": shift_pair_count,
            "n_shifts": len(probes),
            "training_uses_target": False,
            "discovery_uses_mechanism_labels": False,
        },
        "source_only": source_only_metadata(models),
        "models": _model_summary(models),
        "method_response_summary": _method_response_summary(models, matrices.raw, probes),
        "response": {
            "shape": list(matrices.raw.shape),
            "model_ids": matrices.model_ids,
            "shift_ids": matrices.shift_ids,
            "raw": matrices.raw,
            "normalized_by_model": matrices.normalized_by_model,
            "normalized_by_shift": matrices.normalized_by_shift,
            "first_order": matrices.first_order,
            "second_order": matrices.second_order,
            "shift_metadata": [dataclasses.asdict(probe) for probe in matrices.probes],
        },
        "discovery": discovery,
        "validation": validation,
        "stress_test": stress,
    }
    return _jsonable(result)


def _report(result: dict[str, object]) -> str:
    exp = result["experiment"]
    disc = result["discovery"]
    raw = disc["raw"]
    validation = result["validation"]
    stability = disc["bootstrap_stability_raw"]
    purity = validation["mechanism_enrichment"]["weighted_cluster_purity"]
    normalized_stability = disc["bootstrap_stability_model_normalized"]
    if (
        stability >= 0.80
        and normalized_stability >= 0.80
        and raw["silhouette"] >= 0.20
        and purity >= 0.70
        and disc["normalized_by_model"]["silhouette"] >= 0.20
    ):
        verdict = "NEW-DECOMPOSITION-CANDIDATE"
    elif stability >= 0.80 and raw["effective_rank"] < min(result["response"]["shape"]) and purity < 0.70:
        verdict = "LOW-RANK-BUT-NONSEMANTIC"
    elif disc["row_diagnostic"]["effective_rank"] > raw["effective_rank"]:
        verdict = "MODEL-PHENOTYPES-ONLY"
    else:
        verdict = "EMPIRICAL-DISCOVERY-NOT-YET-INFORMATIVE"
    result["verdict"] = verdict
    result["validation"]["mechanism_enrichment"]["reported_weighted_purity"] = purity
    method_summary = result["method_response_summary"]
    method_lines = []
    for method, values in method_summary.items():
        method_lines.append(f"| {method} | {values['mean_source_risk']:.5f} | {values['mean_abs_response']:.5f} | {values['mean_abs_response_by_kind']} |")
    norm_agreement = validation["cross_normalization_pairwise_agreement"]
    return f"""# 第三轮补充实验报告：风险响应结构重做

## 裁决：`{verdict}`

本报告是对旧第三轮 empirical discovery 的补充重做。旧结果保留为历史记录，不再被视为机制分解证据。

## 设计核验

- 真实训练模型：`{exp['n_models']}` 个；方法：`{exp['n_methods']}`；每种方法 seed：`{exp['seed_count']}`；正则强度：`{exp['lambda_count']}`。
- source environments：`{exp['n_sources']}`；shift probes：`{exp['n_shifts']}`。
- response matrix：`{result['response']['shape']}`，主要分析对象是 shift columns。
- discovery 使用 target 风险：`{exp['training_uses_target']}`；使用机制标签：`{exp['discovery_uses_mechanism_labels']}`。

## Blind discovery

| 输入 | effective rank | selected k | silhouette | bootstrap stability |
|---|---:|---:|---:|---:|
| raw | {raw['effective_rank']} | {raw['selected_k']} | {raw['silhouette']:.4f} | {stability:.4f} |
| model-normalized | {disc['normalized_by_model']['effective_rank']} | {disc['normalized_by_model']['selected_k']} | {disc['normalized_by_model']['silhouette']:.4f} | {disc['bootstrap_stability_model_normalized']:.4f} |
| shift-normalized | {disc['normalized_by_shift']['effective_rank']} | {disc['normalized_by_shift']['selected_k']} | {disc['normalized_by_shift']['silhouette']:.4f} | not primary |

`effective rank` 只说明响应矩阵的数值维数，不自动说明存在机制语义。

## Post-hoc validation

机制 enrichment 和 counterfactual validation 只在 blind discovery 完成后运行。加权 cluster purity 为 `{purity:.4f}`，pair cluster consistency 为 `{validation['intervention']['pair_cluster_consistency']:.4f}`。这些数值都不是机制识别定理，必须与跨 seed、跨方法稳定性和 hidden-composition 检验一起解释。

## 正则与风险响应

| method | mean source risk | mean absolute response | response by shift kind |
|---|---:|---:|---|
{chr(10).join(method_lines)}

相同 shift-side clustering 在 raw 与 model-normalized 输入上的 pairwise 一致率为 `{norm_agreement['raw_vs_model_normalized']:.4f}`，raw 与 shift-normalized 的一致率为 `{norm_agreement['raw_vs_shift_normalized']:.4f}`。因此 raw 中的稳定簇不能直接解释为机制簇；需要先排除响应尺度、正则强度和模型表型因素。

## 当前结论

本次实验修复了旧实验的四个结构性问题：模型由目标函数训练、使用二维 core/nuisance、以 shift columns 为发现对象、并扩大了模型和 shift 数量。当前裁决是 `{verdict}`：稳定性存在，但机制 enrichment、归一化一致性和 cluster 几何不足以支持新的语义分解候选。响应结构更像“风险幅度/模型表型分层”，而不是 core、nuisance、relation 等生成机制的自然分解。

完整机器结果见 `results/round3_retry_results.json`，响应矩阵见 `results/response_*.csv`。
"""


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    output = root / "round3" / "empirical_discovery" / "retry"
    results_dir = output / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    result = run()
    report = _report(result)
    (results_dir / "round3_retry_results.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    response = result["response"]
    for key in ("raw", "normalized_by_model", "normalized_by_shift", "first_order", "second_order"):
        _write_matrix_csv(results_dir / f"response_{key}.csv", np.asarray(response[key]), tuple(response["shift_ids"]))
    _write_figures(output, result)
    (output / "round3_retry_report.md").write_text(report, encoding="utf-8")
    print(output / "round3_retry_report.md")


if __name__ == "__main__":
    main()
