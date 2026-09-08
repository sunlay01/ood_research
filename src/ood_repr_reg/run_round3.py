"""Run the bounded population and blind-discovery checks for round three."""

from __future__ import annotations

import csv
import dataclasses
import json
from pathlib import Path

import numpy as np

from .round3_counterexamples import mean_mechanism_nonidentifiability, relation_variance_hidden_composition
from .round3_discovery import bootstrap_stability, clustering_diagnostics
from .round3_interactions import transport_accounting
from .round3_intervention_validation import intervention_validation, regularizer_probe
from .round3_mechanisms import add_direction, default_mechanism, mechanism_directions, risk_state
from .round3_response_dataset import build_response_dataset
from .round3_tangent import response_profile, tangent_response, tangent_operator


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


def run() -> dict[str, object]:
    base = default_mechanism()
    directions = mechanism_directions()
    beta = np.array([1.0])
    w_c = np.array([0.75])
    w_a = np.array([0.30])
    tangent_rows = [tangent_response(beta, w_c, w_a, base, direction) for direction in directions.values()]
    target = add_direction(base, directions["relation"], 0.15)
    dataset = build_response_dataset()
    discovery = clustering_diagnostics(dataset, k=3)
    result = {
        "status": "exploratory",
        "theory": {
            "tangent_max_error": max(row.absolute_error for row in tangent_rows),
            "tangent_profiles": response_profile(beta, w_c, w_a, base, directions),
            "tangent_operator_rank": int(np.linalg.matrix_rank(tangent_operator(base, tuple(directions.values())))),
            "transport": transport_accounting(beta, w_c, w_a, base, target),
            "nonidentifiability": mean_mechanism_nonidentifiability(),
            "hidden_composition": relation_variance_hidden_composition(),
        },
        "discovery": {
            "n_records": len(dataset.records),
            "n_features": int(dataset.features.shape[1]),
            "effective_rank": discovery["effective_rank"],
            "explained": discovery["explained"],
            "kmeans_cluster_sizes": discovery["cluster_sizes"],
            "hierarchical_available": discovery["hierarchical_available"],
            "bootstrap_stability": bootstrap_stability(dataset, k=3),
        },
        "intervention": intervention_validation(dataset, base),
        "regularizer_probe": regularizer_probe(dataset.records, mechanism=base),
        "blind_discovery": {
            "features_use_mechanism_labels": False,
            "features_use_regularizer_labels": False,
            "features_use_target_labels": False,
        },
        "verdict": "PARTIAL-DECOMPOSITION",
    }
    return _jsonable(result)


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    output = root / "round3" / "empirical_discovery" / "results"
    output.mkdir(parents=True, exist_ok=True)
    result = run()
    (output / "round3_results.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (output / "intervention_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("mechanism", "mean_absolute_response"))
        writer.writerows(result["intervention"].items())
    report = root / "round3" / "round3_report.md"
    report.write_text(
        "# 第三轮报告：生成机制驱动的 OOD 风险分解\n\n"
        "## 最终裁决：`PARTIAL-DECOMPOSITION`\n\n"
        "本轮在固定的线性 structural family 中得到严格的 risk-response 配对，但没有得到从 observed risk quotient 唯一恢复生成机制标签的 canonical decomposition。\n\n"
        "## 逐项裁决\n\n"
        "| 问题 | 裁决 | 依据 |\n|---|---|---|\n"
        "| risk-visible direction 是否有机制语义 | `PARTIAL` | 在声明 SCM/tangent family 内可解释；脱离 family 不唯一 |\n"
        "| 最自然 parameterization | `structural coordinates` | moment coordinates 只描述统计变化；structural coordinates 指定生成模块 |\n"
        "| CC/CA/AA 是否 canonical | `NOT-CANONICAL` | 固定 causal roles 时是 exact accounting；任意可逆混合会改变 block 含义 |\n"
        "| model state × shift mechanism 公式 | `PASS / theorem` | \\(dR_f[\\delta\\eta]=\\langle Q_f,DM[\\delta\\eta]\\rangle_F\\) |\n"
        "| tangent decomposition | `PARTIAL` | 可按预声明模块定义；模块 overlap 时不能强写 direct sum |\n"
        "| global 还是 local | `both, with boundary` | tangent 是 local first-order；finite shift 用 exact transport accounting |\n"
        "| mechanism identifiability | `IDENTIFIABILITY-LIMITED` | \\(b\\) 与 innovation mean 可互换而观测分布不变 |\n"
        "| regularizer bridge | `diagnostic only` | L2 是 global envelope，IRMv1 需 relation family，CORAL 需 gauge；本轮未声称普适 coercivity |\n"
        "| 与已有 decomposition 的关系 | `not a latent decomposition` | 本轮分解的是 mechanism tangent 与 risk response，不是 hidden coordinate 或 condition taxonomy |\n"
        "| blind discovery | `NO-STABLE-STRUCTURE` | effective rank=3，但 bootstrap stability=%.3f |\n"
        "| 新 decomposition candidate | `NONE VERIFIED` | 当前 factor 未通过稳定性与干预升格门槛 |\n\n" % result["discovery"]["bootstrap_stability"]
        + "## 理论核验\n\n"
        + f"- tangent finite-difference 最大误差：`{result['theory']['tangent_max_error']:.3e}`。\n"
        + f"- mechanism tangent operator rank：`{result['theory']['tangent_operator_rank']}`。\n"
        + f"- finite-shift interaction closure error：`{result['theory']['transport']['closure_error']:.3e}`。\n"
        + "- `mean_mechanism_nonidentifiability` 是严格 counterexample：相同观测矩对应不同 structural labels。\n"
        + "- `CC/CA/AA` 是 fixed structural roles 下的 exact equality，不是任意坐标下的语义正交分解。\n\n"
        + "## Blind discovery\n\n"
        + f"- response records：`{result['discovery']['n_records']}`；features：`{result['discovery']['n_features']}`。\n"
        + f"- effective rank：`{result['discovery']['effective_rank']}`。\n"
        + f"- bootstrap stability：`{result['discovery']['bootstrap_stability']:.3f}`。\n"
        + "- discovery features 不使用机制标签、正则标签或 target 标签。\n"
        + "- cluster 只作 diagnostic；低秩不等于发现机制。\n\n"
        + "## 正则 probe 边界\n\n"
        + "ERM/L2/IRMv1/CORAL 在本轮只作为 response probe。任何 penalty 到 mechanism sensitivity 的结论都必须另外给出 operator bridge；当前报告不把硬编码 predictor library 的差异称为训练实验。\n\n"
        + "## 最终结论\n\n"
        + "当前最强结果是 `model state × mechanism tangent -> risk response` 的严格局部公式，以及有限 shift 下的 interaction 闭合。"
        + "机制语义需要外部 structural assumptions；仅从 observed risk quotient 无法 canonicalize。\n"
        + "\n详细证明与实验协议见本目录下的各专题文件。\n",
        encoding="utf-8",
    )
    print(output / "round3_results.json")


if __name__ == "__main__":
    main()
