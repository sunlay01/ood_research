"""Run the narrowly scoped 3A response-space algebra audit."""

from __future__ import annotations

import csv
import dataclasses
import json
from pathlib import Path

import numpy as np

from .round3_algebra import (
    Array,
    AlgebraTolerance,
    coordinate_nullspace,
    factorization_from_matrices,
    family_incremental_ranks,
    grouped_response_geometry,
    intersection_dimension,
    nullspace_functionals,
    pure_family_liftings,
    quotient_diagnostics,
    rank_with_tolerance,
    right_nullspace,
    risk_statistic_functional,
    risk_statistic_labels,
    structural_reachability_liftings,
)
from .round3_response_matrix import ResponseMatrices
from .round3_shift_ensemble import PopulationEnvironment, ShiftProbe, base_environment
from .round3_trained_models import TrainedModel


def _array(value: object) -> np.ndarray:
    return np.asarray(value, dtype=float)


def _load_saved_retry(root: Path) -> tuple[tuple[TrainedModel, ...], PopulationEnvironment, tuple[ShiftProbe, ...], ResponseMatrices]:
    path = root / "round3" / "empirical_discovery" / "retry" / "results" / "round3_retry_results.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    models = tuple(
        TrainedModel(
            model_id=item["model_id"], method=item["method"], seed=int(item["seed"]),
            lambda_=float(item["lambda"]), weights=_array(item["weights"]),
            source_risk=float(item["source_risk"]), penalty=float(item["penalty"]),
            optimization_status=item["optimization_status"], representation_dim=item.get("representation_dim"),
        )
        for item in payload["models"]
    )
    source = base_environment()
    probes = tuple(
        ShiftProbe(
            shift_id=item["shift_id"], target=PopulationEnvironment(
                mu_c=_array(item["target"]["mu_c"]), sigma_c=_array(item["target"]["sigma_c"]),
                gamma=_array(item["target"]["gamma"]), b=_array(item["target"]["b"]),
                mu_xi=_array(item["target"]["mu_xi"]), sigma_xi=_array(item["target"]["sigma_xi"]),
                beta=_array(item["target"]["beta"]), noise_variance=float(item["target"].get("noise_variance", 0.25)),
                family=item["target"].get("family", "linear-gaussian"),
                quadratic_alpha=_array(item["target"]["quadratic_alpha"]) if item["target"].get("quadratic_alpha") is not None else None,
            ),
            kind=item["kind"], pair_id=item.get("pair_id"), sign=int(item["sign"]),
            magnitude=float(item["magnitude"]), hidden_metadata=item.get("hidden_metadata", {}),
        )
        for item in payload["response"]["shift_metadata"]
    )
    response = ResponseMatrices(
        model_ids=tuple(payload["response"]["model_ids"]), shift_ids=tuple(payload["response"]["shift_ids"]),
        raw=_array(payload["response"]["raw"]), normalized_by_model=_array(payload["response"]["normalized_by_model"]),
        normalized_by_shift=_array(payload["response"]["normalized_by_shift"]), first_order=_array(payload["response"]["first_order"]),
        second_order=_array(payload["response"]["second_order"]), probes=probes,
    )
    return models, source, probes, response


def _jsonable(value: object) -> object:
    if dataclasses.is_dataclass(value):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    return value


def _groups(models: tuple[TrainedModel, ...], probes: tuple[ShiftProbe, ...]) -> tuple[dict[str, tuple[int, ...]], dict[str, tuple[int, ...]]]:
    model_groups: dict[str, list[int]] = {}
    for index, model in enumerate(models):
        model_groups.setdefault(model.method, []).append(index)
    shift_groups: dict[str, list[int]] = {}
    for index, probe in enumerate(probes):
        shift_groups.setdefault(probe.kind, []).append(index)
    return {k: tuple(v) for k, v in model_groups.items()}, {k: tuple(v) for k, v in shift_groups.items()}


def _strip_spans(value: object) -> object:
    if isinstance(value, dict):
        return {k: _strip_spans(v) for k, v in value.items() if k != "spans"}
    if isinstance(value, np.ndarray):
        return value
    return value


def _null_functional_records(null_data: dict[str, object], labels: tuple[str, ...]) -> list[dict[str, object]]:
    records = []
    basis = np.asarray(null_data["basis"], dtype=float)
    for index in range(basis.shape[1]):
        coefficients = {
            labels[column]: float(basis[column, index])
            for column in range(basis.shape[0])
            if abs(basis[column, index]) > 1e-8
        }
        records.append({"basis_index": index, "coefficients": coefficients, "functional": risk_statistic_functional(basis[:, index], 5)})
    return records


def _coordinate_null_records(coordinates: dict[str, object], labels: tuple[str, ...]) -> list[dict[str, object]]:
    basis = np.asarray(coordinates["basis"], dtype=float)
    free = tuple(int(value) for value in coordinates["free_columns"])
    records = []
    for index, free_column in enumerate(free):
        coefficients = {
            labels[column]: float(basis[column, index])
            for column in range(basis.shape[0])
            if abs(basis[column, index]) > 1e-8
        }
        records.append({"free_coordinate": labels[free_column], "coefficients": coefficients, "functional": risk_statistic_functional(basis[:, index], 5)})
    return records


def _subspace_relation(values: dict[str, object]) -> str:
    first = float(values["first_in_second_residual"])
    second = float(values["second_in_first_residual"])
    intersection = int(values["intersection_dimension"])
    if first < 1e-7 and second < 1e-7:
        return "identical"
    if first < 1e-7:
        return "first-contained-in-second"
    if second < 1e-7:
        return "second-contained-in-first"
    if intersection:
        return "partial-overlap"
    return "zero-intersection"


def _coefficient_text(coefficients: dict[str, float], limit: int = 7) -> str:
    ordered = sorted(coefficients.items(), key=lambda item: abs(item[1]), reverse=True)[:limit]
    return "; ".join(f"{name}={value:+.4f}" for name, value in ordered)


def run(root: Path | None = None) -> dict[str, object]:
    root = root or Path(__file__).resolve().parents[2]
    models, source, probes, matrices = _load_saved_retry(root)
    tolerance = AlgebraTolerance()
    factor = factorization_from_matrices(models, source, probes, matrices)
    quotient = quotient_diagnostics(factor["phi"], factor["psi"], factor["response"], tolerance)
    labels = tuple(risk_statistic_labels(5))
    null_data = nullspace_functionals(factor["psi"], 5, tolerance)
    coordinate_null = coordinate_nullspace(factor["psi"], tolerance)
    null_records = _null_functional_records(null_data, labels)
    coordinate_null_records = _coordinate_null_records(coordinate_null, labels)
    full_structural_liftings = structural_reachability_liftings(source, count=512, seed=20260906, vary_task=True)
    fixed_task_liftings = structural_reachability_liftings(source, count=512, seed=20260907, vary_task=False)
    full_structural_null = right_nullspace(full_structural_liftings, tolerance)
    current_null = np.asarray(null_data["basis"], dtype=float)
    model_groups, shift_groups = _groups(models, probes)
    shift_geometry = grouped_response_geometry(factor["response"], shift_groups, "shift", tolerance)
    model_geometry = grouped_response_geometry(factor["response"], model_groups, "model", tolerance)
    shift_order = [(name, shift_groups[name]) for name in sorted(shift_groups)]
    model_order = [(name, model_groups[name]) for name in sorted(model_groups)]
    pure_liftings, pure_metadata = pure_family_liftings(source)
    pure_names = tuple(pure_liftings)
    pure_groups: dict[str, tuple[int, ...]] = {}
    pure_columns: list[Array] = []
    cursor = 0
    for name in pure_names:
        block = pure_liftings[name]
        pure_columns.append(block)
        pure_groups[name] = tuple(range(cursor, cursor + block.shape[1]))
        cursor += block.shape[1]
    pure_matrix = np.column_stack(pure_columns)
    pure_response = factor["phi"] @ pure_matrix
    pure_lift_geometry = grouped_response_geometry(pure_matrix, pure_groups, "shift", tolerance)
    pure_response_geometry = grouped_response_geometry(pure_response, pure_groups, "shift", tolerance)
    pure_order = [(name, pure_groups[name]) for name in pure_names]
    rank_profile = []
    for relative in (1e-12, 1e-10, 1e-9, 1e-8, 1e-6):
        profile_tolerance = AlgebraTolerance(rank_relative=relative, residual_absolute=tolerance.residual_absolute, angle_absolute=tolerance.angle_absolute)
        rank_profile.append({
            "rank_relative": relative,
            "phi_rank": rank_with_tolerance(factor["phi"], profile_tolerance),
            "psi_rank": rank_with_tolerance(factor["psi"], profile_tolerance),
            "response_rank": rank_with_tolerance(factor["response"], profile_tolerance),
        })
    shift_pairs = {name: values for name, values in shift_geometry["pairwise"].items()}
    model_pairs = {name: values for name, values in model_geometry["pairwise"].items()}
    result = {
        "status": "complete",
        "scope": "3A-response-space-algebra-only",
        "data": {"n_models": len(models), "n_shifts": len(probes), "feature_dimension": factor["feature_dimension"]},
        "factorization": {
            "phi_rank": factor["phi_rank"], "psi_rank": factor["psi_rank"], "response_rank": factor["response_rank"],
            "residual_frobenius": factor["residual_frobenius"], "residual_max_absolute": factor["residual_max_absolute"],
            "exact_within_tolerance": factor["residual_max_absolute"] <= tolerance.residual_absolute,
        },
        "quotient": quotient,
        "shift_nullspace": {
            "dimension": int(null_data["dimension"]),
            "coordinate_labels": labels,
            "orthonormal_basis": null_data["basis"],
            "orthonormal_functionals": null_records,
            "coordinate_basis": coordinate_null["basis"],
            "coordinate_free_columns": coordinate_null["free_columns"],
            "coordinate_functionals": coordinate_null_records,
            "max_annihilation_residual": null_data["max_annihilation_residual"],
        },
        "structural_reachability": {
            "all_structural_parameters_vary": {
                "sample_count": int(full_structural_liftings.shape[0]),
                "numerical_rank": rank_with_tolerance(full_structural_liftings, tolerance),
                "null_dimension": int(full_structural_null.shape[1]),
                "null_functionals": _null_functional_records({"basis": full_structural_null}, labels),
            },
            "task_fixed": {
                "sample_count": int(fixed_task_liftings.shape[0]),
                "numerical_rank": rank_with_tolerance(fixed_task_liftings, tolerance),
                "null_dimension": int(right_nullspace(fixed_task_liftings, tolerance).shape[1]),
            },
            "current_null_intersection_with_all_structural_null": intersection_dimension(current_null, full_structural_null, tolerance),
            "current_design_only_null_dimension": int(current_null.shape[1] - intersection_dimension(current_null, full_structural_null, tolerance)),
            "interpretation": "The current four-dimensional null contains one structural invariant under the beta-varying family and three directions not excited by the finite current probe design; the statement is numerical for the sampled full family.",
        },
        "pure_families": {
            "step": 0.08,
            "metadata": pure_metadata,
            "lifting_geometry": _strip_spans(pure_lift_geometry),
            "response_geometry": _strip_spans(pure_response_geometry),
            "incremental_ranks_lifting": family_incremental_ranks(pure_matrix, pure_order, "shift", tolerance),
            "incremental_ranks_response": family_incremental_ranks(pure_response, pure_order, "shift", tolerance),
            "family_order": pure_names,
        },
        "rank_profile": rank_profile,
        "groups": {"model": {name: list(indices) for name, indices in model_groups.items()}, "shift": {name: list(indices) for name, indices in shift_groups.items()}},
        "shift_geometry": _strip_spans(shift_geometry),
        "model_geometry": _strip_spans(model_geometry),
        "family_ranks": {"shift": shift_geometry["group_ranks"], "model": model_geometry["group_ranks"]},
        "intersection_dimensions": {"shift": {name: values["intersection_dimension"] for name, values in shift_pairs.items()}, "model": {name: values["intersection_dimension"] for name, values in model_pairs.items()}},
        "principal_angles": {"shift": {name: values["principal_angles"] for name, values in shift_pairs.items()}, "model": {name: values["principal_angles"] for name, values in model_pairs.items()}},
        "inclusion_residuals": {"shift": {name: {"first_in_second": values["first_in_second_residual"], "second_in_first": values["second_in_first_residual"]} for name, values in shift_pairs.items()}, "model": {name: {"first_in_second": values["first_in_second_residual"], "second_in_first": values["second_in_first_residual"]} for name, values in model_pairs.items()}},
        "incremental_ranks": {
            "shift": family_incremental_ranks(factor["response"], shift_order, "shift", tolerance),
            "model": family_incremental_ranks(factor["response"], model_order, "model", tolerance),
        },
        "tolerance": dataclasses.asdict(tolerance),
        "deferred_questions": ["DEFER-TO-3B: response-order structure", "DEFER-TO-3C: mechanism semantics and blind discovery", "DEFER-TO-LATER: regularizer control"],
    }
    return _jsonable(result)


def _write_csv(path: Path, header: tuple[str, ...], rows: list[tuple[object, ...]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def _report(result: dict[str, object]) -> str:
    f = result["factorization"]
    q = result["quotient"]
    limiting = "both" if q["model_blind_dimension"] and q["shift_blind_dimension"] else "model ensemble" if q["shift_blind_dimension"] else "shift ensemble" if q["model_blind_dimension"] else "neither"
    null_records = result["shift_nullspace"]["coordinate_functionals"]
    null_lines = []
    for record in null_records:
        null_lines.append(f"| `{record['free_coordinate']}` | `{_coefficient_text(record['coefficients'])}` |")
    reach = result["structural_reachability"]
    pure = result["pure_families"]
    pure_lines = []
    for name in pure["family_order"]:
        pure_lines.append(f"| `{name}` | {pure['lifting_geometry']['group_ranks'][name]} | {pure['response_geometry']['group_ranks'][name]} |")
    pair_lines = []
    for name, values in pure["lifting_geometry"]["pairwise"].items():
        pair_lines.append(f"| `{name}` | {values['intersection_dimension']} | `{_subspace_relation(values)}` | {values['first_in_second_residual']:.3e} | {values['second_in_first_residual']:.3e} |")
    inc_lines = []
    for row in pure["incremental_ranks_lifting"]:
        inc_lines.append(f"`{row['group']}`: +{row['incremental_rank']} (rank {row['rank_before']} -> {row['rank_after']})")
    pure_response_vs_lifting = {
        "group_rank_mismatches": {
            name: pure["lifting_geometry"]["group_ranks"][name] != pure["response_geometry"]["group_ranks"][name]
            for name in pure["family_order"]
        },
        "joint_rank_mismatch": pure["lifting_geometry"]["joint_rank"] != pure["response_geometry"]["joint_rank"],
    }
    return f"""# 3A 报告：OOD Risk-Response Algebra

## 范围

本报告只分析 raw population risk-response matrix 的 lifting factorization、rank、quotient 和线性子空间几何。没有运行 clustering、机制命名、response-order 分解或 regularizer control。

## 1. 精确因子化

对 (X=(1,C,A)) 使用带 \\(\\sqrt{{2}}\\) off-diagonal 权重的 Frobenius-isometric `svec`，定义 \\(\\phi(w)\\) 和 \\(\\psi(T)\\)。结果：

- \\(N_{{model}}={result['data']['n_models']}\\)，\\(N_{{shift}}={result['data']['n_shifts']}\\)，lifting feature dimension=`{result['data']['feature_dimension']}`；
- \\(\\operatorname{{rank}}(\\Phi)={f['phi_rank']}\\)；
- \\(\\operatorname{{rank}}(\\Psi)={f['psi_rank']}\\)；
- \\(\\operatorname{{rank}}(\\mathsf R)={f['response_rank']}\\)；
- factorization max residual=`{f['residual_max_absolute']:.3e}`，Frobenius residual=`{f['residual_frobenius']:.3e}`；
- exact within tolerance: `{f['exact_within_tolerance']}`。

因此当前数据上 \\(\\mathsf R=\\Phi\\Psi^\\top\\) 数值成立。该结论是 quadratic-risk algebra 的 exact equality，不是机制解释。

## 2. Rank 与 quotient

response rank 的瓶颈判断：`{limiting}`。

- model-side blind dimension：`{q['model_blind_dimension']}`；
- shift-side blind dimension：`{q['shift_blind_dimension']}`；
- shift-side risk-visible quotient dimension：`{q['quotient_dimension']}`；
- response rank equals shift lifting rank：`{q['shift_full_visible']}`；
- response rank equals model lifting rank：`{q['model_full_visible']}`。

这里的 blind dimension 只表示 lifting space 上的线性代数 kernel，不表示生成机制盲区。

## 3. 四个 shift-unexcited risk-statistic combinations

当前 shift lifting 的右 null space 为 4 维。下表使用 coordinate-anchored basis，列出的每一行都是一个线性泛函；对当前 300 个 shifts，其值均为零。系数坐标依次对应 \\(\\Delta M_{{XX}}\\) 的 `svec`、\\(\\Delta m_{{XY}}\\) 和 \\(\\Delta m_{{Y^2}}\\)。这是可读代表，不是唯一 basis。

| anchor coordinate | null functional (largest coefficients) |
|---|---|
{chr(10).join(null_lines)}

其中，`dM[0,0]` 这一行是严格的结构不变量：\\(X_0=1\\)，所以 \\(\\Delta M_{{00}}=\\Delta\\mathbb E[1^2]=0\\) 对任何 environment 都成立。其余三行不是由 intercept 恒等式单独推出的普遍约束：在允许所有 structural parameters（包括 \\(\\beta\\)）变化的随机有效 structural family 中，数值 span rank 为 `{reach['all_structural_parameters_vary']['numerical_rank']}`，只剩 `{reach['all_structural_parameters_vary']['null_dimension']}` 维 null；当前 4 维 null 与它的交为 `{reach['current_null_intersection_with_all_structural_null']}` 维。因此当前结果应解释为：`1` 个 universal structural invariant + `3` 个当前 finite shift design 未激活的 combinations，而不是 4 个都被 SCM 普遍禁止。

固定 task mechanism（\\(\\beta\\) 不变）的 structural reachability 数值 rank 为 `{reach['task_fixed']['numerical_rank']}`；这说明是否把 task shifts 纳入声明 family 会改变 quotient dimension，属于 environment-family 选择，而不是同一个 rank 结论。

## 4. Pure structural-family 子空间

以下 probes 每次只改变一个 structural parameter family，并同时加入正负 finite shifts；这不是 3B 的阶数分解。`lifting rank` 在 21 维 risk-statistic coordinate space 中计算，`response rank` 在 660 维 model-response space 中计算。

| family | lifting rank | response rank |
|---|---:|---:|
{chr(10).join(pure_lines)}

| pair | intersection dimension | relation | first in second residual | second in first residual |
|---|---:|---|---:|---:|
{chr(10).join(pair_lines)}

按当前固定顺序累积 pure family 的 lifting incremental rank：{'; '.join(inc_lines)}。

`b` 与 `mu_xi` 的两行在 lifting 与 response geometry 中均相同（intersection 等于各自 rank、两个 inclusion residual 为零），因为在线性观测方程中它们只通过 `b + mu_xi` 进入 A。其余 family 多数是部分重叠或零交，而不是两两正交 direct sum。当前 pure lifting 与 response 的 group/joint ranks 是否一致：`{pure_response_vs_lifting}`；主角本身仍依赖各自 ambient inner product。

## 5. 结论边界

当前可确认：

1. response matrix 存在由 model lifting 与 shift lifting 共同决定的精确双侧因子化；
2. response rank 给出 source model ensemble 实际能看到的 shift lifting quotient 维度；
3. 两侧 rank 差分别量化 model-side 与 shift-side algebraic blind dimensions。

当前不能确认：

- 低秩是否对应 core/nuisance/relation 等语义；`DEFER-TO-3C`；
- 一阶/二阶或更高阶响应如何组成 filtration；`DEFER-TO-3B`；
- 任意正则项是否控制某个 quotient factor；延期，不在 3A 结论中讨论。

详细结果见 `results/`。
"""


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    output = root / "round3" / "response_algebra"
    results = output / "results"
    results.mkdir(parents=True, exist_ok=True)
    result = run(root)
    (results / "round3_3a_results.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    f, q = result["factorization"], result["quotient"]
    _write_csv(results / "rank_summary.csv", ("quantity", "value"), [(k, v) for k, v in {**f, **q}.items() if not isinstance(v, (dict, list))])
    rows = []
    for axis in ("shift", "model"):
        for name, rank in result[f"{axis}_geometry"]["group_ranks"].items():
            rows.append((axis, name, rank, result[f"{axis}_geometry"]["joint_rank"]))
    _write_csv(results / "subspace_summary.csv", ("axis", "group", "rank", "joint_rank"), rows)
    angle_rows = []
    for axis in ("shift", "model"):
        for name, values in result[f"{axis}_geometry"]["pairwise"].items():
            angle_rows.append((axis, name, json.dumps(values["principal_angles"]), values["intersection_dimension"], values["first_in_second_residual"], values["second_in_first_residual"]))
    _write_csv(results / "principal_angles.csv", ("axis", "pair", "angles", "intersection_dimension", "first_in_second_residual", "second_in_first_residual"), angle_rows)
    inc_rows = [(axis, row["group"], row["rank_before"], row["rank_after"], row["incremental_rank"]) for axis, rows_ in result["incremental_ranks"].items() for row in rows_]
    _write_csv(results / "incremental_rank.csv", ("axis", "group", "rank_before", "rank_after", "incremental_rank"), inc_rows)
    null_rows = []
    for record in result["shift_nullspace"]["coordinate_functionals"]:
        null_rows.append((record["free_coordinate"], json.dumps(record["coefficients"], ensure_ascii=False), record["functional"]["frobenius_norm_mxx"], record["functional"]["l2_norm_mxy"], record["functional"]["delta_my2_functional"]))
    _write_csv(results / "nullspace_functionals.csv", ("free_coordinate", "coefficients", "mxx_functional_frobenius_norm", "mxy_functional_l2_norm", "y2_functional"), null_rows)
    pure_rank_rows = []
    for name in result["pure_families"]["family_order"]:
        pure_rank_rows.append((name, result["pure_families"]["lifting_geometry"]["group_ranks"][name], result["pure_families"]["response_geometry"]["group_ranks"][name]))
    _write_csv(results / "pure_family_rank.csv", ("family", "lifting_rank", "response_rank"), pure_rank_rows)
    pure_pair_rows = []
    for name, values in result["pure_families"]["lifting_geometry"]["pairwise"].items():
        pure_pair_rows.append((name, values["intersection_dimension"], _subspace_relation(values), values["first_in_second_residual"], values["second_in_first_residual"], json.dumps(values["principal_angles"])))
    _write_csv(results / "pure_family_pairwise.csv", ("pair", "intersection_dimension", "relation", "first_in_second_residual", "second_in_first_residual", "principal_angles"), pure_pair_rows)
    pure_inc_rows = [("lifting", row["group"], row["rank_before"], row["rank_after"], row["incremental_rank"]) for row in result["pure_families"]["incremental_ranks_lifting"]]
    pure_inc_rows.extend(("response", row["group"], row["rank_before"], row["rank_after"], row["incremental_rank"]) for row in result["pure_families"]["incremental_ranks_response"])
    _write_csv(results / "pure_family_incremental.csv", ("space", "family", "rank_before", "rank_after", "incremental_rank"), pure_inc_rows)
    (output / "round3_3a_report.md").write_text(_report(result), encoding="utf-8")
    print(output / "round3_3a_report.md")


if __name__ == "__main__":
    main()
