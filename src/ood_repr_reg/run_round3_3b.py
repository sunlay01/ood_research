"""Run the scoped 3B differential response-order audit."""

from __future__ import annotations

import csv
import dataclasses
import json
from pathlib import Path

import numpy as np

from .round3_3b_derivatives import (
    combine_directions,
    OrderDirection,
    OrderEnvironment,
    direction_from_environments,
    directional_liftings,
    moment_polynomial,
    model_lift,
    pure_family_directions,
    risk_polynomial,
)
from .round3_3b_finite_difference import (
    analytic_response_matrix,
    finite_difference_response_matrix,
    finite_response_at_one,
    step_size_convergence,
    validation_error,
)
from .round3_3b_filtration import order_overlap, order_ranks, project_residual, rank_profile
from .round3_algebra import AlgebraTolerance, rank_with_tolerance
from .round3_shift_ensemble import PopulationEnvironment, base_environment


def _load_models(root: Path) -> tuple[np.ndarray, ...]:
    path = root / "round3" / "empirical_discovery" / "retry" / "results" / "round3_retry_results.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return tuple(np.asarray(item["weights"], dtype=float) for item in payload["models"])


def _to_order_environment(value: PopulationEnvironment) -> OrderEnvironment:
    return OrderEnvironment(value.mu_c, value.sigma_c, value.gamma, value.b, value.mu_xi, value.sigma_xi, value.beta, value.noise_variance)


def _zero_direction(base: OrderEnvironment) -> OrderDirection:
    return OrderDirection(
        np.zeros_like(base.mu_c), np.zeros_like(base.sigma_c), np.zeros_like(base.gamma),
        np.zeros_like(base.b), np.zeros_like(base.mu_xi), np.zeros_like(base.sigma_xi),
        np.zeros_like(base.beta), "zero", "zero",
    )


def _direction_records(directions: tuple[OrderDirection, ...]) -> list[dict[str, object]]:
    return [{"name": d.name, "family": d.family} for d in directions]


def _finite_rank_diagnostics(
    analytic: np.ndarray,
    finite_difference: dict[str, np.ndarray],
    tolerance: AlgebraTolerance,
    zero_threshold: float = 1e-12,
) -> tuple[dict[str, int | None], dict[str, str]]:
    ranks: dict[str, int | None] = {}
    statuses: dict[str, str] = {}
    analytic_norm = float(np.linalg.norm(analytic))
    for h, matrix in finite_difference.items():
        if analytic_norm <= zero_threshold:
            ranks[h] = None
            statuses[h] = "invalid-zero-derivative"
        else:
            ranks[h] = rank_with_tolerance(matrix, tolerance)
            statuses[h] = "validated"
    return ranks, statuses


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


def _base_from_kind(base: OrderEnvironment, kind: str) -> OrderEnvironment:
    if kind == "generic-2":
        return OrderEnvironment(
            np.array([-0.22, 0.31]), np.array([[0.95, -0.12], [-0.12, 1.15]]),
            np.array([[0.52, -0.18], [0.27, 0.68]]), np.array([-0.12, 0.16]),
            np.array([0.08, -0.05]), np.array([[0.58, -0.04], [-0.04, 0.74]]),
            np.array([0.83, -0.61]), base.noise_variance,
        )
    if kind == "symmetric-zero-mean":
        return OrderEnvironment(
            np.zeros(2), np.eye(2), base.gamma, np.zeros(2), np.zeros(2),
            np.eye(2) * 0.55, base.beta, base.noise_variance,
        )
    return base


def _safe_direction(direction: OrderDirection, base: OrderEnvironment, h_max: float) -> OrderDirection:
    """Scale covariance directions until every +/-2h point remains SPD."""
    result = direction
    for _ in range(20):
        path_times = (-1.0, -2.0 * h_max, 2.0 * h_max, 1.0)
        if all(base.at(result, time).is_valid_covariance() for time in path_times):
            return result
        result = result.scaled(0.5)
    raise ValueError(f"could not find a valid covariance path for {direction.name}")


def _family_analysis(base: OrderEnvironment, weights: tuple[np.ndarray, ...], h_values: tuple[float, ...], tolerance: AlgebraTolerance) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    families = pure_family_directions(base)
    family_results: dict[str, object] = {}
    analytic_by_order: dict[int, list[np.ndarray]] = {1: [], 2: [], 3: []}
    all_finite_columns: list[np.ndarray] = []
    for family, raw_directions in families.items():
        directions = tuple(_safe_direction(d, base, max(h_values)) for d in raw_directions)
        matrices = {order: analytic_response_matrix(weights, base, directions, order) for order in (1, 2, 3)}
        for order in matrices:
            analytic_by_order[order].append(matrices[order])
        finite_directions = directions + tuple(d.scaled(-1.0, name=f"-{d.name}") for d in directions)
        finite = finite_response_at_one(weights, base, finite_directions)
        all_finite_columns.append(finite)
        convergence = {str(order): step_size_convergence(weights, base, directions, order, h_values) for order in (1, 2, 3)}
        errors = {
            str(order): {str(h): validation_error(matrices[order], finite_difference_response_matrix(weights, base, directions, h, order)) for h in h_values}
            for order in (1, 2, 3)
        }
        finite_matrices = {
            str(order): {str(h): finite_difference_response_matrix(weights, base, directions, h, order) for h in h_values}
            for order in (1, 2, 3)
        }
        finite_ranks, finite_rank_status = {}, {}
        for order in (1, 2, 3):
            finite_ranks[str(order)], finite_rank_status[str(order)] = _finite_rank_diagnostics(matrices[order], finite_matrices[str(order)], tolerance)
        risk_degrees = sorted({len(risk_polynomial(weight, base, direction)) - 1 for weight in weights for direction in directions})
        moment_degrees = sorted({
            max(len(poly) for poly in moment_polynomial(base, direction)) - 1
            for direction in directions
        })
        ranks = order_ranks(matrices, tolerance)
        family_results[family] = {
            "n_directions": len(directions),
            "n_finite_directions": len(finite_directions),
            "directions": _direction_records(directions),
            "individual_ranks": ranks["individual_ranks"],
            "cumulative_ranks": ranks["cumulative_ranks"],
            "incremental_ranks": ranks["incremental_ranks"],
            "finite_rank": rank_with_tolerance(finite, tolerance),
            "polynomial_degree_range": [min(risk_degrees), max(risk_degrees)],
            "moment_polynomial_degree_range": [min(moment_degrees), max(moment_degrees)],
            "finite_closure_residual_le_1": project_residual(finite, matrices[1], tolerance),
            "finite_closure_residual_le_2": project_residual(finite, np.column_stack((matrices[1], matrices[2])), tolerance),
            "finite_closure_residual_le_3": project_residual(finite, np.column_stack((matrices[1], matrices[2], matrices[3])), tolerance),
            "step_size_convergence": convergence,
            "relative_fd_error": errors,
            "finite_difference_ranks": finite_ranks,
            "finite_difference_rank_status": finite_rank_status,
            "order_1_2_overlap": order_overlap(matrices[1], matrices[2], tolerance),
            "order_1_3_overlap": order_overlap(matrices[1], matrices[3], tolerance),
            "order_2_3_overlap": order_overlap(matrices[2], matrices[3], tolerance),
        }
    return family_results, {order: np.column_stack(blocks) for order, blocks in analytic_by_order.items()}


def _mixed_analysis(base: OrderEnvironment, weights: tuple[np.ndarray, ...], h_values: tuple[float, ...], pure_matrices: dict[int, np.ndarray], tolerance: AlgebraTolerance, count: int = 50) -> dict[str, object]:
    families = pure_family_directions(base, samples_per_family=count, seed=20260907)
    combinations = (("core_mean+relation", ("core_mean", "relation")), ("core_covariance+relation", ("core_covariance", "relation")), ("relation+b", ("relation", "b")), ("core_mean+task", ("core_mean", "task")), ("relation+task", ("relation", "task")))
    rows = []
    rng = np.random.default_rng(20260908)
    pure_span = np.column_stack(tuple(pure_matrices.values()))
    for name, selected in combinations:
        paths = []
        order_values = {order: [] for order in (1, 2, 3, 4)}
        for index in range(count):
            chosen = tuple(families[family][int(rng.integers(0, len(families[family])))] for family in selected)
            direction = _safe_direction(combine_directions(*chosen, name=f"{name}[{index}]", family="mixed"), base, max(h_values))
            matrices = {order: analytic_response_matrix(weights, base, (direction,), order) for order in (1, 2, 3, 4)}
            for order, matrix in matrices.items():
                order_values[order].append(matrix[:, 0])
            polynomial_degree = max(len(risk_polynomial(weights[0], base, direction)) - 1, 0)
            paths.append({"degree": polynomial_degree, "orders": {str(order): {"norm": float(np.linalg.norm(matrix)), "residual_outside_pure": project_residual(matrix, pure_span, tolerance)} for order, matrix in matrices.items()}})
        stacked = {order: np.column_stack(values) for order, values in order_values.items()}
        pure_cumulative = np.column_stack(tuple(pure_matrices.values()))
        pure_rank = rank_with_tolerance(pure_cumulative, tolerance)
        mixed_cumulative = {
            order: np.column_stack([stacked[index] for index in range(1, order + 1)])
            for order in (1, 2, 3, 4)
        }
        joint_ranks = {
            str(order): rank_with_tolerance(np.column_stack((pure_cumulative, mixed_cumulative[order])), tolerance)
            for order in (1, 2, 3, 4)
        }
        rows.append({
            "path": name,
            "n_directions": count,
            "max_degree": max(row["degree"] for row in paths),
            "order_ranks": {str(order): rank_with_tolerance(np.column_stack(values), tolerance) for order, values in order_values.items()},
            "cumulative_ranks": {
                str(order): rank_with_tolerance(np.column_stack([stacked[index] for index in range(1, order + 1)]), tolerance)
                for order in (1, 2, 3, 4)
            },
            "new_ranks": {
                str(order): rank_with_tolerance(np.column_stack([stacked[index] for index in range(1, order + 1)]), tolerance)
                - (rank_with_tolerance(np.column_stack([stacked[index] for index in range(1, order)]), tolerance) if order > 1 else 0)
                for order in (1, 2, 3, 4)
            },
            "pure_rank": pure_rank,
            "pure_plus_mixed_cumulative_rank": joint_ranks,
            "incremental_over_pure": {key: value - pure_rank for key, value in joint_ranks.items()},
            "paths": paths,
        })
    return {"paths": rows, "interpretation": "mixed paths are a finite interaction diagnostic; no semantic naming is assigned"}


def run(root: Path | None = None) -> dict[str, object]:
    root = root or Path(__file__).resolve().parents[2]
    old_base = _to_order_environment(base_environment())
    weights = _load_models(root)
    h_values = (0.025, 0.05, 0.1, 0.2, 0.4)
    tolerance = AlgebraTolerance()
    base_results: dict[str, object] = {}
    base_matrices: dict[str, dict[int, np.ndarray]] = {}
    base_global_filtration: dict[str, dict[str, object]] = {}
    for name in ("nonzero-anisotropic", "symmetric-zero-mean", "generic-2"):
        base = _base_from_kind(old_base, name)
        result, matrices = _family_analysis(base, weights, h_values, tolerance)
        base_results[name] = result
        base_matrices[name] = matrices
        base_global_filtration[name] = order_ranks(matrices, tolerance)

    primary_base = _base_from_kind(old_base, "nonzero-anisotropic")
    primary_families = pure_family_directions(primary_base)
    primary_directions = tuple(_safe_direction(d, primary_base, max(h_values)) for values in primary_families.values() for d in values)
    analytic = {order: analytic_response_matrix(weights, primary_base, primary_directions, order) for order in (1, 2, 3)}
    finite_directions = primary_directions + tuple(d.scaled(-1.0, name=f"-{d.name}") for d in primary_directions)
    finite = finite_response_at_one(weights, primary_base, finite_directions)
    filtration = order_ranks(analytic, tolerance)
    taylor = {str(order): project_residual(finite, np.column_stack([analytic[k] for k in range(1, order + 1)]), tolerance) for order in (1, 2, 3)}
    profiles = {str(order): rank_profile(analytic[order], (1e-12, 1e-10, 1e-9, 1e-8, 1e-6)) for order in (1, 2, 3)}
    finite_rank_profiles = {
        str(order): {str(h): finite_difference_response_matrix(weights, primary_base, primary_directions, h, order) for h in h_values}
        for order in (1, 2, 3)
    }
    finite_difference_ranks, finite_difference_status = {}, {}
    for order in (1, 2, 3):
        finite_difference_ranks[str(order)], finite_difference_status[str(order)] = _finite_rank_diagnostics(analytic[order], finite_rank_profiles[str(order)], tolerance)
    finite_rank_profiles = finite_difference_ranks
    order_lifting_ranks = {str(order): rank_with_tolerance(directional_liftings(primary_base, primary_directions, order), tolerance) for order in (1, 2, 3)}
    order_factorization = {
        str(order): {
            "shift_lifting_rank": order_lifting_ranks[str(order)],
            "response_rank": rank_with_tolerance(analytic[order], tolerance),
            "rank_equality": order_lifting_ranks[str(order)] == rank_with_tolerance(analytic[order], tolerance),
        }
        for order in (1, 2, 3)
    }
    global_fd_error_at_01 = {
        str(order): validation_error(analytic[order], finite_difference_response_matrix(weights, primary_base, primary_directions, 0.1, order))
        for order in (1, 2, 3)
    }
    # finite 3A pure-family span, made with the same structural axes and no labels in the algebra.
    finite_rank = rank_with_tolerance(finite, tolerance)
    cumulative = filtration["cumulative_ranks"]
    if cumulative["1"] == finite_rank:
        verdict = "FIRST-ORDER-CLOSED"
    elif cumulative["2"] == finite_rank:
        verdict = "SECOND-ORDER-CLOSURE"
    elif cumulative["3"] == finite_rank:
        verdict = "HIGHER-ORDER-CLOSURE"
    else:
        verdict = "NO-STABLE-ORDER-DECOMPOSITION"
    return _jsonable({
        "status": "complete",
        "scope": "3B-differential-response-order-only",
        "data": {"n_models": len(weights), "feature_dimension": 21, "h_values": h_values, "finite_path_time": 1.0},
        "theory": {
            "order_factorization": "R^(k)=Phi (Psi^(k))^T",
            "derivative_definition": "directional derivatives of exact population quadratic risk",
            "semantic_status": "DEFER-TO-3C",
            "regularizer_status": "DEFER-TO-LATER",
        },
        "bases": base_results,
        "base_global_filtration": base_global_filtration,
        "global": {
            "individual_ranks": filtration["individual_ranks"],
            "cumulative_ranks": cumulative,
            "incremental_ranks": filtration["incremental_ranks"],
            "finite_rank": finite_rank,
            "order_1_2_overlap": order_overlap(analytic[1], analytic[2], tolerance),
            "order_1_3_overlap": order_overlap(analytic[1], analytic[3], tolerance),
            "order_2_3_overlap": order_overlap(analytic[2], analytic[3], tolerance),
            "taylor_closure_residual": taylor,
            "rank_profiles": profiles,
            "finite_difference_ranks": finite_rank_profiles,
            "finite_difference_rank_status": finite_difference_status,
            "order_lifting_ranks": order_lifting_ranks,
            "model_lift_rank": rank_with_tolerance(np.asarray([model_lift(model) for model in weights]), tolerance),
            "order_factorization": order_factorization,
            "fd_error_at_h_0_1": global_fd_error_at_01,
            "finite_difference_rank_status": finite_difference_status,
            "finite_response_norm": float(np.linalg.norm(finite)),
            "analytic_derivative_norms": {str(k): float(np.linalg.norm(v)) for k, v in analytic.items()},
            "analytic_response_matrices": {str(k): analytic[k] for k in (1, 2, 3)},
            "finite_response_matrix": finite,
            "step_size_convergence": {str(k): step_size_convergence(weights, primary_base, primary_directions, k, h_values) for k in (1, 2, 3)},
        },
        "mixed_paths": _mixed_analysis(primary_base, weights, h_values, base_matrices["nonzero-anisotropic"], tolerance),
        "verdict": verdict,
        "tolerance": dataclasses.asdict(tolerance),
        "deferred_questions": ["DEFER-TO-3C: mechanism semantics", "DEFER-TO-LATER: regularizer control"],
    })


def _write_csv(path: Path, header: tuple[str, ...], rows: list[tuple[object, ...]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def _report(result: dict[str, object]) -> str:
    g = result["global"]
    lines = []
    for family, value in result["bases"]["nonzero-anisotropic"].items():
        lines.append(f"| {family} | {value['finite_rank']} | {value['individual_ranks']['1']} | {value['cumulative_ranks']['2']} | {value['cumulative_ranks']['3']} | {value['incremental_ranks']['2']} | {value['incremental_ranks']['3']} | {value['moment_polynomial_degree_range']} |")
    base_lines = []
    for name, families in result["bases"].items():
        base_lines.append(f"| {name} | {result['base_global_filtration'][name]['cumulative_ranks']['1']} | {result['base_global_filtration'][name]['cumulative_ranks']['2']} | {result['base_global_filtration'][name]['cumulative_ranks']['3']} |")
    return f"""# 3B 报告：OOD Risk-Response Differential / Order Filtration

## 裁决：`{result['verdict']}`

本轨只分析 3A risk-response space 的 differential order。没有 clustering、机制命名、正则控制或新的 ambient space。语义问题：`DEFER-TO-3C`；正则问题：`DEFER-TO-LATER`。

## 数据与方法

- 使用 3A 已保存的 `{result['data']['n_models']}` 个模型权重；没有重新手写 predictor。
- structural path 为 `eta(t)=eta0+t v`，协方差路径不做 eigenvalue clamp；所有参与有限差分的点均保持正定。
- 使用 `h={result['data']['h_values']}` 的正负点，计算一阶、二阶、三阶中心差分；解析导数来自同一 structural moment polynomial。

## 全局 filtration

| quantity | value |
|---|---:|
| rank D1 | {g['individual_ranks']['1']} |
| rank D2 | {g['individual_ranks']['2']} |
| rank D3 | {g['individual_ranks']['3']} |
| cumulative <=1 | {g['cumulative_ranks']['1']} |
| cumulative <=2 | {g['cumulative_ranks']['2']} |
| cumulative <=3 | {g['cumulative_ranks']['3']} |
| new @2 | {g['incremental_ranks']['2']} |
| new @3 | {g['incremental_ranks']['3']} |
| finite pure span rank | {g['finite_rank']} |
| rho1 | {g['taylor_closure_residual']['1']:.3e} |
| rho2 | {g['taylor_closure_residual']['2']:.3e} |
| rho3 | {g['taylor_closure_residual']['3']:.3e} |

### Pure-family ranks (nonzero anisotropic base)

| family | finite rank | rank D1 | cumulative <=2 | cumulative <=3 | new@2 | new@3 | moment degree range |
|---|---:|---:|---:|---:|---:|---:|---|
{chr(10).join(lines)}

### Base-point comparison

| base | global cumulative <=1 | global cumulative <=2 | global cumulative <=3 |
|---|---:|---:|---:|
{chr(10).join(base_lines)}

## 解析与有限差分

三阶 stencil 采用 ` [f(2h)-2f(h)+2f(-h)-f(-2h)]/(2h^3) `，对 `t^3` 返回 `6`。每个 family 的逐步收敛和相对误差保存在 JSON；rank 只有在解析/有限差分一致且 tolerance profile 稳定时才解释。

一阶有限差分 rank 在所有五个步长均为 16，二阶均为 7，与解析 rank 一致。解析三阶导数严格为 0；由于三阶 stencil 具有 `h^-3` 放大，零导数上的浮点残差在原始 SVD 中产生伪 rank。因此三阶 finite-difference rank 被标记为 `invalid-zero-derivative`，不参与阶数裁决；三阶结论使用解析矩阵及其零范数。

## Mixed paths

Mixed paths 仅报告是否离开 pure-order span，不赋予任何机制语义。它们是 interaction diagnostic，不能替代 pure-family filtration。全量结果显示 `core_mean+relation` 和 `core_mean+task` 的 mixed affine path 可达到四阶；因此本报告的 `SECOND-ORDER-CLOSURE` 只对 pure-family finite span 成立，不是对任意 mixed structural path 的全局二阶结论。机器结果中的 `incremental_over_pure` 是 mixed cumulative span 相对 pure cumulative span 的实际增量；它不等于机制 interaction 的因果解释。

本轮每个 pure family 使用 100 个确定性随机方向，避免旧坐标轴实验的方向欠采样。随机方向下 `core_mean`、`relation`、`b`、`mu_xi` 的 finite ranks 分别为 5、7、5、5；旧坐标轴结果的 4、6、4、4 不能作为充分覆盖下的 family rank。

## 结论边界

- `R^(k)=Phi(Psi^(k))^T` 是 quadratic population risk 下的精确因子化；当 model lifting rank 在相关子空间上无 kernel 时，order rank 与 shift-lifting rank 相等。
- `V^(<=1) subseteq V^(<=2) subseteq V^(<=3)` 是由 span 定义产生的 filtration；新增秩是代数结论，不是机制结论。
- 二阶或三阶新增方向即使存在，也不能命名为 core、nuisance 或 relation：`DEFER-TO-3C`。
- 本报告不把响应幅度或 penalty 下降解释为控制：`DEFER-TO-LATER`。

机器结果见 `results/round3_3b_results.json`，阶数矩阵和汇总 CSV 见同目录。
"""


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    output = root / "round3" / "3B_response_order"
    results = output / "results"
    results.mkdir(parents=True, exist_ok=True)
    result = run(root)
    (results / "round3_3b_results.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    global_result = result["global"]
    _write_csv(results / "global_order_rank.csv", ("quantity", "value"), [(key, value) for key, value in global_result["cumulative_ranks"].items()])
    for order, matrix in global_result["analytic_response_matrices"].items():
        np.savetxt(results / f"response_order_{order}_analytic.csv", np.asarray(matrix), delimiter=",")
    np.savetxt(results / "response_finite_pm.csv", np.asarray(global_result["finite_response_matrix"]), delimiter=",")
    rows = []
    for family, value in result["bases"]["nonzero-anisotropic"].items():
        rows.append((family, value["finite_rank"], value["individual_ranks"]["1"], value["cumulative_ranks"]["2"], value["cumulative_ranks"]["3"], value["incremental_ranks"]["2"], value["incremental_ranks"]["3"], value["moment_polynomial_degree_range"]))
    _write_csv(results / "pure_family_order.csv", ("family", "finite_rank", "rank_d1", "cumulative_le2", "cumulative_le3", "new_at_2", "new_at_3", "moment_degree_range"), rows)
    base_rows = [(name, max(v["cumulative_ranks"]["1"] for v in families.values()), max(v["cumulative_ranks"]["2"] for v in families.values()), max(v["cumulative_ranks"]["3"] for v in families.values())) for name, families in result["bases"].items()]
    _write_csv(results / "base_order_summary.csv", ("base", "max_family_le1", "max_family_le2", "max_family_le3"), base_rows)
    (output / "round3_3b_report.md").write_text(_report(result), encoding="utf-8")
    print(output / "round3_3b_report.md")


if __name__ == "__main__":
    main()
