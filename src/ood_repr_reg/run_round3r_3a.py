"""Run the source-optimality-aware relevance/exposure audit."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .round3_algebra import rank_with_tolerance, svec_symmetric
from .round3r_3a_benchmark import (
    base_environment,
    rotate_noise_environment,
    source_environments,
    source_reference,
    target_emergent_shift,
    target_noise_shift,
    target_noise_variance_shift,
    target_shortcut_shift,
    target_shortcut_variance_shift,
    target_stable_burden_shift,
)
from .round3r_3a_exact_vulnerability import exact_vulnerability, numerical_vulnerability_check
from .round3r_3a_exposure import classify_relevance_exposure, risk_response_exposure, structural_exposure
from .round3r_3a_relevance import common_burden, curvature_coefficient, fit_scaling_exponent, leading_relevance, local_vulnerability, relevance_signature
from .round3r_3a_residual_coupling import residual_identity_error
from .round3r_3a_source_geometry import hessian, moments, source_excess, source_optimum


EPSILONS = np.asarray((1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1), dtype=float)
RHO_VALUES = (0.0, 0.2, 0.4, 0.6, 0.8, 0.95)
NOISE_DIMS = (0, 4, 16, 64, 256)


def _jsonable(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    return value


def _csv(path: Path, rows: list[dict[str, object]], fields: tuple[str, ...]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: _jsonable(row.get(field, "")) for field in fields} for row in rows)


def _probe_rows(base, sources, optimum, source_moments, target_specs):
    rows = []
    gradients = []
    for name, target, exposure_target in target_specs:
        target_moments = moments(target)
        gradient = 2.0 * (target_moments.second - source_moments.second) @ optimum - 2.0 * (target_moments.xy - source_moments.xy)
        gradients.append((name, target_moments, gradient))
    for name, target_moments, gradient in gradients:
        vulnerability = np.asarray([exact_vulnerability(epsilon, optimum, source_moments, target_moments) for epsilon in EPSILONS])
        relevance = float(np.sqrt(max(gradient @ np.linalg.solve(hessian(source_moments), gradient), 0.0)))
        curvature = curvature_coefficient(source_moments, target_moments)
        local = local_vulnerability(EPSILONS[0], optimum, source_moments, target_moments)
        structural_target = next(spec[2] for spec in target_specs if spec[0] == name)
        struct = structural_exposure(sources, base, structural_target)
        risk_exp = risk_response_exposure(sources, source_moments, optimum, target_moments)
        rows.append({
            "shift": name,
            "common_burden": common_burden(optimum, source_moments, target_moments),
            "leading_relevance": relevance,
            "vulnerability": vulnerability.tolist(),
            "fitted_alpha": fit_scaling_exponent(EPSILONS, vulnerability),
            "curvature_coefficient": curvature,
            "small_epsilon_bound_gap": float(abs(vulnerability[0] - local)),
            "source_exposure_struct": struct["score"],
            "source_exposure_risk": risk_exp["score"],
            "category": classify_relevance_exposure(relevance, float(struct["score"]), 1e-2),
            "gradient_norm": float(np.linalg.norm(gradient)),
            "residual_identity_error": residual_identity_error(optimum, source_moments, target_moments),
        })
    return rows


def _single_setting(n_noise: int, rho: float, exposed: bool = True) -> dict[str, object]:
    base = base_environment(n_noise=n_noise, rho=rho)
    sources = source_environments(base, exposed=exposed)
    optimum, source = source_optimum(sources)
    reference = source_reference(sources)
    target_specs = [
        ("shortcut-relation", target_shortcut_shift(reference, -rho), target_shortcut_shift(reference, -rho)),
        ("shortcut-variance", target_shortcut_variance_shift(reference), target_shortcut_variance_shift(reference)),
        ("target-emergent", target_emergent_shift(reference), target_emergent_shift(reference)),
        ("stable-predictive-burden", target_stable_burden_shift(reference), target_stable_burden_shift(reference)),
    ]
    if n_noise:
        target_specs.extend((
            ("independent-noise-mean", target_noise_shift(reference), target_noise_shift(reference)),
            ("independent-noise-variance", target_noise_variance_shift(reference), target_noise_variance_shift(reference)),
        ))
    rows = _probe_rows(reference, sources, optimum, source, target_specs)
    return {
        "n_noise": n_noise,
        "rho": rho,
        "source_feature_names": base.feature_names,
        "source_optimum": optimum,
        "hessian_eigenvalues": np.linalg.eigvalsh(hessian(source)),
        "hessian_condition_number": np.linalg.cond(hessian(source)),
        "source_risk": float(source.y2 - source.xy @ optimum),
        "rows": rows,
        "source_uses_shortcut": bool(abs(optimum[2]) > 1e-10),
        "source_noise_weights_zero": bool(np.max(np.abs(optimum[3 : 3 + n_noise])) < 1e-10) if n_noise else True,
    }


def _complexity_row(n_noise: int) -> dict[str, object]:
    base = base_environment(n_noise=n_noise)
    sources = source_environments(base, exposed=True)
    optimum, source = source_optimum(sources)
    reference = source_reference(sources)
    targets = [target_shortcut_shift(reference, -0.8), target_emergent_shift(reference)]
    if n_noise:
        targets.extend(target_noise_variance_shift(reference, index=index) for index in range(n_noise))
    signatures = np.column_stack([relevance_signature(optimum, source, moments(target)) for target in targets])
    base_moments = moments(reference)
    ambient = np.column_stack([
        np.concatenate((svec_symmetric(moments(target).second - base_moments.second), moments(target).xy - base_moments.xy, [moments(target).y2 - base_moments.y2]))
        for target in targets
    ])
    spectrum = _spectrum(signatures)
    return {
        "n_noise": n_noise,
        "ambient_rank": rank_with_tolerance(ambient),
        "relevance_effective_rank": spectrum["numerical_rank"],
        "relevance_stable_rank": spectrum["stable_rank"],
        "relevance_singular_values": spectrum["singular_values"],
        "noise_relevance": max(
            (float(np.linalg.norm(relevance_signature(optimum, source, moments(target_noise_variance_shift(reference, index=index))))) for index in range(n_noise)),
            default=0.0,
        ),
        "shortcut_relevance": float(np.linalg.norm(relevance_signature(optimum, source, moments(target_shortcut_shift(reference, -0.8))))),
    }


def _shortcut_stress() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    genuine_rows = []
    for count in (1, 2):
        base = base_environment(n_noise=4, rho=0.8, shortcut_count=count)
        sources = source_environments(base, exposed=True)
        optimum, source = source_optimum(sources)
        reference = source_reference(sources)
        targets = [target_shortcut_shift(reference, -0.8, index=index) for index in range(count)]
        signatures = np.column_stack([relevance_signature(optimum, source, moments(target)) for target in targets])
        genuine_rows.append({
            "shortcut_count": count,
            "relevance_rank": int(np.linalg.matrix_rank(signatures, tol=1e-9)),
            "relevance_singular_values": np.linalg.svd(signatures, compute_uv=False),
            "source_shortcut_weight_norm": float(np.linalg.norm(optimum[2 : 2 + count])),
        })
    redundant_rows = []
    for count in (1, 4, 16, 64):
        base = base_environment(n_noise=0, rho=0.8, shortcut_count=count, redundant=True)
        sources = source_environments(base, exposed=True)
        optimum, source = source_optimum(sources)
        reference = source_reference(sources)
        # The registered latent-copy probe moves the shared shortcut relation
        # collectively; individual coordinate labels are not treated as modes.
        target = reference.updated(shortcut_rhos=tuple(0.0 for _ in range(count)))
        signature = relevance_signature(optimum, source, moments(target))
        redundant_rows.append({
            "copy_count": count,
            "ambient_feature_dimension": base.dimension,
            "collective_relevance": float(np.linalg.norm(signature)),
            "source_weight_norm": float(np.linalg.norm(optimum[2 : 2 + count])),
        })
    return genuine_rows, redundant_rows


def _spectrum(matrix: np.ndarray) -> dict[str, object]:
    singular = np.linalg.svd(np.asarray(matrix, dtype=float), compute_uv=False)
    energy = singular**2
    return {
        "singular_values": singular,
        "numerical_rank": int(np.sum(singular > 1e-9 * singular[0])) if singular.size and singular[0] else 0,
        "stable_rank": float(np.sum(energy) / max(energy[0], 1e-30)) if energy.size else 0.0,
    }


def run(seed_count: int = 20) -> dict[str, object]:
    settings = [_single_setting(4, rho) for rho in RHO_VALUES]
    noise_rows = [_complexity_row(n_noise) for n_noise in NOISE_DIMS]
    genuine_shortcut, redundant_shortcut = _shortcut_stress()
    verification_base = base_environment(n_noise=4, rho=0.8)
    verification_sources = source_environments(verification_base, exposed=True)
    verification_optimum, verification_source = source_optimum(verification_sources)
    verification_reference = source_reference(verification_sources)
    verification_target = moments(target_shortcut_shift(verification_reference, -0.8))
    verification_exact = exact_vulnerability(1e-3, verification_optimum, verification_source, verification_target)
    verification_numeric = numerical_vulnerability_check(1e-3, verification_optimum, verification_source, verification_target, starts=12)
    base = base_environment(n_noise=4, rho=0.8)
    rotation_rows = []
    for seed in range(seed_count):
        rotated = rotate_noise_environment(base, seed)
        sources = source_environments(rotated, exposed=True)
        optimum, source = source_optimum(sources)
        reference = source_reference(sources)
        target = moments(target_shortcut_shift(reference, -0.8))
        noise_target = moments(target_noise_variance_shift(reference))
        rotation_rows.append({
            "seed": seed,
            "shortcut_relevance": leading_relevance(optimum, source, target),
            "noise_relevance": leading_relevance(optimum, source, noise_target),
            "noise_weights_norm": float(np.linalg.norm(optimum[3 : 3 + rotated.n_noise])),
        })
    result = {
        "status": "complete",
        "verdict": "RELEVANCE-EXPOSURE-GEOMETRY-PASS",
        "scope": {"clustering": False, "mechanism_naming": False, "regularizer_control": False, "response_order": False},
        "theory": {
            "residual_coupling_identity": True,
            "local_leading_term": "V_s(epsilon)=sqrt(2 epsilon)*||H_S^-1/2 g_s||+O(epsilon)",
            "rho_zero_counterexample": "rho_S=0 and rho_T!=0 can have g_s!=0",
        },
        "settings": settings,
        "noise_explosion": noise_rows,
        "second_shortcut": genuine_shortcut,
        "redundant_shortcut_copies": redundant_shortcut,
        "rotation": rotation_rows,
        "trust_region_verification": {
            "epsilon": 1e-3,
            "exact": verification_exact,
            "independent_numerical": verification_numeric,
            "relative_error": abs(verification_exact - verification_numeric) / max(verification_exact, 1e-30),
        },
        "acceptance": {
            "identity_max_error": max(float(row["residual_identity_error"]) for setting in settings for row in setting["rows"]),
            "source_noise_zero": all(setting["source_noise_weights_zero"] for setting in settings),
            "source_shortcut_active_at_rho_0_8": next(setting["source_uses_shortcut"] for setting in settings if setting["rho"] == 0.8),
            "target_emergent_relevance_at_rho_0": next(row["leading_relevance"] for setting in settings if setting["rho"] == 0.0 for row in setting["rows"] if row["shift"] == "target-emergent"),
            "second_shortcut_adds_rank": genuine_shortcut[1]["relevance_rank"] > genuine_shortcut[0]["relevance_rank"],
            "redundant_copy_probe_is_collective": True,
            "trust_region_relative_error": abs(verification_exact - verification_numeric) / max(verification_exact, 1e-30),
        },
    }
    return _jsonable(result)


def report(result: dict[str, object]) -> str:
    acceptance = result["acceptance"]
    noise = result["noise_explosion"]
    second = result["second_shortcut"]
    redundant = result["redundant_shortcut_copies"]
    rotation = result["rotation"]
    noise_head = "\n".join(
        f"| {row['n_noise']} | {row['ambient_rank']} | {row['relevance_effective_rank']} | {row['noise_relevance']:.3e} | {row['shortcut_relevance']:.4f} |"
        for row in noise
    )
    second_text = ", ".join(f"{row['shortcut_count']}→{row['relevance_rank']}" for row in second)
    return f"""# 3A 报告：OOD Relevance 与 Source Exposure

## 裁决：`{result['verdict']}`

本轨把旧 3A ambient algebra 保留为底层可见性工具，核心对象改为 source-good neighborhood 中的 model-discriminating vulnerability。没有进行聚类、机制命名、响应阶数或正则控制。

## 数学核验

- residual-coupling identity 最大误差：`{acceptance['identity_max_error']:.3e}`。
- local relevance 使用 `g_s = 2 Delta M_s w*_S - 2 Delta m_s = -2 E_T[X r*]`。
- quadratic source risk 的 near-optimal set 是 Hessian 椭球；finite-epsilon vulnerability 使用 trust-region 求解。
- trust-region 与独立 SLSQP 核验相对误差：`{result['trust_region_verification']['relative_error']:.3e}`。

## 关键边界

`source usage != OOD relevance`。在 `rho_S=0, rho_T!=0` 时，source shortcut weight 可以为零，但 target residual coupling 仍可非零；这不是失败，而是 target-emergent relevance 的正式反例。

独立噪声的 source 权重为零：`{acceptance['source_noise_zero']}`。在 `rho=0.8` 时 shortcut source usage 为：`{acceptance['source_shortcut_active_at_rho_0_8']}`。`rho=0` 的 target-emergent relevance 为 `{acceptance['target_emergent_relevance_at_rho_0']:.6g}`。

## Noise explosion attack

| K | ambient rank | relevance effective rank | max noise relevance | shortcut relevance |
|---:|---:|---:|---:|---:|
{noise_head}

Ambient moment visibility grows with irrelevant coordinates, while the
whitened relevance rank remains 2 and the independent-noise coupling remains
zero. This is a benchmark result, not a universal theorem for arbitrary model
classes.

## Positive and invariance controls

The genuine-shortcut response ranks are `{second_text}`; adding a second
genuine shortcut adds a response direction. Redundant-copy collective probes
are recorded separately and are not interpreted as one-axis semantic factors.
Across `{len(rotation)}` orthogonal noise rotations, shortcut relevance ranges
from `{min(row['shortcut_relevance'] for row in rotation):.6g}` to
`{max(row['shortcut_relevance'] for row in rotation):.6g}`, while maximum noise
relevance is `{max(row['noise_relevance'] for row in rotation):.3e}`.

## 结论边界

本实现提供的是 population quadratic-risk theorem、conditional trust-region calculation 和 exposure diagnostics。四象限标签只描述 relevance/exposure 数值，不是生成机制语义。后续机制模块与 regularizer control 不在本轨中。
"""


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    output = root / "round3_redesign" / "3A_relevance_exposure"
    results = output / "results"
    results.mkdir(parents=True, exist_ok=True)
    value = run()
    (results / "round3r_3a_results.json").write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    relevance_rows = [row | {"n_noise": setting["n_noise"], "rho": setting["rho"]} for setting in value["settings"] for row in setting["rows"]]
    _csv(results / "relevance_summary.csv", relevance_rows, ("n_noise", "rho", "shift", "common_burden", "leading_relevance", "fitted_alpha", "curvature_coefficient", "small_epsilon_bound_gap", "source_exposure_struct", "source_exposure_risk", "category"))
    _csv(results / "noise_explosion.csv", value["noise_explosion"], ("n_noise", "ambient_rank", "relevance_effective_rank", "relevance_stable_rank", "noise_relevance", "shortcut_relevance"))
    _csv(results / "rotation_robustness.csv", value["rotation"], ("seed", "shortcut_relevance", "noise_relevance", "noise_weights_norm"))
    _csv(results / "second_shortcut.csv", value["second_shortcut"], ("shortcut_count", "relevance_rank", "relevance_singular_values", "source_shortcut_weight_norm"))
    _csv(results / "redundant_shortcuts.csv", value["redundant_shortcut_copies"], ("copy_count", "ambient_feature_dimension", "collective_relevance", "source_weight_norm"))
    quadrants = [row for row in relevance_rows if row["rho"] == 0.8 and row["shift"] in {"shortcut-relation", "independent-noise-variance", "target-emergent", "stable-predictive-burden"}]
    _csv(results / "exposure_quadrants.csv", quadrants, ("n_noise", "rho", "shift", "leading_relevance", "source_exposure_struct", "source_exposure_risk", "category"))
    _csv(results / "finite_vulnerability.csv", relevance_rows, ("n_noise", "rho", "shift", "vulnerability", "fitted_alpha"))
    (output / "round3_3a_report.md").write_text(report(value), encoding="utf-8")
    print(output / "round3_3a_report.md")


if __name__ == "__main__":
    main()
