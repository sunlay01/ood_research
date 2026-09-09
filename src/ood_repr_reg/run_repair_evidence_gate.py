"""Run the Repair Evidence Gate audit.

This runner is intentionally evidence-only.  By default it reuses the frozen
Task 1 operator snapshot and regenerates Task 2 certificates from that snapshot,
so CMNIST encoders are not retrained unless ``--refresh-task1`` is requested.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .corrected_geometry_snapshot import corrected_snapshot_audit
from .environment_family import (
    build_source_induced_family,
    build_task_geometry,
    family_directional_derivative,
    legacy_gaussian_family,
    metric_operator_summary,
    source_induced_audit,
)
from .environment_family.base import Role
from .environment_family.legacy_gaussian import LegacyGaussianFamily
from .environment_family.source_induced import EnvironmentParameterization, parameter_jacobian
from .round3r_3b_benchmark import ModuleEnvironment
from .round3r_3e_c_spectral import slack_ratio
from .run_algorithm_mechanism import DEFAULT_OUTPUT as TASK1_DEFAULT_OUTPUT
from .run_algorithm_mechanism import run as run_task1
from .run_sharp_optimality import DEFAULT_OUTPUT as TASK2_DEFAULT_OUTPUT
from .run_sharp_optimality import DEFAULT_SNAPSHOTS, run as run_task2
from .sharp_optimality.geometry import response_parts

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "round3_redesign" / "repair_evidence_gate"


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return "inf" if value > 0 else "-inf"
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _read_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, object] | list[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_jsonable), encoding="utf-8")


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
        ).strip()
    except Exception:
        return "unavailable"


def _git_dirty() -> bool:
    try:
        return bool(subprocess.check_output(
            ["git", "status", "--short"], cwd=ROOT, text=True,
        ).strip())
    except Exception:
        return True


def _operator_norm(matrix: np.ndarray) -> float:
    singular = np.linalg.svd(np.asarray(matrix, dtype=float), compute_uv=False)
    return float(singular[0]) if singular.size else 0.0


def _rank(matrix: np.ndarray, tolerance: float = 1e-9) -> int:
    singular = np.linalg.svd(np.asarray(matrix, dtype=float), compute_uv=False)
    if singular.size == 0 or singular[0] <= 0.0:
        return 0
    return int(np.sum(singular > tolerance * singular[0]))


def r1_slack_audit() -> dict[str, object]:
    irreducible = np.array([[1.0], [0.0]])
    scales = []
    for magnitude in (1e-10, 1.0, 1e10):
        result = slack_ratio(irreducible, np.array([[magnitude], [0.0]]))
        scales.append({"magnitude": magnitude, **result})
    compatible = slack_ratio(irreducible, np.array([[0.0], [2.0]]))
    transition_bad = slack_ratio(irreducible, np.array([[1e-10], [1e-10]]))
    passed = (
        all(not row["support_compatible"] and np.isinf(row["rho_slack"]) for row in scales)
        and compatible["support_compatible"]
        and abs(float(compatible["rho_slack"]) - 4.0) < 1e-9
        and not transition_bad["support_compatible"]
    )
    return {
        "status": "REPAIR-PASS" if passed else "REPAIR-FAIL",
        "scale_sweep": scales,
        "compatible_case": compatible,
        "near_tolerance_transition": transition_bad,
    }


def _family_record(config_id: str, role: str, family) -> dict[str, object]:
    geometry = build_task_geometry(family)
    summary = metric_operator_summary(geometry.response, geometry.observation, geometry.spec.metric)
    parts = response_parts(geometry.response, geometry.observation, None, geometry.spec.metric)
    metadata = family.metadata()
    record: dict[str, object] = {
        "config_id": config_id,
        "family_name": metadata.get("family_name", geometry.spec.family_name),
        "benchmark_role": role,
        "family_construction": geometry.spec.coordinate_description,
        "family_dimension": geometry.spec.dimension,
        "direction_names": list(geometry.spec.directions),
        "metric_definition": "declared tangent metric",
        "reference_environment": repr(geometry.reference),
        "source_reference_index": metadata.get("source_reference_index", 0),
        "A_shape": list(geometry.response.shape),
        "O_S_shape": list(geometry.observation.shape),
        "rank_O_S": summary["rank_O"],
        "dim_ker_O_S": summary["kernel_dimension"],
        "rank_A": summary["rank_A"],
        "rank_A_irr": _rank(np.asarray(parts["R"])),
        "information_floor": summary["information_floor"],
        "target_risk_used": bool(metadata.get("target_risk_used", False)),
        "response_operator_used_in_family_construction": bool(metadata.get("response_operator_used", False)),
        "semantic_labels_used": bool(metadata.get("semantic_labels_used", False)),
        "cluster_labels_used": bool(metadata.get("cluster_labels_used", False)),
    }
    if record["family_name"] == "source_induced":
        audit = source_induced_audit(family)
        record.update({
            "retained_modes": list(audit["retained_mode_indices"]),
            "realization_residuals": list(audit["realization_residuals"]),
            "realized_metric_deviation": audit["realized_metric_identity_error"],
            "state_span_rank": audit["state_span_rank"],
            "realizable_rank": audit["realizable_rank"],
            "unrealizable_modes": list(audit["unrealizable_modes"]),
        })
    return record


def r2_family_provenance() -> dict[str, object]:
    records = [
        _family_record("legacy_hidden_u_primary", "hidden-U primary", legacy_gaussian_family()),
        _family_record("source_induced_comparison", "source-induced comparison", build_source_induced_family()),
    ]
    legacy = next(row for row in records if row["config_id"] == "legacy_hidden_u_primary")
    source = next(row for row in records if row["config_id"] == "source_induced_comparison")
    passed = (
        legacy["family_name"] != source["family_name"]
        and legacy["benchmark_role"] == "hidden-U primary"
        and source["benchmark_role"] == "source-induced comparison"
        and int(legacy["family_dimension"]) == 8
        and int(legacy["rank_O_S"]) == 7
        and abs(float(legacy["information_floor"]) - 0.05118145108608892) < 1e-9
        and float(source["information_floor"]) >= -1e-12
        and not any(bool(row[key]) for row in records for key in (
            "target_risk_used", "semantic_labels_used", "cluster_labels_used",
        ))
    )
    return {"status": "REPAIR-PASS" if passed else "REPAIR-FAIL", "records": records}


@dataclass(frozen=True)
class ToyEnvironment:
    value: float


class ToyUnitIntervalFamily:
    def reference_environment(self) -> ToyEnvironment:
        return ToyEnvironment(0.5)

    def perturb(self, reference: ToyEnvironment, coordinate: str, signed_step: float, *, role: Role) -> ToyEnvironment:
        del role
        if coordinate != "x":
            raise ValueError(coordinate)
        value = reference.value + signed_step
        if value < 0.0 or value > 1.0:
            raise ValueError("illegal unit-interval point")
        return ToyEnvironment(value)

    def legal_step_interval(self, reference: ToyEnvironment, coordinate: str, role: Role) -> tuple[float, float]:
        del role
        if coordinate != "x":
            raise ValueError(coordinate)
        return float(reference.value), float(1.0 - reference.value)


def _legacy_zero_variance_boundary_audit() -> dict[str, object]:
    base = ModuleEnvironment(
        shortcut_rhos=(0.75, 0.57),
        shortcut_means=(0.18, 0.08),
        shortcut_variances=(0.0, 0.61),
        n_noise=4,
    )
    family = LegacyGaussianFamily(base=base)
    negative, positive = family.legal_step_interval(base, "S1_variance", role="source")
    derivative, diagnostic = family_directional_derivative(
        family, base, "S1_variance",
        lambda env: np.array([env.shortcut_variances[0]]),
        1e-3, role="source",
    )
    expected = family.tangent_spec(base).scales[family.tangent_spec(base).index("S1_variance")]
    return {
        "case": "legacy_gaussian_zero_variance",
        "negative_radius": negative,
        "positive_radius": positive,
        "scheme": diagnostic.scheme,
        "derivative": float(derivative[0]),
        "expected": float(expected),
        "pass": bool(
            negative == 0.0 and np.isinf(positive)
            and diagnostic.scheme == "forward_second_order"
            and abs(float(derivative[0]) - float(expected)) < 1e-10
        ),
        "diagnostic": diagnostic.as_dict(),
    }


def _source_induced_parameter_boundary_audit() -> dict[str, object]:
    base = ModuleEnvironment(
        shortcut_rhos=(0.75, 0.57),
        shortcut_means=(0.18, 0.08),
        shortcut_variances=(0.0, 0.61),
        n_noise=4,
    )
    parameters = EnvironmentParameterization(len(base.shortcut_rhos), base.n_noise)
    jacobian, diagnostics = parameter_jacobian(base, parameters, 1e-6, return_diagnostics=True)
    s1 = next(row for row in diagnostics if row["parameter"] == "variance_1")
    return {
        "case": "source_induced_parameter_jacobian_zero_variance",
        "jacobian_shape": list(jacobian.shape),
        "variance_1_scheme": s1["scheme"],
        "variance_1_legal_minus": s1["legal_minus"],
        "variance_1_legal_plus": s1["legal_plus"],
        "finite": bool(np.all(np.isfinite(jacobian))),
        "pass": bool(
            np.all(np.isfinite(jacobian))
            and s1["scheme"] == "forward_second_order"
            and not s1["legal_minus"]
            and s1["legal_plus"]
        ),
    }


def r3_boundary_audit() -> dict[str, object]:
    family = ToyUnitIntervalFamily()
    calls: list[float] = []

    def evaluate(env: ToyEnvironment) -> np.ndarray:
        if env.value < 0.0 or env.value > 1.0:
            raise AssertionError("illegal evaluation")
        calls.append(env.value)
        return np.array([env.value ** 2 + 3.0 * env.value])

    rows = []
    expected = {"interior": 4.0, "lower": 3.0, "upper": 5.0}
    for name, reference in (
        ("interior", ToyEnvironment(0.5)),
        ("lower", ToyEnvironment(0.0)),
        ("upper", ToyEnvironment(1.0)),
    ):
        derivative, diagnostic = family_directional_derivative(
            family, reference, "x", evaluate, 0.1, role="source",
        )
        rows.append({
            "case": name,
            "derivative": float(derivative[0]),
            "expected": expected[name],
            "abs_error": abs(float(derivative[0]) - expected[name]),
            **diagnostic.as_dict(),
        })
    legacy = _legacy_zero_variance_boundary_audit()
    source = _source_induced_parameter_boundary_audit()
    passed = (
        all(row["abs_error"] < 1e-10 for row in rows)
        and all(0.0 <= value <= 1.0 for value in calls)
        and legacy["pass"] and source["pass"]
    )
    return {
        "status": "REPAIR-PASS" if passed else "REPAIR-FAIL",
        "rows": rows,
        "evaluations": calls,
        "legacy_zero_variance": legacy,
        "source_induced_parameter_jacobian": source,
    }


def _parse_pytest_counts(output: str) -> dict[str, int]:
    counts = {"tests_passed": 0, "tests_failed": 0, "tests_skipped": 0, "warnings": 0}
    for key, pattern in (
        ("tests_passed", r"(\d+) passed"),
        ("tests_failed", r"(\d+) failed"),
        ("tests_skipped", r"(\d+) skipped"),
        ("warnings", r"(\d+) warnings?"),
    ):
        matches = re.findall(pattern, output)
        if matches:
            counts[key] = int(matches[-1])
    return counts


def _run_recorded_command(name: str, command: str, *, cwd: Path = ROOT) -> dict[str, object]:
    completed = subprocess.run(
        command, cwd=cwd, shell=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    output = completed.stdout or ""
    record: dict[str, object] = {
        "name": name,
        "command": command,
        "cwd": str(cwd),
        "returncode": completed.returncode,
        "passed": completed.returncode == 0,
        "output_tail": output[-4000:],
    }
    if "pytest" in command:
        record.update(_parse_pytest_counts(output))
    if "lake build" in command:
        record["lean_build_passed"] = completed.returncode == 0
    return record


def _copy_rows(source: Path, target: Path, transform: Callable[[dict[str, str]], dict[str, object]] | None = None) -> list[dict[str, object]]:
    rows = _read_csv(source)
    payload = [dict(row) if transform is None else transform(row) for row in rows]
    _write_csv(target, payload)
    return payload


def _key(row: dict[str, object]) -> tuple[str, str, str, str]:
    return (
        str(row.get("setting", "")), str(row.get("method", "")),
        str(row.get("lambda", "")), str(row.get("seed", "")),
    )


def _natural_task2_cases(rows: list[dict[str, object]]) -> dict[str, object]:
    natural = [row for row in rows if row.get("kind") in {"gaussian", "cmnist"}]
    nonzero_inside = []
    violations = []
    for row in natural:
        e_norm = float(row.get("E_operator_norm", 0.0) or 0.0)
        condition = str(row.get("full_minimax_condition", "False")) == "True"
        adaptive_condition = str(row.get("support_compatible", "False")) == "True" and float(row.get("spectral_gap_min", -1.0) or -1.0) >= -1e-8
        if e_norm > 1e-8 and adaptive_condition:
            nonzero_inside.append(row)
        if e_norm > 1e-8 and not adaptive_condition:
            violations.append(row)
        row["full_condition_bool"] = condition
        row["adaptive_condition_bool"] = adaptive_condition
    return {
        "natural_row_count": len(natural),
        "nonzero_E_inside_slack_count": len(nonzero_inside),
        "natural_slack_violation_count": len(violations),
        "nonzero_E_inside_slack_examples": nonzero_inside[:10],
        "natural_slack_violation_examples": violations[:10],
    }


def _write_docs(output: Path, results: Path, summary: dict[str, object]) -> None:
    status = summary["overall_verdict"]
    header = (
        f"repair_baseline_commit: `{summary['repair_baseline_commit']}`\n"
        f"code_base_commit: `{summary['code_base_commit']}`\n"
        f"artifact_commit: `{summary['artifact_commit']}`\n"
        f"working_tree_dirty_when_generated: `{summary['working_tree_dirty_when_generated']}`\n"
        f"task1_runner: `{summary['task1_runner']}`\n"
        f"task2_runner: `{summary['task2_runner']}`\n"
        f"regression_commands_recorded: `{summary['regression_commands_recorded']}`\n"
        f"date: `{summary['date']}`\n"
        f"python_version: `{summary['python_version']}`\n"
        f"config_ids: `{', '.join(summary['config_ids'])}`\n\n"
    )
    docs = {
        "repair_notes.md": header + "# Repair Notes\n\nR1 updates slack support compatibility, R2 records family provenance with distinct config IDs, and R3 audits boundary-safe finite differences. Historical result directories are preserved.\n",
        "family_provenance_audit.md": header + "# Family Provenance Audit\n\nSee `results/family_provenance.json`. Legacy hidden-U and source-induced comparison are rebuilt through the family interface and reported with separate roles.\n",
        "boundary_fd_audit.md": header + "# Boundary FD Audit\n\nSee `results/regression_status.json`. The toy `[0,1]` family checks central and one-sided second-order stencils without illegal evaluations.\n",
        "task1_numerical_evidence.md": header + "# Task 1 Numerical Evidence\n\nSee `results/task1_actual_mechanism.csv`, `results/task1_common_base_attribution.csv`, and `results/task1_summary.json`. Learner-side mechanism rows remain source-only; response residuals are post-hoc.\n",
        "task2_numerical_evidence.md": header + "# Task 2 Numerical Evidence\n\nSee `results/task2_theorem_audit.csv`, `results/task2_spectral_rows.csv`, and `results/task2_natural_cases.json`. Corrected `E=A_rec+PiO` is used throughout.\n",
        "joint_interpretation.md": header + "# Joint Interpretation\n\nTask 1 explains source-side algorithm response. Task 2 checks whether that response is family-conditionally minimax under spectral slack and static steering tax. Neither track states a target-risk lower bound.\n",
        "repair_evidence_report.md": (
            header + "# Repair Evidence Gate\n\n"
            f"Overall verdict: `{status}`. Readiness: `{summary['readiness']}`. "
            f"R1/R2/R3: `{summary['r1_status']}`, `{summary['r2_status']}`, `{summary['r3_status']}`. "
            f"Task 1: `{summary['task1_evidence_status']}`. Task 2: `{summary['task2_evidence_status']}`. "
            "All numbers in this report are read from the JSON/CSV artifacts under `results/`.\n"
        ),
    }
    for name, text in docs.items():
        (output / name).write_text(text, encoding="utf-8")


def run(
    output: Path = DEFAULT_OUTPUT,
    *,
    refresh_task1: bool = True,
    record_regressions: bool = False,
    cmnist: bool = True,
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    results = output / "results"
    results.mkdir(exist_ok=True)
    code_base_commit = _git_commit()
    dirty_at_generation_start = _git_dirty()

    r1 = r1_slack_audit()
    r2 = r2_family_provenance()
    r3 = r3_boundary_audit()
    _write_json(results / "family_provenance.json", r2)

    task1_source = TASK1_DEFAULT_OUTPUT
    if refresh_task1:
        task1_source = output / "task1_regenerated"
        run_task1(task1_source, cmnist=cmnist, seeds=seeds)
    task1_results = task1_source / "results"
    task1_summary = _read_json(task1_results / "summary.json")
    snapshot_path = task1_results / "operator_snapshots.npz"

    task2_output = output / "task2_regenerated"
    task2_summary = run_task2(task2_output, cmnist=cmnist, seeds=seeds, snapshot_path=snapshot_path)
    task2_results = task2_output / "results"

    actual = _copy_rows(task1_results / "exact_reconstruction.csv", results / "task1_actual_mechanism.csv")
    common = _copy_rows(task1_results / "counterfactual_residuals.csv", results / "task1_common_base_attribution.csv")
    residual = _copy_rows(task1_results / "per_mode_rows.csv", results / "task1_task_residual_attribution.csv")
    theorem = _copy_rows(task2_results / "theorem_audit.csv", results / "task2_theorem_audit.csv")
    spectral = _copy_rows(task2_results / "spectral_rows.csv", results / "task2_spectral_rows.csv")

    natural = _natural_task2_cases(theorem)
    _write_json(results / "task2_natural_cases.json", natural)
    _write_json(results / "task1_summary.json", task1_summary)
    _write_json(results / "task2_summary.json", task2_summary)

    task2_by_key = {_key(row): row for row in theorem}
    joint = []
    for row in actual:
        partner = task2_by_key.get(_key(row))
        if partner is None:
            continue
        joint.append({
            "setting": row.get("setting"), "method": row.get("method"),
            "lambda": row.get("lambda"), "seed": row.get("seed"),
            "task1_E_norm": row.get("E_norm"),
            "task1_pi_error": row.get("pi_reconstruction_relative_error"),
            "task2_information_floor": partner.get("information_floor"),
            "task2_E_operator_norm": partner.get("E_operator_norm"),
            "task2_slack_ratio": partner.get("slack_ratio"),
            "task2_total_regret": partner.get("total_regret"),
            "task2_total_excess": partner.get("total_excess"),
        })
    _write_csv(results / "task1_task2_joint.csv", joint)

    snapshot_probe = corrected_snapshot_audit(np.eye(2), np.eye(2), np.zeros((2, 2)))
    regression_status = {
        "r1": r1,
        "r2_status": r2["status"],
        "r3": r3,
        "snapshot_endpoint_probe_pass": snapshot_probe["passes"],
        "task1_rows": len(actual),
        "task2_rows": len(theorem),
        "joint_rows": len(joint),
        "recorded_commands": [],
    }

    if record_regressions:
        commands = [
            ("repair_evidence_gate", "PYTHONPATH=src pytest -q tests/test_repair_evidence_gate.py"),
            ("task1_task2", "PYTHONPATH=src pytest -q tests/test_algorithm_mechanism.py tests/test_sharp_optimality.py"),
            ("full_repository", "PYTHONPATH=src pytest -q"),
            ("lean_build", "source ../scripts/lean_env.sh && lake build"),
        ]
        for name, command in commands:
            cwd = ROOT / "formalization" if name == "lean_build" else ROOT
            regression_status["recorded_commands"].append(_run_recorded_command(name, command, cwd=cwd))
    _write_json(results / "regression_status.json", regression_status)

    task1_ok = (
        task1_summary.get("verdict") == "ALGORITHM-MECHANISM-PASS"
        and float(task1_summary.get("max_pi_reconstruction_relative_error", 1.0)) < 2e-4
        and bool(task1_summary.get("corrected_geometry_snapshots_pass", False))
        and bool(task1_summary.get("stable_task_residual_attributions", []))
    )
    task2_correct = (
        task2_summary.get("verdict") == "SHARP-OPTIMALITY-PASS"
        and bool(task2_summary.get("corrected_snapshot_pass", False))
        and bool(task2_summary.get("cross_operator_audit_pass", False))
        and bool(task2_summary.get("counterexamples_pass", False))
    )
    task2_support = task2_correct and int(natural["nonzero_E_inside_slack_count"]) > 0
    r_pass = all(item == "REPAIR-PASS" for item in (r1["status"], r2["status"], r3["status"]))
    task1_status = "TASK1-EVIDENCE-SUPPORT" if task1_ok else "TASK1-EVIDENCE-PARTIAL"
    task2_status = "TASK2-EVIDENCE-SUPPORT" if task2_support else ("TASK2-EVIDENCE-PARTIAL" if task2_correct else "TASK2-EVIDENCE-FAIL")
    regressions_ok = (
        not record_regressions
        or all(bool(row["passed"]) for row in regression_status["recorded_commands"])
    )
    if r_pass and task1_ok and task2_support and regressions_ok:
        overall = "REPAIR-EVIDENCE-PASS"
    elif r1["status"] == "REPAIR-FAIL" or r2["status"] == "REPAIR-FAIL" or r3["status"] == "REPAIR-FAIL" or task2_status == "TASK2-EVIDENCE-FAIL" or not regressions_ok:
        overall = "REPAIR-EVIDENCE-FAIL"
    else:
        overall = "REPAIR-EVIDENCE-PARTIAL"

    summary = {
        "repair_baseline_commit": "da48363055869cc6af72c6122b733cbb2dedada8",
        "code_base_commit": code_base_commit,
        "artifact_commit": "see git log -- round3_redesign/repair_evidence_gate/results/repair_status.json",
        "working_tree_dirty_when_generated": dirty_at_generation_start,
        "git_dirty_after_generation": _git_dirty(),
        "task1_runner": "ood_repr_reg.run_algorithm_mechanism",
        "task2_runner": "ood_repr_reg.run_sharp_optimality",
        "task1_refreshed_post_repair": bool(refresh_task1),
        "regression_commands_recorded": bool(record_regressions),
        "date": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "config_ids": ["legacy_hidden_u_primary", "source_induced_comparison"],
        "r1_status": r1["status"],
        "r2_status": r2["status"],
        "r3_status": r3["status"],
        "task1_evidence_status": task1_status,
        "task2_evidence_status": task2_status,
        "overall_verdict": overall,
        "readiness": "READY-FOR-TASK-3" if overall == "REPAIR-EVIDENCE-PASS" else "NOT-READY-FOR-TASK-3",
        "task1_summary_source": str(task1_results / "summary.json"),
        "task2_regenerated_output": str(task2_output),
        "regression_status_source": str(results / "regression_status.json"),
        "task1_row_count": len(actual),
        "task2_row_count": len(theorem),
        "task1_task2_joint_row_count": len(joint),
        "natural_nonzero_E_inside_slack_count": natural["nonzero_E_inside_slack_count"],
        "target_risk_lower_bound_claimed": False,
        "semantic_or_cluster_used": False,
        "finite_sample_guarantee_claimed": False,
    }
    _write_json(results / "repair_status.json", summary)
    _write_docs(output, results, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--reuse-task1", dest="refresh_task1", action="store_false")
    parser.add_argument("--refresh-task1", dest="refresh_task1", action="store_true")
    parser.set_defaults(refresh_task1=True)
    parser.add_argument("--record-regressions", action="store_true")
    parser.add_argument("--no-cmnist", action="store_true")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    args = parser.parse_args()
    seeds = tuple(int(value) for value in args.seeds.split(",") if value)
    print(json.dumps(
        run(
            args.output, refresh_task1=args.refresh_task1,
            record_regressions=args.record_regressions,
            cmnist=not args.no_cmnist, seeds=seeds,
        ),
        indent=2, default=_jsonable,
    ))


if __name__ == "__main__":
    main()
