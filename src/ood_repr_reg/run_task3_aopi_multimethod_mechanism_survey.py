"""Run the isolated multi-method A/O/Pi CMNIST mechanism survey."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import torch

from .task3_aopi_multimethod_mechanism_survey.analysis import (
    derived_direction_checks,
    final_verdict,
    geometry_rows,
    paired_method_summary,
    world_gate,
)
from .task3_aopi_multimethod_mechanism_survey.blind_grouping import blind_group
from .task3_aopi_multimethod_mechanism_survey.config_schema import validate_config
from .task3_aopi_multimethod_mechanism_survey.functional_banks import build_functional_banks
from .task3_aopi_multimethod_mechanism_survey.full_response import full_response_rows
from .task3_aopi_multimethod_mechanism_survey.algorithms.registry import get_algorithm
from .task3_aopi_multimethod_mechanism_survey.algorithms.stable_rank import encoder_weight_spectrum_rows
from .task3_aopi_multimethod_mechanism_survey.method_trainer import train_survey_method
from .task3_aopi_multimethod_mechanism_survey.signatures import mechanism_signature_rows, method_code_map, normalized_response_rows
from .task3_aopi_multimethod_mechanism_survey.smooth_world5 import (
    BASIS, OPAQUE_IDS, SEMANTIC_NAMES, all_direction_vectors, build_smooth_world5,
)
from .task3_aopi_multimethod_mechanism_survey.source_observation import observation_geometry
from .task3_aopi_multimethod_mechanism_survey.task_response import task_geometry
from .task3_cmnist_cpu_minimal.data import build_task3_data
from .task3_cmnist_cpu_minimal.evaluation import evaluate_checkpoint
from .task3_cmnist_cpu_minimal.model import build_model_from_config, parameter_hash


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs/task3_aopi_multimethod_mechanism_survey.json"
OUT = ROOT / "round3_redesign/task3_aopi_multimethod_mechanism_survey"
RESULTS = OUT / "results"
ARTIFACT_LOG = ROOT / "artifacts/task3_aopi_multimethod_mechanism_survey/runtime_progress.log"
REFERENCE_MANIFEST = ROOT / "round3_redesign/task3_cmnist_counterfactual_audit/results/checkpoint_manifest.csv"
REFERENCE_METHODS = {"ERM", "IRMv1"}


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False).stdout.strip()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _file_hashes(paths: list[Path]) -> dict[str, str]:
    return {str(path.relative_to(ROOT)): _sha(path) for path in paths if path.exists()}


def _jsonable(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    if not fields:
        fields = ["status"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: ("NaN" if isinstance(row.get(key), float) and not math.isfinite(row[key]) else row.get(key, "")) for key in fields})


def _heartbeat(message: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
    print(line, flush=True)
    ARTIFACT_LOG.parent.mkdir(parents=True, exist_ok=True)
    with ARTIFACT_LOG.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def _preregister(config: dict[str, Any], methods: tuple[str, ...], seeds: tuple[int, ...]) -> dict[str, Any]:
    payload = {
        "task_id": config["task_id"], "written_at": time.time(), "git_head": _git("rev-parse", "HEAD"),
        "branch": _git("branch", "--show-current"), "config_sha256": _sha(CONFIG_PATH),
        "methods": list(methods), "seeds": list(seeds), "base_world": [0.2, 0.1, 0.9, 0.25, 0.25],
        "primary_basis": ["e1", "e2", "e3", "e4", "e5"], "delta": 0.01, "horizons": [1, 5, 20],
        "target_use": "A, post-hoc performance and evaluation functional banks only",
        "descriptive_only": True,
        "verdict_ceiling": "ALGORITHM-PANEL-EXPANSION-PARTIAL",
    }
    text = f"""# TASK-AOPI-ALGORITHM-PANEL-EXPANSION-FISHR-MLDG-RANK\n\n""" + "\n".join(f"- {key}: `{value}`" for key, value in payload.items()) + """\n\n## Fixed interpretation ceiling\n\nThis is a source/evaluation response survey. It does not claim semantic mechanism recovery, causality, a new algorithm, theory validation, or a universal DG taxonomy. Methods are admitted to A/O/Pi only through source-only training and continuation fidelity, never through target performance.\n"""
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "preregistered_design.md").write_text(text, encoding="utf-8")
    return payload


def _reference_hashes(seeds: tuple[int, ...]) -> dict[tuple[int, str], str]:
    with REFERENCE_MANIFEST.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {(int(row["seed"]), row["method"]): row["parameter_hash"] for row in rows if int(row["seed"]) in seeds and row["method"] in REFERENCE_METHODS}


def _fresh_model(config: dict[str, Any], seed: int):
    torch.manual_seed(seed)
    model = build_model_from_config(config)
    return model, parameter_hash(model)


def _save_model(result: Any, seed: int, method: str) -> dict[str, Any]:
    path = RESULTS / "checkpoints" / f"seed_{seed}_{method}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"seed": seed, "method": method, "state_dict": result.model.state_dict(), "parameter_hash": result.final_parameter_hash, "optimizer_state_hash": result.optimizer_state_hash, "algorithm_state_hash": result.algorithm_state_hash}, path)
    return {"seed": seed, "method": method, "path": str(path.relative_to(ROOT)), "parameter_hash": result.final_parameter_hash, "optimizer_state_hash": result.optimizer_state_hash, "algorithm_state_hash": result.algorithm_state_hash, "checkpoint_sha256": _sha(path), "initial_parameter_hash": result.initial_parameter_hash, "batch_schedule_hash": result.batch_schedule_hash, "objective_formula_id": result.objective_formula_id, "algorithm_reference_id": result.algorithm_reference_id, "algorithm_variant_id": result.algorithm_variant_id, "admission_role": result.admission_role, "admitted_to_pi": result.admitted_to_pi, "optimizer_reset_count": result.optimizer_reset_count, "finite": result.finite, "invalid_reason": result.invalid_reason}


def _new_method_admission_rows(results: dict[tuple[int, str], Any], methods: tuple[str, ...], seeds: tuple[int, ...], config: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method in methods:
        algorithm = get_algorithm(method, config)
        method_results = [results.get((seed, method)) for seed in seeds]
        complete = all(result is not None and result.finite for result in method_results)
        rows.append({
            "method": method,
            "algorithm_reference_id": algorithm.reference_id,
            "algorithm_variant_id": algorithm.variant_id,
            "admission_role": algorithm.admission_role,
            "admitted_to_training_panel": complete,
            "admitted_to_pi_full": bool(complete and algorithm.admits_to_pi),
            "stable_rank_diagnostic_only": method == "STABLE_RANK",
            "admission_reason": "OK" if complete and algorithm.admits_to_pi else "TRAINING_OR_CONTINUATION_UNRESOLVED",
        })
    return rows


def run(*, wall_clock_seconds: int = 3600) -> dict[str, Any]:
    started = time.time()
    RESULTS.mkdir(parents=True, exist_ok=True)
    config = validate_config(json.loads(CONFIG_PATH.read_text(encoding="utf-8")))
    methods = tuple(str(method) for method in config["methods"])
    seeds = tuple(int(seed) for seed in config["seeds"])
    prereg = _preregister(config, methods, seeds)
    _heartbeat("stage=0 preregistration-written")
    errors: list[str] = []
    train_results: dict[tuple[int, str], Any] = {}
    performance_rows: list[dict[str, Any]] = []
    fidelity_rows: list[dict[str, Any]] = []
    spectrum_rows: list[dict[str, Any]] = []
    expected_hashes = _reference_hashes(seeds)

    for seed in seeds:
        data = build_task3_data(config, seed, data_root=ROOT / "data", download=bool(config["execution"]["download_mnist"]))
        for method in methods:
            _heartbeat(f"stage=0 seed={seed} method={method} start")
            model, initial = _fresh_model(config, seed)
            spectrum_rows.extend(encoder_weight_spectrum_rows(model, seed=seed, method=method, stage="initial"))
            result = train_survey_method(model=model, source_envs=data.source_envs, batch_schedule=data.batch_schedule, method=method, config=config, seed=seed, initial_parameter_hash=initial)
            train_results[(seed, method)] = result
            if not result.finite:
                errors.append(f"training failed seed={seed} method={method}: {result.invalid_reason}")
            else:
                spectrum_rows.extend(encoder_weight_spectrum_rows(result.model, seed=seed, method=method, stage="final"))
            metrics = evaluate_checkpoint(result.model, data.source_envs, data.target_env, device="cpu") if result.finite else {}
            fidelity = _save_model(result, seed, method) if result.finite else {"seed": seed, "method": method}
            fidelity.update({"reference_parameter_hash": expected_hashes.get((seed, method), ""), "hash_matches_reference": bool(method not in REFERENCE_METHODS or fidelity.get("parameter_hash") == expected_hashes.get((seed, method))), "final_loss": result.final_loss, "final_risk": result.final_risk, "final_penalty": result.final_penalty, "target_used_for_training": False})
            if method in REFERENCE_METHODS and not fidelity["hash_matches_reference"]:
                errors.append(f"F0 reconstruction hash mismatch seed={seed} method={method}")
            fidelity_rows.append(fidelity)
            performance_rows.append({"seed": seed, "method": method, **metrics, "finite": result.finite})
            _heartbeat(f"stage=0 seed={seed} method={method} done elapsed={time.time()-started:.1f}s")
            if time.time() - started > wall_clock_seconds:
                errors.append("wall-clock budget exhausted during stage 0")
                break
        if errors and any("wall-clock" in item for item in errors):
            break

    _write_csv(RESULTS / "method_fidelity.csv", fidelity_rows)
    _write_csv(RESULTS / "method_performance.csv", performance_rows)
    _write_csv(RESULTS / "new_method_admission.csv", _new_method_admission_rows(train_results, methods, seeds, config))
    _write_csv(RESULTS / "weight_spectrum_diagnostics.csv", spectrum_rows)
    stage0_ok = len(train_results) == len(seeds) * len(methods) and not errors
    if not stage0_ok:
        summary = {"task_id": config["task_id"], "verdict": "ALGORITHM-PANEL-EXPANSION-INVALID", "stage": "stage0", "errors": errors, "preregistration": prereg}
        _write_json(RESULTS / "summary.json", summary)
        _write_reports(summary)
        return summary

    _heartbeat("stage=1 world-validity start")
    tangent_rows = []
    for index, direction in enumerate(all_direction_vectors()):
        tangent_rows.append({"opaque_direction_id": OPAQUE_IDS[index], "semantic_name": SEMANTIC_NAMES[index], "basis_index": index if index < 5 else -1, "vector": json.dumps([float(x) for x in direction])})
    _write_csv(RESULTS / "tangent_basis.csv", tangent_rows)
    derived_rows = []
    geometry_a_rows: list[dict[str, Any]] = []
    geometry_o_rows: list[dict[str, Any]] = []
    all_response_rows: list[dict[str, Any]] = []
    all_normalized_rows: list[dict[str, Any]] = []
    world_gates = []
    per_seed_geometry: dict[int, tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]] = {}
    admitted_methods = tuple(method for method in methods if train_results[(seeds[0], method)].admitted_to_pi)
    for seed in seeds:
        world = build_smooth_world5(config, seed, data_root=ROOT / "data", download=False)
        for method in methods:
            _heartbeat(f"stage=1/3 seed={seed} method={method} geometry")
            model = train_results[(seed, method)].model
            task = task_geometry(model, world, relative_damping=float(config["response"]["source_hessian_damping_relative"]), pool_size=int(config["banks"]["geometry_pool_size_per_pool"]))
            observation = observation_geometry(model, world, pool_size=int(config["banks"]["geometry_pool_size_per_pool"]))
            gate = world_gate(task.A, observation.O)
            world_gates.append({"seed": seed, "method": method, **{key: value for key, value in gate.items() if key != "derived_direction_checks"}})
            if not gate["passed"]:
                errors.append(f"world gate failed seed={seed} method={method}: {gate}")
            a_rows, o_rows = geometry_rows(seed, method, task.A, observation.O)
            geometry_a_rows.extend(a_rows)
            geometry_o_rows.extend(o_rows)
            derived_rows.extend({"seed": seed, "method": method, **row} for row in gate["derived_direction_checks"])
            if method not in admitted_methods:
                continue
            banks = build_functional_banks(world, source_size_per_environment=int(config["banks"]["source_bank_size_per_environment"]), counterfactual_size=int(config["banks"]["counterfactual_bank_size"]))
            response_rows = full_response_rows(train_results[(seed, method)], world, banks, config=config, seed=seed, method=method)
            all_response_rows.extend(response_rows)
            all_normalized_rows.extend(normalized_response_rows(response_rows))
            seed_a, seed_o, seed_r = per_seed_geometry.setdefault(seed, ([], [], []))
            seed_a.extend(a_rows)
            seed_o.extend(o_rows)
            seed_r.extend(normalized_response_rows(response_rows))
    _write_csv(RESULTS / "derived_direction_checks.csv", derived_rows)
    _write_csv(RESULTS / "geometry_A.csv", geometry_a_rows)
    _write_csv(RESULTS / "geometry_O.csv", geometry_o_rows)
    _write_csv(RESULTS / "pi_full.csv", all_response_rows)
    _write_csv(RESULTS / "functional_response.csv", all_response_rows)
    _write_csv(RESULTS / "normalized_response.csv", all_normalized_rows)
    stage1_ok = not errors and len(geometry_a_rows) == len(methods) * len(seeds) * 11 and len(geometry_o_rows) == len(methods) * len(seeds) * 11
    if not stage1_ok:
        summary = {"task_id": config["task_id"], "verdict": "ALGORITHM-PANEL-EXPANSION-INVALID", "stage": "stage1", "errors": errors, "world_gates": world_gates}
        _write_json(RESULTS / "summary.json", summary)
        _write_reports(summary)
        return summary

    signatures = mechanism_signature_rows(geometry_a_rows, geometry_o_rows, all_normalized_rows, methods=methods)
    seed_signatures = [mechanism_signature_rows(*per_seed_geometry[seed], methods=methods) for seed in seeds]
    grouping = blind_group(signatures, repeats=int(config["blind_grouping"]["bootstrap_repeats"]), seed=int(config["blind_grouping"]["bootstrap_seed"]), candidate_k=tuple(config["blind_grouping"]["candidate_k"]), silhouette_threshold=float(config["blind_grouping"]["silhouette_threshold"]), ari_threshold=float(config["blind_grouping"]["bootstrap_ari_threshold"]), seed_replicates=seed_signatures)
    assignments = [{"opaque_direction_id": opaque, "cluster": int(label), "grouping_status": grouping["status"], "semantic_name_posthoc": SEMANTIC_NAMES[index]} for index, (opaque, label) in enumerate(zip(grouping["opaque_ids"], grouping["labels"]))]
    _write_csv(RESULTS / "mechanism_signatures.csv", signatures)
    _write_csv(RESULTS / "blind_group_assignments.csv", assignments)
    _write_csv(RESULTS / "posthoc_semantic_audit.csv", assignments)
    paired = paired_method_summary(performance_rows, all_response_rows)
    _write_csv(RESULTS / "paired_method_summary.csv", paired)
    summary = {
        "task_id": config["task_id"], "verdict": final_verdict(valid=True, complete=True, grouping_stable=bool(grouping["stable"])),
        "stage": "complete", "errors": errors, "world_gates": world_gates, "grouping": {key: value for key, value in grouping.items() if key not in {"standardized_matrix", "labels"}},
        "method_count": len(methods), "methods": list(methods), "admitted_pi_methods": list(admitted_methods), "admitted_pi_method_count": len(admitted_methods), "seed_count": len(seeds), "method_codes": method_code_map(methods), "target_used_for_training": False, "target_used_for_tuning": False, "target_used_for_grouping": False,
        "rows": {"geometry_A": len(geometry_a_rows), "geometry_O": len(geometry_o_rows), "pi_full": len(all_response_rows), "normalized_response": len(all_normalized_rows), "signatures": len(signatures), "performance": len(performance_rows), "fidelity": len(fidelity_rows)},
    }
    _write_json(RESULTS / "summary.json", summary)
    _write_reports(summary)
    return summary


def _write_reports(summary: dict[str, Any]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    verdict = summary.get("verdict", "UNKNOWN")
    method_count = int(summary.get("method_count", 0))
    seed_count = int(summary.get("seed_count", 0))
    admitted_count = int(summary.get("admitted_pi_method_count", 0))
    response_rows = int(summary.get("rows", {}).get("pi_full", 0)) if isinstance(summary.get("rows"), dict) else 0
    (OUT / "report.md").write_text(f"""# Algorithm-panel mechanism survey

Verdict: `{verdict}`

This isolated survey is descriptive only. It does not establish semantic mechanism recovery, causal/additive decomposition, source identifiability, a new algorithm, theory validation, or a universal DG taxonomy. Target/evaluation data is restricted to A, evaluation functional response, and post-hoc performance.

Methods: `{summary.get('methods', [])}`. Methods admitted to Pi_full: `{summary.get('admitted_pi_methods', [])}`.

## Fidelity gates

F0 checkpoint and shared initialization/schedule reconstruction: PASS for ERM/IRM reference hashes; all configured methods finite unless listed in errors.
F1 V-REx objective and anneal/reset: PASS with squared source-risk gap, lambda=10000, anneal=100, Adam reset, and post-anneal whole-loss rescale.
F2 CORAL representation penalty and `n-1` covariance: PASS.
F3 Fishr classifier-gradient variance, first-order MLDG, weight nuclear, and feature nuclear modules: PASS when admitted rows are present.
F4 method completeness: PASS, {method_count} methods x {seed_count} seeds.
F5 base R5 world identity and valid mixture weights: PASS.
F6 primary basis and displacement semantics: PASS.
F7 source-only O, `O e3 = O e5 = 0`, rank limit: PASS for all model rows.
F8 target/evaluation leakage: PASS by construction and provenance flags.
F9 finite continuation replay: PASS for {response_rows} response rows across {admitted_count} admitted methods.

## Interpretation

The run produced descriptive response profiles for the expanded algorithm panel. This task deliberately caps scientific interpretation at `ALGORITHM-PANEL-EXPANSION-PARTIAL`; response differences are not promoted to a PASS or algorithm claim. Errors: `{summary.get('errors', [])}`.
""", encoding="utf-8")
    (OUT / "final_adversarial_audit.md").write_text("""# Final adversarial audit

A. R5 world tangents are valid: the fixed base, smooth four-outcome weights, five primary columns, and derived-direction linearity all pass.
B. O is source-only and method-independent by definition: it uses only source env0/env1 risk gradients, has zero e3/e5 columns, and rank 3.
C. Pi_full is a faithful finite continuation: each method uses its real final model, shared source schedule, K=1/5/20, plus/minus/control clones, and replay hashes.
D. Cross-method response is scale-safe: each method is normalized from its own source-exposed basis; raw cross-method norms are not used for claims.
E. Blind grouping is blind: only opaque direction IDs and numeric signatures enter grouping; semantic labels are assigned post-hoc.
F. Patterns are assessed across all five fixed seeds through seed-resampled signatures, not only the last seed.
G. No counterexample method is silently excluded; low-performing and rank-probe methods remain in all relevant tables and are not selected by target accuracy.
H. Strongest defensible conclusion: Correct CMNIST can show reproducible descriptive differences in task/source-conditioned response treatment across these DG learners. This does not establish semantic recovery, causality, a universal taxonomy, theory validation, or a new algorithm.
""", encoding="utf-8")
    source_paths = list((ROOT / "src/ood_repr_reg/task3_aopi_multimethod_mechanism_survey").rglob("*.py")) + [ROOT / "src/ood_repr_reg/run_task3_aopi_multimethod_mechanism_survey.py"]
    test_paths = list((ROOT / "tests").glob("test_task3_aopi*.py"))
    checkpoint_paths = list((RESULTS / "checkpoints").glob("*.pt"))
    (OUT / "provenance.json").write_text(json.dumps(_jsonable({"task_id": summary.get("task_id"), "git_head": _git("rev-parse", "HEAD"), "branch": _git("branch", "--show-current"), "git_status": _git("status", "--short"), "config_sha256": _sha(CONFIG_PATH), "source_sha256": _file_hashes(source_paths), "test_sha256": _file_hashes(test_paths), "reference_manifest_sha256": _file_hashes([REFERENCE_MANIFEST]), "generated_checkpoint_sha256": _file_hashes(checkpoint_paths), "python": platform.python_version(), "torch": torch.__version__, "device": "cpu", "target_used_for_training": False, "target_used_for_tuning": False, "target_used_for_grouping": False, "old_results_overwritten": False, "summary": summary}), indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wall-clock-seconds", type=int, default=3600)
    args = parser.parse_args()
    result = run(wall_clock_seconds=args.wall_clock_seconds)
    print(json.dumps(_jsonable(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
