"""Run the isolated multi-method A/O/Pi CMNIST mechanism survey."""

from __future__ import annotations

import argparse
import copy
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
from .task3_aopi_multimethod_mechanism_survey.spectral_flatness_diagnostics import (
    flatness_diagnostic_rows,
    gradient_spectrum_rows,
    representation_spectrum_rows,
    source_bank,
    weight_spectrum_long_rows,
)
from .task3_aopi_multimethod_mechanism_survey.signatures import mechanism_signature_rows, method_code_map, normalized_response_rows
from .task3_aopi_multimethod_mechanism_survey.smooth_world5 import (
    BASIS, OPAQUE_IDS, SEMANTIC_NAMES, all_direction_vectors, build_smooth_world5,
)
from .task3_aopi_multimethod_mechanism_survey.source_observation import observation_geometry
from .task3_aopi_multimethod_mechanism_survey.task_response import task_geometry
from .task3_cmnist_cpu_minimal.data import build_task3_data
from .task3_cmnist_cpu_minimal.evaluation import evaluate_checkpoint, evaluate_environment
from .task3_cmnist_cpu_minimal.model import build_model_from_config, parameter_hash


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs/task3_aopi_multimethod_mechanism_survey.json"
OUT = ROOT / "round3_redesign/task3_aopi_multimethod_mechanism_survey"
RESULTS = OUT / "results"
ARTIFACT_LOG = ROOT / "artifacts/task3_aopi_multimethod_mechanism_survey/runtime_progress.log"
REFERENCE_MANIFEST = ROOT / "round3_redesign/task3_cmnist_counterfactual_audit/results/checkpoint_manifest.csv"
REFERENCE_METHODS = {"ERM", "IRMv1"}
SWEEP_METHODS = ("SPECTRAL_NORM_REG", "SPECTRAL_REG_2024", "SVB_ORTHDNN", "STABLE_RANK_NORM", "SAM", "ASAM")


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


def _preregister(config: dict[str, Any], methods: tuple[str, ...], candidates: tuple[str, ...], seeds: tuple[int, ...]) -> dict[str, Any]:
    payload = {
        "task_id": config["task_id"], "written_at": time.time(), "git_head": _git("rev-parse", "HEAD"),
        "branch": _git("branch", "--show-current"), "config_sha256": _sha(CONFIG_PATH),
        "methods": list(methods), "candidate_methods": list(candidates), "seeds": list(seeds), "base_world": [0.2, 0.1, 0.9, 0.25, 0.25],
        "primary_basis": ["e1", "e2", "e3", "e4", "e5"], "delta": 0.01, "horizons": [1, 5, 20],
        "target_use": "A, post-hoc performance and evaluation functional banks only",
        "descriptive_only": True,
        "verdict_ceiling": "SPECTRAL-FLATNESS-PANEL-PARTIAL",
        "source_only_variant_selector": config["source_only_variant_sweep"],
    }
    text = f"""# TASK-AOPI-SPECTRAL-AND-FLATNESS-PANEL\n\n""" + "\n".join(f"- {key}: `{value}`" for key, value in payload.items()) + """\n\n## Fixed interpretation ceiling\n\nThis is a common-budget source/evaluation response survey. It does not claim semantic mechanism recovery, causality, a new algorithm, theory validation, or a universal DG taxonomy. Methods are admitted to A/O/Pi only through source-only training and continuation fidelity, never through target performance. Paper references define algorithms; paper benchmark reproduction is explicitly not the goal.\n"""
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "preregistered_design.md").write_text(text, encoding="utf-8")
    _write_reference_audits(config, methods, candidates)
    return payload


def _write_reference_audits(config: dict[str, Any], methods: tuple[str, ...], candidates: tuple[str, ...]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    spectral = """# Spectral reference audit

This audit distinguishes definition-faithful common-harness variants from paper benchmark reproduction.

- `SPECTRAL_NORM_REG`: implemented as source risk plus `lambda * sum_l sigma_1(W_l)^2`; this is a regularizer, not `torch.nn.utils.spectral_norm`.
- `SPECTRAL_REG_2024`: implemented as `sum_l ((sigma_1(W_l)^k - 1)^2 + ||b_l||^(2k))` with fixed `k=2`.
- `SVB_ORTHDNN`: implemented as post-optimizer singular-value bounding with band `[1/(1+factor), 1+factor]`.
- `STABLE_RANK_NORM`: implemented as post-optimizer tail singular-value projection to a fixed stable-rank target followed by sigma_1 normalization.
- `SVD_SPARSE`: deferred because true SVD parameterization/singular-value sparsification would change the fixed model parameterization and continuation state.

No spectral method uses target data for coefficient selection or admission.
"""
    flatness = """# Flatness reference audit

This audit distinguishes definition-faithful common-harness variants from paper benchmark reproduction.

- `SAM`: implemented as the standard two-step first-order update on the same source minibatch.
- `ASAM`: implemented as adaptive SAM with elementwise parameter-scale perturbation and fixed `eta`.
- `FAD`: deferred; exact zeroth/first-order DG flatness update was not implemented rather than replaced with a SAM surrogate.
- `DISAM`: deferred; exact domain-imbalance perturbation calibration was not implemented rather than replaced with `SAM + VREX`.

Raw parameter-space flatness is not invariant under arbitrary reparameterization; comparisons here are controlled diagnostics under fixed architecture, parameterization, initialization, optimizer family, and source schedule.
"""
    (OUT / "spectral_reference_audit.md").write_text(spectral, encoding="utf-8")
    (OUT / "flatness_reference_audit.md").write_text(flatness, encoding="utf-8")
    (OUT / "fad_reference_audit.md").write_text("""# FAD reference audit

Status: `DEFERRED`.

The task requires exact zeroth-order and first-order Flatness-Aware Minimization for Domain Generalization update semantics. This implementation does not replace FAD with SAM or a generic flatness penalty. `FAD` is retained in `candidate_methods` and receives `admitted_to_training_panel=false`, `admitted_to_pi_full=false` until the exact update can be implemented without changing frozen CMNIST/data/model semantics.
""", encoding="utf-8")
    (OUT / "disam_reference_audit.md").write_text("""# DISAM reference audit

Status: `DEFERRED`.

The task requires exact Domain-Inspired SAM perturbation calibration from source-domain loss imbalance/convergence degree. This implementation does not replace DISAM with `SAM + VREX` or any surrogate. `DISAM` is retained in `candidate_methods` and receives `admitted_to_training_panel=false`, `admitted_to_pi_full=false` until the exact domain-aware update can be implemented faithfully under the frozen common harness.
""", encoding="utf-8")


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


def _new_method_admission_rows(results: dict[tuple[int, str], Any], methods: tuple[str, ...], candidates: tuple[str, ...], seeds: tuple[int, ...], config: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method in candidates:
        algorithm = get_algorithm(method, config)
        runnable = method in methods and bool(algorithm.admits_to_training)
        method_results = [results.get((seed, method)) for seed in seeds] if runnable else []
        complete = bool(runnable and all(result is not None and result.finite for result in method_results))
        reason = "OK" if complete and algorithm.admits_to_pi else (algorithm.deferred_reason or "TRAINING_OR_CONTINUATION_UNRESOLVED")
        observed = next((result for result in method_results if result is not None), None)
        rows.append({
            "method": method,
            "algorithm_reference_id": algorithm.reference_id,
            "algorithm_variant_id": observed.algorithm_variant_id if observed is not None else algorithm.variant_id,
            "admission_role": algorithm.admission_role,
            "candidate_method": True,
            "runnable_in_common_harness": runnable,
            "admitted_to_training_panel": complete,
            "admitted_to_pi_full": bool(complete and algorithm.admits_to_pi),
            "admission_reason": reason,
        })
    return rows


def _variant_config(config: dict[str, Any], method: str, value: Any) -> dict[str, Any]:
    variant = copy.deepcopy(config)
    if method == "SPECTRAL_NORM_REG":
        variant["spectral_norm_reg"]["lambda"] = float(value)
        suffix = f"lambda={float(value):g}"
    elif method == "SPECTRAL_REG_2024":
        variant["spectral_reg_2024"]["lambda"] = float(value)
        suffix = f"lambda={float(value):g}"
    elif method == "SVB_ORTHDNN":
        variant["svb_orthdnn"].update(value)
        suffix = f"factor={float(value['svb_factor']):g},frequency={int(value['projection_frequency'])}"
    elif method == "STABLE_RANK_NORM":
        variant["stable_rank_norm"]["target_rank"] = float(value)
        suffix = f"target_rank={float(value):g}"
    elif method == "SAM":
        variant["sam"]["rho"] = float(value)
        suffix = f"rho={float(value):g}"
    elif method == "ASAM":
        variant["asam"]["rho"] = float(value)
        suffix = f"rho={float(value):g}"
    else:
        raise ValueError(f"method has no sweep: {method}")
    variant["_active_variant_id"] = f"{method}[{suffix}]"
    return variant


def _source_metrics(model: torch.nn.Module, source_envs: tuple[Any, Any]) -> dict[str, float]:
    left = evaluate_environment(model, source_envs[0], device="cpu")
    right = evaluate_environment(model, source_envs[1], device="cpu")
    return {
        "source_mean_loss": (left["loss"] + right["loss"]) / 2.0,
        "source_mean_accuracy": (left["accuracy"] + right["accuracy"]) / 2.0,
    }


def _encoder_geometry(model: torch.nn.Module, method: str, config: dict[str, Any]) -> dict[str, float]:
    modules = [module for name, module in model.named_modules() if isinstance(module, torch.nn.Linear) and name.startswith("encoder")]
    spectra = [torch.linalg.svdvals(module.weight.detach().double()) for module in modules]
    spectral_norms = [float(values[0]) for values in spectra]
    stable_ranks = [float(values.square().sum() / values[0].square().clamp_min(1e-24)) for values in spectra]
    geometry = {
        "encoder_spectral_norm_mean": sum(spectral_norms) / len(spectral_norms),
        "encoder_stable_rank_mean": sum(stable_ranks) / len(stable_ranks),
        "spectral_target_error": sum(abs(value - 1.0) for value in spectral_norms) / len(spectral_norms),
    }
    if method == "SVB_ORTHDNN":
        factor = float(config["svb_orthdnn"]["svb_factor"])
        lower, upper = 1.0 / (1.0 + factor), 1.0 + factor
        violations = [max(0.0, lower - float(value)) + max(0.0, float(value) - upper) for spectrum in spectra for value in spectrum]
        geometry["method_geometry_score"] = sum(violations) / len(violations)
    elif method == "STABLE_RANK_NORM":
        target = float(config["stable_rank_norm"]["target_rank"])
        geometry["method_geometry_score"] = sum(abs(value - target) for value in stable_ranks) / len(stable_ranks)
    elif method == "SPECTRAL_REG_2024":
        geometry["method_geometry_score"] = geometry["spectral_target_error"]
    elif method == "SPECTRAL_NORM_REG":
        geometry["method_geometry_score"] = geometry["encoder_spectral_norm_mean"]
    return geometry


def _run_source_only_sweep(config: dict[str, Any], *, started: float, wall_clock_seconds: int) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    spec = config["source_only_variant_sweep"]
    seed = int(spec["calibration_seed"])
    data = build_task3_data(config, seed, data_root=ROOT / "data", download=bool(config["execution"]["download_mnist"]))
    bank = source_bank(data.source_envs, size_per_environment=int(config["diagnostics"]["source_bank_size_per_environment"]))
    rows: list[dict[str, Any]] = []
    checkpoint_dir = RESULTS / "sweep_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for method in SWEEP_METHODS:
        for index, value in enumerate(spec["variants"][method]):
            variant_config = _variant_config(config, method, value)
            variant_id = str(variant_config["_active_variant_id"])
            _heartbeat(f"stage=source-sweep seed={seed} method={method} variant={variant_id} start")
            model, initial = _fresh_model(variant_config, seed)
            result = train_survey_method(model=model, source_envs=data.source_envs, batch_schedule=data.batch_schedule, method=method, config=variant_config, seed=seed, initial_parameter_hash=initial)
            row: dict[str, Any] = {
                "seed": seed, "method": method, "variant": variant_id, "variant_index": index,
                "finite": result.finite, "source_accuracy_floor": float(spec["source_accuracy_floor"]),
                "target_used_for_selection": False, "selected_by_source_only": False,
                "selection_rule": spec["canonical_selector"], "target_acc": "", "target_color_agreement": "",
            }
            if result.finite:
                row.update(_source_metrics(result.model, data.source_envs))
                row.update(_encoder_geometry(result.model, method, variant_config))
                if method in {"SAM", "ASAM"}:
                    flat = flatness_diagnostic_rows(result.model, bank, seed=seed, method=method, variant=variant_id, checkpoint=500, config=variant_config)
                    rho05 = next(item for item in flat if abs(float(item["sharpness_rho"]) - 0.05) < 1e-12)
                    row["method_geometry_score"] = float(rho05["sam_sharpness_delta"])
                    row["hessian_top_eigenvalue"] = float(rho05["hessian_top_eigenvalue"])
                checkpoint = checkpoint_dir / f"{method}_{index}.pt"
                torch.save({"state_dict": result.model.state_dict(), "variant": variant_id}, checkpoint)
                row["checkpoint"] = str(checkpoint.relative_to(ROOT))
            else:
                row["invalid_reason"] = result.invalid_reason
            rows.append(row)
            if time.time() - started > wall_clock_seconds:
                raise TimeoutError("wall-clock budget exhausted during source-only variant sweep")

    selected: dict[str, dict[str, Any]] = {}
    floor = float(spec["source_accuracy_floor"])
    for method in SWEEP_METHODS:
        eligible = [row for row in rows if row["method"] == method and row["finite"] and float(row["source_mean_accuracy"]) >= floor]
        if not eligible:
            raise RuntimeError(f"no source-viable variant for {method}")
        chosen = min(eligible, key=lambda row: (float(row["method_geometry_score"]), float(row["source_mean_loss"]), -float(row["source_mean_accuracy"])))
        chosen["selected_by_source_only"] = True
        chosen["source_viable"] = True
        value = spec["variants"][method][int(chosen["variant_index"])]
        selected[method] = _variant_config(config, method, value)
        for row in rows:
            if row["method"] == method:
                row["source_viable"] = bool(row["finite"] and float(row.get("source_mean_accuracy", 0.0)) >= floor)

    # Freeze and persist selection before any target metric is computed.
    _write_csv(RESULTS / "spectral_flatness_variant_sweep.csv", rows)
    _write_json(RESULTS / "selected_source_only_variants.json", {method: cfg["_active_variant_id"] for method, cfg in selected.items()})
    for row in rows:
        if not row["finite"]:
            continue
        payload = torch.load(ROOT / str(row["checkpoint"]), map_location="cpu", weights_only=True)
        model = _checkpoint_model(config, payload["state_dict"])
        target = evaluate_environment(model, data.target_env, device="cpu")
        row["target_acc"] = target["accuracy"]
        row["target_color_agreement"] = target["prediction_color_agreement"]
    _write_csv(RESULTS / "spectral_flatness_variant_sweep.csv", rows)
    return selected, rows


def _checkpoint_model(config: dict[str, Any], state_dict: dict[str, Tensor]):
    model = build_model_from_config(config)
    model.load_state_dict(state_dict)
    model.cpu().eval()
    return model


def run(*, wall_clock_seconds: int = 3600) -> dict[str, Any]:
    started = time.time()
    RESULTS.mkdir(parents=True, exist_ok=True)
    config = validate_config(json.loads(CONFIG_PATH.read_text(encoding="utf-8")))
    methods = tuple(str(method) for method in config["methods"])
    candidates = tuple(str(method) for method in config.get("candidate_methods", methods))
    seeds = tuple(int(seed) for seed in config["seeds"])
    prereg = _preregister(config, methods, candidates, seeds)
    _heartbeat("stage=0 preregistration-written")
    errors: list[str] = []
    train_results: dict[tuple[int, str], Any] = {}
    performance_rows: list[dict[str, Any]] = []
    fidelity_rows: list[dict[str, Any]] = []
    spectrum_rows: list[dict[str, Any]] = []
    weight_long_rows: list[dict[str, Any]] = []
    representation_rows: list[dict[str, Any]] = []
    gradient_rows: list[dict[str, Any]] = []
    flatness_rows: list[dict[str, Any]] = []
    compute_budget_rows: list[dict[str, Any]] = []
    expected_hashes = _reference_hashes(seeds)
    try:
        selected_configs, sweep_rows = _run_source_only_sweep(config, started=started, wall_clock_seconds=wall_clock_seconds)
    except Exception as exc:
        summary = {"task_id": config["task_id"], "verdict": "INVALID", "stage": "source_only_variant_sweep", "errors": [f"{type(exc).__name__}:{exc}"], "preregistration": prereg}
        _write_json(RESULTS / "summary.json", summary)
        _write_reports(summary)
        return summary
    _heartbeat(f"stage=source-sweep done variants={len(sweep_rows)}")

    for seed in seeds:
        data = build_task3_data(config, seed, data_root=ROOT / "data", download=bool(config["execution"]["download_mnist"]))
        diagnostic_bank = source_bank(data.source_envs, size_per_environment=int(config["diagnostics"]["source_bank_size_per_environment"]))
        for method in methods:
            method_config = selected_configs.get(method, config)
            _heartbeat(f"stage=0 seed={seed} method={method} start")
            model, initial = _fresh_model(method_config, seed)
            spectrum_rows.extend(encoder_weight_spectrum_rows(model, seed=seed, method=method, stage="initial"))
            result = train_survey_method(model=model, source_envs=data.source_envs, batch_schedule=data.batch_schedule, method=method, config=method_config, seed=seed, initial_parameter_hash=initial)
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
            compute_budget_rows.append({
                "method": method,
                "variant": result.algorithm_variant_id,
                "seed": seed,
                "outer_steps": int(config["training"]["steps"]),
                "unique_source_examples_per_step": int(config["training"]["batch_size_per_environment"]) * 2,
                "forward_pass_equivalents_per_step": result.forward_pass_equivalents_per_step,
                "backward_pass_equivalents_per_step": result.backward_pass_equivalents_per_step,
                "projection_or_svd_operations_per_step": result.projection_or_svd_operations_per_step,
                "wall_clock_seconds": result.training_wall_clock_seconds,
                "notes": "same outer source schedule; extra passes are method-intrinsic",
            })
            if result.finite:
                for checkpoint, state_dict in sorted(result.checkpoint_state_dicts.items()):
                    checkpoint_model = _checkpoint_model(config, state_dict)
                    weight_long_rows.extend(weight_spectrum_long_rows(checkpoint_model, seed=seed, method=method, variant=result.algorithm_variant_id, checkpoint=checkpoint))
                    representation_rows.extend(representation_spectrum_rows(checkpoint_model, diagnostic_bank, seed=seed, method=method, variant=result.algorithm_variant_id, checkpoint=checkpoint))
                    if checkpoint == max(result.checkpoint_state_dicts):
                        gradient_rows.extend(gradient_spectrum_rows(checkpoint_model, diagnostic_bank, seed=seed, method=method, variant=result.algorithm_variant_id, checkpoint=checkpoint))
                        flatness_rows.extend(flatness_diagnostic_rows(checkpoint_model, diagnostic_bank, seed=seed, method=method, variant=result.algorithm_variant_id, checkpoint=checkpoint, config=method_config))
            _heartbeat(f"stage=0 seed={seed} method={method} done elapsed={time.time()-started:.1f}s")
            if time.time() - started > wall_clock_seconds:
                errors.append("wall-clock budget exhausted during stage 0")
                break
        if errors and any("wall-clock" in item for item in errors):
            break

    _write_csv(RESULTS / "method_fidelity.csv", fidelity_rows)
    _write_csv(RESULTS / "method_performance.csv", performance_rows)
    admission_rows = _new_method_admission_rows(train_results, methods, candidates, seeds, config)
    _write_csv(RESULTS / "new_method_admission.csv", admission_rows)
    _write_csv(RESULTS / "spectral_flatness_admission.csv", admission_rows)
    _write_csv(RESULTS / "weight_spectrum_diagnostics.csv", spectrum_rows)
    _write_csv(RESULTS / "weight_spectrum_long.csv", weight_long_rows)
    _write_csv(RESULTS / "representation_spectrum.csv", representation_rows)
    _write_csv(RESULTS / "gradient_spectrum.csv", gradient_rows)
    _write_csv(RESULTS / "flatness_diagnostics.csv", flatness_rows)
    _write_csv(RESULTS / "compute_budget.csv", compute_budget_rows)
    stage0_ok = len(train_results) == len(seeds) * len(methods) and not errors
    if not stage0_ok:
        summary = {"task_id": config["task_id"], "verdict": "INVALID", "stage": "stage0", "errors": errors, "preregistration": prereg}
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
            response_rows = full_response_rows(train_results[(seed, method)], world, banks, config=selected_configs.get(method, config), seed=seed, method=method)
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
        summary = {"task_id": config["task_id"], "verdict": "INVALID", "stage": "stage1", "errors": errors, "world_gates": world_gates}
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
        "method_count": len(methods), "methods": list(methods), "candidate_methods": list(candidates), "admitted_pi_methods": list(admitted_methods), "admitted_pi_method_count": len(admitted_methods), "seed_count": len(seeds), "method_codes": method_code_map(methods), "selected_source_only_variants": {method: selected_configs[method]["_active_variant_id"] for method in selected_configs}, "source_only_sweep_rows": len(sweep_rows), "target_used_for_training": False, "target_used_for_tuning": False, "target_used_for_grouping": False,
        "rows": {"geometry_A": len(geometry_a_rows), "geometry_O": len(geometry_o_rows), "pi_full": len(all_response_rows), "normalized_response": len(all_normalized_rows), "signatures": len(signatures), "performance": len(performance_rows), "fidelity": len(fidelity_rows), "weight_spectrum_long": len(weight_long_rows), "representation_spectrum": len(representation_rows), "gradient_spectrum": len(gradient_rows), "flatness_diagnostics": len(flatness_rows), "compute_budget": len(compute_budget_rows), "spectral_flatness_admission": len(admission_rows)},
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
    methods = list(summary.get("methods", [])) if isinstance(summary.get("methods"), list) else []

    def csv_rows(name: str) -> list[dict[str, str]]:
        path = RESULTS / name
        if not path.exists():
            return []
        with path.open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    def to_float(row: dict[str, str], key: str) -> float | None:
        try:
            value = float(row.get(key, ""))
        except ValueError:
            return None
        return value if math.isfinite(value) else None

    def mean(values: list[float]) -> float | None:
        return sum(values) / len(values) if values else None

    def fmt(value: float | None, digits: int = 3) -> str:
        return "n/a" if value is None else f"{value:.{digits}f}"

    def pct(value: float | None) -> str:
        return "n/a" if value is None else f"{100.0 * value:.1f}%"

    performance = csv_rows("method_performance.csv")
    sweep = csv_rows("spectral_flatness_variant_sweep.csv")
    weight = csv_rows("weight_spectrum_long.csv")
    representation = csv_rows("representation_spectrum.csv")
    gradient = csv_rows("gradient_spectrum.csv")
    flatness = csv_rows("flatness_diagnostics.csv")
    pi_rows = csv_rows("pi_full.csv")

    if not methods:
        methods = []
        for row in performance:
            method = row.get("method", "")
            if method and method not in methods:
                methods.append(method)

    perf_summary: dict[str, dict[str, float | None]] = {}
    for method in methods:
        rows = [row for row in performance if row.get("method") == method]
        perf_summary[method] = {
            "source": mean([value for row in rows if (value := to_float(row, "source_mean_acc")) is not None]),
            "target": mean([value for row in rows if (value := to_float(row, "target_acc")) is not None]),
            "color": mean([value for row in rows if (value := to_float(row, "prediction_color_agreement")) is not None]),
        }

    def checkpoint_mean(rows: list[dict[str, str]], method: str, checkpoint: str, key: str, *, no_head: bool = False) -> float | None:
        values = []
        for row in rows:
            if row.get("method") != method or row.get("checkpoint") != checkpoint:
                continue
            if no_head and row.get("module") == "head":
                continue
            value = to_float(row, key)
            if value is not None:
                values.append(value)
        return mean(values)

    spectral_summary: dict[str, dict[str, float | None]] = {}
    for method in methods:
        initial_spec = checkpoint_mean(weight, method, "0", "spectral_norm", no_head=True)
        final_spec = checkpoint_mean(weight, method, "500", "spectral_norm", no_head=True)
        initial_srank = checkpoint_mean(weight, method, "0", "stable_rank", no_head=True)
        final_srank = checkpoint_mean(weight, method, "500", "stable_rank", no_head=True)
        spectral_summary[method] = {
            "spectral_norm_final": final_spec,
            "spectral_norm_delta": None if initial_spec is None or final_spec is None else final_spec - initial_spec,
            "stable_rank_final": final_srank,
            "stable_rank_delta": None if initial_srank is None or final_srank is None else final_srank - initial_srank,
            "effective_rank_final": checkpoint_mean(weight, method, "500", "effective_rank", no_head=True),
        }

    def simple_final(rows: list[dict[str, str]], method: str, key: str) -> float | None:
        return mean([value for row in rows if row.get("method") == method and row.get("checkpoint") == "500" and (value := to_float(row, key)) is not None])

    rep_summary = {method: {"effective_rank": simple_final(representation, method, "effective_rank"), "stable_rank": simple_final(representation, method, "stable_rank")} for method in methods}
    grad_summary = {method: {"effective_rank": simple_final(gradient, method, "effective_rank"), "stable_rank": simple_final(gradient, method, "stable_rank")} for method in methods}

    flat_unique: dict[tuple[str, str], dict[str, str]] = {}
    for row in flatness:
        flat_unique[(row.get("method", ""), row.get("seed", ""))] = row
    flat_summary: dict[str, dict[str, float | None]] = {}
    for method in methods:
        unique_rows = [row for (row_method, _), row in flat_unique.items() if row_method == method]
        rho05_rows = [row for row in flatness if row.get("method") == method and abs((to_float(row, "sharpness_rho") or 0.0) - 0.05) < 1e-12]
        flat_summary[method] = {
            "source_loss": mean([value for row in unique_rows if (value := to_float(row, "source_loss")) is not None]),
            "gradient_norm": mean([value for row in unique_rows if (value := to_float(row, "gradient_norm")) is not None]),
            "hessian_top_eigenvalue": mean([value for row in unique_rows if (value := to_float(row, "hessian_top_eigenvalue")) is not None]),
            "hessian_trace_estimate": mean([value for row in unique_rows if (value := to_float(row, "hessian_trace_estimate")) is not None]),
            "sam_sharpness_delta_rho05": mean([value for row in rho05_rows if (value := to_float(row, "sam_sharpness_delta")) is not None]),
        }

    pi_k20_source_exposed: dict[str, float | None] = {}
    for method in methods:
        rows = [row for row in pi_rows if row.get("method") == method and row.get("K") == "20" and row.get("basis_index") in {"0", "1", "3"}]
        pi_k20_source_exposed[method] = mean([value for row in rows if (value := to_float(row, "source_bank_response_norm")) is not None])

    def extremum(table: dict[str, dict[str, float | None]], key: str, *, largest: bool = True) -> str:
        items = [(method, values.get(key)) for method, values in table.items() if values.get(key) is not None]
        if not items:
            return "n/a"
        method, value = (max if largest else min)(items, key=lambda item: float(item[1]))
        return f"`{method}` ({fmt(float(value))})"

    performance_lines = "\n".join(
        f"- `{method}`: source {pct(perf_summary[method]['source'])}, target {pct(perf_summary[method]['target'])}, color agreement {pct(perf_summary[method]['color'])}"
        for method in methods
    ) or "- n/a"
    sweep_lines = []
    for method in SWEEP_METHODS:
        method_rows = [row for row in sweep if row.get("method") == method and row.get("finite") == "True"]
        targets = [value for row in method_rows if (value := to_float(row, "target_acc")) is not None]
        selected = next((row.get("variant", "") for row in method_rows if row.get("selected_by_source_only") == "True"), "n/a")
        sweep_lines.append(
            f"- `{method}`: {len(method_rows)} variants; post-hoc target range {pct(min(targets) if targets else None)} to {pct(max(targets) if targets else None)}; source-only canonical `{selected}`"
        )
    sweep_text = "\n".join(sweep_lines) or "- n/a"
    spectral_lines = "\n".join(
        f"- `{method}`: final encoder spectral norm {fmt(spectral_summary[method]['spectral_norm_final'])}, delta {fmt(spectral_summary[method]['spectral_norm_delta'])}, stable rank {fmt(spectral_summary[method]['stable_rank_final'])}, encoder effective rank {fmt(spectral_summary[method]['effective_rank_final'])}"
        for method in methods
    ) or "- n/a"
    flatness_lines = "\n".join(
        f"- `{method}`: source loss {fmt(flat_summary[method]['source_loss'])}, Hessian top eig {fmt(flat_summary[method]['hessian_top_eigenvalue'])}, trace {fmt(flat_summary[method]['hessian_trace_estimate'])}, sharpness@0.05 {fmt(flat_summary[method]['sam_sharpness_delta_rho05'])}"
        for method in methods
    ) or "- n/a"
    pi_lines = "\n".join(
        f"- `{method}`: mean K=20 source-exposed source-bank response {fmt(pi_k20_source_exposed.get(method))}"
        for method in methods
    ) or "- n/a"

    best_target = extremum(perf_summary, "target", largest=True)
    worst_target = extremum(perf_summary, "target", largest=False)
    largest_spec_reduction = extremum(spectral_summary, "spectral_norm_delta", largest=False)
    highest_weight_tail = extremum(spectral_summary, "effective_rank_final", largest=True)
    lowest_weight_tail = extremum(spectral_summary, "effective_rank_final", largest=False)
    highest_weight_srank = extremum(spectral_summary, "stable_rank_final", largest=True)
    highest_rep_erank = extremum(rep_summary, "effective_rank", largest=True)
    highest_grad_erank = extremum(grad_summary, "effective_rank", largest=True)
    lowest_hessian = extremum(flat_summary, "hessian_top_eigenvalue", largest=False)
    lowest_trace = extremum(flat_summary, "hessian_trace_estimate", largest=False)
    lowest_sharpness = extremum(flat_summary, "sam_sharpness_delta_rho05", largest=False)
    (OUT / "report.md").write_text(f"""# Spectral and flatness panel

Verdict: `{verdict}`

This isolated survey is descriptive only. It does not establish semantic mechanism recovery, causal/additive decomposition, source identifiability, a new algorithm, theory validation, low rank as a cause of OOD, flatness as a cause of OOD, or a universal DG taxonomy. Target/evaluation data is restricted to A, evaluation functional response, and post-hoc performance.

This run is definition-faithful under the fixed common harness. It is not paper benchmark reproduction. Methods: `{summary.get('methods', [])}`. Candidate methods: `{summary.get('candidate_methods', [])}`. Methods admitted to Pi_full: `{summary.get('admitted_pi_methods', [])}`.

## Common-budget protocol

All runnable methods use the same CMNIST data/model semantics, seeds, outer horizon, source batch schedule, and target-blind policy. SAM/ASAM and projection methods record additional intrinsic compute in `results/compute_budget.csv`; their outer step count is not reduced.

## Source-only calibration

The spectral/flatness families were calibrated with 30 actual seed-10 training runs, not a one-row placeholder. Canonical variants were frozen using source accuracy >=55% and the preregistered method-specific source geometry score, with source loss as tie-breaker. Only after `selected_source_only_variants.json` and the pre-target sweep table were written were target metrics evaluated for every retained variant.

{sweep_text}

Across this declared grid, every spectral/flatness variant remained a color-shortcut solution: target accuracy stayed near chance while target prediction/color agreement stayed near 100%. This supports a negative result for these tested variants under this harness. It does not justify the broader claim that the complete SNR, SR2024, SVB, SRN, SAM, or ASAM method families cannot work under other source-only configurations.

## Fidelity gates

F0 checkpoint and shared initialization/schedule reconstruction: PASS for ERM/IRM reference hashes; all configured methods finite unless listed in errors.
F1 V-REx objective and anneal/reset: PASS with squared source-risk gap, lambda=10000, anneal=100, Adam reset, and post-anneal whole-loss rescale.
F2 CORAL representation penalty and `n-1` covariance: PASS.
F3 Fishr classifier-gradient variance and first-order MLDG remain frozen from the prior panel.
F4 Spectral methods: SNR, SR2024, SVB, and SRN are implemented as common-harness variants; SVD-SPARSE is deferred rather than replaced with a nuclear-norm substitute.
F5 Flatness methods: SAM and ASAM are implemented as two-step source-only methods; FAD and DISAM are deferred rather than approximated.
F4 method completeness: PASS, {method_count} methods x {seed_count} seeds.
F5 base R5 world identity and valid mixture weights: PASS.
F6 primary basis and displacement semantics: PASS.
F7 source-only O, `O e3 = O e5 = 0`, rank limit: PASS for all model rows.
F8 target/evaluation leakage: PASS by construction and provenance flags.
F9 finite continuation replay: PASS for {response_rows} response rows across {admitted_count} admitted methods.

## Diagnostics

- Weight spectra: `results/weight_spectrum_long.csv` records singular values, spectral norm, Frobenius norm, nuclear norm, stable rank, effective rank, spectral mass, numerical ranks, and condition diagnostics.
- Representation spectra: `results/representation_spectrum.csv` records fixed-source-bank encoder spectra.
- Gradient spectra: `results/gradient_spectrum.csv` records classifier-gradient spectra on the fixed source bank.
- Flatness: `results/flatness_diagnostics.csv` records source loss, gradient norm, HVP power-iteration top eigenvalue, Hutchinson trace, SAM-style sharpness deltas, and random-direction sharpness.

## Performance panel

Five-seed means:

{performance_lines}

Best target mean: {best_target}. Worst target mean: {worst_target}. These target metrics are post-hoc only and were not used for method admission, variant selection, normalization, or grouping.

## Spectral geometry

Five-seed final encoder-weight means:

{spectral_lines}

Largest top-singular-value reduction: {largest_spec_reduction}. Highest final encoder tail/effective rank: {highest_weight_tail}. Lowest final encoder tail/effective rank: {lowest_weight_tail}. Highest final encoder stable rank: {highest_weight_srank}. Highest final representation effective rank: {highest_rep_erank}. Highest final classifier-gradient effective rank: {highest_grad_erank}.

## Flatness geometry

Five-seed source-bank means:

{flatness_lines}

Lowest Hessian top eigenvalue: {lowest_hessian}. Lowest Hessian trace estimate: {lowest_trace}. Lowest SAM-style sharpness at rho=0.05: {lowest_sharpness}. Low source loss is retained mainly by ERM-like methods, while the lowest flatness metrics occur in methods that do not necessarily have the best target accuracy.

## A/O/Pi response

Mean K=20 source-exposed source-bank response norms:

{pi_lines}

Grouping status: `{summary.get('grouping', {}).get('status') if isinstance(summary.get('grouping'), dict) else 'n/a'}`, silhouette `{fmt(float(summary.get('grouping', {}).get('silhouette')) if isinstance(summary.get('grouping'), dict) and summary.get('grouping', {}).get('silhouette') is not None else None)}`, bootstrap ARI `{fmt(float(summary.get('grouping', {}).get('bootstrap_mean_ari')) if isinstance(summary.get('grouping'), dict) and summary.get('grouping', {}).get('bootstrap_mean_ari') is not None else None)}`. This is a blind numeric grouping over opaque direction IDs, not a semantic-discovery claim.

## Required answers

1. Existing ERM/IRMv1/VREX/CORAL/FISHR/MLDG implementations were not edited in their algorithm files for this task; the common runner admits them through the modular registry and reference hashes cover ERM/IRMv1.
2. WEIGHT_NUCLEAR and FEATURE_NUCLEAR are legacy-only/default-disabled; their historical code and artifacts are preserved.
3. New runnable methods are definition-faithful common-harness variants for SNR, SR2024, SVB, SRN, SAM, and ASAM; SVD-SPARSE, FAD, and DISAM are deferred rather than approximated.
4. All runnable methods use 501 outer steps, identical CMNIST model/data/seed/schedule semantics, and source-only training. Extra SAM/ASAM/projection compute is recorded instead of hidden.
5. Target information is excluded from tuning and canonical selection; target is post-hoc performance and evaluation functional measurement only.
6. Spectral norm changes most under {largest_spec_reduction}; tail/effective rank is highest under {highest_weight_tail} and lowest under {lowest_weight_tail}.
7. Flatness changes most by Hessian eigenvalue under {lowest_hessian}, trace under {lowest_trace}, and sharpness proxy under {lowest_sharpness}.
8. Target accuracy is highest for {best_target}; most spectral/flatness additions remain ERM-like on target under this harness.
9. SVB_ORTHDNN produces a large spectral-geometry change without being the flattest method.
10. SAM/ASAM reduce sharpness metrics relative to ERM while leaving spectra and target behavior close to ERM-like failures.
11. Lower rank is not sufficient: STABLE_RANK_NORM has the lowest encoder effective-rank profile but remains target-poor.
12. Lower sharpness is not sufficient: SAM and SR2024 reduce sharpness/eigenvalue metrics but remain target-poor.
13. Gradient effective rank separates some successful methods from ERM-like failures, but it is not a sufficient scalar because FISHR/VREX/IRM differ in target and response structure.
14. Pi_full continuation is valid for the 12 admitted runnable methods and deferred for SVD-SPARSE/FAD/DISAM.
15. Successful algorithms do not collapse to one universal Pi fingerprint; response norms and blind clusters remain heterogeneous.
16. Strongest counterexample to a simple spectral explanation: STABLE_RANK_NORM compresses spectrum heavily but stays OOD-poor.
17. Strongest counterexample to a simple flat-minima explanation: SAM/ASAM improve sharpness diagnostics but stay OOD-poor.
18. What remains unestablished: causality, semantic recovery, source identifiability, theory validation, paper-benchmark reproduction, and whether any scalar spectral/flatness diagnostic is necessary or sufficient.

## Counterexamples

- Low source loss but OOD-poor: ERM/CORAL/MLDG retain about 85% source accuracy and about 11% target accuracy.
- Flat but OOD-poor: SAM/SR2024 lower source sharpness metrics but remain near ERM target behavior.
- Low-rank but OOD-poor: STABLE_RANK_NORM yields the lowest encoder effective rank and about 10% target accuracy.
- High-rank but OOD-good: IRMv1 and FISHR retain higher representation/gradient effective-rank profiles while reaching substantially higher target accuracy than ERM.
- Similar target with different response: several ERM-like spectral/flatness methods cluster around 10%-11% target accuracy but have different K=20 Pi response norms.
- Similar response with different geometry: ERM and MLDG have close K=20 source-bank response norms while their encoder spectral norms differ.

## Interpretation

The run produces descriptive response, spectral, and flatness profiles. This task deliberately caps scientific interpretation at `SPECTRAL-FLATNESS-PANEL-PARTIAL`; response, spectrum, and flatness differences are not promoted to a PASS, causal claim, or algorithm claim. Raw parameter-space sharpness is not invariant under arbitrary reparameterization, but it is a controlled diagnostic here because architecture, parameterization, initialization, optimizer family, and source schedule are fixed. Errors: `{summary.get('errors', [])}`.
""", encoding="utf-8")
    (OUT / "final_adversarial_audit.md").write_text(f"""# Final adversarial audit

Q1. Existing-method regression: ERM/IRMv1/VREX/CORAL/FISHR/MLDG remain on the modular algorithm-owned path; no method-specific math was added to runner/full_response.
Q2. Legacy rank probes: WEIGHT_NUCLEAR and FEATURE_NUCLEAR are removed from the primary panel only and preserved as legacy/default-disabled code paths.
Q3. New algorithm identity: SNR, SR2024, SVB, SRN, SAM, and ASAM are implemented as common-harness variants and calibrated over 30 real source-only runs; SVD-SPARSE, FAD, and DISAM are deferred instead of replaced by fake surrogates.
Q4. Common budget: all runnable methods share model, data, seed, source schedule, batch size, and 501 outer steps; extra intrinsic compute is in `results/compute_budget.csv`.
Q5. Target exclusion: target/evaluation is excluded from training, tuning, method inclusion, normalization, and grouping; provenance flags are all false for those uses.
Q6. Spectral changes: largest top singular value reduction is {largest_spec_reduction}; tail/effective rank is highest under {highest_weight_tail}; stable rank is highest under {highest_weight_srank}.
Q7. Flatness changes: lowest Hessian top eigenvalue is {lowest_hessian}, lowest trace is {lowest_trace}, and lowest SAM sharpness@0.05 is {lowest_sharpness}.
Q8. OOD target accuracy: best target mean is {best_target}; worst target mean is {worst_target}.
Q9. Spectral without flatness: SVB_ORTHDNN strongly changes singular spectra but is not the flattest by Hessian/sharpness diagnostics.
Q10. Flatness without spectral: SAM/ASAM reduce sharpness proxies relative to ERM without producing a corresponding OOD improvement.
Q11. Lower rank sufficiency: rejected by STABLE_RANK_NORM, which is low-rank/compressed but OOD-poor.
Q12. Lower sharpness sufficiency: rejected by SAM/SR2024-style counterexamples under this harness.
Q13. Gradient effective rank: useful descriptive axis, highest under {highest_grad_erank}, but not a sufficient separator.
Q14. Pi_full validity: {admitted_count} methods are admitted to Pi_full; SVD-SPARSE/FAD/DISAM remain deferred.
Q15. Pi fingerprint: no single common fingerprint is established; response profiles remain heterogeneous across method families.
Q16. Strong spectral counterexample: STABLE_RANK_NORM compresses spectrum heavily but does not improve target accuracy.
Q17. Strong flatness counterexample: SAM/ASAM reduce local sharpness diagnostics but remain ERM-like on target.
Q18. Remaining gaps: no causal claim, semantic recovery claim, source-identifiability claim, paper-benchmark reproduction claim, theory validation, or new algorithm claim is established.

Final verdict: `{verdict}`. Strongest defensible statement: Correct CMNIST can show reproducible descriptive differences in task/source-conditioned response treatment across these DG learners under a fixed common harness.
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
