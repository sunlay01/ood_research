"""Run TASK-AOPI-CMNIST-REINSTANTIATION-AUDIT."""

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

from .task3_aopi_cmnist_reinstantiation.analysis import ALLOWED_VERDICTS, discrimination_gate, finite_prediction_gate, geometry_gate, overall_verdict
from .task3_aopi_cmnist_reinstantiation.finite_validation import finite_response, sign_agreement
from .task3_aopi_cmnist_reinstantiation.learner_response import REFERENCE_GRADIENT_NORM_MAX, augmented_head_weights, pi_operator, refine_head
from .task3_aopi_cmnist_reinstantiation.source_geometry import (
    encoder_parameter_hash,
    evaluation_gradient,
    freeze_encoder,
    geometry_summary,
    normalized_gram,
    observation_O,
    response_A,
    source_features,
    source_labels,
    source_risk_hessian,
    whiten_source_hessian,
)
from .task3_aopi_cmnist_reinstantiation.world_tangents import EPSILONS, REPORT_DIRECTIONS, WorldFactory, centered_pair, tangent_manifest_rows
from .task3_ood_capability_decomposition.artifacts import load_checkpoint_manifest, load_verified_model, sha256_file
from .task3_cmnist_cpu_minimal.model import linear_layer_count, parameter_hash


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs/task3_cmnist_cpu_minimal.json"
MANIFEST_PATH = ROOT / "round3_redesign/task3_cmnist_counterfactual_audit/results/checkpoint_manifest.csv"
OUT_DIR = ROOT / "round3_redesign/task3_aopi_cmnist_reinstantiation"
RESULTS_DIR = OUT_DIR / "results"
STATE_DELTA_PATH = ROOT / "active/STATE_DELTA.md"
TASK_ID = "TASK-AOPI-CMNIST-REINSTANTIATION-AUDIT"
SEEDS = [10, 11, 12, 13, 14]
METHODS = ["ERM", "IRMv1"]
TARGET_GAP_MIN = 0.20


def _git(args: list[str]) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False).stdout.strip()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_clean(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _json_clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_clean(item) for item in value]
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_clean(value), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "NaN" if isinstance(row.get(key), float) and not math.isfinite(float(row[key])) else row.get(key, "NA") for key in fields})


def _load_config() -> dict[str, Any]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if config.get("task_id") != "TASK3-CMNIST-CPU-MINIMAL" or config.get("stage_b", {}).get("seeds") != SEEDS:
        raise ValueError("audit requires the fixed corrected CPU-minimal config")
    return config


def write_preregistration(config: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "task_id": TASK_ID,
        "written_at_unix": time.time(),
        "git_head": _git(["rev-parse", "HEAD"]),
        "config_sha256": _sha(CONFIG_PATH),
        "manifest": str(MANIFEST_PATH.relative_to(ROOT)),
        "seeds": SEEDS,
        "methods": METHODS,
        "parameter_block": "frozen_encoder_trainable_augmented_final_head_65d",
        "epsilons": list(EPSILONS),
        "head_lbfgs": {"max_iter": 100, "tolerance_grad": 1e-9, "tolerance_change": 1e-12, "accepted_gradient_norm_max": REFERENCE_GRADIENT_NORM_MAX},
        "source_hessian_damping_relative": 1e-6,
        "ift_damping_relative": 1e-8,
        "primary_metric": "frobenius_norm_A_plus_PiO",
        "gates": {"g1_gram": 0.15, "g1_reparameterization": 0.05, "g2_cosine": 0.90, "g2_relative_vector_error": 0.25, "g2_min_rows": 12},
        "config": config,
    }
    text = f"""# {TASK_ID} Preregistered Design

The audit freezes each verified nonlinear encoder and re-optimizes only its
65-dimensional final linear head from the saved head initialization.

- Methods: `ERM`, `IRMv1`; seeds: `10..14`.
- Source tangent basis: env0 color flip, env1 color flip, shared label noise.
- Report directions: env0, env1, label, common color, antisymmetric color.
- Centered probability finite differences: `0.01`, `0.02` with common random numbers.
- Source-only quantities: refined head, `H_S`, `O_S`, `Pi`, all gates.
- Held-out evaluation quantities: `A` and post-hoc finite-response checks only.
- Main metric: `||A + Pi O_S||_F`; all thresholds are in `provenance.json`.

Interpretation ceiling: this is a frozen-encoder head-block local response
audit. It is not a full-network response, new algorithm, causal claim, or
universal DG result.
"""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "preregistered_design.md").write_text(text, encoding="utf-8")
    return payload


def verify_inputs(config: dict[str, Any]) -> tuple[list[Any], dict[str, Any]]:
    records = load_checkpoint_manifest(MANIFEST_PATH, seeds=SEEDS, methods=METHODS, root=ROOT)
    config_sha = sha256_file(CONFIG_PATH)
    rows: list[dict[str, Any]] = []
    targets = {method: [] for method in METHODS}
    for record in records:
        if sha256_file(record.path) != record.checkpoint_sha256 or record.config_sha256 != config_sha:
            raise ValueError(f"checkpoint provenance mismatch: {record.seed}/{record.method}")
        model = load_verified_model(record, config, config_sha256=config_sha)
        if linear_layer_count(model) != 3 or parameter_hash(model) != record.parameter_hash:
            raise ValueError("checkpoint model identity mismatch")
        payload = torch.load(record.path, map_location="cpu")
        metrics = {key: float(value) for key, value in payload["official_metrics"].items()}
        targets[record.method].append(metrics["final_target_acc"])
        rows.append({"seed": record.seed, "method": record.method, "checkpoint_path": str(record.path.relative_to(ROOT)), "checkpoint_sha256": record.checkpoint_sha256, "parameter_hash": record.parameter_hash, "config_sha256": record.config_sha256, "encoder_parameter_hash": encoder_parameter_hash(model), **metrics})
    erm_mean, irm_mean = sum(targets["ERM"]) / 5.0, sum(targets["IRMv1"]) / 5.0
    if irm_mean - erm_mean < TARGET_GAP_MIN:
        raise ValueError("corrected ERM/IRMv1 target gap is not present")
    return records, {"manifest_rows": rows, "erm_target_acc_mean": erm_mean, "irmv1_target_acc_mean": irm_mean, "target_gap": irm_mean - erm_mean, "target_gap_verified": True}


def _reparameterized_gram(hessian: torch.Tensor, derivative_columns: torch.Tensor) -> torch.Tensor:
    scale = torch.cat((torch.full((32,), 2.0, dtype=torch.double), torch.full((32,), 0.5, dtype=torch.double), torch.ones(1, dtype=torch.double)))
    transform = torch.diag(scale)
    inverse = torch.diag(scale.reciprocal())
    transformed_h = inverse.T @ hessian @ inverse
    transformed_g = inverse.T @ derivative_columns
    values, vectors = torch.linalg.eigh((transformed_h + transformed_h.T) / 2.0)
    values = values.clamp_min(1e-12)
    inverse_root = vectors @ torch.diag(values.rsqrt()) @ vectors.T
    response = inverse_root @ transformed_g
    return normalized_gram(response)


def _timeout(started: float, limit: int) -> None:
    if time.monotonic() - started > limit:
        raise TimeoutError(f"audit exceeded preregistered wall clock limit of {limit}s")


def audit_record(record: Any, config: dict[str, Any], *, started: float, wall_clock_seconds: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    _timeout(started, wall_clock_seconds)
    print(f"[aopi] source-reference seed={record.seed} method={record.method}", flush=True)
    model = load_verified_model(record, config, config_sha256=sha256_file(CONFIG_PATH))
    complete_hash_before, encoder = parameter_hash(model), freeze_encoder(model)
    factory = WorldFactory(config, record.seed, data_root=ROOT / "data", download=False)
    base = factory.build()
    base_features, base_labels = source_features(encoder, base), source_labels(base)
    initial_head = augmented_head_weights(model.head.weight, model.head.bias)
    reference = refine_head(record.method, base_features, base_labels, initial_head)
    if not reference.converged:
        raise ValueError(f"source-only head reference did not converge for {record.seed}/{record.method}: {reference.gradient_norm}")
    whitening = whiten_source_hessian(source_risk_hessian(base_features, reference.weights))
    pi = pi_operator(record.method, base_features, base_labels, reference.weights, whitening.root)
    a_rows: list[dict[str, Any]] = []
    o_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    mismatch_rows: list[dict[str, Any]] = []
    by_epsilon: dict[float, dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = {float(epsilon): {} for epsilon in EPSILONS}
    for epsilon in EPSILONS:
        for tangent, direction in REPORT_DIRECTIONS.items():
            _timeout(started, wall_clock_seconds)
            print(f"[aopi] direction seed={record.seed} method={record.method} tangent={tangent} epsilon={epsilon}", flush=True)
            plus, minus = centered_pair(factory, direction, epsilon)
            plus_features, plus_labels = source_features(encoder, plus), source_labels(plus)
            minus_features, minus_labels = source_features(encoder, minus), source_labels(minus)
            task = response_A(encoder, plus, minus, reference.weights, whitening.inverse_root, epsilon)
            observation = observation_O(plus_features, plus_labels, minus_features, minus_labels, reference.weights, epsilon)
            predicted = pi.pi @ observation
            finite = finite_response(record.method, plus_features, plus_labels, minus_features, minus_labels, reference, whitening.root, predicted, epsilon)
            mismatch = task + predicted
            by_epsilon[float(epsilon)][tangent] = (task, observation, mismatch)
            a_rows.append({"seed": record.seed, "method": record.method, "tangent": tangent, "epsilon": epsilon, "response_norm": float(task.norm().item()), "whitened_response_norm": float(task.norm().item()), "retained_hessian_rank": int(whitening.retained.sum().item()), "hessian_min_eig": whitening.min_eigenvalue, "hessian_max_eig": whitening.max_eigenvalue, "hessian_condition": whitening.condition_number, "hessian_damping": whitening.damping, "whitening_identity_error": whitening.identity_error, "finite": bool(torch.isfinite(task).all())})
            o_rows.append({"seed": record.seed, "method": record.method, "tangent": tangent, "epsilon": epsilon, "observation_norm": float(observation.norm().item()), "finite": bool(torch.isfinite(observation).all())})
            validation_rows.append({"seed": record.seed, "method": record.method, "tangent": tangent, "epsilon": epsilon, "predicted_response_norm": float(finite.predicted.norm().item()), "actual_response_norm": float(finite.actual.norm().item()), "cosine_similarity": finite.cosine_similarity, "relative_norm_error": finite.relative_norm_error, "relative_vector_error": finite.relative_vector_error, "task_inner_product_sign_agreement": sign_agreement(task, finite), "finite": finite.finite})
            mismatch_rows.append({"seed": record.seed, "method": record.method, "tangent": tangent, "epsilon": epsilon, "A_norm": float(task.norm().item()), "PiO_norm": float(predicted.norm().item()), "mismatch_norm": float(mismatch.norm().item()), "normalized_mismatch_if_preregistered": float(mismatch.norm().item() / max(task.norm().item(), 1e-12)), "finite": bool(torch.isfinite(mismatch).all())})
    for epsilon, values in by_epsilon.items():
        a_matrix = torch.stack([values[name][0] for name in REPORT_DIRECTIONS], dim=1)
        o_matrix = torch.stack([values[name][1] for name in REPORT_DIRECTIONS], dim=1)
        a_gram = normalized_gram(a_matrix)
        other = normalized_gram(torch.stack([by_epsilon[0.02 if epsilon == 0.01 else 0.01][name][0] for name in REPORT_DIRECTIONS], dim=1))
        epsilon_disagreement = float((a_gram - other).norm().item())
        derivative_columns = whitening.root @ a_matrix
        reparameterized = _reparameterized_gram(whitening.effective_hessian, derivative_columns)
        reparameterized_disagreement = float((a_gram - reparameterized).norm().item())
        a_info, o_info = geometry_summary(a_matrix), geometry_summary(o_matrix)
        for row in a_rows:
            if row["seed"] == record.seed and row["method"] == record.method and float(row["epsilon"]) == epsilon:
                row.update(a_info | {"gram_disagreement_with_epsilon_pair": epsilon_disagreement, "reparameterization_gram_disagreement": reparameterized_disagreement})
        for row in o_rows:
            if row["seed"] == record.seed and row["method"] == record.method and float(row["epsilon"]) == epsilon:
                row.update({"source_observation_rank": o_info["response_rank"], "kernel_dimension": 3 - min(3, int(o_info["response_rank"]))})
    if parameter_hash(model) != complete_hash_before or encoder_parameter_hash(model) != encoder.encoder_hash:
        raise RuntimeError("audit mutated stored checkpoint model parameters")
    provenance = {"seed": record.seed, "method": record.method, "checkpoint_parameter_hash": complete_hash_before, "encoder_parameter_hash": encoder.encoder_hash, "source_reference_gradient_norm": reference.gradient_norm, "head_initial_displacement_norm": reference.initial_displacement_norm, "pi_condition_number": pi.condition_number, "pi_damping": pi.damping}
    return a_rows, o_rows, validation_rows, mismatch_rows, provenance


def _report(summary: dict[str, Any], provenance: dict[str, Any]) -> str:
    return f"""# {TASK_ID} Report

## Provenance And Validity

- git head: `{provenance['git_head']}`
- methods: ERM, IRMv1; seeds: 10..14
- parameter block: frozen nonlinear encoder, trainable 65D final head
- checkpoint target-gap integrity: `{provenance['input_verification']['target_gap_verified']}`

## Why This Audit Was Run

The formal/synthetic A/O/Pi construction predates corrected neural CMNIST evidence. This audit tests a restricted frozen-encoder final-head instantiation; it does not replace the frozen 3A--3D theory.

## A Geometry

G1 stable geometry: `{summary['geometry_stability_gate']['pass']}`.

## O_S Geometry

O_S is the source-only concatenation of per-environment risk and IRMv1 penalty gradients. Its ranks and kernels are in `results/geometry_O.csv`.

## Pi Finite Validation

G2 finite source-perturbation prediction: `{summary['finite_prediction_gate']['pass']}`.

## ERM Vs IRMv1 Discrimination

G3: `{summary['method_discrimination_gate']['label']}`. All paired seeds are in `results/paired_method_summary.csv`.

## Gate Verdict

`{summary['overall_verdict']}`

## Interpretation Boundary

This result concerns a frozen-encoder, final-head local response only. It does not establish semantic mechanism discovery, causal recovery, finite-sample theory, universal DG validity, a new algorithm, or novelty.
"""


def run(*, wall_clock_seconds: int = 3600) -> dict[str, Any]:
    started = time.monotonic()
    config = _load_config()
    prereg = write_preregistration(config)
    records, verification = verify_inputs(config)
    all_a: list[dict[str, Any]] = []
    all_o: list[dict[str, Any]] = []
    all_validation: list[dict[str, Any]] = []
    all_mismatch: list[dict[str, Any]] = []
    references: list[dict[str, Any]] = []
    valid, error = True, None
    try:
        for record in records:
            rows = audit_record(record, config, started=started, wall_clock_seconds=wall_clock_seconds)
            all_a.extend(rows[0]); all_o.extend(rows[1]); all_validation.extend(rows[2]); all_mismatch.extend(rows[3]); references.append(rows[4])
    except Exception as exc:
        valid, error = False, f"{type(exc).__name__}: {exc}"
        print(f"[aopi] invalid: {error}", flush=True)
    _write_csv(RESULTS_DIR / "tangent_manifest.csv", tangent_manifest_rows())
    _write_csv(RESULTS_DIR / "geometry_A.csv", all_a)
    _write_csv(RESULTS_DIR / "geometry_O.csv", all_o)
    _write_csv(RESULTS_DIR / "pi_finite_validation.csv", all_validation)
    _write_csv(RESULTS_DIR / "mismatch_by_direction.csv", all_mismatch)
    geometry = geometry_gate(all_a) if all_a else {"pass": False, "reason": "no geometry rows"}
    finite = finite_prediction_gate(all_validation) if all_validation else {"pass": False, "reason": "no finite rows"}
    discrimination = discrimination_gate(all_mismatch, geometry_pass=bool(geometry["pass"]), finite_pass=bool(finite["pass"])) if all_mismatch else {"pass": False, "label": "NONDISCRIMINATIVE", "paired_rows": []}
    _write_csv(RESULTS_DIR / "paired_method_summary.csv", discrimination["paired_rows"])
    verdict = overall_verdict(valid=valid, geometry=geometry, finite=finite, discrimination=discrimination)
    summary = {"task_id": TASK_ID, "valid": valid, "methods": METHODS, "seeds": SEEDS, "geometry_stability_gate": geometry, "finite_prediction_gate": finite, "method_discrimination_gate": discrimination, "primary_mismatch_metric": "frobenius_norm_A_plus_PiO", "paired_method_effects": discrimination.get("paired_rows", []), "overall_verdict": verdict, "interpretation_ceiling": "correct-neural frozen-encoder head-block re-instantiation audit only", "error": error}
    provenance = {"git_head": _git(["rev-parse", "HEAD"]), "git_status": _git(["status", "--short"]), "config_sha256": _sha(CONFIG_PATH), "manifest_sha256": _sha(MANIFEST_PATH), "python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu", "old_results_used_as_inputs": False, "input_verification": verification, "head_references": references, "preregistration": prereg}
    _write_json(OUT_DIR / "provenance.json", provenance)
    _write_json(RESULTS_DIR / "summary.json", summary)
    (OUT_DIR / "report.md").write_text(_report(summary, provenance), encoding="utf-8")
    STATE_DELTA_PATH.write_text(f"""# Proposed State Delta

Task: `{TASK_ID}`

Verdict: `{verdict}`

Proposed result ID: `R-AOPI-CMNIST-HEAD-AUDIT`.

This is a frozen-encoder, 65-dimensional final-head local response audit only. It does not update canonical theorem status. Task 3 scientific verdict remains pending broader authorized evaluation.
""", encoding="utf-8")
    if verdict not in ALLOWED_VERDICTS:
        raise RuntimeError("invalid verdict enum")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wall-clock-seconds", type=int, default=3600)
    args = parser.parse_args()
    summary = run(wall_clock_seconds=int(args.wall_clock_seconds))
    print(f"[aopi] verdict={summary['overall_verdict']}", flush=True)


if __name__ == "__main__":
    main()
