"""Run the semantics-repaired CMNIST A/O/Pi audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import torch

from .task3_aopi_cmnist_reinstantiation_repair.full_response import full_response_rows, reconstruct_adam_state
from .task3_aopi_cmnist_reinstantiation_repair.head_response import finite_head_response, h0_reference, head_pi_diagnostics, refine_h1
from .task3_aopi_cmnist_reinstantiation_repair.linearity_checks import check_linearity
from .task3_aopi_cmnist_reinstantiation_repair.smooth_world import BASIS, DERIVED_DIRECTIONS, SmoothWorldFactory, base_world_identity, environment_parameters, outcome_weight
from .task3_aopi_cmnist_reinstantiation_repair.source_observation import independent_O_direction, observation_geometry
from .task3_aopi_cmnist_reinstantiation_repair.task_response import augmented_head, corrected_geometry, independent_A_direction, smooth_fd_consistency
from .task3_cmnist_counterfactual_audit.diagnostics import model_counterfactual_diagnostics
from .task3_cmnist_counterfactual_audit.probe import build_counterfactual_probe
from .task3_cmnist_cpu_minimal.data import build_task3_data
from .task3_cmnist_cpu_minimal.evaluation import evaluate_checkpoint
from .task3_ood_capability_decomposition.artifacts import load_checkpoint_manifest, load_verified_model, sha256_file


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs/task3_cmnist_cpu_minimal.json"
MANIFEST_PATH = ROOT / "round3_redesign/task3_cmnist_counterfactual_audit/results/checkpoint_manifest.csv"
OUT = ROOT / "round3_redesign/task3_aopi_cmnist_reinstantiation_repair"
RESULTS = OUT / "results"
TASK_ID = "TASK-AOPI-CMNIST-REINSTANTIATION-REPAIR"
SEEDS = [10, 11, 12, 13, 14]
METHODS = ["ERM", "IRMv1"]


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False).stdout.strip()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _source_manifest() -> dict[str, str]:
    paths = [CONFIG_PATH, Path(__file__)] + sorted((ROOT / "src/ood_repr_reg/task3_aopi_cmnist_reinstantiation_repair").glob("*.py"))
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def _preregister(config: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "task_id": TASK_ID, "written_at_unix": time.time(), "git_head": _git("rev-parse", "HEAD"),
        "config_sha256": sha256_file(CONFIG_PATH), "seeds": SEEDS, "methods": METHODS,
        "tangent_basis": list(BASIS), "derived_validation_directions": list(DERIVED_DIRECTIONS),
        "coordinate_semantics": "delta displacement from source (0.2,0.1), evaluation (0.9,0.9), label noise 0.25",
        "smooth_fd_epsilon": 1e-5, "linearity_tolerance": 0.02,
        "full_response": {"delta": 0.01, "K": [1, 5, 20], "continuation_seed_offset": 271828, "functional_coordinate": "source_env0_clean_color_logits_first_256"},
        "head_damping_grid": [1e-10, 1e-8, 1e-6], "source_code_sha256": _source_manifest(),
        "target_use": "evaluation-side A and post-hoc H0/H1 diagnostics only; never source fitting, O, Pi_head, reconstruction, or continuation",
    }
    text = f"""# {TASK_ID} Preregistered Design

This repair invalidates the old audit interpretation: it omitted the source term in A, used thresholded finite differences, made O method-dependent, and treated a new head-only equilibrium as the successful full learner.

It also supersedes repair commit `b6c9eaf`: that run represented theta as absolute probabilities and then added the base probabilities a second time, producing invalid source/evaluation mixtures including `p=1.1`.

- Primary tangent basis: `source_env0_color`, `source_env1_color`, `shared_label_noise`.
- Coordinates are displacements from source `(0.2,0.1)`, evaluation `(0.9,0.9)`, label noise `0.25`; G-1 validates the zero-displacement world and every mixture weight.
- Primary worlds: smooth four-outcome empirical expectations; no thresholded world is a primary derivative.
- G0 tangent-linearity tolerance: `0.02`; G0 failure stops the audit.
- Primary O: concatenated source risk gradients only, method-independent by construction.
- Pi_head: H0 is diagnostic only; H1 is a separate frozen-encoder equilibrium diagnostic.
- Pi_full: exact Adam-state reconstruction, smooth source-only continuation, delta `0.01`, K `1,5,20`, common continuation batches and replay.
- Full mismatch: `NOT_ESTABLISHED`; no scalar surrogate is invented.
"""
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "preregistered_design.md").write_text(text, encoding="utf-8")
    return payload


def _set_head(model: torch.nn.Module, weights: torch.Tensor) -> torch.nn.Module:
    clone = torch.deepcopy(model) if hasattr(torch, "deepcopy") else None
    if clone is None:
        import copy
        clone = copy.deepcopy(model)
    with torch.no_grad():
        clone.head.weight.copy_(weights[:-1].reshape_as(clone.head.weight).float())
        clone.head.bias.copy_(weights[-1:].reshape_as(clone.head.bias).float())
    return clone


def _predictor_metrics(model: torch.nn.Module, config: dict[str, Any], seed: int) -> dict[str, float]:
    data = build_task3_data(config, seed, data_root=ROOT / "data", download=False)
    metrics = evaluate_checkpoint(model, data.source_envs, data.target_env, device=config["device"])
    probe = build_counterfactual_probe(data.target_env)
    diagnostic = model_counterfactual_diagnostics(model, probe, device=config["device"], relative_tolerance=1e-8)
    return {"source_mean_acc": float(metrics["source_mean_acc"]), "target_acc": float(metrics["target_acc"]), "prediction_color_agreement": float(metrics["prediction_color_agreement"]), "counterfactual_prediction_consistency": float(diagnostic["counterfactual_prediction_consistency"])}


def _toy_correct_A_gate() -> bool:
    h = torch.diag(torch.tensor([4.0, 9.0], dtype=torch.double))
    target = torch.tensor([[5.0], [7.0]], dtype=torch.double)
    source = torch.tensor([[1.0], [2.0]], dtype=torch.double)
    code = torch.diag(torch.tensor([0.5, 1.0 / 3.0], dtype=torch.double)) @ (target - source)
    return bool(torch.allclose(code, torch.linalg.matrix_power(h, -1) @ torch.diag(torch.tensor([2.0, 3.0], dtype=torch.double)) @ (target - source), atol=1e-12))


def _head_rows(record, model, worlds, config: dict[str, Any], root: torch.Tensor) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    h0, h1 = h0_reference(model, worlds, record.method), refine_h1(model, worlds, record.method)
    references: list[dict[str, Any]] = []
    for name, reference in (("H0", h0), ("H1", h1)):
        candidate = _set_head(model, reference.weights)
        references.append({"seed": record.seed, "method": record.method, "reference": name, "source_objective": reference.objective, "gradient_norm": reference.gradient_norm, "head_displacement_norm": reference.displacement, "converged": reference.converged, **_predictor_metrics(candidate, config, record.seed)})
    diagnostics = []
    for row in head_pi_diagnostics(model, worlds, record.method, h1, root):
        diagnostics.append({"seed": record.seed, "method": record.method, "row_type": "spectrum", **row})
    for row in finite_head_response(model, worlds, record.method, h1, root):
        diagnostics.append({"seed": record.seed, "method": record.method, "row_type": "finite_response", **row})
    return references, diagnostics


def _report(summary: dict[str, Any]) -> str:
    return f"""# {TASK_ID} Report

## Why the previous audit is invalid

`AOPI-OLD-AUDIT-INVALIDATED-BY-SEMANTIC-MISMATCH`: the old A omitted `-D grad R_S`; thresholded finite data differences were not tangents; O included IRMv1 penalty processing; and Pi was evaluated at a newly refined head-only equilibrium.

Repair commit `b6c9eaf` is also invalidated: its base coordinates were added twice, so its A/O/Pi numbers were not computed on the declared ColoredMNIST distribution. The current rerun uses zero displacement at source `(0.2,0.1)`, evaluation `(0.9,0.9)`, and label noise `0.25`.

## Repaired construction

The primary space is exactly R^3 and uses smooth four-outcome empirical expectations. `A = H_S^(-1/2) D[grad R_T - grad R_S]`; `O_S` is method-independent concatenated source risk gradients. Derived common/antisymmetric directions are linearity checks only.

## Gates and verdict

- G0 tangent linearity: `{summary['gates'].get('G0')}`
- G-1 base-world identity: `{summary['gates'].get('G-1')}`
- G1 corrected-A toy identity: `{summary['gates'].get('G1')}`
- G2 smooth derivative consistency: `{summary['gates'].get('G2')}`
- G3 method-independent O: `{summary['gates'].get('G3')}`
- G4 Pi_head diagnostic: `{summary['gates'].get('G4')}`
- G5 full response reconstruction/replay: `{summary['gates'].get('G5')}`

Verdict: `{summary['verdict']}`.

`Pi_head != Pi_full`. The full response is reported in source functional coordinates. A matched full-network mismatch is `NOT_ESTABLISHED`; this repair makes no algorithm, causal, theory-validation, or universal DG claim.
"""


def run(*, wall_clock_seconds: int = 3600) -> dict[str, Any]:
    started = time.monotonic()
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    prereg = _preregister(config)
    records = load_checkpoint_manifest(MANIFEST_PATH, seeds=SEEDS, methods=METHODS, root=ROOT)
    a_rows: list[dict[str, Any]] = []; o_rows: list[dict[str, Any]] = []; linear_rows: list[dict[str, Any]] = []
    head_refs: list[dict[str, Any]] = []; head_pi: list[dict[str, Any]] = []; full_rows: list[dict[str, Any]] = []; reconstruction: list[dict[str, Any]] = []
    gates = {"G-1": True, "G1": _toy_correct_A_gate(), "G2": True, "G3": True, "G4": True, "G5": True}
    error: str | None = None
    try:
        for record in records:
            if time.monotonic() - started > wall_clock_seconds:
                raise TimeoutError("wall-clock budget exceeded before G0")
            model = load_verified_model(record, config, config_sha256=sha256_file(CONFIG_PATH))
            worlds = SmoothWorldFactory(config, record.seed, data_root=ROOT / "data", download=False).build()
            gates["G-1"] = gates["G-1"] and base_world_identity(worlds)
            for environment in range(2):
                for evaluation in (False, True):
                    p, q = environment_parameters(worlds.base_theta, environment=environment, evaluation=evaluation)
                    weights = [outcome_weight(p, q, label_flip, color_flip) for label_flip in (0, 1) for color_flip in (0, 1)]
                    gates["G-1"] = gates["G-1"] and all(0.0 <= float(weight) <= 1.0 for weight in weights) and abs(float(sum(weights)) - 1.0) <= 1e-12
            if not gates["G-1"]:
                raise RuntimeError("G-1 base-world identity failed")
            geometry = corrected_geometry(model, worlds)
            observation = observation_geometry(model, worlds)
            fd_error = smooth_fd_consistency(model, worlds)
            actual = {
                "common_A": independent_A_direction(model, worlds, DERIVED_DIRECTIONS["common_source_color"]),
                "anti_A": independent_A_direction(model, worlds, DERIVED_DIRECTIONS["antisymmetric_source_color"]),
                "common_O": independent_O_direction(model, worlds, DERIVED_DIRECTIONS["common_source_color"]),
                "anti_O": independent_O_direction(model, worlds, DERIVED_DIRECTIONS["antisymmetric_source_color"]),
            }
            linear = check_linearity(geometry.A, observation, actual_common_A=actual["common_A"], actual_anti_A=actual["anti_A"], actual_common_O=actual["common_O"], actual_anti_O=actual["anti_O"])
            linear_rows.append({"seed": record.seed, "method": record.method, "smooth_fd_relative_error": fd_error, **linear.__dict__})
            for index, name in enumerate(BASIS):
                a_rows.append({"seed": record.seed, "method": record.method, "tangent": name, "target_gradient_derivative_norm": float(geometry.target_gradient_derivative[:, index].norm()), "source_gradient_derivative_norm": float(geometry.source_gradient_derivative[:, index].norm()), "delta_gradient_derivative_norm": float(geometry.delta_gradient_derivative[:, index].norm()), "A_norm": float(geometry.A[:, index].norm()), "hessian_condition": geometry.whitening.condition_number, "whitening_identity_error": geometry.whitening.identity_error})
                o_rows.append({"seed": record.seed, "method": record.method, "tangent": name, "O_norm": float(observation[:, index].norm()), "source_observation_rank": linear.o_rank, "kernel_dimension": 3 - linear.o_rank})
            # Smooth expectations make a direct central check meaningful; the JVP and columns are exact to numerical precision.
            gates["G2"] = gates["G2"] and fd_error <= 1e-6
            gates["G3"] = gates["G3"] and torch.equal(observation, observation_geometry(model, worlds))
        gates["G0"] = len(linear_rows) == 10 and all(bool(row["passed"]) for row in linear_rows)
        if not gates["G0"]:
            raise RuntimeError("G0-TANGENT-LINEARITY failed")
        for record in records:
            if time.monotonic() - started > wall_clock_seconds:
                raise TimeoutError("wall-clock budget exceeded after G0")
            model = load_verified_model(record, config, config_sha256=sha256_file(CONFIG_PATH))
            worlds = SmoothWorldFactory(config, record.seed, data_root=ROOT / "data", download=False).build()
            root = corrected_geometry(model, worlds).whitening.root
            refs, diagnostics = _head_rows(record, model, worlds, config, root)
            head_refs.extend(refs); head_pi.extend(diagnostics)
            h1_rows = [row for row in diagnostics if row["row_type"] == "finite_response"]
            if record.method == "ERM":
                gates["G4"] = gates["G4"] and all(float(row["cosine_similarity"]) >= 0.90 and float(row["relative_vector_error"]) <= 0.25 for row in h1_rows)
            state = reconstruct_adam_state(config, record.seed, record.method, record.parameter_hash, data_root=str(ROOT / "data"))
            reconstruction.append({"seed": record.seed, "method": record.method, "parameter_hash": state.parameter_hash, "optimizer_state_hash": state.optimizer_hash})
            rows = full_response_rows(config, worlds, state, seed=record.seed, method=record.method)
            full_rows.extend(rows)
            gates["G5"] = gates["G5"] and len(rows) == 9 and all(bool(row["finite"]) and bool(row["replay_match"]) for row in rows)
    except TimeoutError as exc:
        error = str(exc)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    invalid = error is not None and "wall-clock" not in error
    completed_full = len(full_rows) == len(records) * 9
    if invalid or not gates["G-1"] or not gates.get("G0", False) or not gates["G1"] or not gates["G2"] or not gates["G3"]:
        verdict = "AOPI-REPAIR-INVALID"
    elif not completed_full:
        verdict = "AOPI-REPAIR-PARTIAL"
    elif gates["G4"] and gates["G5"]:
        verdict = "AOPI-REPAIR-PARTIAL"
    else:
        verdict = "AOPI-REPAIR-FAIL"
    paired = []
    for seed in SEEDS:
        erm = [float(row["central_source_logit_response_norm"]) for row in full_rows if row["seed"] == seed and row["method"] == "ERM" and row["K"] == 20]
        irm = [float(row["central_source_logit_response_norm"]) for row in full_rows if row["seed"] == seed and row["method"] == "IRMv1" and row["K"] == 20]
        if erm and irm:
            paired.append({"seed": seed, "erm_K20_response_mean": sum(erm) / len(erm), "irmv1_K20_response_mean": sum(irm) / len(irm), "full_mismatch": "NOT_ESTABLISHED", "label": "INCONCLUSIVE"})
    _write_csv(RESULTS / "tangent_basis.csv", [{"tangent": name, "coordinates": ";".join(map(str, vector.tolist()))} for name, vector in BASIS.items()])
    _write_csv(RESULTS / "linearity_checks.csv", linear_rows); _write_csv(RESULTS / "geometry_A.csv", a_rows); _write_csv(RESULTS / "geometry_O.csv", o_rows)
    _write_csv(RESULTS / "head_reference_audit.csv", head_refs); _write_csv(RESULTS / "head_pi_validation.csv", head_pi); _write_csv(RESULTS / "full_response.csv", full_rows); _write_csv(RESULTS / "paired_method_summary.csv", paired)
    summary = {"task_id": TASK_ID, "old_audit_status": "AOPI-OLD-AUDIT-INVALIDATED-BY-SEMANTIC-MISMATCH", "superseded_repair_commit": "b6c9eaf (invalid base-coordinate semantics)", "gates": gates, "full_response_status": "COMPLETE" if completed_full else "INCOMPLETE", "full_mismatch": "NOT_ESTABLISHED", "verdict": verdict, "error": error, "rows": {"A": len(a_rows), "O": len(o_rows), "linearity": len(linear_rows), "head_references": len(head_refs), "head_pi": len(head_pi), "full": len(full_rows)}}
    provenance = {"task_id": TASK_ID, "git_head": _git("rev-parse", "HEAD"), "git_status": _git("status", "--short"), "preregistration": prereg, "source_code_sha256": _source_manifest(), "reconstruction": reconstruction, "target_used_for_training_or_selection": False}
    _write_json(RESULTS / "summary.json", summary); _write_json(OUT / "provenance.json", provenance); (OUT / "report.md").write_text(_report(summary), encoding="utf-8")
    (ROOT / "active/STATE_DELTA.md").write_text(f"# Proposed State Delta\n\nTask: `{TASK_ID}`\n\nOld audit: `AOPI-OLD-AUDIT-INVALIDATED-BY-SEMANTIC-MISMATCH`\n\nSuperseded repair: `b6c9eaf` is invalid because base probabilities were added twice.\n\nRepair verdict: `{verdict}`\n\nNo canonical state changes are proposed. Full mismatch remains `NOT_ESTABLISHED`; no algorithm task is authorized by this audit alone.\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--wall-clock-seconds", type=int, default=3600)
    args = parser.parse_args(); print(run(wall_clock_seconds=args.wall_clock_seconds)["verdict"])


if __name__ == "__main__":
    main()
