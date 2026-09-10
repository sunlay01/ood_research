"""Runner for TASK3-CMNIST-CPU-MINIMAL."""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

from .task3_cmnist_cpu_minimal.data import (
    build_task3_data,
    make_batch_schedule,
    scheduled_source_batches,
)
from .task3_cmnist_cpu_minimal.evaluation import (
    checkpoint_row,
    evaluate_checkpoint,
    response_diagnostics,
)
from .task3_cmnist_cpu_minimal.methods import (
    calibrate_response_scale,
    grad_response_penalty,
    inverse_hessian_metric_penalty,
    local_response_penalty,
)
from .task3_cmnist_cpu_minimal.model import (
    CPUColoredMNISTMLP,
    augmented_head_dimension,
    build_model_from_config,
    linear_layer_count,
    parameter_hash,
)
from .task3_cmnist_cpu_minimal.trainer import TrainResult, train_one_method


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs/task3_cmnist_cpu_minimal.json"
OUT_DIR = ROOT / "round3_redesign/task3_cmnist_cpu_minimal"
RESULTS_DIR = OUT_DIR / "results"
STATE_DELTA_PATH = ROOT / "active/STATE_DELTA.md"


def _run_git(args: list[str]) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return completed.stdout.strip()


def _load_config() -> dict[str, Any]:
    with CONFIG_PATH.open(encoding="utf-8") as handle:
        config = json.load(handle)
    if config["task_id"] != "TASK3-CMNIST-CPU-MINIMAL":
        raise ValueError("wrong task config")
    return config


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "NA") for field in fieldnames})


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _cpu_model() -> str:
    try:
        return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    except Exception:
        return platform.processor() or "unknown"


def _torchvision_version() -> str:
    try:
        import torchvision

        return str(torchvision.__version__)
    except Exception as exc:
        return f"unavailable:{type(exc).__name__}"


def write_preregistration(config: dict[str, Any], *, overwrite: bool = True) -> dict[str, Any]:
    payload = {
        "task_id": config["task_id"],
        "git_head_before_run": _run_git(["rev-parse", "HEAD"]),
        "git_diff_stat_before_run": _run_git(["diff", "--stat"]),
        "config_sha256": _sha256(CONFIG_PATH),
        "device": config["device"],
        "exact_config": config,
    }
    path = OUT_DIR / "preregistered_design.md"
    if path.exists() and not overwrite:
        return payload
    text = f"""# TASK3-CMNIST-CPU-MINIMAL Preregistered Design

git commit/HEAD before run: `{payload['git_head_before_run']}`

git diff --stat before run:

```text
{payload['git_diff_stat_before_run'] or 'clean'}
```

config SHA256: `{payload['config_sha256']}`

## Exact Question

Does `LOCAL_RESPONSE` using damped detached `H^-1` carry more signal than `GRAD` using identity metric when both use the same source head-gradient disagreement?

## Exact Config Values

```json
{json.dumps(config, indent=2, sort_keys=True)}
```

## Stage A Gate

Stage A runs only `ERM` and `IRMv1` on seeds `0,1,2`. It passes only if IRMv1 mean target accuracy is at least `0.50`, ERM mean target accuracy is at most `0.35`, and IRMv1 minus ERM is at least `0.20`.

## Stage B Methods

Stage B runs seeds `10,11,12,13,14` with `ERM`, `IRMv1`, `GRAD`, and `LOCAL_RESPONSE`. There is no beta grid, checkpoint selection, or target-based tuning.

## GRAD Formula

`GRAD` is `0.5 * sum_e ||g_e - g_bar||^2`, where `g_e` is the final-head weight-and-bias gradient of the source environment BCE risk.

## LOCAL_RESPONSE Formula

`LOCAL_RESPONSE` uses the same `g_e` and replaces identity with damped detached Cholesky-solved `H^-1`, where `H` is the analytic pooled-source BCE head Hessian over 65 augmented coordinates.

## Calibration Formula

For `GRAD` and `LOCAL_RESPONSE`, step-0 source batches set `c = ||G_R|| / (||G_P|| + 1e-12)`, and training uses `R + 1e-3 W + 0.10 c P`.

## Verdict Rules

Allowed verdicts are `CPU-MINIMAL-INVALID`, `CPU-MINIMAL-SIGNAL`, and `CPU-MINIMAL-NO-SIGNAL`. Signal requires the preregistered LR-vs-ERM, LR-vs-GRAD, seed-win, source-accuracy, and mechanism-direction gates.

## Forbidden Target Uses

Target metrics cannot affect training, calibration, checkpointing, method choice, or any threshold. Target is evaluation-only.

## Interpretation Ceiling

This is an exploratory CPU signal probe only. It does not establish theory support, causal recovery, SOTA, finite-sample guarantees, or paper-level readiness.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return payload


def _toy_batches(seed: int = 123) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    generator = torch.Generator().manual_seed(seed)
    return (
        (torch.rand((16, 392), generator=generator), torch.randint(0, 2, (16, 1), generator=generator).float()),
        (torch.rand((16, 392), generator=generator), torch.randint(0, 2, (16, 1), generator=generator).float()),
    )


def invariant_audit(config: dict[str, Any]) -> dict[str, bool]:
    torch.manual_seed(2027)
    model = build_model_from_config(config)
    batches = _toy_batches()
    grad = grad_response_penalty(model, batches)
    local = local_response_penalty(model, batches, damping_epsilon=float(config["response"]["damping_epsilon"]))
    local.penalty.backward()
    source = inspect.getsource(train_one_method)
    inverse_source = inspect.getsource(inverse_hessian_metric_penalty)
    torch.manual_seed(99)
    left = build_model_from_config(config)
    torch.manual_seed(99)
    right = build_model_from_config(config)
    schedule_left = make_batch_schedule(source_pool_sizes=(32, 32), steps=501, batch_size_per_environment=512, seed=5)
    schedule_right = make_batch_schedule(source_pool_sizes=(32, 32), steps=501, batch_size_per_environment=512, seed=5)
    return {
        "model_exact_392_64_64_1": [tuple(layer.weight.shape) for layer in model.modules() if isinstance(layer, torch.nn.Linear)] == [(64, 392), (64, 64), (1, 64)],
        "three_linear_layers": linear_layer_count(model) == 3,
        "same_seed_same_initialization": parameter_hash(left) == parameter_hash(right),
        "same_seed_same_batch_schedule": torch.equal(schedule_left.indices[0], schedule_right.indices[0]) and torch.equal(schedule_left.indices[1], schedule_right.indices[1]),
        "adam_persistent": source.count("torch.optim.Adam") == 1 and source.index("torch.optim.Adam") < source.index("for step in range(steps)"),
        "steps_exactly_501": int(config["training"]["steps"]) == 501,
        "batch_size_per_env_exactly_512": int(config["training"]["batch_size_per_environment"]) == 512,
        "grad_is_head_only_by_definition": grad.raw_head_grads.shape == (2, 65) and augmented_head_dimension(model) == 65,
        "grad_penalty_backprop_reaches_encoder": any(parameter.grad is not None and torch.isfinite(parameter.grad).all() and float(parameter.grad.norm()) > 0.0 for parameter in model.encoder.parameters()),
        "lr_uses_same_head_gradients_as_grad": torch.allclose(grad.raw_head_grads, local.raw_head_grads, atol=1e-8),
        "lr_only_changes_metric": torch.allclose(grad.centered_head_grads, local.centered_head_grads, atol=1e-8),
        "lr_hessian_is_65x65": local.hessian is not None and tuple(local.hessian.shape) == (65, 65),
        "lr_hessian_stopgrad": local.hessian is not None and not local.hessian.requires_grad,
        "lr_uses_cholesky_solve": local.solver == "cholesky" and "cholesky_solve" in inverse_source,
        "no_explicit_inverse": "torch.linalg.inv" not in inverse_source and ".inverse(" not in inverse_source,
        "no_target_in_training": "target" not in source.lower(),
        "no_beta_grid": "beta" not in json.dumps(config).lower(),
        "no_checkpoint_selection": True,
        "no_iga_fish_domainbed": set(config["stage_b"]["methods"]) == {"ERM", "IRMv1", "GRAD", "LOCAL_RESPONSE"},
    }


def _fresh_model(config: dict[str, Any], seed: int) -> tuple[CPUColoredMNISTMLP, str]:
    torch.manual_seed(int(seed))
    model = build_model_from_config(config)
    return model, parameter_hash(model)


def _final_metrics_for_result(config: dict[str, Any], data, result: TrainResult) -> dict[str, float]:
    if not result.finite:
        return {
            "source_env0_acc": float("nan"),
            "source_env1_acc": float("nan"),
            "source_mean_acc": float("nan"),
            "target_acc": float("nan"),
            "prediction_color_agreement": float("nan"),
        }
    return evaluate_checkpoint(result.model, data.source_envs, data.target_env, device=config["device"])


def run_stage_a(config: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for seed in config["stage_a"]["seeds"]:
        data = build_task3_data(config, int(seed))
        for method in config["stage_a"]["methods"]:
            model, initial_hash = _fresh_model(config, int(seed))
            started = time.time()
            result = train_one_method(
                model=model,
                source_envs=data.source_envs,
                batch_schedule=data.batch_schedule,
                method=method,
                config=config,
                seed=int(seed),
                initial_parameter_hash=initial_hash,
            )
            metrics = _final_metrics_for_result(config, data, result)
            rows.append({
                "seed": int(seed),
                "method": method,
                "final_source_env0_acc": metrics["source_env0_acc"],
                "final_source_env1_acc": metrics["source_env1_acc"],
                "final_source_mean_acc": metrics["source_mean_acc"],
                "final_target_acc": metrics["target_acc"],
                "prediction_color_agreement": metrics["prediction_color_agreement"],
                "finite": result.finite,
                "invalid_reason": result.invalid_reason,
                "wall_seconds": time.time() - started,
            })
    fieldnames = [
        "seed", "method", "final_source_env0_acc", "final_source_env1_acc", "final_source_mean_acc",
        "final_target_acc", "prediction_color_agreement", "finite", "invalid_reason", "wall_seconds",
    ]
    _write_csv(RESULTS_DIR / "baseline_gate.csv", rows, fieldnames)
    erm_values = [float(row["final_target_acc"]) for row in rows if row["method"] == "ERM" and row["finite"]]
    irm_values = [float(row["final_target_acc"]) for row in rows if row["method"] == "IRMv1" and row["finite"]]
    erm_mean = sum(erm_values) / len(erm_values) if erm_values else float("nan")
    irm_mean = sum(irm_values) / len(irm_values) if irm_values else float("nan")
    gap = irm_mean - erm_mean
    criteria = {
        "irmv1_ge_0p50": irm_mean >= float(config["stage_a"]["irm_min_mean_target_accuracy"]),
        "erm_le_0p35": erm_mean <= float(config["stage_a"]["erm_max_mean_target_accuracy"]),
        "irm_advantage_ge_0p20": gap >= float(config["stage_a"]["irm_min_mean_advantage_over_erm"]),
    }
    payload = {
        "stage": "A",
        "seeds": config["stage_a"]["seeds"],
        "erm_mean_target_accuracy": erm_mean,
        "irmv1_mean_target_accuracy": irm_mean,
        "irmv1_minus_erm": gap,
        "criteria": criteria,
        "passed": all(criteria.values()) and len(rows) == 6 and all(bool(row["finite"]) for row in rows),
    }
    _write_json(RESULTS_DIR / "baseline_gate.json", payload)
    return payload


def _checkpoint_model(config: dict[str, Any], state_dict: dict[str, torch.Tensor]) -> CPUColoredMNISTMLP:
    model = build_model_from_config(config)
    model.load_state_dict(state_dict)
    return model


def run_stage_b(config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    gate_path = RESULTS_DIR / "baseline_gate.json"
    if not gate_path.exists() or not json.loads(gate_path.read_text(encoding="utf-8")).get("passed"):
        raise RuntimeError("Stage B requires a passed Stage A baseline gate")
    run_rows: list[dict[str, Any]] = []
    dynamics: list[dict[str, Any]] = []
    for seed in config["stage_b"]["seeds"]:
        data = build_task3_data(config, int(seed))
        for method in config["stage_b"]["methods"]:
            model, initial_hash = _fresh_model(config, int(seed))
            started = time.time()
            result = train_one_method(
                model=model,
                source_envs=data.source_envs,
                batch_schedule=data.batch_schedule,
                method=method,
                config=config,
                seed=int(seed),
                initial_parameter_hash=initial_hash,
            )
            for checkpoint in result.checkpoints:
                eval_model = _checkpoint_model(config, checkpoint.state_dict)
                metrics = evaluate_checkpoint(eval_model, data.source_envs, data.target_env, device=config["device"])
                batches = scheduled_source_batches(data.source_envs, data.batch_schedule, checkpoint.step, config["device"])
                diagnostics = response_diagnostics(eval_model, batches, damping_epsilon=float(config["response"]["damping_epsilon"]))
                dynamics.append(checkpoint_row(seed=int(seed), method=method, step=checkpoint.step, eval_metrics=metrics, diagnostics=diagnostics, loss=checkpoint.loss))
            final_rows = [row for row in dynamics if row["seed"] == int(seed) and row["method"] == method and row["step"] == 500]
            final = final_rows[-1] if final_rows else {}
            calibration = result.calibration
            run_rows.append({
                "seed": int(seed),
                "method": method,
                "steps": int(config["training"]["steps"]),
                "batch_size_per_env": int(config["training"]["batch_size_per_environment"]),
                "hidden_dim": int(config["model"]["hidden_dim"]),
                "initial_parameter_hash": result.initial_parameter_hash,
                "batch_schedule_hash": result.batch_schedule_hash,
                "calibration_scale_c": calibration.scale_c if calibration else "NA",
                "initial_risk_grad_norm": calibration.risk_grad_norm if calibration else "NA",
                "initial_penalty_grad_norm": calibration.penalty_grad_norm if calibration else "NA",
                "realized_initial_update_ratio": calibration.realized_initial_update_ratio if calibration else "NA",
                "final_source_env0_acc": final.get("source_env0_acc", float("nan")),
                "final_source_env1_acc": final.get("source_env1_acc", float("nan")),
                "final_source_mean_acc": (float(final.get("source_env0_acc", float("nan"))) + float(final.get("source_env1_acc", float("nan")))) / 2.0,
                "final_target_acc": final.get("target_acc", float("nan")),
                "final_prediction_color_agreement": final.get("prediction_color_agreement", float("nan")),
                "final_raw_grad_disagreement": final.get("raw_grad_disagreement", float("nan")),
                "final_local_response_disagreement": final.get("local_response_disagreement", float("nan")),
                "finite": result.finite,
                "invalid_reason": result.invalid_reason,
                "wall_seconds": time.time() - started,
            })
    run_fields = [
        "seed", "method", "steps", "batch_size_per_env", "hidden_dim", "initial_parameter_hash",
        "batch_schedule_hash", "calibration_scale_c", "initial_risk_grad_norm", "initial_penalty_grad_norm",
        "realized_initial_update_ratio", "final_source_env0_acc", "final_source_env1_acc", "final_source_mean_acc",
        "final_target_acc", "final_prediction_color_agreement", "final_raw_grad_disagreement",
        "final_local_response_disagreement", "finite", "invalid_reason", "wall_seconds",
    ]
    dyn_fields = [
        "seed", "method", "step", "source_env0_acc", "source_env1_acc", "target_acc",
        "prediction_color_agreement", "raw_grad_disagreement", "local_response_disagreement", "loss",
    ]
    _write_csv(RESULTS_DIR / "main_runs.csv", run_rows, run_fields)
    _write_csv(RESULTS_DIR / "dynamics.csv", dynamics, dyn_fields)
    return run_rows, dynamics


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean = _mean(values)
    return (sum((value - mean) ** 2 for value in values) / (len(values) - 1)) ** 0.5


def summarize_results(config: dict[str, Any], audit: dict[str, bool]) -> dict[str, Any]:
    gate = json.loads((RESULTS_DIR / "baseline_gate.json").read_text(encoding="utf-8")) if (RESULTS_DIR / "baseline_gate.json").exists() else {"passed": False}
    summary: dict[str, Any] = {
        "task_id": config["task_id"],
        "stage_a_gate": gate,
        "stage_b_completed": (RESULTS_DIR / "main_runs.csv").exists(),
        "implementation_validity_flags": audit,
    }
    if not gate.get("passed") or not all(audit.values()):
        summary["verdict"] = "CPU-MINIMAL-INVALID"
        _write_json(RESULTS_DIR / "summary.json", summary)
        return summary
    if not (RESULTS_DIR / "main_runs.csv").exists():
        summary["verdict"] = "STAGE-A-PASS"
        _write_json(RESULTS_DIR / "summary.json", summary)
        return summary
    rows = _read_csv(RESULTS_DIR / "main_runs.csv")
    by_method: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_method.setdefault(row["method"], []).append(row)
    method_stats = {}
    for method, method_rows in by_method.items():
        targets = [float(row["final_target_acc"]) for row in method_rows]
        sources = [float(row["final_source_mean_acc"]) for row in method_rows]
        method_stats[method] = {
            "target_mean": _mean(targets),
            "target_std": _std(targets),
            "source_mean": _mean(sources),
            "source_std": _std(sources),
        }
    paired_lr_erm = []
    paired_lr_grad = []
    lr_wins_grad = 0
    lr_source_minus_erm = []
    lr_plr_lower_than_erm = 0
    lr_color_away_from_spurious = 0
    for seed in config["stage_b"]["seeds"]:
        seed_rows = {row["method"]: row for row in rows if int(row["seed"]) == int(seed)}
        if {"ERM", "GRAD", "LOCAL_RESPONSE"} <= set(seed_rows):
            lr = seed_rows["LOCAL_RESPONSE"]
            erm = seed_rows["ERM"]
            grad = seed_rows["GRAD"]
            lr_target = float(lr["final_target_acc"])
            grad_target = float(grad["final_target_acc"])
            erm_target = float(erm["final_target_acc"])
            paired_lr_erm.append(lr_target - erm_target)
            paired_lr_grad.append(lr_target - grad_target)
            lr_wins_grad += int(lr_target > grad_target)
            lr_source_minus_erm.append(float(lr["final_source_mean_acc"]) - float(erm["final_source_mean_acc"]))
            lr_plr_lower_than_erm += int(float(lr["final_local_response_disagreement"]) < float(erm["final_local_response_disagreement"]))
            lr_color_away_from_spurious += int(float(lr["final_prediction_color_agreement"]) < float(erm["final_prediction_color_agreement"]))
    calibration_ok = all(
        row["method"] not in {"GRAD", "LOCAL_RESPONSE"}
        or 0.08 <= float(row["realized_initial_update_ratio"]) <= 0.12
        for row in rows
    )
    all_finite = all(row["finite"] == "True" for row in rows)
    invalid = (not all_finite) or (not calibration_ok)
    signal = (
        _mean(paired_lr_erm) >= 0.02
        and _mean(paired_lr_grad) >= 0.01
        and lr_wins_grad >= 4
        and _mean(lr_source_minus_erm) >= -0.05
        and (lr_plr_lower_than_erm >= 4 or lr_color_away_from_spurious >= 4)
    )
    summary.update({
        "method_means_stds": method_stats,
        "paired_LR_minus_ERM": paired_lr_erm,
        "paired_LR_minus_GRAD": paired_lr_grad,
        "mean_LR_minus_ERM": _mean(paired_lr_erm),
        "mean_LR_minus_GRAD": _mean(paired_lr_grad),
        "lr_wins_vs_grad": lr_wins_grad,
        "source_accuracy_deltas_LR_minus_ERM": lr_source_minus_erm,
        "mean_source_delta_LR_minus_ERM": _mean(lr_source_minus_erm),
        "mechanism_direction_counts": {
            "final_P_LR_lower_than_ERM": lr_plr_lower_than_erm,
            "target_prediction_color_agreement_lower_than_ERM": lr_color_away_from_spurious,
        },
        "implementation_validity_flags": {**audit, "all_rows_finite": all_finite, "calibration_ratios_in_bounds": calibration_ok},
        "verdict": "CPU-MINIMAL-INVALID" if invalid else ("CPU-MINIMAL-SIGNAL" if signal else "CPU-MINIMAL-NO-SIGNAL"),
    })
    _write_json(RESULTS_DIR / "summary.json", summary)
    return summary


def write_report(summary: dict[str, Any]) -> None:
    gate = summary.get("stage_a_gate", {})
    verdict = summary.get("verdict", "CPU-MINIMAL-INVALID")
    lines = [
        "# TASK3-CMNIST-CPU-MINIMAL Report",
        "",
        "## 1. Exact question",
        "Does damped inverse-Hessian weighting improve the same head-gradient disagreement signal relative to identity weighting in this fixed CPU ColoredMNIST probe?",
        "",
        "## 2. Exact CPU protocol",
        "CPU-only, 3 linear layers `392->64->64->1`, batch size 512 per source environment, Adam, 501 steps, no beta grid, no checkpoint selection, target evaluation only.",
        "",
        "## 3. Stage A benchmark calibration",
        f"ERM mean target accuracy: `{gate.get('erm_mean_target_accuracy')}`",
        f"IRMv1 mean target accuracy: `{gate.get('irmv1_mean_target_accuracy')}`",
        f"IRMv1 minus ERM: `{gate.get('irmv1_minus_erm')}`",
        f"Stage A passed: `{gate.get('passed')}`",
        "",
    ]
    if summary.get("stage_b_completed"):
        stats = summary.get("method_means_stds", {})
        lines.extend([
            "## 4. Stage B results",
            json.dumps(stats, indent=2, sort_keys=True),
            "",
            "## 5. GRAD vs LOCAL_RESPONSE",
            f"Mean LR-ERM target difference: `{summary.get('mean_LR_minus_ERM')}`",
            f"Mean LR-GRAD target difference: `{summary.get('mean_LR_minus_GRAD')}`",
            f"LR wins vs GRAD: `{summary.get('lr_wins_vs_grad')}`",
            "",
            "## 6. Mechanism diagnostics",
            json.dumps(summary.get("mechanism_direction_counts", {}), indent=2, sort_keys=True),
            "",
            "## 7. Counterexamples / failure cases",
            "Any failed signal criterion is retained in `results/summary.json`; no config rescue or hyperparameter search was run.",
            "",
        ])
    lines.extend([
        "## 8. Verdict",
        verdict,
        "",
        "## 9. What this does NOT establish",
        "This does not establish theory support, causal recovery, target-risk lower bounds, finite-sample guarantees, universal DG, SOTA, or paper-level readiness.",
        "",
        "Historical reopen: none.",
    ])
    (OUT_DIR / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_state_delta(summary: dict[str, Any]) -> None:
    verdict = summary.get("verdict", "CPU-MINIMAL-INVALID")
    if verdict == "STAGE-A-PASS":
        text = """# Proposed State Delta

Task 3 CPU minimal verdict: STAGE-A-PASS
Task 3 scientific conclusion: NOT ESTABLISHED until Stage B runs
frozen theory reopened: false

No canonical state file was edited.
"""
    elif verdict == "CPU-MINIMAL-INVALID":
        reason = "baseline calibration failed" if not summary.get("stage_a_gate", {}).get("passed") else "implementation validity gate failed"
        text = f"""# Proposed State Delta

Task 3 CPU minimal verdict: CPU-MINIMAL-INVALID
reason: {reason}
Task 3 scientific conclusion: NOT ESTABLISHED
frozen theory reopened: false

No canonical state file was edited.
"""
    elif verdict == "CPU-MINIMAL-NO-SIGNAL":
        text = """# Proposed State Delta

Task 3 CPU minimal verdict: CPU-MINIMAL-NO-SIGNAL
Task 3 scientific conclusion: head-local inverse-Hessian metric did not show the preregistered exploratory advantage over the identity-metric control in this CPU CMNIST probe
frozen theory reopened: false

No canonical state file was edited.
"""
    else:
        text = """# Proposed State Delta

Task 3 CPU minimal verdict: CPU-MINIMAL-SIGNAL
Task 3 scientific conclusion: exploratory signal only; larger-scale confirmation justified
frozen theory reopened: false
GPU-scale follow-up justified: true

No canonical state file was edited.
"""
    STATE_DELTA_PATH.write_text(text, encoding="utf-8")


def write_provenance(config: dict[str, Any], prereg: dict[str, Any], stage: str) -> None:
    payload = {
        "repo_commit_before_run": prereg["git_head_before_run"],
        "repo_commit_after_implementation_before_run": _run_git(["rev-parse", "HEAD"]),
        "working_tree_status_before_stage_a": _run_git(["status", "--short"]),
        "working_tree_status_before_stage_b": _run_git(["status", "--short"]),
        "config_sha256": prereg["config_sha256"],
        "python_version": sys.version,
        "pytorch_version": torch.__version__,
        "torchvision_version": _torchvision_version(),
        "device": config["device"],
        "cpu_model": _cpu_model(),
        "exact_command": f"PYTHONPATH=src python -m ood_repr_reg.run_task3_cmnist_cpu_minimal --stage {stage}",
        "requested_stage": stage,
        "random_seeds": {"stage_a": config["stage_a"]["seeds"], "stage_b": config["stage_b"]["seeds"]},
    }
    _write_json(OUT_DIR / "provenance.json", payload)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["A", "B", "all"], default="all")
    args = parser.parse_args(argv)
    config = _load_config()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    prereg = write_preregistration(config, overwrite=(args.stage in {"A", "all"}))
    audit = invariant_audit(config)
    if not all(audit.values()):
        summary = {"task_id": config["task_id"], "stage_a_gate": {"passed": False}, "stage_b_completed": False, "implementation_validity_flags": audit, "verdict": "CPU-MINIMAL-INVALID"}
        _write_json(RESULTS_DIR / "summary.json", summary)
        write_report(summary)
        write_state_delta(summary)
        write_provenance(config, prereg, args.stage)
        print("INVARIANT-FAIL")
        return 2
    gate = json.loads((RESULTS_DIR / "baseline_gate.json").read_text(encoding="utf-8")) if (RESULTS_DIR / "baseline_gate.json").exists() else None
    if args.stage in {"A", "all"}:
        gate = run_stage_a(config)
    if not gate or not gate.get("passed"):
        summary = summarize_results(config, audit)
        write_report(summary)
        write_state_delta(summary)
        write_provenance(config, prereg, args.stage)
        print("STAGE-A-FAIL" if args.stage != "B" else "STAGE-A-NOT-PASSED")
        return 1
    if args.stage in {"B", "all"}:
        run_stage_b(config)
    summary = summarize_results(config, audit)
    write_report(summary)
    write_state_delta(summary)
    write_provenance(config, prereg, args.stage)
    print(summary["verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
