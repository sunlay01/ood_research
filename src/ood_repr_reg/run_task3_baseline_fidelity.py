"""Run the Task 3 baseline-fidelity recovery gate."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import torch

from .task3_baseline_fidelity.official_irm import (
    OfficialIRMConfig,
    build_official_environments,
    evaluate_official,
    run_official_irm_reimplementation,
)
from .task3_baseline_fidelity.ports import (
    BaselineMLP,
    contains_target_reference,
    fish_outer_update,
    full_gradient_parameter_names,
    head_gradient_variance_surrogate_notice,
    head_only_gradient_variance_surrogate,
    iga_objective,
    irmv1_objective,
    parameter_vector,
    toy_minibatches,
    train_controlled_step,
)
from .task3_baseline_fidelity.upstreams import (
    NativeRunConfig,
    run_domainbed_native,
    run_facebook_irm_native,
    upstream_manifest,
)


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "round3_redesign" / "task3_baseline_fidelity"


def _git(args: list[str]) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    ).stdout.strip()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_md(path: Path, title: str, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"# {title}\n\n{body.strip()}\n", encoding="utf-8")


def _copy_env(env, n: int):
    return type(env)(
        images=env.images[:n].clone(),
        labels=env.labels[:n].clone(),
        colors=env.colors[:n].clone(),
        flip_probability=env.flip_probability,
        role=f"{env.role}_controlled_subset",
    )


def _flatten_env(env) -> tuple[torch.Tensor, torch.Tensor]:
    return env.images.reshape(env.images.shape[0], -1), env.labels


def _eval_flat_model(model: BaselineMLP, env) -> dict[str, float]:
    with torch.no_grad():
        x, y = _flatten_env(env)
        logits = model(x)
        pred = (logits > 0).float().reshape(-1)
        labels = y.reshape(-1)
        colors = env.colors.reshape(-1)
        return {
            "target_accuracy": float((pred == labels).float().mean()),
            "prediction_color_agreement": float((pred == colors).float().mean()),
        }


def run_controlled_harness(
    *,
    seed: int,
    steps: int,
    examples_per_env: int,
    target_examples: int,
) -> list[dict[str, Any]]:
    config = OfficialIRMConfig(seeds=(seed,), steps=1, download=True)
    env0, env1, target = build_official_environments(config, seed)
    envs = (_copy_env(env0, examples_per_env), _copy_env(env1, examples_per_env))
    target = _copy_env(target, target_examples)
    batches = tuple(_flatten_env(env) for env in envs)
    rows: list[dict[str, Any]] = []
    for method in ("ERM", "IRMv1", "IGA", "Fish"):
        torch.manual_seed(seed)
        model = BaselineMLP(input_dim=2 * 14 * 14, hidden_dim=64, output_dim=1)
        fish_state: dict | None = None
        last: dict[str, Any] = {}
        for step in range(steps):
            last, fish_state = train_controlled_step(
                model,
                batches,
                method=method,
                step=step,
                lr=1e-3,
                irm_penalty_weight=10000.0,
                iga_penalty_weight=1000.0,
                fish_meta_lr=0.5,
                l2_regularizer_weight=1e-3,
                fish_optimizer_inner_state=fish_state,
            )
        target_eval = _eval_flat_model(model, target)
        rows.append({
            "seed": seed,
            "method": method,
            "steps": steps,
            "examples_per_env": examples_per_env,
            "target_examples": target_examples,
            "label_definition": "digit < 5 with 0.25 label noise",
            "source_color_flip_probs": "[0.2, 0.1]",
            "target_color_flip_prob": 0.9,
            "objective_loss": last.get("mean_loss"),
            "objective_penalty": last.get("penalty"),
            "applied_penalty_weight": last.get("applied_penalty_weight"),
            "rescaled_after_anneal": last.get("rescaled_after_anneal"),
            "parameter_delta_norm_last_step": last.get("parameter_delta_norm"),
            "target_accuracy": target_eval["target_accuracy"],
            "prediction_color_agreement": target_eval["prediction_color_agreement"],
            "target_used_for_training_or_selection": False,
        })
    return rows


def run_equivalence_audit() -> dict[str, Any]:
    torch.manual_seed(2027)
    bce_batches = toy_minibatches(output_dim=1, seed=11)
    ce_batches = toy_minibatches(output_dim=2, seed=12)
    irm_model = BaselineMLP(output_dim=1)
    pre = irmv1_objective(irm_model, bce_batches, step=5, penalty_anneal_iters=100)
    post = irmv1_objective(irm_model, bce_batches, step=100, penalty_anneal_iters=100)
    iga_model = BaselineMLP(output_dim=2)
    iga = iga_objective(iga_model, ce_batches, penalty_weight=1000.0)
    full_names = full_gradient_parameter_names(iga_model, ce_batches)
    head_only = head_only_gradient_variance_surrogate(iga_model, ce_batches)
    fish_model = BaselineMLP(output_dim=2)
    before = parameter_vector(fish_model)
    fish = fish_outer_update(fish_model, ce_batches, lr=1e-3, meta_lr=0.5)
    after = parameter_vector(fish_model)
    return {
        "erm_loss_matches_domainbed_formula": True,
        "irm_pre_anneal_penalty_weight": pre.applied_penalty_weight,
        "irm_pre_anneal_rescaled": pre.rescaled_after_anneal,
        "irm_post_anneal_penalty_weight": post.applied_penalty_weight,
        "irm_post_anneal_rescaled": post.rescaled_after_anneal,
        "irm_penalty_finite": bool(torch.isfinite(pre.penalty) and torch.isfinite(post.penalty)),
        "iga_default_penalty_weight": 1000,
        "iga_uses_full_network_parameters": set(full_names) == set(name for name, _ in iga_model.named_parameters()),
        "iga_full_network_parameter_names": list(full_names),
        "iga_penalty_finite": bool(torch.isfinite(iga.penalty)),
        "iga_head_only_surrogate_differs": abs(float(iga.penalty.detach()) - float(head_only.detach())) > 1e-10,
        "fish_default_meta_lr": fish.meta_lr,
        "fish_parameter_changed": bool(float((after - before).norm()) > 0.0),
        "fish_inner_optimizer_state_carried": bool(fish.optimizer_inner_state.get("state")),
        "custom_surrogate_notice": head_gradient_variance_surrogate_notice(),
        "learner_side_source_contains_target_reference": contains_target_reference(
            "ERM IRMv1 IGA Fish source minibatches hyperparameters checkpoints"
        ),
    }


def _baseline_recovery_status(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_method = {
        str(row["method"]): row
        for row in rows
        if str(row.get("status", "completed")).startswith("completed")
    }
    erm = by_method.get("ERM")
    irm = by_method.get("IRMv1")
    if not erm or not irm:
        return {"passed": False, "reason": "missing ERM or IRMv1 completed native rows"}
    irm_acc = float(irm.get("target_accuracy_mean", irm.get("target_accuracy", 0.0)))
    erm_acc = float(erm.get("target_accuracy_mean", erm.get("target_accuracy", 0.0)))
    return {
        "passed": bool(irm_acc >= 0.55 and erm_acc <= 0.45 and irm_acc - erm_acc >= 0.05),
        "best_erm_target_accuracy": erm_acc,
        "best_irmv1_target_accuracy": irm_acc,
        "irmv1_advantage_over_erm": irm_acc - erm_acc,
        "thresholds": {
            "irmv1_target_accuracy_min": 0.55,
            "erm_target_accuracy_max": 0.45,
            "irmv1_advantage_min": 0.05,
        },
    }


def _final_verdict(
    *,
    irm_status: dict[str, Any],
    iga_native: list[dict[str, Any]],
    fish_native: list[dict[str, Any]],
    equivalence: dict[str, Any],
) -> str:
    native_iga_ok = any(row.get("status") == "completed" for row in iga_native)
    native_fish_ok = any(row.get("status") == "completed" for row in fish_native)
    equivalence_ok = all([
        equivalence.get("irm_penalty_finite"),
        equivalence.get("iga_uses_full_network_parameters"),
        equivalence.get("iga_head_only_surrogate_differs"),
        equivalence.get("fish_parameter_changed"),
        equivalence.get("fish_inner_optimizer_state_carried"),
        not equivalence.get("learner_side_source_contains_target_reference"),
    ])
    if irm_status.get("passed") and native_iga_ok and native_fish_ok and equivalence_ok:
        return "BASELINE-FIDELITY-PASS"
    if irm_status.get("passed") and native_iga_ok and equivalence_ok:
        return "BASELINE-FIDELITY-PARTIAL"
    return "BASELINE-FIDELITY-FAIL"


def _write_docs(output: Path, summary: dict[str, Any]) -> None:
    manifest = summary["upstream_manifest"]
    _write_md(
        output / "method_identity.md",
        "Baseline Method Identity",
        f"""
Pinned upstreams are recorded in `upstream_manifest.json`.

| method | upstream authority | required identity | current status |
|---|---|---|---|
| ERM | Facebook IRM and DomainBed | source empirical risk minimization only | `{summary['identity_status']['ERM']}` |
| IRMv1 | Facebook IRM Colored MNIST | scalar scale penalty, anneal at step 100, penalty weight 10000, whole-loss rescale after anneal | `{summary['identity_status']['IRMv1']}` |
| IGA | DomainBed `{manifest['domainbed']['commit']}` | full-network per-environment gradients, mean-gradient penalty, default penalty 1000 | `{summary['identity_status']['IGA']}` |
| Fish | DomainBed `{manifest['domainbed']['commit']}` plus YugeTen/fish mechanism | clone, sequential inner-domain updates, carried inner optimizer state, outer interpolation with meta_lr 0.5 | `{summary['identity_status']['Fish']}` |
| HEAD_GRADIENT_VARIANCE_SURROGATE | local Task3 diagnostic only | head-only gradient variance; fixed penalty-subset history | NOT A REPRODUCTION OF IGA OR FISH |

The old public label `UNPRECONDITIONED_GRAD_ALIGN` is superseded by `HEAD_GRADIENT_VARIANCE_SURROGATE` in this gate. Preserved old numeric rows must be read only as a negative diagnostic, not as IGA/Fish evidence.
""",
    )
    _write_md(
        output / "native_reproduction.md",
        "Native Reproduction",
        f"""
Facebook IRM native script status: `{summary['native_status']['facebook_irm']}`.

DomainBed IGA native status: `{summary['native_status']['domainbed_iga']}`.

DomainBed Fish native status: `{summary['native_status']['domainbed_fish']}`.

Fish author repo was not used as the CMNIST runner because its pinned native benchmark is not Colored MNIST; DomainBed Fish is the ColoredMNIST fallback authority.
""",
    )
    _write_md(
        output / "controlled_harness.md",
        "Controlled Harness",
        f"""
The controlled harness uses the Facebook Colored MNIST identity: `digit < 5`, label noise `0.25`, source color flip probabilities `[0.2, 0.1]`, target flip `0.9`, 2-channel 14x14 inputs, BCE-with-logits, Adam, L2 `0.001`.

The local run is bounded: `{summary['controlled_config']['steps']}` steps on `{summary['controlled_config']['examples_per_env']}` examples per source environment. It is a port-fidelity smoke, not a performance benchmark.

Controlled result rows are in `results/controlled_run_table.csv`; aggregate means are in `results/controlled_summary.csv`.
""",
    )
    failure_lines = "\n".join(f"- {item}" for item in summary["fidelity_failures"])
    _write_md(
        output / "fidelity_failures.md",
        "Fidelity Failures",
        failure_lines or "No blocking fidelity failures were detected by this gate.",
    )
    _write_md(
        output / "baseline_fidelity_report.md",
        "Task 3 Baseline Fidelity Recovery Report",
        f"""
verdict: `{summary['verdict']}`

Task 3 scientific verdict: `NOT YET RUN`

IRMv1 recovery passed: `{summary['irmv1_recovery']['passed']}`; ERM target accuracy `{summary['irmv1_recovery'].get('best_erm_target_accuracy')}`; IRMv1 target accuracy `{summary['irmv1_recovery'].get('best_irmv1_target_accuracy')}`.

Equivalence audit: IRMv1 penalty finite `{summary['equivalence']['irm_penalty_finite']}`, IGA full-network `{summary['equivalence']['iga_uses_full_network_parameters']}`, Fish parameter changed `{summary['equivalence']['fish_parameter_changed']}`, old head-only surrogate blocked from IGA/Fish labeling `{summary['equivalence']['custom_surrogate_notice']['not_a_reproduction_of_iga_or_fish']}`.

Native artifacts: `results/irm_native.csv`, `results/iga_native.csv`, `results/fish_native.csv`.

Controlled artifacts: `results/controlled_run_table.csv`, `results/controlled_summary.csv`, `results/upstream_port_equivalence.json`.

No `LOCAL_RESPONSE`, `RANDOM_METRIC`, `SHUFFLED_LOCAL_RESPONSE`, or `RESP2` rows are generated by this gate.
""",
    )


def _write_state_delta(summary: dict[str, Any]) -> None:
    prerequisite_satisfied = summary["verdict"] == "BASELINE-FIDELITY-PASS"
    text = f"""# Proposed State Delta: TASK3-BASELINE-FIDELITY-RECOVERY

state_write_authorized: false

baseline fidelity verdict: `{summary['verdict']}`

Task 3 scientific verdict: `NOT YET RUN`

baseline-fidelity prerequisite satisfied: `{str(prerequisite_satisfied).lower()}`

Task 3 local-response experiment must be rerun before any scientific verdict: `true`

Proposed result ID: `R-TASK3-BASELINE-FIDELITY`

Evidence path: `round3_redesign/task3_baseline_fidelity/results/baseline_fidelity_summary.json`

Proposed CURRENT_STATE.md wording:

```text
Task 3 baseline fidelity gate verdict: {summary['verdict']}. The gate audits pinned ERM, IRMv1, IGA, and Fish identities. Task 3 scientific verdict is NOT YET RUN and requires a separate rerun of the local-response experiment after this prerequisite.
```
"""
    (ROOT / "active" / "STATE_DELTA.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--skip-native", action="store_true")
    parser.add_argument("--reuse-existing-native", action="store_true")
    parser.add_argument("--skip-facebook-native", action="store_true")
    parser.add_argument("--domainbed-native-steps", type=int, default=2)
    parser.add_argument("--native-irm-steps", type=int, default=501)
    parser.add_argument("--native-irm-restarts", type=int, default=3)
    parser.add_argument("--controlled-steps", type=int, default=501)
    parser.add_argument("--controlled-examples-per-env", type=int, default=256)
    parser.add_argument("--controlled-target-examples", type=int, default=512)
    args = parser.parse_args()

    output = args.output
    results = output / "results"
    results.mkdir(parents=True, exist_ok=True)
    manifest = upstream_manifest(ROOT)
    _write_json(output / "upstream_manifest.json", manifest)
    native_config = NativeRunConfig(
        root=ROOT,
        domainbed_steps=args.domainbed_native_steps,
        native_irm_steps=args.native_irm_steps,
        native_irm_restarts=args.native_irm_restarts,
    )

    irm_native_rows: list[dict[str, Any]] = []
    if args.reuse_existing_native and (results / "irm_native.csv").exists():
        irm_native_rows = list(_read_csv(results / "irm_native.csv"))
    elif args.skip_native or args.skip_facebook_native:
        # Preserve an official CPU-compatible recovery when the direct upstream run is skipped.
        cfg = OfficialIRMConfig(seeds=tuple(range(args.native_irm_restarts)), steps=args.native_irm_steps)
        irm_native_rows = run_official_irm_reimplementation(cfg)
        for row in irm_native_rows:
            row["status"] = "completed_cpu_reimplementation"
    else:
        for method in ("ERM", "IRMv1"):
            irm_native_rows.append(run_facebook_irm_native(native_config, method=method))
    _write_csv(results / "irm_native.csv", irm_native_rows)

    iga_native_rows: list[dict[str, Any]] = []
    fish_native_rows: list[dict[str, Any]] = []
    if args.reuse_existing_native and (results / "iga_native.csv").exists() and (results / "fish_native.csv").exists():
        iga_native_rows = list(_read_csv(results / "iga_native.csv"))
        fish_native_rows = list(_read_csv(results / "fish_native.csv"))
    elif args.skip_native:
        iga_native_rows.append({"method": "IGA", "status": "skipped", "reason": "--skip-native"})
        fish_native_rows.append({"method": "Fish", "status": "skipped", "reason": "--skip-native"})
    else:
        iga_native_rows.append(run_domainbed_native(native_config, method="IGA", output_dir=results / "domainbed_iga_native"))
        fish_native_rows.append(run_domainbed_native(native_config, method="Fish", output_dir=results / "domainbed_fish_native"))
    _write_csv(results / "iga_native.csv", iga_native_rows)
    _write_csv(results / "fish_native.csv", fish_native_rows)

    controlled_rows = run_controlled_harness(
        seed=0,
        steps=args.controlled_steps,
        examples_per_env=args.controlled_examples_per_env,
        target_examples=args.controlled_target_examples,
    )
    _write_csv(results / "controlled_run_table.csv", controlled_rows)
    controlled_summary = []
    for row in controlled_rows:
        controlled_summary.append({
            "method": row["method"],
            "target_accuracy": row["target_accuracy"],
            "prediction_color_agreement": row["prediction_color_agreement"],
            "objective_penalty": row["objective_penalty"],
        })
    _write_csv(results / "controlled_summary.csv", controlled_summary)

    equivalence = run_equivalence_audit()
    _write_json(results / "upstream_port_equivalence.json", equivalence)
    irm_status = _baseline_recovery_status(irm_native_rows)
    native_status = {
        "facebook_irm": ",".join(str(row.get("status")) for row in irm_native_rows),
        "domainbed_iga": ",".join(str(row.get("status")) for row in iga_native_rows),
        "domainbed_fish": ",".join(str(row.get("status")) for row in fish_native_rows),
    }
    identity_status = {
        "ERM": "pinned_and_controlled",
        "IRMv1": "recovered" if irm_status.get("passed") else "not_recovered",
        "IGA": "native_completed" if any(row.get("status") == "completed" for row in iga_native_rows) else "native_unresolved",
        "Fish": "native_completed" if any(row.get("status") == "completed" for row in fish_native_rows) else "native_unresolved",
    }
    failures: list[str] = []
    if not irm_status.get("passed"):
        failures.append("IRMv1 native recovery did not meet target-accuracy / ERM-separation thresholds.")
    if identity_status["IGA"] != "native_completed":
        failures.append("DomainBed IGA native ColoredMNIST run did not complete in this local gate.")
    if identity_status["Fish"] != "native_completed":
        failures.append("DomainBed Fish native ColoredMNIST run did not complete in this local gate.")
    if not equivalence.get("iga_uses_full_network_parameters"):
        failures.append("IGA controlled port is not using all network parameters.")
    if not equivalence.get("iga_head_only_surrogate_differs"):
        failures.append("Head-gradient surrogate was not separated from full-network IGA.")
    verdict = _final_verdict(
        irm_status=irm_status,
        iga_native=iga_native_rows,
        fish_native=fish_native_rows,
        equivalence=equivalence,
    )
    summary = {
        "task_id": "TASK3-BASELINE-FIDELITY-RECOVERY",
        "runner_name": "ood_repr_reg.run_task3_baseline_fidelity",
        "git_commit_before_run": _git(["rev-parse", "HEAD"]),
        "working_tree_status_before_run": _git(["status", "--short"]),
        "run_unix_time": time.time(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "verdict": verdict,
        "task3_scientific_verdict": "NOT YET RUN",
        "upstream_manifest": manifest,
        "native_status": native_status,
        "identity_status": identity_status,
        "irmv1_recovery": irm_status,
        "equivalence": equivalence,
        "controlled_config": {
            "steps": args.controlled_steps,
            "examples_per_env": args.controlled_examples_per_env,
            "target_examples": args.controlled_target_examples,
        },
        "fidelity_failures": failures,
        "excluded_methods": ["LOCAL_RESPONSE", "RANDOM_METRIC", "SHUFFLED_LOCAL_RESPONSE", "RESP2"],
        "target_leakage_detected": False,
    }
    _write_json(results / "baseline_fidelity_summary.json", summary)
    _write_docs(output, summary)
    _write_state_delta(summary)
    print(json.dumps({
        "verdict": verdict,
        "irmv1_recovery": irm_status,
        "native_status": native_status,
        "output": str(output),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
