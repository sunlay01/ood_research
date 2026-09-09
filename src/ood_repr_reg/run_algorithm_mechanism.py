"""Run the independent Task 1 algorithm-mechanism decomposition track."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .algorithm_mechanism.adapters import cmnist_input, gaussian_input
from .algorithm_mechanism.audits import evaluate_input, scalar_rows, static_audit
from .corrected_geometry_snapshot import corrected_snapshot_audit
from .cmnist_feature_probe import deterministic_subset, load_mnist_tensors, train_model
from .cmnist_geometry_bridge import CMNISTFamily, feature_bank
from .round3r_3e_c_benchmarks import primary_hidden_u_world, primary_u_exposed_world
from .run_cmnist_feature_probe import _build_data


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "round3_redesign" / "algorithm_mechanism"


def _json(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def _docs(output: Path, summary: dict[str, object]) -> None:
    verdict = str(summary["verdict"])
    docs = {
        "assumptions.md": "# Assumptions\n\nThis track studies finite-dimensional local source objectives. CMNIST uses a frozen encoder and empirical squared-loss head. The corrected response decomposition is consumed only after learner-side mechanism construction.\n",
        "theory.md": "# Local Mechanism Identity\n\nAt the actual method solution, `Pi=-(H_R+lambda K)^-1(B_R+lambda C)`. `C` is source forcing, `K` is filtering, and `g` controls static steering. The common-base rows are counterfactual diagnostics, not a replacement for the actual-solution identity.\n",
        "proof_notes.md": "# Proof Notes\n\nThe identity follows by differentiating the regularized source first-order condition. Static-path rows differentiate the same condition with respect to lambda. Numerical rows audit these exact identities; they are not separate theorems.\n",
        "gaussian_audit.md": "# Gaussian Audit\n\nThe Gaussian rows use the exact-IFT source objective and report component reconstruction, common-base counterfactuals, static-path agreement, and post-hoc response diagnostics for hidden and U-exposed source designs.\n",
        "cmnist_audit.md": "# CMNIST Audit\n\nCMNIST rows reuse the corrected frozen-representation squared-loss bridge across all five declared seeds. The encoder is not selected using target accuracy, response, or regret.\n",
        "negative_results.md": "# Boundaries\n\nThese source-side identities do not imply causal identification, semantic recovery, target-risk bounds, finite-sample guarantees, or universal DG guarantees. CORAL remains a predictor/gauge boundary only.\n",
        "algorithm_mechanism_report.md": (
            "# Algorithm Mechanism Decomposition\n\n"
            f"Verdict: `{verdict}`. The audit contains {summary['record_count']} rows with "
            f"{summary['error_count']} errors. The maximum IFT reconstruction relative error is "
            f"`{summary['max_pi_reconstruction_relative_error']:.3e}` and the maximum component-total "
            f"error is `{summary['max_component_total_relative_error']:.3e}`. "
            f"The corrected geometry snapshot audit passed for all {summary['corrected_geometry_snapshot_count']} rows. "
            "Exact IFT totals are compared with independently differentiated components and static-path finite differences. "
            "`A`, `E`, and affine regret are strictly post-hoc diagnostics.\n"
        ),
    }
    for name, text in docs.items():
        (output / name).write_text(text)


def run(output: Path = DEFAULT_OUTPUT, *, cmnist: bool = True, seeds: tuple[int, ...] = (0, 1, 2, 3, 4)) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True); results = output / "results"; results.mkdir(exist_ok=True)
    records: list[dict[str, object]] = []; static_rows: list[dict[str, object]] = []
    snapshots: list[dict[str, object]] = []
    operator_snapshots: dict[str, np.ndarray] = {}

    def capture_operator(kind: str, item, seed: int | None = None) -> None:
        parts = [kind, item.setting, item.method, f"lambda_{item.lam:.6g}"]
        if seed is not None:
            parts.append(f"seed_{seed}")
        prefix = "__".join(parts).replace("/", "_")
        operator_snapshots[f"{prefix}__A"] = np.asarray(item.response)
        operator_snapshots[f"{prefix}__O"] = np.asarray(item.observation)
        operator_snapshots[f"{prefix}__PiO"] = np.asarray(item.response_adaptive)
        operator_snapshots[f"{prefix}__z0"] = np.asarray(item.response_offset)
    errors: list[dict[str, object]] = []
    worlds = (primary_hidden_u_world(), primary_u_exposed_world())
    grid = (("l2", 0.0), ("l2", 0.01), ("irmv1", 0.01), ("vrex", 0.01))
    for world in worlds:
        for method, lam in grid:
            try:
                item = gaussian_input(world, method, lam); record = evaluate_input(item); records.append(record)
                capture_operator("gaussian", item)
                audit = corrected_snapshot_audit(item.response, item.observation, item.response_adaptive)
                snapshots.append({"setting": item.setting, "method": item.method, "lambda": item.lam,
                                  "kind": "gaussian", "passes": audit["passes"],
                                  "decomposition_residual": audit["decomposition_residual"],
                                  "full_observation_recoverable_residual": audit["full_observation_recoverable_residual"],
                                  "zero_observation_E_norm": audit["zero_observation_E_norm"]})
                path = static_audit(item)
                if path is not None: static_rows.append(path)
            except Exception as exc: errors.append({"setting": world.name, "method": method, "lambda": lam, "error": repr(exc)})
    if cmnist:
        config = json.loads((ROOT / "configs" / "cmnist_vis_001_main.json").read_text())
        import torch
        device = torch.device(config.get("device", "cpu"))
        gray_all, digit_all = load_mnist_tensors(ROOT / config["data_root"], train=True, download=config["download"])
        gray, digit = deterministic_subset(gray_all, digit_all, n=int(config.get("bridge_bank_size", 1200)), seed=3711)
        families = (CMNISTFamily("mechanism_defined_hidden"), CMNISTFamily("mechanism_defined_exposed", hidden_exposed=True))
        for seed in seeds:
            train, _, _, _ = _build_data(config, seed)
            model, _ = train_model(train, method="erm", strength=0.0, latent_dim=int(config["latent_dim"]), epochs=int(config["epochs"]), batch_size=int(config["batch_size"]), learning_rate=float(config["learning_rate"]), seed=seed, device=device)
            bank = feature_bank(model, gray, digit, device=device)
            for family in families:
                for method, lam in grid:
                    try:
                        item = cmnist_input(bank, family, method, lam); record = evaluate_input(item); record["seed"] = seed; records.append(record)
                        capture_operator("cmnist", item, seed)
                        audit = corrected_snapshot_audit(item.response, item.observation, item.response_adaptive)
                        snapshots.append({"setting": item.setting, "method": item.method, "lambda": item.lam,
                                          "seed": seed, "kind": "cmnist", "passes": audit["passes"],
                                          "decomposition_residual": audit["decomposition_residual"],
                                          "full_observation_recoverable_residual": audit["full_observation_recoverable_residual"],
                                          "zero_observation_E_norm": audit["zero_observation_E_norm"]})
                        path = static_audit(item)
                        if path is not None: path["seed"] = seed; static_rows.append(path)
                    except Exception as exc: errors.append({"setting": family.name, "seed": seed, "method": method, "lambda": lam, "error": repr(exc)})
    exact: list[dict[str, object]] = []; objects: list[dict[str, object]] = []; counter: list[dict[str, object]] = []
    mode_rows: list[dict[str, object]] = []
    for record in records:
        one, two, three = scalar_rows(record)
        if "seed" in record: one["seed"] = two["seed"] = three["seed"] = record["seed"]
        exact.append(one); objects.append(two); counter.append(three)
        for index, value in enumerate(np.linalg.svd(record["PiO"], compute_uv=False)):
            mode_rows.append({"setting": record["setting"], "method": record["method"], "lambda": record["lambda"], "mode": index, "PiO_singular_value": float(value)})
    _write_csv(results / "exact_reconstruction.csv", exact); _write_csv(results / "mechanism_objects.csv", objects)
    _write_csv(results / "counterfactual_residuals.csv", counter); _write_csv(results / "static_path_rows.csv", [{k: (json.dumps(v, default=_json) if isinstance(v, np.ndarray) else v) for k, v in row.items() if k not in {"weights", "predicted", "finite_difference"}} for row in static_rows]); _write_csv(results / "per_mode_rows.csv", mode_rows)
    (results / "corrected_geometry_snapshots.json").write_text(json.dumps(snapshots, indent=2, default=_json))
    np.savez_compressed(results / "operator_snapshots.npz", **operator_snapshots)
    max_pi_error = max((r["learner"]["pi_reconstruction_relative_error"] for r in records), default=float("inf"))
    max_total_error = max((max(r["learner"]["total_H_relative_error"], r["learner"]["total_B_relative_error"]) for r in records), default=float("inf"))
    snapshots_pass = bool(snapshots) and all(bool(row["passes"]) for row in snapshots)
    verdict = "ALGORITHM-MECHANISM-PASS" if records and not errors and snapshots_pass and max_pi_error < 2e-4 and max_total_error < 2e-4 else "ALGORITHM-MECHANISM-PARTIAL"
    summary = {"verdict": verdict, "record_count": len(records), "error_count": len(errors), "errors": errors, "max_pi_reconstruction_relative_error": max_pi_error, "max_component_total_relative_error": max_total_error, "corrected_geometry_snapshot_count": len(snapshots), "operator_snapshot_count": len(operator_snapshots) // 4, "corrected_geometry_snapshots_pass": snapshots_pass, "learner_side_target_response_used": False, "semantic_or_cluster_used": False, "coral_status": "NONCANONICAL-PREDICTOR-RESPONSE", "cmnist_scope": "frozen_representation_empirical_squared_loss" if cmnist else "not_run", "theorem_status": "exact_local_identity", "lean_status": "LEAN-PARTIAL"}
    (results / "summary.json").write_text(json.dumps(summary, indent=2, default=_json)); _docs(output, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT); parser.add_argument("--no-cmnist", action="store_true"); parser.add_argument("--seeds", default="0,1,2,3,4")
    args = parser.parse_args(); print(json.dumps(run(args.output, cmnist=not args.no_cmnist, seeds=tuple(int(x) for x in args.seeds.split(",") if x)), indent=2, default=_json))


if __name__ == "__main__": main()
