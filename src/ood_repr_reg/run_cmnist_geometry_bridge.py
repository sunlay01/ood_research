"""Run the isolated CMNIST Round-3 geometry bridge."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from .cmnist_feature_probe import (
    accuracy,
    deterministic_subset,
    load_mnist_tensors,
    train_model,
)
from .run_cmnist_feature_probe import _build_data
from .cmnist_geometry_bridge import (
    CMNISTFamily,
    build_geometry,
    feature_bank,
    moment_state,
    method_row,
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("")
        return
    fields = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _families() -> tuple[CMNISTFamily, ...]:
    return (
        CMNISTFamily("declared_source_target_coupled_correlation", directions=("rho_source_1", "rho_source_2")),
        CMNISTFamily("mechanism_defined_hidden", hidden_exposed=False),
        CMNISTFamily("mechanism_defined_exposed", hidden_exposed=True),
        CMNISTFamily("irrelevant_source_diversity", directions=("rho_source_1", "rho_source_2", "rho_hidden", "brightness_nuisance")),
    )


def _make_training_data(config: dict, seed: int):
    # Keep the exact CMNIST generator and source sample schedule of VIS-001.
    train, source_eval, target, _ = _build_data(config, seed)
    return train, source_eval, target


def run(config_path: Path, output: Path, *, smoke: bool = False) -> dict[str, object]:
    config = json.loads(config_path.read_text())
    output.mkdir(parents=True, exist_ok=True)
    (output / "results").mkdir(exist_ok=True)
    if smoke:
        seeds = [int(config["seeds"][0])]
        epochs = min(int(config["epochs"]), 2)
        methods = [("erm", 0.0), ("l2", 0.001)]
        steps = (2e-3, 1e-3)
    else:
        seeds = [int(x) for x in config.get("bridge_seeds", [0, 1, 2, 3, 4])]
        epochs = int(config["epochs"])
        methods = [("erm", 0.0)] + [(method, lam) for method in ("l2", "irmv1", "vrex") for lam in (0.001, 0.01, 0.1)]
        steps = (1e-3, 5e-4, 2.5e-4)
    device = torch.device(config.get("device", "cpu"))
    train_gray, train_digit = load_mnist_tensors(Path(config["data_root"]), train=True, download=config["download"])
    test_gray, test_digit = load_mnist_tensors(Path(config["data_root"]), train=False, download=config["download"])
    # Geometry uses a fixed source-training bank. Test/target labels are used
    # only by the post-hoc accuracy context below.
    gray, digit = deterministic_subset(train_gray, train_digit, n=int(config.get("bridge_bank_size", 1200)), seed=3711, offset=0)
    geometry_rows: list[dict[str, object]] = []
    method_rows: list[dict[str, object]] = []
    error_rows: list[dict[str, object]] = []
    snapshots: dict[str, Array] = {}
    for seed in seeds:
        train, source_eval, target = _make_training_data(config, seed)
        model, train_metrics = train_model(
            train, method="erm", strength=0.0, latent_dim=int(config["latent_dim"]),
            epochs=epochs, batch_size=int(config["batch_size"]),
            learning_rate=float(config["learning_rate"]), seed=seed, device=device,
        )
        bank = feature_bank(model, gray, digit, device=device)
        source_accuracy = float(np.mean([accuracy(model, env, device) for env in source_eval]))
        target_accuracy = float(accuracy(model, target, device))
        for family in _families():
            geometry = build_geometry(bank, family, steps=steps)
            geometry_rows.append({
                "seed": seed, "family": family.name, "feature_dimension": bank.dimension,
                "source_accuracy": source_accuracy, "target_accuracy": target_accuracy,
                "rank_O": geometry["rank_O"], "rank_A": geometry["rank_A"],
                "kernel_dimension": geometry["kernel_dimension"], "alpha": geometry["alpha"],
                "information_floor": geometry["information_floor"],
                "A_step_relative_errors": geometry["A_step_relative_errors"],
                "O_step_relative_errors": geometry["O_step_relative_errors"],
                "A_direction_norms": np.linalg.norm(geometry["A"], axis=0).tolist(),
                "O_singular_values": np.linalg.svd(geometry["O"], compute_uv=False).tolist(),
                "A_singular_values": np.linalg.svd(geometry["A"], compute_uv=False).tolist(),
                "feature_covariance_singular_values": np.linalg.svd(
                    np.cov(np.concatenate((bank.red, bank.green), axis=0), rowvar=False)
                )[1].tolist(),
                "head_hessian_eigenvalues": np.linalg.eigvalsh(
                    2.0 * np.mean([moment_state(bank, rho)[0] for rho in family.source_rhos], axis=0)
                ).tolist(),
                "train_objective": train_metrics["final_train_risk"],
            })
            snapshots[f"A_seed{seed}_{family.name}"] = geometry["A"]
            snapshots[f"O_seed{seed}_{family.name}"] = geometry["O"]
            for method, lam in methods:
                try:
                    row = method_row(bank, family, method, lam, family.name, steps)
                except Exception as exc:
                    row = {"method": method.upper(), "lambda": lam, "source_design": family.name, "valid": False, "status": f"error:{exc}"}
                    error_rows.append({"seed": seed, "family": family.name, "method": method, "lambda": lam, "error": repr(exc)})
                row["seed"] = seed
                row.setdefault("status", "PASS" if row.get("valid", False) else "INVALID")
                method_rows.append(row)
    fd_rows = []
    for row in method_rows:
        for item in row.get("fd_direction_rows", []):
            fd_rows.append({"seed": row["seed"], "source_design": row["source_design"],
                            "method": row["method"], "lambda": row["lambda"], **item,
                            "valid": row["valid"]})
    _write_csv(output / "results" / "ift_fd_rows.csv", fd_rows)
    # Keep JSON-valued diagnostics in the method table readable and stable.
    method_csv_rows = []
    for row in method_rows:
        method_csv_rows.append({k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in row.items() if k != "fd_direction_rows"})
    _write_csv(output / "results" / "method_comparison.csv", method_csv_rows)
    (output / "results" / "geometry_by_seed.json").write_text(json.dumps(geometry_rows, indent=2, default=float))
    exposure = {}
    for row in geometry_rows:
        if row["family"] in {"mechanism_defined_hidden", "mechanism_defined_exposed", "irrelevant_source_diversity"}:
            exposure.setdefault(row["family"], []).append(row)
    (output / "results" / "exposure_comparison.json").write_text(json.dumps(exposure, indent=2, default=float))
    np.savez_compressed(output / "results" / "operator_snapshots.npz", **snapshots)
    hidden = [r for r in geometry_rows if r["family"] == "mechanism_defined_hidden"]
    exposed = [r for r in geometry_rows if r["family"] == "mechanism_defined_exposed"]
    stable_A = bool(geometry_rows) and all(
        max(r["A_step_relative_errors"]) < 1e-3 for r in geometry_rows
    )
    stable_O = bool(geometry_rows) and all(
        max(r["O_step_relative_errors"]) < 1e-3 for r in geometry_rows
    )
    hidden_alpha = float(np.mean([r["alpha"] for r in hidden])) if hidden else float("nan")
    exposed_alpha = float(np.mean([r["alpha"] for r in exposed])) if exposed else float("nan")
    valid_rows = [r for r in method_rows if r.get("valid")]
    decomposition_residual_max = max(
        (float(r.get("decomposition_residual", 0.0)) for r in method_rows),
        default=0.0,
    )
    fd_audit = [item for row in valid_rows for item in row.get("fd_direction_rows", [])]
    # Zero-response directions carry no angular information; the audit applies
    # the cosine criterion only to numerically active directions.
    active_fd = [item for item in fd_audit if item["fd_norm"] > 1e-8 or item["ift_norm"] > 1e-8]
    ift_ok = bool(active_fd) and max(item["relative_error"] for item in active_fd) < 1e-2 and min(item["cosine"] for item in active_fd) > 0.99
    method_difference = False
    for family_name in ("mechanism_defined_hidden", "mechanism_defined_exposed"):
        means = []
        for method in ("L2", "IRMV1", "VREX"):
            values = [r["E_operator_norm"] for r in valid_rows if r["source_design"] == family_name and r["method"] == method]
            if values:
                means.append(float(np.mean(values)))
        if means and max(means) - min(means) > 1e-4:
            method_difference = True
    verdict = "CMNIST-GEOMETRY-PASS" if (
        not error_rows and valid_rows and stable_A and stable_O and ift_ok
        and method_difference and hidden_alpha > exposed_alpha * 1.01
    ) else "CMNIST-GEOMETRY-PARTIAL"
    summary = {
        "verdict": verdict, "status": "empirical_finite_sample_geometry_bridge",
        "representation_seeds": seeds, "head_objective": "empirical_squared_loss",
        "encoder_frozen_for_bridge": True, "target_used_for_training_or_selection": False,
        "target_labels_used_for_geometry": False,
        "semantic_labels_used_for_geometry": False, "regularizer_used_for_geometry": False,
        "stable_A": stable_A, "stable_O": stable_O, "ift_ok": ift_ok,
        "method_difference": method_difference,
        "hidden_alpha_mean": hidden_alpha, "exposed_alpha_mean": exposed_alpha,
        "geometry_row_count": len(geometry_rows), "method_row_count": len(method_rows),
        "valid_method_row_count": len(valid_rows), "error_row_count": len(error_rows), "error_rows": error_rows,
        "response_decomposition": "A_irreducible=A P_ker(O_S); A_recoverable=A(I-P_ker(O_S)); E=A_recoverable+Pi O_S",
        "decomposition_residual_max": decomposition_residual_max,
        "families": [f.metadata() for f in _families()],
        "world_metric": "Euclidean on declared CMNIST correlation/nuisance tangent",
        "response_metric": "source-risk H_S^{1/2} head displacement",
        "target_risk_lower_bound_claimed": False,
    }
    (output / "results" / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    try:
        import matplotlib.pyplot as plt
        family_names = ["mechanism_defined_hidden", "mechanism_defined_exposed", "irrelevant_source_diversity"]
        grouped = {name: [r for r in geometry_rows if r["family"] == name] for name in family_names}
        figure, axis = plt.subplots(figsize=(7, 4))
        for name, values in grouped.items():
            if values:
                axis.plot([r["seed"] for r in values], [r["alpha"] for r in values], "o-", label=name)
        axis.set(xlabel="representation seed", ylabel="||A P_ker(O)||", title="CMNIST information floor geometry")
        axis.legend(fontsize=8); axis.grid(alpha=0.25); figure.tight_layout()
        figure.savefig(output / "results" / "exposure_floor.png", dpi=160); plt.close(figure)

        figure, axis = plt.subplots(figsize=(7, 4))
        for name, values in grouped.items():
            if values:
                axis.plot([r["seed"] for r in values], [max(r["A_step_relative_errors"]) for r in values], "o-", label=name)
        axis.set(xlabel="representation seed", ylabel="max A FD relative error", title="A step-size stability")
        axis.legend(fontsize=8); axis.grid(alpha=0.25); figure.tight_layout()
        figure.savefig(output / "results" / "A_step_stability.png", dpi=160); plt.close(figure)

        method_names = ["ERM", "L2", "IRMV1", "VREX"]
        figure, axis = plt.subplots(figsize=(8, 4))
        for name in family_names:
            means = []
            for method in method_names:
                values = [r for r in method_rows if r["source_design"] == name and r["method"] == method and r.get("valid")]
                means.append(np.mean([r["E_operator_norm"] for r in values]) if values else np.nan)
            axis.plot(method_names, means, "o-", label=name)
        axis.set(ylabel="mean ||E|| operator norm", title="Method residual by family")
        axis.legend(fontsize=8); axis.grid(alpha=0.25); figure.tight_layout()
        figure.savefig(output / "results" / "method_residuals.png", dpi=160); plt.close(figure)

        figure, axis = plt.subplots(figsize=(7, 4))
        for name in family_names:
            values = grouped[name]
            if values:
                spectrum = np.mean([r["A_singular_values"] for r in values], axis=0)
                axis.semilogy(np.arange(1, len(spectrum) + 1), spectrum, "o-", label=name)
        axis.set(xlabel="singular-value index", ylabel="singular value", title="A singular spectra")
        axis.legend(fontsize=8); axis.grid(alpha=0.25); figure.tight_layout()
        figure.savefig(output / "results" / "singular_spectra.png", dpi=160); plt.close(figure)
    except Exception as exc:  # plotting is diagnostic and never changes the verdict
        summary["plot_error"] = str(exc)
        (output / "results" / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    method_stats = []
    for family_name in sorted({str(r["source_design"]) for r in method_rows}):
        for method in ("ERM", "L2", "IRMV1", "VREX"):
            values = [r for r in method_rows if r["source_design"] == family_name and r["method"] == method and r.get("valid")]
            if values:
                method_stats.append(
                    f"| {family_name} | {method} | {np.mean([r['norm_z0'] for r in values]):.5g} | "
                    f"{np.mean([r['pi_O_operator_norm'] for r in values]):.5g} | "
                    f"{np.mean([r['E_operator_norm'] for r in values]):.5g} | "
                    f"{np.mean([r['affine_regret'] for r in values]):.5g} | "
                    f"{np.mean([r['fd_error_max'] for r in values]):.3g} |"
                )
    report = f"""# CMNIST Round-3 Geometry Bridge Report

## Verdict

`{verdict}`

This is an empirical finite-sample bridge for a frozen learned
representation and a trainable squared-loss linear head. It is not a proof of
the Gaussian population theorems on neural networks.

## Main findings

- Five ERM encoder seeds were trained with the existing CMNIST generator.
- The feature bank was reduced to the empirical non-degenerate support; no
  damping was added to the head Hessian.
- `A` step stability: `{stable_A}`; `O_S` step stability: `{stable_O}`.
- Hidden mechanism alpha mean: `{hidden_alpha:.6g}`.
- Exposed mechanism alpha mean: `{exposed_alpha:.6g}`.
- The hidden-to-exposed decrease is a source-exposure diagnostic, not a
  causal identification or target-risk theorem.

## Method summary

The recoverable residual is defined as
`E = A_recoverable + Pi O_S`, where `A_recoverable = A(I-P_ker(O_S))`.
The irreducible response `A_irreducible = A P_ker(O_S)` is used only for the
information floor and spectral-slack diagnostic. The maximum reconstruction
residual `||A-A_irreducible-A_recoverable||` over method rows is
`{decomposition_residual_max:.3g}`.

| family | method | mean ||z0|| | mean ||Pi O|| | mean ||E|| | mean affine regret | mean FD error |
|---|---|---:|---:|---:|---:|---:|
{chr(10).join(method_stats)}

All target accuracy values are post-hoc context only. The source state and
operator construction read no target risk, semantic label, cluster label or
regularizer geometry. `ift_fd_rows.csv` contains the per-direction,
per-step audit; `method_comparison.csv` contains method rows; and
`operator_snapshots.npz` contains the A/O matrices.

## Interpretation boundary

The result supports or rejects applicability only for the stated frozen
representation, empirical moments, tangent family and metric. It does not
establish a finite-sample guarantee, causal mechanism identification,
universal DG theorem, or target-risk lower bound. Full-CNN cross-entropy and
the prior CMNIST probe remain separate tracks.
"""
    (output / "cmnist_geometry_bridge_report.md").write_text(report)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/cmnist_vis_001_main.json"))
    parser.add_argument("--output", type=Path, default=Path("round3_redesign/cmnist_geometry_bridge"))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.output, smoke=args.smoke), indent=2))


if __name__ == "__main__":
    main()
