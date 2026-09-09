"""Run the independent Task 2 sharp family-conditional optimality audit."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .algorithm_mechanism.adapters import cmnist_input, gaussian_input
from .cmnist_feature_probe import deterministic_subset, load_mnist_tensors, train_model
from .cmnist_geometry_bridge import CMNISTFamily, feature_bank
from .corrected_geometry_snapshot import corrected_snapshot_audit
from .round3r_3e_c_benchmarks import primary_hidden_u_world, primary_u_exposed_world
from .run_cmnist_feature_probe import _build_data
from .sharp_optimality.counterexamples import counterexamples
from .sharp_optimality.geometry import affine_policy_audit, response_parts, transported_coordinate_audit

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "round3_redesign" / "sharp_optimality"
DEFAULT_SNAPSHOTS = ROOT / "round3_redesign" / "algorithm_mechanism" / "results" / "operator_snapshots.npz"
GRID = (("l2", 0.0), ("l2", 0.001), ("l2", 0.01), ("l2", 0.1),
        ("irmv1", 0.001), ("irmv1", 0.01), ("irmv1", 0.1),
        ("vrex", 0.001), ("vrex", 0.01), ("vrex", 0.1))


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
        writer.writeheader()
        writer.writerows(rows)


def _matrix_audit(kind: str, setting: str, method: str, lam: float,
                  response: np.ndarray, observation: np.ndarray, pi_o: np.ndarray,
                  z0: np.ndarray, *, seed: int | None = None) -> tuple[dict[str, object], dict[str, object]]:
    snapshot = corrected_snapshot_audit(response, observation, pi_o)
    audit = affine_policy_audit(z0, response, observation, pi_o)
    parts = response_parts(response, observation, pi_o)
    dimension = response.shape[1]
    transform = np.diag(np.linspace(0.8, 1.2, dimension))
    coordinate = transported_coordinate_audit(
        z0, response, observation, pi_o,
        np.eye(dimension), transform,
    )
    row = {
        "kind": kind, "setting": setting, "method": method, "lambda": lam,
        "seed": seed, "dim_world": dimension,
        "kernel_dimension": int(parts["kernel_dimension"]),
        "information_floor": audit["information_floor"],
        "adaptive_regret": audit["adaptive_regret"],
        "total_regret": audit["total_regret"],
        "total_excess": audit["total_excess"],
        "static_tax_lower_bound": audit["static_tax_lower_bound"],
        "static_tax_holds": audit["static_tax_holds"],
        "full_minimax_condition": audit["full_minimax_condition"],
        "z0_norm": audit["z0_norm"],
        "R_operator_norm": audit["R_operator_norm"],
        "E_operator_norm": audit["E_operator_norm"],
        "PiO_operator_norm": audit["PiO_operator_norm"],
        "slack_ratio": audit["slack_ratio"],
        "support_compatible": audit["support_compatible"],
        "spectral_gap_min": audit["psd_min_gap"],
        "decomposition_residual": audit["decomposition_residual"],
        "snapshot_passes": snapshot["passes"],
        "full_observation_recoverable_residual": snapshot["full_observation_recoverable_residual"],
        "zero_observation_E_norm": snapshot["zero_observation_E_norm"],
        "coordinate_audit_passes": coordinate["pass"],
    }
    spectral = {
        "kind": kind, "setting": setting, "method": method, "lambda": lam,
        "seed": seed, "alpha": audit["alpha"], "slack_ratio": audit["slack_ratio"],
        "support_compatible": audit["support_compatible"], "psd_min_gap": audit["psd_min_gap"],
        "adaptive_condition_holds": audit["condition_holds"],
        "recoverable_exact": audit["recoverable_exact"],
        "cross_irr_E_star_norm": audit["orthogonality"]["Airr_E_star_norm"],
        "cross_E_irr_star_norm": audit["orthogonality"]["E_Airr_star_norm"],
        "gram_sum_residual": audit["orthogonality"]["gram_sum_residual"],
    }
    return row, {"coordinate": coordinate, "spectral": spectral}


def _scalar_audit(kind: str, item, *, seed: int | None = None) -> tuple[dict[str, object], dict[str, object]]:
    return _matrix_audit(kind, item.setting, item.method, item.lam,
                         np.asarray(item.response), np.asarray(item.observation),
                         np.asarray(item.response_adaptive), np.asarray(item.response_offset), seed=seed)


def _snapshot_rows(path: Path) -> list[tuple[str, str, str, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int | None]]:
    arrays = np.load(path, allow_pickle=False)
    rows = []
    for key in arrays.files:
        if not key.endswith("__A"):
            continue
        prefix = key[:-3]
        parts = prefix.split("__")
        if len(parts) not in {4, 5}:
            raise ValueError(f"unrecognized operator snapshot key: {key}")
        kind, setting, method, lambda_token = parts[:4]
        seed = int(parts[4].removeprefix("seed_")) if len(parts) == 5 else None
        lam = float(lambda_token.removeprefix("lambda_"))
        rows.append((kind, setting, method, lam, arrays[key], arrays[f"{prefix}__O"],
                     arrays[f"{prefix}__PiO"], arrays[f"{prefix}__z0"], seed))
    return rows


def _docs(output: Path, summary: dict[str, object]) -> None:
    verdict = str(summary["verdict"])
    documents = {
        "assumptions.md": "# Assumptions\n\nThe audit is finite-dimensional, local, affine, and family-conditional. CMNIST uses frozen learned representations and squared-loss heads. All world-norm comparisons use the declared Euclidean metric after whitening.\n",
        "theory.md": "# Sharp Family-Conditional Optimality\n\nAfter metric whitening, `R=A P_ker(O)` and `E=A(I-P_ker(O))+Pi O`. Adaptive optimality holds exactly when `E E^T` is bounded by spectral slack `alpha^2 I-R R^T`; full affine optimality additionally requires `z0=0`.\n",
        "proof_notes.md": "# Proof Notes\n\nThe cross terms vanish because `E P=0` and `R(I-P)=0`. The spectral condition is equivalent to the operator-norm adaptive minimax equality. The static tax uses the two source-indistinguishable worlds on a top invisible direction.\n",
        "metric_whitening.md": "# Metric Whitening\n\nFor `u^T G u <= 1`, this track evaluates `A G^-1/2`, `O G^-1/2`, and `(Pi O)G^-1/2`. Coordinate changes transport the metric and must preserve each regret certificate.\n",
        "coordinate_audit.md": "# Coordinate Audit\n\nEvery reported row is re-evaluated after a non-orthogonal invertible world-coordinate transform with transported metric. The output records all certificate differences explicitly.\n",
        "static_tax.md": "# Static Steering Tax\n\nThe indistinguishable pair on a top invisible direction gives the lower bound `R_total >= R_info + 0.5 ||z0||^2`. This is an affine response result, not a target-risk lower bound.\n",
        "gaussian_audit.md": "# Gaussian Audit\n\nHidden-U and U-exposed Gaussian rows are evaluated from the frozen exact-IFT operators. They are numerical audits of the finite-dimensional theorem conditions.\n",
        "cmnist_audit.md": "# CMNIST Audit\n\nCMNIST rows consume Task 1's immutable corrected operator snapshots. They are empirical frozen-head checks, not population or causal claims.\n",
        "negative_results.md": "# Boundaries\n\nNo causal identification, semantic recovery, finite-sample guarantee, target-risk lower bound, or universal DG claim is made. Numerical CMNIST rows are empirical bridge audits only.\n",
        "sharp_optimality_report.md": (
            "# Sharp Family-Conditional Optimality\n\n"
            f"Verdict: `{verdict}`. The audit contains {summary['audit_row_count']} rows with "
            f"{summary['error_count']} errors. Corrected decomposition, metric-transport, cross-operator, "
            f"and counterexample audits respectively passed: `{summary['corrected_snapshot_pass']}`, "
            f"`{summary['coordinate_invariance_pass']}`, `{summary['cross_operator_audit_pass']}`, and "
            f"`{summary['counterexamples_pass']}`. The report separates exact finite-dimensional identities, "
            "numerical Gaussian audits, and empirical frozen-head CMNIST audits.\n"
        ),
    }
    for name, content in documents.items():
        (output / name).write_text(content)


def run(output: Path = DEFAULT_OUTPUT, *, cmnist: bool = True,
        seeds: tuple[int, ...] = (0, 1, 2, 3, 4), snapshot_path: Path = DEFAULT_SNAPSHOTS) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    results = output / "results"
    results.mkdir(exist_ok=True)
    audits: list[dict[str, object]] = []
    spectral_rows: list[dict[str, object]] = []
    coordinate_rows: list[dict[str, object]] = []
    errors: list[dict[str, object]] = []

    use_snapshots = snapshot_path.exists()
    if use_snapshots:
        source_rows = _snapshot_rows(snapshot_path)
        if not cmnist:
            source_rows = [row for row in source_rows if row[0] == "gaussian"]
        for kind, setting, method, lam, a, o, pi_o, z0, seed in source_rows:
            try:
                row, detail = _matrix_audit(kind, setting, method, lam, a, o, pi_o, z0, seed=seed)
                audits.append(row); spectral_rows.append(detail["spectral"])
                coordinate_rows.append({"kind": kind, "setting": setting, "method": method,
                                        "lambda": lam, "seed": seed,
                                        **{f"error_{key}": value for key, value in detail["coordinate"]["errors"].items()},
                                        "passes": detail["coordinate"]["pass"]})
            except Exception as exc:
                errors.append({"kind": kind, "setting": setting, "method": method,
                               "lambda": lam, "seed": seed, "error": repr(exc)})
    else:
        for world in (primary_hidden_u_world(), primary_u_exposed_world()):
            for method, lam in GRID:
                try:
                    item = gaussian_input(world, method, lam)
                    row, detail = _scalar_audit("gaussian", item)
                    audits.append(row); spectral_rows.append(detail["spectral"])
                    coordinate_rows.append({"kind": "gaussian", "setting": item.setting,
                                            "method": item.method, "lambda": item.lam,
                                            **{f"error_{key}": value for key, value in detail["coordinate"]["errors"].items()},
                                            "passes": detail["coordinate"]["pass"]})
                except Exception as exc:
                    errors.append({"kind": "gaussian", "setting": world.name, "method": method,
                                   "lambda": lam, "error": repr(exc)})

    if cmnist and not use_snapshots:
        config = json.loads((ROOT / "configs" / "cmnist_vis_001_main.json").read_text())
        import torch
        device = torch.device(config.get("device", "cpu"))
        gray_all, digit_all = load_mnist_tensors(ROOT / config["data_root"], train=True,
                                                 download=config["download"])
        gray, digit = deterministic_subset(gray_all, digit_all,
                                           n=int(config.get("bridge_bank_size", 1200)), seed=3711)
        families = (
            CMNISTFamily("declared_source_target_coupled_correlation", directions=("rho_source_1", "rho_source_2")),
            CMNISTFamily("mechanism_defined_hidden"),
            CMNISTFamily("mechanism_defined_exposed", hidden_exposed=True),
            CMNISTFamily("irrelevant_source_diversity", directions=("rho_source_1", "rho_source_2", "brightness_nuisance")),
        )
        for seed in seeds:
            train, _, _, _ = _build_data(config, seed)
            model, _ = train_model(train, method="erm", strength=0.0,
                                   latent_dim=int(config["latent_dim"]), epochs=int(config["epochs"]),
                                   batch_size=int(config["batch_size"]),
                                   learning_rate=float(config["learning_rate"]), seed=seed, device=device)
            bank = feature_bank(model, gray, digit, device=device)
            for family in families:
                for method, lam in GRID:
                    try:
                        item = cmnist_input(bank, family, method, lam)
                        row, detail = _scalar_audit("cmnist", item, seed=seed)
                        audits.append(row); spectral_rows.append(detail["spectral"])
                        coordinate_rows.append({"kind": "cmnist", "setting": item.setting,
                                                "method": item.method, "lambda": item.lam, "seed": seed,
                                                **{f"error_{key}": value for key, value in detail["coordinate"]["errors"].items()},
                                                "passes": detail["coordinate"]["pass"]})
                    except Exception as exc:
                        errors.append({"kind": "cmnist", "seed": seed, "setting": family.name,
                                       "method": method, "lambda": lam, "error": repr(exc)})

    fixtures = counterexamples()
    fixture_rows = []
    for name, result in fixtures.items():
        fixture_rows.append({"fixture": name, "condition_holds": result["condition_holds"],
                             "information_floor": result["information_floor"],
                             "adaptive_regret": result["adaptive_regret"],
                             "total_regret": result["total_regret"],
                             "full_minimax_condition": result["full_minimax_condition"],
                             "E_operator_norm": result["E_operator_norm"]})
    _write_csv(results / "theorem_audit.csv", audits)
    _write_csv(results / "spectral_rows.csv", spectral_rows)
    _write_csv(results / "counterexamples.csv", fixture_rows)
    (results / "counterexamples.json").write_text(json.dumps(fixtures, indent=2, default=_json))
    (results / "coordinate_audit.json").write_text(json.dumps(coordinate_rows, indent=2, default=_json))
    _write_csv(results / "gaussian_audit.csv", [row for row in audits if row["kind"] == "gaussian"])
    _write_csv(results / "cmnist_audit.csv", [row for row in audits if row["kind"] == "cmnist"])

    snapshot_ok = bool(audits) and all(bool(row["snapshot_passes"]) for row in audits)
    coordinate_ok = bool(coordinate_rows) and all(bool(row["passes"]) for row in coordinate_rows)
    cross_ok = bool(spectral_rows) and all(
        abs(float(row["cross_irr_E_star_norm"])) < 1e-7
        and abs(float(row["cross_E_irr_star_norm"])) < 1e-7
        and abs(float(row["gram_sum_residual"])) < 1e-7 for row in spectral_rows
    )
    required_fixtures = ("nonzero_E_within_slack", "small_E_in_zero_slack_direction",
                         "static_steering_breaks_optimality", "same_norm_different_placement_good",
                         "same_norm_different_placement_bad")
    fixtures_ok = (
        fixtures["nonzero_E_within_slack"]["condition_holds"]
        and not fixtures["small_E_in_zero_slack_direction"]["condition_holds"]
        and not fixtures["static_steering_breaks_optimality"]["full_minimax_condition"]
        and fixtures["same_norm_different_placement_good"]["condition_holds"]
        and not fixtures["same_norm_different_placement_bad"]["condition_holds"]
        and all(name in fixtures for name in required_fixtures)
    )
    expected_rows = (2 * len(GRID)) + (len(seeds) * 4 * len(GRID)) if cmnist else 2 * len(GRID)
    coverage_ok = len(audits) == expected_rows and not errors
    verdict = "SHARP-OPTIMALITY-PASS" if coverage_ok and snapshot_ok and coordinate_ok and cross_ok and fixtures_ok else "SHARP-OPTIMALITY-PARTIAL"
    summary = {
        "verdict": verdict, "audit_row_count": len(audits), "error_count": len(errors), "errors": errors,
        "corrected_snapshot_pass": snapshot_ok, "coordinate_invariance_pass": coordinate_ok,
        "cross_operator_audit_pass": cross_ok, "counterexamples_pass": fixtures_ok,
        "expected_audit_row_count": expected_rows, "full_coverage_pass": coverage_ok,
        "cmnist_scope": "frozen_representation_empirical_squared_loss" if cmnist else "not_run",
        "operator_snapshot_source": str(snapshot_path) if use_snapshots else "regenerated_from_frozen_bridge",
        "metric": "Euclidean after declared family-metric whitening", "lean_status": "LEAN-PARTIAL",
        "target_risk_lower_bound_claimed": False, "semantic_or_cluster_used": False,
    }
    (results / "summary.json").write_text(json.dumps(summary, indent=2, default=_json))
    _docs(output, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-cmnist", action="store_true")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--snapshot-path", type=Path, default=DEFAULT_SNAPSHOTS)
    args = parser.parse_args()
    print(json.dumps(run(args.output, cmnist=not args.no_cmnist,
                         seeds=tuple(int(value) for value in args.seeds.split(",") if value),
                         snapshot_path=args.snapshot_path),
                     indent=2, default=_json))


if __name__ == "__main__":
    main()
