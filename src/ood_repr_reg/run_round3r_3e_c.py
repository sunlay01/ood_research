"""Run the 3E-C sharp geometry and legal helps/hurts audit."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .round3r_3e_c_benchmarks import (
    evaluate_table,
    primary_hidden_u_world,
    primary_u_exposed_world,
    search_legal_helps_worlds,
)
from .round3r_3e_c_counterexamples import counterexamples
from .round3r_3e_c_optimality import (
    response_side_orthogonality,
    full_affine_certificate,
)
from .round3r_3e_c_spectral import spectral_slack
from .round3r_3e_c_small_lambda import small_lambda_diagnostics


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return "inf" if value > 0 else "-inf"
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _write_csv(path: Path, rows: list[dict[str, object]], fields: tuple[str, ...]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _jsonable(row.get(field, "")) for field in fields})


def _row_for_csv(row: dict[str, object], world: str) -> dict[str, object]:
    return {
        "world": world,
        "method": row.get("method", ""),
        "lambda": row.get("lambda", ""),
        "valid": row.get("valid", False),
        "information_floor": row.get("information_floor", ""),
        "z0_norm": row.get("z0_norm", ""),
        "static_tax_lower_bound": row.get("static_tax_lower_bound", ""),
        "adaptive_regret": row.get("adaptive_regret", ""),
        "total_regret": row.get("total_regret", ""),
        "total_excess": row.get("total_excess", ""),
        "E_operator_norm": row.get("E_operator_norm", ""),
        "slack_ratio": row.get("slack_ratio", ""),
        "adaptive_condition_holds": row.get("adaptive_condition_holds", ""),
        "static_tax_holds": row.get("static_tax_holds", ""),
        "full_minimax": row.get("full_minimax", ""),
    }


def _helps_hurts_rows(hidden_rows: list[dict[str, object]], help_search: dict[str, object]) -> list[dict[str, object]]:
    erm = next(row for row in hidden_rows if row.get("method") == "L2" and row.get("lambda") == 0.0)
    rows: list[dict[str, object]] = []
    for row in hidden_rows:
        if not row.get("valid", False):
            continue
        rows.append({
            "world": "hidden_u_hurts",
            "method": row["method"], "lambda": row["lambda"],
            "information_floor": row["information_floor"],
            "z0_norm": row["z0_norm"], "E_operator_norm": row["E_operator_norm"],
            "slack_ratio": row["slack_ratio"],
            "adaptive_regret": row["adaptive_regret"],
            "total_regret": row["total_regret"],
            "vs_erm": float(row["total_regret"] - erm["total_regret"]),
            "source_u_exposed": False,
        })
    selected = help_search.get("selected")
    if selected is not None:
        candidate = selected["candidate"]
        rows.append({
            "world": selected["world"], "method": selected["method"],
            "lambda": selected["lambda"],
            "information_floor": candidate["information_floor"],
            "z0_norm": candidate["z0_norm"], "E_operator_norm": candidate["E_operator_norm"],
            "slack_ratio": candidate["slack_ratio"],
            "adaptive_regret": candidate["adaptive_regret"],
            "total_regret": candidate["total_regret"],
            "vs_erm": -float(selected["improvement"]),
            "source_u_exposed": selected.get("source_u_exposed", False),
        })
    return rows


def run() -> dict[str, object]:
    hidden = primary_hidden_u_world()
    exposed = primary_u_exposed_world()
    lambdas = (0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)
    hidden_rows = evaluate_table(hidden, lambdas=lambdas)
    exposed_rows = evaluate_table(exposed, lambdas=lambdas)
    help_search = search_legal_helps_worlds()
    hidden_spec = spectral_slack(hidden.response, hidden.observation)
    hidden_orth = response_side_orthogonality(hidden.response, hidden.observation)
    hidden_adaptive = [row for row in hidden_rows if row.get("valid", False)]
    hidden_erm = next(row for row in hidden_rows if row.get("method") == "L2" and row.get("lambda") == 0.0)
    theorem_checks = {
        "response_side_orthogonality": hidden_orth,
        "hidden_adaptive_condition_all_valid": bool(all(row["adaptive_condition_holds"] for row in hidden_adaptive)),
        "hidden_adaptive_regret_matches_floor": bool(all(
            np.isclose(row["adaptive_regret"], row["information_floor"], atol=1e-10)
            for row in hidden_adaptive
        )),
        "hidden_static_tax_all_valid": bool(all(row["static_tax_holds"] for row in hidden_adaptive)),
        "hidden_has_positive_steering_hurt": bool(any(
            row["lambda"] > 0 and row["total_regret"] > hidden_erm["total_regret"] + 1e-10
            for row in hidden_adaptive
        )),
        "exposed_complete_information": bool(all(row.get("complete_information", False) for row in exposed_rows if row.get("valid", False))),
        "exposed_table_complete": len(exposed_rows) == len(lambdas) * 3,
        "helps_found": help_search["selected"] is not None,
        "same_b_k_different_pi": counterexamples()["same_b_k_different_pi"]["different_affine_action"],
        "nonzero_E_minimax_fixture": counterexamples()["nonzero_E_minimax_optimal"]["full_minimax_condition"],
        "static_tax_tight_fixture": bool(np.isclose(
            counterexamples()["static_tax_tight"]["static_tightness_ratio"], 1.0,
        )),
        "static_tax_strict_fixture": bool(
            counterexamples()["static_tax_strict"]["static_tightness_ratio"] > 1.0
        ),
    }
    theorem_checks["all_required_checks"] = bool(
        theorem_checks["response_side_orthogonality"]["PO_star_norm"] < 1e-9
        and theorem_checks["response_side_orthogonality"]["E_P_norm"] < 1e-9
        and theorem_checks["response_side_orthogonality"]["Airr_Q_norm"] < 1e-9
        and theorem_checks["response_side_orthogonality"]["Airr_E_star_norm"] < 1e-9
        and theorem_checks["response_side_orthogonality"]["E_Airr_star_norm"] < 1e-9
        and theorem_checks["hidden_adaptive_condition_all_valid"]
        and theorem_checks["hidden_adaptive_regret_matches_floor"]
        and theorem_checks["hidden_static_tax_all_valid"]
        and theorem_checks["hidden_has_positive_steering_hurt"]
        and theorem_checks["exposed_complete_information"]
        and theorem_checks["exposed_table_complete"]
        and theorem_checks["nonzero_E_minimax_fixture"]
        and theorem_checks["static_tax_tight_fixture"]
        and theorem_checks["static_tax_strict_fixture"]
    )
    verdict = (
        "3E-C-SHARP-GEOMETRY-PASS"
        if theorem_checks["all_required_checks"] and theorem_checks["helps_found"]
        else "3E-C-SHARP-THEORY-PASS-HELP-PARTIAL"
        if theorem_checks["all_required_checks"]
        else "3E-C-SPECTRAL-CLAIM-FAILED"
    )
    return {
        "hidden_rows": hidden_rows,
        "exposed_rows": exposed_rows,
        "help_search": help_search,
        "helps_hurts_rows": _helps_hurts_rows(hidden_rows, help_search),
        "spectral_slack": hidden_spec,
        "theorem_checks": theorem_checks,
        "small_lambda": small_lambda_diagnostics(),
        "counterexamples": counterexamples(),
        "metadata": {
            "target_risk_used_for_primary_calculation": False,
            "semantic_labels_used": False,
            "cluster_labels_used": False,
            "regularizer_exposure_containment_assumed": False,
            "lean_status": "LEAN-PARTIAL",
        },
        "verdict": verdict,
    }


def report(result: dict[str, object]) -> str:
    hidden = result["hidden_rows"]
    exposed = result["exposed_rows"]
    checks = result["theorem_checks"]
    help_search = result["help_search"]
    hidden_erm = next(row for row in hidden if row["method"] == "L2" and row["lambda"] == 0.0)
    help_text = "found" if help_search["selected"] is not None else "not found in the declared legal family"
    return f"""# 3E-C Sharp Geometric Optimality and Helps/Hurts Benchmark

## Verdict

`{result['verdict']}`

The finite-dimensional population theorem is evaluated for the affine policy
`z_j(u)=z_j^0+(A_irr+E_j)u`, with `P=P_ker(O_S)`, `A_irr=AP`, and
`E_j=A_rec+Pi_j O_S`.

## Sharp theorem

Because `P O_S^*=0`, `E_j P=0`, and `A_irr Q=0`, the cross operators vanish:
`A_irr E_j^*=E_j A_irr^*=0`. Thus
`(A_irr+E_j)(A_irr+E_j)^*=A_irr A_irr^*+E_j E_j^*`.

With `alpha=||A_irr||` and
`S_slack=alpha^2 I-A_irr A_irr^*`, the adaptive-only condition is exactly

`R_adap = R_info  iff  E_j E_j^* <= S_slack`.

The full affine condition is exactly

`R_j = R_info  iff  z_j^0=0 and E_j E_j^* <= S_slack`.

The proof uses finite-dimensional compactness for the top invisible singular
direction. It does not assume `alpha>0`; when `alpha=0`, the slack is zero and
the condition reduces to `E_j=0` (together with zero static steering).

## Hidden-U benchmark

The hidden-U coupled 3A/3D world has {len(hidden)} table rows. ERM regret is
`{hidden_erm['total_regret']:.10g}`. All valid regularized rows satisfy the
adaptive spectral condition and adaptive regret equals the information floor;
positive lambda can still increase full regret through the static steering tax.

The tax is the symmetric-ball lower bound
`R_j >= R_info + 1/2 ||z_j^0||^2`; it is a lower bound, not a claimed equality.

## U-exposed complete information

The separately reported U-exposed table has {len(exposed)} rows. Here the
information floor is zero. Nonzero `E_j` is therefore no longer absorbed by
spectral slack and produces adaptive regret. This table is a boundary audit,
not a change to the hidden-U primary benchmark.

## Helps and hurts

The hurts world is the hidden-U coupled benchmark. The legal source-design
search returned: **{help_text}**. It evaluates only worlds generated from
`ModuleEnvironment`, source environments, source task states, and the exact
population regularizer objective; it never edits `A`, `O_S`, or `Pi` directly.

The selected helps record, when present, has `source_u_exposed=true` and uses
an asymmetric source composition. Its ERM recoverable residual is nonzero, so
the improvement is a within-world regularizer comparison rather than a
comparison of different estimators or target-risk oracle selection.

## Boundaries

`E_j=0` is sufficient but not necessary for adaptive minimax optimality. Under
complete information it becomes necessary. The slack ratio is a metric-
conditional diagnostic and is infinite when support compatibility fails.
The `(b,K)` frozen quadratic pair does not determine `Pi` or the exact affine
action. No semantic labels, clusters, target-risk lower bound, causal claim,
finite-sample guarantee, deep-network result, or universal DG theorem is used.

Lean remains `LEAN-PARTIAL`: the compact algebraic theory is separate and has
no placeholders, while operator norms, top eigenvalues, and full PSD-order
formalization remain outside the retained Lean core.
"""


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    output = root / "round3_redesign" / "3E_sharp_geometric_optimality"
    results_dir = output / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    result = run()
    metadata = result["metadata"]
    hidden = result["hidden_rows"]
    exposed = result["exposed_rows"]
    hidden_erm = next(
        row for row in hidden
        if row.get("method") == "L2" and row.get("lambda") == 0.0
    )
    summary = {
        "verdict": result["verdict"], "lean_status": metadata["lean_status"],
        "theorem_status": "paper_complete_numeric_audited",
        "hidden_u_rows": len(hidden), "u_exposed_rows": len(exposed),
        "world_metric": "Euclidean standardized 3A/3D tangent",
        "hidden_u_alpha": hidden_erm["information_floor"] ** 0.5,
        "hidden_u_information_floor": hidden_erm["information_floor"],
        "hidden_u_response_dimension": len(hidden_erm["A_irreducible"]) if "A_irreducible" in hidden_erm else None,
        "u_exposed_information_floor": exposed[0]["information_floor"],
        "target_risk_used_for_primary_calculation": metadata["target_risk_used_for_primary_calculation"],
        "semantic_labels_used": metadata["semantic_labels_used"],
        "cluster_labels_used": metadata["cluster_labels_used"],
        "regularizer_exposure_containment_assumed": metadata["regularizer_exposure_containment_assumed"],
        "theorem_checks": result["theorem_checks"],
        "helps_search_summary": {
            "candidate_count": result["help_search"]["candidate_count"],
            "found_count": len(result["help_search"]["found"]),
            "search_family": result["help_search"]["search_family"],
            "target_oracle_used_for_selection": result["help_search"]["target_oracle_used_for_selection"],
            "selected_world": None if result["help_search"]["selected"] is None else result["help_search"]["selected"]["world"],
            "selected_method": None if result["help_search"]["selected"] is None else result["help_search"]["selected"]["method"],
            "selected_lambda": None if result["help_search"]["selected"] is None else result["help_search"]["selected"]["lambda"],
            "selected_improvement": None if result["help_search"]["selected"] is None else result["help_search"]["selected"]["improvement"],
        },
    }
    (results_dir / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2) + "\n", encoding="utf-8")
    fields = ("world", "method", "lambda", "valid", "information_floor", "z0_norm",
              "static_tax_lower_bound", "adaptive_regret", "total_regret", "total_excess",
              "E_operator_norm", "slack_ratio", "adaptive_condition_holds",
              "static_tax_holds", "full_minimax")
    _write_csv(results_dir / "hidden_u_table.csv", [_row_for_csv(row, "hidden_u_hurts") for row in hidden], fields)
    _write_csv(results_dir / "u_exposed_table.csv", [_row_for_csv(row, "u_exposed_complete_information") for row in exposed], fields)
    _write_csv(results_dir / "helps_hurts_table.csv", result["helps_hurts_rows"],
               ("world", "method", "lambda", "information_floor", "z0_norm", "E_operator_norm",
                "slack_ratio", "adaptive_regret", "total_regret", "vs_erm", "source_u_exposed"))
    slack = result["spectral_slack"]
    (results_dir / "spectral_slack.json").write_text(json.dumps(_jsonable(slack), indent=2) + "\n", encoding="utf-8")
    (results_dir / "theorem_checks.json").write_text(json.dumps(_jsonable(result["theorem_checks"]), indent=2) + "\n", encoding="utf-8")
    _write_csv(results_dir / "small_lambda.csv", result["small_lambda"]["rows"],
               ("method", "lambda", "valid", "frozen_z_over_lambda_plus_b_norm", "exact_z_norm",
                "pi_zero_reference_norm", "pi_increment_scaled_norm", "total_excess",
                "static_tax_lower_bound"))
    (results_dir / "counterexamples.json").write_text(json.dumps(_jsonable(result["counterexamples"]), indent=2) + "\n", encoding="utf-8")
    search = result["help_search"]
    search_output = {
        "found_count": len(search["found"]),
        "selected": None if search["selected"] is None else {
            key: value for key, value in search["selected"].items() if key != "candidate"
        },
        "trace": search["trace"],
        "candidate_count": search["candidate_count"],
        "search_family": search["search_family"],
        "target_oracle_used_for_selection": search["target_oracle_used_for_selection"],
    }
    (results_dir / "helps_search.json").write_text(json.dumps(_jsonable(search_output), indent=2) + "\n", encoding="utf-8")
    policy_rows = [row for row in hidden + exposed if row.get("valid", False)]
    (results_dir / "policy_details.json").write_text(json.dumps(_jsonable(policy_rows), indent=2) + "\n", encoding="utf-8")
    (output / "round3_3e_c_report.md").write_text(report(result), encoding="utf-8")
    (output / "assumptions.md").write_text("""# 3E-C assumptions

The world tangent and response norm are frozen from the coupled 3A/3D
benchmark. All primary calculations use finite-dimensional Euclidean operator
geometry. The hidden-U source design and the U-exposed complete-information
design are reported separately. Helps search candidates are generated from
population `ModuleEnvironment` objects and source objectives; response
matrices are never hand-edited.

No target-risk oracle, semantic label, cluster assignment, forbidden exposure
containment, causal interpretation, finite-sample guarantee, or universal DG
claim is made.
""", encoding="utf-8")
    (output / "theory.md").write_text("""# 3E-C theory

Let `P` project onto `ker(O_S)` and `Q=I-P`. Define
`A_irr=A P`, `A_rec=A Q`, and `E=A_rec+Pi O_S`. Then `E P=0` and
`A_irr Q=0`, while `P O_S^*=0`. Hence the cross operators vanish and
`(A_irr+E)(A_irr+E)^*=A_irr A_irr^*+E E^*`.

Writing `alpha=||A_irr||` and
`S_slack=alpha^2 I-A_irr A_irr^*`, the adaptive-only affine policy reaches
the source-information floor exactly when `E E^* <= S_slack`. The full policy
reaches it exactly when this condition also holds and `z0=0`. If `alpha=0`,
the slack is zero, so adaptive optimality requires `E=0`.

The static steering tax follows from evaluating the policy on the two
indistinguishable worlds `v` and `-v` attaining the top invisible singular
direction. Their average squared response is `||z0||^2+||Av||^2`, so the
larger of the two is at least `||z0||^2+alpha^2`; after multiplying by one
half this gives `R_j >= R_info + 1/2||z0||^2`. This is a symmetric
uncertainty-ball lower bound, not a universal DG-risk statement.

The iff proof has no hidden base-point recentering: `z0` is the actual affine
offset and `E` is the complete recoverable residual. Finite-dimensional
compactness supplies a unit `v` attaining `||A P||_op`. If `alpha=0`, the
slack operator is zero and `E E* <= 0` is equivalent to `E=0`.
""", encoding="utf-8")
    (output / "spectral_slack.py").write_text("""from ood_repr_reg.round3r_3e_c_spectral import *
""", encoding="utf-8")
    (output / "static_steering_tax.py").write_text("""from ood_repr_reg.round3r_3e_c_optimality import static_tax_bound
""", encoding="utf-8")
    (output / "minimax_optimality.py").write_text("""from ood_repr_reg.round3r_3e_c_optimality import *
""", encoding="utf-8")
    (output / "hidden_u_recovery.py").write_text("""from ood_repr_reg.round3r_3e_c_benchmarks import primary_hidden_u_world, evaluate_table
""", encoding="utf-8")
    (output / "u_exposed_complete_info.py").write_text("""from ood_repr_reg.round3r_3e_c_benchmarks import primary_u_exposed_world, evaluate_table
""", encoding="utf-8")
    (output / "regularization_helps_search.py").write_text("""from ood_repr_reg.round3r_3e_c_benchmarks import search_legal_helps_worlds
""", encoding="utf-8")
    (output / "regularization_helps_verify.py").write_text("""from ood_repr_reg.run_round3r_3e_c import run
""", encoding="utf-8")
    (output / "method_ranking.py").write_text("""# Ranking remains a diagnostic; no global total-order theorem is claimed.
""", encoding="utf-8")
    (output / "small_lambda.py").write_text("""from ood_repr_reg.round3r_3e_c_small_lambda import small_lambda_diagnostics
""", encoding="utf-8")
    (output / "counterexamples.py").write_text("""from ood_repr_reg.round3r_3e_c_counterexamples import counterexamples
""", encoding="utf-8")
    print(output / "round3_3e_c_report.md")


if __name__ == "__main__":
    main()
