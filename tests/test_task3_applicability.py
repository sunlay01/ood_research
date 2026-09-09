from __future__ import annotations

import math

import numpy as np

from ood_repr_reg.run_task3_applicability import run
from ood_repr_reg.task3_applicability.evaluation import (
    _permute_theory_features,
    _verdict,
    evaluate_applicability,
    grouped_cv_keeps_runs_together,
    logic_audit,
)
from ood_repr_reg.task3_applicability.features import FEATURE_SETS, build_feature_dictionary
from ood_repr_reg.task3_applicability.targets import (
    MaterializedRun,
    _cmnist_target_rows,
    _gaussian_source_metrics,
    _gaussian_target_rows,
    build_design,
    cmnist_families,
    gaussian_worlds,
)


def test_preregistration_logic_only_writes_no_outcomes(tmp_path):
    summary = run(tmp_path, smoke=True, logic_only=True)
    assert summary["verdict"] == "LOGIC-AUDIT-PASS"
    assert (tmp_path / "task3_preregistered_design.json").exists()
    assert not (tmp_path / "results" / "per_target_outcomes.csv").exists()


def test_feature_dictionary_has_no_target_outcome_leakage():
    dictionary = build_feature_dictionary()
    assert dictionary
    assert all(not row["algebraically_contains_outcome"] for row in dictionary.values())
    assert all(not row["uses_target_outcome_for_feature_construction"] for row in dictionary.values())
    assert logic_audit(build_design(), dictionary)["passes"]


def test_b1_contains_method_lambda_controls():
    assert "method" in FEATURE_SETS["B1"]
    assert "lambda" in FEATURE_SETS["B1"]
    assert "source_risk" in FEATURE_SETS["B1"]


def test_grouped_cv_keeps_training_runs_together():
    rows = []
    for seed in ("0", "1"):
        for run in ("a", "b"):
            for target in ("t0", "t1"):
                rows.append({"benchmark": "cmnist", "seed": seed, "training_run_id": f"{seed}-{run}", "target_id": target})
    assert grouped_cv_keeps_runs_together(rows, "cmnist")


def test_rho_slack_inf_handling_is_deterministic(tmp_path):
    feature_rows = []
    outcome_rows = []
    for seed in ("0", "1"):
        for index, method in enumerate(("L2", "IRMV1", "VREX")):
            run_id = f"cmnist|fam|{method}|0.01|{seed}"
            feature_rows.append({
                "benchmark": "cmnist", "family_config": "fam", "method": method, "lambda": 0.01,
                "seed": seed, "training_run_id": run_id, "source_risk": index, "regularizer_value": 0.1,
                "source_objective": index + 0.1, "parameter_displacement_from_erm": 0.0,
                "z0_norm": 0.1 * index, "pi_operator_norm": 1.0, "pi_frobenius_norm": 1.0,
                "K_operator_norm": 1.0, "C_operator_norm": 0.1, "g_norm": 0.2,
                "E_operator_norm": index + 1.0, "E_frobenius_norm": index + 1.0,
                "rho_slack": math.inf if index == 1 else float(index), "slack_margin": index,
                "R_info": 0.0, "rank_O_S": 2, "dim_ker_O_S": 0, "rank_A_irr": 0,
            })
            for target in ("t0", "t1"):
                outcome_rows.append({
                    "benchmark": "cmnist", "family_config": "fam", "method": method, "lambda": 0.01,
                    "seed": seed, "training_run_id": run_id, "target_id": target,
                    "radius_regime": "local", "delta_target_risk_vs_erm": float(index),
                    "target_squared_loss_risk": float(index), "win_loss_vs_erm": index == 0,
                })
    first = evaluate_applicability(feature_rows, outcome_rows, tmp_path / "a", build_design(smoke=True))
    second = evaluate_applicability(feature_rows, outcome_rows, tmp_path / "b", build_design(smoke=True))
    assert first["predictive_row_count"] == second["predictive_row_count"]


def test_permutation_breaks_feature_run_correspondence():
    rows = []
    for seed in range(5):
        rows.append({
            "benchmark": "cmnist", "family_config": "fam", "method": "L2", "lambda": 0.01,
            "seed": str(seed), "training_run_id": f"run-{seed}", "E_operator_norm": float(seed),
        })
    permuted = _permute_theory_features(rows, seed=3)
    assert sorted(row["E_operator_norm"] for row in permuted) == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert [row["E_operator_norm"] for row in permuted] != [row["E_operator_norm"] for row in rows]


def test_gaussian_finite_targets_are_legal():
    design = build_design(smoke=True)
    world = gaussian_worlds()["hidden_u_hurts"]
    run_row = _gaussian_source_metrics(world, "L2", 0.0)
    rows = _gaussian_target_rows(run_row, design)
    valid = [row for row in rows if np.isfinite(float(row["target_squared_loss_risk"]))]
    assert valid
    assert all(row["target_outcome_generation"] == "legal_gaussian_environment_family" for row in valid)


def test_cmnist_target_generation_does_not_mutate_source_inputs():
    family = cmnist_families()["mechanism_defined_hidden"]
    bank = type("Bank", (), {})()
    bank.red = np.asarray([[0.0], [1.0], [0.5]])
    bank.green = np.asarray([[1.0], [0.0], [0.25]])
    bank.labels = np.asarray([0.0, 1.0, 1.0])
    bank.dimension = 2
    weights = np.asarray([0.2, 0.4])
    before = (bank.red.copy(), bank.green.copy(), bank.labels.copy())
    run_row = MaterializedRun(
        "cmnist", family.name, "L2", 0.0, "0", "run", weights, weights, 0.0, 0.0, 0.0, 0.0,
        family=family, bank=bank,
    )
    rows = _cmnist_target_rows(run_row, build_design(smoke=True))
    assert any(np.isfinite(float(row["target_squared_loss_risk"])) for row in rows)
    assert np.array_equal(bank.red, before[0])
    assert np.array_equal(bank.green, before[1])
    assert np.array_equal(bank.labels, before[2])


def test_verdict_gate_support_requires_both_benchmarks():
    logic = {"passes": True}
    rows = [
        {"benchmark": "gaussian", "test": "ranking", "B3_minus_B1": 0.2},
        {"benchmark": "gaussian", "test": "finite_risk_prediction", "B3_minus_B1": 0.2},
    ]
    assert _verdict(rows, [], logic) == "TASK3-APPLICABILITY-PARTIAL"
    rows.extend([
        {"benchmark": "cmnist", "test": "ranking", "B3_minus_B1": 0.2},
        {"benchmark": "cmnist", "test": "finite_risk_prediction", "B3_minus_B1": 0.2},
    ])
    assert _verdict(rows, [], logic) == "TASK3-APPLICABILITY-SUPPORT"
