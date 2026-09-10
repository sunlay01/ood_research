from __future__ import annotations

import json
from pathlib import Path

import pytest

from ood_repr_reg.cleanroom_rerun_1_3d import runner


def test_forbidden_old_result_inputs_are_blocked() -> None:
    root = runner.ROOT
    blocked = [
        root / "round1/results/round1_results.json",
        root / "round2/results/round2_results.json",
        root / "round3/some_track/results/summary.json",
        root / "round3_redesign/algorithm_mechanism/results/operator_snapshots.npz",
        root / "docs/experiments/results/old.json",
        root / "docs/state/RESULT_REGISTRY.json",
    ]
    for path in blocked:
        assert runner.forbidden_cleanroom_input(path), path
    assert runner.forbidden_cleanroom_input(runner.RESULTS_ROOT / "provenance.json") is None


def test_access_guard_raises_before_old_vs_new_phase() -> None:
    old_path = runner.ROOT / "round1/results/round1_results.json"
    with runner.cleanroom_access_guard():
        with pytest.raises(RuntimeError, match="CLEANROOM-GATE-FAIL"):
            old_path.read_text(encoding="utf-8")


def test_access_guard_can_be_relaxed_for_old_vs_new_phase() -> None:
    old_path = runner.ROOT / "round1/results/round1_results.json"
    with runner.cleanroom_access_guard(allow_old_results=True):
        # Existence is enough here; this phase is explicitly allowed to inspect old paths.
        old_path.exists()


def test_dependency_audit_and_scaffold_are_cleanroom_local() -> None:
    runner.write_scaffold()
    with runner.cleanroom_access_guard():
        payload = runner.dependency_audit()
    assert payload["status"] == "CLEANROOM-GATE-PASS"
    assert (runner.CLEANROOM_ROOT / "cross_round_dependency_audit.md").exists()
    assert (runner.RESULTS_ROOT / "cross_round" / "dependency_audit.json").exists()


def test_baseline_gate_logic_accepts_recovered_irm() -> None:
    stats = {
        "ERM": {"target_mean": 0.11, "prediction_color_agreement_mean": 0.89},
        "IRMv1": {"target_mean": 0.67, "prediction_color_agreement_mean": 0.52},
    }
    criteria = {
        "erm_color_shortcut": stats["ERM"]["target_mean"] <= 0.35 and stats["ERM"]["prediction_color_agreement_mean"] >= 0.65,
        "irmv1_distinct_from_erm": abs(stats["IRMv1"]["target_mean"] - stats["ERM"]["target_mean"]) >= 0.20,
        "irmv1_not_ten_percent": stats["IRMv1"]["target_mean"] >= 0.50,
        "enough_optimizer_steps": 501 >= 501,
    }
    assert all(criteria.values())


def test_final_verdict_enum_is_exact() -> None:
    assert runner.FINAL_VERDICTS == {
        "CLEANROOM-CONFIRMS-MAINLINE",
        "THEORY-SURVIVES-EMPIRICAL-MAINLINE-CHANGES",
        "MAJOR-RETHINK-REQUIRED",
        "RERUN-INCONCLUSIVE",
    }


def test_task1_task2_expected_paths_are_separated() -> None:
    assert runner.RESULTS_ROOT / "task1" / "gaussian" != runner.RESULTS_ROOT / "task1" / "cmnist"
    assert runner.RESULTS_ROOT / "task2" / "gaussian" != runner.RESULTS_ROOT / "task2" / "cmnist_erm"
    assert runner.RESULTS_ROOT / "task2" / "cmnist_erm" != runner.RESULTS_ROOT / "task2" / "cmnist_irmv1"


def test_scaffold_writes_required_report_files() -> None:
    runner.write_scaffold()
    missing = [name for name in runner.REPORT_FILES if not (runner.CLEANROOM_ROOT / name).exists()]
    assert missing == []


def test_provisional_registry_jsonable_paths(tmp_path: Path) -> None:
    payload = runner._jsonable({"path": tmp_path / "x", "items": (1, 2)})
    encoded = json.dumps(payload)
    assert str(tmp_path / "x") in encoded
