import json
import importlib.util
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _state_checker_module():
    spec = importlib.util.spec_from_file_location(
        "check_project_state", ROOT / "scripts/check_project_state.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_project_state_checker_passes():
    completed = subprocess.run(
        [sys.executable, "scripts/check_project_state.py"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout
    result = json.loads(completed.stdout)
    assert result["ok"]
    # Active task context is included when a task is in progress; the two
    # canonical files remain mandatory in every boot context.
    assert result["boot_context"]["file_count"] in (2, 4)
    if result["boot_context"]["file_count"] == 4:
        assert "TASK3-" in (ROOT / "active/TASK.md").read_text(encoding="utf-8")
    assert result["boot_context"]["line_count"] < 600


def test_boot_context_reconstructs_current_state_without_history():
    state = (ROOT / "CURRENT_STATE.md").read_text(encoding="utf-8")
    theorem = (ROOT / "docs/state/THEOREM_REGISTRY.md").read_text(encoding="utf-8")
    decision = (ROOT / "docs/state/DECISION_LOG.md").read_text(encoding="utf-8")
    results = json.loads((ROOT / "docs/state/RESULT_REGISTRY.json").read_text(encoding="utf-8"))

    assert "Round 3 redesign after Task 1, Task 2, and repair evidence gate" in state
    assert "Task 3 is next" in state
    assert "Task 3 is not executed" in state
    for theorem_id in (
        "T-3A-A", "T-3C-PI", "T-3D-IDENTIFIABILITY",
        "T-3E-INFO-FLOOR", "T-3E-SPECTRAL-SLACK", "T-3E-AFFINE-REGRET",
    ):
        assert theorem_id in theorem
    assert "D-SEMANTIC-LATENT-SUPERSEDED" in decision
    assert "D-REG-EXPOSURE-CONTAINMENT-REJECTED" in decision
    assert {row["id"] for row in results} >= {
        "R-REPAIR-GATE", "R-PYTEST-FULL", "R-LEAN-BUILD",
        "R-TASK1-MECHANISM", "R-TASK2-SHARP",
        "R-LEGACY-HIDDEN-U", "R-SOURCE-INDUCED",
    }


def test_legacy_state_files_are_neutralized():
    for path in (
        "README.md",
        "CONTEXT_MANAGEMENT.md",
        "PROJECT_PROMPT_CN.md",
        "docs/research/research_state.md",
        "docs/research/open_questions.md",
    ):
        text = (ROOT / path).read_text(encoding="utf-8").lower()
        assert "latent-001 is current" not in text
        assert "repair gate is blocked" not in text
        assert "task 1 / task 2 are pending" not in text


def test_file_status_marks_history_default_read_false_and_deleted_material_protected():
    rows = json.loads((ROOT / "docs/state/FILE_STATUS.json").read_text(encoding="utf-8"))
    by_path = {row["path_or_glob"]: row for row in rows}
    for glob in ("round1/**", "round2/**", "round3/**", "round3_redesign/**"):
        assert by_path[glob]["default_read"] is False
    mech = by_path["MECH-001/C011"]
    assert mech["status"] == "DO_NOT_RESTORE"
    assert mech["do_not_restore"] is True
    assert mech["do_not_reference"] is True


def test_do_not_restore_glob_absence_is_success_and_tracked_presence_fails(monkeypatch):
    checker = _state_checker_module()
    records = [{
        "path_or_glob": "artifacts/MECH-001*",
        "status": "DO_NOT_RESTORE",
        "default_read": False,
        "do_not_restore": True,
        "do_not_reference": True,
    }]

    errors: list[str] = []
    monkeypatch.setattr(checker, "_tracked_paths_matching", lambda pattern: [])
    checker._check_file_status_records(records, errors)
    assert errors == []

    monkeypatch.setattr(checker, "_tracked_paths_matching", lambda pattern: ["artifacts/MECH-001"])
    checker._check_file_status_records(records, errors)
    assert errors == ["forbidden do-not-restore tracked path exists: artifacts/MECH-001*"]
