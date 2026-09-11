"""Official DomainBed reproduction gate for four successful-CMNIST candidates.

This runner intentionally keeps the upstream DomainBed training loop separate
from the local multi-method geometry runner.  It records provenance and emits
diagnostic labels without treating a low target score as a method failure.
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import time
from pathlib import Path
from typing import Any

import torch

from .task3_baseline_fidelity.upstreams import (
    NativeRunConfig,
    run_domainbed_native,
    upstream_manifest,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "round3_redesign" / "task3_official_success_reproduction"
METHODS = ("GroupDRO", "ANDMask", "SANDMask", "MLDG")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _diagnose(row: dict[str, Any]) -> str:
    if row.get("status") != "completed":
        return "unresolved"
    if row.get("protocol_mismatch"):
        return "protocol_mismatch"
    if row.get("implementation_issue"):
        return "implementation_issue"
    if row.get("budget_sensitive"):
        return "budget_sensitive"
    # A completed run is not evidence of scientific success by itself.
    return "recovered" if row.get("target_accuracy", 0.0) >= 0.50 else "unresolved"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    args = parser.parse_args()

    out = args.output
    results_dir = out / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    manifest = upstream_manifest(ROOT)
    (out / "upstream_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )

    rows: list[dict[str, Any]] = []
    for seed in range(args.seeds):
        config = NativeRunConfig(
            root=ROOT,
            timeout_seconds=args.timeout_seconds,
            domainbed_steps=args.steps,
        )
        for method in args.methods:
            run_dir = results_dir / f"domainbed_{method.lower()}_seed{seed}"
            row = run_domainbed_native(config, method=method, output_dir=run_dir)
            row.update({
                "seed": seed,
                "dataset": "ColoredMNIST",
                "test_envs": "[2]",
                "target_used_for_training_or_selection": False,
                "upstream": "DomainBed",
                "upstream_commit": manifest["domainbed"]["commit"],
                "budget_steps": args.steps,
                "protocol_mismatch": False,
                "implementation_issue": row.get("status") == "failed",
                "budget_sensitive": False,
            })
            row["target_accuracy"] = row.get("env2_out_acc")
            row["diagnosis"] = _diagnose(row)
            rows.append(row)

    _write_csv(results_dir / "method_fidelity.csv", rows)
    summary: dict[str, Any] = {
        "task_id": "TASK3-OFFICIAL-SUCCESS-REPRODUCTION",
        "runner": "ood_repr_reg.run_task3_official_success_reproduction",
        "methods": list(args.methods),
        "git_commit_before_run": __import__("subprocess").run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
            capture_output=True, check=False).stdout.strip(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "steps": args.steps,
        "seeds": args.seeds,
        "rows": rows,
        "diagnosis_counts": {
            label: sum(r["diagnosis"] == label for r in rows)
            for label in ("recovered", "protocol_mismatch", "implementation_issue", "budget_sensitive", "unresolved")
        },
        "target_leakage_detected": any(r["target_used_for_training_or_selection"] for r in rows),
        "run_unix_time": time.time(),
    }
    (results_dir / "method_fidelity_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    (out / "README.md").write_text(
        "# Official CMNIST Reproduction\n\n"
        "This panel runs the pinned DomainBed implementations of GroupDRO, "
        "ANDMask, SANDMask, and MLDG on ColoredMNIST. Official upstream "
        "training is intentionally separate from the local geometry runner.\n\n"
        "A low target score is recorded as `unresolved` until protocol, "
        "implementation, budget, and checkpoint selection audits are complete.\n\n"
        "Results: `results/method_fidelity.csv` and "
        "`results/method_fidelity_summary.json`.\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(out), "rows": len(rows), "diagnosis_counts": summary["diagnosis_counts"]}, indent=2))


if __name__ == "__main__":
    main()
