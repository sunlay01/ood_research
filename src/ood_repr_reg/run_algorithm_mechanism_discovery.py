"""Run the first data-first mechanism-family discovery pass."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .algorithm_mechanism_discovery import (
    build_fullnetwork_features,
    build_rephead_features,
    discover_numeric_families,
    evaluate_help_hurt_prediction,
    find_counterexample_pairs,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "round3_redesign" / "algorithm_mechanism_discovery"


def _write_report(output: Path, full: pd.DataFrame, rep: pd.DataFrame, fs: dict, rs: dict, pred: pd.DataFrame, pairs: pd.DataFrame) -> None:
    fs_sil = next((row["silhouette"] for row in fs["candidate_scores"] if row["k"] == fs["selected_k"]), 0.0)
    rs_sil = next((row["silhouette"] for row in rs["candidate_scores"] if row["k"] == rs["selected_k"]), 0.0)
    full_composition = full.groupby("full_cluster")["method"].apply(lambda s: ", ".join(sorted(set(map(str, s))))).to_dict()
    rep_composition = rep.groupby("rep_cluster")["method"].apply(lambda s: ", ".join(sorted(set(map(str, s))))).to_dict()
    lines = [
        "# Data-first mechanism discovery report",
        "",
        "This is a discovery and falsification artifact. Discovery features do not contain target accuracy or target loss. Target outcomes enter only in the external help/hurt audit.",
        "",
        "## Inputs and scope",
        "",
        f"- Full-network rows: {len(full)} across ERM, IRMv1, VREX, FISHR.",
        f"- Representation/head rows: {len(rep)} across IRMv1, Full-BIRM, LoRA-BIRM.",
        "- BIRM/LoRA rows are kept in a separate feature space because their current Pi is fixed-encoder/head-only.",
        "",
        "## Numeric families",
        "",
        f"- Full-network selected k={fs['selected_k']}, silhouette={fs_sil:.3f}, bootstrap pair agreement={fs['bootstrap_pair_agreement_mean']:.3f} +/- {fs['bootstrap_pair_agreement_std']:.3f}.",
        f"- Representation/head selected k={rs['selected_k']}, silhouette={rs_sil:.3f}, bootstrap pair agreement={rs['bootstrap_pair_agreement_mean']:.3f} +/- {rs['bootstrap_pair_agreement_std']:.3f}.",
        f"- Full-network cluster composition: {full_composition}.",
        f"- Representation/head cluster composition: {rep_composition}.",
        "- Cluster IDs are numeric family IDs. They are not semantic mechanism labels.",
        "",
        "## External help/hurt audit",
        "",
        pred.to_string(index=False) if len(pred) else "No predictive rows.",
        "",
        "The method-identity baseline is expected to be strong in this first pass because the existing four-method table has one training configuration per method. This is a confounding warning, not evidence of a mechanism.",
        "The current predictive audit confirms this warning: method identity, source risk, and numeric geometry all obtain perfect leave-one-seed-out accuracy because the help label is almost identical to the method split. No beyond-identity mechanism evidence is claimed.",
        "",
        "## Counterexamples",
        "",
        f"Candidate counterexample pairs: {len(pairs)}. Same-method close-geometry target gaps and different-method close-target geometry gaps are retained for adversarial review.",
        "",
        "## Interpretation ceiling",
        "",
        "The first pass can establish stable response-geometry families only. Forcing/filtering/interaction names require common-base C/K counterfactuals. A family that disappears after controlling method identity, lambda, or environment family is treated as a confounder.",
    ]
    (output / "mechanism_discovery_report.md").write_text("\n".join(lines) + "\n")


def run(output: Path = DEFAULT_OUTPUT) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    full, outcomes = build_fullnetwork_features(ROOT)
    rep = build_rephead_features(ROOT)
    full_clustered, full_stability = discover_numeric_families(full, prefix="full")
    rep_clustered, rep_stability = discover_numeric_families(rep, prefix="rep")
    prediction = evaluate_help_hurt_prediction(full, outcomes)
    pairs = find_counterexample_pairs(full, outcomes)
    full_clustered.to_csv(output / "fullnetwork_feature_matrix.csv", index=False)
    rep_clustered.to_csv(output / "representation_head_feature_matrix.csv", index=False)
    clusters = pd.concat([
        full_clustered[["method", "seed", "full_cluster"]].rename(columns={"full_cluster": "cluster"}).assign(layer="fullnetwork"),
        rep_clustered[["method", "seed", "rep_cluster"]].rename(columns={"rep_cluster": "cluster"}).assign(layer="representation_head"),
    ], ignore_index=True, sort=False)
    clusters.to_csv(output / "cluster_assignments.csv", index=False)
    prediction.to_csv(output / "help_hurt_predictive_audit.csv", index=False)
    pairs.to_csv(output / "counterexample_pairs.csv", index=False)
    stability = {"fullnetwork": full_stability, "representation_head": rep_stability}
    (output / "cluster_stability.json").write_text(json.dumps(stability, indent=2, sort_keys=True))
    provenance = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "target_used_for_discovery": False,
        "target_used_for_external_audit_only": True,
        "full_methods": ["ERM", "IRMv1", "VREX", "FISHR"],
        "representation_head_methods": ["IRMv1", "Full-BIRM (Official)", "LoRA-BIRM (Official)"],
        "source_paths": [
            "round3_redesign/task3_aopi_four_methods_rerun/results",
            "round3_redesign/birm_cmnist_checkpoint_rerun",
        ],
        "feature_columns_full": [c for c in full.columns if c not in {"method", "seed"}],
        "feature_columns_representation_head": [c for c in rep.columns if c not in {"method", "seed", "checkpoint_step"}],
        "limitations": [
            "No lambda sweep in the current four-method inputs.",
            "BIRM/LoRA Pi is head-only and is not merged with full-network Pi.",
            "C/K common-base attribution is not part of this discovery-only pass.",
        ],
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True))
    _write_report(output, full_clustered, rep_clustered, full_stability, rep_stability, prediction, pairs)
    summary = {
        "status": "PROBE",
        "fullnetwork_rows": len(full), "representation_head_rows": len(rep),
        "fullnetwork_stability": full_stability, "representation_head_stability": rep_stability,
        "prediction_models": prediction.to_dict(orient="records"), "counterexample_pairs": len(pairs),
        "target_used_for_discovery": False,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
