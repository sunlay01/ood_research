"""Run the gated LATENT-001 semantic latent-space experiment."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .latent_semantic import (
    COMPONENTS,
    METHODS,
    component_risk_accounting,
    conservative_component_bound,
    default_scm,
    fit_semantic_decomposition,
    sample_batches,
    source_environments,
    target_environments,
    train_model,
)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path.name}")
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run(config: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    projector_dir = output_dir / "projectors"
    projector_dir.mkdir(exist_ok=True)
    scm = default_scm()
    sources = source_environments(scm)
    targets = target_environments(scm)
    methods = tuple(config["methods"])
    unknown = sorted(set(methods) - set(METHODS))
    if unknown:
        raise ValueError(f"unknown methods: {unknown}")

    run_rows: list[dict[str, Any]] = []
    semantic_rows: list[dict[str, Any]] = []
    risk_rows: list[dict[str, Any]] = []
    started = time.time()
    for seed in config["seeds"]:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        train_batches = sample_batches(scm, sources, int(config["n_train"]), seed)
        semantic_a = sample_batches(scm, sources, int(config["n_semantic"]), seed + 10_000)
        semantic_b = sample_batches(scm, sources, int(config["n_semantic"]), seed + 20_000)
        for latent_dim in config["latent_dims"]:
            for method in methods:
                strengths = [0.0] if method == "erm" else config["strengths"]
                for strength in strengths:
                    model, training = train_model(
                        method,
                        train_batches,
                        input_dim=scm.input_dim,
                        latent_dim=int(latent_dim),
                        seed=int(seed),
                        strength=float(strength),
                        steps=int(config["steps"]),
                        learning_rate=float(config["learning_rate"]),
                    )
                    decomposition = fit_semantic_decomposition(
                        model,
                        scm,
                        semantic_a,
                        semantic_b,
                        permutation_seed=int(seed) + 30_000,
                    )
                    run_id = f"seed{seed}-z{latent_dim}-{method}-l{float(strength):g}"
                    np.savez_compressed(
                        projector_dir / f"{run_id}.npz",
                        **{f"projector_{name}": decomposition.projectors[name] for name in COMPONENTS},
                        whitening_mean=decomposition.whitening.mean,
                        whitening_transform=decomposition.whitening.transform,
                    )
                    base = {
                        "run_id": run_id,
                        "seed": seed,
                        "latent_dim": latent_dim,
                        "support_rank": decomposition.whitening.rank,
                        "method": method,
                        "lambda": strength,
                    }
                    run_rows.append({**base, **training, **decomposition.diagnostics})
                    for component in COMPONENTS:
                        semantic_rows.append(
                            {
                                **base,
                                "component": component,
                                "rank": float(np.trace(decomposition.projectors[component])),
                                "crossfit_overlap": decomposition.diagnostics[
                                    f"crossfit_overlap_{component}"
                                ],
                                "oracle_recovery": decomposition.diagnostics[
                                    f"oracle_recovery_{component}"
                                ],
                                "oracle_angle_radians": decomposition.diagnostics[
                                    f"oracle_angle_{component}"
                                ],
                                "permuted_recovery": decomposition.diagnostics[
                                    f"permuted_recovery_{component}"
                                ],
                                "semantic_status": (
                                    "IDENTIFIED"
                                    if decomposition.diagnostics["semantic_identified"]
                                    else "SEMANTIC_DECOMPOSITION_NOT_IDENTIFIED"
                                ),
                            }
                        )
                    bound = conservative_component_bound(
                        model, scm, sources, decomposition, **config["intervention_budget"]
                    )
                    for target in targets:
                        accounting = component_risk_accounting(
                            model, scm, sources, target, decomposition
                        )
                        risk_rows.append(
                            {
                                **base,
                                "target": target.name,
                                "shift_type": target.shift_type,
                                "covered": target.covered,
                                **accounting,
                                **bound,
                                "bound_ratio": bound["component_bound"]
                                / max(abs(accounting["target_transport"]), 1e-12),
                            }
                        )
                    print(
                        f"{run_id}: source={training['final_source_risk']:.4f} "
                        f"penalty={training['final_penalty']:.4g} "
                        f"semantic={int(decomposition.diagnostics['semantic_identified'])}",
                        flush=True,
                    )

    _write_csv(output_dir / "runs.csv", run_rows)
    _write_csv(output_dir / "semantic_components.csv", semantic_rows)
    _write_csv(output_dir / "risk_components.csv", risk_rows)
    (output_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True))
    environment = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "elapsed_seconds": time.time() - started,
        "target_used_for_training_or_decomposition": False,
    }
    (output_dir / "environment.json").write_text(json.dumps(environment, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    run(json.loads(arguments.config.read_text()), arguments.output)


if __name__ == "__main__":
    main()
