"""Recover source-defined semantic response vectors from the CMNIST survey.

This is an isolated analysis runner: it reuses the existing trainer and
continuation implementation and writes a vector archive plus norm summary.
Target metrics are recorded only post-hoc and never affect response choices.
"""
from __future__ import annotations
import copy, csv, hashlib, json, subprocess, time
from pathlib import Path
import numpy as np
import torch

from .task3_aopi_multimethod_mechanism_survey.config_schema import validate_config
from .task3_aopi_multimethod_mechanism_survey.smooth_world5 import build_smooth_world5, all_direction_vectors, SEMANTIC_NAMES
from .task3_aopi_multimethod_mechanism_survey.functional_banks import build_functional_banks, bank_logits
from .task3_aopi_multimethod_mechanism_survey.full_response import full_response_rows, _run_path
from .task3_aopi_multimethod_mechanism_survey.method_trainer import train_survey_method
from .task3_aopi_multimethod_mechanism_survey.algorithms.registry import get_algorithm
from .task3_aopi_multimethod_mechanism_survey.method_trainer import optimizer_state_hash
from .task3_cmnist_cpu_minimal.data import build_task3_data
from .task3_cmnist_cpu_minimal.evaluation import evaluate_checkpoint
from .task3_cmnist_cpu_minimal.model import build_model_from_config

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/task3_aopi_multimethod_mechanism_survey.json"
OUT = ROOT / "round3_redesign/semantic_mechanism_bridge"
METHODS = ("ERM", "IRMv1", "VREX", "FISHR", "CORAL")
SEEDS = (10, 11, 12, 13, 14)

def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for r in rows for k in r})
    with path.open("w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)

def _train_model(config, seed, method, data):
    torch.manual_seed(seed)
    model = build_model_from_config(config)
    initial = hashlib.sha256(torch.cat([p.detach().reshape(-1) for p in model.parameters()]).numpy().tobytes()).hexdigest()
    return train_survey_method(model=model, source_envs=data.source_envs, batch_schedule=data.batch_schedule, method=method, config=config, seed=seed, initial_parameter_hash=initial)

def run():
    raw = json.loads(CONFIG.read_text()); raw["methods"] = list(METHODS); raw["candidate_methods"] = list(METHODS); raw["seeds"] = list(SEEDS)
    config = copy.deepcopy(validate_config(raw))
    horizons = tuple(int(x) for x in config["response"]["horizons"])
    summary_rows, vector_rows, arrays = [], [], {}
    target_rows = []
    for seed in SEEDS:
        data = build_task3_data(config, seed, data_root=ROOT / "data", download=bool(config["execution"]["download_mnist"]))
        world = build_smooth_world5(config, seed, data_root=ROOT / "data", download=False)
        banks = build_functional_banks(world, source_size_per_environment=int(config["banks"]["source_bank_size_per_environment"]), counterfactual_size=int(config["banks"]["counterfactual_bank_size"]))
        for method in METHODS:
            result = _train_model(config, seed, method, data)
            if not result.finite: continue
            metrics = evaluate_checkpoint(result.model, data.source_envs, data.target_env, device="cpu")
            target_rows.append({"seed": seed, "method": method, **metrics, "target_posthoc": True})
            rows = full_response_rows(result, world, banks, config=config, seed=seed, method=method)
            for row in rows:
                row = dict(row)
                direction_index = int(str(row["opaque_direction_id"])[1:])
                row["semantic_direction"] = SEMANTIC_NAMES[direction_index]
                row["source_only"] = row["semantic_direction"] in {"source_env0_color", "source_env1_color", "source_label_noise", "source_color_common", "source_color_contrast"}
                row["target_acc_posthoc"] = float(metrics.get("target_accuracy", float("nan")))
                summary_rows.append(row)
            # Reconstruct exact response vectors using the same common schedule and continuation.
            max_h = max(horizons); gen = torch.Generator().manual_seed(seed + 271828)
            schedule = (torch.randint(len(world.source[0].digits), (max_h, int(config["training"]["batch_size_per_environment"])), generator=gen), torch.randint(len(world.source[1].digits), (max_h, int(config["training"]["batch_size_per_environment"])), generator=gen))
            zero = torch.zeros(5, dtype=torch.double)
            control = _run_path(result, world, banks, zero, method, config, schedule, horizons)
            basis_vectors = {}
            for basis in (0, 1, 3):
                d = torch.zeros(5, dtype=torch.double); d[basis] = 1
                plus = _run_path(result, world, banks, float(config["response"]["delta"]) * d, method, config, schedule, horizons)
                minus = _run_path(result, world, banks, -float(config["response"]["delta"]) * d, method, config, schedule, horizons)
                for h in horizons:
                    for bank in ("source", "counterfactual_red", "counterfactual_green", "clean_task"):
                        vec = ((plus[h][3][bank] - minus[h][3][bank]) / (2 * float(config["response"]["delta"]))).numpy()
                        basis_vectors[(basis, h, bank)] = vec
            # Derived semantic directions are fixed linear combinations of the
            # measured basis responses; archive their full vectors as well.
            for direction_index, direction in enumerate(all_direction_vectors()):
                for h in horizons:
                    for bank in ("source", "counterfactual_red", "counterfactual_green", "clean_task"):
                        vec = sum(float(direction[b]) * basis_vectors[(b, h, bank)] for b in (0, 1, 3))
                        key = f"{method}__{seed}__u{direction_index:03d}__K{h}__{bank}"
                        arrays[key] = vec
                        vector_rows.append({"method": method, "seed": seed, "semantic_direction": SEMANTIC_NAMES[direction_index], "basis_index": direction_index if direction_index < 5 else -1, "K": h, "bank": bank, "array_key": key, "response_norm": float(np.linalg.norm(vec))})
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "semantic_response_vectors.npz", **arrays)
    _write_csv(OUT / "legacy_semantic_response_summary.csv", summary_rows)
    _write_csv(OUT / "semantic_response_vectors_index.csv", vector_rows)
    _write_csv(OUT / "target_metrics_posthoc.csv", target_rows)
    prov = {"methods": METHODS, "seeds": SEEDS, "horizons": horizons, "source_only": True, "target_used_for": "posthoc_metrics_only", "vector_archive": "semantic_response_vectors.npz", "git_commit": subprocess.run(["git","rev-parse","HEAD"], cwd=ROOT, text=True, capture_output=True).stdout.strip()}
    (OUT / "provenance.json").write_text(json.dumps(prov, indent=2) + "\n")
    return {"rows": len(summary_rows), "vectors": len(vector_rows)}

if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
