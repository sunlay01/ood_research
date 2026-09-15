"""Controlled synthetic audit of OOD failure regimes.

This is a mechanism diagnostic, not an OOD algorithm and not a theory test.
It uses only source data for training and held-out target data for reporting.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import platform
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "round3_redesign/task_failure_regime_synthetic/results"
FACTORS = {
    "rho": {"low": (0.65, 0.55), "high": (0.95, 0.85)},
    "k": {"linear": 1, "nonlinear": 2},
    "sigma_c": {"clean": 0.0, "noisy": 0.2},
    "d_z": {"narrow": 2, "wide": 8},
    "optimizer": {"sgd": "sgd", "adam": "adam"},
}
SEEDS = (10, 11, 12, 13, 14)
CHECKPOINTS = (0, 50, 100, 200, 500)


def _seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)


def _orthogonal(seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed + 88001)
    q, r = torch.linalg.qr(torch.randn(5, 5, generator=g, dtype=torch.double))
    signs = torch.sign(torch.diag(r)); signs[signs == 0] = 1
    return q @ torch.diag(signs)


def _core(u: torch.Tensor, k: int) -> torch.Tensor:
    score = u[:, 0] if k == 1 else u[:, 0] * u[:, 1]
    return (score >= 0).double() * 2.0 - 1.0


def _make_env(seed: int, k: int, sigma: float, agreement: float, n: int, offset: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed + offset)
    u = torch.randn(n, 2, generator=g, dtype=torch.double)
    y = _core(u, k)
    if sigma:
        y = torch.where(torch.rand(n, generator=g, dtype=torch.double) < sigma, -y, y)
    s = torch.where(torch.rand(n, generator=g, dtype=torch.double) < agreement, y, -y)
    noise = torch.randn(n, 2, generator=g, dtype=torch.double)
    return torch.cat((u, s[:, None], noise), dim=1), y, u


class Net(nn.Module):
    def __init__(self, mixing: torch.Tensor, d_z: int) -> None:
        super().__init__()
        self.register_buffer("mixing", mixing)
        self.feature = nn.Sequential(nn.Linear(5, 32), nn.ReLU(), nn.Linear(32, d_z), nn.ReLU())
        self.head = nn.Linear(d_z, 1)
        self.double()

    def representation(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature(x @ self.mixing.T)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.representation(x))[:, 0]


def _risk(logits: torch.Tensor, y: torch.Tensor) -> float:
    return float(F.binary_cross_entropy_with_logits(logits, (y > 0).double()).item())


def _fit_head(z: torch.Tensor, y: torch.Tensor, *, nonlinear: bool, seed: int, steps: int = 250) -> nn.Module:
    _seed_all(seed + (991 if nonlinear else 773))
    if nonlinear:
        probe: nn.Module = nn.Sequential(nn.Linear(z.shape[1], 32), nn.ReLU(), nn.Linear(32, 1))
    else:
        probe = nn.Linear(z.shape[1], 1)
    probe = probe.double()
    opt = torch.optim.Adam(probe.parameters(), lr=0.01)
    target = (y > 0).double()
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = F.binary_cross_entropy_with_logits(probe(z)[:, 0], target)
        loss.backward(); opt.step()
    return probe


def _fit_full(seed: int, k: int, sigma: float, rho: tuple[float, float], dz: int, optimizer: str, mixing: torch.Tensor, *, balanced: bool, steps: int = 300) -> Net:
    _seed_all(seed + 5000 + (17 if balanced else 0))
    model = Net(mixing, dz)
    opt = torch.optim.Adam(model.parameters(), lr=0.003) if optimizer == "adam" else torch.optim.SGD(model.parameters(), lr=0.05)
    for step in range(steps):
        if balanced:
            a0, y0, _ = _make_env(seed + step, k, sigma, 0.5, 128, 61000)
            a1, y1, _ = _make_env(seed + step, k, sigma, 0.5, 128, 62000)
        else:
            a0, y0, _ = _make_env(seed + step, k, sigma, rho[0], 128, 61000)
            a1, y1, _ = _make_env(seed + step, k, sigma, rho[1], 128, 62000)
        opt.zero_grad(set_to_none=True)
        loss = 0.5 * (F.binary_cross_entropy_with_logits(model(a0), (y0 > 0).double()) + F.binary_cross_entropy_with_logits(model(a1), (y1 > 0).double()))
        loss.backward(); opt.step()
    return model


def _counterfactual_x(pair: tuple[torch.Tensor, torch.Tensor, torch.Tensor], seed: int, agreement: float = 0.10) -> torch.Tensor:
    """Keep u fixed while independently resampling s and nuisance n."""
    _, y, u = pair
    g = torch.Generator().manual_seed(seed + 77001)
    s = torch.where(torch.rand(len(y), generator=g, dtype=torch.double) < agreement, y, -y)
    noise = torch.randn(len(y), 2, generator=g, dtype=torch.double)
    return torch.cat((u, s[:, None], noise), dim=1)


def _mean_sd(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    return float(np.mean(values)), float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def _run_one(seed: int, labels: dict[str, str], *, steps: int = 500) -> list[dict[str, object]]:
    rho = FACTORS["rho"][labels["rho"]]; k = FACTORS["k"][labels["k"]]; sigma = FACTORS["sigma_c"][labels["sigma_c"]]; dz = FACTORS["d_z"][labels["d_z"]]; optimizer = labels["optimizer"]
    _seed_all(seed)
    mixing = _orthogonal(seed)
    model = Net(mixing, dz)
    opt = torch.optim.Adam(model.parameters(), lr=0.003) if optimizer == "adam" else torch.optim.SGD(model.parameters(), lr=0.05)
    target = _make_env(seed, k, sigma, 0.10, 4096, 10300)
    probe = _make_env(seed, k, sigma, 0.50, 4096, 10400)
    repair = _make_env(seed, k, sigma, 0.50, 4096, 10600)
    pair_a = _make_env(seed, k, sigma, 0.50, 2048, 10500)
    pair_b_x = _counterfactual_x(pair_a, seed)
    rows: list[dict[str, object]] = []
    for step in range(steps + 1):
        if step in CHECKPOINTS:
            # Probe fitting must retain autograd; only feature extraction and
            # evaluation are inference-only.  Wrapping _fit_head in no_grad
            # silently disables loss.backward() and makes the audit unrunnable.
            with torch.no_grad():
                actual = _risk(model(target[0]), target[1])
                z_probe = model.representation(probe[0]); z_target = model.representation(target[0])
                z_repair = model.representation(repair[0])
            head = _fit_head(z_probe.detach(), probe[1], nonlinear=False, seed=seed + step)
            nprobe = _fit_head(z_probe.detach(), probe[1], nonlinear=True, seed=seed + step)
            core = _fit_head(probe[2], probe[1], nonlinear=True, seed=seed + step + 100)
            repair_head = _fit_head(z_repair.detach(), repair[1], nonlinear=False, seed=seed + step + 50000)
            with torch.no_grad():
                r_head = _risk(head(z_target)[:, 0], target[1]); r_probe = _risk(nprobe(z_target)[:, 0], target[1]); r_core = _risk(core(target[2])[:, 0], target[1])
                r_head_repair = _risk(repair_head(z_target)[:, 0], target[1])
                z_a = model.representation(pair_a[0]); z_b = model.representation(pair_b_x)
                c_pred = float(((nprobe(z_a) - nprobe(z_b)) ** 2).mean().item())
                c_linear = float(((head(z_a) - head(z_b)) ** 2).mean().item())
                c_prob = float((torch.sigmoid(nprobe(z_a)) - torch.sigmoid(nprobe(z_b))).pow(2).mean().item())
                c_linear_prob = float((torch.sigmoid(head(z_a)) - torch.sigmoid(head(z_b))).pow(2).mean().item())
                c_actual = float((model(pair_a[0]) - model(pair_b_x)).pow(2).mean().item())
                head_gain = actual - r_head
                repair_gain = actual - r_head_repair
            # Full retraining is another optimization run, so it must also stay
            # outside no_grad.  Its target evaluation is inference-only.
            full_model = _fit_full(seed + step, k, sigma, rho, dz, optimizer, mixing, balanced=True, steps=160)
            with torch.no_grad():
                full_risk = _risk(full_model(target[0]), target[1])
                full_c_actual = float((full_model(pair_a[0]) - full_model(pair_b_x)).pow(2).mean().item())
            rows.append({**labels, "seed": seed, "checkpoint": step, "R_actual": actual, "R_head": r_head, "R_head_repair": r_head_repair, "R_probe": r_probe, "R_core": r_core, "G_use": head_gain, "G_repr": r_probe - r_core, "C_pred": c_pred, "C_linear": c_linear, "C_prob": c_prob, "C_linear_prob": c_linear_prob, "C_actual": c_actual, "C_actual_after_balanced": full_c_actual, "head_repair_gain": repair_gain, "full_retrain_gain": actual - full_risk, "contamination_repair": c_actual - full_c_actual, "R_full_retrain": full_risk, "d_z_value": dz, "k_value": k, "sigma_c_value": sigma, "rho0": rho[0], "rho1": rho[1], "optimizer": optimizer})
        if step == steps: break
        batches = (_make_env(seed + step, k, sigma, rho[0], 128, 20100), _make_env(seed + step, k, sigma, rho[1], 128, 20200))
        opt.zero_grad(set_to_none=True)
        loss = 0.5 * (F.binary_cross_entropy_with_logits(model(batches[0][0]), (batches[0][1] > 0).double()) + F.binary_cross_entropy_with_logits(model(batches[1][0]), (batches[1][1] > 0).double()))
        loss.backward(); opt.step()
    return rows


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for row in rows for k in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


def _summary(rows: list[dict[str, object]], smoke: bool) -> dict[str, object]:
    final = [r for r in rows if int(r["checkpoint"]) == 500]
    keys = ("rho", "k", "sigma_c", "d_z", "optimizer")

    def cell_key(r: dict[str, object]) -> tuple[str, ...]:
        return tuple(str(r[k]) for k in keys)

    def aggregate(items: list[dict[str, object]]) -> dict[str, object]:
        out: dict[str, object] = {k: items[0][k] for k in keys} if items else {}
        out["n"] = len(items)
        for metric in ("G_use", "G_repr", "C_pred", "C_linear", "C_prob", "C_linear_prob", "C_actual", "contamination_repair", "head_repair_gain", "full_retrain_gain"):
            mean, sd = _mean_sd([float(r[metric]) for r in items])
            out[metric] = mean; out[f"{metric}_sd"] = sd
        return out

    cell_rows = [aggregate([r for r in final if cell_key(r) == key]) for key in sorted({cell_key(r) for r in final})]
    checkpoint_rows = []
    for checkpoint in CHECKPOINTS:
        at_checkpoint = [r for r in rows if int(r["checkpoint"]) == checkpoint]
        for key in sorted({cell_key(r) for r in at_checkpoint}):
            grouped = aggregate([r for r in at_checkpoint if cell_key(r) == key]); grouped["checkpoint"] = checkpoint; checkpoint_rows.append(grouped)

    metrics = ("G_use", "G_repr", "C_prob", "head_repair_gain", "full_retrain_gain", "contamination_repair")
    means = {metric: float(np.mean([float(r[metric]) for r in final])) if final else float("nan") for metric in metrics}

    def contrast_stats(factor_a: str, factor_b: str, metric: str) -> dict[str, float]:
        values = []
        levels_a = sorted({str(r[factor_a]) for r in final}); levels_b = sorted({str(r[factor_b]) for r in final})
        for seed in sorted({str(r["seed"]) for r in final}):
            sample = [r for r in final if str(r["seed"]) == seed]
            grouped = {(a, b): [float(r[metric]) for r in sample if str(r[factor_a]) == a and str(r[factor_b]) == b] for a in levels_a for b in levels_b}
            # A 2x2 interaction contrast, averaging over all other factors.
            a0, a1 = levels_a[0], levels_a[-1]; b0, b1 = levels_b[0], levels_b[-1]
            values.append(np.mean(grouped[(a1, b1)]) - np.mean(grouped[(a1, b0)]) - np.mean(grouped[(a0, b1)]) + np.mean(grouped[(a0, b0)]))
        mean, sd = _mean_sd([float(v) for v in values]); se = sd / math.sqrt(len(values)) if len(values) > 1 else float("nan")
        return {"mean": mean, "sd": sd, "z": mean / se if se and np.isfinite(se) else float("nan"), "n_seeds": len(values)}

    contrasts = {
        "rho_x_sigma_to_G_use": contrast_stats("rho", "sigma_c", "G_use"),
        "k_x_optimizer_to_G_repr": contrast_stats("k", "optimizer", "G_repr"),
        "k_x_dz_to_G_repr": contrast_stats("k", "d_z", "G_repr"),
    }
    incremental_r2 = float("nan")
    if len(final) >= 4:
        design = np.c_[np.ones(len(final)), np.asarray([[float(r["G_use"]), float(r["G_repr"]) ] for r in final])]
        response = np.asarray([float(r["C_prob"]) for r in final]); fitted = design @ np.linalg.lstsq(design, response, rcond=None)[0]
        total = float(((response - response.mean()) ** 2).sum()); incremental_r2 = 1.0 - float(((response - fitted) ** 2).sum()) / total if total > 1e-12 else 1.0
    repair_corr = float(np.corrcoef([float(r["G_use"]) for r in final], [float(r["head_repair_gain"]) for r in final])[0, 1]) if len(final) > 2 else float("nan")
    verdict = "SMOKE-ONLY" if smoke else ("TWO-AXIS-STRUCTURE-SUPPORTED; CONTAMINATION-UNRESOLVED" if final and abs(contrasts["rho_x_sigma_to_G_use"]["z"]) >= 2 and abs(contrasts["k_x_optimizer_to_G_repr"]["z"]) >= 2 else "TWO-AXIS-STRUCTURE-INCONCLUSIVE")
    return {"task_id": "TASK-FAILURE-REGIME-SYNTHETIC", "verdict": verdict, "n_rows": len(rows), "n_final": len(final), "n_cells": len(cell_rows), "means": means, "factorial_contrasts": contrasts, "contamination_prob_incremental_r2": incremental_r2, "independent_head_repair_corr": repair_corr, "cell_rows": cell_rows, "checkpoint_rows": checkpoint_rows, "interpretation": "continuous diagnostic evidence only; categorical regime claims and contamination causality are unresolved", "runtime": platform.platform()}


def _persist_results(rows: list[dict[str, object]], smoke: bool, elapsed: float) -> dict[str, object]:
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    _write_csv(OUT / ("smoke_rows.csv" if smoke else "factorial_rows.csv"), rows)
    summary = _summary(rows, smoke); summary["elapsed_seconds"] = elapsed
    _write_csv(OUT / ("smoke_summary.csv" if smoke else "factorial_summary.csv"), list(summary.get("cell_rows", [])))
    _write_csv(OUT / ("smoke_checkpoint_summary.csv" if smoke else "checkpoint_summary.csv"), list(summary.get("checkpoint_rows", [])))
    verdict = str(summary["verdict"])
    report = [
        "# Failure-regime synthetic audit: final report", "", f"- Verdict: **{verdict}**",
        f"- Rows: {summary['n_rows']}; final-checkpoint rows: {summary['n_final']}; data/optimizer cells: {summary['n_cells']}",
        f"- Mean diagnostic vector `(G_use, G_repr, C_prob)`: `({summary['means']['G_use']:.6g}, {summary['means']['G_repr']:.6g}, {summary['means']['C_prob']:.6g})`",
        f"- Independent head-repair correlation with `G_use`: `{summary['independent_head_repair_corr']:.4f}`",
        f"- Probability-sensitivity incremental R2 from `(G_use, G_repr)`: `{summary['contamination_prob_incremental_r2']:.4f}`",
        "", "This is a continuous synthetic diagnostic. It does not establish a theorem, a universal taxonomy, or a claim about any specific OOD algorithm.",
        "Categorical regime labels, raw-logit sensitivity, and contamination causality are intentionally not used as acceptance criteria.",
    ]
    (OUT / ("smoke_final_verdict.md" if smoke else "final_verdict.md")).write_text("\n".join(report) + "\n", encoding="utf-8")
    summary.pop("cell_rows", None); summary.pop("checkpoint_rows", None)
    (OUT / ("smoke_summary.json" if smoke else "summary.json")).write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=True) + "\n")
    return summary


def run(smoke: bool = False) -> dict[str, object]:
    started = time.time(); OUT.mkdir(parents=True, exist_ok=True)
    configs = [dict(zip(FACTORS, values)) for values in itertools.product(*[tuple(v) for v in FACTORS.values()])]
    if smoke: configs, seeds = configs[:1], (10,)
    else: seeds = SEEDS
    rows: list[dict[str, object]] = []
    for idx, cfg in enumerate(configs):
        for seed in seeds:
            rows.extend(_run_one(seed, cfg, steps=500))
            print(f"completed config {idx+1}/{len(configs)} seed={seed}", flush=True)
    return _persist_results(rows, smoke, time.time() - started)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--smoke", action="store_true"); ap.add_argument("--analyze-existing", action="store_true"); ap.add_argument("--elapsed-seconds", type=float, default=0.0); args = ap.parse_args()
    if args.analyze_existing:
        raw_path = OUT / ("smoke_rows.csv" if args.smoke else "factorial_rows.csv")
        with raw_path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            for key in ("seed", "checkpoint", "d_z_value", "k_value"):
                row[key] = int(row[key])
            for key in ("sigma_c_value", "rho0", "rho1", "R_actual", "R_head", "R_head_repair", "R_probe", "R_core", "G_use", "G_repr", "C_pred", "C_linear", "C_prob", "C_linear_prob", "C_actual", "C_actual_after_balanced", "head_repair_gain", "full_retrain_gain", "contamination_repair", "R_full_retrain"):
                row[key] = float(row[key])
        print(json.dumps(_persist_results(rows, args.smoke, args.elapsed_seconds), indent=2, allow_nan=True))
    else:
        print(json.dumps(run(args.smoke), indent=2, allow_nan=True))
