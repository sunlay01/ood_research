"""Convergence-matched optimizer and dense complexity/capacity audit.

This is a narrow follow-up to the synthetic failure diagnostic.  It does not
assign categorical regimes or test a phase-transition theorem.  Its purpose is
to separate optimization speed from optimizer-dependent representation quality
and to check whether a continuous task-difficulty/capacity sweep has a stable
representation boundary.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .run_failure_regime_synthetic import Net, _fit_head, _orthogonal, _risk, _seed_all


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "round3_redesign/task_failure_regime_synthetic/results"
SEEDS = (10, 11, 12)
CONV_CHECKPOINTS = (0, 100, 250, 500, 1000, 2000, 5000)
CONV_CELLS = (
    ("high", "clean", "narrow"),
    ("high", "clean", "wide"),
    ("high", "noisy", "narrow"),
    ("high", "noisy", "wide"),
)
OPTIMIZERS = (
    ("sgd", "sgd_fast", 0.05, 0.0),
    ("sgd", "sgd_slow", 0.01, 0.0),
    ("sgd", "sgd_momentum", 0.05, 0.9),
    ("adam", "adam_default", 0.003, 0.0),
    ("adam", "adam_slow", 0.001, 0.0),
)
ALPHAS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
DZS = (1, 2, 3, 4, 6, 8, 16)


def _core_alpha(u: torch.Tensor, alpha: float) -> torch.Tensor:
    score = (1.0 - alpha) * u[:, 0] + alpha * u[:, 0] * u[:, 1]
    return (score >= 0).double() * 2.0 - 1.0


def _make_env(seed: int, alpha: float, sigma: float, agreement: float, n: int, offset: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed + offset)
    u = torch.randn(n, 2, generator=g, dtype=torch.double)
    y = _core_alpha(u, alpha)
    if sigma:
        y = torch.where(torch.rand(n, generator=g, dtype=torch.double) < sigma, -y, y)
    s = torch.where(torch.rand(n, generator=g, dtype=torch.double) < agreement, y, -y)
    noise = torch.randn(n, 2, generator=g, dtype=torch.double)
    return torch.cat((u, s[:, None], noise), dim=1), y, u


def _diagnose(model: Net, target: tuple[torch.Tensor, torch.Tensor, torch.Tensor], probe: tuple[torch.Tensor, torch.Tensor, torch.Tensor], steps: int = 120) -> dict[str, float]:
    with torch.no_grad():
        actual = _risk(model(target[0]), target[1])
        z_probe = model.representation(probe[0]); z_target = model.representation(target[0])
    head = _fit_head(z_probe.detach(), probe[1], nonlinear=False, seed=9181, steps=steps)
    nprobe = _fit_head(z_probe.detach(), probe[1], nonlinear=True, seed=9182, steps=steps)
    core = _fit_head(probe[2], probe[1], nonlinear=True, seed=9183, steps=steps)
    with torch.no_grad():
        r_head = _risk(head(z_target)[:, 0], target[1])
        r_probe = _risk(nprobe(z_target)[:, 0], target[1])
        r_core = _risk(core(target[2])[:, 0], target[1])
    return {"R_actual": actual, "R_head": r_head, "R_probe": r_probe, "R_core": r_core, "G_use": actual - r_head, "G_repr": r_probe - r_core}


def _train_trajectory(seed: int, alpha: float, sigma: float, dz: int, optimizer: str, lr: float, momentum: float, max_steps: int, checkpoints: tuple[int, ...]) -> list[dict[str, object]]:
    _seed_all(seed + 70000 + int(alpha * 1000) + dz)
    mixing = _orthogonal(seed)
    model = Net(mixing, dz)
    rho = (0.95, 0.85)
    src0 = _make_env(seed, alpha, sigma, rho[0], 1024, 30100)
    src1 = _make_env(seed, alpha, sigma, rho[1], 1024, 30200)
    target = _make_env(seed, alpha, sigma, 0.10, 2048, 30300)
    probe = _make_env(seed, alpha, sigma, 0.50, 2048, 30400)
    if optimizer == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    rows: list[dict[str, object]] = []
    for step in range(max_steps + 1):
        if step in checkpoints:
            with torch.no_grad():
                source_risk = float(0.5 * (F.binary_cross_entropy_with_logits(model(src0[0]), (src0[1] > 0).double()) + F.binary_cross_entropy_with_logits(model(src1[0]), (src1[1] > 0).double())).item())
            diag = _diagnose(model, target, probe)
            rows.append({"seed": seed, "alpha": alpha, "sigma_c": sigma, "d_z": dz, "optimizer": optimizer, "optimizer_config": config_name(optimizer, lr, momentum), "lr": lr, "momentum": momentum, "checkpoint": step, "source_risk": source_risk, **diag})
        if step == max_steps:
            break
        opt.zero_grad(set_to_none=True)
        loss = 0.5 * (F.binary_cross_entropy_with_logits(model(src0[0]), (src0[1] > 0).double()) + F.binary_cross_entropy_with_logits(model(src1[0]), (src1[1] > 0).double()))
        loss.backward(); opt.step()
    return rows


def config_name(optimizer: str, lr: float, momentum: float) -> str:
    if optimizer == "adam":
        return "adam_default" if lr == 0.003 else "adam_slow"
    if momentum > 0:
        return "sgd_momentum"
    return "sgd_fast" if lr == 0.05 else "sgd_slow"


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def _matched_pairs(rows: list[dict[str, object]]) -> dict[str, object]:
    """Match SGD and Adam states by source BCE, not by training step."""
    pairs: list[dict[str, float]] = []
    groups = {(r["seed"], r["alpha"], r["sigma_c"], r["d_z"]) for r in rows}
    for group in groups:
        members = [r for r in rows if (r["seed"], r["alpha"], r["sigma_c"], r["d_z"]) == group]
        sgd = [r for r in members if r["optimizer"] == "sgd"]
        adam = [r for r in members if r["optimizer"] == "adam"]
        for left in sgd:
            right = min(adam, key=lambda r: abs(float(r["source_risk"]) - float(left["source_risk"])))
            pairs.append({"source_risk_sgd": float(left["source_risk"]), "source_risk_adam": float(right["source_risk"]), "source_risk_abs_diff": abs(float(left["source_risk"]) - float(right["source_risk"])), "G_repr_sgd": float(left["G_repr"]), "G_repr_adam": float(right["G_repr"]), "G_repr_diff_sgd_minus_adam": float(left["G_repr"]) - float(right["G_repr"]), "G_use_sgd": float(left["G_use"]), "G_use_adam": float(right["G_use"])})
    matched = [p for p in pairs if p["source_risk_abs_diff"] <= 0.01]
    return {"n_pairs": len(pairs), "n_matched_pairs": len(matched), "matched_source_risk_tolerance": 0.01, "mean_abs_source_risk_diff": float(np.mean([p["source_risk_abs_diff"] for p in matched])) if matched else float("nan"), "mean_G_repr_diff_sgd_minus_adam": float(np.mean([p["G_repr_diff_sgd_minus_adam"] for p in matched])) if matched else float("nan"), "mean_abs_G_repr_diff": float(np.mean([abs(p["G_repr_diff_sgd_minus_adam"]) for p in matched])) if matched else float("nan"), "matched_pairs": matched}


def _dense_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    final = [r for r in rows if int(r["checkpoint"]) == 1000]
    means: dict[str, float] = {}
    violations: dict[str, int] = {}
    critical: dict[str, dict[str, int | None]] = {}
    for optimizer in ("sgd", "adam"):
        for alpha in ALPHAS:
            series = []
            for dz in DZS:
                values = [float(r["G_repr"]) for r in final if r["optimizer"] == optimizer and float(r["alpha"]) == alpha and int(r["d_z"]) == dz]
                series.append(float(np.mean(values)) if values else float("nan"))
                means[f"{optimizer}|alpha={alpha:.1f}|dz={dz}"] = series[-1]
            valid = [v for v in series if np.isfinite(v)]
            violations[f"{optimizer}|alpha={alpha:.1f}"] = sum(valid[i + 1] > valid[i] + 0.02 for i in range(len(valid) - 1))
            under = [dz for dz, value in zip(DZS, series) if np.isfinite(value) and value < 0.10]
            critical[f"{optimizer}|alpha={alpha:.1f}"] = {"first_dz_below_0.10": min(under) if under else None}
    return {"n_final": len(final), "mean_G_repr": means, "capacity_monotonicity_violations": violations, "critical_capacity": critical}


def run(smoke: bool = False) -> dict[str, object]:
    started = time.time()
    seeds = (10,) if smoke else SEEDS
    conv_checkpoints = (0, 100, 500) if smoke else CONV_CHECKPOINTS
    conv_rows: list[dict[str, object]] = []
    conv_cells = CONV_CELLS[:1] if smoke else CONV_CELLS
    for rho_label, sigma_label, dz_label in conv_cells:
        del rho_label
        sigma = 0.0 if sigma_label == "clean" else 0.2
        dz = 2 if dz_label == "narrow" else 8
        for seed in seeds:
            optimizer_configs = (OPTIMIZERS[0], OPTIMIZERS[3]) if smoke else OPTIMIZERS
            for optimizer, _, lr, momentum in optimizer_configs:
                conv_rows.extend(_train_trajectory(seed, 1.0, sigma, dz, optimizer, lr, momentum, max(conv_checkpoints), conv_checkpoints))
    dense_rows: list[dict[str, object]] = []
    dense_alphas = ALPHAS[:2] if smoke else ALPHAS
    dense_dzs = DZS[:2] if smoke else DZS
    for seed in seeds:
        for alpha, dz, optimizer in itertools.product(dense_alphas, dense_dzs, ("sgd", "adam")):
            lr = 0.05 if optimizer == "sgd" else 0.003
            dense_rows.extend(_train_trajectory(seed, alpha, 0.0, dz, optimizer, lr, 0.0, 1000, (1000,)))
    rows = conv_rows + dense_rows
    OUT.mkdir(parents=True, exist_ok=True)
    _write_csv(OUT / ("optimizer_capacity_smoke_rows.csv" if smoke else "optimizer_capacity_rows.csv"), rows)
    matched = _matched_pairs(conv_rows)
    dense = _dense_summary(dense_rows)
    summary = {"task_id": "TASK-OPTIMIZER-CAPACITY-AUDIT", "verdict": "SMOKE-ONLY" if smoke else "CONTINUOUS-AXIS-AUDIT", "n_rows": len(rows), "n_convergence_rows": len(conv_rows), "n_dense_rows": len(dense_rows), "matched_optimizer": {key: value for key, value in matched.items() if key != "matched_pairs"}, "dense": dense, "elapsed_seconds": time.time() - started, "interpretation": "No categorical regime or phase-boundary claim; matched source loss tests speed confounding, and dense sweep tests boundary stability."}
    (OUT / ("optimizer_capacity_smoke_summary.json" if smoke else "optimizer_capacity_summary.json")).write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=True) + "\n", encoding="utf-8")
    report = ["# Optimizer/capacity audit", "", f"- Verdict: **{summary['verdict']}**", f"- Convergence rows: {len(conv_rows)}; dense rows: {len(dense_rows)}", f"- Matched source-risk pairs: {matched['n_matched_pairs']}", f"- Mean matched `G_repr` difference (SGD - Adam): `{matched['mean_G_repr_diff_sgd_minus_adam']}`", f"- Mean absolute matched `G_repr` difference: `{matched['mean_abs_G_repr_diff']}`", f"- Dense first-capacity-below-0.10 map: `{dense['critical_capacity']}`", "", "This report tests optimization-speed and capacity confounds. It does not validate a phase-boundary theorem or a categorical failure taxonomy."]
    (OUT / ("optimizer_capacity_smoke_report.md" if smoke else "optimizer_capacity_report.md")).write_text("\n".join(report) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--smoke", action="store_true"); args = parser.parse_args(); print(json.dumps(run(args.smoke), indent=2, allow_nan=True))
