"""Pinned upstream provenance and native-run helpers."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any


PINNED_UPSTREAMS: dict[str, dict[str, str]] = {
    "facebook_irm": {
        "repo": "facebookresearch/InvariantRiskMinimization",
        "url": "https://github.com/facebookresearch/InvariantRiskMinimization.git",
        "commit": "fc185d0f828a98f57030ba3647efc7394d1be95a",
        "local_dir": "artifacts/baseline_fidelity/upstreams/InvariantRiskMinimization",
        "authority_files": "code/colored_mnist/main.py",
    },
    "domainbed": {
        "repo": "facebookresearch/DomainBed",
        "url": "https://github.com/facebookresearch/DomainBed.git",
        "commit": "b93c22a1cfc3b2428398272c1a116c8de1f4139e",
        "local_dir": "artifacts/baseline_fidelity/upstreams/DomainBed",
        "authority_files": "domainbed/algorithms.py; domainbed/hparams_registry.py; domainbed/datasets.py",
    },
    "fish_author": {
        "repo": "YugeTen/fish",
        "url": "https://github.com/YugeTen/fish.git",
        "commit": "333efa24572d99da0a4107ab9cc4af93a915d2a9",
        "local_dir": "artifacts/baseline_fidelity/upstreams/fish",
        "authority_files": "README.md; src/main.py; src/utils.py",
    },
    "fishr_optional": {
        "repo": "alexrame/fishr",
        "url": "https://github.com/alexrame/fishr.git",
        "commit": "7b8fdf1e0b15226ded9b58efd37698e74e616ab7",
        "local_dir": "artifacts/baseline_fidelity/upstreams/fishr",
        "authority_files": "optional only; not used by this gate",
    },
}


@dataclass(frozen=True)
class NativeRunConfig:
    root: Path
    timeout_seconds: int = 1800
    domainbed_steps: int = 2
    native_irm_steps: int = 501
    native_irm_restarts: int = 3


def git_head(path: Path) -> str | None:
    if not (path / ".git").exists():
        return None
    completed = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def upstream_manifest(root: Path) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for key, meta in PINNED_UPSTREAMS.items():
        local = root / meta["local_dir"]
        head = git_head(local)
        rows[key] = {
            **meta,
            "local_path": str(local),
            "present": local.exists(),
            "checked_out_commit": head,
            "pinned_commit_matches": head == meta["commit"] if head else False,
        }
    rows["fish_author_cmnist_status"] = {
        "used_for_native_cmnist": False,
        "reason": "Pinned YugeTen/fish native benchmark is not Colored MNIST; DomainBed Fish is the CMNIST fallback authority.",
    }
    return rows


def _ensure_domainbed_import_stubs(root: Path) -> Path:
    """Create import-only stubs for optional DomainBed dependencies.

    The ColoredMNIST path does not instantiate WILDS datasets or timm models,
    but the pinned DomainBed modules import them at module import time.
    """
    base = root / "artifacts/baseline_fidelity/domainbed_stubs"
    wilds = base / "wilds/datasets"
    wilds.mkdir(parents=True, exist_ok=True)
    (base / "wilds/__init__.py").write_text("", encoding="utf-8")
    (wilds / "__init__.py").write_text("", encoding="utf-8")
    stub_class = """
class {name}:
    def __init__(self, *args, **kwargs):
        raise RuntimeError('{name} stub is import-only; ColoredMNIST should not instantiate it')
"""
    (wilds / "camelyon17_dataset.py").write_text(stub_class.format(name="Camelyon17Dataset"), encoding="utf-8")
    (wilds / "fmow_dataset.py").write_text(stub_class.format(name="FMoWDataset"), encoding="utf-8")
    (base / "timm.py").write_text(
        "def create_model(*args, **kwargs):\n"
        "    raise RuntimeError('timm stub is import-only; ColoredMNIST should not instantiate it')\n",
        encoding="utf-8",
    )
    return base


def run_facebook_irm_native(config: NativeRunConfig, *, method: str) -> dict[str, Any]:
    script = config.root / PINNED_UPSTREAMS["facebook_irm"]["local_dir"] / "code/colored_mnist/main.py"
    if not script.exists():
        return {"method": method, "status": "missing_upstream_script", "returncode": -1}
    args = [
        "--n_restarts",
        str(config.native_irm_restarts),
        "--steps",
        str(config.native_irm_steps),
    ]
    if method.upper() == "ERM":
        args.extend(["--penalty_anneal_iters", "0", "--penalty_weight", "0"])
    wrapper = """
import runpy, sys, torch
torch.Tensor.cuda = lambda self, *args, **kwargs: self
torch.nn.Module.cuda = lambda self, *args, **kwargs: self
script = sys.argv[1]
sys.argv = [script] + sys.argv[2:]
runpy.run_path(script, run_name='__main__')
"""
    env = os.environ.copy()
    env["HOME"] = str(config.root / "artifacts/baseline_fidelity/home")
    completed = subprocess.run(
        [sys.executable, "-c", wrapper, str(script), *args],
        cwd=config.root,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=config.timeout_seconds,
        check=False,
    )
    stdout = completed.stdout[-12000:]
    test_matches = re.findall(
        r"Final test acc \(mean/std across restarts so far\):\s*\n\s*([0-9.eE+-]+)\s+([0-9.eE+-]+)",
        completed.stdout,
    )
    train_matches = re.findall(
        r"Final train acc \(mean/std across restarts so far\):\s*\n\s*([0-9.eE+-]+)\s+([0-9.eE+-]+)",
        completed.stdout,
    )
    row: dict[str, Any] = {
        "method": "IRMv1" if method.upper() == "IRMV1" else "ERM",
        "status": "completed" if completed.returncode == 0 else "failed",
        "returncode": completed.returncode,
        "native_script": str(script),
        "cpu_cuda_shim": True,
        "home_redirected": env["HOME"],
        "n_restarts": config.native_irm_restarts,
        "steps": config.native_irm_steps,
        "stdout_tail": stdout.replace("\n", "\\n")[-4000:],
    }
    if test_matches:
        row["target_accuracy_mean"] = float(test_matches[-1][0])
        row["target_accuracy_std"] = float(test_matches[-1][1])
    if train_matches:
        row["train_accuracy_mean"] = float(train_matches[-1][0])
        row["train_accuracy_std"] = float(train_matches[-1][1])
    return row


def run_domainbed_native(config: NativeRunConfig, *, method: str, output_dir: Path) -> dict[str, Any]:
    domainbed = config.root / PINNED_UPSTREAMS["domainbed"]["local_dir"]
    if not (domainbed / "domainbed/scripts/train.py").exists():
        return {"method": method, "status": "missing_upstream_script", "returncode": -1}
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = config.root / "artifacts/baseline_fidelity/domainbed_data"
    env = os.environ.copy()
    stubs = _ensure_domainbed_import_stubs(config.root)
    env["PYTHONPATH"] = str(stubs) + os.pathsep + str(domainbed) + os.pathsep + env.get("PYTHONPATH", "")
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "domainbed.scripts.train",
            "--data_dir",
            str(data_dir),
            "--dataset",
            "ColoredMNIST",
            "--algorithm",
            method,
            "--test_envs",
            "2",
            "--steps",
            str(config.domainbed_steps),
            "--checkpoint_freq",
            str(max(1, config.domainbed_steps)),
            "--output_dir",
            str(output_dir),
            "--skip_model_save",
        ],
        cwd=domainbed,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=config.timeout_seconds,
        check=False,
    )
    results_path = output_dir / "results.jsonl"
    last_result: dict[str, Any] = {}
    if results_path.exists():
        lines = [line for line in results_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if lines:
            last_result = json.loads(lines[-1])
    return {
        "method": method,
        "status": "completed" if completed.returncode == 0 else "failed",
        "returncode": completed.returncode,
        "steps": config.domainbed_steps,
        "native_script": str(domainbed / "domainbed/scripts/train.py"),
        "default_hparams_first": True,
        "output_dir": str(output_dir),
        "env2_out_acc": last_result.get("env2_out_acc"),
        "loss": last_result.get("loss"),
        "penalty": last_result.get("penalty"),
        "stdout_tail": completed.stdout.replace("\n", "\\n")[-4000:],
    }
