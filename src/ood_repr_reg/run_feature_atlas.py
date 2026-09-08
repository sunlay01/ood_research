"""Generate direct latent-neuron visualizations for CMNIST checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .cmnist_feature_probe import (
    SmallCMNISTCNN,
    deterministic_subset,
    load_mnist_tensors,
    make_counterfactual_probe,
)
from .feature_atlas import (
    activation_table,
    plot_activation_maximization,
    plot_saliency,
    plot_top_activations,
    write_activation_csv,
)


def _device(name: str) -> torch.device:
    return torch.device(name)


def run(config_path: Path, checkpoint_dir: Path, output_dir: Path) -> None:
    config = json.loads(config_path.read_text())
    output_dir.mkdir(parents=True, exist_ok=True)
    gray, digit = load_mnist_tensors(
        Path(config["data_root"]), train=False, download=bool(config.get("download", False))
    )
    probe_gray, probe_digit = deterministic_subset(
        gray,
        digit,
        n=int(config["probe_size"]),
        seed=int(config.get("probe_seed", 211)),
        offset=int(config.get("probe_offset", 0)),
    )
    probe = make_counterfactual_probe(probe_gray, probe_digit)
    device = _device(config.get("device", "cpu"))
    atlas_rows: list[dict[str, object]] = []
    for item in config["checkpoints"]:
        label = str(item["label"])
        checkpoint = checkpoint_dir / str(item["file"])
        model = SmallCMNISTCNN(latent_dim=int(config["latent_dim"])).to(device)
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        model.eval()
        rows, examples = activation_table(
            model, probe, device=device, top_k=int(config.get("top_k", 8))
        )
        for row in rows:
            atlas_rows.append({"model": label, **row})
        safe_label = label.lower().replace(" ", "-").replace("/", "-")
        write_activation_csv(output_dir / f"{safe_label}-activation_stats.csv", rows)
        plot_top_activations(
            output_dir / f"{safe_label}-top_activations.png",
            model_label=label,
            rows=rows,
            examples=examples,
            neurons=int(config.get("plot_neurons", 8)),
        )
        plot_saliency(
            output_dir / f"{safe_label}-saliency.png",
            model=model,
            model_label=label,
            rows=rows,
            examples=examples,
            device=device,
            neurons=int(config.get("saliency_neurons", 4)),
        )
        plot_activation_maximization(
            output_dir / f"{safe_label}-activation_maximization.png",
            model=model,
            model_label=label,
            rows=rows,
            examples=examples,
            device=device,
            neurons=int(config.get("synthesis_neurons", 8)),
            steps=int(config.get("synthesis_steps", 220)),
        )
    write_activation_csv(output_dir / "feature_activation_stats.csv", atlas_rows)
    torch.save(
        {"gray": probe_gray, "digit": probe_digit, "y": probe.y},
        output_dir / "probe.pt",
    )
    (output_dir / "README.txt").write_text(
        "Top activation images are selected from a fixed MNIST test probe. "
        "Saliency is the absolute input gradient of the displayed latent neuron. "
        "These are observational feature preferences, not causal semantic labels.\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.config, args.checkpoint_dir, args.output_dir)


if __name__ == "__main__":
    main()
