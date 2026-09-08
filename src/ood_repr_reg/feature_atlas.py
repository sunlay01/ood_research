"""Direct visual inspection of CMNIST latent neurons.

The atlas deliberately does not assign semantic names to neurons. It reports
which fixed probe images maximize each neuron and where the activation is
sensitive in the input image.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from .cmnist_feature_probe import CounterfactualProbe, SmallCMNISTCNN


@dataclass(frozen=True)
class AtlasExample:
    image: Tensor
    activation: float
    digit: int
    color: str
    sample_index: int


def probe_batch(probe: CounterfactualProbe) -> tuple[Tensor, np.ndarray, np.ndarray, np.ndarray]:
    """Return a fixed red/green probe pool and human-readable metadata."""
    images = torch.cat((probe.red, probe.green), dim=0)
    n = int(probe.y.shape[0])
    digits = torch.cat((probe.digit, probe.digit)).numpy()
    labels = torch.cat((probe.y, probe.y)).numpy()
    colors = np.asarray(["red"] * n + ["green"] * n)
    return images, digits, labels, colors


@torch.no_grad()
def activation_table(
    model: SmallCMNISTCNN,
    probe: CounterfactualProbe,
    *,
    device: torch.device,
    top_k: int = 8,
) -> tuple[list[dict[str, object]], list[list[AtlasExample]]]:
    """Compute all-neuron diagnostics and top activation examples."""
    images, digits, labels, colors = probe_batch(probe)
    z = model.encode(images.to(device)).cpu()
    latent_dim = z.shape[1]
    rows: list[dict[str, object]] = []
    examples: list[list[AtlasExample]] = []
    for neuron in range(latent_dim):
        values = z[:, neuron]
        red = values[: len(probe.y)]
        green = values[len(probe.y) :]
        top_indices = torch.argsort(values, descending=True)[:top_k].tolist()
        examples.append(
            [
                AtlasExample(
                    image=images[index].cpu(),
                    activation=float(values[index]),
                    digit=int(digits[index]),
                    color=str(colors[index]),
                    sample_index=int(index),
                )
                for index in top_indices
            ]
        )
        task0 = values[torch.as_tensor(labels == 0)].mean()
        task1 = values[torch.as_tensor(labels == 1)].mean()
        rows.append(
            {
                "neuron": neuron,
                "max_activation": float(values.max()),
                "mean_activation": float(values.mean()),
                "std_activation": float(values.std(unbiased=False)),
                "red_mean": float(red.mean()),
                "green_mean": float(green.mean()),
                "color_difference_green_minus_red": float(green.mean() - red.mean()),
                "color_selectivity": float(
                    (green.mean() - red.mean()).abs() / values.std(unbiased=False).clamp_min(1e-8)
                ),
                "task0_mean": float(task0),
                "task1_mean": float(task1),
                "task_difference": float(task1 - task0),
                "top_color": str(colors[top_indices[0]]),
                "top_digit": int(digits[top_indices[0]]),
            }
        )
    return rows, examples


def _draw_image(axis: plt.Axes, image: Tensor, *, title: str) -> None:
    axis.imshow(image.permute(1, 2, 0).numpy().clip(0.0, 1.0))
    axis.set_title(title, fontsize=7, pad=2)
    axis.set_xticks([])
    axis.set_yticks([])


def plot_top_activations(
    path: Path,
    *,
    model_label: str,
    rows: list[dict[str, object]],
    examples: list[list[AtlasExample]],
    neurons: int = 8,
) -> None:
    """Plot rows of real input images that maximize selected latent neurons."""
    ranked = sorted(range(len(rows)), key=lambda i: float(rows[i]["max_activation"]), reverse=True)
    selected = ranked[:neurons]
    columns = 1 + max(len(examples[i]) for i in selected)
    figure, axes = plt.subplots(
        len(selected), columns, figsize=(1.45 * columns, 1.65 * len(selected)), squeeze=False
    )
    for row_number, neuron in enumerate(selected):
        record = rows[neuron]
        axes[row_number, 0].axis("off")
        axes[row_number, 0].text(
            0.0,
            0.82,
            f"neuron {neuron}\nmax {float(record['max_activation']):.2f}\n"
            f"top {record['top_color']} digit {record['top_digit']}\n"
            f"color sel. {float(record['color_selectivity']):.2f}",
            va="top",
            fontsize=8,
        )
        for column, example in enumerate(examples[neuron], start=1):
            _draw_image(
                axes[row_number, column],
                example.image,
                title=f"{example.color} d{example.digit}\n{example.activation:.2f}",
            )
    figure.suptitle(f"Top latent-neuron activations | {model_label}", fontsize=12)
    figure.text(
        0.5,
        0.01,
        "Rows are selected by largest activation on a fixed red/green probe; no target images are used.",
        ha="center",
        fontsize=8,
    )
    figure.tight_layout(rect=(0, 0.03, 1, 0.96))
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _saliency(model: SmallCMNISTCNN, image: Tensor, neuron: int, device: torch.device) -> Tensor:
    """Absolute input gradient of one latent neuron, normalized per image."""
    x = image[None].to(device).detach().requires_grad_(True)
    model.zero_grad(set_to_none=True)
    activation = model.encode(x)[0, neuron]
    activation.backward()
    gradient = x.grad.detach()[0].abs().sum(dim=0).cpu()
    return gradient / gradient.max().clamp_min(1e-8)


def plot_saliency(
    path: Path,
    *,
    model: SmallCMNISTCNN,
    model_label: str,
    rows: list[dict[str, object]],
    examples: list[list[AtlasExample]],
    device: torch.device,
    neurons: int = 4,
) -> None:
    """Plot top examples with input-gradient overlays."""
    selected_neurons = sorted(
        range(len(rows)), key=lambda i: float(rows[i]["max_activation"]), reverse=True
    )[:neurons]
    selected = [examples[index] for index in selected_neurons]
    columns = 1 + max(len(items) for items in selected)
    figure, axes = plt.subplots(
        len(selected), columns, figsize=(1.45 * columns, 1.65 * len(selected)), squeeze=False
    )
    model.eval()
    for row_number, (neuron, neuron_examples) in enumerate(zip(selected_neurons, selected)):
        axes[row_number, 0].axis("off")
        axes[row_number, 0].text(0.0, 0.78, f"latent neuron {neuron}\ninput gradient", fontsize=8, va="top")
        for column, example in enumerate(neuron_examples, start=1):
            heatmap = _saliency(model, example.image, neuron, device)
            axes[row_number, column].imshow(example.image.permute(1, 2, 0).numpy().clip(0.0, 1.0))
            axes[row_number, column].imshow(heatmap.numpy(), cmap="magma", alpha=0.55, vmin=0, vmax=1)
            axes[row_number, column].set_title(
                f"{example.color} d{example.digit}\nmax {example.activation:.2f}", fontsize=7, pad=2
            )
            axes[row_number, column].set_xticks([])
            axes[row_number, column].set_yticks([])
    figure.suptitle(f"Input sensitivity of top latent neurons | {model_label}", fontsize=12)
    figure.text(
        0.5,
        0.01,
        "Warm regions have larger absolute d(neuron activation)/d(pixel) for the displayed input.",
        ha="center",
        fontsize=8,
    )
    figure.tight_layout(rect=(0, 0.03, 1, 0.96))
    figure.savefig(path, dpi=180)
    plt.close(figure)


def write_activation_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _total_variation(image: Tensor) -> Tensor:
    return (image[:, :, 1:] - image[:, :, :-1]).abs().mean() + (
        image[:, :, :, 1:] - image[:, :, :, :-1]
    ).abs().mean()


def activation_maximization(
    model: SmallCMNISTCNN,
    neuron: int,
    *,
    color: str,
    device: torch.device,
    steps: int = 220,
    restarts: int = 2,
    start_image: Tensor | None = None,
) -> tuple[Tensor, float]:
    """Synthesize a red or green CMNIST input that maximizes one neuron.

    The grayscale shape is optimized while the color channel is fixed. When a
    real starting image is supplied, a data-anchor term keeps the result near
    the observed MNIST manifold; unconstrained pixel optimization is prone to
    high-frequency adversarial patterns.
    """
    if color not in {"red", "green"}:
        raise ValueError("color must be red or green")
    channel = 0 if color == "red" else 1
    parameters = [parameter.requires_grad for parameter in model.parameters()]
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    best_image: Tensor | None = None
    best_value = -float("inf")
    try:
        for restart in range(restarts):
            generator = torch.Generator(device=device).manual_seed(1701 + neuron * 31 + restart)
            if start_image is None:
                # A low-resolution image prior suppresses adversarial pixel-scale
                # patterns while retaining enough capacity for digit strokes.
                initial = (
                    torch.randn((1, 1, 14, 14), generator=generator, device=device) * 0.45 - 2.5
                )
                anchor = None
            else:
                anchor = start_image[channel : channel + 1].to(device=device).unsqueeze(0)
                anchor = F.interpolate(anchor, size=(14, 14), mode="area").clamp(0.01, 0.99)
                initial = torch.logit(anchor) + torch.randn(
                    (1, 1, 14, 14), generator=generator, device=device
                ) * 0.05
            raw = initial.requires_grad_(True)
            optimizer = torch.optim.Adam((raw,), lr=0.08)
            for _ in range(steps):
                gray = F.interpolate(
                    raw.sigmoid(), size=(28, 28), mode="bilinear", align_corners=False
                )
                image = torch.zeros((1, 3, 28, 28), device=device)
                image[:, channel] = gray[:, 0]
                activation = model.encode(image)[0, neuron]
                objective = -activation + 0.001 * gray.square().mean() + 0.02 * _total_variation(gray)
                if anchor is not None:
                    objective = objective + 0.06 * (gray - F.interpolate(
                        anchor, size=(28, 28), mode="bilinear", align_corners=False
                    )).square().mean()
                optimizer.zero_grad(set_to_none=True)
                objective.backward()
                optimizer.step()
            with torch.no_grad():
                gray = F.interpolate(
                    raw.sigmoid(), size=(28, 28), mode="bilinear", align_corners=False
                )
                image = torch.zeros((1, 3, 28, 28), device=device)
                image[:, channel] = gray[:, 0]
                value = float(model.encode(image)[0, neuron])
                if value > best_value:
                    best_value = value
                    best_image = image[0].cpu()
    finally:
        for parameter, requires_grad in zip(model.parameters(), parameters):
            parameter.requires_grad_(requires_grad)
    if best_image is None:
        raise RuntimeError("activation maximization produced no image")
    return best_image, best_value


def plot_activation_maximization(
    path: Path,
    *,
    model: SmallCMNISTCNN,
    model_label: str,
    rows: list[dict[str, object]],
    examples: list[list[AtlasExample]],
    device: torch.device,
    neurons: int = 8,
    steps: int = 220,
) -> None:
    """Show synthesized inputs for the most active latent neurons."""
    selected = sorted(
        range(len(rows)), key=lambda i: float(rows[i]["max_activation"]), reverse=True
    )[:neurons]
    figure, axes = plt.subplots(len(selected), 3, figsize=(4.5, 1.5 * len(selected)), squeeze=False)
    for row_number, neuron in enumerate(selected):
        axes[row_number, 0].axis("off")
        axes[row_number, 0].text(
            0.0,
            0.8,
            f"neuron {neuron}\nprobe max {float(rows[neuron]['max_activation']):.2f}",
            va="top",
            fontsize=8,
        )
        for column, color in enumerate(("red", "green"), start=1):
            color_examples = [example for example in examples[neuron] if example.color == color]
            start_image = color_examples[0].image if color_examples else None
            image, value = activation_maximization(
                model,
                neuron,
                color=color,
                device=device,
                steps=steps,
                start_image=start_image,
            )
            _draw_image(axes[row_number, column], image, title=f"{color} synthetic\nact {value:.2f}")
    figure.suptitle(f"Activation-maximized latent features | {model_label}", fontsize=12)
    figure.text(
        0.5,
        0.01,
        "Each image is optimized to maximize one neuron with color fixed; it is a model-preferred prototype, not a recovered training image.",
        ha="center",
        fontsize=8,
    )
    figure.tight_layout(rect=(0, 0.03, 1, 0.96))
    figure.savefig(path, dpi=180)
    plt.close(figure)
