"""Counterfactual color probe construction for corrected CPU-minimal CMNIST."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ..task3_cmnist_cpu_minimal.data import ColoredEnvironment, binary_labels_from_digits


@dataclass(frozen=True)
class ColorCounterfactualProbe:
    red: Tensor
    green: Tensor
    grayscale: Tensor
    clean_labels: Tensor
    noisy_labels: Tensor
    digits: Tensor
    source_role: str

    @property
    def n_examples(self) -> int:
        return int(self.red.shape[0])


def _as_two_channel_images(flattened: Tensor) -> Tensor:
    if flattened.ndim != 2 or flattened.shape[1] != 392:
        raise ValueError("CPU-minimal CMNIST inputs must have shape [N, 392]")
    return flattened.detach().cpu().reshape(flattened.shape[0], 2, 14, 14)


def build_counterfactual_probe(target_env: ColoredEnvironment) -> ColorCounterfactualProbe:
    """Build red/green interventions from the target images without using target color."""
    two_channel = _as_two_channel_images(target_env.images)
    grayscale = two_channel.sum(dim=1)
    zeros = torch.zeros_like(grayscale)
    red = torch.stack((grayscale, zeros), dim=1).reshape(grayscale.shape[0], -1).contiguous()
    green = torch.stack((zeros, grayscale), dim=1).reshape(grayscale.shape[0], -1).contiguous()
    clean_labels = binary_labels_from_digits(target_env.digits.detach().cpu()).reshape(-1).float()
    return ColorCounterfactualProbe(
        red=red.float(),
        green=green.float(),
        grayscale=grayscale.float(),
        clean_labels=clean_labels,
        noisy_labels=target_env.labels.detach().cpu().reshape(-1).float().clone(),
        digits=target_env.digits.detach().cpu().reshape(-1).long().clone(),
        source_role=str(target_env.role),
    )


def probe_invariant_checks(probe: ColorCounterfactualProbe, *, atol: float = 0.0) -> dict[str, bool]:
    red = probe.red.reshape(probe.n_examples, 2, 14, 14)
    green = probe.green.reshape(probe.n_examples, 2, 14, 14)
    grayscale = probe.grayscale
    return {
        "red_grayscale_preserved": bool(torch.allclose(red.sum(dim=1), grayscale, atol=atol, rtol=0.0)),
        "green_grayscale_preserved": bool(torch.allclose(green.sum(dim=1), grayscale, atol=atol, rtol=0.0)),
        "red_channel_carries_grayscale": bool(torch.allclose(red[:, 0], grayscale, atol=atol, rtol=0.0)),
        "red_other_channel_zero": bool(torch.allclose(red[:, 1], torch.zeros_like(grayscale), atol=atol, rtol=0.0)),
        "green_channel_carries_grayscale": bool(torch.allclose(green[:, 1], grayscale, atol=atol, rtol=0.0)),
        "green_other_channel_zero": bool(torch.allclose(green[:, 0], torch.zeros_like(grayscale), atol=atol, rtol=0.0)),
        "clean_label_is_digit_less_than_5": bool(torch.equal(probe.clean_labels, (probe.digits < 5).float())),
    }
