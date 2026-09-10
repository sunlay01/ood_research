"""Fixed CPU-minimal MLP for ColoredMNIST."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class CPUColoredMNISTMLP(nn.Module):
    """Exactly Linear(392,64), ReLU, Linear(64,64), ReLU, Linear(64,1)."""

    def __init__(self, input_dim: int = 392, hidden_dim: int = 64) -> None:
        super().__init__()
        if input_dim != 392 or hidden_dim != 64:
            raise ValueError("TASK3-CMNIST-CPU-MINIMAL fixes model dimensions to 392->64->64->1")
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.head = nn.Linear(hidden_dim, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def encode(self, x: Tensor) -> Tensor:
        return self.encoder(x.reshape(x.shape[0], -1))

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.encode(x))


def build_model_from_config(config: dict) -> CPUColoredMNISTMLP:
    model_cfg = config["model"]
    if model_cfg != {"input_dim": 392, "hidden_dim": 64, "activation": "relu", "linear_layers": 3}:
        raise ValueError("config model block must match the CPU-minimal fixed MLP")
    return CPUColoredMNISTMLP(input_dim=392, hidden_dim=64)


def linear_layer_count(model: nn.Module) -> int:
    return sum(1 for module in model.modules() if isinstance(module, nn.Linear))


def augmented_head_dimension(model: CPUColoredMNISTMLP) -> int:
    return int(model.head.weight.shape[1] + 1)


def parameter_hash(model: nn.Module) -> str:
    import hashlib

    hasher = hashlib.sha256()
    with torch.no_grad():
        for parameter in model.parameters():
            hasher.update(parameter.detach().cpu().numpy().tobytes())
    return hasher.hexdigest()
