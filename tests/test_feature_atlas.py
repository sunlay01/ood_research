import numpy as np
import torch

from ood_repr_reg.cmnist_feature_probe import SmallCMNISTCNN, make_counterfactual_probe
from ood_repr_reg.feature_atlas import activation_maximization, activation_table


def test_activation_table_reports_top_images_and_color_statistics() -> None:
    gray = torch.rand(12, 1, 28, 28)
    digit = torch.arange(12) % 10
    probe = make_counterfactual_probe(gray, digit)
    model = SmallCMNISTCNN(latent_dim=4)
    rows, examples = activation_table(model, probe, device=torch.device("cpu"), top_k=3)

    assert len(rows) == 4
    assert all(len(items) == 3 for items in examples)
    assert all(row["top_color"] in {"red", "green"} for row in rows)
    assert all(np.isfinite(float(row["color_selectivity"])) for row in rows)


def test_activation_maximization_returns_colored_input() -> None:
    gray = torch.rand(4, 1, 28, 28)
    digit = torch.arange(4)
    probe = make_counterfactual_probe(gray, digit)
    model = SmallCMNISTCNN(latent_dim=2)
    image, value = activation_maximization(
        model, 0, color="green", device=torch.device("cpu"), steps=2, restarts=1
    )
    assert image.shape == (3, 28, 28)
    assert np.isfinite(value)
    assert torch.count_nonzero(image[0]) == 0
    assert torch.count_nonzero(image[2]) == 0
