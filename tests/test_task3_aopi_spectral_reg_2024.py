import torch
from torch import nn

from ood_repr_reg.task3_aopi_multimethod_mechanism_survey.algorithms.spectral_reg_2024 import spectral_reg_2024_penalty


def test_spectral_reg_2024_minimum_at_sigma_power_one_and_zero_bias():
    model = nn.Sequential(nn.Linear(2, 2))
    with torch.no_grad():
        model[0].weight.copy_(torch.eye(2))
        model[0].bias.zero_()
    assert torch.allclose(spectral_reg_2024_penalty(model, exponent=2), torch.tensor(0.0), atol=1e-7)
    with torch.no_grad():
        model[0].weight.mul_(2.0)
    assert spectral_reg_2024_penalty(model, exponent=2).item() > 0.0
