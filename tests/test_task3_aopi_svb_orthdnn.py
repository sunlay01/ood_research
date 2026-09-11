import torch

from ood_repr_reg.task3_aopi_multimethod_mechanism_survey.algorithms.svb_orthdnn import svb_project_weight


def test_svb_projection_clamps_singular_values_to_band():
    weight = torch.diag(torch.tensor([3.0, 0.1, 1.0]))
    projected = svb_project_weight(weight, factor=0.05)
    singular = torch.linalg.svdvals(projected)
    assert float(singular.max()) <= 1.05 + 1e-6
    assert float(singular.min()) >= 1.0 / 1.05 - 1e-6
