import torch

from ood_repr_reg.task3_aopi_multimethod_mechanism_survey.algorithms.stable_rank_norm import stable_rank_project_weight
from ood_repr_reg.task3_aopi_multimethod_mechanism_survey.algorithms.stable_rank import stable_rank


def test_stable_rank_norm_projection_sets_top_norm_and_reduces_tail_rank():
    weight = torch.eye(10)
    projected = stable_rank_project_weight(weight, target_rank=4.0, spectral_norm_target=1.0)
    singular = torch.linalg.svdvals(projected)
    assert float(singular[0]) == torch.tensor(1.0).item()
    assert stable_rank(projected) <= 4.0001
