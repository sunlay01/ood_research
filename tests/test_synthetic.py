import torch

from ood_repr_reg.synthetic import make_dataset


def test_dataset_is_reproducible_and_source_standardized() -> None:
    first = make_dataset(seed=7, n_train=64, n_eval=32, n_probe=24)
    second = make_dataset(seed=7, n_train=64, n_eval=32, n_probe=24)
    first_x = torch.cat([batch.x for batch in first.train])
    second_x = torch.cat([batch.x for batch in second.train])

    assert torch.equal(first_x, second_x)
    assert torch.allclose(first_x.mean(0), torch.zeros(5), atol=1e-5)
    assert torch.allclose(first_x.std(0), torch.ones(5), atol=1e-5)


def test_target_domain_reverses_spurious_correlation() -> None:
    bundle = make_dataset(seed=3, n_train=256, n_eval=1024, n_probe=64)
    source = torch.cat([batch.x[:, 2] * batch.y for batch in bundle.source_eval]).mean()
    target = torch.cat([batch.x[:, 2] * batch.y for batch in bundle.target_domain_eval]).mean()

    assert source > 0.3
    assert target < -0.3
