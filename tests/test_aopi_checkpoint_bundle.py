import copy
import json
from pathlib import Path

import torch

from ood_repr_reg.task3_aopi_multimethod_mechanism_survey.method_trainer import (
    train_survey_method,
)
from ood_repr_reg.task3_cmnist_cpu_minimal.data import (
    BatchSchedule,
    ColoredEnvironment,
)
from ood_repr_reg.task3_cmnist_cpu_minimal.model import CPUColoredMNISTMLP


def test_logical_checkpoint_bundle_is_complete_and_off_by_one_free():
    config = json.loads(
        Path("configs/task3_aopi_multimethod_mechanism_survey.json").read_text()
    )
    config = copy.deepcopy(config)
    config["training"].update(
        steps=2,
        batch_size_per_environment=8,
        checkpoint_steps=[0, 1],
    )
    generator = torch.Generator().manual_seed(7)
    environments = tuple(
        ColoredEnvironment(
            torch.rand((8, 392), generator=generator),
            torch.randint(0, 2, (8, 1), generator=generator).float(),
            torch.arange(8),
            torch.randint(0, 2, (8,), generator=generator).float(),
            probability,
            name,
        )
        for probability, name in ((0.2, "source_env0"), (0.1, "source_env1"))
    )
    schedule = BatchSchedule(
        (
            torch.zeros((2, 8), dtype=torch.long),
            torch.zeros((2, 8), dtype=torch.long),
        ),
        7,
        2,
        8,
    )
    result = train_survey_method(
        model=CPUColoredMNISTMLP(),
        source_envs=environments,
        batch_schedule=schedule,
        method="ERM",
        config=config,
        seed=7,
        initial_parameter_hash="",
    )

    assert result.finite
    assert sorted(result.checkpoint_state_dicts) == [0, 1]
    assert sorted(result.checkpoint_bundles) == [0, 2]
    assert result.checkpoint_bundles[0]["continuation_step"] == 0
    assert result.checkpoint_bundles[2]["continuation_step"] == 2
    assert result.checkpoint_bundles[2]["training_loop_step"] == 1
    assert result.checkpoint_bundles[2]["checkpoint_step"] == 2
    for key, value in result.checkpoint_state_dicts[1].items():
        assert torch.equal(value, result.checkpoint_bundles[2]["model_state"][key])
    assert result.checkpoint_bundles[2]["model_parameter_hash"]
    assert result.checkpoint_bundles[2]["optimizer_state_hash"]
    assert result.checkpoint_bundles[2]["algorithm_state_hash"]
