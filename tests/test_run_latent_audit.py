import json

import pandas as pd

import ood_repr_reg.run_latent_audit as runner


def test_runner_keeps_source_training_and_semantics_separate_from_targets(
    tmp_path, monkeypatch
):
    train_calls = []
    semantic_calls = []
    target_calls = []
    original_train = runner.train_model
    original_fit = runner.fit_semantic_decomposition
    original_accounting = runner.component_risk_accounting

    def checked_train(method, batches, **kwargs):
        assert all(batch.environment.shift_type == "source" for batch in batches)
        train_calls.append((method, id(batches), kwargs["seed"], kwargs["latent_dim"]))
        return original_train(method, batches, **kwargs)

    def checked_fit(model, scm, estimator, validation, **kwargs):
        assert all(batch.environment.shift_type == "source" for batch in estimator)
        assert all(batch.environment.shift_type == "source" for batch in validation)
        semantic_calls.append((id(estimator), id(validation)))
        return original_fit(model, scm, estimator, validation, **kwargs)

    def checked_accounting(model, scm, sources, target, decomposition):
        assert all(environment.shift_type == "source" for environment in sources)
        assert target.shift_type != "source"
        target_calls.append(target.name)
        return original_accounting(model, scm, sources, target, decomposition)

    monkeypatch.setattr(runner, "train_model", checked_train)
    monkeypatch.setattr(runner, "fit_semantic_decomposition", checked_fit)
    monkeypatch.setattr(runner, "component_risk_accounting", checked_accounting)
    config = {
        "methods": ["erm", "l2"],
        "seeds": [4],
        "latent_dims": [4],
        "strengths": [0.03],
        "n_train": 24,
        "n_semantic": 48,
        "steps": 2,
        "learning_rate": 0.01,
        "intervention_budget": {
            "relation_budget": 2.0,
            "mean_budget": 1.6,
            "covariance_budget": 1.8,
        },
    }
    runner.run(config, tmp_path)

    assert len(train_calls) == 2
    assert {call[0] for call in train_calls} == {"erm", "l2"}
    assert len({call[1] for call in train_calls}) == 1
    assert {(call[2], call[3]) for call in train_calls} == {(4, 4)}
    assert len(semantic_calls) == 2
    assert len(target_calls) == 14
    runs = pd.read_csv(tmp_path / "runs.csv")
    assert set(runs["method"]) == {"erm", "l2"}
    assert set(runs["seed"]) == {4}
    assert set(runs["latent_dim"]) == {4}
    environment = json.loads((tmp_path / "environment.json").read_text())
    assert environment["target_used_for_training_or_decomposition"] is False
