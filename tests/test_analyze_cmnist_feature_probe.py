import numpy as np
import pandas as pd

from ood_repr_reg.analyze_cmnist_feature_probe import aggregate_metrics, paired_metrics


def test_paired_metrics_use_same_seed_erm() -> None:
    rows = []
    for seed, baseline in ((0, 0.5), (1, 0.7)):
        shared = {
            "seed": seed,
            "diagnostic_l2": 2.0,
            "diagnostic_irmv1": 3.0,
            "diagnostic_coral": 4.0,
            "source_accuracy": 0.9,
            "target_accuracy": baseline,
            "balanced_accuracy": 0.8,
            "latent_color_response": 10.0,
            "prediction_color_response": 5.0,
            "probability_color_response": 0.2,
            "task_signal": 2.0,
            "counterfactual_prediction_consistency": 0.6,
        }
        rows.append({"method": "erm", "strength": 0.0, **shared})
        rows.append(
            {
                "method": "coral",
                "strength": 1.0,
                **shared,
                "diagnostic_coral": 2.0,
                "target_accuracy": baseline + 0.1,
                "prediction_color_response": 2.5,
            }
        )
    paired = paired_metrics(pd.DataFrame(rows))
    summary = aggregate_metrics(paired)

    assert np.allclose(paired["target_accuracy_delta"], 0.1)
    assert np.allclose(paired["prediction_color_ratio"], 0.5)
    assert np.allclose(paired["own_penalty_ratio"], 0.5)
    assert np.isclose(summary.loc[0, "target_accuracy_delta_mean"], 0.1)
