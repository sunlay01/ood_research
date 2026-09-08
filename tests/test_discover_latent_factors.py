import numpy as np
import pandas as pd
import pytest

from ood_repr_reg.discover_latent_factors import (
    DISCOVERY_FEATURES,
    adjusted_rand_index,
    erm_paired_frame,
    pca,
)


def _trajectory() -> pd.DataFrame:
    rows = []
    for seed in range(2):
        for method in ("erm", "mmd"):
            for step in (0, 1):
                row = {
                    "method": method,
                    "lambda": 0.1 if method != "erm" else 0.0,
                    "seed": seed,
                    "step": step,
                    "cross_domain_mse": 1.0 + seed + (0.2 if method != "erm" else 0.0),
                }
                for index, feature in enumerate(DISCOVERY_FEATURES):
                    row[feature] = float(seed + step + index + (0.5 if method != "erm" else 0.0))
                rows.append(row)
    return pd.DataFrame(rows)


def test_erm_pairing_uses_final_checkpoint_and_excludes_erm() -> None:
    paired = erm_paired_frame(_trajectory())

    assert len(paired) == 2
    assert set(paired["method"]) == {"mmd"}
    assert set(paired["step"]) == {1}
    assert paired["delta_source_mse"].tolist() == [0.5, 0.5]
    assert paired["target_signed_gap_vs_erm"].tolist() == pytest.approx([0.2, 0.2])


def test_pca_drops_constant_columns_without_misaligning_loadings() -> None:
    matrix = np.array(
        [
            [0.0, 1.0, 4.0],
            [1.0, 1.0, 3.0],
            [2.0, 1.0, 2.0],
            [3.0, 1.0, 1.0],
        ]
    )
    result = pca(matrix, max_components=2)

    assert result["keep"].tolist() == [True, False, True]
    assert result["components"].shape == (2, 2)
    assert result["scores"].shape == (4, 2)


def test_adjusted_rand_index_ignores_cluster_label_names() -> None:
    first = np.array([1, 1, 2, 2, 3, 3])
    second = np.array([7, 7, 4, 4, 9, 9])

    assert adjusted_rand_index(first, second) == 1.0
