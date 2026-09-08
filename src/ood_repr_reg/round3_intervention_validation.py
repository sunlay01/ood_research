"""Post-hoc mechanism intervention checks and regularizer probes."""

from __future__ import annotations

import numpy as np

from .round3_mechanisms import StructuralMechanism, add_direction, default_mechanism, mechanism_directions, risk
from .round3_tangent import response_profile
from .round3_response_dataset import ResponseDataset


def intervention_validation(dataset: ResponseDataset, base: StructuralMechanism, scale: float = 0.15) -> dict[str, float]:
    directions = mechanism_directions()
    outcomes: dict[str, float] = {}
    for name, direction in directions.items():
        target = add_direction(base, direction, scale)
        values = []
        for record in dataset.records:
            values.append(float(risk(np.array([1.0]), record.w_c, record.w_a, target) - risk(np.array([1.0]), record.w_c, record.w_a, base)))
        outcomes[name] = float(np.mean(np.abs(values)))
    return outcomes


def regularizer_probe(
    records: tuple,
    beta: np.ndarray | None = None,
    mechanism: StructuralMechanism | None = None,
) -> list[dict[str, float | str]]:
    beta = np.array([1.0]) if beta is None else np.asarray(beta, dtype=float)
    mechanism = default_mechanism() if mechanism is None else mechanism
    directions = mechanism_directions()
    result = []
    for record in records:
        norm = float(np.sum(record.w_c * record.w_c) + np.sum(record.w_a * record.w_a))
        profile = response_profile(beta, record.w_c, record.w_a, mechanism, directions)
        row: dict[str, float | str] = {
            "model_id": record.model_id,
            "regularizer": record.regularizer,
            "l2": norm,
            "nuisance_energy": float(np.sum(record.w_a * record.w_a)),
        }
        row.update({f"sensitivity_{name}": abs(value) for name, value in profile.items()})
        result.append(row)
    return result
