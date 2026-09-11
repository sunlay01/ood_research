"""Registry for survey algorithms; no method math lives here."""

from __future__ import annotations

from typing import Any

from .base import SurveyAlgorithm
from .coral import CORALAlgorithm
from .erm import ERMAlgorithm
from .feature_nuclear import FeatureNuclearAlgorithm
from .fishr import FishrAlgorithm
from .irmv1 import IRMv1Algorithm
from .mldg import MLDGAlgorithm
from .vrex import VRExAlgorithm
from .weight_nuclear import WeightNuclearAlgorithm


ALGORITHM_CLASSES = {
    "ERM": ERMAlgorithm,
    "IRMv1": IRMv1Algorithm,
    "VREX": VRExAlgorithm,
    "CORAL": CORALAlgorithm,
    "FISHR": FishrAlgorithm,
    "MLDG": MLDGAlgorithm,
    "WEIGHT_NUCLEAR": WeightNuclearAlgorithm,
    "FEATURE_NUCLEAR": FeatureNuclearAlgorithm,
}


def get_algorithm(name: str, config: dict[str, Any]) -> SurveyAlgorithm:
    try:
        return ALGORITHM_CLASSES[name](config)
    except KeyError as exc:
        raise ValueError(f"unsupported survey method: {name}") from exc


def algorithm_names() -> tuple[str, ...]:
    return tuple(ALGORITHM_CLASSES)
