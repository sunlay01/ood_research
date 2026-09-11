"""Registry for survey algorithms; no method math lives here."""

from __future__ import annotations

from typing import Any

from .base import SurveyAlgorithm
from .asam import ASAMAlgorithm
from .coral import CORALAlgorithm
from .disam import DISAMAlgorithm
from .erm import ERMAlgorithm
from .fad import FADAlgorithm
from .feature_nuclear import FeatureNuclearAlgorithm
from .fishr import FishrAlgorithm
from .irmv1 import IRMv1Algorithm
from .mldg import MLDGAlgorithm
from .sam import SAMAlgorithm
from .spectral_norm_reg import SpectralNormRegAlgorithm
from .spectral_reg_2024 import SpectralReg2024Algorithm
from .stable_rank_norm import StableRankNormAlgorithm
from .svb_orthdnn import SVBOrthDNNAlgorithm
from .svd_sparse import SVDSparseAlgorithm
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
    "SPECTRAL_NORM_REG": SpectralNormRegAlgorithm,
    "SPECTRAL_REG_2024": SpectralReg2024Algorithm,
    "SVB_ORTHDNN": SVBOrthDNNAlgorithm,
    "STABLE_RANK_NORM": StableRankNormAlgorithm,
    "SVD_SPARSE": SVDSparseAlgorithm,
    "SAM": SAMAlgorithm,
    "ASAM": ASAMAlgorithm,
    "FAD": FADAlgorithm,
    "DISAM": DISAMAlgorithm,
}


def get_algorithm(name: str, config: dict[str, Any]) -> SurveyAlgorithm:
    try:
        return ALGORITHM_CLASSES[name](config)
    except KeyError as exc:
        raise ValueError(f"unsupported survey method: {name}") from exc


def algorithm_names() -> tuple[str, ...]:
    return tuple(ALGORITHM_CLASSES)
