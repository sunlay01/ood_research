"""Per-method algorithm definitions for the CMNIST survey."""

from .coral import CORALAlgorithm
from .asam import ASAMAlgorithm
from .disam import DISAMAlgorithm
from .erm import ERMAlgorithm
from .fad import FADAlgorithm
from .feature_nuclear import FeatureNuclearAlgorithm
from .fishr import FishrAlgorithm
from .irmv1 import IRMv1Algorithm
from .mldg import MLDGAlgorithm
from .registry import algorithm_names, get_algorithm
from .sam import SAMAlgorithm
from .spectral_norm_reg import SpectralNormRegAlgorithm
from .spectral_reg_2024 import SpectralReg2024Algorithm
from .stable_rank_norm import StableRankNormAlgorithm
from .svb_orthdnn import SVBOrthDNNAlgorithm
from .svd_sparse import SVDSparseAlgorithm
from .vrex import VRExAlgorithm
from .weight_nuclear import WeightNuclearAlgorithm

__all__ = [
    "CORALAlgorithm",
    "ASAMAlgorithm",
    "DISAMAlgorithm",
    "ERMAlgorithm",
    "FADAlgorithm",
    "FeatureNuclearAlgorithm",
    "FishrAlgorithm",
    "IRMv1Algorithm",
    "MLDGAlgorithm",
    "SAMAlgorithm",
    "SpectralNormRegAlgorithm",
    "SpectralReg2024Algorithm",
    "StableRankNormAlgorithm",
    "SVBOrthDNNAlgorithm",
    "SVDSparseAlgorithm",
    "VRExAlgorithm",
    "WeightNuclearAlgorithm",
    "algorithm_names",
    "get_algorithm",
]
