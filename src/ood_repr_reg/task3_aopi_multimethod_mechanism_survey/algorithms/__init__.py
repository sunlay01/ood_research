"""Per-method algorithm definitions for the CMNIST survey."""

from .coral import CORALAlgorithm
from .erm import ERMAlgorithm
from .feature_nuclear import FeatureNuclearAlgorithm
from .fishr import FishrAlgorithm
from .irmv1 import IRMv1Algorithm
from .mldg import MLDGAlgorithm
from .registry import algorithm_names, get_algorithm
from .vrex import VRExAlgorithm
from .weight_nuclear import WeightNuclearAlgorithm

__all__ = [
    "CORALAlgorithm",
    "ERMAlgorithm",
    "FeatureNuclearAlgorithm",
    "FishrAlgorithm",
    "IRMv1Algorithm",
    "MLDGAlgorithm",
    "VRExAlgorithm",
    "WeightNuclearAlgorithm",
    "algorithm_names",
    "get_algorithm",
]
