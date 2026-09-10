"""First-round OOD capability decomposition audit for corrected CMNIST."""

from .analysis import ALLOWED_DOMINANT_BOTTLENECKS, ALLOWED_VERDICTS, summarize_capabilities
from .artifacts import load_verified_checkpoint_bundles
from .experiments import run_capability_experiments
from .linear import fit_ridge_classifier, projection_matrix

__all__ = [
    "ALLOWED_DOMINANT_BOTTLENECKS",
    "ALLOWED_VERDICTS",
    "fit_ridge_classifier",
    "load_verified_checkpoint_bundles",
    "projection_matrix",
    "run_capability_experiments",
    "summarize_capabilities",
]
