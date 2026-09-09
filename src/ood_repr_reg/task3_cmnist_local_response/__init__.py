"""End-to-end CMNIST local-response geometry experiment."""

from .curvature import (
    CurvatureDiagnostics,
    MetricBundle,
    damped_metric,
    gradient_disagreement_penalty,
    local_response_penalty,
    matched_random_metric,
)
from .trainer import Task3Config, default_config, run_experiment

__all__ = [
    "CurvatureDiagnostics",
    "MetricBundle",
    "Task3Config",
    "damped_metric",
    "default_config",
    "gradient_disagreement_penalty",
    "local_response_penalty",
    "matched_random_metric",
    "run_experiment",
]
