"""Counterfactual diagnostics for corrected CPU-minimal ColoredMNIST models."""

from .analysis import build_paired_effects, summarize_audit
from .diagnostics import compute_counterfactual_diagnostics, model_counterfactual_diagnostics
from .probe import ColorCounterfactualProbe, build_counterfactual_probe

__all__ = [
    "ColorCounterfactualProbe",
    "build_counterfactual_probe",
    "compute_counterfactual_diagnostics",
    "model_counterfactual_diagnostics",
    "build_paired_effects",
    "summarize_audit",
]
