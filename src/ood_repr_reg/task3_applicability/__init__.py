"""Task 3 applicability track.

This package tests whether frozen Round-3 source/mechanism/family quantities
discriminate finite held-out target behavior.  Feature construction is kept
separate from held-out outcome generation so the preregistration barrier and
leakage audit are explicit.
"""

from .features import FEATURE_SETS, build_feature_dictionary, build_feature_rows
from .targets import build_design, generate_target_outcomes, materialize_training_runs
from .evaluation import evaluate_applicability, logic_audit

__all__ = [
    "FEATURE_SETS",
    "build_design",
    "build_feature_dictionary",
    "build_feature_rows",
    "evaluate_applicability",
    "generate_target_outcomes",
    "logic_audit",
    "materialize_training_runs",
]
