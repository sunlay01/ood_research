"""Blind module-discovery entry points for the 3B audit.

The algorithms intentionally return diagnostics rather than semantic labels.
Mechanism metadata is not accepted by any discovery function.
"""

from .round3r_3b_discovery import (
    bootstrap_stability,
    discover_gram_modules,
    discover_point_modules,
    discover_self_expression_modules,
    discover_subspace_modules,
    normalized_responses,
)

__all__ = [
    "bootstrap_stability", "discover_gram_modules", "discover_point_modules",
    "discover_self_expression_modules",
    "discover_subspace_modules", "normalized_responses",
]
