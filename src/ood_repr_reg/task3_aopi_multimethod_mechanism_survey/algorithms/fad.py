"""Deferred Flatness-Aware Minimization for Domain Generalization."""

from __future__ import annotations

from .deferred import DeferredAlgorithm


class FADAlgorithm(DeferredAlgorithm):
    name = "FAD"
    formula_id = "FAD_DEFERRED_REFERENCE_AUDIT_REQUIRED"
    reference_id = "ZHANG_2023_FLATNESS_AWARE_MINIMIZATION_FOR_DOMAIN_GENERALIZATION"
    variant_id = "FAD_DEFERRED_NO_GUESSED_SAM_SURROGATE"
    deferred_reason = "FAD_REFERENCE_UNRESOLVED_EXACT_ZERO_AND_FIRST_ORDER_UPDATE_NOT_IMPLEMENTED"
