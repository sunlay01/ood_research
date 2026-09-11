"""Deferred Domain-Inspired SAM method."""

from __future__ import annotations

from .deferred import DeferredAlgorithm


class DISAMAlgorithm(DeferredAlgorithm):
    name = "DISAM"
    formula_id = "DISAM_DEFERRED_REFERENCE_AUDIT_REQUIRED"
    reference_id = "MEDIABRAIN_SJTU_DISAM_DOMAIN_INSPIRED_SAM"
    variant_id = "DISAM_DEFERRED_NO_SAM_PLUS_VREX_SURROGATE"
    deferred_reason = "DISAM_REFERENCE_UNRESOLVED_DOMAIN_IMBALANCE_CALIBRATION_NOT_IMPLEMENTED"
