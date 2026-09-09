"""Metric-aware sharp optimality audits for affine source-adaptive policies."""

from .geometry import affine_policy_audit, metric_whiten, response_parts

__all__ = ["affine_policy_audit", "metric_whiten", "response_parts"]
