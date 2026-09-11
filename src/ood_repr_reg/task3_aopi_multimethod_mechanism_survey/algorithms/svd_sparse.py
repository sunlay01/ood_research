"""Deferred SVD sparse training method."""

from __future__ import annotations

from .deferred import DeferredAlgorithm


class SVDSparseAlgorithm(DeferredAlgorithm):
    name = "SVD_SPARSE"
    formula_id = "SVD_SPARSE_DEFERRED"
    reference_id = "LOW_RANK_DNN_SVD_ORTHOGONALITY_AND_SINGULAR_VALUE_SPARSIFICATION"
    variant_id = "SVD_SPARSE_DEFERRED_COMMON_MODEL_COMPARABILITY"
    deferred_reason = "SVD_SPARSE_DEFERRED_TRUE_FACTORISED_PARAMETERIZATION_WOULD_CHANGE_MODEL_COMPARABILITY"
