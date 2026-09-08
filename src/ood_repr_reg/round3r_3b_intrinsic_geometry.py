"""Public names for the intrinsic 3B response geometry."""

from .round3r_3b_geometry import (
    GeometryTolerance,
    filter_relevant_shifts,
    intersection_dimension,
    matrix_rank,
    mixed_shift_decomposition,
    module_subspaces,
    principal_angles,
    relevance_gram,
    subspace_basis,
    transformed_geometry,
    whitened_responses,
)

__all__ = [
    "GeometryTolerance", "filter_relevant_shifts", "intersection_dimension",
    "matrix_rank", "mixed_shift_decomposition", "module_subspaces",
    "principal_angles", "relevance_gram", "subspace_basis",
    "transformed_geometry", "whitened_responses",
]
