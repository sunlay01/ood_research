"""Exact finite-shift and local interaction accounting."""

from __future__ import annotations

import numpy as np

from .round3_mechanisms import StructuralMechanism, interaction_terms, moment_matrix, risk_state, risk


def transport_accounting(
    beta: np.ndarray,
    w_c: np.ndarray,
    w_a: np.ndarray,
    source: StructuralMechanism,
    target: StructuralMechanism,
) -> dict[str, float]:
    delta = moment_matrix(target) - moment_matrix(source)
    terms = interaction_terms(risk_state(beta, w_c, w_a), delta, source.c_dim)
    exact = risk(beta, w_c, w_a, target) - risk(beta, w_c, w_a, source)
    terms["exact_transport"] = float(exact)
    terms["closure_error"] = float(terms["total"] - exact)
    return terms


def mixed_shift_accounting(
    beta: np.ndarray,
    w_c: np.ndarray,
    w_a: np.ndarray,
    source: StructuralMechanism,
    targets: tuple[StructuralMechanism, ...],
) -> dict[str, object]:
    values = [transport_accounting(beta, w_c, w_a, source, target) for target in targets]
    return {"rows": values, "max_closure_error": max(abs(row["closure_error"]) for row in values)}
