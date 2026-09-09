"""Exact source-only quadratic prototype for Task 3R.

The fitted parameter is the forcing block ``C = U V.T`` of a centered
quadratic regularizer.  ``Pi`` is always reconstructed from the head FOC; it
is never exposed as a learner parameter.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..round3r_3b_benchmark import (
    ModuleEnvironment,
    MomentState,
    environment_state,
    mixture_state,
    risk,
    source_design,
    source_optimum,
)

Array = np.ndarray


@dataclass(frozen=True)
class ProbeConfiguration:
    config_id: str
    base: ModuleEnvironment
    sources: tuple[ModuleEnvironment, ...]
    role: str = "independent_primary"


@dataclass(frozen=True)
class Fold:
    index: int
    reference: MomentState
    pseudo_target: MomentState
    reference_vector: Array
    delta: Array
    reference_weights: Array
    required_response: Array


@dataclass(frozen=True)
class FittedRegularizer:
    method: str
    beta: float
    regularizer_strength: float
    C: Array
    contrast_basis: Array
    source_selection_score: float
    mean_pseudo_risk: float
    mean_response_mismatch: float
    optimizer_success: bool
    optimizer_message: str
    response_target_kind: str
    target_information_used: bool = False
    pi_is_free_parameter: bool = False


def encode_state(state: MomentState) -> Array:
    """Encode a task-complete quadratic state without semantic metadata."""
    return np.concatenate((np.asarray(state.second).reshape(-1), state.xy, [state.y2]))


def decode_state(vector: Array, dimension: int) -> MomentState:
    value = np.asarray(vector, dtype=float)
    width = dimension * dimension
    if value.shape != (width + dimension + 1,):
        raise ValueError("state vector has incompatible dimension")
    return MomentState(value[:width].reshape(dimension, dimension), value[width:width + dimension], float(value[-1]))


def _sqrt_and_inverse(matrix: Array) -> tuple[Array, Array]:
    values, vectors = np.linalg.eigh((matrix + matrix.T) / 2.0)
    if values.min(initial=1.0) <= 1e-12:
        raise ValueError("head Hessian must be positive definite")
    root = vectors @ np.diag(np.sqrt(values)) @ vectors.T
    inverse_root = vectors @ np.diag(1.0 / np.sqrt(values)) @ vectors.T
    return root, inverse_root


def _gradient(weights: Array, state: MomentState) -> Array:
    return 2.0 * (state.second @ weights - state.xy)


def make_configurations() -> tuple[ProbeConfiguration, ...]:
    """Return the preregistered source configurations.

    The first two are independent primary configurations.  The third removes
    variation in shortcut 2 and is used only for H4.
    """
    bases = (
        ModuleEnvironment(shortcut_rhos=(0.75, 0.45), shortcut_means=(0.18, -0.12),
                          shortcut_variances=(0.49, 0.64), n_noise=4),
        ModuleEnvironment(shortcut_rhos=(0.62, 0.28), shortcut_means=(0.08, -0.20),
                          shortcut_variances=(0.58, 0.72), n_noise=4),
    )
    primary = tuple(
        ProbeConfiguration(f"gaussian_primary_{index + 1}", base, source_design(base, True))
        for index, base in enumerate(bases)
    )
    base = bases[0]
    reduced = [base]
    for sign in (-1.0, 1.0):
        rhos = list(base.shortcut_rhos)
        rhos[0] += sign * 0.08
        reduced.append(base.updated(shortcut_rhos=tuple(rhos)))
    return primary + (ProbeConfiguration("gaussian_reduced_exposure", base, tuple(reduced), "source_information_ablation"),)


def _contrast_basis(states: tuple[MomentState, ...], tolerance: float = 1e-10) -> Array:
    encoded = np.stack([encode_state(state) for state in states])
    centered = encoded - encoded.mean(axis=0, keepdims=True)
    _, singular, vh = np.linalg.svd(centered, full_matrices=False)
    if not singular.size or singular[0] <= tolerance:
        raise ValueError("source domains do not provide a response contrast")
    rank = int(np.sum(singular > tolerance * singular[0]))
    return vh[:rank].T


def leave_one_out_folds(states: tuple[MomentState, ...]) -> tuple[Fold, ...]:
    if len(states) < 3:
        raise ValueError("at least three source domains are required")
    folds: list[Fold] = []
    for index, pseudo in enumerate(states):
        meta = tuple(state for j, state in enumerate(states) if j != index)
        reference = MomentState(
            np.mean([state.second for state in meta], axis=0),
            np.mean([state.xy for state in meta], axis=0),
            float(np.mean([state.y2 for state in meta])),
        )
        weights = np.linalg.solve(reference.second, reference.xy)
        hessian = 2.0 * reference.second
        root, _ = _sqrt_and_inverse(hessian)
        pseudo_optimum = np.linalg.solve(pseudo.second, pseudo.xy)
        # Estimate the finite required response from source head optima.
        # Frozen A uses the opposite sign to the oracle optimum displacement.
        required = -root @ (pseudo_optimum - weights)
        folds.append(Fold(index, reference, pseudo, encode_state(reference),
                          encode_state(pseudo) - encode_state(reference), weights, required))
    return tuple(folds)


def directional_risk_jacobian(weights: Array, reference: MomentState, delta: Array) -> Array:
    shifted = decode_state(encode_state(reference) + np.asarray(delta), weights.size)
    return _gradient(weights, shifted) - _gradient(weights, reference)


def exact_directional_response(weights: Array, reference: MomentState, delta: Array,
                               C: Array, regularizer_strength: float,
                               K: Array | None = None) -> dict[str, Array | float]:
    """Evaluate ``Pi delta`` from the exact centered quadratic FOC."""
    p = weights.size
    k = np.eye(p) if K is None else np.asarray(K, dtype=float)
    hessian = 2.0 * reference.second
    root, _ = _sqrt_and_inverse(hessian)
    forcing = directional_risk_jacobian(weights, reference, delta)
    metric = hessian + regularizer_strength * k
    response_w = -np.linalg.solve(metric, forcing + regularizer_strength * np.asarray(C) @ delta)
    response_z = root @ response_w
    return {
        "response_w": response_w,
        "response_z": response_z,
        "forcing": forcing,
        "local_metric_min_eigenvalue": float(np.linalg.eigvalsh(metric).min()),
    }


def exact_ift_blocks(weights: Array, reference: MomentState, C: Array,
                     regularizer_strength: float) -> dict[str, Array | float]:
    """Return the exact local FOC blocks and ``Pi`` in state coordinates.

    The state coordinate is the deterministic vector returned by
    :func:`encode_state`.  The finite difference below differentiates the
    analytic quadratic gradient, not a trained model or a target outcome.
    """
    state_vector = encode_state(reference)
    p = weights.size
    h_r = 2.0 * reference.second
    b_r = _finite_state_jacobian(
        lambda vector: _gradient(weights, decode_state(vector, p)), state_vector
    )
    K = np.eye(p)
    C_array = np.asarray(C, dtype=float)
    if C_array.shape != (p, state_vector.size):
        raise ValueError("C has incompatible state dimension")
    total_h = h_r + regularizer_strength * K
    total_b = b_r + regularizer_strength * C_array
    pi_w = -np.linalg.solve(total_h, total_b)
    root, _ = _sqrt_and_inverse(h_r)
    pi_z = root @ pi_w
    return {
        "H_R": h_r,
        "B_R": b_r,
        "K": K,
        "C": C_array,
        "total_H": total_h,
        "total_B": total_b,
        "Pi_w": pi_w,
        "Pi": pi_z,
        "min_local_metric_eigenvalue": float(np.linalg.eigvalsh(total_h).min()),
    }


def _finite_state_jacobian(function, point: Array, step: float = 1e-6) -> Array:
    point = np.asarray(point, dtype=float)
    value = np.asarray(function(point), dtype=float)
    result = np.empty((value.size, point.size))
    for index in range(point.size):
        plus, minus = point.copy(), point.copy()
        plus[index] += step
        minus[index] -= step
        result[:, index] = (function(plus) - function(minus)) / (2.0 * step)
    return result


def finite_difference_response(reference_weights: Array, reference: MomentState,
                               delta: Array, C: Array, regularizer_strength: float,
                               step: float = 1e-5) -> dict[str, Array | float]:
    """Audit an IFT directional response against retrained quadratic heads."""
    base = encode_state(reference)
    plus = decode_state(base + step * np.asarray(delta), reference_weights.size)
    minus = decode_state(base - step * np.asarray(delta), reference_weights.size)
    wp = retrain_centered_head(reference_weights, reference, plus, C, regularizer_strength)
    wm = retrain_centered_head(reference_weights, reference, minus, C, regularizer_strength)
    root, _ = _sqrt_and_inverse(2.0 * reference.second)
    finite = root @ (wp - wm) / (2.0 * step)
    analytic = exact_directional_response(reference_weights, reference, delta, C,
                                          regularizer_strength)["response_z"]
    return {
        "finite_difference": finite,
        "ift": np.asarray(analytic),
        "relative_error": float(np.linalg.norm(finite - analytic) /
                                  max(1.0, np.linalg.norm(analytic))),
        "step": float(step),
    }


def source_only_response_target(reference: MomentState, pseudo_target: MomentState,
                                reference_weights: Array) -> Array:
    """Estimate a required response using only two source moment states."""
    root, _ = _sqrt_and_inverse(2.0 * reference.second)
    pseudo_optimum = np.linalg.solve(pseudo_target.second, pseudo_target.xy)
    return -root @ (pseudo_optimum - reference_weights)


def retrain_centered_head(reference_weights: Array, reference: MomentState,
                          shifted: MomentState, C: Array,
                          regularizer_strength: float) -> Array:
    """Solve the exact shifted head FOC for a frozen learned regularizer."""
    delta = encode_state(shifted) - encode_state(reference)
    p = reference_weights.size
    metric = 2.0 * shifted.second + regularizer_strength * np.eye(p)
    rhs = 2.0 * shifted.xy + regularizer_strength * reference_weights
    rhs -= regularizer_strength * np.asarray(C) @ delta
    return np.linalg.solve(metric, rhs)


def matched_random_response_targets(folds: tuple[Fold, ...], seed: int) -> tuple[Array, ...]:
    """Randomize response direction while preserving each source target norm."""
    rng = np.random.default_rng(seed)
    values = []
    for fold in folds:
        draw = rng.normal(size=fold.required_response.size)
        norm = np.linalg.norm(fold.required_response)
        values.append(draw * (norm / max(np.linalg.norm(draw), 1e-15)))
    return tuple(values)


def fit_source_only_regularizer(states: tuple[MomentState, ...], beta: float, *,
                                regularizer_strength: float = 0.1,
                                response_term: bool = True,
                                randomize_response: bool = False,
                                random_seed: int = 0) -> FittedRegularizer:
    """Fit low-rank ``C=U V.T`` from source-only LOO folds.

    The objective and exact gradient depend only on the supplied source states.
    Target moments cannot be passed to this API.
    """
    from scipy.optimize import minimize

    folds = leave_one_out_folds(states)
    basis = _contrast_basis(states)
    p, rank = states[0].xy.size, basis.shape[1]
    targets = matched_random_response_targets(folds, random_seed) if randomize_response else tuple(
        fold.required_response for fold in folds
    )
    ridge = 1e-6

    def value_gradient(flat: Array) -> tuple[float, Array]:
        u = flat.reshape(p, rank)
        c = u @ basis.T
        total = 0.0
        gradient_u = np.zeros_like(u)
        for fold, required in zip(folds, targets):
            v = basis.T @ fold.delta
            t = u @ v
            h_pseudo = 2.0 * fold.pseudo_target.second
            b_pseudo = 2.0 * fold.pseudo_target.xy
            pseudo_inverse = np.linalg.inv(h_pseudo + regularizer_strength * np.eye(p))
            candidate = pseudo_inverse @ (
                b_pseudo + regularizer_strength * fold.reference_weights
                - regularizer_strength * t
            )
            pseudo_risk = risk(candidate, fold.pseudo_target)
            d_candidate_dt = -regularizer_strength * pseudo_inverse
            grad_t = d_candidate_dt.T @ _gradient(candidate, fold.pseudo_target)

            h_reference = 2.0 * fold.reference.second
            root, _ = _sqrt_and_inverse(h_reference)
            reference_inverse = np.linalg.inv(h_reference + regularizer_strength * np.eye(p))
            forcing = directional_risk_jacobian(fold.reference_weights, fold.reference, fold.delta)
            response_map = -regularizer_strength * root @ reference_inverse
            response = root @ (-reference_inverse @ (forcing + regularizer_strength * t))
            mismatch = required + response
            response_loss = float(mismatch @ mismatch) if response_term else 0.0
            if response_term:
                grad_t += beta * 2.0 * response_map.T @ mismatch
            total += pseudo_risk + beta * response_loss
            gradient_u += np.outer(grad_t, v)
        count = float(len(folds))
        total = total / count + ridge * float(u.ravel() @ u.ravel())
        gradient_u = gradient_u / count + 2.0 * ridge * u
        return total, gradient_u.ravel()

    result = minimize(lambda x: value_gradient(x), np.zeros(p * rank), jac=True,
                      method="L-BFGS-B", options={"maxiter": 500, "ftol": 1e-13, "gtol": 1e-10})
    u = result.x.reshape(p, rank)
    c = u @ basis.T
    pseudo_risks: list[float] = []
    mismatches: list[float] = []
    for fold, required in zip(folds, targets):
        candidate = retrain_centered_head(fold.reference_weights, fold.reference,
                                          fold.pseudo_target, c, regularizer_strength)
        response = exact_directional_response(fold.reference_weights, fold.reference,
                                              fold.delta, c, regularizer_strength)["response_z"]
        pseudo_risks.append(risk(candidate, fold.pseudo_target))
        mismatches.append(float(np.linalg.norm(required + np.asarray(response)) ** 2))
    mean_risk = float(np.mean(pseudo_risks))
    mean_mismatch = float(np.mean(mismatches))
    selection_score = mean_risk + 0.1 * mean_mismatch
    if randomize_response:
        kind = "matched_norm_random_source_response"
    elif response_term:
        kind = "loo_finite_source_optimum_response"
    else:
        kind = "response_term_removed"
    return FittedRegularizer(
        "PSEUDO_RESPONSE" if response_term and not randomize_response else (
            "RANDOM_RESPONSE" if randomize_response else "NO_RESPONSE"
        ), float(beta), float(regularizer_strength), c, basis, selection_score,
        mean_risk, mean_mismatch, bool(result.success), str(result.message), kind,
    )


def mldg_head(states: tuple[MomentState, ...], step: float = 0.05) -> Array:
    """Exact one-step MLDG-style quadratic baseline using source LOO folds."""
    folds = leave_one_out_folds(states)
    normal = np.zeros_like(states[0].second)
    rhs = np.zeros_like(states[0].xy)
    for fold in folds:
        h_meta = 2.0 * fold.reference.second
        b_meta = 2.0 * fold.reference.xy
        transform = np.eye(h_meta.shape[0]) - step * h_meta
        offset = step * b_meta
        h_pseudo = 2.0 * fold.pseudo_target.second
        b_pseudo = 2.0 * fold.pseudo_target.xy
        normal += transform.T @ h_pseudo @ transform
        rhs += transform.T @ (b_pseudo - h_pseudo @ offset)
    return np.linalg.solve(normal, rhs)


def shift_configuration(configuration: ProbeConfiguration, direction: int,
                        amount: float) -> tuple[tuple[ModuleEnvironment, ...], ModuleEnvironment]:
    """Apply one legal relation shift to source collection and held-out environment."""
    if direction not in (0, 1):
        raise ValueError("only the two declared shortcut-relation directions are legal")
    shifted_sources = []
    for environment in configuration.sources:
        rhos = list(environment.shortcut_rhos)
        rhos[direction] += amount
        if abs(rhos[direction]) >= 0.99:
            raise ValueError("held-out shift leaves the preregistered legal interval")
        shifted_sources.append(environment.updated(shortcut_rhos=tuple(rhos)))
    target_rhos = list(configuration.base.shortcut_rhos)
    target_rhos[direction] += amount
    if abs(target_rhos[direction]) >= 0.99:
        raise ValueError("target shift leaves the preregistered legal interval")
    return tuple(shifted_sources), configuration.base.updated(shortcut_rhos=tuple(target_rhos))


def source_states(configuration: ProbeConfiguration) -> tuple[MomentState, ...]:
    return tuple(environment_state(environment) for environment in configuration.sources)


__all__ = [
    "ProbeConfiguration", "Fold", "FittedRegularizer", "encode_state", "decode_state",
    "make_configurations", "leave_one_out_folds", "directional_risk_jacobian",
    "exact_directional_response", "retrain_centered_head", "fit_source_only_regularizer",
    "matched_random_response_targets", "mldg_head", "shift_configuration", "source_states",
    "mixture_state", "source_optimum",
]
