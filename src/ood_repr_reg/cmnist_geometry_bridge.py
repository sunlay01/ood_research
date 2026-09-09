"""CMNIST bridge for the Round-3 source/response geometry.

This module deliberately uses a frozen encoder and an empirical squared-loss
head.  Color probabilities are integrated analytically over a fixed grayscale
bank, so environment finite differences do not contain Bernoulli resampling
noise.  The output is an empirical finite-sample bridge, not a population
theorem for neural networks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

Array = np.ndarray


@dataclass(frozen=True)
class RepresentationBank:
    red: Array
    green: Array
    labels: Array

    @property
    def dimension(self) -> int:
        return int(self.red.shape[1] + 1)


@dataclass(frozen=True)
class CMNISTFamily:
    """Low-dimensional declared CMNIST environment family."""

    name: str
    source_rhos: tuple[float, float] = (0.9, 0.8)
    target_rho: float = 0.1
    hidden_exposed: bool = False
    directions: tuple[str, ...] = ("rho_source_1", "rho_source_2", "rho_hidden")

    def __post_init__(self) -> None:
        if len(self.source_rhos) != 2 or any(not 0 < r < 1 for r in self.source_rhos):
            raise ValueError("source_rhos must contain two probabilities in (0, 1)")
        if not 0 < self.target_rho < 1:
            raise ValueError("target_rho must be in (0, 1)")

    @property
    def dimension(self) -> int:
        return len(self.directions)

    def metadata(self) -> dict[str, object]:
        return {
            "family": self.name,
            "directions": list(self.directions),
            "source_rhos": list(self.source_rhos),
            "target_rho": self.target_rho,
            "hidden_source_exposed": self.hidden_exposed,
            "target_labels_used_for_construction": False,
            "unit_scales": {direction: 1.0 for direction in self.directions},
        }

    def source_rhos_at(self, theta: Array) -> tuple[float, float]:
        theta = np.asarray(theta, dtype=float)
        result = np.asarray(self.source_rhos, dtype=float)
        result += theta[:2]
        if self.hidden_exposed and "rho_hidden" in self.directions:
            result += theta[self.directions.index("rho_hidden")]
        if np.any((result <= 0) | (result >= 1)):
            raise ValueError("CMNIST correlation perturbation left (0,1)")
        return float(result[0]), float(result[1])

    def target_rho_at(self, theta: Array) -> float:
        theta = np.asarray(theta, dtype=float)
        value = self.target_rho
        for index, direction in enumerate(self.directions):
            if direction.startswith("rho_"):
                value += float(theta[index])
        if not 0 < value < 1:
            raise ValueError("CMNIST target correlation perturbation left (0,1)")
        return value


def svec_symmetric(matrix: Array) -> Array:
    value = np.asarray(matrix, dtype=float)
    entries: list[float] = []
    for row in range(value.shape[0]):
        entries.append(float(value[row, row]))
        entries.extend(float(np.sqrt(2.0) * value[row, col]) for col in range(row))
    return np.asarray(entries)


def decode_svec(vector: Array, dimension: int) -> Array:
    vector = np.asarray(vector, dtype=float)
    matrix = np.zeros((dimension, dimension), dtype=float)
    index = 0
    for row in range(dimension):
        matrix[row, row] = vector[index]
        index += 1
        for col in range(row):
            matrix[row, col] = matrix[col, row] = vector[index] / np.sqrt(2.0)
            index += 1
    return matrix


def task_state(matrix: Array, cross: Array, constant: float) -> Array:
    return np.concatenate((svec_symmetric(matrix), np.asarray(cross, dtype=float), [float(constant)]))


def feature_bank(model, gray, digit, *, device) -> RepresentationBank:
    """Cache exact red/green representations for one fixed grayscale bank."""
    import torch
    from .cmnist_feature_probe import colorize

    model.eval()
    with torch.no_grad():
        n = gray.shape[0]
        red = model.encode(colorize(gray, torch.zeros(n, dtype=torch.long)).to(device)).cpu().numpy()
        green = model.encode(colorize(gray, torch.ones(n, dtype=torch.long)).to(device)).cpu().numpy()
    # Work on the empirical non-degenerate support of the frozen features.
    # This is a coordinate reduction, not a ridge/damping modification.
    pooled = np.concatenate((red, green), axis=0)
    centered = pooled - pooled.mean(axis=0, keepdims=True)
    _, singular, vh = np.linalg.svd(centered, full_matrices=False)
    threshold = max(float(singular[0]) if singular.size else 0.0, 1e-12) * 1e-10
    keep = singular > threshold
    basis = vh[keep].T if np.any(keep) else np.zeros((red.shape[1], 0))
    center = pooled.mean(axis=0)
    red = (red - center) @ basis
    green = (green - center) @ basis
    return RepresentationBank(red=red, green=green, labels=digit.detach().cpu().numpy().astype(float))


def moment_state(bank: RepresentationBank, rho: float) -> tuple[Array, Array, float]:
    """Return E[xx^T], E[xY], E[Y^2] with x=(1, frozen representation)."""
    if not 0 < rho < 1:
        raise ValueError("rho must be in (0,1)")
    y = bank.labels
    # P(red | y=0)=rho and P(red | y=1)=1-rho.
    red_probability = np.where(y < 0.5, rho, 1.0 - rho)
    xr = np.column_stack((np.ones(len(y)), bank.red))
    xg = np.column_stack((np.ones(len(y)), bank.green))
    weighted_xx = red_probability[:, None, None] * (xr[:, :, None] * xr[:, None, :])
    weighted_xx += (1.0 - red_probability)[:, None, None] * (xg[:, :, None] * xg[:, None, :])
    x = red_probability[:, None] * xr + (1.0 - red_probability)[:, None] * xg
    matrix = weighted_xx.mean(axis=0)
    cross = (x * y[:, None]).mean(axis=0)
    return matrix, cross, float(np.mean(y * y))


def state_vector(bank: RepresentationBank, rho: float) -> Array:
    return task_state(*moment_state(bank, rho))


def source_state_stack(bank: RepresentationBank, family: CMNISTFamily, theta: Array) -> Array:
    states = []
    for index, rho in enumerate(family.source_rhos_at(theta)):
        local_bank = bank
        if "brightness_nuisance" in family.directions:
            nuisance_index = family.directions.index("brightness_nuisance")
            scale = 1.0 + float(theta[nuisance_index])
            local_bank = RepresentationBank(bank.red * scale, bank.green * scale, bank.labels)
        states.append(state_vector(local_bank, rho))
    return np.concatenate(states)


def target_state(bank: RepresentationBank, family: CMNISTFamily, theta: Array) -> tuple[Array, Array, float]:
    return moment_state(bank, family.target_rho_at(theta))


def source_observation(bank: RepresentationBank, family: CMNISTFamily, theta: Array | None = None,
                       step: float = 1e-4) -> Array:
    theta = np.zeros(family.dimension) if theta is None else np.asarray(theta, dtype=float)
    columns = []
    for index in range(family.dimension):
        direction = np.zeros(family.dimension)
        direction[index] = step
        columns.append((source_state_stack(bank, family, theta + direction)
                        - source_state_stack(bank, family, theta - direction)) / (2.0 * step))
    return np.column_stack(columns)


def _sqrt_spd(matrix: Array) -> Array:
    values, vectors = np.linalg.eigh((matrix + matrix.T) / 2.0)
    if values.min() <= 0:
        raise ValueError("source Hessian is not positive definite")
    return vectors @ np.diag(np.sqrt(values)) @ vectors.T


def head_from_state(vector: Array, dimension: int, *, method: str = "erm", lam: float = 0.0) -> Array:
    width = dimension * (dimension + 1) // 2 + dimension + 1
    if vector.size % width:
        raise ValueError("invalid stacked task state")
    states = []
    for start in range(0, vector.size, width):
        block = vector[start:start + width]
        matrix = decode_svec(block, dimension)
        nmat = dimension * (dimension + 1) // 2
        cross = block[nmat:nmat + dimension]
        constant = block[-1]
        states.append((matrix, cross, constant))

    def value_grad(w: Array) -> tuple[float, Array]:
        risks = np.asarray([w @ m @ w - 2 * w @ c + k for m, c, k in states])
        grads = np.asarray([2 * (m @ w - c) for m, c, _ in states])
        mean_risk = float(risks.mean())
        mean_grad = grads.mean(axis=0)
        key = method.lower()
        if key == "erm":
            return mean_risk, mean_grad
        if key == "l2":
            return mean_risk + 0.5 * lam * float(w @ w), mean_grad + lam * w
        if key == "vrex":
            centered = risks - risks.mean()
            return mean_risk + lam * float(np.mean(centered ** 2)), mean_grad + lam * 2 * np.mean(centered[:, None] * grads, axis=0)
        if key == "irmv1":
            radial = np.asarray([2 * (w @ m @ w - w @ c) for m, c, _ in states])
            radial_grad = np.asarray([4 * m @ w - 2 * c for m, c, _ in states])
            return mean_risk + lam * float(np.mean(radial ** 2)), mean_grad + lam * 2 * np.mean(radial[:, None] * radial_grad, axis=0)
        raise ValueError(f"unsupported method {method}")

    if method.lower() == "erm" or lam == 0.0:
        matrix = np.mean([s[0] for s in states], axis=0)
        cross = np.mean([s[1] for s in states], axis=0)
        return np.linalg.solve(matrix, cross)
    from scipy.optimize import minimize
    matrix = np.mean([s[0] for s in states], axis=0)
    cross = np.mean([s[1] for s in states], axis=0)
    start = np.linalg.solve(matrix, cross)
    result = minimize(lambda w: value_grad(w), start, jac=True, method="BFGS",
                      options={"gtol": 1e-10, "maxiter": 1000})
    if not result.success and np.linalg.norm(value_grad(result.x)[1]) > 2e-7:
        raise RuntimeError(f"head optimization failed: {result.message}")
    return np.asarray(result.x, dtype=float)


def head_objective_gradient(w: Array, state: Array, dimension: int, method: str, lam: float) -> Array:
    # This independent numpy gradient is used by the FD/IFT audit.
    width = dimension * (dimension + 1) // 2 + dimension + 1
    states = []
    for start in range(0, state.size, width):
        block = state[start:start + width]
        nmat = dimension * (dimension + 1) // 2
        states.append((decode_svec(block, dimension), block[nmat:nmat + dimension], block[-1]))
    risks = np.asarray([w @ m @ w - 2 * w @ c + k for m, c, k in states])
    grads = np.asarray([2 * (m @ w - c) for m, c, _ in states])
    result = grads.mean(axis=0)
    if method.lower() == "l2":
        return result + lam * w
    if method.lower() == "vrex":
        centered = risks - risks.mean()
        return result + lam * 2 * np.mean(centered[:, None] * grads, axis=0)
    if method.lower() == "irmv1":
        radial = np.asarray([2 * (w @ m @ w - w @ c) for m, c, _ in states])
        radial_grad = np.asarray([4 * m @ w - 2 * c for m, c, _ in states])
        return result + lam * 2 * np.mean(radial[:, None] * radial_grad, axis=0)
    return result


def ift_matrices(state: Array, dimension: int, method: str, lam: float, w: Array) -> tuple[Array, Array]:
    """Compute D_w F and D_state F through torch double autodiff."""
    import torch
    wt = torch.tensor(w, dtype=torch.float64, requires_grad=True)
    yt = torch.tensor(state, dtype=torch.float64, requires_grad=True)
    width = dimension * (dimension + 1) // 2 + dimension + 1

    def unpack(block):
        m = torch.zeros((dimension, dimension), dtype=block.dtype, device=block.device)
        idx = 0
        for row in range(dimension):
            m[row, row] = block[idx]; idx += 1
            for col in range(row):
                m[row, col] = block[idx] / np.sqrt(2.0)
                m[col, row] = m[row, col]; idx += 1
        nmat = dimension * (dimension + 1) // 2
        return m, block[nmat:nmat + dimension], block[-1]

    def F(wv, yv):
        blocks = [unpack(yv[start:start + width]) for start in range(0, yv.numel(), width)]
        risks = torch.stack([wv @ m @ wv - 2 * wv @ c + k for m, c, k in blocks])
        gradients = torch.stack([2 * (m @ wv - c) for m, c, _ in blocks])
        result = gradients.mean(dim=0)
        if method.lower() == "l2":
            result = result + lam * wv
        elif method.lower() == "vrex":
            centered = risks - risks.mean()
            result = result + lam * 2 * torch.mean(centered[:, None] * gradients, dim=0)
        elif method.lower() == "irmv1":
            radial = torch.stack([2 * (wv @ m @ wv - wv @ c) for m, c, _ in blocks])
            radial_grad = torch.stack([4 * m @ wv - 2 * c for m, c, _ in blocks])
            result = result + lam * 2 * torch.mean(radial[:, None] * radial_grad, dim=0)
        return result

    dw = torch.autograd.functional.jacobian(lambda x: F(x, yt), wt)
    dy = torch.autograd.functional.jacobian(lambda y: F(wt, y), yt)
    return dw.detach().numpy(), dy.detach().numpy()


def rank(matrix: Array, tolerance: float = 1e-8) -> int:
    singular = np.linalg.svd(np.asarray(matrix), compute_uv=False)
    return int(np.sum(singular > tolerance * singular[0])) if singular.size and singular[0] > 0 else 0


def null_basis(matrix: Array, tolerance: float = 1e-8) -> Array:
    _, singular, vh = np.linalg.svd(matrix, full_matrices=True)
    r = int(np.sum(singular > tolerance * singular[0])) if singular.size and singular[0] > 0 else 0
    return vh[r:].T


def geometry_metrics(A: Array, O: Array, tolerance: float = 1e-8) -> dict[str, object]:
    irreducible, recoverable, projector = response_decomposition(A, O, tolerance)
    N = null_basis(O, tolerance)
    alpha = float(np.linalg.svd(A @ N, compute_uv=False)[0]) if N.shape[1] else 0.0
    return {
        "dim_U": int(O.shape[1]), "rank_O": rank(O, tolerance), "rank_A": rank(A, tolerance),
        "kernel_dimension": int(N.shape[1]), "alpha": alpha,
        "information_floor": 0.5 * alpha * alpha,
        "A_irreducible": irreducible,
        "A_recoverable": recoverable,
        "null_projector": projector,
        "decomposition_residual": float(np.linalg.norm(A - irreducible - recoverable)),
        "singular_values_A": np.linalg.svd(A, compute_uv=False).tolist(),
        "singular_values_O": np.linalg.svd(O, compute_uv=False).tolist(),
    }


def response_decomposition(A: Array, O: Array, tolerance: float = 1e-8) -> tuple[Array, Array, Array]:
    """Split a response operator into source-invisible and source-recoverable parts.

    The projector is built from an orthonormal basis of ``ker(O)``.  This is
    the response-side decomposition used by the 3E theory:
    ``A_irreducible = A P`` and ``A_recoverable = A (I-P)``.
    """
    a = np.asarray(A, dtype=float)
    o = np.asarray(O, dtype=float)
    if a.ndim != 2 or o.ndim != 2 or a.shape[1] != o.shape[1]:
        raise ValueError("A and O must be matrices with the same world dimension")
    n = null_basis(o, tolerance)
    p = n @ n.T if n.shape[1] else np.zeros((o.shape[1], o.shape[1]))
    identity = np.eye(o.shape[1])
    return a @ p, a @ (identity - p), p


def affine_regret(z0: Array, A: Array, adaptive: Array) -> float:
    """Independent finite-dimensional shifted-ball audit."""
    from .round3r_3e_joint_regret import shifted_ball_maximum
    return float(shifted_ball_maximum(z0, A + adaptive).value)


def method_row(bank: RepresentationBank, family: CMNISTFamily, method: str, lam: float,
               source_design: str, steps: tuple[float, ...]) -> dict[str, object]:
    zero = np.zeros(family.dimension)
    state = source_state_stack(bank, family, zero)
    p = bank.dimension
    source_matrix = np.mean([moment_state(bank, r)[0] for r in family.source_rhos], axis=0)
    H = 2.0 * source_matrix
    Hsqrt = _sqrt_spd(H)
    erm = head_from_state(state, p, method="erm")
    weights = head_from_state(state, p, method=method, lam=lam)
    z0 = Hsqrt @ (weights - erm)
    dw, dy = ift_matrices(state, p, method, lam, weights)
    eig = np.linalg.eigvalsh((dw + dw.T) / 2.0)
    # Do not use a determinant threshold: it is scale-dependent and can reject
    # a well-defined but ill-conditioned local metric.  The eigenvalue and
    # finite condition number are the declared validity checks.
    valid = bool(eig.min() > 1e-10 and np.isfinite(np.linalg.cond(dw)))
    A = response_operator(bank, family, Hsqrt, steps[0])
    O = source_observation(bank, family, step=steps[0])
    pi_state = -np.linalg.pinv(dw) @ dy
    pi_o = Hsqrt @ pi_state @ O
    fd_errors = []
    fd_direction_rows = []
    for step in steps:
        cols = []
        for index in range(family.dimension):
            direction = np.zeros(family.dimension); direction[index] = step
            wp = head_from_state(source_state_stack(bank, family, direction), p, method=method, lam=lam)
            wm = head_from_state(source_state_stack(bank, family, -direction), p, method=method, lam=lam)
            fd = Hsqrt @ (wp - wm) / (2.0 * step)
            cols.append(fd)
        fd_matrix = np.column_stack(cols)
        pred = Hsqrt @ (-np.linalg.pinv(dw) @ dy @ O)
        fd_errors.append(float(np.linalg.norm(fd_matrix - pred) / max(np.linalg.norm(fd_matrix), 1e-10)))
        for index, (fd_column, pred_column) in enumerate(zip(fd_matrix.T, pred.T, strict=True)):
            fd_direction_rows.append({
                "step": float(step), "direction": family.directions[index],
                "fd_norm": float(np.linalg.norm(fd_column)),
                "ift_norm": float(np.linalg.norm(pred_column)),
                "cosine": float(fd_column @ pred_column / max(np.linalg.norm(fd_column) * np.linalg.norm(pred_column), 1e-12)),
                "relative_error": float(np.linalg.norm(fd_column - pred_column) / max(np.linalg.norm(fd_column), 1e-10)),
            })
    metrics = geometry_metrics(A, O)
    A_irr = np.asarray(metrics["A_irreducible"])
    A_rec = np.asarray(metrics["A_recoverable"])
    E = A_rec + pi_o
    regret = affine_regret(z0, A, pi_o)
    N = null_basis(O)
    R = A_irr
    alpha = float(np.linalg.svd(A @ N, compute_uv=False)[0]) if N.shape[1] else 0.0
    slack = alpha * alpha * np.eye(A.shape[0]) - R @ R.T
    slack_gap = float(np.linalg.eigvalsh((slack - E @ E.T + (slack - E @ E.T).T) / 2.0).min())
    return {
        "method": method.upper(), "lambda": float(lam), "source_design": source_design,
        "valid": valid, "norm_z0": float(np.linalg.norm(z0)),
        "pi_O_operator_norm": float(np.linalg.svd(pi_o, compute_uv=False)[0]) if pi_o.size else 0.0,
        "A_irreducible_operator_norm": float(np.linalg.svd(A_irr, compute_uv=False)[0]) if A_irr.size else 0.0,
        "A_recoverable_operator_norm": float(np.linalg.svd(A_rec, compute_uv=False)[0]) if A_rec.size else 0.0,
        "E_operator_norm": float(np.linalg.svd(E, compute_uv=False)[0]) if E.size else 0.0,
        "decomposition_residual": float(np.linalg.norm(A - A_irr - A_rec)),
        "affine_regret": regret, "information_floor": metrics["information_floor"],
        "total_excess": regret - float(metrics["information_floor"]),
        "min_local_metric_eigenvalue": float(eig.min()), "condition_number": float(np.linalg.cond(dw)),
        "fd_error_max": max(fd_errors), "fd_errors": fd_errors,
        "fd_direction_rows": fd_direction_rows,
        "factorization_residual": float(np.linalg.norm(pi_o - Hsqrt @ pi_state @ O)),
        "spectral_slack_min_eigenvalue": slack_gap,
    }


def response_operator(bank: RepresentationBank, family: CMNISTFamily, Hsqrt: Array, step: float) -> Array:
    columns = []
    for index in range(family.dimension):
        direction = np.zeros(family.dimension); direction[index] = step
        mp, cp, _ = target_state(bank, family, direction)
        mm, cm, _ = target_state(bank, family, -direction)
        qp = np.linalg.pinv(mp, rcond=1e-10) @ cp
        qm = np.linalg.pinv(mm, rcond=1e-10) @ cm
        columns.append(Hsqrt @ (qp - qm) / (2.0 * step))
    return np.column_stack(columns)


def build_geometry(bank: RepresentationBank, family: CMNISTFamily, steps=(1e-3, 5e-4, 2.5e-4)) -> dict[str, object]:
    source_matrix = np.mean([moment_state(bank, rho)[0] for rho in family.source_rhos], axis=0)
    Hsqrt = _sqrt_spd(2.0 * source_matrix)
    A_by_step = [response_operator(bank, family, Hsqrt, step) for step in steps]
    O_by_step = [source_observation(bank, family, step=step) for step in steps]
    main_A, main_O = A_by_step[0], O_by_step[0]
    metrics = geometry_metrics(main_A, main_O)
    return {"family": family.metadata(), "A": main_A, "O": main_O, **metrics,
            "A_step_relative_errors": [float(np.linalg.norm(a-main_A)/max(np.linalg.norm(main_A),1e-12)) for a in A_by_step],
            "O_step_relative_errors": [float(np.linalg.norm(o-main_O)/max(np.linalg.norm(main_O),1e-12)) for o in O_by_step]}


__all__ = ["RepresentationBank", "CMNISTFamily", "feature_bank", "moment_state",
           "task_state", "source_state_stack", "target_state", "source_observation",
           "response_operator", "head_from_state", "head_objective_gradient",
           "ift_matrices", "geometry_metrics", "response_decomposition", "build_geometry", "method_row"]
