"""Source-supervised semantic latent decomposition for LATENT-001.

The semantic projectors are estimated only from source folds and the frozen
factorial intervention design. Target environments are used only by the risk
accounting functions after projectors and models have been fixed.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, permutations
from typing import Iterable, Sequence

import numpy as np
import torch
from torch import Tensor, nn

COMPONENTS = ("task", "relation", "mean", "covariance", "residual")
METHODS = ("erm", "l1", "l2", "irmv1", "mmd", "coral", "grad_align", "hess_align")
BLOCKS = {
    "task": slice(0, 3),
    "relation": slice(3, 6),
    "mean": slice(6, 9),
    "covariance": slice(9, 12),
    "residual": slice(12, 15),
}


@dataclass(frozen=True)
class InterventionEnvironment:
    name: str
    relation: np.ndarray
    mean: np.ndarray
    covariance_scale: np.ndarray
    design: np.ndarray
    shift_type: str
    covered: bool


@dataclass(frozen=True)
class LatentSCM:
    loading: np.ndarray
    beta: np.ndarray
    relation_base: np.ndarray
    sigma_u: float = 0.35
    sigma_relation: float = 0.45
    sigma_mean: float = 0.55
    sigma_covariance: float = 0.55
    sigma_residual: float = 0.7
    sigma_y: float = 0.15
    relation_amplitude: float = 0.45
    mean_amplitude: float = 0.8
    covariance_log_amplitude: float = 0.55

    @property
    def input_dim(self) -> int:
        return 15


@dataclass(frozen=True)
class EnvironmentBatch:
    environment: InterventionEnvironment
    x: Tensor
    y: Tensor


@dataclass(frozen=True)
class Whitening:
    mean: np.ndarray
    transform: np.ndarray
    inverse: np.ndarray

    @property
    def rank(self) -> int:
        return self.transform.shape[0]


@dataclass(frozen=True)
class SemanticDecomposition:
    whitening: Whitening
    projectors: dict[str, np.ndarray]
    bases: dict[str, np.ndarray]
    raw_bases: dict[str, np.ndarray]
    oracle_projectors: dict[str, np.ndarray]
    diagnostics: dict[str, float]


def default_scm() -> LatentSCM:
    return LatentSCM(
        loading=np.diag([1.0, 0.75, 0.55]),
        beta=np.array([1.0, -0.8, 0.65]),
        # Axis three is predictive but never varied by the source design.
        relation_base=np.diag([0.4, 0.35, 0.65]),
    )


def _environment_from_design(
    scm: LatentSCM,
    name: str,
    design: Sequence[float],
    *,
    shift_type: str,
    covered: bool,
    relation_axis3: float = 0.0,
) -> InterventionEnvironment:
    design_array = np.asarray(design, dtype=float)
    if design_array.shape != (6,):
        raise ValueError("factorial design must have six entries")
    relation = scm.relation_base.copy()
    relation[0, 0] += scm.relation_amplitude * design_array[0]
    relation[1, 1] += scm.relation_amplitude * design_array[1]
    relation[2, 2] += scm.relation_amplitude * relation_axis3
    mean = scm.mean_amplitude * np.array([design_array[2], design_array[3], 0.0])
    covariance_scale = np.ones(3)
    covariance_scale[:2] = np.exp(
        0.5 * scm.covariance_log_amplitude * design_array[4:6]
    )
    return InterventionEnvironment(
        name=name,
        relation=relation,
        mean=mean,
        covariance_scale=covariance_scale,
        design=design_array,
        shift_type=shift_type,
        covered=covered,
    )


def source_environments(scm: LatentSCM) -> tuple[InterventionEnvironment, ...]:
    environments = [
        _environment_from_design(
            scm, "source_baseline", np.zeros(6), shift_type="source", covered=True
        )
    ]
    names = ("relation_1", "relation_2", "mean_1", "mean_2", "covariance_1", "covariance_2")
    for axis, name in enumerate(names):
        for sign in (-1.0, 1.0):
            design = np.zeros(6)
            design[axis] = sign
            environments.append(
                _environment_from_design(
                    scm,
                    f"source_{name}_{'plus' if sign > 0 else 'minus'}",
                    design,
                    shift_type="source",
                    covered=True,
                )
            )
    return tuple(environments)


def target_environments(scm: LatentSCM) -> tuple[InterventionEnvironment, ...]:
    targets = [
        ("covered_interpolation", [0.5, -0.35, 0, 0, 0, 0], "relation", True, 0.0),
        ("covered_extrapolation", [1.8, -1.5, 0, 0, 0, 0], "relation", True, 0.0),
        ("unseen_relation_axis3", [0, 0, 0, 0, 0, 0], "relation", False, -2.0),
        ("relation_sign_flip", [-2.0, -1.8, 0, 0, 0, 0], "relation", True, 0.0),
        ("mean_axis1", [0, 0, 1.6, 0, 0, 0], "mean", True, 0.0),
        ("covariance_axis2", [0, 0, 0, 0, 0, 1.8], "covariance", True, 0.0),
        ("compound_covered", [-1.5, 1.4, 1.2, -1.0, 1.2, -1.1], "compound", True, 0.0),
    ]
    return tuple(
        _environment_from_design(
            scm,
            name,
            design,
            shift_type=shift_type,
            covered=covered,
            relation_axis3=axis3,
        )
        for name, design, shift_type, covered, axis3 in targets
    )


def input_moments(
    scm: LatentSCM, environment: InterventionEnvironment
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return E[X], E[XX^T], E[XY], and E[Y^2] exactly."""
    mean = np.zeros(scm.input_dim)
    mean[BLOCKS["mean"]] = environment.mean
    covariance = np.zeros((scm.input_dim, scm.input_dim))
    loading = scm.loading
    relation = environment.relation
    covariance[BLOCKS["task"], BLOCKS["task"]] = (
        loading @ loading.T + scm.sigma_u * np.eye(3)
    )
    covariance[BLOCKS["relation"], BLOCKS["relation"]] = (
        relation @ relation.T + scm.sigma_relation * np.eye(3)
    )
    covariance[BLOCKS["task"], BLOCKS["relation"]] = loading @ relation.T
    covariance[BLOCKS["relation"], BLOCKS["task"]] = relation @ loading.T
    covariance[BLOCKS["mean"], BLOCKS["mean"]] = scm.sigma_mean * np.eye(3)
    covariance[BLOCKS["covariance"], BLOCKS["covariance"]] = (
        scm.sigma_covariance * np.diag(environment.covariance_scale**2)
    )
    covariance[BLOCKS["residual"], BLOCKS["residual"]] = (
        scm.sigma_residual * np.eye(3)
    )
    second = covariance + np.outer(mean, mean)
    cross = np.zeros(scm.input_dim)
    cross[BLOCKS["task"]] = loading @ scm.beta
    cross[BLOCKS["relation"]] = relation @ scm.beta
    y_second = float(scm.beta @ scm.beta + scm.sigma_y)
    return mean, second, cross, y_second


def sample_environment(
    scm: LatentSCM,
    environment: InterventionEnvironment,
    n: int,
    generator: torch.Generator,
) -> EnvironmentBatch:
    dtype = torch.float64
    c = torch.randn((n, 3), generator=generator, dtype=dtype)
    loading = torch.as_tensor(scm.loading, dtype=dtype)
    relation = torch.as_tensor(environment.relation, dtype=dtype)
    u = c @ loading.T + scm.sigma_u**0.5 * torch.randn((n, 3), generator=generator, dtype=dtype)
    a_relation = c @ relation.T + scm.sigma_relation**0.5 * torch.randn(
        (n, 3), generator=generator, dtype=dtype
    )
    a_mean = torch.as_tensor(environment.mean, dtype=dtype) + scm.sigma_mean**0.5 * torch.randn(
        (n, 3), generator=generator, dtype=dtype
    )
    a_covariance = (
        scm.sigma_covariance**0.5
        * torch.randn((n, 3), generator=generator, dtype=dtype)
        * torch.as_tensor(environment.covariance_scale, dtype=dtype)
    )
    residual = scm.sigma_residual**0.5 * torch.randn((n, 3), generator=generator, dtype=dtype)
    y = c @ torch.as_tensor(scm.beta, dtype=dtype) + scm.sigma_y**0.5 * torch.randn(
        n, generator=generator, dtype=dtype
    )
    return EnvironmentBatch(
        environment=environment,
        x=torch.cat((u, a_relation, a_mean, a_covariance, residual), dim=1),
        y=y,
    )


def sample_batches(
    scm: LatentSCM,
    environments: Iterable[InterventionEnvironment],
    n: int,
    seed: int,
) -> tuple[EnvironmentBatch, ...]:
    generator = torch.Generator().manual_seed(seed)
    return tuple(sample_environment(scm, environment, n, generator) for environment in environments)


class LinearLatentModel(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, seed: int) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.encoder = nn.Linear(input_dim, latent_dim, bias=False, dtype=torch.float64)
        self.head = nn.Linear(latent_dim, 1, bias=True, dtype=torch.float64)
        with torch.no_grad():
            initial = torch.randn((latent_dim, input_dim), generator=generator, dtype=torch.float64)
            q, _ = torch.linalg.qr(initial.T, mode="reduced")
            self.encoder.weight.copy_(q.T)
            self.head.weight.copy_(
                0.1 * torch.randn((1, latent_dim), generator=generator, dtype=torch.float64)
            )
            self.head.bias.zero_()

    def encode(self, x: Tensor) -> Tensor:
        return self.encoder(x)

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.encode(x)).squeeze(-1)


def mean_source_risk(model: LinearLatentModel, batches: Sequence[EnvironmentBatch]) -> Tensor:
    return torch.stack([(model(batch.x) - batch.y).square().mean() for batch in batches]).mean()


def _covariance(values: Tensor) -> Tensor:
    centered = values - values.mean(0, keepdim=True)
    return centered.T @ centered / max(values.shape[0] - 1, 1)


def _gaussian_rbf_expectation(
    mean_left: Tensor,
    covariance_left: Tensor,
    mean_right: Tensor,
    covariance_right: Tensor,
    bandwidth: float,
) -> Tensor:
    total = covariance_left + covariance_right
    identity = torch.eye(total.shape[0], dtype=total.dtype, device=total.device)
    scaled = identity + total / bandwidth**2
    delta = mean_left - mean_right
    return torch.linalg.det(scaled).clamp_min(1e-18).rsqrt() * torch.exp(
        -0.5 * delta @ torch.linalg.solve(bandwidth**2 * identity + total, delta)
    )


def gaussian_rbf_mmd(left: Tensor, right: Tensor) -> Tensor:
    left_mean, right_mean = left.mean(0), right.mean(0)
    left_covariance, right_covariance = _covariance(left), _covariance(right)
    terms = []
    for bandwidth in (0.75, 1.5, 3.0):
        xx = _gaussian_rbf_expectation(
            left_mean, left_covariance, left_mean, left_covariance, bandwidth
        )
        yy = _gaussian_rbf_expectation(
            right_mean, right_covariance, right_mean, right_covariance, bandwidth
        )
        xy = _gaussian_rbf_expectation(
            left_mean, left_covariance, right_mean, right_covariance, bandwidth
        )
        terms.append(xx + yy - 2.0 * xy)
    return torch.stack(terms).mean()


def regularizer_value(
    method: str, model: LinearLatentModel, batches: Sequence[EnvironmentBatch]
) -> Tensor:
    if method == "erm":
        return torch.zeros((), dtype=torch.float64)
    parameters = torch.cat([parameter.flatten() for parameter in model.parameters()])
    if method == "l1":
        return parameters.abs().mean()
    if method == "l2":
        return parameters.square().mean()

    encoded = [model.encode(batch.x) for batch in batches]
    if method == "mmd":
        return torch.stack(
            [gaussian_rbf_mmd(encoded[i], encoded[j]) for i, j in combinations(range(len(encoded)), 2)]
        ).mean()
    if method == "coral":
        covariances = [_covariance(z) for z in encoded]
        return torch.stack(
            [(covariances[i] - covariances[j]).square().mean() for i, j in combinations(range(len(encoded)), 2)]
        ).mean()

    gradients, hessians, radial = [], [], []
    head = model.head.weight.squeeze(0)
    bias = model.head.bias.squeeze(0)
    for z, batch in zip(encoded, batches, strict=True):
        augmented = torch.cat((z, torch.ones((z.shape[0], 1), dtype=z.dtype)), dim=1)
        prediction = z @ head + bias
        residual = prediction - batch.y
        gradients.append(2.0 * augmented.T @ residual / z.shape[0])
        hessians.append(2.0 * augmented.T @ augmented / z.shape[0])
        radial.append(2.0 * (residual * prediction).mean())
    if method == "irmv1":
        return torch.stack(radial).square().mean()
    if method == "grad_align":
        stacked = torch.stack(gradients)
        return (stacked - stacked.mean(0, keepdim=True)).square().mean()
    if method == "hess_align":
        stacked = torch.stack(hessians)
        return (stacked - stacked.mean(0, keepdim=True)).square().mean()
    raise ValueError(f"unknown method: {method}")


def train_model(
    method: str,
    batches: Sequence[EnvironmentBatch],
    *,
    input_dim: int,
    latent_dim: int,
    seed: int,
    strength: float,
    steps: int,
    learning_rate: float,
) -> tuple[LinearLatentModel, dict[str, float]]:
    if method not in METHODS:
        raise ValueError(f"unknown method: {method}")
    model = LinearLatentModel(input_dim, latent_dim, seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    with torch.no_grad():
        initial_risk = float(mean_source_risk(model, batches))
        initial_penalty = float(regularizer_value(method, model, batches))
    for _ in range(steps):
        optimizer.zero_grad()
        risk_value = mean_source_risk(model, batches)
        penalty = regularizer_value(method, model, batches)
        objective = risk_value + strength * penalty
        if not torch.isfinite(objective):
            raise FloatingPointError(f"non-finite {method} objective")
        objective.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=20.0)
        optimizer.step()
    with torch.no_grad():
        final_risk = float(mean_source_risk(model, batches))
        final_penalty = float(regularizer_value(method, model, batches))
    return model, {
        "initial_source_risk": initial_risk,
        "final_source_risk": final_risk,
        "initial_penalty": initial_penalty,
        "final_penalty": final_penalty,
    }


def fit_whitening(encoded: Sequence[np.ndarray], tolerance: float = 1e-8) -> Whitening:
    pooled = np.concatenate(encoded, axis=0)
    mean = pooled.mean(axis=0)
    covariance = np.cov(pooled - mean, rowvar=False)
    covariance = np.atleast_2d(covariance)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    keep = eigenvalues > tolerance * max(float(eigenvalues.max()), 1.0)
    if not np.any(keep):
        raise ValueError("latent source support is degenerate")
    values = eigenvalues[keep]
    vectors = eigenvectors[:, keep]
    transform = (vectors / np.sqrt(values)).T
    inverse = vectors * np.sqrt(values)
    return Whitening(mean=mean, transform=transform, inverse=inverse)


def _orthonormal_basis(matrix: np.ndarray, tolerance: float = 1e-7) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    if matrix.shape[1] == 0:
        return np.zeros((matrix.shape[0], 0))
    u, singular_values, _ = np.linalg.svd(matrix, full_matrices=False)
    keep = singular_values > tolerance * max(float(singular_values[0]), 1.0)
    return u[:, keep]


def _residualized_projectors(
    raw_bases: dict[str, np.ndarray], rank: int, order: Sequence[str]
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    accumulated = np.zeros((rank, 0))
    bases: dict[str, np.ndarray] = {}
    for component in order:
        raw = raw_bases[component]
        residual = raw - accumulated @ (accumulated.T @ raw)
        basis = _orthonormal_basis(residual)
        bases[component] = basis
        accumulated = _orthonormal_basis(np.concatenate((accumulated, basis), axis=1))
    residual_basis = _orthonormal_basis(np.eye(rank) - accumulated @ accumulated.T)
    bases["residual"] = residual_basis
    projectors = {name: basis @ basis.T for name, basis in bases.items()}
    return projectors, bases


def _environment_statistics(
    model: LinearLatentModel,
    batches: Sequence[EnvironmentBatch],
    whitening: Whitening,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    means, covariances, label_cross = [], [], []
    with torch.no_grad():
        for batch in batches:
            z = model.encode(batch.x).cpu().numpy()
            white = (z - whitening.mean) @ whitening.transform.T
            centered = white - white.mean(0)
            y = batch.y.cpu().numpy()
            means.append(white.mean(0))
            covariances.append(centered.T @ centered / max(len(white) - 1, 1))
            label_cross.append(centered.T @ (y - y.mean()) / max(len(y) - 1, 1))
    return np.stack(means), np.stack(covariances), np.stack(label_cross)


def _raw_semantic_bases(
    model: LinearLatentModel,
    batches: Sequence[EnvironmentBatch],
    whitening: Whitening,
    design_override: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    means, covariances, label_cross = _environment_statistics(model, batches, whitening)
    design = np.stack([batch.environment.design for batch in batches])
    if design_override is not None:
        design = np.asarray(design_override, dtype=float)
    regression = np.column_stack((np.ones(len(design)), design))
    mean_coefficients = np.linalg.lstsq(regression, means, rcond=None)[0]
    cross_coefficients = np.linalg.lstsq(regression, label_cross, rcond=None)[0]
    covariance_coefficients = np.linalg.lstsq(
        regression, covariances.reshape(len(covariances), -1), rcond=None
    )[0].reshape(7, whitening.rank, whitening.rank)
    covariance_vectors = []
    for coefficient in covariance_coefficients[5:7]:
        values, vectors = np.linalg.eigh(0.5 * (coefficient + coefficient.T))
        covariance_vectors.append(vectors[:, int(np.argmax(np.abs(values)))])
    return {
        "task": _orthonormal_basis(cross_coefficients[0]),
        "relation": _orthonormal_basis(cross_coefficients[1:3].T),
        "mean": _orthonormal_basis(mean_coefficients[3:5].T),
        "covariance": _orthonormal_basis(np.column_stack(covariance_vectors)),
    }


def _oracle_raw_bases(
    model: LinearLatentModel, scm: LatentSCM, whitening: Whitening
) -> dict[str, np.ndarray]:
    encoder = model.encoder.weight.detach().cpu().numpy()
    mapped = whitening.transform @ encoder
    task = mapped[:, BLOCKS["task"]] @ scm.loading @ scm.beta
    return {
        "task": _orthonormal_basis(task),
        "relation": _orthonormal_basis(mapped[:, BLOCKS["relation"]][:, :2]),
        "mean": _orthonormal_basis(mapped[:, BLOCKS["mean"]][:, :2]),
        "covariance": _orthonormal_basis(mapped[:, BLOCKS["covariance"]][:, :2]),
    }


def projector_overlap(left: np.ndarray, right: np.ndarray) -> float:
    rank = max(int(round(np.trace(left))), int(round(np.trace(right))), 1)
    return float(np.trace(left @ right) / rank)


def maximum_principal_angle(left: np.ndarray, right: np.ndarray) -> float:
    left_basis = _orthonormal_basis(left)
    right_basis = _orthonormal_basis(right)
    if left_basis.shape[1] == 0 or right_basis.shape[1] == 0:
        return float(np.pi / 2)
    singular_values = np.linalg.svd(left_basis.T @ right_basis, compute_uv=False)
    return float(np.arccos(np.clip(singular_values.min(), 0.0, 1.0)))


def projector_diagnostics(projectors: dict[str, np.ndarray]) -> dict[str, float]:
    names = list(COMPONENTS)
    identity = np.eye(next(iter(projectors.values())).shape[0])
    symmetry = max(np.linalg.norm(projectors[name] - projectors[name].T) for name in names)
    idempotence = max(
        np.linalg.norm(projectors[name] @ projectors[name] - projectors[name]) for name in names
    )
    orthogonality = max(
        np.linalg.norm(projectors[left] @ projectors[right])
        for left, right in combinations(names, 2)
    )
    completeness = np.linalg.norm(sum(projectors.values()) - identity)
    return {
        "projector_symmetry_error": float(symmetry),
        "projector_idempotence_error": float(idempotence),
        "projector_orthogonality_error": float(orthogonality),
        "projector_completeness_error": float(completeness),
    }


def fit_semantic_decomposition(
    model: LinearLatentModel,
    scm: LatentSCM,
    estimator_batches: Sequence[EnvironmentBatch],
    validation_batches: Sequence[EnvironmentBatch],
    *,
    permutation_seed: int = 0,
    order: Sequence[str] = ("task", "relation", "mean", "covariance"),
) -> SemanticDecomposition:
    with torch.no_grad():
        encoded = [model.encode(batch.x).cpu().numpy() for batch in estimator_batches]
    whitening = fit_whitening(encoded)
    raw = _raw_semantic_bases(model, estimator_batches, whitening)
    projectors, bases = _residualized_projectors(raw, whitening.rank, order)
    validation_raw = _raw_semantic_bases(model, validation_batches, whitening)
    validation_projectors, _ = _residualized_projectors(validation_raw, whitening.rank, order)
    oracle_raw = _oracle_raw_bases(model, scm, whitening)
    oracle_projectors, _ = _residualized_projectors(oracle_raw, whitening.rank, order)

    generator = np.random.default_rng(permutation_seed)
    permuted_design = np.stack([batch.environment.design for batch in estimator_batches])[
        generator.permutation(len(estimator_batches))
    ]
    permuted_raw = _raw_semantic_bases(
        model, estimator_batches, whitening, design_override=permuted_design
    )
    permuted_projectors, _ = _residualized_projectors(permuted_raw, whitening.rank, order)

    # Repeat the complete construction with the folds exchanged. The canonical
    # attribution projectors remain those from the preregistered A fold; only
    # validation diagnostics are symmetrically aggregated across A->B and B->A.
    with torch.no_grad():
        encoded_validation = [
            model.encode(batch.x).cpu().numpy() for batch in validation_batches
        ]
    reverse_whitening = fit_whitening(encoded_validation)
    reverse_raw = _raw_semantic_bases(model, validation_batches, reverse_whitening)
    reverse_projectors, reverse_bases = _residualized_projectors(
        reverse_raw, reverse_whitening.rank, order
    )
    reverse_validation_raw = _raw_semantic_bases(
        model, estimator_batches, reverse_whitening
    )
    reverse_validation_projectors, _ = _residualized_projectors(
        reverse_validation_raw, reverse_whitening.rank, order
    )
    reverse_oracle_raw = _oracle_raw_bases(model, scm, reverse_whitening)
    reverse_oracle_projectors, _ = _residualized_projectors(
        reverse_oracle_raw, reverse_whitening.rank, order
    )
    reverse_design = np.stack(
        [batch.environment.design for batch in validation_batches]
    )
    reverse_permuted_raw = _raw_semantic_bases(
        model,
        validation_batches,
        reverse_whitening,
        design_override=reverse_design[generator.permutation(len(validation_batches))],
    )
    reverse_permuted_projectors, _ = _residualized_projectors(
        reverse_permuted_raw, reverse_whitening.rank, order
    )

    diagnostics = projector_diagnostics(projectors)
    for component in COMPONENTS:
        diagnostics[f"crossfit_overlap_{component}"] = float(
            np.mean(
                [
                    projector_overlap(
                        projectors[component], validation_projectors[component]
                    ),
                    projector_overlap(
                        reverse_projectors[component],
                        reverse_validation_projectors[component],
                    ),
                ]
            )
        )
        diagnostics[f"oracle_recovery_{component}"] = float(
            np.mean(
                [
                    projector_overlap(
                        projectors[component], oracle_projectors[component]
                    ),
                    projector_overlap(
                        reverse_projectors[component],
                        reverse_oracle_projectors[component],
                    ),
                ]
            )
        )
        diagnostics[f"oracle_angle_{component}"] = float(
            np.mean(
                [
                    maximum_principal_angle(
                        bases[component], _orthonormal_basis(oracle_projectors[component])
                    ),
                    maximum_principal_angle(
                        reverse_bases[component],
                        _orthonormal_basis(reverse_oracle_projectors[component]),
                    ),
                ]
            )
        )
        diagnostics[f"permuted_recovery_{component}"] = float(
            np.mean(
                [
                    projector_overlap(
                        permuted_projectors[component], oracle_projectors[component]
                    ),
                    projector_overlap(
                        reverse_permuted_projectors[component],
                        reverse_oracle_projectors[component],
                    ),
                ]
            )
        )

    canonical_energy_reference = projectors
    order_deviations = []
    for alternate in permutations(("task", "relation", "mean", "covariance")):
        alternate_projectors, _ = _residualized_projectors(raw, whitening.rank, alternate)
        order_deviations.append(
            max(
                np.linalg.norm(alternate_projectors[name] - canonical_energy_reference[name])
                for name in COMPONENTS
            )
        )
        reverse_alternate_projectors, _ = _residualized_projectors(
            reverse_raw, reverse_whitening.rank, alternate
        )
        order_deviations.append(
            max(
                np.linalg.norm(
                    reverse_alternate_projectors[name] - reverse_projectors[name]
                )
                for name in COMPONENTS
            )
        )
    diagnostics["maximum_order_projector_deviation"] = float(max(order_deviations))
    diagnostics["semantic_identified"] = float(
        diagnostics["projector_orthogonality_error"] < 1e-6
        and diagnostics["projector_completeness_error"] < 1e-6
        and min(diagnostics[f"crossfit_overlap_{name}"] for name in COMPONENTS[:-1]) >= 0.5
        and min(diagnostics[f"oracle_recovery_{name}"] for name in COMPONENTS[:-1]) >= 0.5
        and np.mean(
            [diagnostics[f"oracle_recovery_{name}"] for name in COMPONENTS[:-1]]
        )
        > np.mean([diagnostics[f"permuted_recovery_{name}"] for name in COMPONENTS[:-1]]) + 0.1
        and diagnostics["maximum_order_projector_deviation"] <= 0.25
    )
    return SemanticDecomposition(
        whitening=whitening,
        projectors=projectors,
        bases=bases,
        raw_bases=raw,
        oracle_projectors=oracle_projectors,
        diagnostics=diagnostics,
    )


def latent_population_moments(
    model: LinearLatentModel,
    scm: LatentSCM,
    environment: InterventionEnvironment,
    whitening: Whitening,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    x_mean, x_second, x_cross, y_second = input_moments(scm, environment)
    encoder = model.encoder.weight.detach().cpu().numpy()
    z_mean_original = encoder @ x_mean
    z_second_original = encoder @ x_second @ encoder.T
    z_cross_original = encoder @ x_cross
    transform = whitening.transform
    center = whitening.mean
    z_mean = transform @ (z_mean_original - center)
    z_second = transform @ (
        z_second_original
        - np.outer(z_mean_original, center)
        - np.outer(center, z_mean_original)
        + np.outer(center, center)
    ) @ transform.T
    z_cross = transform @ z_cross_original
    return z_mean, z_second, z_cross, y_second


def mixture_population_moments(
    model: LinearLatentModel,
    scm: LatentSCM,
    environments: Sequence[InterventionEnvironment],
    whitening: Whitening,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    values = [latent_population_moments(model, scm, environment, whitening) for environment in environments]
    return tuple(np.mean([value[index] for value in values], axis=0) for index in range(4))  # type: ignore[return-value]


def whitened_head(
    model: LinearLatentModel, whitening: Whitening
) -> tuple[np.ndarray, float]:
    head = model.head.weight.detach().cpu().numpy().reshape(-1)
    white_head = whitening.inverse.T @ head
    white_bias = float(model.head.bias.detach()) + float(head @ whitening.mean)
    return white_head, white_bias


def population_risk_from_moments(
    head: np.ndarray,
    bias: float,
    moments: tuple[np.ndarray, np.ndarray, np.ndarray, float],
) -> float:
    mean, second, cross, y_second = moments
    return float(
        head @ second @ head
        + 2.0 * bias * head @ mean
        + bias**2
        - 2.0 * head @ cross
        + y_second
    )


def component_risk_accounting(
    model: LinearLatentModel,
    scm: LatentSCM,
    source: Sequence[InterventionEnvironment],
    target: InterventionEnvironment,
    decomposition: SemanticDecomposition,
) -> dict[str, float]:
    source_moments_value = mixture_population_moments(
        model, scm, source, decomposition.whitening
    )
    target_moments_value = latent_population_moments(
        model, scm, target, decomposition.whitening
    )
    head, bias = whitened_head(model, decomposition.whitening)
    delta_mean = target_moments_value[0] - source_moments_value[0]
    delta_second = target_moments_value[1] - source_moments_value[1]
    # Absorb the fixed intercept's mean-shift term into an adjusted cross moment.
    delta_cross = (
        target_moments_value[2] - source_moments_value[2] - bias * delta_mean
    )
    source_risk = population_risk_from_moments(head, bias, source_moments_value)
    target_risk = population_risk_from_moments(head, bias, target_moments_value)
    rows: dict[str, float] = {
        "source_population_risk": source_risk,
        "target_population_risk": target_risk,
        "target_transport": target_risk - source_risk,
    }
    projected_heads = {
        name: decomposition.projectors[name] @ head for name in COMPONENTS
    }
    for name, value in projected_heads.items():
        rows[f"head_energy_{name}"] = float(value @ value)
        rows[f"main_{name}"] = float(
            value @ delta_second @ value - 2.0 * value @ delta_cross
        )
        rows[f"shapley_{name}"] = rows[f"main_{name}"]
    interaction_sum = 0.0
    for left, right in combinations(COMPONENTS, 2):
        interaction = float(
            2.0 * projected_heads[left] @ delta_second @ projected_heads[right]
        )
        rows[f"interaction_{left}_{right}"] = interaction
        rows[f"shapley_{left}"] += 0.5 * interaction
        rows[f"shapley_{right}"] += 0.5 * interaction
        interaction_sum += interaction
    accounted = sum(rows[f"main_{name}"] for name in COMPONENTS) + interaction_sum
    rows["interaction_total"] = interaction_sum
    rows["accounted_transport"] = accounted
    rows["accounting_residual"] = rows["target_transport"] - accounted
    return rows


def conservative_component_bound(
    model: LinearLatentModel,
    scm: LatentSCM,
    source: Sequence[InterventionEnvironment],
    decomposition: SemanticDecomposition,
    *,
    relation_budget: float = 2.0,
    mean_budget: float = 1.6,
    covariance_budget: float = 1.8,
) -> dict[str, float]:
    """A target-independent population bound over the declared intervention box.

    This uses a conservative global second-moment envelope. It is valid but is
    not an Omega-to-component bridge.
    """
    encoder = model.encoder.weight.detach().cpu().numpy()
    source_moments_x = [input_moments(scm, environment) for environment in source]
    source_second_norm = max(np.linalg.norm(value[1], 2) for value in source_moments_x)
    source_cross_norm = max(np.linalg.norm(value[2]) for value in source_moments_x)
    max_relation = np.linalg.norm(scm.relation_base, 2) + scm.relation_amplitude * relation_budget
    target_trace_bound = (
        np.trace(scm.loading @ scm.loading.T) + 3 * scm.sigma_u
        + 3 * max_relation**2 + 3 * scm.sigma_relation
        + 3 * scm.sigma_mean + 2 * (scm.mean_amplitude * mean_budget) ** 2
        + 3 * scm.sigma_covariance * np.exp(scm.covariance_log_amplitude * covariance_budget)
        + 3 * scm.sigma_residual
    )
    target_cross_bound = np.linalg.norm(scm.loading @ scm.beta) + max_relation * np.linalg.norm(scm.beta)
    map_norm = np.linalg.norm(decomposition.whitening.transform @ encoder, 2)
    rho_sigma = map_norm**2 * (target_trace_bound + source_second_norm)
    rho_cross = map_norm * (target_cross_bound + source_cross_norm)
    head, _ = whitened_head(model, decomposition.whitening)
    norms = {
        name: np.linalg.norm(decomposition.projectors[name] @ head) for name in COMPONENTS
    }
    diagonal = sum(2.0 * rho_cross * value + rho_sigma * value**2 for value in norms.values())
    interaction = sum(
        2.0 * rho_sigma * norms[left] * norms[right]
        for left, right in combinations(COMPONENTS, 2)
    )
    return {
        "rho_cross_global": float(rho_cross),
        "rho_second_global": float(rho_sigma),
        "component_bound": float(diagonal + interaction),
        "omega_component_bound_status": "NO OMEGA-TO-COMPONENT BOUND",
    }
