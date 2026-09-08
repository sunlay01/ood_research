"""Representation-level risk decompositions.

The functions in this module operate on population quantities or on oracle
conditional estimates.  They do not infer conditional expectations from a
finite sample and therefore do not turn an empirical diagnostic into a
theorem.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor


ScalarFunction = Callable[[Tensor], Tensor]


def _check_tensor(value: Tensor, name: str) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.numel() == 0:
        raise ValueError(f"{name} must be non-empty")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values")
    return value


def _check_same_shape(*named_values: tuple[str, Tensor]) -> None:
    shapes = {tuple(_check_tensor(value, name).shape) for name, value in named_values}
    if len(shapes) != 1:
        details = ", ".join(f"{name}={tuple(value.shape)}" for name, value in named_values)
        raise ValueError(f"all inputs must have the same shape: {details}")


@dataclass(frozen=True)
class SquaredRiskDecomposition:
    """Squared-risk terms for one environment.

    ``noise`` is the irreducible term relative to ``m_x = E[Y|X]``;
    ``representation_insufficiency`` is the information lost by replacing X
    with Z; and ``head_mismatch`` is the regret of the chosen head relative
    to ``m_z = E[Y|Z]``.

    Exact equality holds at the population level when ``m_x`` and ``m_z`` are
    the stated conditional expectations.  ``residual`` is exposed so finite
    sample oracle checks can detect violations of that condition.
    """

    risk: Tensor
    noise: Tensor
    representation_insufficiency: Tensor
    head_mismatch: Tensor
    residual: Tensor

    @property
    def explained_sum(self) -> Tensor:
        return self.noise + self.representation_insufficiency + self.head_mismatch


@dataclass(frozen=True)
class NestedOracleDecomposition:
    """Sequential oracle-risk increments for cross-domain failure analysis.

    The cumulative quantities ``e_prime`` are deliberately kept alongside
    their increments.  Only the telescoping equality is unconditional; the
    three latter increments need not be non-negative for a particular
    train/test split.
    """

    e_prime: tuple[Tensor, Tensor, Tensor, Tensor]
    increments: tuple[Tensor, Tensor, Tensor, Tensor]
    residual: Tensor

    @property
    def training_underfitting(self) -> Tensor:
        return self.increments[0]

    @property
    def test_inseparability(self) -> Tensor:
        return self.increments[1]

    @property
    def training_test_misalignment(self) -> Tensor:
        return self.increments[2]

    @property
    def classifier_noninvariance(self) -> Tensor:
        return self.increments[3]

    @property
    def final_error(self) -> Tensor:
        return self.e_prime[3]


def nested_oracle_decomposition(
    training_error: Tensor,
    test_representation_oracle_error: Tensor,
    joint_representation_oracle_error: Tensor,
    learned_model_test_error: Tensor,
) -> NestedOracleDecomposition:
    """Build the four sequential DG failure increments.

    The arguments correspond to Galstyan-style cumulative metrics:

    ``e'_0``: learned model evaluated on training domains;
    ``e'_1``: best allowed head evaluated on test domains;
    ``e'_2``: head fitted on the train+test mixture, evaluated on test;
    ``e'_3``: learned model evaluated on test domains.

    This is an oracle/evaluation decomposition, not a claim that the
    increments are independent causal sources or source-only observables.
    ``e'_1`` and ``e'_2`` may use test labels and are therefore offline
    diagnostics only.
    """

    _check_same_shape(
        ("training_error", training_error),
        ("test_representation_oracle_error", test_representation_oracle_error),
        ("joint_representation_oracle_error", joint_representation_oracle_error),
        ("learned_model_test_error", learned_model_test_error),
    )
    e_prime = (
        training_error,
        test_representation_oracle_error,
        joint_representation_oracle_error,
        learned_model_test_error,
    )
    increments = (
        e_prime[0],
        e_prime[1] - e_prime[0],
        e_prime[2] - e_prime[1],
        e_prime[3] - e_prime[2],
    )
    residual = e_prime[3] - sum(increments)
    return NestedOracleDecomposition(
        e_prime=e_prime,
        increments=increments,
        residual=residual,
    )


def squared_risk_decomposition(
    y: Tensor,
    conditional_mean_x: Tensor,
    conditional_mean_z: Tensor,
    prediction: Tensor,
) -> SquaredRiskDecomposition:
    """Compute the conditional-expectation Pythagorean decomposition.

    The tensors may contain arbitrary leading dimensions but must have the
    same shape.  The returned residual is zero under the population identity
    and is useful as a numerical audit quantity for oracle simulations.
    """

    _check_same_shape(
        ("y", y),
        ("conditional_mean_x", conditional_mean_x),
        ("conditional_mean_z", conditional_mean_z),
        ("prediction", prediction),
    )
    risk = (y - prediction).square().mean()
    noise = (y - conditional_mean_x).square().mean()
    insufficiency = (conditional_mean_x - conditional_mean_z).square().mean()
    head_mismatch = (conditional_mean_z - prediction).square().mean()
    residual = risk - noise - insufficiency - head_mismatch
    return SquaredRiskDecomposition(
        risk=risk,
        noise=noise,
        representation_insufficiency=insufficiency,
        head_mismatch=head_mismatch,
        residual=residual,
    )


@dataclass(frozen=True)
class ProperLossDecomposition:
    """Regret decomposition for a differentiable strictly proper loss."""

    bayes_risk: Tensor
    risk: Tensor
    representation_insufficiency: Tensor
    head_mismatch: Tensor
    residual: Tensor

    @property
    def regret(self) -> Tensor:
        return self.risk - self.bayes_risk


def bregman_divergence(
    point: Tensor,
    reference: Tensor,
    potential: ScalarFunction,
    gradient: ScalarFunction,
) -> Tensor:
    """Return pointwise ``B_F(point, reference)`` for a convex potential F."""

    _check_same_shape(("point", point), ("reference", reference))
    values = potential(point) - potential(reference)
    linear = (point - reference) * gradient(reference)
    if values.shape != point.shape[:-1] or linear.shape != point.shape:
        raise ValueError(
            "potential must map (..., K) to (...) and gradient must map (..., K) to (..., K)"
        )
    return values - linear.sum(dim=-1)


def proper_loss_decomposition(
    conditional_prob_x: Tensor,
    conditional_prob_z: Tensor,
    prediction: Tensor,
    potential: ScalarFunction,
    gradient: ScalarFunction,
) -> ProperLossDecomposition:
    """Compute the proper-loss Bayes/Bregman decomposition.

    ``potential`` is ``F = -H`` where ``H`` is the Bayes envelope of the
    strictly proper loss.  At the population level

    ``R(q(Z)) = R* + E B_F(eta_X, eta_Z) + E B_F(eta_Z, q(Z))``.

    The conditional-probability simplex constraints are checked here because
    otherwise the Bregman terms need not have the interpretation of a proper
    scoring-rule regret.
    """

    _check_same_shape(
        ("conditional_prob_x", conditional_prob_x),
        ("conditional_prob_z", conditional_prob_z),
        ("prediction", prediction),
    )
    if conditional_prob_x.ndim < 2:
        raise ValueError("probability inputs must have shape (..., number_of_classes)")
    for name, probabilities in (
        ("conditional_prob_x", conditional_prob_x),
        ("conditional_prob_z", conditional_prob_z),
        ("prediction", prediction),
    ):
        if (probabilities < 0).any():
            raise ValueError(f"{name} must be non-negative")
        if not torch.allclose(
            probabilities.sum(dim=-1),
            torch.ones_like(probabilities[..., 0]),
            atol=1e-6,
            rtol=1e-6,
        ):
            raise ValueError(f"{name} must lie on the probability simplex")

    insufficiency = bregman_divergence(
        conditional_prob_x, conditional_prob_z, potential, gradient
    ).mean()
    head_mismatch = bregman_divergence(
        conditional_prob_z, prediction, potential, gradient
    ).mean()
    total_regret = bregman_divergence(
        conditional_prob_x, prediction, potential, gradient
    ).mean()
    bayes_risk = (-potential(conditional_prob_x)).mean()
    risk = bayes_risk + total_regret
    residual = total_regret - insufficiency - head_mismatch
    return ProperLossDecomposition(
        bayes_risk=bayes_risk,
        risk=risk,
        representation_insufficiency=insufficiency,
        head_mismatch=head_mismatch,
        residual=residual,
    )


@dataclass(frozen=True)
class TransportDecomposition:
    """Discrete form of the absolute-continuous/singular transport identity."""

    functional_shift: Tensor
    density_reweighting: Tensor
    singular_coverage: Tensor
    difference: Tensor
    residual: Tensor


def transport_decomposition(
    source_weights: Tensor,
    density_ratio: Tensor,
    source_integrand: Tensor,
    target_integrand_on_source: Tensor,
    singular_weights: Tensor,
    target_integrand_on_singular: Tensor,
    *,
    atol: float = 1e-6,
) -> TransportDecomposition:
    """Evaluate a discrete Lebesgue transport refinement.

    The inputs encode ``P_T = r P_S + P_T^perp``.  The first term measures a
    change in the integrand on shared support, the second changes only the
    source mass weighting, and the third is the target mass outside source
    support.  This identity is exact when the supplied measures and
    integrands are exact.
    """

    _check_same_shape(
        ("source_weights", source_weights),
        ("density_ratio", density_ratio),
        ("source_integrand", source_integrand),
        ("target_integrand_on_source", target_integrand_on_source),
    )
    _check_tensor(singular_weights, "singular_weights")
    _check_tensor(target_integrand_on_singular, "target_integrand_on_singular")
    if singular_weights.shape != target_integrand_on_singular.shape:
        raise ValueError("singular weights and integrands must have the same shape")
    if (source_weights < 0).any() or (density_ratio < 0).any() or (singular_weights < 0).any():
        raise ValueError("measure weights and density ratios must be non-negative")
    if not torch.allclose(source_weights.sum(), source_weights.new_tensor(1.0), atol=atol):
        raise ValueError("source_weights must sum to one")
    shared_mass = (source_weights * density_ratio).sum()
    singular_mass = singular_weights.sum()
    if not torch.allclose(shared_mass + singular_mass, source_weights.new_tensor(1.0), atol=atol):
        raise ValueError("shared and singular target masses must sum to one")

    functional_shift = (
        source_weights * density_ratio * (target_integrand_on_source - source_integrand)
    ).sum()
    density_reweighting = (source_weights * (density_ratio - 1.0) * source_integrand).sum()
    singular_coverage = (singular_weights * target_integrand_on_singular).sum()
    difference = (
        (source_weights * density_ratio * target_integrand_on_source).sum()
        + singular_coverage
        - (source_weights * source_integrand).sum()
    )
    residual = difference - functional_shift - density_reweighting - singular_coverage
    return TransportDecomposition(
        functional_shift=functional_shift,
        density_reweighting=density_reweighting,
        singular_coverage=singular_coverage,
        difference=difference,
        residual=residual,
    )


def regularization_path_derivative(
    gradient_of_error: Tensor,
    objective_hessian: Tensor,
    gradient_of_regularizer: Tensor,
) -> Tensor:
    """Compute the local implicit derivative ``dE/dlambda``.

    This is a conditional local result: the objective Hessian must be
    invertible at an isolated differentiable stationary point.
    """

    _check_tensor(gradient_of_error, "gradient_of_error")
    _check_tensor(objective_hessian, "objective_hessian")
    _check_tensor(gradient_of_regularizer, "gradient_of_regularizer")
    if gradient_of_error.ndim != 1 or gradient_of_regularizer.ndim != 1:
        raise ValueError("gradients must be one-dimensional vectors")
    if gradient_of_error.shape != gradient_of_regularizer.shape:
        raise ValueError("gradient vectors must have the same shape")
    if objective_hessian.shape != (gradient_of_error.numel(), gradient_of_error.numel()):
        raise ValueError("objective_hessian must be square with one row per parameter")
    try:
        response = torch.linalg.solve(objective_hessian, gradient_of_regularizer)
    except RuntimeError as error:
        raise ValueError("objective_hessian must be invertible") from error
    return -(gradient_of_error * response).sum()


def log_loss_potential(probabilities: Tensor) -> Tensor:
    """Return ``F(p) = sum p log p``, the negative log-loss Bayes envelope."""

    if (probabilities <= 0).any():
        raise ValueError("log-loss probabilities must be strictly positive")
    return (probabilities * probabilities.log()).sum(dim=-1)


def log_loss_potential_gradient(probabilities: Tensor) -> Tensor:
    """Gradient of :func:`log_loss_potential`."""

    if (probabilities <= 0).any():
        raise ValueError("log-loss probabilities must be strictly positive")
    return probabilities.log() + 1.0
