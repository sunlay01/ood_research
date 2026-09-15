"""Finite-dimensional admissible response-controller classes.

The counterfactual repairability layer works with the induced response map
``Pi : Y -> Z``.  A class is therefore represented by a finite collection of
response-space basis matrices, rather than by a parameter norm or a semantic
label such as ``representation``.  Parameter/Jacobian construction is kept as
an optional bridge at the bottom of this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from ..round3r_3e_recovery import operator_norm

Array = np.ndarray


def _matrix(value: Array, *, name: str) -> Array:
    result = np.asarray(value, dtype=float)
    if result.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result.copy()


def _is_invertible(value: Array, tolerance: float) -> bool:
    singular = np.linalg.svd(value, compute_uv=False)
    return bool(singular.size and singular[-1] > tolerance * max(1.0, singular[0]))


@dataclass(frozen=True)
class ResponseControllerClass:
    """A finite-dimensional linear class ``C subseteq L(Y, Z)``.

    ``basis[i]`` is a ``(dim_Z, dim_Y)`` response operator.  The class is the
    span of these matrices, optionally intersected with a response-space
    budget.  An empty basis is a valid zero class when ``response_shape`` is
    supplied.
    """

    basis: Sequence[Array]
    budget_type: str = "none"
    budget_radius: float = float("inf")
    response_shape: tuple[int, int] | None = None
    name: str = "controller"

    def __post_init__(self) -> None:
        raw = tuple(_matrix(value, name="basis element") for value in self.basis)
        shape = self.response_shape
        if raw:
            shape = raw[0].shape
            if any(value.shape != shape for value in raw):
                raise ValueError("all basis elements must have the same shape")
        elif shape is None:
            raise ValueError("response_shape is required for an empty basis")
        if len(shape) != 2 or any(int(size) <= 0 for size in shape):
            raise ValueError("response_shape must contain two positive dimensions")
        budget_type = str(self.budget_type).lower()
        if budget_type not in {"none", "operator", "frobenius"}:
            raise ValueError("budget_type must be 'none', 'operator', or 'frobenius'")
        radius = float(self.budget_radius)
        if budget_type == "none":
            radius = float("inf")
        elif not np.isfinite(radius) or radius < 0.0:
            raise ValueError("a bounded budget needs a finite non-negative radius")
        object.__setattr__(self, "basis", raw)
        object.__setattr__(self, "response_shape", (int(shape[0]), int(shape[1])))
        object.__setattr__(self, "budget_type", budget_type)
        object.__setattr__(self, "budget_radius", radius)

    @property
    def response_dimension(self) -> int:
        return int(self.response_shape[0])

    @property
    def source_dimension(self) -> int:
        return int(self.response_shape[1])

    @property
    def dimension(self) -> int:
        return len(self.basis)

    def response_operator(self, coeffs: Array | Sequence[float]) -> Array:
        """Construct ``Pi`` from coordinates in the declared basis."""
        values = np.asarray(coeffs, dtype=float).reshape(-1)
        if values.size != self.dimension:
            raise ValueError(f"expected {self.dimension} coefficients, got {values.size}")
        if not np.all(np.isfinite(values)):
            raise ValueError("controller coefficients must be finite")
        if not self.basis:
            return np.zeros(self.response_shape, dtype=float)
        return np.tensordot(values, np.stack(self.basis, axis=0), axes=(0, 0))

    def _basis_matrix(self) -> Array:
        if not self.basis:
            return np.empty((self.response_shape[0] * self.response_shape[1], 0))
        return np.stack([value.reshape(-1) for value in self.basis], axis=1)

    def coefficients_for(self, response_operator: Array) -> tuple[Array, Array, Array]:
        """Return least-squares coefficients, projection, and residual.

        The residual is returned as a matrix, so callers can inspect the exact
        response-space mismatch rather than only a scalar membership score.
        """
        value = _matrix(response_operator, name="response_operator")
        if value.shape != self.response_shape:
            raise ValueError("response_operator has incompatible shape")
        design = self._basis_matrix()
        if design.shape[1] == 0:
            coefficients = np.empty(0, dtype=float)
        else:
            coefficients = np.linalg.lstsq(design, value.reshape(-1), rcond=None)[0]
        projection = self.response_operator(coefficients)
        residual = value - projection
        return coefficients, projection, residual

    def membership_residual(self, response_operator: Array, *, relative: bool = True) -> float:
        """Return the response-space distance to the linear class span."""
        value = _matrix(response_operator, name="response_operator")
        _, _, residual = self.coefficients_for(value)
        norm = float(np.linalg.norm(residual, ord="fro"))
        if not relative:
            return norm
        return norm / max(1.0, float(np.linalg.norm(value, ord="fro")))

    def budget_value(self, response_operator: Array) -> float:
        value = _matrix(response_operator, name="response_operator")
        if value.shape != self.response_shape:
            raise ValueError("response_operator has incompatible shape")
        if self.budget_type == "none":
            return 0.0
        if self.budget_type == "operator":
            return operator_norm(value)
        return float(np.linalg.norm(value, ord="fro"))

    def budget_residual(self, response_operator: Array) -> float:
        """Positive slack of the response-space budget."""
        if self.budget_type == "none":
            return float("inf")
        return float(self.budget_radius - self.budget_value(response_operator))

    def contains(self, response_operator: Array, *, tolerance: float = 1e-9) -> bool:
        residual = self.membership_residual(response_operator)
        scale = max(1.0, float(np.linalg.norm(np.asarray(response_operator, dtype=float), ord="fro")))
        return bool(residual <= tolerance and self.budget_residual(response_operator) >= -tolerance * scale)

    def membership_audit(self, response_operator: Array, *, tolerance: float = 1e-9) -> dict[str, object]:
        value = _matrix(response_operator, name="response_operator")
        coefficients, projection, residual = self.coefficients_for(value)
        residual_norm = float(np.linalg.norm(residual, ord="fro"))
        value_norm = float(np.linalg.norm(value, ord="fro"))
        relative = residual_norm / max(1.0, value_norm)
        budget = self.budget_value(value)
        budget_residual = self.budget_radius - budget if self.budget_type != "none" else float("inf")
        scale = max(1.0, value_norm)
        return {
            "class_name": self.name,
            "in_span": bool(relative <= tolerance),
            "membership_residual": relative,
            "membership_residual_absolute": residual_norm,
            "coefficients": coefficients,
            "projection": projection,
            "budget_type": self.budget_type,
            "budget_radius": self.budget_radius,
            "budget_value": budget,
            "budget_residual": float(budget_residual),
            "budget_feasible": bool(budget_residual >= -tolerance * scale),
            "contains": bool(relative <= tolerance and budget_residual >= -tolerance * scale),
            "tolerance": float(tolerance),
        }

    def is_feasible(self, response_operator: Array, *, tolerance: float = 1e-9) -> bool:
        return bool(self.membership_audit(response_operator, tolerance=tolerance)["contains"])

    def subset_of(self, other: "ResponseControllerClass", *, tolerance: float = 1e-9) -> bool:
        """Numerically check linear-span and budget containment."""
        if self.response_shape != other.response_shape:
            return False
        if any(other.membership_residual(value, relative=False) > tolerance for value in self.basis):
            return False
        if other.budget_type == "none":
            return True
        if self.budget_type == "none":
            return False
        # A span inclusion alone is not sufficient for different gauges.  This
        # conservative check is exact for the common equal-gauge case.
        if self.budget_type != other.budget_type:
            return False
        return bool(self.budget_radius <= other.budget_radius + tolerance)

    def transported(self, *, source_transform: Array | None = None,
                    response_transform: Array | None = None,
                    tolerance: float = 1e-10,
                    name: str | None = None) -> "ResponseControllerClass":
        """Transport the class while preserving the induced response ``Pi O``.

        For ``Y' = S Y`` the response map is ``Pi' = Pi S^{-1}``.  A response
        transform is allowed only when it is orthogonal, because the current
        regret uses the Euclidean norm on ``Z``.
        """
        source = np.eye(self.source_dimension) if source_transform is None else _matrix(source_transform, name="source_transform")
        if source.shape != (self.source_dimension, self.source_dimension) or not _is_invertible(source, tolerance):
            raise ValueError("source_transform must be square and invertible")
        if self.budget_type != "none" and not np.allclose(
            source.T @ source, np.eye(self.source_dimension), atol=tolerance * 100
        ):
            raise ValueError(
                "a bounded Euclidean response budget needs an orthogonal source transport; "
                "transport a source metric explicitly before using a non-orthogonal basis change"
            )
        response = np.eye(self.response_dimension) if response_transform is None else _matrix(response_transform, name="response_transform")
        if response.shape != (self.response_dimension, self.response_dimension):
            raise ValueError("response_transform has incompatible shape")
        if not np.allclose(response.T @ response, np.eye(self.response_dimension), atol=tolerance * 100):
            raise ValueError("response_transform must be orthogonal for Euclidean regret")
        inverse_source = np.linalg.inv(source)
        basis = tuple(response @ value @ inverse_source for value in self.basis)
        return ResponseControllerClass(
            basis=basis,
            budget_type=self.budget_type,
            budget_radius=self.budget_radius,
            response_shape=self.response_shape,
            name=name or f"{self.name}:transported",
        )

    @classmethod
    def from_jacobian(cls, jacobian: Array, parameter_controller_basis: Iterable[Array], *,
                      budget_type: str = "none", budget_radius: float = float("inf"),
                      name: str = "jacobian-controller") -> "ResponseControllerClass":
        """Build an induced response class ``Pi = J_B L``.

        Each element of ``parameter_controller_basis`` is a parameter-space
        map ``L : Y -> Theta_B``.  This is the bridge used later to connect an
        actual local update to the response-space class.
        """
        jac = _matrix(jacobian, name="jacobian")
        induced = []
        parameter_dimension = jac.shape[1]
        for value in parameter_controller_basis:
            controller = _matrix(value, name="parameter controller basis")
            if controller.shape[0] != parameter_dimension:
                raise ValueError("parameter controller basis has incompatible parameter dimension")
            induced.append(jac @ controller)
        if not induced:
            raise ValueError("at least one parameter controller basis is required")
        return cls(induced, budget_type=budget_type, budget_radius=budget_radius, name=name)


def standard_response_basis(response_dimension: int, source_dimension: int) -> tuple[Array, ...]:
    """Return the canonical basis of all ``Z x Y`` response maps."""
    if response_dimension <= 0 or source_dimension <= 0:
        raise ValueError("dimensions must be positive")
    result = []
    for row in range(response_dimension):
        for column in range(source_dimension):
            basis = np.zeros((response_dimension, source_dimension), dtype=float)
            basis[row, column] = 1.0
            result.append(basis)
    return tuple(result)


__all__ = ["ResponseControllerClass", "standard_response_basis"]
