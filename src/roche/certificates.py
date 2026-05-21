"""Contour certificates for Schur stability.

The functions here are intentionally small and transparent. They are starter
implementations for Paper 1, not final high-performance routines.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import det, eigvals, norm, solve

ComplexMatrix = NDArray[np.complex128]
FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class GridCertificateResult:
    """Result of evaluating a contour certificate on a finite grid."""

    certified: bool
    min_margin: float
    lipschitz_bound: float | None
    num_points: int
    criterion_value: float | None
    method: str


def unit_circle_grid(num_points: int, radius: float = 1.0) -> NDArray[np.complex128]:
    """Return equispaced points on a circle."""
    if num_points < 4:
        raise ValueError("num_points must be at least 4")
    theta = np.linspace(0.0, 2.0 * np.pi, num_points, endpoint=False)
    return (radius * np.exp(1j * theta)).astype(np.complex128)


def spectral_radius(a: ComplexMatrix) -> float:
    """Return the spectral radius of a square matrix."""
    return float(np.max(np.abs(eigvals(a))))


def is_schur_stable(a: ComplexMatrix, tol: float = 1e-10) -> bool:
    """Return True when all eigenvalues are strictly inside the unit disk."""
    return spectral_radius(a) < 1.0 - tol


def characteristic_value(a: ComplexMatrix, z: complex) -> complex:
    """Compute det(zI - A)."""
    n = a.shape[0]
    return complex(det(z * np.eye(n, dtype=np.complex128) - a))


def determinant_margin(
    a: ComplexMatrix,
    a0: ComplexMatrix,
    z: complex,
) -> float:
    """Rouché determinant margin at a contour point.

    Positive margin means the scalar Rouché inequality holds at this point:
    |p0(z)| > |pA(z) - p0(z)|.
    """
    p = characteristic_value(a, z)
    p0 = characteristic_value(a0, z)
    return float(abs(p0) - abs(p - p0))


def determinant_margins(
    a: ComplexMatrix,
    a0: ComplexMatrix,
    points: NDArray[np.complex128],
) -> FloatArray:
    """Evaluate determinant margins on contour points."""
    return np.array([determinant_margin(a, a0, z) for z in points], dtype=np.float64)


def resolvent_quantity(
    a: ComplexMatrix,
    a0: ComplexMatrix,
    z: complex,
    ord: int | float | str | None = 2,
) -> float:
    """Compute ||(zI - A0)^(-1)(A0 - A)||.

    The resolvent certificate holds on the contour when this quantity is < 1.
    """
    n = a.shape[0]
    identity = np.eye(n, dtype=np.complex128)
    perturbation = a0 - a
    x = solve(z * identity - a0, perturbation, assume_a="gen")
    return float(norm(x, ord=ord))


def resolvent_margin(
    a: ComplexMatrix,
    a0: ComplexMatrix,
    z: complex,
    ord: int | float | str | None = 2,
) -> float:
    """Return 1 - ||(zI - A0)^(-1)(A0 - A)|| at a contour point."""
    return 1.0 - resolvent_quantity(a, a0, z, ord=ord)


def resolvent_margins(
    a: ComplexMatrix,
    a0: ComplexMatrix,
    points: NDArray[np.complex128],
    ord: int | float | str | None = 2,
) -> FloatArray:
    """Evaluate resolvent margins on contour points."""
    return np.array([resolvent_margin(a, a0, z, ord=ord) for z in points], dtype=np.float64)


def finite_difference_lipschitz(values: FloatArray, period: float = 2.0 * np.pi) -> float:
    """Estimate a periodic Lipschitz constant from sampled values.

    This is an estimate, not a proof. For formal certification, pass a rigorous
    upper bound into `grid_certificate`.
    """
    if values.ndim != 1 or len(values) < 4:
        raise ValueError("values must be a one-dimensional array of length >= 4")
    step = period / len(values)
    diffs = np.abs(np.roll(values, -1) - values) / step
    return float(np.max(diffs))


def grid_certificate(
    margins: FloatArray,
    lipschitz_bound: float | None = None,
    method: str = "unknown",
) -> GridCertificateResult:
    """Apply the deterministic finite-grid contour criterion.

    If a rigorous Lipschitz bound L is provided and

        min_k m(theta_k) > pi L / N,

    then m(theta) > 0 on the full circle.

    If no Lipschitz bound is provided, this returns a sampling-only result where
    `certified` simply means all sampled margins are positive. That is useful for
    experiments but is not a mathematical certificate for the continuum.
    """
    if margins.ndim != 1 or len(margins) < 4:
        raise ValueError("margins must be a one-dimensional array of length >= 4")
    min_margin = float(np.min(margins))
    n = len(margins)
    if lipschitz_bound is None:
        return GridCertificateResult(
            certified=min_margin > 0.0,
            min_margin=min_margin,
            lipschitz_bound=None,
            num_points=n,
            criterion_value=None,
            method=method,
        )
    criterion = np.pi * float(lipschitz_bound) / n
    return GridCertificateResult(
        certified=min_margin > criterion,
        min_margin=min_margin,
        lipschitz_bound=float(lipschitz_bound),
        num_points=n,
        criterion_value=criterion,
        method=method,
    )


def certify_on_unit_circle(
    a: ComplexMatrix,
    a0: ComplexMatrix,
    num_points: int = 512,
    method: Literal["determinant", "resolvent"] = "resolvent",
    lipschitz_bound: float | None = None,
) -> GridCertificateResult:
    """Evaluate a determinant or resolvent certificate on the unit circle."""
    points = unit_circle_grid(num_points)
    if method == "determinant":
        margins = determinant_margins(a, a0, points)
    elif method == "resolvent":
        margins = resolvent_margins(a, a0, points)
    else:
        raise ValueError(f"unknown method: {method}")
    return grid_certificate(margins, lipschitz_bound=lipschitz_bound, method=method)
