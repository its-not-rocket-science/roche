"""Chebyshev surrogate fitting for path margin functions.

Given a path gamma:[0,1]->R^d and margin function g_j(t) = f_c(gamma(t)) - f_j(gamma(t)),
fit a polynomial surrogate Q_j at Chebyshev nodes and estimate approximation error.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


def chebyshev_nodes(n: int, a: float = 0.0, b: float = 1.0) -> FloatArray:
    """n Chebyshev nodes of the first kind mapped to [a, b]."""
    k = np.arange(1, n + 1)
    nodes_11 = np.cos((2 * k - 1) * np.pi / (2 * n))   # in [-1, 1]
    return ((b - a) * (nodes_11 + 1.0) / 2.0 + a).astype(np.float64)


def fit_poly(t_nodes: FloatArray, g_vals: FloatArray, degree: int) -> np.poly1d:
    """Fit degree-d polynomial to (t_nodes, g_vals) via least squares."""
    coeffs = np.polyfit(t_nodes, g_vals, degree)
    return np.poly1d(coeffs)


def eval_poly_complex(poly: np.poly1d, z: ComplexArray) -> ComplexArray:
    """Evaluate polynomial at complex array z via np.polyval."""
    return np.polyval(poly.coeffs, z).astype(np.complex128)


def approx_error(
    poly: np.poly1d,
    g_vals_fine: FloatArray,
    t_fine: FloatArray | None = None,
) -> float:
    """Max |g(t) - poly(t)| on supplied fine grid (or evaluated poly values)."""
    if t_fine is not None:
        q_fine = np.real(np.polyval(poly.coeffs, t_fine))
    else:
        q_fine = np.real(np.polyval(poly.coeffs, np.linspace(0.0, 1.0, len(g_vals_fine))))
    return float(np.max(np.abs(g_vals_fine - q_fine)))


def fit_surrogate(
    g_fn: callable,
    degree: int,
    n_nodes: int | None = None,
    n_error_check: int = 500,
) -> tuple[np.poly1d, float]:
    """Fit Chebyshev surrogate and estimate approximation error.

    Returns (poly, epsilon) where epsilon = max |g(t) - poly(t)| on fine grid.
    """
    if n_nodes is None:
        n_nodes = max(degree + 4, 2 * degree)
    t_nodes = chebyshev_nodes(n_nodes)
    g_vals = np.array([g_fn(t) for t in t_nodes], dtype=np.float64)
    poly = fit_poly(t_nodes, g_vals, degree)
    t_fine = np.linspace(0.0, 1.0, n_error_check)
    g_fine = np.array([g_fn(t) for t in t_fine], dtype=np.float64)
    eps = approx_error(poly, g_fine, t_fine)
    return poly, eps
