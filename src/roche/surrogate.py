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


def _interval_horner(coeffs: FloatArray, lo: float, hi: float) -> tuple[float, float]:
    """Evaluate polynomial on interval [lo, hi] via Horner with interval arithmetic.

    Returns (a, b) such that p(t) ∈ [a, b] for all t ∈ [lo, hi].
    Uses natural interval extension (dependency problem causes O(h) overestimate
    per evaluation; narrow sub-intervals control this).
    """
    a = b = float(coeffs[0])
    for c in coeffs[1:]:
        prods = (a * lo, a * hi, b * lo, b * hi)
        a = min(prods) + float(c)
        b = max(prods) + float(c)
    return a, b


def verified_error_bound(
    error_coeffs: FloatArray,
    n_sub: int = 1000,
) -> float:
    """Rigorous upper bound on max|e(t)| for t ∈ [0,1] via interval subdivision.

    Args:
        error_coeffs: coefficients of e(t) = g_j(t) - Q_j(t), highest degree
                      first (np.poly1d.coeffs convention).
        n_sub:        number of uniform sub-intervals; more → tighter bound.

    Returns:
        eps_rig ≥ max_{t ∈ [0,1]} |e(t)|.
        (Rigorous modulo IEEE 754 rounding; use directed rounding for full
        formal correctness.)
    """
    h = 1.0 / n_sub
    eps_rig = 0.0
    for i in range(n_sub):
        lo = i * h
        v_lo, v_hi = _interval_horner(error_coeffs, lo, lo + h)
        if abs(v_lo) > eps_rig:
            eps_rig = abs(v_lo)
        if abs(v_hi) > eps_rig:
            eps_rig = abs(v_hi)
    return eps_rig


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
