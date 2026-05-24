"""Rouche zero-counting and real-polynomial positivity certificates for paths.

Certificate pipeline for a single rival class margin g_j(t) on [0,1]:

  1. Fit polynomial surrogate Q via Chebyshev nodes (see surrogate.py).
  2. Estimate approximation error  eps = max|g_j(t) - Q(t)|.
  3. Real cert:   min_{t in [0,1]} Q(t) > eps  =>  g_j(t) > 0 everywhere.
  4. Rouche cert: Q zero-free in domain D containing [0,1]  +  Q(0) > eps
                  =>  Q > 0 on [0,1]  =>  g_j > 0 everywhere.
  5. Dense truth: sample g_j directly on fine grid (ground truth, not a cert).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from roche.surrogate import eval_poly_complex

ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]


def exact_polynomial_minimum(
    poly: np.poly1d,
    interval: tuple[float, float] = (0.0, 1.0),
) -> float:
    """Exact minimum of a polynomial on a closed interval.

    Evaluates at endpoints plus all real critical points (derivative roots)
    that lie in the interior.  Returns the true minimum, not a grid estimate.
    """
    a, b = interval
    candidates = [float(np.real(np.polyval(poly.coeffs, a))),
                  float(np.real(np.polyval(poly.coeffs, b)))]
    if poly.order >= 1:
        dpoly = poly.deriv()
        roots = np.roots(dpoly.coeffs)
        for r in roots:
            if abs(r.imag) < 1e-10 and a < r.real < b:
                candidates.append(float(np.real(np.polyval(poly.coeffs, r.real))))
    return float(np.min(candidates))


@dataclass(frozen=True)
class PathCertResult:
    method: str          # 'real_poly' | 'rouche' | 'dense'
    certified: bool
    margin: float        # min Q(t) - eps (real) or min Q(t) (dense)
    approx_error: float  # surrogate eps (0 for dense)
    rouche_margin: float | None  # Rouche condition slack; None for real/dense


def rectangular_contour(
    n_pts: int = 200,
    width: float = 0.3,
    x_lo: float = 0.0,
    x_hi: float = 1.0,
) -> ComplexArray:
    """Rectangle contour enclosing [x_lo, x_hi] with half-height width."""
    a, b, h = x_lo - width, x_hi + width, width
    n4 = n_pts // 4
    bottom = np.linspace(a,  b,  n4, endpoint=False) + (-1j * h)
    right  = np.linspace(b,  b,  n4, endpoint=False) + 1j * np.linspace(-h, h, n4, endpoint=False)
    top    = np.linspace(b,  a,  n4, endpoint=False) + (1j * h)
    left   = np.linspace(a,  a,  n4, endpoint=False) + 1j * np.linspace(h, -h, n4, endpoint=False)
    return np.concatenate([bottom, right, top, left]).astype(np.complex128)


def certify_real(
    poly: np.poly1d,
    epsilon: float,
) -> PathCertResult:
    """Real certificate: exact min_{t in [0,1]} Q(t) > eps.

    Uses derivative-root exact minimisation, not grid sampling.
    """
    min_q = exact_polynomial_minimum(poly, (0.0, 1.0))
    margin = min_q - epsilon
    return PathCertResult(
        method="real_poly",
        certified=margin > 0.0,
        margin=margin,
        approx_error=epsilon,
        rouche_margin=None,
    )


def certify_real_grid(
    poly: np.poly1d,
    epsilon: float,
    n_check: int = 500,
) -> PathCertResult:
    """Grid-sampled real certificate (diagnostic only, not exact)."""
    t = np.linspace(0.0, 1.0, n_check)
    q_vals = np.real(np.polyval(poly.coeffs, t))
    min_q = float(np.min(q_vals))
    margin = min_q - epsilon
    return PathCertResult(
        method="real_poly_grid",
        certified=margin > 0.0,
        margin=margin,
        approx_error=epsilon,
        rouche_margin=None,
    )


def certify_rouche(
    poly: np.poly1d,
    epsilon: float,
    contour_pts: ComplexArray,
) -> PathCertResult:
    """Rouche certificate using constant reference c = centroid of Q on contour.

    Rouche condition: |c| > |Q(z) - c| for all z on boundary  =>  Q zero-free in D.
    Sign fixed by Q(0) > eps.  Combined => g_j(t) > 0 on [0,1].
    """
    q_on_contour = eval_poly_complex(poly, contour_pts)          # complex
    c = np.mean(q_on_contour)                                    # complex centroid
    deviations = np.abs(q_on_contour - c)
    rouche_margin = float(np.abs(c) - np.max(deviations))
    q0 = float(np.real(np.polyval(poly.coeffs, 0.0)))
    sign_ok = q0 > epsilon
    certified = (rouche_margin > 0.0) and sign_ok
    real_margin = q0 - epsilon
    return PathCertResult(
        method="rouche",
        certified=certified,
        margin=real_margin,
        approx_error=epsilon,
        rouche_margin=rouche_margin,
    )


def certify_dense(
    g_fn: callable,
    n_check: int = 1000,
) -> PathCertResult:
    """Ground truth by dense sampling of actual g_j(t). Not a certificate."""
    t = np.linspace(0.0, 1.0, n_check)
    g_vals = np.array([g_fn(t_) for t_ in t], dtype=np.float64)
    min_g = float(np.min(g_vals))
    return PathCertResult(
        method="dense",
        certified=min_g > 0.0,
        margin=min_g,
        approx_error=0.0,
        rouche_margin=None,
    )


def certify_path(
    g_fn: callable,
    degree: int = 8,
    contour_width: float = 0.3,
    n_contour: int = 200,
    n_nodes: int | None = None,
    n_error_check: int = 500,
) -> dict[str, PathCertResult]:
    """Run all three methods on a single margin function g_fn:[0,1]->R.

    Returns dict with keys 'real_poly', 'rouche', 'dense'.
    """
    from roche.surrogate import fit_surrogate
    poly, eps = fit_surrogate(g_fn, degree, n_nodes=n_nodes, n_error_check=n_error_check)
    contour = rectangular_contour(n_contour, contour_width)
    return {
        "real_poly": certify_real(poly, eps),
        "rouche":    certify_rouche(poly, eps, contour),
        "dense":     certify_dense(g_fn),
    }
