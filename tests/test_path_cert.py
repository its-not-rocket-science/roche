"""Tests for surrogate fitting and path certificates (Paper 3)."""

import numpy as np
import pytest

from roche.surrogate import fit_surrogate, chebyshev_nodes
from roche.path_cert import (
    certify_real,
    certify_rouche,
    rectangular_contour,
)


def _build_poly8_excursion():
    """(2t-1)^8 + 0.1: positive on [0,1], huge off-axis growth at corners."""
    inner = np.poly1d([2.0, -1.0])
    p = np.poly1d([1.0])
    for _ in range(8):
        p = np.polymul(p, inner)
    coeffs = p.coeffs.copy()
    coeffs[-1] += 0.1
    return np.poly1d(coeffs)


# ---------------------------------------------------------------------------

def test_chebyshev_surrogate_recovers_low_degree_polynomial():
    """Enough nodes => surrogate exactly recovers low-degree polynomial."""
    coeffs = np.array([3.0, -2.0, 1.0])   # 3t^2 - 2t + 1
    g_fn = lambda t: float(np.polyval(coeffs, t))
    poly, eps = fit_surrogate(g_fn, degree=2, n_nodes=20, n_error_check=1000)
    assert eps < 1e-10


def test_real_cert_passes_when_min_exceeds_epsilon():
    """Real cert passes when min Q(t) - epsilon > 0."""
    # t^2 - t + 0.5 => min = 0.25 at t=0.5
    poly = np.poly1d([1.0, -1.0, 0.5])
    result = certify_real(poly, epsilon=0.1)
    assert result.certified
    assert result.margin > 0.0


def test_real_cert_fails_when_min_below_epsilon():
    """Real cert fails when min Q(t) < epsilon."""
    poly = np.poly1d([1.0, -1.0, 0.5])
    result = certify_real(poly, epsilon=0.3)   # min=0.25 < 0.3
    assert not result.certified
    assert result.margin < 0.0


def test_rouche_fails_on_positive_poly_with_large_complex_excursion():
    """(2t-1)^8 + 0.1 is positive on [0,1] but Rouché fails due to corner excursion.

    At corner z = -0.3 + 0.3i: |2z-1| = sqrt(1.6^2 + 0.6^2) = sqrt(2.92) ~ 1.71,
    so |(2z-1)^8| ~ 72.  The real certificate trivially passes (min = 0.1 >> 0).
    """
    poly = _build_poly8_excursion()
    contour = rectangular_contour(n_pts=400, width=0.3)

    # Sanity: polynomial is positive on [0,1]
    t_fine = np.linspace(0.0, 1.0, 1000)
    min_real = float(np.min(np.polyval(poly.coeffs, t_fine)))
    assert min_real > 0.05

    # Real cert passes
    real_result = certify_real(poly, epsilon=0.0)
    assert real_result.certified

    # Rouché fails due to complex excursion
    rouche_result = certify_rouche(poly, epsilon=0.0, contour_pts=contour)
    assert not rouche_result.certified
    assert rouche_result.rouche_margin is not None
    assert rouche_result.rouche_margin < 0.0


def test_rectangular_contour_shape_and_bounds():
    """Contour encloses [0,1] with correct real and imaginary ranges."""
    w = 0.25
    pts = rectangular_contour(n_pts=200, width=w)
    assert pts.shape == (200,)
    assert np.all(np.real(pts) >= -w - 1e-9)
    assert np.all(np.real(pts) <= 1.0 + w + 1e-9)
    assert np.all(np.abs(np.imag(pts)) <= w + 1e-9)
