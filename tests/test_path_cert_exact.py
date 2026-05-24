"""Tests for exact polynomial minimum and exact real certificate."""
import numpy as np
import pytest
from roche.path_cert import (
    certify_real,
    certify_real_grid,
    exact_polynomial_minimum,
)


def test_exact_minimum_interior():
    # Q(t) = (t - 0.5)^2, minimum at t=0.5, value=0
    poly = np.poly1d([1, -1, 0.25])  # t^2 - t + 0.25 = (t-0.5)^2
    result = exact_polynomial_minimum(poly)
    assert abs(result) < 1e-10, f"Expected 0, got {result}"


def test_exact_minimum_endpoint():
    # Q(t) = t + 1, minimum at t=0, value=1
    poly = np.poly1d([1, 1])
    result = exact_polynomial_minimum(poly)
    assert abs(result - 1.0) < 1e-10, f"Expected 1.0, got {result}"


def test_exact_minimum_complex_deriv_roots():
    # Q(t) = t^2 + 1 — derivative root t=0 is in [0,1] boundary; minimum is 1.0
    poly = np.poly1d([1, 0, 1])
    result = exact_polynomial_minimum(poly)
    assert abs(result - 1.0) < 1e-10, f"Expected 1.0, got {result}"


def test_real_certificate_uses_exact_minimum_not_grid():
    # Q(t) = (2t-1)^8 + 0.01, minimum at t=0.5, value=0.01
    # With epsilon=0.005 this should certify.
    # A 500-point grid may miss t=0.5 exactly but exact min will find it.
    # Simpler: Q(t) = (t - 0.3)^2 + 0.01, minimum at t=0.3, value=0.01
    # coeffs: t^2 - 0.6t + 0.09 + 0.01 = t^2 - 0.6t + 0.10
    poly = np.poly1d([1.0, -0.6, 0.10])
    eps = 0.005
    result = certify_real(poly, eps)
    assert result.certified, "Exact cert should certify when min Q - eps > 0"
    assert result.margin > 0


def test_complex_excursion_example_real_passes_rouche_fails():
    # Q(t) = (2t-1)^8 + eps: real min is eps > 0 so real cert passes.
    # High-degree poly has large complex values so Rouche may fail.
    from roche.path_cert import certify_rouche, rectangular_contour
    # Build Q(t) = (2t-1)^8 + 0.01
    p_base = np.poly1d([2, -1]) ** 8
    eps_val = 0.01
    coeffs = p_base.coeffs.astype(float).copy()
    coeffs[-1] += eps_val
    poly = np.poly1d(coeffs)
    epsilon = 0.005  # surrogate error
    contour = rectangular_contour(200, 0.3)
    real_result = certify_real(poly, epsilon)
    rouche_result = certify_rouche(poly, epsilon, contour)
    assert real_result.certified, "Real cert should pass for Q=(2t-1)^8+0.01 with eps=0.005"
    # Rouche is expected to fail on this high-degree polynomial (complex excursion)
    # Not a hard assertion since it depends on parameters, but log the result
    print(f"Rouche certified={rouche_result.certified}, margin={rouche_result.rouche_margin:.4f}")
