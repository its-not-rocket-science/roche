"""Tests for GridCertificateResult.validation_level and analytic Lipschitz bound."""
import numpy as np
import pytest
from roche.certificates import (
    GridCertificateResult,
    grid_certificate,
    resolvent_lipschitz_bound_analytic_diagonal,
    resolvent_margins,
    unit_circle_grid,
)


def test_validation_level_sampled_only():
    margins = np.array([0.5, 0.6, 0.7, 0.8])
    result = grid_certificate(margins)
    assert result.validation_level == "sampled_only"
    assert result.certified is True


def test_validation_level_numerical_lipschitz():
    margins = np.array([0.5, 0.6, 0.7, 0.8])
    result = grid_certificate(margins, lipschitz_bound=1.0)
    assert result.validation_level == "numerical_lipschitz"


def test_validation_level_analytic_lipschitz():
    margins = np.array([0.5, 0.6, 0.7, 0.8])
    result = grid_certificate(margins, lipschitz_bound=1.0, lipschitz_analytic=True)
    assert result.validation_level == "analytic_lipschitz"


def _make_diagonal(n=4, rho=0.6):
    rng = np.random.default_rng(42)
    radii = rng.uniform(0.3, rho, n)
    angles = rng.uniform(0, 2*np.pi, n)
    return np.diag(radii * np.exp(1j * angles))


def test_analytic_diagonal_bound_is_analytic():
    a0 = _make_diagonal()
    a = _make_diagonal(rho=0.4)
    bound, is_analytic = resolvent_lipschitz_bound_analytic_diagonal(a, a0)
    assert is_analytic is True
    assert bound > 0


def test_analytic_bound_geq_numerical_estimate():
    """Analytic bound should be >= numerical estimate for diagonal A0."""
    n = 4
    a0 = _make_diagonal(n, rho=0.5)
    a = _make_diagonal(n, rho=0.3)
    analytic_bound, _ = resolvent_lipschitz_bound_analytic_diagonal(a, a0)
    from roche.certificates import resolvent_lipschitz_bound
    points = unit_circle_grid(512)
    numerical_bound = resolvent_lipschitz_bound(a, a0, points)
    # Analytic bound is conservative; should be >= numerical
    assert analytic_bound >= numerical_bound * 0.9, (
        f"Analytic {analytic_bound:.4f} < numerical {numerical_bound:.4f}"
    )


def test_analytic_bound_fallback_nondiagl():
    """Non-diagonal A0 falls back to numerical (is_analytic=False)."""
    n = 4
    rng = np.random.default_rng(7)
    a0 = rng.standard_normal((n, n)) * 0.1
    a0 = a0 * 0.3 / np.max(np.abs(np.linalg.eigvals(a0)))
    a = np.eye(n, dtype=complex) * 0.2
    bound, is_analytic = resolvent_lipschitz_bound_analytic_diagonal(a, a0)
    assert is_analytic is False
    assert bound > 0
