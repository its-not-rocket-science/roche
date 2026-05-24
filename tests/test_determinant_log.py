"""Tests for log-scale determinant margins."""
import numpy as np
import pytest
from roche.certificates import determinant_margins, determinant_margins_log, unit_circle_grid


def _stable_random(n, rho=0.5, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    eigs = np.linalg.eigvals(a)
    a = a * (rho / np.max(np.abs(eigs)) * 0.98)
    return a.astype(np.complex128)


def test_log_margins_agree_sign_with_standard_small_n():
    """For small n, log margins and standard margins should agree on sign."""
    a = _stable_random(4)
    a0 = _stable_random(4, rho=0.3)
    points = unit_circle_grid(64)
    std_margins = determinant_margins(a, a0, points)
    log_margins, finite_mask = determinant_margins_log(a, a0, points)
    assert np.all(finite_mask), "All points should be finite for n=4"
    # Signs should agree
    assert np.all(np.sign(std_margins) == np.sign(log_margins)), (
        "Log and standard margins disagree on sign"
    )


def test_log_margins_finite_mask_all_true_small_n():
    a = _stable_random(6)
    a0 = _stable_random(6, rho=0.3)
    points = unit_circle_grid(32)
    _, finite_mask = determinant_margins_log(a, a0, points)
    assert np.all(finite_mask)
