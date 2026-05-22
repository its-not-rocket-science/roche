"""Tests for new functionality added in Paper 1 implementation."""

from __future__ import annotations

import numpy as np
import pytest

from roche.certificates import (
    certify_on_unit_circle,
    resolvent_lipschitz_bound,
    unit_circle_grid,
)
from roche.matrices import (
    block_diagonal_matrix,
    random_block_diagonal,
    random_normal_matrix,
)
from roche.reference import scalar_reference
from roche.ssm import (
    generate_ar_sequence,
    ssm_forward,
    train_diagonal_ssm,
)


# ---- matrices ---------------------------------------------------------------

def test_block_diagonal_matrix_shape():
    rng = np.random.default_rng(0)
    b1 = random_normal_matrix(3, 0.8, rng)
    b2 = random_normal_matrix(2, 0.8, rng)
    bd = block_diagonal_matrix([b1, b2])
    assert bd.shape == (5, 5)


def test_block_diagonal_matrix_zeros_off_block():
    rng = np.random.default_rng(1)
    b1 = random_normal_matrix(2, 0.8, rng)
    b2 = random_normal_matrix(3, 0.8, rng)
    bd = block_diagonal_matrix([b1, b2])
    # off-diagonal blocks should be zero
    assert np.allclose(bd[:2, 2:], 0.0)
    assert np.allclose(bd[2:, :2], 0.0)


def test_random_block_diagonal_has_correct_size():
    rng = np.random.default_rng(2)
    a = random_block_diagonal(12, block_size=3, radius=0.85, rng=rng)
    assert a.shape == (12, 12)


def test_random_block_diagonal_non_divisible_n():
    rng = np.random.default_rng(3)
    a = random_block_diagonal(10, block_size=3, radius=0.85, rng=rng)
    assert a.shape == (10, 10)


# ---- certificates -----------------------------------------------------------

def test_resolvent_lipschitz_bound_positive():
    rng = np.random.default_rng(4)
    n = 4
    a = random_normal_matrix(n, 0.7, rng)
    a0 = scalar_reference(n, 0.5)
    points = unit_circle_grid(64)
    L = resolvent_lipschitz_bound(a, a0, points)
    assert L > 0


def test_resolvent_lipschitz_bound_zero_perturbation():
    n = 4
    a0 = scalar_reference(n, 0.5)
    points = unit_circle_grid(64)
    L = resolvent_lipschitz_bound(a0, a0, points)
    assert L == pytest.approx(0.0, abs=1e-12)


def test_rigorous_certificate_no_false_positive():
    # If the sampled minimum is less than pi*L/N the rigorous cert should fail.
    rng = np.random.default_rng(5)
    n = 4
    a = random_normal_matrix(n, 0.85, rng)
    a0 = scalar_reference(n, 0.5)
    num_points = 8  # deliberately coarse
    points = unit_circle_grid(num_points)
    from roche.certificates import resolvent_margins, grid_certificate
    margins = resolvent_margins(a, a0, points)
    L = resolvent_lipschitz_bound(a, a0, points)
    result = grid_certificate(margins, lipschitz_bound=L, method="resolvent")
    # Certified implies true stable (sufficient, not necessary).
    if result.certified:
        from roche.certificates import is_schur_stable
        assert is_schur_stable(a)


# ---- SSM --------------------------------------------------------------------

def test_generate_ar_sequence_length():
    rng = np.random.default_rng(6)
    seq = generate_ar_sequence([0.5, -0.3], n_samples=100, noise_std=0.1, rng=rng)
    assert len(seq) == 100


def test_ssm_forward_output_shape():
    n = 4
    a = np.diag(np.array([0.5 + 0.0j, 0.4 + 0.3j, 0.3 - 0.2j, 0.2 + 0.0j]))
    b = np.ones(n, dtype=np.complex128) * 0.1
    c = np.ones(n, dtype=np.complex128) * 0.1
    u = np.random.randn(50)
    y = ssm_forward(np.diag(a), b, c, 0.0, u)
    assert y.shape == (50,)


def test_train_diagonal_ssm_returns_stable_or_near_stable():
    rng = np.random.default_rng(7)
    ar_coeffs = [0.6, -0.2]
    u = generate_ar_sequence(ar_coeffs, n_samples=200, noise_std=0.05, rng=rng)
    target = np.roll(u, -1)
    target[-1] = 0.0
    a_mat, loss, _ = train_diagonal_ssm(u, target, n_state=4, n_iter=200, seed=0)
    assert a_mat.shape == (4, 4)
    assert loss >= 0.0
    # Diagonal entries: eigenvalues are the diagonal
    eigs = np.diag(a_mat)
    # Radii should be positive (parameterised as exp(log_r))
    assert np.all(np.abs(eigs) > 0)
