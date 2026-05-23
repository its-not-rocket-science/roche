import numpy as np
import pytest
import torch
from roche.regularisers import lyapunov_penalty, make_contour_barrier, spectral_penalty
from roche.ssm import generate_ar_sequence, train_diagonal_ssm_adam


def _diag(radii):
    """Make complex diagonal tensor from real radii."""
    return torch.tensor(radii, dtype=torch.complex128)


# ---------------------------------------------------------------------------
# spectral_penalty
# ---------------------------------------------------------------------------

def test_spectral_penalty_zero_for_clearly_stable():
    a = _diag([0.3, 0.4, 0.2])
    p = spectral_penalty(a, margin=0.01)
    assert float(p) < 1e-6


def test_spectral_penalty_positive_near_boundary():
    a = _diag([0.995, 0.5, 0.3])  # 0.995 > 1 - 0.01 = 0.99 threshold
    p = spectral_penalty(a, margin=0.01)
    assert float(p) > 0.0


def test_spectral_penalty_positive_for_unstable():
    a = _diag([1.1, 0.5])
    p = spectral_penalty(a, margin=0.01)
    assert float(p) > 0.0


def test_spectral_penalty_gradient_flows():
    # 0.995 > threshold 0.99 → relu is in active region → nonzero gradient
    a = torch.tensor([0.995 + 0j, 0.5 + 0j], dtype=torch.complex128, requires_grad=True)
    p = spectral_penalty(a, margin=0.01)
    p.backward()
    assert a.grad is not None and not torch.all(a.grad == 0)


# ---------------------------------------------------------------------------
# lyapunov_penalty
# ---------------------------------------------------------------------------

def test_lyapunov_penalty_zero_for_clearly_stable():
    a = _diag([0.3, 0.4, 0.2])
    p = lyapunov_penalty(a, margin=0.01)
    assert float(p) < 1e-6


def test_lyapunov_penalty_positive_near_boundary():
    a = _diag([0.995, 0.5])
    p = lyapunov_penalty(a, margin=0.01)
    assert float(p) > 0.0


def test_lyapunov_penalty_penalises_all_near_unstable_modes():
    # All modes above threshold (|a|^2 > 0.99) → lyapunov sums all, spectral only max
    a = _diag([0.999, 0.998, 0.997])  # |a|^2 = [0.998, 0.996, 0.994] all > 0.99
    lp = float(lyapunov_penalty(a, margin=0.01))
    sp = float(spectral_penalty(a, margin=0.01))
    assert lp > sp


# ---------------------------------------------------------------------------
# contour_barrier
# ---------------------------------------------------------------------------

def test_contour_barrier_returns_callable():
    d0 = np.array([0.5 + 0j, 0.4 + 0.3j], dtype=np.complex128)
    barrier = make_contour_barrier(d0, num_contour_points=64)
    assert callable(barrier)


def test_contour_barrier_near_zero_for_good_margin():
    d0 = np.array([0.3 + 0j, 0.2 + 0.1j], dtype=np.complex128)
    barrier = make_contour_barrier(d0, num_contour_points=64, margin=0.01)
    # Same eigenvalues as d0 → large positive margin → barrier ≈ 0
    a = torch.tensor(d0, dtype=torch.complex128)
    p = float(barrier(a))
    assert p < 1e-4


def test_contour_barrier_positive_for_poor_margin():
    d0 = np.array([0.5 + 0j, 0.4 + 0j], dtype=np.complex128)
    barrier = make_contour_barrier(d0, num_contour_points=64, margin=0.01)
    # Push eigenvalues to 0.999 → resolvent margin at z=1 collapses below 0.01 → fires
    a = torch.tensor([0.999 + 0j, 0.999 + 0j], dtype=torch.complex128)
    p = float(barrier(a))
    assert p > 0.0


# ---------------------------------------------------------------------------
# train_diagonal_ssm_adam
# ---------------------------------------------------------------------------

def test_train_diagonal_ssm_adam_returns_stable():
    rng = np.random.default_rng(0)
    u = generate_ar_sequence([0.6, -0.2], 200, 0.1, rng)
    target = np.roll(u, -1); target[-1] = 0.0
    a_mat, task_hist, reg_hist = train_diagonal_ssm_adam(
        u, target, n_state=4, n_epochs=50, lr=1e-3, seed=0
    )
    rho = float(np.max(np.abs(np.linalg.eigvals(a_mat))))
    assert rho < 1.0


def test_train_diagonal_ssm_adam_with_spectral_regulariser():
    rng = np.random.default_rng(1)
    u = generate_ar_sequence([0.7, -0.3], 200, 0.1, rng)
    target = np.roll(u, -1); target[-1] = 0.0
    a_mat, task_hist, reg_hist = train_diagonal_ssm_adam(
        u, target, n_state=4, n_epochs=50, lr=1e-3,
        regulariser=spectral_penalty, reg_weight=0.1, seed=0,
    )
    assert len(task_hist) == 50
    assert len(reg_hist) == 50
    rho = float(np.max(np.abs(np.linalg.eigvals(a_mat))))
    assert rho < 1.0


def test_train_diagonal_ssm_adam_unconstrained_accepts_near_boundary_init():
    rng = np.random.default_rng(2)
    u = generate_ar_sequence([0.5, -0.1], 100, 0.1, rng)
    target = np.roll(u, -1); target[-1] = 0.0
    # Just check it runs without error; stability not guaranteed unconstrained
    a_mat, _, _ = train_diagonal_ssm_adam(
        u, target, n_state=4, n_epochs=20, lr=1e-3,
        seed=0, constrained=False, init_log_r=float(np.log(0.96)),
    )
    assert a_mat.shape == (4, 4)


def test_schur_stable_trainer_always_stable_at_high_lr():
    """schur_stable=True guarantees rho < 1 by construction regardless of lr."""
    rng = np.random.default_rng(7)
    u = generate_ar_sequence([0.5, -0.1], 200, 0.1, rng)
    target = np.roll(u, -1); target[-1] = 0.0
    a_mat, _, _ = train_diagonal_ssm_adam(
        u, target, n_state=4, n_epochs=30, lr=0.1,
        seed=0, constrained=False, init_log_r=float(np.log(0.96)),
        schur_stable=True,
    )
    rho = float(np.max(np.abs(np.linalg.eigvals(a_mat))))
    assert rho < 1.0, f"schur_stable trainer produced rho={rho:.4f} >= 1"


def test_rho_history_out_has_correct_length():
    rng = np.random.default_rng(3)
    u = generate_ar_sequence([0.4, -0.2], 150, 0.1, rng)
    target = np.roll(u, -1); target[-1] = 0.0
    rho_hist: list[float] = []
    train_diagonal_ssm_adam(
        u, target, n_state=4, n_epochs=25, lr=1e-3, seed=0,
        rho_history_out=rho_hist,
    )
    assert len(rho_hist) == 25
    assert all(r > 0 for r in rho_hist)
