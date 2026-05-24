"""Priority 6: validate PyTorch reference_opt objectives against NumPy certificates."""
import numpy as np
import pytest
import torch
from roche.certificates import resolvent_margins, unit_circle_grid
from roche.reference_opt import _resolvent_margins_torch, _unit_circle

_DTYPE = torch.complex128
_RDTYPE = torch.float64


def _make_diag_matrix(n=4, rho=0.5, seed=0):
    rng = np.random.default_rng(seed)
    radii = rng.uniform(0.2, rho, n)
    angles = rng.uniform(0, 2 * np.pi, n)
    d = radii * np.exp(1j * angles)
    return np.diag(d)


def test_torch_diagonal_fast_path_matches_numpy():
    """Diagonal A: PyTorch fast path should match NumPy resolvent_margins."""
    n = 4
    K = 32
    a_np = _make_diag_matrix(n, rho=0.5, seed=1)
    a0_np = _make_diag_matrix(n, rho=0.3, seed=2)

    # NumPy
    points_np = unit_circle_grid(K)
    np_margins = resolvent_margins(a_np, a0_np, points_np)

    # PyTorch fast path (diagonal A passed as 1D)
    d_a = torch.tensor(np.diag(a_np), dtype=_DTYPE)     # (n,)
    d0 = torch.tensor(np.diag(a0_np), dtype=_DTYPE)     # (n,)
    points_t = _unit_circle(K)
    torch_margins = _resolvent_margins_torch(d_a, d0, points_t).detach().numpy()

    np.testing.assert_allclose(
        np_margins, torch_margins, atol=1e-10,
        err_msg="PyTorch diagonal fast path disagrees with NumPy resolvent_margins"
    )


def test_torch_general_svd_path_matches_numpy():
    """Non-diagonal A: PyTorch SVD path should match NumPy resolvent_margins."""
    n = 4
    K = 32
    rng = np.random.default_rng(5)
    # A is non-diagonal (random), A0 is diagonal
    a_full = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    a_full = (a_full * 0.4 / np.max(np.abs(np.linalg.eigvals(a_full)))).astype(np.complex128)
    a0_np = _make_diag_matrix(n, rho=0.3, seed=3)

    points_np = unit_circle_grid(K)
    np_margins = resolvent_margins(a_full, a0_np, points_np)

    # PyTorch SVD path (A passed as 2D)
    a_t = torch.tensor(a_full, dtype=_DTYPE)
    d0 = torch.tensor(np.diag(a0_np), dtype=_DTYPE)
    points_t = _unit_circle(K)
    torch_margins = _resolvent_margins_torch(a_t, d0, points_t).detach().numpy()

    np.testing.assert_allclose(
        np_margins, torch_margins, atol=1e-10,
        err_msg="PyTorch SVD path disagrees with NumPy resolvent_margins"
    )


def test_finite_difference_gradient_diagonal_n2():
    """Finite-difference gradient check for diagonal reference optimiser, n=2."""
    from roche.reference_opt import _softmin
    n = 2
    K = 16
    rng = np.random.default_rng(99)
    a_np = _make_diag_matrix(n, rho=0.4, seed=10)
    d_a = torch.tensor(np.diag(a_np), dtype=_DTYPE)
    points_t = _unit_circle(K)

    # Parameterize d0 via log_r_raw, phi
    d0_init_np = np.array([0.3 + 0j, 0.2 + 0.1j], dtype=np.complex128)
    r0 = np.abs(d0_init_np)
    phi0 = np.angle(d0_init_np)
    log_r_raw = torch.tensor(np.log(r0 / (1.0 - r0)), dtype=_RDTYPE, requires_grad=True)
    phi = torch.tensor(phi0, dtype=_RDTYPE, requires_grad=True)

    def objective(log_r, ph):
        r = torch.sigmoid(log_r) * 0.98
        d0 = torch.polar(r, ph).to(_DTYPE)
        margins = _resolvent_margins_torch(d_a, d0, points_t)
        return _softmin(margins, 20.0)

    # Autograd gradient
    val = objective(log_r_raw, phi)
    val.backward()
    grad_log_r = log_r_raw.grad.clone()

    # Finite-difference gradient for log_r_raw[0]
    eps_fd = 1e-5
    with torch.no_grad():
        lrr_plus = log_r_raw.clone(); lrr_plus[0] += eps_fd
        lrr_minus = log_r_raw.clone(); lrr_minus[0] -= eps_fd
        fd_grad = (objective(lrr_plus, phi) - objective(lrr_minus, phi)).item() / (2 * eps_fd)

    np.testing.assert_allclose(
        grad_log_r[0].item(), fd_grad, rtol=1e-4,
        err_msg="Autograd gradient disagrees with finite difference"
    )
