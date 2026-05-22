"""Minimal diagonal SSM for Paper 1 Experiment 3.

Implements a complex-diagonal state-space model trained via L-BFGS-B
on a synthetic autoregressive prediction task.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from roche.certificates import spectral_radius

ComplexMatrix = NDArray[np.complex128]


def generate_ar_sequence(
    ar_coeffs: list[float],
    n_samples: int,
    noise_std: float,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Generate a stationary AR(p) sequence.

    ar_coeffs[k] is the coefficient at lag k+1. The process is assumed stable
    (caller is responsible for choosing coefficients with roots inside the unit
    circle). Warmup of len(ar_coeffs) steps is discarded.
    """
    p = len(ar_coeffs)
    warmup = 200
    total = n_samples + warmup
    x = np.zeros(total)
    x[:p] = rng.normal(0, 0.1, p)
    noise = rng.normal(0, noise_std, total)
    coeffs = np.asarray(ar_coeffs, dtype=np.float64)
    for t in range(p, total):
        x[t] = float(coeffs @ x[t - p : t][::-1]) + noise[t]
    return x[warmup:].astype(np.float64)


def _unpack(params: NDArray[np.float64], n: int) -> tuple:
    log_radii = params[:n]
    angles = params[n : 2 * n]
    b_re = params[2 * n : 3 * n]
    b_im = params[3 * n : 4 * n]
    c_re = params[4 * n : 5 * n]
    c_im = params[5 * n : 6 * n]
    d = params[6 * n]
    radii = np.exp(np.clip(log_radii, -20.0, 2.0))
    a = radii * np.exp(1j * angles)
    b = b_re + 1j * b_im
    c = c_re + 1j * c_im
    return a, b, c, d


def ssm_forward(
    a: NDArray[np.complex128],
    b: NDArray[np.complex128],
    c: NDArray[np.complex128],
    d: float,
    u_seq: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Run diagonal SSM forward pass. Returns real-valued output sequence."""
    T = len(u_seq)
    n = len(a)
    h = np.zeros(n, dtype=np.complex128)
    y = np.zeros(T)
    for t in range(T):
        h = a * h + b * u_seq[t]
        y[t] = 2.0 * float(np.real(c @ h)) + d * u_seq[t]
    return y


def _loss_and_grad(
    params: NDArray[np.float64],
    u_seq: NDArray[np.float64],
    target: NDArray[np.float64],
    n: int,
) -> tuple[float, NDArray[np.float64]]:
    """MSE loss with BPTT gradients for the diagonal complex SSM."""
    _bad = (1e30, np.zeros_like(params))
    with np.errstate(over="ignore", invalid="ignore"):
        a, b, c, d = _unpack(params, n)
        T = len(u_seq)

        # Forward pass, cache states
        H = np.zeros((T, n), dtype=np.complex128)
        h = np.zeros(n, dtype=np.complex128)
        for t in range(T):
            h = a * h + b * u_seq[t]
            if not np.isfinite(h).all():
                return _bad
            H[t] = h

        y = 2.0 * np.real(H @ c) + d * u_seq
        residual = y - target
        if not np.isfinite(residual).all():
            return _bad
        loss = float(np.mean(residual**2))
        if not np.isfinite(loss):
            return _bad
        scale = 2.0 / T

        # Backward pass
        dL_dy = scale * residual  # (T,) real

        # Gradient w.r.t. c: dL/dc = sum_t dL/dy_t * 2 * H[t] (complex)
        dc = 2.0 * (H.T @ dL_dy)  # (n,) complex
        dc_re = np.real(dc)
        dc_im = np.imag(dc)

        # Gradient w.r.t. d
        dd = float(dL_dy @ u_seq)

        # BPTT for a, b
        dh = np.zeros(n, dtype=np.complex128)
        da_sum = np.zeros(n, dtype=np.complex128)
        db_sum = np.zeros(n, dtype=np.complex128)
        for t in reversed(range(T)):
            # dL/dy_t = dL_dy[t]; dL/dh_t from output = dL_dy[t] * 2 * c (conj for Wirtinger)
            dh += dL_dy[t] * 2.0 * np.conj(c)
            h_prev = H[t - 1] if t > 0 else np.zeros(n, dtype=np.complex128)
            da_sum += dh * np.conj(h_prev)
            db_sum += dh * u_seq[t]
            dh = dh * np.conj(a)

        # a = radii * exp(i*angles), radii = exp(log_radii)
        # da/d_log_radii = radii * exp(i*angles) = a  => chain rule: dL/d_log_r = Re(dL/da * conj(a))
        # da/d_angles = i * radii * exp(i*angles) = i*a => dL/d_angle = Re(dL/da * conj(i*a)) = Im(dL/da * conj(a))
        da_log_r = np.real(da_sum * np.conj(a))
        da_angle = np.imag(da_sum * np.conj(a))
        db_re = np.real(db_sum)
        db_im = np.imag(db_sum)

        grad = np.concatenate([da_log_r, da_angle, db_re, db_im, dc_re, dc_im, [dd]])
        if not np.isfinite(grad).all():
            return _bad
        return loss, grad


def train_diagonal_ssm(
    u_seq: NDArray[np.float64],
    target_seq: NDArray[np.float64],
    n_state: int = 8,
    n_iter: int = 2000,
    seed: int = 0,
) -> tuple[ComplexMatrix, float, bool]:
    """Train a diagonal SSM on (u_seq -> target_seq) via L-BFGS-B.

    Returns (A_matrix, final_loss, converged).
    A_matrix is the diagonal transition matrix diag(a).
    """
    rng = np.random.default_rng(seed)
    # Initialise: log_radii near -0.7 (radii ~0.5), random angles
    x0 = np.concatenate([
        rng.normal(-0.7, 0.2, n_state),
        rng.uniform(0.0, 2.0 * np.pi, n_state),
        rng.normal(0.0, 0.1, 4 * n_state),
        [0.0],
    ])
    # Bounds: log_radii in [-10, 1.5] (radii up to ~4.5); angles unbounded;
    # b/c/d unbounded.
    log_r_bounds = [(-10.0, 1.5)] * n_state
    angle_bounds = [(None, None)] * n_state
    other_bounds = [(None, None)] * (4 * n_state + 1)
    bounds = log_r_bounds + angle_bounds + other_bounds

    result = minimize(
        _loss_and_grad,
        x0,
        args=(u_seq, target_seq, n_state),
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        options={"maxiter": n_iter, "ftol": 1e-12, "gtol": 1e-7},
    )
    a_vec, _, _, _ = _unpack(result.x, n_state)
    a_matrix = np.diag(a_vec)
    return a_matrix, float(result.fun), bool(result.success)


def diagonal_plus_lowrank_matrix(
    diag_a: NDArray[np.complex128],
    u: NDArray[np.complex128],
    v: NDArray[np.complex128],
) -> ComplexMatrix:
    """Return diag(diag_a) + u @ v (low-rank correction)."""
    return np.diag(diag_a) + u @ v


def ssm_transition_matrix(params: NDArray[np.float64], n: int) -> ComplexMatrix:
    """Extract the diagonal transition matrix A = diag(a) from parameter vector."""
    a, _, _, _ = _unpack(params, n)
    return np.diag(a)


def schur_stability_summary(a_matrix: ComplexMatrix) -> dict:
    rho = spectral_radius(a_matrix)
    return {
        "spectral_radius": rho,
        "schur_stable": rho < 1.0,
        "max_eigenvalue_abs": float(np.max(np.abs(np.linalg.eigvals(a_matrix)))),
    }
