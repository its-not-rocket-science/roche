"""Diagonal SSM training for Papers 1 and 2.

Paper 1: train_diagonal_ssm  — L-BFGS-B, no regularisation.
Paper 2: train_diagonal_ssm_adam — Adam, pluggable regulariser.

Implements a complex-diagonal state-space model trained via L-BFGS-B
on a synthetic autoregressive prediction task.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch
import torch.nn as nn
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


# ---------------------------------------------------------------------------
# Paper 2: Adam trainer with pluggable regulariser
# ---------------------------------------------------------------------------

class _DiagonalSSM(nn.Module):
    """Diagonal complex SSM as a PyTorch module.

    constrained=True (default): radii clamped to (0, ~4.5) via exp(clamp(log_r,-10,1.5)).
    constrained=False: radii = exp(log_r), unconstrained; eigenvalues can escape unit disk.
    init_log_r: override default initialisation (e.g. near-boundary adversarial init).
    """

    def __init__(
        self,
        n_state: int,
        seed: int = 0,
        constrained: bool = True,
        init_log_r: float | None = None,
        schur_stable: bool = False,
    ) -> None:
        super().__init__()
        rng = np.random.default_rng(seed)
        default_log_r = init_log_r if init_log_r is not None else -0.7
        log_r = torch.tensor(rng.normal(default_log_r, 0.05, n_state), dtype=torch.float64)
        angles = torch.tensor(rng.uniform(0.0, 2 * np.pi, n_state), dtype=torch.float64)
        b_re = torch.tensor(rng.normal(0.0, 0.1, n_state), dtype=torch.float64)
        b_im = torch.tensor(rng.normal(0.0, 0.1, n_state), dtype=torch.float64)
        c_re = torch.tensor(rng.normal(0.0, 0.1, n_state), dtype=torch.float64)
        c_im = torch.tensor(rng.normal(0.0, 0.1, n_state), dtype=torch.float64)
        self.log_r = nn.Parameter(log_r)
        self.angles = nn.Parameter(angles)
        self.b_re = nn.Parameter(b_re)
        self.b_im = nn.Parameter(b_im)
        self.c_re = nn.Parameter(c_re)
        self.c_im = nn.Parameter(c_im)
        self.d = nn.Parameter(torch.zeros(1, dtype=torch.float64))
        self.constrained = constrained
        self.schur_stable = schur_stable

    @property
    def a_diag(self) -> torch.Tensor:
        """Complex diagonal eigenvalues."""
        if self.schur_stable:
            # sigmoid maps R -> (0,1); multiply by (1-eps) gives rho < 0.98 always
            radii = torch.sigmoid(self.log_r) * (1.0 - 0.02)
        elif self.constrained:
            radii = torch.exp(torch.clamp(self.log_r, -10.0, 1.5))
        else:
            radii = torch.exp(self.log_r)   # unconstrained: can exceed 1
        return torch.polar(radii, self.angles).to(torch.complex128)

    def forward(self, u_seq: torch.Tensor) -> torch.Tensor:
        """u_seq: (T,) float64 → y: (T,) float64."""
        T = u_seq.shape[0]
        a = self.a_diag                            # (n,) complex
        b = (self.b_re + 1j * self.b_im).to(torch.complex128)  # (n,)
        c = (self.c_re + 1j * self.c_im).to(torch.complex128)  # (n,)
        h = torch.zeros(a.shape[0], dtype=torch.complex128)
        ys = []
        for t in range(T):
            h = a * h + b * u_seq[t].to(torch.complex128)
            y_t = 2.0 * torch.real(c @ h) + self.d[0] * u_seq[t]
            ys.append(y_t)
        return torch.stack(ys)  # (T,) real (via real-part extraction)

    def transition_matrix(self) -> ComplexMatrix:
        with torch.no_grad():
            return np.diag(self.a_diag.numpy().astype(np.complex128))


def train_diagonal_ssm_adam(
    u_seq: NDArray[np.float64],
    target_seq: NDArray[np.float64],
    n_state: int = 8,
    n_epochs: int = 500,
    lr: float = 1e-3,
    reg_weight: float = 0.1,
    regulariser: Callable[[torch.Tensor], torch.Tensor] | None = None,
    seed: int = 0,
    constrained: bool = True,
    init_log_r: float | None = None,
    schur_stable: bool = False,
    rho_history_out: list | None = None,
) -> tuple[ComplexMatrix, list[float], list[float]]:
    """Train a diagonal complex SSM via Adam with an optional stability regulariser.

    constrained=False: eigenvalue radii unconstrained (can exceed 1 without regulariser).
    schur_stable=True: sigmoid parameterisation guarantees rho < 0.98 by construction.
    init_log_r: initial log-radius value (e.g. log(0.96) ≈ -0.041 for near-boundary init).
    rho_history_out: if provided, append per-epoch spectral radius (for trajectory figures).
    Returns (A_matrix, task_loss_history, reg_loss_history).
    """
    model = _DiagonalSSM(
        n_state, seed=seed, constrained=constrained,
        init_log_r=init_log_r, schur_stable=schur_stable,
    )
    u_t = torch.tensor(u_seq, dtype=torch.float64)
    target_t = torch.tensor(target_seq, dtype=torch.float64)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)

    task_hist: list[float] = []
    reg_hist: list[float] = []

    for _ in range(n_epochs):
        optimiser.zero_grad()
        y_pred = model(u_t)
        task_loss = torch.mean((y_pred - target_t) ** 2)
        if regulariser is not None:
            reg_loss = reg_weight * regulariser(model.a_diag)
        else:
            reg_loss = torch.zeros(1, dtype=torch.float64)[0]
        total = task_loss + reg_loss
        total.backward()
        optimiser.step()
        task_hist.append(task_loss.item())
        reg_hist.append(reg_loss.item())
        if rho_history_out is not None:
            with torch.no_grad():
                rho_history_out.append(float(torch.max(torch.abs(model.a_diag)).item()))

    return model.transition_matrix(), task_hist, reg_hist
