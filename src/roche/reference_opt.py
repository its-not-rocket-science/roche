"""Gradient-based reference matrix optimisation for Paper 2.

For a diagonal reference A0 = diag(d0) and any matrix A, the resolvent
product (zI - A0)^{-1}(A0 - A) has a closed-form diagonal representation
when A is also diagonal, reducing the margin evaluation to O(Kn) with no
matrix solve or SVD.  For general A the product is non-diagonal but the
diagonal fast path for A0 still avoids a batched solve.

All optimisation is done in PyTorch (complex128) for autograd; results
are returned as NumPy arrays for compatibility with the Paper 1 pipeline.
"""

from __future__ import annotations

import time
from typing import Literal

import numpy as np
import torch
from numpy.typing import NDArray

ComplexMatrix = NDArray[np.complex128]

_DTYPE = torch.complex128
_RDTYPE = torch.float64


def _unit_circle(K: int) -> torch.Tensor:
    theta = torch.arange(K, dtype=_RDTYPE) * (2.0 * torch.pi / K)
    return torch.polar(torch.ones(K, dtype=_RDTYPE), theta)


def _resolvent_margins_torch(
    a: torch.Tensor,   # (n,n) or (n,) complex — full matrix or diagonal entries of A
    d0: torch.Tensor,  # (n,) complex — diagonal entries of A0
    points: torch.Tensor,  # (K,) complex
) -> torch.Tensor:
    """Resolvent margins for diagonal A0 = diag(d0), general A.

    Returns shape (K,) float64.

    Fast path (A diagonal): op-norm = max_i |(d0_i-a_i)/(z-d0_i)|, O(Kn).
    General path (A non-diagonal): row-scale (A0-A) by 1/(z-d0_i), SVD, O(Kn^2).
    """
    denom = points[:, None] - d0[None, :]    # (K, n) complex

    if a.ndim == 1:
        # Diagonal A fast path: O(Kn), no SVD
        diff = d0 - a                        # (n,)
        vals = diff[None, :] / denom         # (K, n)
        op_norm = torch.max(torch.abs(vals), dim=1).values  # (K,)
    else:
        # General A: (zI-A0)^{-1}(A0-A) = diag(1/(z-d0_i)) @ (A0-A)
        # Row i of result scaled by 1/(z-d0_i); 2-norm via SVD
        perturbation = torch.diag(d0) - a    # (n,n)
        scale = 1.0 / denom                  # (K,n)
        scaled_M = scale[:, :, None] * perturbation[None, :, :]  # (K,n,n)
        sv = torch.linalg.svd(scaled_M, full_matrices=False).S   # (K,n)
        op_norm = sv[:, 0]                   # (K,)

    return 1.0 - op_norm


def _softmin(x: torch.Tensor, beta: float = 20.0) -> torch.Tensor:
    """Differentiable approximation to min(x) via log-sum-exp."""
    return -(1.0 / beta) * torch.logsumexp(-beta * x, dim=0)


def optimise_diagonal_reference(
    a: ComplexMatrix,
    n_steps: int = 300,
    lr: float = 0.05,
    num_contour_points: int = 256,
    method: Literal["resolvent"] = "resolvent",
    init: Literal["eigs", "scalar", "random"] = "eigs",
    softmin_beta: float = 20.0,
    seed: int = 0,
) -> tuple[ComplexMatrix, list[float]]:
    """Gradient ascent on min_theta resolvent_margin(A, A0, theta) w.r.t. diagonal A0.

    A0 is parameterised as diag(r * exp(i*phi)) where
    r = sigmoid(log_r_raw) * (1 - eps) ensures strict Schur stability.

    Returns (A0_matrix, margin_history).
    """
    torch.manual_seed(seed)
    a_full = a if a.ndim == 2 else np.diag(a)
    n = a_full.shape[0]
    # Diagonal A: pass (n,) for O(Kn) fast path; general A: pass (n,n) for SVD path
    _diag = np.diag(np.diag(a_full))
    _is_diag = np.allclose(a_full, _diag, atol=1e-12)
    if _is_diag:
        a_t = torch.tensor(np.diag(a_full), dtype=_DTYPE)  # (n,) fast path
    else:
        a_t = torch.tensor(a_full, dtype=_DTYPE)            # (n,n) SVD path
    points = _unit_circle(num_contour_points)

    # Initialise
    if init == "eigs":
        from roche.reference import diagonal_reference_from_eigs
        a0_np = diagonal_reference_from_eigs(a if a.ndim == 2 else np.diag(a))
        d0_init = np.diag(a0_np)
    elif init == "scalar":
        d0_init = np.full(n, 0.5 + 0.0j, dtype=np.complex128)
    else:
        rng = np.random.default_rng(seed)
        r0 = rng.uniform(0.1, 0.8, n)
        phi0 = rng.uniform(0.0, 2.0 * np.pi, n)
        d0_init = r0 * np.exp(1j * phi0)

    r0 = np.abs(d0_init).clip(1e-4, 0.97)
    phi0 = np.angle(d0_init)
    log_r_raw = torch.tensor(np.log(r0 / (1.0 - r0)), dtype=_RDTYPE, requires_grad=True)
    phi = torch.tensor(phi0, dtype=_RDTYPE, requires_grad=True)

    optimiser = torch.optim.Adam([log_r_raw, phi], lr=lr)
    history: list[float] = []

    for _ in range(n_steps):
        optimiser.zero_grad()
        eps = 0.02
        r = torch.sigmoid(log_r_raw) * (1.0 - eps)
        d0 = torch.polar(r, phi).to(_DTYPE)
        margins = _resolvent_margins_torch(a_t, d0, points)
        loss = -_softmin(margins, softmin_beta)
        loss.backward()
        optimiser.step()
        history.append(-loss.item())

    with torch.no_grad():
        eps = 0.02
        r = torch.sigmoid(log_r_raw) * (1.0 - eps)
        d0 = torch.polar(r, phi).to(_DTYPE)

    d0_np = d0.numpy().astype(np.complex128)
    return np.diag(d0_np), history


def optimise_dlr_reference(
    a: ComplexMatrix,
    rank: int = 1,
    n_steps: int = 400,
    lr: float = 0.02,
    num_contour_points: int = 256,
    softmin_beta: float = 20.0,
    seed: int = 0,
    init_lowrank_scale: float = 1e-2,
) -> tuple[ComplexMatrix, list[float], dict]:
    """Gradient ascent on resolvent margin w.r.t. DLR reference A0 = diag(d0) + U @ V.H.

    A0 is guaranteed stable via a softplus stability penalty on its eigenvalues.
    Diagonal part parameterised as diag(sigmoid(log_r_raw)*(1-eps)).
    U, V free complex (n x rank); their contribution is scaled by a small factor
    to keep A0 close to diagonal and aid stability.

    Parameters
    ----------
    init_lowrank_scale:
        Scale of random noise used to initialise U, V away from zero. Nonzero
        initialisation breaks the gradient symmetry and allows the low-rank term
        to learn.

    Returns (A0_matrix, margin_history, diagnostics) where diagnostics contains:
        spectral_radius_a0, lowrank_frobenius_norm, diag_norm,
        final_sampled_margin, stability_penalty_active.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    n = a.shape[0]
    a_t = torch.tensor(a, dtype=_DTYPE)
    points = _unit_circle(num_contour_points)

    # Diagonal part: stable reparameterisation
    from roche.reference import diagonal_reference_from_eigs
    a0_diag_np = diagonal_reference_from_eigs(a)
    d0_init = np.diag(a0_diag_np)
    r0 = np.abs(d0_init).clip(1e-4, 0.97)
    phi0 = np.angle(d0_init)
    log_r_raw = torch.tensor(np.log(r0 / (1.0 - r0)), dtype=_RDTYPE, requires_grad=True)
    phi = torch.tensor(phi0, dtype=_RDTYPE, requires_grad=True)

    # Low-rank part: initialise with small nonzero noise to break symmetry
    scale = 0.1
    u_re_init = rng.standard_normal((n, rank)) * init_lowrank_scale
    u_im_init = rng.standard_normal((n, rank)) * init_lowrank_scale
    v_re_init = rng.standard_normal((n, rank)) * init_lowrank_scale
    v_im_init = rng.standard_normal((n, rank)) * init_lowrank_scale
    u_re = torch.tensor(u_re_init, dtype=_RDTYPE, requires_grad=True)
    u_im = torch.tensor(u_im_init, dtype=_RDTYPE, requires_grad=True)
    v_re = torch.tensor(v_re_init, dtype=_RDTYPE, requires_grad=True)
    v_im = torch.tensor(v_im_init, dtype=_RDTYPE, requires_grad=True)

    params = [log_r_raw, phi, u_re, u_im, v_re, v_im]
    optimiser = torch.optim.Adam(params, lr=lr)
    history: list[float] = []

    for _ in range(n_steps):
        optimiser.zero_grad()
        eps = 0.02
        r = torch.sigmoid(log_r_raw) * (1.0 - eps)
        d0 = torch.polar(r, phi).to(_DTYPE)                           # (n,)
        U = (u_re + 1j * u_im).to(_DTYPE) * scale                    # (n, rank)
        V = (v_re + 1j * v_im).to(_DTYPE) * scale                    # (n, rank)
        A0 = torch.diag(d0) + U @ V.conj().T                         # (n, n)

        # Resolvent margins via batched SVD
        lhs = points[:, None, None] * torch.eye(n, dtype=_DTYPE)[None] - A0[None]  # (K,n,n)
        perturbation = A0 - a_t                                                      # (n,n)
        resolved = torch.linalg.solve(lhs, perturbation[None].expand(len(points), -1, -1))
        sv = torch.linalg.svd(resolved, full_matrices=False).S
        op_norm = sv[:, 0]
        margins = 1.0 - op_norm

        # Stability penalty: softplus on spectral radius of A0 exceeding (1 - eps)
        eigs = torch.linalg.eigvals(A0)
        rho_a0 = torch.max(torch.abs(eigs))
        stability_penalty = torch.nn.functional.softplus(rho_a0 - (1.0 - eps))

        loss = -_softmin(margins, softmin_beta) + 5.0 * stability_penalty
        loss.backward()
        optimiser.step()
        history.append(float(margins.min().item()))

    with torch.no_grad():
        eps = 0.02
        r = torch.sigmoid(log_r_raw) * (1.0 - eps)
        d0 = torch.polar(r, phi).to(_DTYPE)
        U = (u_re + 1j * u_im).to(_DTYPE) * scale
        V = (v_re + 1j * v_im).to(_DTYPE) * scale
        A0 = torch.diag(d0) + U @ V.conj().T

        # Compute diagnostics
        eigs_final = torch.linalg.eigvals(A0)
        rho_final = float(torch.max(torch.abs(eigs_final)).item())
        lr_matrix = (U @ V.conj().T)
        lr_frob = float(torch.linalg.norm(lr_matrix, ord="fro").item())
        diag_frob = float(torch.linalg.norm(torch.diag(d0), ord="fro").item())
        final_margin = history[-1] if history else float("nan")
        penalty_active = rho_final >= (1.0 - eps)

    a0_np = A0.numpy().astype(np.complex128)
    diagnostics = {
        "spectral_radius_a0": rho_final,
        "lowrank_frobenius_norm": lr_frob,
        "diag_norm": diag_frob,
        "final_sampled_margin": final_margin,
        "stability_penalty_active": penalty_active,
    }
    return a0_np, history, diagnostics


def compare_reference_methods(
    a: ComplexMatrix,
    num_contour_points: int = 512,
    random_search_candidates: int = 200,
    opt_steps: int = 300,
    seed: int = 0,
) -> dict[str, dict]:
    """Compare four reference methods on a single matrix A.

    Returns dict: method -> {min_margin, certified, wall_time}.
    """
    from roche.certificates import certify_on_unit_circle
    from roche.reference import (
        diagonal_reference_from_eigs,
        random_search_diagonal_reference,
        scalar_reference,
    )

    n = a.shape[0]
    results: dict[str, dict] = {}

    for name, ref_fn in [
        ("scalar",   lambda: scalar_reference(n, 0.5)),
        ("eig_shrunk", lambda: diagonal_reference_from_eigs(a)),
        ("random_search", lambda: random_search_diagonal_reference(
            a, random_search_candidates, num_contour_points, "resolvent", seed)[0]),
        ("gradient_opt", lambda: optimise_diagonal_reference(
            a, opt_steps, num_contour_points=num_contour_points, seed=seed)[0]),
    ]:
        t0 = time.perf_counter()
        a0 = ref_fn()
        wall = time.perf_counter() - t0
        cert = certify_on_unit_circle(a, a0, num_contour_points, "resolvent")
        results[name] = {
            "min_margin": cert.min_margin,
            "certified": cert.certified,
            "wall_time": wall,
            "a0": a0,
        }

    return results
