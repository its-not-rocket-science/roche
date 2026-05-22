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
    a: torch.Tensor,   # (n,) complex — diagonal entries of A
    d0: torch.Tensor,  # (n,) complex — diagonal entries of A0
    points: torch.Tensor,  # (K,) complex
) -> torch.Tensor:
    """Resolvent margins for diagonal A0, general diagonal A.

    Returns shape (K,) float64.

    For diagonal A0: (zI-A0)^{-1}(A0-A) has entries (d0_i - a_i)/(z - d0_i).
    Operator 2-norm of a diagonal matrix = max |entry|.
    """
    # (K, n): scale[k, i] = (d0_i - a_i) / (z_k - d0_i)
    diff = d0 - a                              # (n,) complex
    denom = points[:, None] - d0[None, :]      # (K, n) complex
    vals = diff[None, :] / denom               # (K, n) complex
    op_norm = torch.max(torch.abs(vals), dim=1).values  # (K,)
    return 1.0 - op_norm                       # (K,) real (autograd-able)


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
    n = a.shape[0]
    a_t = torch.tensor(np.diag(a) if a.ndim == 2 else a, dtype=_DTYPE)
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
