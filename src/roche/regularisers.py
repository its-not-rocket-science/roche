"""Training-time stability regularisers for Paper 2.

All regularisers take the diagonal SSM eigenvalue vector a_diag
(PyTorch complex128 tensor of shape (n,)) and return a real scalar loss.

Three regulariser classes:
  spectral_penalty  -- hinge on max|lambda_i|
  lyapunov_penalty  -- per-mode sum, different gradient field from spectral
  contour_barrier   -- differentiable resolvent margin barrier

For diagonal A and diagonal A0 the resolvent barrier is O(Kn) with no SVD.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray

ComplexMatrix = NDArray[np.complex128]
_DTYPE = torch.complex128
_RDTYPE = torch.float64


def spectral_penalty(
    a_diag: torch.Tensor,
    margin: float = 0.01,
) -> torch.Tensor:
    """Hinge loss on spectral radius: relu(max|a_i| - (1 - margin)).

    Exactly zero for clearly stable systems; gradient fires only when rho
    exceeds the threshold.  relu is preferred over softplus here because
    softplus adds a constant background even far below the threshold.
    """
    rho = torch.max(torch.abs(a_diag))
    return F.relu(rho - (1.0 - margin))


def spectral_softplus(
    a_diag: torch.Tensor,
    margin: float = 0.01,
    beta: float = 10.0,
) -> torch.Tensor:
    """Softplus hinge on spectral radius: softplus(beta*(rho - (1-margin))) / beta.

    Unlike relu, provides non-zero gradient everywhere below the threshold,
    testing whether the relu dead-zone is the cause of adversarial failure.
    The background penalty at rho << threshold decays exponentially with beta.
    """
    rho = torch.max(torch.abs(a_diag))
    return F.softplus(beta * (rho - (1.0 - margin))) / beta


def lyapunov_penalty(
    a_diag: torch.Tensor,
    margin: float = 0.01,
) -> torch.Tensor:
    """Per-mode Lyapunov hinge: sum_i relu(|a_i|^2 - (1 - margin)).

    For diagonal A with P=I the Lyapunov condition A^H P A - P < 0 reduces to
    |a_i|^2 < 1 for each i.  Using a sum rather than max gives different
    gradient dynamics: all near-unstable modes are pulled toward stability jointly.
    relu ensures exactly zero penalty for clearly stable modes.
    """
    return torch.sum(F.relu(torch.abs(a_diag) ** 2 - (1.0 - margin)))


def make_contour_barrier(
    a0_diag_np: ComplexMatrix,
    num_contour_points: int = 256,
    softmin_beta: float = 20.0,
    margin: float = 0.01,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return a contour barrier regulariser with a fixed reference A0.

    The reference A0 is computed once (e.g. from gradient optimisation on the
    initial model) and frozen throughout training.

    The barrier fires when min_k margin(z_k) < margin, penalising the
    shortfall via softplus(-softmin(margins) + margin).
    """
    d0_np = np.diag(a0_diag_np) if a0_diag_np.ndim == 2 else a0_diag_np
    d0 = torch.tensor(d0_np, dtype=_DTYPE)

    theta = torch.arange(num_contour_points, dtype=_RDTYPE) * (2.0 * torch.pi / num_contour_points)
    points = torch.polar(torch.ones(num_contour_points, dtype=_RDTYPE), theta)

    def barrier(a_diag: torch.Tensor) -> torch.Tensor:
        # a_diag: (n,) complex — eigenvalues of learned A (diagonal)
        diff = d0 - a_diag                         # (n,) complex
        denom = points[:, None] - d0[None, :]      # (K, n) complex
        vals = diff[None, :] / denom               # (K, n) complex
        op_norm = torch.max(torch.abs(vals), dim=1).values  # (K,)
        margins = 1.0 - op_norm                    # (K,) real
        soft_min = -(1.0 / softmin_beta) * torch.logsumexp(-softmin_beta * margins, dim=0)
        return F.relu(-soft_min + margin)

    return barrier
