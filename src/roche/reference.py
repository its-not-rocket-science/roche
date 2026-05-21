"""Reference matrix construction for Rouché-style certificates."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import eigvals

from roche.certificates import certify_on_unit_circle

ComplexMatrix = NDArray[np.complex128]


def scalar_reference(n: int, r: float = 0.5) -> ComplexMatrix:
    """Return A0 = rI."""
    if not (0 <= r < 1):
        raise ValueError("r must be in [0, 1)")
    return (r * np.eye(n, dtype=np.complex128)).astype(np.complex128)


def diagonal_reference_from_eigs(a: ComplexMatrix, shrink: float = 0.9) -> ComplexMatrix:
    """Build a diagonal stable reference from shrunken eigenvalues of A.

    This does not preserve eigenvectors. It is a simple baseline for the
    reference-optimisation section.
    """
    if not (0 < shrink < 1):
        raise ValueError("shrink must be in (0, 1)")
    eigs = eigvals(a)
    radii = np.minimum(np.abs(eigs), 1.0) * shrink
    phases = np.exp(1j * np.angle(eigs))
    return np.diag(radii * phases).astype(np.complex128)


def random_diagonal_reference(
    n: int,
    rng: np.random.Generator,
    min_radius: float = 0.0,
    max_radius: float = 0.95,
) -> ComplexMatrix:
    """Sample a random stable diagonal reference."""
    radii = rng.uniform(min_radius, max_radius, size=n)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=n)
    return np.diag(radii * np.exp(1j * phases)).astype(np.complex128)


def random_search_diagonal_reference(
    a: ComplexMatrix,
    num_candidates: int = 100,
    num_points: int = 256,
    method: str = "resolvent",
    seed: int = 0,
) -> tuple[ComplexMatrix, float]:
    """Naive random search for a diagonal reference with large sampled margin."""
    rng = np.random.default_rng(seed)
    n = a.shape[0]
    best_a0 = scalar_reference(n, 0.5)
    best_score = certify_on_unit_circle(
        a, best_a0, num_points=num_points, method=method  # type: ignore[arg-type]
    ).min_margin
    for _ in range(num_candidates):
        candidate = random_diagonal_reference(n, rng)
        result = certify_on_unit_circle(
            a, candidate, num_points=num_points, method=method  # type: ignore[arg-type]
        )
        if result.min_margin > best_score:
            best_score = result.min_margin
            best_a0 = candidate
    return best_a0, float(best_score)
