"""Matrix generators for certificate experiments."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import qr

from roche.certificates import spectral_radius

ComplexMatrix = NDArray[np.complex128]


def random_unitary(n: int, rng: np.random.Generator) -> ComplexMatrix:
    """Generate a random unitary matrix using QR decomposition."""
    x = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    q, r = qr(x)
    phases = np.diag(r) / np.maximum(np.abs(np.diag(r)), 1e-12)
    return (q * phases.conj()).astype(np.complex128)


def random_normal_matrix(
    n: int,
    radius: float,
    rng: np.random.Generator,
) -> ComplexMatrix:
    """Generate a normal matrix with eigenvalues inside a disk of given radius."""
    angles = rng.uniform(0.0, 2.0 * np.pi, size=n)
    radii = radius * np.sqrt(rng.uniform(0.0, 1.0, size=n))
    eigs = radii * np.exp(1j * angles)
    u = random_unitary(n, rng)
    return (u @ np.diag(eigs) @ u.conj().T).astype(np.complex128)


def random_nonnormal_matrix(
    n: int,
    radius: float,
    rng: np.random.Generator,
    departure: float = 5.0,
) -> ComplexMatrix:
    """Generate a stable but potentially non-normal matrix.

    The construction uses a random similarity transform with controllable
    conditioning, then rescales to the requested spectral radius.
    """
    eigs = radius * rng.uniform(0.2, 1.0, size=n) * np.exp(
        1j * rng.uniform(0.0, 2.0 * np.pi, size=n)
    )
    s = np.diag(np.geomspace(1.0, departure, n))
    u = random_unitary(n, rng)
    v = u @ s @ u.conj().T
    a = v @ np.diag(eigs) @ np.linalg.inv(v)
    rho = spectral_radius(a)
    if rho > 0:
        a = a * (radius / rho)
    return a.astype(np.complex128)


def jordan_like_matrix(n: int, eigenvalue: complex, superdiag: float = 1.0) -> ComplexMatrix:
    """Return an upper triangular near-Jordan block."""
    a = np.eye(n, dtype=np.complex128) * eigenvalue
    for i in range(n - 1):
        a[i, i + 1] = superdiag
    return a


def diagonal_plus_low_rank(
    n: int,
    radius: float,
    rank: int,
    rng: np.random.Generator,
    low_rank_scale: float = 0.05,
) -> ComplexMatrix:
    """Generate a diagonal-plus-low-rank matrix and rescale it."""
    eigs = radius * rng.uniform(0.1, 1.0, size=n) * np.exp(
        1j * rng.uniform(0.0, 2.0 * np.pi, size=n)
    )
    d = np.diag(eigs)
    u = rng.normal(size=(n, rank)) + 1j * rng.normal(size=(n, rank))
    v = rng.normal(size=(rank, n)) + 1j * rng.normal(size=(rank, n))
    a = d + low_rank_scale * (u @ v) / np.sqrt(n * rank)
    rho = spectral_radius(a)
    if rho > 0:
        a = a * (radius / rho)
    return a.astype(np.complex128)


def scale_to_radius(a: ComplexMatrix, target_radius: float) -> ComplexMatrix:
    """Scale a matrix to a target spectral radius."""
    rho = spectral_radius(a)
    if rho == 0:
        return a.copy()
    return (a * (target_radius / rho)).astype(np.complex128)
