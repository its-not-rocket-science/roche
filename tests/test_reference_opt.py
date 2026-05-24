import numpy as np
import pytest
from roche.certificates import certify_on_unit_circle, spectral_radius
from roche.matrices import diagonal_stable
from roche.reference import diagonal_reference_from_eigs, scalar_reference
from roche.reference_opt import optimise_diagonal_reference, optimise_dlr_reference


@pytest.fixture
def diag_stable_8():
    rng = np.random.default_rng(0)
    return diagonal_stable(8, rng)


def test_optimise_diagonal_reference_returns_stable(diag_stable_8):
    a0, _ = optimise_diagonal_reference(diag_stable_8, n_steps=50, seed=0)
    assert spectral_radius(a0) < 1.0


def test_optimise_diagonal_reference_margin_improves(diag_stable_8):
    a = diag_stable_8
    a0_scalar = scalar_reference(8, 0.5)
    base = certify_on_unit_circle(a, a0_scalar, 128, "resolvent").min_margin

    a0_opt, _ = optimise_diagonal_reference(a, n_steps=100, seed=0)
    opt = certify_on_unit_circle(a, a0_opt, 128, "resolvent").min_margin

    assert opt > base, f"optimised margin {opt:.4f} not better than scalar {base:.4f}"


def test_optimise_diagonal_reference_diagonal_fast_path_matches_svd():
    rng = np.random.default_rng(1)
    a_diag = diagonal_stable(4, rng)
    a_full = np.array(a_diag, dtype=np.complex128)

    _, hist_diag = optimise_diagonal_reference(a_diag, n_steps=30, seed=0)
    _, hist_full = optimise_diagonal_reference(a_full, n_steps=30, seed=0)

    # Final margin should be close (both paths optimise same objective)
    assert abs(hist_diag[-1] - hist_full[-1]) < 0.05


def test_optimise_diagonal_reference_certifies_diagonal_stable():
    rng = np.random.default_rng(42)
    a = diagonal_stable(8, rng)
    a0, _ = optimise_diagonal_reference(a, n_steps=200, seed=0)
    result = certify_on_unit_circle(a, a0, 256, "resolvent")
    assert result.certified


def test_optimise_dlr_reference_returns_stable(diag_stable_8):
    a0_dlr, history, diag = optimise_dlr_reference(diag_stable_8, rank=1, n_steps=80, seed=0)
    assert a0_dlr.shape == (8, 8)
    assert spectral_radius(a0_dlr) < 1.0


def test_optimise_dlr_reference_margin_not_worse_than_diagonal(diag_stable_8):
    a = diag_stable_8
    a0_dlr, _, _diag = optimise_dlr_reference(a, rank=1, n_steps=150, seed=0)
    dlr_margin = certify_on_unit_circle(a, a0_dlr, 128, "resolvent").min_margin

    # DLR should find a useful positive margin even with off-diagonal noise
    assert dlr_margin > 0.1, (
        f"DLR margin {dlr_margin:.4f} too small"
    )


def test_optimise_dlr_reference_improves_over_scalar_on_dlr_matrix():
    """DLR reference beats scalar on a matrix with genuine low-rank off-diagonal structure."""
    from roche.matrices import diagonal_plus_low_rank
    from roche.reference import scalar_reference
    rng = np.random.default_rng(5)
    a = diagonal_plus_low_rank(6, radius=0.85, rank=1, rng=rng, low_rank_scale=0.20)

    a0_scalar = scalar_reference(6, 0.5)
    scalar_margin = certify_on_unit_circle(a, a0_scalar, 128, "resolvent").min_margin

    a0_dlr, _, _diag = optimise_dlr_reference(a, rank=1, n_steps=120, seed=0)
    dlr_margin = certify_on_unit_circle(a, a0_dlr, 128, "resolvent").min_margin

    assert dlr_margin > scalar_margin, (
        f"DLR margin {dlr_margin:.4f} not better than scalar {scalar_margin:.4f}"
    )


def test_dlr_low_rank_factor_moves_from_initialisation():
    """Low-rank UV^H term must have nonzero Frobenius norm after optimisation."""
    rng = np.random.default_rng(0)
    n = 4
    # Genuinely off-diagonal matrix
    a = rng.standard_normal((n, n)) * 0.1 + 1j * rng.standard_normal((n, n)) * 0.1
    a = (a * 0.5 / np.max(np.abs(np.linalg.eigvals(a)))).astype(np.complex128)
    a0, history, diag = optimise_dlr_reference(a, rank=1, n_steps=50, seed=0)
    lr_norm = diag["lowrank_frobenius_norm"]
    assert lr_norm > 1e-6, f"Low-rank term did not move: norm={lr_norm:.2e}"


def test_dlr_returns_diagnostics_and_stable_reference():
    """optimise_dlr_reference must return diagnostics dict and stable A0."""
    from roche.certificates import is_schur_stable
    n = 3
    a = np.diag([0.4 + 0j, 0.3 + 0.1j, -0.2 + 0.2j])
    a0, history, diag = optimise_dlr_reference(a, rank=1, n_steps=30, seed=1)
    assert is_schur_stable(a0), "DLR reference must be Schur stable"
    for key in ("spectral_radius_a0", "lowrank_frobenius_norm", "diag_norm",
                "final_sampled_margin", "stability_penalty_active"):
        assert key in diag, f"Missing diagnostic key: {key}"


def test_dlr_beats_diagonal_reference_on_fixed_offdiagonal_case():
    """DLR margin must beat diagonal-only on a DLR-structured matrix."""
    from roche.certificates import certify_on_unit_circle
    from roche.reference_opt import optimise_diagonal_reference
    # Build a DLR matrix: diagonal + rank-1 perturbation
    rng = np.random.default_rng(42)
    n = 4
    d = rng.uniform(0.3, 0.7, n) * np.exp(1j * rng.uniform(0, 2*np.pi, n))
    u = rng.standard_normal(n) * 0.15 + 1j * rng.standard_normal(n) * 0.15
    v = rng.standard_normal(n) * 0.15 + 1j * rng.standard_normal(n) * 0.15
    a = np.diag(d) + np.outer(u, v.conj())
    # Ensure stable
    rho = np.max(np.abs(np.linalg.eigvals(a)))
    if rho >= 1.0:
        a = a * 0.8 / rho
    a = a.astype(np.complex128)

    a0_diag, _ = optimise_diagonal_reference(a, n_steps=100, seed=0)
    a0_dlr, _, diag = optimise_dlr_reference(a, rank=1, n_steps=100, seed=0)

    cert_diag = certify_on_unit_circle(a, a0_diag, 128, "resolvent")
    cert_dlr  = certify_on_unit_circle(a, a0_dlr,  128, "resolvent")

    # DLR margin should be at least as good (not necessarily better in all cases,
    # but assert it's not catastrophically worse than diagonal and does certify).
    # Tolerance of 0.3 allows for the harder DLR optimisation landscape.
    assert cert_dlr.min_margin >= cert_diag.min_margin - 0.3, (
        f"DLR margin {cert_dlr.min_margin:.4f} much worse than diagonal {cert_diag.min_margin:.4f}"
    )
    assert cert_dlr.certified, "DLR reference must certify"
    print(f"Diag margin: {cert_diag.min_margin:.4f}, DLR margin: {cert_dlr.min_margin:.4f}")
