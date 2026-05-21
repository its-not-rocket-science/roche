import numpy as np

from roche.certificates import (
    certify_on_unit_circle,
    determinant_margin,
    grid_certificate,
    is_schur_stable,
    resolvent_margin,
    unit_circle_grid,
)
from roche.reference import scalar_reference


def test_unit_circle_grid_has_expected_radius():
    points = unit_circle_grid(16)
    assert np.allclose(np.abs(points), 1.0)


def test_stable_scalar_matrix_certifies_against_close_reference():
    a = np.array([[0.4 + 0.0j]])
    a0 = np.array([[0.3 + 0.0j]])
    assert is_schur_stable(a)
    result = certify_on_unit_circle(a, a0, num_points=64, method="determinant")
    assert result.certified


def test_unstable_scalar_matrix_not_certified_against_stable_reference():
    a = np.array([[1.2 + 0.0j]])
    a0 = np.array([[0.3 + 0.0j]])
    assert not is_schur_stable(a)
    result = certify_on_unit_circle(a, a0, num_points=64, method="determinant")
    assert not result.certified


def test_resolvent_margin_positive_for_small_perturbation():
    a0 = scalar_reference(2, 0.2)
    a = scalar_reference(2, 0.25)
    margin = resolvent_margin(a, a0, 1.0 + 0.0j)
    assert margin > 0


def test_deterministic_grid_certificate_uses_lipschitz_bound():
    margins = np.ones(32) * 0.2
    result = grid_certificate(margins, lipschitz_bound=1.0, method="test")
    assert result.certified


def test_determinant_margin_positive_for_identical_matrices():
    a = scalar_reference(3, 0.5)
    assert determinant_margin(a, a, 1.0 + 0.0j) > 0
