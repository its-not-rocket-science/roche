import numpy as np

from roche.certificates import spectral_radius
from roche.matrices import diagonal_plus_low_rank, random_nonnormal_matrix, random_normal_matrix


def test_random_normal_matrix_scaled_inside_radius():
    rng = np.random.default_rng(0)
    a = random_normal_matrix(5, radius=0.8, rng=rng)
    assert spectral_radius(a) <= 0.8 + 1e-8


def test_random_nonnormal_matrix_scaled_inside_radius():
    rng = np.random.default_rng(1)
    a = random_nonnormal_matrix(5, radius=0.8, rng=rng)
    assert spectral_radius(a) <= 0.8 + 1e-8


def test_diagonal_plus_low_rank_scaled_inside_radius():
    rng = np.random.default_rng(2)
    a = diagonal_plus_low_rank(8, radius=0.9, rank=2, rng=rng)
    assert spectral_radius(a) <= 0.9 + 1e-8
