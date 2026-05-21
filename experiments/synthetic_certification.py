"""Starter synthetic experiment for Paper 1.

Example:
    python experiments/synthetic_certification.py --n 8 --num-matrices 100 --num-contour 256
"""

from __future__ import annotations

import argparse
from collections import Counter

import numpy as np

from roche.certificates import certify_on_unit_circle, is_schur_stable, spectral_radius
from roche.matrices import diagonal_plus_low_rank, random_nonnormal_matrix, random_normal_matrix
from roche.reference import scalar_reference


def build_matrix(kind: str, n: int, rng: np.random.Generator) -> np.ndarray:
    if kind == "normal_stable":
        return random_normal_matrix(n, radius=0.9, rng=rng)
    if kind == "normal_unstable":
        return random_normal_matrix(n, radius=1.1, rng=rng)
    if kind == "nonnormal_stable":
        return random_nonnormal_matrix(n, radius=0.9, rng=rng, departure=20.0)
    if kind == "dlr_stable":
        return diagonal_plus_low_rank(n, radius=0.9, rank=max(1, n // 8), rng=rng)
    raise ValueError(f"unknown kind: {kind}")


def run(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)
    kinds = ["normal_stable", "normal_unstable", "nonnormal_stable", "dlr_stable"]
    counts: Counter[tuple[str, str, bool, bool]] = Counter()

    for kind in kinds:
        for _ in range(args.num_matrices):
            a = build_matrix(kind, args.n, rng)
            a0 = scalar_reference(args.n, args.reference_radius)
            stable = is_schur_stable(a, tol=1e-8)
            rho = spectral_radius(a)

            det_result = certify_on_unit_circle(
                a, a0, num_points=args.num_contour, method="determinant"
            )
            res_result = certify_on_unit_circle(
                a, a0, num_points=args.num_contour, method="resolvent"
            )

            counts[(kind, "determinant", stable, det_result.certified)] += 1
            counts[(kind, "resolvent", stable, res_result.certified)] += 1

            if args.verbose:
                print(
                    f"{kind:18s} rho={rho:.3f} stable={stable} "
                    f"det_margin={det_result.min_margin:+.3e} det={det_result.certified} "
                    f"res_margin={res_result.min_margin:+.3e} res={res_result.certified}"
                )

    print("\nSummary counts: (kind, method, truly_stable, sampled_certified) -> count")
    for key, value in sorted(counts.items()):
        print(f"{key}: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--num-matrices", type=int, default=100)
    parser.add_argument("--num-contour", type=int, default=256)
    parser.add_argument("--reference-radius", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
