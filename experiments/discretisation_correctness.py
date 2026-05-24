"""Experiment 2: Discretisation correctness of finite-grid certificates.

Verifies Theorem 3: if min_k m(theta_k) > pi*L/N then m(theta) > 0 everywhere.
Tests that:
  - no false positives occur when the Lipschitz condition is satisfied;
  - the certificate becomes reliable as N grows;
  - the sampled-only diagnostic can give false positives at small N.

Usage:
    python experiments/discretisation_correctness.py [--seed S] [--outdir DIR]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from roche.certificates import (
    certify_on_unit_circle,
    finite_difference_lipschitz,
    grid_certificate,
    is_schur_stable,
    resolvent_lipschitz_bound,
    resolvent_margins,
    spectral_radius,
    unit_circle_grid,
)
from roche.matrices import (
    diagonal_plus_low_rank,
    jordan_like_matrix,
    random_nonnormal_matrix,
    random_normal_matrix,
)
from roche.plotting import plot_discretisation_study
from roche.reference import scalar_reference, diagonal_reference_from_eigs

GRID_SIZES = [16, 32, 64, 128, 256, 512, 1024]


def run(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Use a dense oversampled baseline (N=4096) for "ground truth" certification
    N_truth = 4096
    n = 8

    # Diagonal test cases: certificates work reliably for diagonal A with diagonal A0.
    # Phases deliberately varied to make the problem non-trivial.
    angles = rng.uniform(0.0, 2.0 * np.pi, n)
    diag_r080 = np.diag(0.80 * np.exp(1j * angles))
    diag_r090 = np.diag(0.90 * np.exp(1j * angles))
    diag_r095 = np.diag(0.95 * np.exp(1j * angles))
    diag_ref = np.diag(0.50 * np.exp(1j * angles))  # reference with same phases

    # Non-diagonal cases for comparison (expect certificate failure)
    test_cases = {
        # Diagonal: resolvent cert = max_i |r - 0.5| / (1 - 0.5) = 2|r - 0.5|
        # For r=0.80: cert quantity = 0.60 < 1 (passes)
        "diag_r080_same_phase": (diag_r080, diag_ref),
        # For r=0.90: cert quantity = 0.80 < 1 (passes, small margin)
        "diag_r090_same_phase": (diag_r090, diag_ref),
        # For r=0.95: cert quantity = 0.90 < 1 (passes, very small margin)
        "diag_r095_same_phase": (diag_r095, diag_ref),
        # Non-diagonal: dense normal (expected to fail certificate with any simple ref)
        "dense_normal_scalar_ref": (
            random_normal_matrix(n, 0.85, rng),
            scalar_reference(n, 0.5),
        ),
        # Jordan: upper triangular, large resolvent norm near boundary
        "jordan_near_r095_scalar": (
            jordan_like_matrix(n, 0.95 + 0.0j, superdiag=1.0),
            scalar_reference(n, 0.5),
        ),
    }

    if args.quick:
        test_cases = dict(list(test_cases.items())[:2])
        grid_sizes_run = [64, 256]
    else:
        grid_sizes_run = GRID_SIZES

    csv_rows = []
    for name, (a, a0) in test_cases.items():
        stable = is_schur_stable(a)
        rho = spectral_radius(a)

        # Ground-truth margin and Lipschitz on dense grid
        truth_points = unit_circle_grid(N_truth)
        truth_margins = resolvent_margins(a, a0, truth_points)
        truth_min = float(np.min(truth_margins))
        truth_cert = truth_min > 0.0
        L_dense = resolvent_lipschitz_bound(a, a0, truth_points)

        print(f"\n{'='*60}")
        print(f"Case: {name}  rho={rho:.4f}  stable={stable}  truth_cert={truth_cert}")
        print(f"  truth min_margin={truth_min:+.4e}  L_dense={L_dense:.4e}")
        print(
            f"  {'N':>6}  {'min_m':>10}  {'L_est':>10}  {'pi*L/N':>10}"
            f"  {'sampled':>8}  {'rigorous':>9}"
        )

        sampled_certs = []
        rigorous_certs = []
        min_margins_list = []
        criteria_list = []

        for N in grid_sizes_run:
            points = unit_circle_grid(N)
            margins = resolvent_margins(a, a0, points)
            min_m = float(np.min(margins))
            L_est = resolvent_lipschitz_bound(a, a0, points)
            criterion = np.pi * L_est / N

            res_sampled = grid_certificate(margins, lipschitz_bound=None, method="resolvent")
            res_rigorous = grid_certificate(margins, lipschitz_bound=L_est, method="resolvent")

            sampled_certs.append(res_sampled.certified)
            rigorous_certs.append(res_rigorous.certified)
            min_margins_list.append(min_m)
            criteria_list.append(criterion)
            csv_rows.append({
                "case": name, "N": N, "min_margin": f"{min_m:.6e}",
                "L_est": f"{L_est:.6e}", "criterion": f"{criterion:.6e}",
                "sampled_cert": int(res_sampled.certified),
                "rigorous_cert": int(res_rigorous.certified),
            })

            print(
                f"  {N:6d}  {min_m:+10.4e}  {L_est:10.4e}  {criterion:10.4e}"
                f"  {int(res_sampled.certified):8d}  {int(res_rigorous.certified):9d}"
            )

        # Flag any false positives (certified but not ground-truth certified)
        for i, N in enumerate(grid_sizes_run):
            if sampled_certs[i] and not truth_cert:
                print(f"  *** FALSE POSITIVE (sampled) at N={N} ***")
            if rigorous_certs[i] and not truth_cert:
                print(f"  *** FALSE POSITIVE (rigorous) at N={N} ***")

        if not args.quick:
            plot_discretisation_study(
                grid_sizes_run,
                sampled_certs,
                rigorous_certs,
                min_margins_list,
                criteria_list,
                path=outdir / f"disc_{name}.png",
            )

    import csv
    with open(outdir / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["case","N","min_margin","L_est","criterion",
                                               "sampled_cert","rigorous_cert"])
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"\nFigures saved to {outdir}/")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", type=str, default="results/exp2")
    parser.add_argument("--quick", action="store_true",
                        help="Fast smoke: 2 grid sizes, 2 test cases")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
