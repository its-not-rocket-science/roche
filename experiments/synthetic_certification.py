"""Experiment 1: Certificate geometry across matrix families.

Usage:
    python experiments/synthetic_certification.py [--n N] [--num-matrices M]
                                                   [--num-contour K] [--seed S]
                                                   [--outdir DIR] [--verbose]
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
    is_schur_stable,
    resolvent_lipschitz_bound,
    spectral_radius,
    unit_circle_grid,
)
from roche.matrices import (
    block_diagonal_matrix,
    diagonal_plus_low_rank,
    jordan_like_matrix,
    random_block_diagonal,
    random_nonnormal_matrix,
    random_normal_matrix,
)
from roche.plotting import plot_margin_comparison
from roche.reference import (
    diagonal_reference_from_eigs,
    random_search_diagonal_reference,
    scalar_reference,
)


KINDS = [
    "normal_stable",
    "normal_unstable",
    "nonnormal_stable",
    "nonnormal_unstable",
    "jordan_stable",
    "jordan_near_boundary",
    "block_diagonal_stable",
    "dlr_stable",
]


def build_matrix(kind: str, n: int, rng: np.random.Generator) -> np.ndarray:
    if kind == "normal_stable":
        return random_normal_matrix(n, radius=0.85, rng=rng)
    if kind == "normal_unstable":
        return random_normal_matrix(n, radius=1.1, rng=rng)
    if kind == "nonnormal_stable":
        return random_nonnormal_matrix(n, radius=0.85, rng=rng, departure=20.0)
    if kind == "nonnormal_unstable":
        return random_nonnormal_matrix(n, radius=1.05, rng=rng, departure=20.0)
    if kind == "jordan_stable":
        return jordan_like_matrix(n, eigenvalue=0.8 + 0.0j, superdiag=0.5)
    if kind == "jordan_near_boundary":
        return jordan_like_matrix(n, eigenvalue=0.95 + 0.0j, superdiag=1.0)
    if kind == "block_diagonal_stable":
        block_sz = max(2, n // 4)
        return random_block_diagonal(n, block_size=block_sz, radius=0.85, rng=rng)
    if kind == "dlr_stable":
        return diagonal_plus_low_rank(n, radius=0.85, rank=max(1, n // 8), rng=rng)
    raise ValueError(f"unknown kind: {kind}")


def run(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []

    for kind in KINDS:
        det_margins_all: list[float] = []
        res_margins_all: list[float] = []
        det_cert_count = 0
        res_cert_count = 0
        true_stable_count = 0
        fn_det = 0  # false negatives for stable matrices
        fn_res = 0

        for trial in range(args.num_matrices):
            a = build_matrix(kind, args.n, rng)
            a0_scalar = scalar_reference(args.n, args.reference_radius)
            a0_diag, diag_score = random_search_diagonal_reference(
                a, num_candidates=50, num_points=128, method="resolvent", seed=trial
            )
            # Use diagonal reference if it gives better margin
            a0 = a0_diag if diag_score > certify_on_unit_circle(
                a, a0_scalar, num_points=128, method="resolvent"
            ).min_margin else a0_scalar

            stable = is_schur_stable(a, tol=1e-8)
            rho = spectral_radius(a)

            det_result = certify_on_unit_circle(
                a, a0, num_points=args.num_contour, method="determinant"
            )
            res_result = certify_on_unit_circle(
                a, a0, num_points=args.num_contour, method="resolvent"
            )

            det_margins_all.append(det_result.min_margin)
            res_margins_all.append(res_result.min_margin)

            if stable:
                true_stable_count += 1
                if not det_result.certified:
                    fn_det += 1
                if not res_result.certified:
                    fn_res += 1
            if det_result.certified:
                det_cert_count += 1
            if res_result.certified:
                res_cert_count += 1

            if args.verbose:
                print(
                    f"{kind:24s} trial={trial:3d} rho={rho:.3f} "
                    f"stable={int(stable)} "
                    f"det={int(det_result.certified)} ({det_result.min_margin:+.3e}) "
                    f"res={int(res_result.certified)} ({res_result.min_margin:+.3e})"
                )

        fn_rate_det = fn_det / max(1, true_stable_count)
        fn_rate_res = fn_res / max(1, true_stable_count)
        results.append(
            {
                "kind": kind,
                "true_stable": true_stable_count,
                "det_cert": det_cert_count,
                "res_cert": res_cert_count,
                "fn_rate_det": fn_rate_det,
                "fn_rate_res": fn_rate_res,
                "mean_det_margin": float(np.mean(det_margins_all)),
                "mean_res_margin": float(np.mean(res_margins_all)),
            }
        )

    # Print table
    header = (
        f"{'kind':26s} {'stable':>7} {'det':>5} {'res':>5} "
        f"{'fn_det':>7} {'fn_res':>7} {'mean_det':>10} {'mean_res':>10}"
    )
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for r in results:
        print(
            f"{r['kind']:26s} {r['true_stable']:7d} {r['det_cert']:5d} {r['res_cert']:5d} "
            f"{r['fn_rate_det']:7.3f} {r['fn_rate_res']:7.3f} "
            f"{r['mean_det_margin']:+10.3e} {r['mean_res_margin']:+10.3e}"
        )
    print(sep)
    print("(fn_det/fn_res = false-negative rate among truly stable matrices)")

    # Save margin profile plot for last instance of each kind
    rng2 = np.random.default_rng(args.seed + 9999)
    theta = np.linspace(0.0, 2.0 * np.pi, args.num_contour, endpoint=False)
    for kind in KINDS[:4]:  # plot first 4 for space
        a = build_matrix(kind, args.n, rng2)
        a0_scalar = scalar_reference(args.n, args.reference_radius)
        points = unit_circle_grid(args.num_contour)
        from roche.certificates import determinant_margins, resolvent_margins
        dm = determinant_margins(a, a0_scalar, points)
        rm = resolvent_margins(a, a0_scalar, points)
        plot_margin_comparison(
            theta, dm, rm,
            path=outdir / f"margins_{kind}.png",
            title=kind,
        )

    import csv
    csv_path = outdir / "summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["kind","true_stable","det_cert","res_cert",
                                               "fn_rate_det","fn_rate_res","mean_det_margin","mean_res_margin"])
        writer.writeheader()
        for r in results:
            writer.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in r.items()})

    print(f"\nFigures saved to {outdir}/")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--num-matrices", type=int, default=200)
    parser.add_argument("--num-contour", type=int, default=512)
    parser.add_argument("--reference-radius", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--outdir", type=str, default="results/exp1")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--quick", action="store_true",
                        help="Fast smoke: 3 matrices, K=64, first 2 families")
    args = parser.parse_args()
    if args.quick:
        args.num_matrices = 3
        args.num_contour = 64
    return args


if __name__ == "__main__":
    run(parse_args())
