"""Experiment 3: Contour certificates on trained diagonal SSMs.

Trains small diagonal complex SSMs on a synthetic long-memory AR task, then
evaluates Rouché-type certificates on the learned transition matrices.

Usage:
    python experiments/trained_ssm.py [--n-state N] [--seed S] [--outdir DIR]
                                       [--seq-len T] [--n-iter I]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from roche.certificates import (
    certify_on_unit_circle,
    is_schur_stable,
    spectral_radius,
    unit_circle_grid,
    resolvent_margins,
    resolvent_lipschitz_bound,
)
from roche.plotting import plot_margin_comparison
from roche.reference import scalar_reference, diagonal_reference_from_eigs, random_search_diagonal_reference
from roche.ssm import generate_ar_sequence, train_diagonal_ssm, schur_stability_summary


def make_ar_coeffs(n_poles: int, rng: np.random.Generator, max_radius: float = 0.92) -> list[float]:
    """Generate stable AR coefficients by placing complex conjugate pole pairs."""
    poles = []
    for _ in range(n_poles // 2):
        r = rng.uniform(0.7, max_radius)
        omega = rng.uniform(0.1, np.pi - 0.1)
        poles.extend([r * np.exp(1j * omega), r * np.exp(-1j * omega)])
    if n_poles % 2 == 1:
        poles.append(complex(rng.uniform(0.5, max_radius)))
    # Convert poles to AR coefficients via Vieta's formulas (polynomial product)
    poly = np.poly(poles)  # monic polynomial with given roots
    # poly[0]=1, poly[1..p] are the polynomial coefficients; AR coeffs = -poly[1:]
    ar_coeffs = [-float(c.real) for c in poly[1:]]
    return ar_coeffs


def run(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Experiment 3: trained diagonal SSM  n_state={args.n_state}  seq_len={args.seq_len}")
    print("=" * 70)

    configs = [
        {"desc": "long-memory AR(4)", "ar_poles": 4, "max_r": 0.92},
        {"desc": "long-memory AR(4) hard", "ar_poles": 4, "max_r": 0.97},
        {"desc": "short-memory AR(2)", "ar_poles": 2, "max_r": 0.70},
        {"desc": "oscillatory AR(6)", "ar_poles": 6, "max_r": 0.88},
    ]

    if args.quick:
        configs = configs[:1]

    csv_rows = []
    for cfg in configs:
        ar_coeffs = make_ar_coeffs(cfg["ar_poles"], rng, max_radius=cfg["max_r"])
        u_seq = generate_ar_sequence(ar_coeffs, args.seq_len, noise_std=0.05, rng=rng)
        target_seq = np.roll(u_seq, -1)  # next-step prediction
        target_seq[-1] = 0.0

        print(f"\nTask: {cfg['desc']}")
        print(f"  AR coeffs: {[f'{c:.3f}' for c in ar_coeffs]}")

        a_matrix, final_loss, converged = train_diagonal_ssm(
            u_seq, target_seq,
            n_state=args.n_state,
            n_iter=args.n_iter,
            seed=int(rng.integers(10000)),
        )

        summary = schur_stability_summary(a_matrix)
        print(f"  Training: loss={final_loss:.4e}  converged={converged}")
        print(f"  Spectral radius: {summary['spectral_radius']:.4f}  "
              f"Schur stable: {summary['schur_stable']}")

        # Evaluate with scalar reference
        a0_scalar = scalar_reference(args.n_state, 0.5)
        res_scalar = certify_on_unit_circle(a_matrix, a0_scalar, num_points=512, method="resolvent")
        det_scalar = certify_on_unit_circle(a_matrix, a0_scalar, num_points=512, method="determinant")

        # Evaluate with shrunken-eigenvalue diagonal reference (if stable)
        if summary["schur_stable"]:
            a0_diag = diagonal_reference_from_eigs(a_matrix, shrink=0.9)
        else:
            a0_diag = a0_scalar
        res_diag = certify_on_unit_circle(a_matrix, a0_diag, num_points=512, method="resolvent")
        det_diag = certify_on_unit_circle(a_matrix, a0_diag, num_points=512, method="determinant")

        # Evaluate with best random-search diagonal reference
        a0_best, best_score = random_search_diagonal_reference(
            a_matrix, num_candidates=100, num_points=256, method="resolvent", seed=0
        )
        res_best = certify_on_unit_circle(a_matrix, a0_best, num_points=512, method="resolvent")

        print(f"  Scalar ref:       det={int(det_scalar.certified)} (m={det_scalar.min_margin:+.3e})  "
              f"res={int(res_scalar.certified)} (m={res_scalar.min_margin:+.3e})")
        print(f"  Diag-shrunk ref:  det={int(det_diag.certified)} (m={det_diag.min_margin:+.3e})  "
              f"res={int(res_diag.certified)} (m={res_diag.min_margin:+.3e})")
        print(f"  Best-search ref:  res={int(res_best.certified)} (m={res_best.min_margin:+.3e})")

        # Compute Lipschitz bound for rigorous certificate with best reference
        points = unit_circle_grid(512)
        L = resolvent_lipschitz_bound(a_matrix, a0_best, points)
        from roche.certificates import resolvent_margins, grid_certificate
        margins = resolvent_margins(a_matrix, a0_best, points)
        rig = grid_certificate(margins, lipschitz_bound=L, method="resolvent")
        print(f"  Rigorous cert (best ref, N=512): certified={rig.certified}  "
              f"L={L:.3e}  pi*L/N={rig.criterion_value:.3e}")
        csv_rows.append({
            "task": cfg["desc"],
            "final_loss": f"{final_loss:.6e}",
            "converged": int(converged),
            "spectral_radius": f"{summary['spectral_radius']:.6f}",
            "schur_stable": int(summary["schur_stable"]),
            "res_scalar_margin": f"{res_scalar.min_margin:.6e}",
            "res_diag_margin": f"{res_diag.min_margin:.6e}",
            "res_best_margin": f"{res_best.min_margin:.6e}",
            "rigorous_cert": int(rig.certified),
        })

        # Save margin plot
        theta = np.linspace(0, 2 * np.pi, 512, endpoint=False)
        from roche.certificates import determinant_margins
        dm = determinant_margins(a_matrix, a0_best, points)
        rm = resolvent_margins(a_matrix, a0_best, points)
        tag = cfg["desc"].replace(" ", "_").replace("(", "").replace(")", "")
        plot_margin_comparison(
            theta, dm, rm,
            path=outdir / f"margins_{tag}.png",
            title=f"{cfg['desc']}  rho={summary['spectral_radius']:.4f}",
        )

    import csv
    with open(outdir / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["task","final_loss","converged","spectral_radius",
                                               "schur_stable","res_scalar_margin","res_diag_margin",
                                               "res_best_margin","rigorous_cert"])
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"\nFigures saved to {outdir}/")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-state", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=500)
    parser.add_argument("--n-iter", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--outdir", type=str, default="results/exp3")
    parser.add_argument("--quick", action="store_true",
                        help="Fast smoke: 1 config, 100 iterations")
    args = parser.parse_args()
    if args.quick:
        args.n_iter = 100
    return args


if __name__ == "__main__":
    run(parse_args())
