"""Experiment P2-1: Reference quality comparison across methods and matrix families.

Compares four reference selection methods:
  scalar        -- A0 = 0.5 * I
  eig_shrunk    -- A0 = diag(0.9 * rho_i * exp(i*theta_i))
  random_search -- best of 200 random diagonal candidates
  gradient_opt  -- gradient ascent on min_margin (300 steps, Adam)

Usage:
    python experiments/reference_quality.py [--n N] [--m M] [--seed S] [--outdir DIR]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from roche.certificates import certify_on_unit_circle, is_schur_stable
from roche.matrices import (
    diagonal_plus_low_rank,
    jordan_like_matrix,
    random_nonnormal_matrix,
    random_normal_matrix,
)
from roche.reference import (
    diagonal_reference_from_eigs,
    random_search_diagonal_reference,
    scalar_reference,
)
from roche.reference_opt import optimise_diagonal_reference
from roche.ssm import generate_ar_sequence, train_diagonal_ssm

FAMILIES = [
    "diagonal_stable",
    "nonnormal_stable",
    "jordan_near_boundary",
    "trained_ssm",
]

METHODS = ["scalar", "eig_shrunk", "random_search", "gradient_opt"]


def build_matrix(kind: str, n: int, rng: np.random.Generator, seed: int) -> np.ndarray:
    if kind == "diagonal_stable":
        angles = rng.uniform(0.0, 2.0 * np.pi, n)
        r = rng.uniform(0.7, 0.95, n)
        return np.diag(r * np.exp(1j * angles))
    if kind == "nonnormal_stable":
        return random_nonnormal_matrix(n, 0.85, rng=rng, departure=20.0)
    if kind == "jordan_near_boundary":
        return jordan_like_matrix(n, 0.95 + 0.0j, superdiag=1.0)
    if kind == "trained_ssm":
        ar_coeffs = [-0.282, 0.481, -0.285, -0.532]
        rng2 = np.random.default_rng(seed)
        u = generate_ar_sequence(ar_coeffs, 500, 0.05, rng2)
        target = np.roll(u, -1); target[-1] = 0.0
        a_mat, _, _ = train_diagonal_ssm(u, target, n_state=n, n_iter=1000, seed=seed)
        return a_mat
    raise ValueError(kind)


def evaluate_reference(
    a: np.ndarray,
    method: str,
    num_points: int,
    seed: int,
) -> dict:
    n = a.shape[0]
    t0 = time.perf_counter()
    if method == "scalar":
        a0 = scalar_reference(n, 0.5)
    elif method == "eig_shrunk":
        a0 = diagonal_reference_from_eigs(a)
    elif method == "random_search":
        a0, _ = random_search_diagonal_reference(a, 200, num_points, "resolvent", seed)
    elif method == "gradient_opt":
        a0, _ = optimise_diagonal_reference(a, n_steps=300, num_contour_points=num_points, seed=seed)
    else:
        raise ValueError(method)
    wall = time.perf_counter() - t0
    cert = certify_on_unit_circle(a, a0, num_points, "resolvent")
    return {"min_margin": cert.min_margin, "certified": cert.certified, "wall_time": wall}


def run(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Results: family -> method -> list of per-trial dicts
    results: dict[str, dict[str, list]] = {f: {m: [] for m in METHODS} for f in FAMILIES}

    for kind in FAMILIES:
        print(f"\nFamily: {kind}")
        for trial in range(args.m):
            a = build_matrix(kind, args.n, rng, seed=trial)
            if not is_schur_stable(a):
                continue
            for method in METHODS:
                r = evaluate_reference(a, method, args.num_points, seed=trial)
                results[kind][method].append(r)
            if (trial + 1) % 10 == 0:
                print(f"  {trial+1}/{args.m}", end="\r")
        print()

    # Print summary table
    header = f"{'family':22s}  {'method':14s}  {'cert%':>6}  {'mean_margin':>12}  {'mean_time_ms':>12}"
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))
    for kind in FAMILIES:
        for method in METHODS:
            rs = results[kind][method]
            if not rs:
                continue
            cert_pct = 100.0 * np.mean([r["certified"] for r in rs])
            mean_m = np.mean([r["min_margin"] for r in rs])
            mean_t = 1e3 * np.mean([r["wall_time"] for r in rs])
            print(f"{kind:22s}  {method:14s}  {cert_pct:6.1f}  {mean_m:+12.4f}  {mean_t:12.2f}")

    # Figure: bar chart of mean min-margin by method and family
    fig, axes = plt.subplots(1, len(FAMILIES), figsize=(14, 4), sharey=False)
    colours = {"scalar": "C0", "eig_shrunk": "C1", "random_search": "C2", "gradient_opt": "C3"}
    for ax, kind in zip(axes, FAMILIES):
        means = [np.mean([r["min_margin"] for r in results[kind][m]]) for m in METHODS]
        ax.bar(METHODS, means, color=[colours[m] for m in METHODS])
        ax.axhline(0, color="k", lw=0.8, ls="--")
        ax.set_title(kind.replace("_", "\n"), fontsize=8)
        ax.set_xticks(range(len(METHODS)))
        ax.set_xticklabels([m.replace("_", "\n") for m in METHODS], fontsize=7)
        ax.set_ylabel("Mean min margin")
    fig.suptitle("Reference quality: mean resolvent min-margin by method")
    fig.tight_layout()
    fig.savefig(outdir / "reference_quality_margins.png", dpi=150)
    plt.close(fig)

    # Figure: gradient ascent convergence on one example per family
    fig, axes = plt.subplots(1, len(FAMILIES), figsize=(14, 3))
    rng2 = np.random.default_rng(args.seed + 99)
    for ax, kind in zip(axes, FAMILIES):
        a = build_matrix(kind, args.n, rng2, seed=0)
        if is_schur_stable(a):
            _, hist = optimise_diagonal_reference(a, n_steps=300, num_contour_points=args.num_points, seed=0)
            ax.plot(hist)
            ax.axhline(0, color="k", lw=0.8, ls="--")
            ax.set_xlabel("Step")
            ax.set_ylabel("Softmin margin")
            ax.set_title(kind.replace("_", "\n"), fontsize=8)
    fig.suptitle("Gradient ascent convergence")
    fig.tight_layout()
    fig.savefig(outdir / "reference_opt_convergence.png", dpi=150)
    plt.close(fig)

    print(f"\nFigures saved to {outdir}/")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--m", type=int, default=50, help="instances per family")
    p.add_argument("--num-points", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", type=str, default="results/p2exp1")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
