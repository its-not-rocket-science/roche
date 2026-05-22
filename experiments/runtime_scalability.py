"""Experiment 4: Runtime and scalability of certificate computation.

Measures wall-clock time for determinant and resolvent certificates as a
function of matrix dimension n and contour grid size K.

Usage:
    python experiments/runtime_scalability.py [--seed S] [--outdir DIR]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from roche.certificates import certify_on_unit_circle
from roche.matrices import random_normal_matrix, diagonal_plus_low_rank
from roche.reference import scalar_reference

DIMS = [4, 8, 16, 32, 64, 128, 256]
GRID_SIZES = [64, 128, 256, 512]
N_REPEATS = 3


def time_certificate(a, a0, num_points, method, n_repeats=N_REPEATS):
    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        certify_on_unit_circle(a, a0, num_points=num_points, method=method)
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def run(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("Timing resolvent certificate (dense normal matrix)")
    print(f"{'n':>6}  " + "  ".join(f"K={K:4d}" for K in GRID_SIZES))
    print("-" * (8 + 10 * len(GRID_SIZES)))

    res_times = np.zeros((len(DIMS), len(GRID_SIZES)))
    det_times = np.zeros((len(DIMS), len(GRID_SIZES)))

    for i, n in enumerate(DIMS):
        a = random_normal_matrix(n, 0.85, rng)
        a0 = scalar_reference(n, 0.5)
        row_res = []
        row_det = []
        for j, K in enumerate(GRID_SIZES):
            t_res = time_certificate(a, a0, K, "resolvent")
            t_det = time_certificate(a, a0, K, "determinant")
            res_times[i, j] = t_res
            det_times[i, j] = t_det
            row_res.append(f"{t_res*1e3:8.2f}ms")
            row_det.append(f"{t_det*1e3:8.2f}ms")
        print(f"{n:6d}  res: " + "  ".join(row_res))
        print(f"{'':6}  det: " + "  ".join(row_det))

    print()
    print("Timing resolvent certificate (diagonal-plus-low-rank, rank=n//8)")
    print(f"{'n':>6}  " + "  ".join(f"K={K:4d}" for K in GRID_SIZES))

    dlr_times = np.zeros((len(DIMS), len(GRID_SIZES)))
    for i, n in enumerate(DIMS):
        rank = max(1, n // 8)
        a = diagonal_plus_low_rank(n, 0.85, rank, rng)
        a0 = scalar_reference(n, 0.5)
        row = []
        for j, K in enumerate(GRID_SIZES):
            t = time_certificate(a, a0, K, "resolvent")
            dlr_times[i, j] = t
            row.append(f"{t*1e3:8.2f}ms")
        print(f"{n:6d}  " + "  ".join(row))

    # Scaling plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    K_idx = GRID_SIZES.index(256) if 256 in GRID_SIZES else 2
    ax = axes[0]
    ax.loglog(DIMS, res_times[:, K_idx] * 1e3, "o-", label="resolvent (dense)")
    ax.loglog(DIMS, det_times[:, K_idx] * 1e3, "s--", label="determinant (dense)")
    ax.loglog(DIMS, dlr_times[:, K_idx] * 1e3, "^:", label="resolvent (DLR)")
    ax.set_xlabel("Matrix dimension n")
    ax.set_ylabel("Time (ms, K=256)")
    ax.set_title("Wall-clock time vs. matrix dimension")
    ax.legend()

    n_idx = DIMS.index(32) if 32 in DIMS else 3
    ax = axes[1]
    ax.semilogy(GRID_SIZES, res_times[n_idx, :] * 1e3, "o-", label="resolvent (dense)")
    ax.semilogy(GRID_SIZES, det_times[n_idx, :] * 1e3, "s--", label="determinant (dense)")
    ax.set_xlabel("Grid size K")
    ax.set_ylabel("Time (ms, n=32)")
    ax.set_title("Wall-clock time vs. contour grid size")
    ax.legend()

    fig.tight_layout()
    fig.savefig(outdir / "runtime_scaling.png", dpi=200)
    plt.close(fig)

    print(f"\nFigures saved to {outdir}/")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--outdir", type=str, default="results/exp4")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
