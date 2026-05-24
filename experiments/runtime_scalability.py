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

    dims = [4, 8] if args.quick else DIMS
    grid_sizes = [64, 128] if args.quick else GRID_SIZES

    print("Timing resolvent certificate (dense normal matrix)")
    print(f"{'n':>6}  " + "  ".join(f"K={K:4d}" for K in grid_sizes))
    print("-" * (8 + 10 * len(grid_sizes)))

    res_times = np.zeros((len(dims), len(grid_sizes)))
    det_times = np.zeros((len(dims), len(grid_sizes)))

    for i, n in enumerate(dims):
        a = random_normal_matrix(n, 0.85, rng)
        a0 = scalar_reference(n, 0.5)
        row_res = []
        row_det = []
        for j, K in enumerate(grid_sizes):
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
    print(f"{'n':>6}  " + "  ".join(f"K={K:4d}" for K in grid_sizes))

    dlr_times = np.zeros((len(dims), len(grid_sizes)))
    for i, n in enumerate(dims):
        rank = max(1, n // 8)
        a = diagonal_plus_low_rank(n, 0.85, rank, rng)
        a0 = scalar_reference(n, 0.5)
        row = []
        for j, K in enumerate(grid_sizes):
            t = time_certificate(a, a0, K, "resolvent")
            dlr_times[i, j] = t
            row.append(f"{t*1e3:8.2f}ms")
        print(f"{n:6d}  " + "  ".join(row))

    # CSV output
    import csv
    with open(outdir / "summary.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["matrix_type","n","K","time_ms"])
        for i, n in enumerate(dims):
            for j, K in enumerate(grid_sizes):
                writer.writerow(["resolvent_dense", n, K, f"{res_times[i,j]*1e3:.4f}"])
                writer.writerow(["determinant_dense", n, K, f"{det_times[i,j]*1e3:.4f}"])
                writer.writerow(["resolvent_dlr", n, K, f"{dlr_times[i,j]*1e3:.4f}"])

    if not args.quick:
        # Scaling plots
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        K_ref = 256 if 256 in grid_sizes else grid_sizes[-1]
        K_idx = grid_sizes.index(K_ref)
        ax = axes[0]
        ax.loglog(dims, res_times[:, K_idx] * 1e3, "o-", label="resolvent (dense)")
        ax.loglog(dims, det_times[:, K_idx] * 1e3, "s--", label="determinant (dense)")
        ax.loglog(dims, dlr_times[:, K_idx] * 1e3, "^:", label="resolvent (DLR)")
        ax.set_xlabel("Matrix dimension n")
        ax.set_ylabel(f"Time (ms, K={K_ref})")
        ax.set_title("Wall-clock time vs. matrix dimension")
        ax.legend()

        n_ref = 32 if 32 in dims else dims[-1]
        n_idx = dims.index(n_ref)
        ax = axes[1]
        ax.semilogy(grid_sizes, res_times[n_idx, :] * 1e3, "o-", label="resolvent (dense)")
        ax.semilogy(grid_sizes, det_times[n_idx, :] * 1e3, "s--", label="determinant (dense)")
        ax.set_xlabel("Grid size K")
        ax.set_ylabel(f"Time (ms, n={n_ref})")
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
    parser.add_argument("--quick", action="store_true",
                        help="Fast smoke: dims=[4,8], grid_sizes=[64,128]")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
