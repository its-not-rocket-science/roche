"""Exp P3-4: Verified vs. grid approximation error bounds.

True margin g_j is a random degree-TRUE_DEG polynomial with controlled minimum.
A degree-SURR_DEG surrogate Q_j is fitted at Chebyshev nodes (intentional
underfitting when SURR_DEG < TRUE_DEG).  The error polynomial e = g_j - Q_j
has exactly known coefficients; interval-arithmetic subdivision gives a
rigorous upper bound eps_rig >= max|e(t)| on [0,1].

We compare eps_grid (500-point grid estimate) with eps_rig (interval bound) and
count false positives: instances where the grid certifies (min Q > eps_grid) but
the verified bound invalidates (min Q <= eps_rig).

Grid:
  min_margin in {0.10, 0.15, 0.20}   (tight to large)
  N_INSTANCES = 200 per cell

Output: Table P3-4, scatter figure, results/p3exp4/
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from roche.surrogate import (
    chebyshev_nodes,
    fit_poly,
    verified_error_bound,
)

RESULTS_DIR = Path("results/p3exp4")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TRUE_DEG   = 8     # true margin polynomial degree
SURR_DEG   = 4     # surrogate degree (underfitting -> non-trivial error)
N_NODES    = 12    # Chebyshev nodes for fitting (> SURR_DEG+1 -> LS fit)
N_GRID     = 500   # points for grid estimate
N_SUB      = 1000  # interval sub-divisions for verified bound
N_INSTANCES = 300
MIN_MARGINS = [0.005, 0.010, 0.020]
RNG_SEED   = 99


def make_poly(degree: int, min_margin: float, rng: np.random.Generator) -> np.poly1d:
    """Random degree-d polynomial on [0,1] with minimum value = min_margin."""
    coeffs = rng.standard_normal(degree + 1)
    coeffs /= np.max(np.abs(coeffs)) + 1e-9
    p = np.poly1d(coeffs)
    t_dense = np.linspace(0.0, 1.0, 2000)
    shift = min_margin - float(np.min(p(t_dense)))
    coeffs[-1] += shift
    return np.poly1d(coeffs)


def run_instance(
    poly_true: np.poly1d,
) -> dict:
    """Fit surrogate, compute eps_grid and eps_rig, certify with each."""
    t_nodes = chebyshev_nodes(N_NODES)
    g_vals  = np.polyval(poly_true.coeffs, t_nodes).astype(np.float64)
    poly_q  = fit_poly(t_nodes, g_vals, SURR_DEG)

    # Error polynomial: coefficients known exactly
    error_poly = poly_true - poly_q   # np.poly1d subtraction

    # Grid estimate
    t_fine   = np.linspace(0.0, 1.0, N_GRID)
    eps_grid = float(np.max(np.abs(np.polyval(error_poly.coeffs, t_fine))))

    # Verified bound via interval subdivision
    eps_rig = verified_error_bound(error_poly.coeffs, n_sub=N_SUB)

    # Real-polynomial certificate: min_t Q(t) > eps
    q_fine  = np.polyval(poly_q.coeffs, t_fine).real
    min_q   = float(np.min(q_fine))

    cert_grid = min_q > eps_grid
    cert_rig  = min_q > eps_rig

    return {
        "eps_grid": eps_grid,
        "eps_rig":  eps_rig,
        "min_q":    min_q,
        "cert_grid": cert_grid,
        "cert_rig":  cert_rig,
        "false_pos": cert_grid and not cert_rig,   # grid yes, rig no
    }


def run_cell(min_margin: float, rng: np.random.Generator) -> list[dict]:
    results = []
    for _ in range(N_INSTANCES):
        poly_true = make_poly(TRUE_DEG, min_margin, rng)
        results.append(run_instance(poly_true))
    return results


def summarise(results: list[dict]) -> dict:
    eps_g  = np.array([r["eps_grid"] for r in results])
    eps_r  = np.array([r["eps_rig"]  for r in results])
    ratios = eps_r / np.maximum(eps_g, 1e-15)
    cert_g = np.array([r["cert_grid"] for r in results])
    fp     = np.array([r["false_pos"] for r in results])
    # Conditional false positive: fraction of grid-certified that rig rejects
    n_cert_g = cert_g.sum()
    fp_cond  = float(fp.sum() / n_cert_g) * 100 if n_cert_g > 0 else 0.0
    return {
        "mean_eps_grid": float(np.mean(eps_g)),
        "mean_eps_rig":  float(np.mean(eps_r)),
        "mean_ratio":    float(np.mean(ratios)),
        "cert_grid_pct": float(np.mean(cert_g)) * 100,
        "cert_rig_pct":  float(np.mean([r["cert_rig"] for r in results])) * 100,
        "false_pos_pct": float(np.mean(fp)) * 100,
        "false_pos_cond_pct": fp_cond,
    }


def main() -> None:
    rng = np.random.default_rng(RNG_SEED)
    all_rows: list[tuple[float, list[dict]]] = []

    for mm in MIN_MARGINS:
        print(f"  min_margin={mm:.2f}", flush=True)
        results = run_cell(mm, rng)
        all_rows.append((mm, results))

    # Print table
    print()
    hdr = (f"{'min_margin':>10}  {'eps_grid':>10}  {'eps_rig':>10}  "
           f"{'ratio':>6}  {'cert_grid%':>10}  {'cert_rig%':>9}  "
           f"{'fp%':>6}  {'fp%|grid':>8}")
    print(hdr)
    print("=" * len(hdr))
    for mm, results in all_rows:
        s = summarise(results)
        print(
            f"{mm:>10.3f}  {s['mean_eps_grid']:>10.4f}  {s['mean_eps_rig']:>10.4f}  "
            f"{s['mean_ratio']:>6.3f}  {s['cert_grid_pct']:>10.1f}  "
            f"{s['cert_rig_pct']:>9.1f}  {s['false_pos_pct']:>6.1f}  "
            f"{s['false_pos_cond_pct']:>8.1f}"
        )

    # Figure: scatter eps_rig vs eps_grid for all instances, coloured by false_pos
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for ax, (mm, results) in zip(axes, all_rows):
        eg = np.array([r["eps_grid"] for r in results])
        er = np.array([r["eps_rig"]  for r in results])
        fp = np.array([r["false_pos"] for r in results])
        ax.scatter(eg[~fp], er[~fp], s=8, alpha=0.5, color="steelblue", label="agree")
        ax.scatter(eg[ fp], er[ fp], s=20, alpha=0.9, color="crimson",  label="false pos")
        lo = min(eg.min(), er.min()) * 0.9
        hi = max(eg.max(), er.max()) * 1.1
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.8, label="y = x")
        ax.set_xlabel(r"$\hat\epsilon^{\rm grid}$")
        ax.set_ylabel(r"$\epsilon^{\rm rig}$")
        ax.set_title(f"min margin = {mm}")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "verified_vs_grid.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved to {RESULTS_DIR}/verified_vs_grid.pdf")

    # Save summary CSV for paper
    with open(RESULTS_DIR / "summary.txt", "w") as f:
        f.write("min_margin,mean_eps_grid,mean_eps_rig,mean_ratio,"
                "cert_grid_pct,cert_rig_pct,false_pos_pct\n")
        for mm, results in all_rows:
            s = summarise(results)
            f.write(
                f"{mm},{s['mean_eps_grid']:.6f},{s['mean_eps_rig']:.6f},"
                f"{s['mean_ratio']:.4f},{s['cert_grid_pct']:.1f},"
                f"{s['cert_rig_pct']:.1f},{s['false_pos_pct']:.1f},"
                f"{s['false_pos_cond_pct']:.1f}\n"
            )


if __name__ == "__main__":
    main()
