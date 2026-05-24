"""Exp P3-1: Rouche path certificates on synthetic polynomial classifiers.

True margin g_j(t) is a polynomial of controlled degree and minimum value.
We fit surrogate Q of matching degree and compare three methods:
  - dense:     ground truth (direct sampling)
  - real_poly: min Q(t) > eps certificate
  - rouche:    Rouche zero-counting certificate

Grid:
  degree_true  in {2, 4, 8}
  min_margin   in {0.5, 0.2, 0.05}
  100 random instances per cell

Output: Table P3-1, convergence figure, results/p3exp1/
"""
from __future__ import annotations

import sys
import argparse
import csv
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from roche.path_cert import certify_path


RESULTS_DIR = Path("results/p3exp1")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEGREES = [2, 4, 8]
MIN_MARGINS = [0.5, 0.2, 0.05]
N_INSTANCES = 100
CONTOUR_WIDTH = 0.3
N_CONTOUR = 200
RNG_SEED = 42


def make_random_margin_fn(
    degree: int,
    min_margin: float,
    rng: np.random.Generator,
) -> callable:
    """Random degree-d polynomial on [0,1] with min value = min_margin.

    Sample random roots/coefficients, then shift so minimum = min_margin.
    """
    # Random polynomial: coefficients in standard normal, normalised
    raw_coeffs = rng.standard_normal(degree + 1)
    raw_coeffs /= np.max(np.abs(raw_coeffs)) + 1e-9
    poly_raw = np.poly1d(raw_coeffs)
    # Find minimum on [0,1] by dense sampling
    t_dense = np.linspace(0.0, 1.0, 2000)
    vals = poly_raw(t_dense)
    shift = min_margin - np.min(vals)
    poly_shifted = np.poly1d(np.append(raw_coeffs[:-1], raw_coeffs[-1] + shift))

    def g_fn(t: float) -> float:
        return float(poly_shifted(t))

    return g_fn



def run_cell_n(deg: int, min_margin: float, rng: np.random.Generator, n: int) -> dict:
    counts = {"dense": 0, "real_poly": 0, "rouche": 0}
    margins = {"real_poly": [], "rouche": []}
    for _ in range(n):
        g_fn = make_random_margin_fn(deg, min_margin, rng)
        results = certify_path(g_fn, degree=deg, contour_width=CONTOUR_WIDTH, n_contour=N_CONTOUR)
        for method, res in results.items():
            if res.certified:
                counts[method] += 1
            if method in margins:
                margins[method].append(res.margin)
    return {
        "cert_rate": {m: counts[m] / n * 100 for m in counts},
        "mean_margin": {m: float(np.mean(margins[m])) for m in margins},
    }


def main(args=None) -> None:
    if args is None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--quick", action="store_true",
                            help="Fast smoke: 2 instances, deg=2, min_margin=0.2")
        parser.add_argument("--outdir", type=str, default=str(RESULTS_DIR))
        parser.add_argument("--seed", type=int, default=RNG_SEED)
        args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    degrees = [2] if args.quick else DEGREES
    min_margins = [0.2] if args.quick else MIN_MARGINS
    n_instances = 2 if args.quick else N_INSTANCES

    rows = []
    for deg, mm in product(degrees, min_margins):
        print(f"  deg={deg} min={mm:.2f}", flush=True)
        cell = run_cell_n(deg, mm, rng, n_instances)
        rows.append((deg, mm, cell))

    # Print table
    print()
    header = f"{'degree':>8}  {'min_margin':>10}  {'dense%':>7}  {'real%':>7}  {'rouche%':>8}  {'real_margin':>12}  {'rouche_margin':>14}"
    print(header)
    print("=" * len(header))
    for deg, mm, cell in rows:
        cr = cell["cert_rate"]
        mg = cell["mean_margin"]
        print(
            f"{deg:>8}  {mm:>10.2f}  {cr['dense']:>7.1f}  {cr['real_poly']:>7.1f}"
            f"  {cr['rouche']:>8.1f}  {mg['real_poly']:>+12.4f}  {mg['rouche']:>+14.4f}"
        )

    # CSV output
    with open(outdir / "summary.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["degree","min_margin","dense_pct","real_pct","rouche_pct",
                         "mean_margin_real","mean_margin_rouche"])
        for deg, mm, cell in rows:
            cr = cell["cert_rate"]
            mg = cell["mean_margin"]
            writer.writerow([deg, mm, f"{cr['dense']:.1f}", f"{cr['real_poly']:.1f}",
                             f"{cr['rouche']:.1f}", f"{mg['real_poly']:.4f}", f"{mg['rouche']:.4f}"])

    if not args.quick:
        # Figure: cert rate heatmap
        methods = ["dense", "real_poly", "rouche"]
        method_labels = ["Dense (truth)", "Real poly", "Rouché"]
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
        for ax, method, label in zip(axes, methods, method_labels):
            data = np.zeros((len(DEGREES), len(MIN_MARGINS)))
            for i, deg in enumerate(DEGREES):
                for j, mm in enumerate(MIN_MARGINS):
                    cell = next(c for d, m, c in rows if d == deg and m == mm)
                    data[i, j] = cell["cert_rate"][method]
            im = ax.imshow(data, vmin=0, vmax=100, cmap="RdYlGn", aspect="auto")
            ax.set_xticks(range(len(MIN_MARGINS)))
            ax.set_xticklabels([str(m) for m in MIN_MARGINS])
            ax.set_yticks(range(len(DEGREES)))
            ax.set_yticklabels([str(d) for d in DEGREES])
            ax.set_xlabel("min margin")
            ax.set_ylabel("poly degree")
            ax.set_title(label)
            for ii in range(len(DEGREES)):
                for jj in range(len(MIN_MARGINS)):
                    ax.text(jj, ii, f"{data[ii,jj]:.0f}", ha="center", va="center", fontsize=9)
            plt.colorbar(im, ax=ax, label="cert %")
        fig.tight_layout()
        fig.savefig(outdir / "cert_rate_heatmap.pdf", bbox_inches="tight")
        plt.close(fig)

        # Figure: margin distribution for deg=4, mm=0.2
        target_deg, target_mm = 4, 0.2
        rng2 = np.random.default_rng(args.seed + 1)
        real_margins, rouche_margins = [], []
        for _ in range(N_INSTANCES):
            g_fn = make_random_margin_fn(target_deg, target_mm, rng2)
            res = certify_path(g_fn, degree=target_deg, contour_width=CONTOUR_WIDTH)
            real_margins.append(res["real_poly"].margin)
            rouche_margins.append(res["rouche"].rouche_margin)
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.hist(real_margins, bins=30, alpha=0.6, label="real poly margin")
        ax2.hist(rouche_margins, bins=30, alpha=0.6, label="Rouché margin")
        ax2.axvline(0, color="k", linewidth=1, linestyle="--")
        ax2.set_xlabel("margin")
        ax2.set_ylabel("count")
        ax2.set_title(f"degree={target_deg}, min_margin={target_mm}")
        ax2.legend()
        fig2.tight_layout()
        fig2.savefig(outdir / "margin_distribution.pdf", bbox_inches="tight")
        plt.close(fig2)

    print(f"\nFigures saved to {outdir}/")


if __name__ == "__main__":
    main()
