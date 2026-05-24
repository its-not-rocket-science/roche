"""Exp P3-2+3: Rouche certificates on small neural networks + scaling limits.

Phase 2: Train 2-layer MLP on two-moons.  For 200 same-class path pairs
         (straight-line interpolation), fit Chebyshev surrogates of degree
         4, 8, 16 and compare real_poly vs rouche vs dense ground truth.

Phase 3: Scaling limits.  Fix degree=8, vary path scale (1x, 2x, 4x, 8x)
         by stretching the straight-line interpolation.  Report how cert rate
         and approx error degrade with path length.

Output: Table P3-2 (method comparison), Table P3-3 (scaling), figures,
        saved to results/p3exp2/
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from roche.path_cert import certify_path

RESULTS_DIR = Path("results/p3exp2")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

N_PATHS = 200
DEGREES = [4, 8, 16]
PATH_SCALES = [1, 2, 4, 8]
RNG_SEED = 0
N_TRAIN = 1000
N_HIDDEN = 32
N_EPOCHS = 300
LR = 1e-3


# ---------------------------------------------------------------------------
# Tiny MLP (no external dependencies beyond numpy/torch)
# ---------------------------------------------------------------------------

def make_two_moons(n: int, rng: np.random.Generator, noise: float = 0.15):
    from sklearn.datasets import make_moons
    X, y = make_moons(n_samples=n, noise=noise, random_state=int(rng.integers(0, 10000)))
    return X.astype(np.float32), y.astype(np.int64)


def train_mlp(X_train, y_train, n_hidden: int, n_epochs: int, lr: float, rng):
    import torch
    import torch.nn as nn
    torch.manual_seed(int(rng.integers(0, 10000)))
    model = nn.Sequential(
        nn.Linear(2, n_hidden), nn.Tanh(),
        nn.Linear(n_hidden, n_hidden), nn.Tanh(),
        nn.Linear(n_hidden, 2),
    )
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    X_t = torch.tensor(X_train)
    y_t = torch.tensor(y_train)
    for ep in range(n_epochs):
        opt.zero_grad()
        logits = model(X_t)
        loss = nn.functional.cross_entropy(logits, y_t)
        loss.backward()
        opt.step()
        if (ep + 1) % 50 == 0:
            print(f"    epoch {ep+1}/{n_epochs}  loss={loss.item():.4f}", flush=True)
    return model


def model_margins(model, x_batch):
    """Return logits[0] - logits[1] for each row in x_batch (class 0 margin)."""
    with torch.no_grad():
        logits = model(torch.tensor(x_batch, dtype=torch.float32)).numpy()
    return logits[:, 0] - logits[:, 1]   # positive => predicted class 0


def make_path_margin_fn(model, x_a, x_b, scale: float = 1.0, pred_class: int = 0):
    """g(t) = logit_{pred_class}(gamma(t)) - logit_{1-pred_class}(gamma(t)).

    gamma(t) = x_a + t*(x_b - x_a) * scale  (scale>1 extends path beyond x_b).
    """
    import torch
    sign = 1 if pred_class == 0 else -1

    def g_fn(t: float) -> float:
        x_t = x_a + t * (x_b - x_a) * scale
        x_tensor = torch.tensor(x_t[None], dtype=torch.float32)
        with torch.no_grad():
            logits = model(x_tensor).numpy()[0]
        return float(sign * (logits[0] - logits[1]))

    return g_fn


# ---------------------------------------------------------------------------
# Phase 2: method comparison
# ---------------------------------------------------------------------------

def _run_phase2(model, X_test, y_test, rng: np.random.Generator,
                n_paths: int, degrees: list) -> list[dict]:
    """Same-class pairs x degrees x methods."""
    idx0 = np.where(y_test == 0)[0]
    pairs = []
    while len(pairs) < n_paths:
        i, j = rng.choice(idx0, size=2, replace=False)
        pairs.append((X_test[i], X_test[j]))

    results = []
    for deg in degrees:
        print(f"  degree={deg}", flush=True)
        dense_cert = real_cert = rouche_cert = 0
        approx_errors = []
        for x_a, x_b in pairs:
            g_fn = make_path_margin_fn(model, x_a, x_b, scale=1.0, pred_class=0)
            res = certify_path(g_fn, degree=deg, contour_width=0.3, n_contour=200)
            if res["dense"].certified:
                dense_cert += 1
            if res["real_poly"].certified:
                real_cert += 1
            if res["rouche"].certified:
                rouche_cert += 1
            approx_errors.append(res["real_poly"].approx_error)
        n = len(pairs)
        results.append({
            "degree": deg,
            "dense_cert%": 100 * dense_cert / n,
            "real_cert%":  100 * real_cert  / n,
            "rouche_cert%": 100 * rouche_cert / n,
            "mean_approx_error": float(np.mean(approx_errors)),
        })
    return results


# ---------------------------------------------------------------------------
# Phase 3: scaling limits
# ---------------------------------------------------------------------------

def _run_phase3(model, X_test, y_test, rng: np.random.Generator, n_paths: int) -> list[dict]:
    """Fix degree=8, vary path scale."""
    idx0 = np.where(y_test == 0)[0]
    pairs = []
    while len(pairs) < n_paths:
        i, j = rng.choice(idx0, size=2, replace=False)
        pairs.append((X_test[i], X_test[j]))

    deg = 8
    results = []
    for scale in PATH_SCALES:
        print(f"  scale={scale}", flush=True)
        real_cert = rouche_cert = 0
        approx_errors = []
        for x_a, x_b in pairs:
            g_fn = make_path_margin_fn(model, x_a, x_b, scale=float(scale), pred_class=0)
            res = certify_path(g_fn, degree=deg, contour_width=0.3, n_contour=200)
            if res["real_poly"].certified:
                real_cert += 1
            if res["rouche"].certified:
                rouche_cert += 1
            approx_errors.append(res["real_poly"].approx_error)
        n = len(pairs)
        results.append({
            "scale": scale,
            "real_cert%":   100 * real_cert  / n,
            "rouche_cert%": 100 * rouche_cert / n,
            "mean_approx_error": float(np.mean(approx_errors)),
        })
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    import csv
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="Fast smoke: 5 paths, deg=4 only, 10 epochs")
    parser.add_argument("--outdir", type=str, default=str(RESULTS_DIR))
    args = parser.parse_args()

    n_paths = 5 if args.quick else N_PATHS
    degrees = [4] if args.quick else DEGREES
    n_epochs = 10 if args.quick else N_EPOCHS
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(RNG_SEED)

    print("Generating two-moons data...", flush=True)
    X_train, y_train = make_two_moons(N_TRAIN, rng)
    X_test,  y_test  = make_two_moons(500,     rng)

    print("Training MLP...", flush=True)
    model = train_mlp(X_train, y_train, N_HIDDEN, n_epochs, LR, rng)

    import torch
    with torch.no_grad():
        logits = model(torch.tensor(X_test)).numpy()
    pred = np.argmax(logits, axis=1)
    acc = float(np.mean(pred == y_test))
    print(f"Test accuracy: {acc*100:.1f}%", flush=True)

    print("\nPhase 2: method comparison...", flush=True)
    p2_results = _run_phase2(model, X_test, y_test, rng, n_paths, degrees)

    print("\nPhase 3: scaling limits...", flush=True)
    p3_results = _run_phase3(model, X_test, y_test, rng, n_paths)

    # Print tables
    print()
    print(f"{'degree':>8}  {'dense%':>7}  {'real%':>7}  {'rouche%':>8}  {'mean_eps':>10}")
    print("=" * 55)
    for r in p2_results:
        print(
            f"{r['degree']:>8}  {r['dense_cert%']:>7.1f}  {r['real_cert%']:>7.1f}"
            f"  {r['rouche_cert%']:>8.1f}  {r['mean_approx_error']:>10.4e}"
        )

    print()
    print(f"{'scale':>7}  {'real%':>7}  {'rouche%':>8}  {'mean_eps':>10}")
    print("=" * 40)
    for r in p3_results:
        print(
            f"{r['scale']:>7}  {r['real_cert%']:>7.1f}"
            f"  {r['rouche_cert%']:>8.1f}  {r['mean_approx_error']:>10.4e}"
        )

    # CSV output
    with open(outdir / "summary_phase2.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["degree","dense_pct","real_pct","rouche_pct","mean_approx_error"])
        for r in p2_results:
            writer.writerow([r["degree"], f"{r['dense_cert%']:.1f}", f"{r['real_cert%']:.1f}",
                             f"{r['rouche_cert%']:.1f}", f"{r['mean_approx_error']:.4e}"])
    with open(outdir / "summary_phase3.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["scale","real_pct","rouche_pct","mean_approx_error"])
        for r in p3_results:
            writer.writerow([r["scale"], f"{r['real_cert%']:.1f}",
                             f"{r['rouche_cert%']:.1f}", f"{r['mean_approx_error']:.4e}"])

    if not args.quick:
        # Figures
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))

        ax = axes[0]
        degs = [r["degree"] for r in p2_results]
        real_rates = [r["real_cert%"] for r in p2_results]
        rouche_rates = [r["rouche_cert%"] for r in p2_results]
        dense_rates = [r["dense_cert%"] for r in p2_results]
        x = np.arange(len(degs))
        ax.bar(x - 0.25, dense_rates,  width=0.25, label="Dense (truth)", alpha=0.8)
        ax.bar(x,        real_rates,   width=0.25, label="Real poly",      alpha=0.8)
        ax.bar(x + 0.25, rouche_rates, width=0.25, label="Rouché",         alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([f"deg={d}" for d in degs])
        ax.set_ylabel("cert %")
        ax.set_title("Two-moons: method comparison")
        ax.legend()
        ax.set_ylim(0, 110)

        ax2 = axes[1]
        scales = [r["scale"] for r in p3_results]
        real_s = [r["real_cert%"] for r in p3_results]
        rouche_s = [r["rouche_cert%"] for r in p3_results]
        eps_s = [r["mean_approx_error"] for r in p3_results]
        ax2.plot(scales, real_s,   "o-", label="Real poly cert%")
        ax2.plot(scales, rouche_s, "s-", label="Rouché cert%")
        ax2.set_xlabel("path scale")
        ax2.set_ylabel("cert %")
        ax2.set_title("Scaling limits (degree=8)")
        ax2_r = ax2.twinx()
        ax2_r.plot(scales, eps_s, "^--", color="gray", label="approx error")
        ax2_r.set_ylabel("mean approx error")
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_r.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

        fig.tight_layout()
        fig.savefig(outdir / "two_moons_certs.pdf", bbox_inches="tight")
        plt.close(fig)

        # Decision boundary plot
        fig3, ax3 = plt.subplots(figsize=(5, 4))
        xx, yy = np.meshgrid(np.linspace(-2.5, 3.5, 200), np.linspace(-1.5, 2.5, 200))
        import torch
        grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
        with torch.no_grad():
            logits_grid = model(grid).numpy()
        Z = (logits_grid[:, 0] - logits_grid[:, 1]).reshape(xx.shape)
        ax3.contourf(xx, yy, Z, levels=[-10, 0, 10], colors=["#FFDDC1", "#C1E1FF"], alpha=0.5)
        ax3.contour(xx, yy, Z, levels=[0], colors="k", linewidths=1)
        ax3.scatter(X_test[y_test == 0, 0], X_test[y_test == 0, 1], s=5, c="blue", alpha=0.4)
        ax3.scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1], s=5, c="red",  alpha=0.4)
        ax3.set_title("Two-moons decision boundary")
        fig3.tight_layout()
        fig3.savefig(outdir / "decision_boundary.pdf", bbox_inches="tight")
        plt.close(fig3)

    print(f"\nFigures saved to {outdir}/")


if __name__ == "__main__":
    main()
