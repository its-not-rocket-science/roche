"""Exp P2-3: Stability regularisers under genuine gradient pressure.

Training regime designed to push unconstrained eigenvalues outside the unit
disk: lr=0.1, adversarial near-boundary initialisation (radii at 0.96), and
a long-memory AR(4) task (rho=0.97, seq_len=2000).  Without any regulariser,
several seeds overshoot the unit circle.  Regularisers use margin=0.1
(firing threshold at rho=0.9) so they provide gradient tension before
escape, not just at the boundary.

Methods: none, spectral, lyapunov, contour_barrier.
10 seeds per method.  n_state=8, n_epochs=300.

Output: Table P2-3 with instability rate + cert%, results/p2exp3/
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import functools

from roche.certificates import certify_on_unit_circle
from roche.reference import diagonal_reference_from_eigs
from roche.regularisers import lyapunov_penalty, make_contour_barrier, spectral_penalty
from roche.ssm import generate_ar_sequence, train_diagonal_ssm_adam

RESULTS_DIR = Path("results/p2exp3")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

N_STATE    = 8
N_EPOCHS   = 300
LR         = 0.1        # high lr creates gradient-pressure instability
REG_WEIGHT = 1.0        # strong enough to counter high-lr gradients
REG_MARGIN = 0.1        # fire at rho=0.9; provides tension before unit-circle escape
SEQ_LEN    = 2000
N_SEEDS    = 10
K_CONTOUR  = 256
INIT_LOG_R = np.log(0.96)   # near-boundary adversarial init

# AR(4) with poles close to unit circle (rho ≈ 0.97)
AR_COEFFS  = [1.2, -0.5, 0.2, -0.08]   # rho ≈ 0.97 after normalization
NOISE_STD  = 0.05


def _ar_task(seed: int):
    rng = np.random.default_rng(seed + 9999)
    u = generate_ar_sequence(AR_COEFFS, SEQ_LEN, NOISE_STD, rng)
    target = np.roll(u, -1)
    target[-1] = 0.0
    return u, target


def _cert_rate(a_matrices):
    certified = 0
    for a_mat in a_matrices:
        a0 = diagonal_reference_from_eigs(a_mat)
        result = certify_on_unit_circle(a_mat, a0, K_CONTOUR, "resolvent")
        if result.certified:
            certified += 1
    return 100.0 * certified / len(a_matrices)


def run_method(name: str, reg_fn) -> dict:
    rho_vals, stable_count, a_matrices = [], 0, []
    for seed in range(N_SEEDS):
        u, target = _ar_task(seed)
        a_mat, _, _ = train_diagonal_ssm_adam(
            u, target,
            n_state=N_STATE,
            n_epochs=N_EPOCHS,
            lr=LR,
            reg_weight=REG_WEIGHT,
            regulariser=reg_fn,
            seed=seed,
            constrained=False,
            init_log_r=INIT_LOG_R,
        )
        if not np.isfinite(a_mat).all():
            rho = float("inf")
        else:
            rho = float(np.max(np.abs(np.linalg.eigvals(a_mat))))
        rho_vals.append(rho)
        if rho < 1.0:
            stable_count += 1
        a_matrices.append(a_mat)
        print(f"  {name:16s} seed={seed}  rho={rho:.4f}  stable={rho<1.0}", flush=True)

    stable_mats = [
        m for m in a_matrices
        if np.isfinite(m).all() and np.max(np.abs(np.linalg.eigvals(m))) < 1.0
    ]
    cert = _cert_rate(stable_mats) if stable_mats else 0.0
    finite_rhos = [r for r in rho_vals if np.isfinite(r)]
    mean_rho = float(np.mean(finite_rhos)) if finite_rhos else float("inf")
    return {
        "method":       name,
        "stable%":      100.0 * stable_count / N_SEEDS,
        "mean_rho":     mean_rho,
        "n_diverged":   sum(1 for r in rho_vals if not np.isfinite(r)),
        "cert%_stable": cert,
        "rho_vals":     rho_vals,
    }


def main():
    rng0 = np.random.default_rng(0)
    u0, target0 = _ar_task(0)
    # Build contour barrier from eig-shrunk reference on seed-0 task
    a_init_mat, _, _ = train_diagonal_ssm_adam(
        u0, target0,
        n_state=N_STATE, n_epochs=50, lr=LR,
        seed=0, constrained=False, init_log_r=INIT_LOG_R,
    )
    a0_init = diagonal_reference_from_eigs(a_init_mat)
    contour_fn = make_contour_barrier(np.diag(a0_init), num_contour_points=K_CONTOUR, margin=REG_MARGIN)

    spectral_fn = functools.partial(spectral_penalty, margin=REG_MARGIN)
    lyapunov_fn = functools.partial(lyapunov_penalty, margin=REG_MARGIN)
    methods = [
        ("none",           None),
        ("spectral",       spectral_fn),
        ("lyapunov",       lyapunov_fn),
        ("contour_barrier",contour_fn),
    ]

    rows = []
    for name, reg_fn in methods:
        print(f"\n{name}", flush=True)
        rows.append(run_method(name, reg_fn))

    # Print table
    print()
    hdr = f"{'method':18s}  {'stable%':>8}  {'mean_rho*':>9}  {'diverged':>8}  {'cert%_stable':>13}"
    print(hdr)
    print("=" * len(hdr))
    for r in rows:
        print(
            f"{r['method']:18s}  {r['stable%']:>8.1f}  {r['mean_rho']:>9.4f}"
            f"  {r['n_diverged']:>8d}  {r['cert%_stable']:>13.1f}"
        )
    print("  * mean_rho excludes diverged (NaN/inf) runs")

    # Figure: spectral radius distribution per method
    fig, ax = plt.subplots(figsize=(8, 4))
    labels = [r["method"] for r in rows]
    data   = [r["rho_vals"] for r in rows]
    ax.boxplot(data, labels=labels)
    ax.axhline(1.0, color="r", linestyle="--", linewidth=1, label="unit circle")
    ax.set_ylabel("final spectral radius ρ")
    ax.set_title(f"Unconstrained SSM (lr={LR}, init ρ=0.96, {N_SEEDS} seeds)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "rho_distribution.pdf", bbox_inches="tight")
    plt.close(fig)

    # Figure: rho trajectories not available (only final), so bar chart of stable%
    fig2, axes2 = plt.subplots(1, 2, figsize=(9, 4))
    ax1, ax2 = axes2
    stable_rates = [r["stable%"] for r in rows]
    cert_rates   = [r["cert%_stable"] for r in rows]
    x = np.arange(len(labels))
    ax1.bar(x, stable_rates, color=["#d62728", "#2ca02c", "#2ca02c", "#2ca02c"])
    ax1.set_xticks(x); ax1.set_xticklabels(labels, rotation=15, ha="right")
    ax1.set_ylabel("stable runs (%)"); ax1.set_ylim(0, 110)
    ax1.set_title("Stability rate")
    ax2.bar(x, cert_rates)
    ax2.set_xticks(x); ax2.set_xticklabels(labels, rotation=15, ha="right")
    ax2.set_ylabel("cert% (stable runs only)"); ax2.set_ylim(0, 110)
    ax2.set_title("Post-hoc cert rate (stable only)")
    fig2.tight_layout()
    fig2.savefig(RESULTS_DIR / "stability_and_cert.pdf", bbox_inches="tight")
    plt.close(fig2)

    print(f"\nFigures saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
