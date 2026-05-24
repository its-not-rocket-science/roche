"""Experiment P2-2: Stability regulariser comparison during SSM training.

Trains diagonal SSMs on three AR tasks under four regularisation regimes:
  none      -- no regulariser, Adam baseline
  spectral  -- hinge on spectral radius
  lyapunov  -- per-mode Lyapunov sum penalty
  contour   -- contour barrier with gradient-optimised frozen reference

Usage:
    python experiments/regulariser_comparison.py [--n-state N] [--n-epochs E]
                                                  [--seeds K] [--outdir DIR]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import functools

import torch

from roche.certificates import certify_on_unit_circle, is_schur_stable, spectral_radius
from roche.reference import diagonal_reference_from_eigs
from roche.reference_opt import optimise_diagonal_reference
from roche.regularisers import (
    lyapunov_penalty,
    make_contour_barrier,
    spectral_penalty,
)
from roche.ssm import generate_ar_sequence, train_diagonal_ssm_adam


REG_MARGIN = 0.1   # consistent with unstable_regime.py

TASKS = [
    {"desc": "AR(2) short",   "ar_coeffs": [-1.347, -0.490], "max_r": 0.70},
    {"desc": "AR(4) long",    "ar_coeffs": [-0.282, 0.481, -0.285, -0.532], "max_r": 0.92},
    {"desc": "AR(4) hard",    "ar_coeffs": [-0.301, 0.163, -0.131, -0.295], "max_r": 0.97},
]

REG_NAMES = ["none", "spectral", "lyapunov", "contour"]
REG_WEIGHT = 0.1


def make_regulariser(name: str, initial_a: np.ndarray) -> object:
    if name == "none":
        return None
    if name == "spectral":
        return functools.partial(spectral_penalty, margin=REG_MARGIN)
    if name == "lyapunov":
        return functools.partial(lyapunov_penalty, margin=REG_MARGIN)
    if name == "contour":
        a0, _ = optimise_diagonal_reference(initial_a, n_steps=200, num_contour_points=128)
        return make_contour_barrier(a0, num_contour_points=128, margin=REG_MARGIN)
    raise ValueError(name)


def run_one(
    task: dict,
    reg_name: str,
    n_state: int,
    n_epochs: int,
    seq_len: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    u = generate_ar_sequence(task["ar_coeffs"], seq_len, 0.05, rng)
    target = np.roll(u, -1); target[-1] = 0.0

    # Build a quick initial model to initialise contour reference
    from roche.ssm import _DiagonalSSM
    init_model = _DiagonalSSM(n_state, seed=seed)
    initial_a = init_model.transition_matrix()

    regulariser = make_regulariser(reg_name, initial_a)

    a_mat, task_hist, reg_hist = train_diagonal_ssm_adam(
        u, target,
        n_state=n_state,
        n_epochs=n_epochs,
        lr=1e-3,
        reg_weight=REG_WEIGHT,
        regulariser=regulariser,
        seed=seed,
    )

    rho = spectral_radius(a_mat)
    stable = is_schur_stable(a_mat)

    # Post-hoc certificate with eig-shrunk reference
    a0_eval = diagonal_reference_from_eigs(a_mat) if stable else None
    if a0_eval is not None:
        cert = certify_on_unit_circle(a_mat, a0_eval, 512, "resolvent")
        min_margin = cert.min_margin
        certified = cert.certified
    else:
        min_margin = -99.0
        certified = False

    return {
        "final_task_loss": task_hist[-1],
        "spectral_radius": rho,
        "min_margin": min_margin,
        "certified": certified,
        "task_hist": task_hist,
        "reg_hist": reg_hist,
    }


def run(args: argparse.Namespace) -> None:
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    seeds = list(range(args.seeds))

    # All results: task -> reg -> list of per-seed dicts
    all_results: dict[str, dict[str, list]] = {
        t["desc"]: {r: [] for r in REG_NAMES} for t in TASKS
    }

    for task in TASKS:
        print(f"\nTask: {task['desc']}")
        for reg in REG_NAMES:
            for seed in seeds:
                print(f"  reg={reg:10s}  seed={seed}", end="\r")
                r = run_one(task, reg, args.n_state, args.n_epochs, args.seq_len, seed)
                all_results[task["desc"]][reg].append(r)
            print()

    # Print summary table with std
    header = (
        f"{'task':16s}  {'reg':10s}  {'task_loss':>10}  {'±':>9}  "
        f"{'rho':>6}  {'±':>6}  {'margin':>8}  {'±':>6}  {'cert%':>6}"
    )
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))
    csv_rows = ["task,reg,mean_loss,std_loss,mean_rho,std_rho,mean_margin,std_margin,cert_pct\n"]
    for task in TASKS:
        for reg in REG_NAMES:
            rs = all_results[task["desc"]][reg]
            tl_vals = [r["final_task_loss"] for r in rs]
            rho_vals = [r["spectral_radius"] for r in rs]
            mm_vals  = [r["min_margin"] for r in rs]
            tl  = np.mean(tl_vals);  tl_s  = np.std(tl_vals, ddof=1)
            rho = np.mean(rho_vals); rho_s = np.std(rho_vals, ddof=1)
            mm  = np.mean(mm_vals);  mm_s  = np.std(mm_vals, ddof=1)
            cp  = 100.0 * np.mean([r["certified"] for r in rs])
            print(
                f"{task['desc']:16s}  {reg:10s}  {tl:10.4e}  {tl_s:9.4e}  "
                f"{rho:6.4f}  {rho_s:6.4f}  {mm:+8.4f}  {mm_s:6.4f}  {cp:6.1f}"
            )
            csv_rows.append(
                f"{task['desc']},{reg},{tl:.6e},{tl_s:.6e},"
                f"{rho:.6f},{rho_s:.6f},{mm:.6f},{mm_s:.6f},{cp:.1f}\n"
            )
    with open(outdir / "summary_with_std.csv", "w") as f:
        f.writelines(csv_rows)

    # Figure: learning curves on AR(4) hard, all regularisers
    hard_task = TASKS[2]["desc"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    colours = {"none": "C0", "spectral": "C1", "lyapunov": "C2", "contour": "C3"}
    for reg in REG_NAMES:
        rs = all_results[hard_task][reg]
        mean_task = np.mean([r["task_hist"] for r in rs], axis=0)
        mean_reg = np.mean([r["reg_hist"] for r in rs], axis=0)
        ax1.semilogy(mean_task, label=reg, color=colours[reg])
        ax2.plot(mean_reg, label=reg, color=colours[reg])
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Task loss (log)"); ax1.legend(); ax1.set_title(f"{hard_task}: task loss")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Reg loss"); ax2.set_title(f"{hard_task}: regulariser loss")
    fig.tight_layout()
    fig.savefig(outdir / "training_curves.png", dpi=150)
    plt.close(fig)

    # Figure: bar chart — cert% by method and task
    fig, axes = plt.subplots(1, len(TASKS), figsize=(12, 4), sharey=True)
    for ax, task in zip(axes, TASKS):
        cert_pcts = [100.0 * np.mean([r["certified"] for r in all_results[task["desc"]][reg]]) for reg in REG_NAMES]
        ax.bar(range(len(REG_NAMES)), cert_pcts, color=[colours[r] for r in REG_NAMES])
        ax.set_ylim(0, 105)
        ax.set_title(task["desc"])
        ax.set_xticks(range(len(REG_NAMES)))
        ax.set_xticklabels(REG_NAMES, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel("Certification rate (%)")
    fig.suptitle("Post-hoc certification rate by regulariser and task")
    fig.tight_layout()
    fig.savefig(outdir / "certification_rates.png", dpi=150)
    plt.close(fig)

    print(f"\nFigures saved to {outdir}/")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--n-state", type=int, default=8)
    p.add_argument("--n-epochs", type=int, default=500)
    p.add_argument("--seq-len", type=int, default=500)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--outdir", type=str, default="results/p2exp2")
    p.add_argument("--quick", action="store_true",
                   help="Fast smoke: 2 seeds, 20 epochs, seq_len=100")
    args = p.parse_args()
    if args.quick:
        args.seeds = 2
        args.n_epochs = 20
        args.seq_len = 100
    return args


if __name__ == "__main__":
    run(parse_args())
