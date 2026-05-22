"""Plotting helpers for starter experiments."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


def plot_margins(theta: NDArray[np.float64], margins: NDArray[np.float64], path: str | Path) -> None:
    """Plot contour margins as a function of angle."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(theta, margins)
    ax.axhline(0.0, linestyle="--", linewidth=1)
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel("margin")
    ax.set_title("Contour certificate margin")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_margin_comparison(
    theta: NDArray[np.float64],
    det_margins: NDArray[np.float64],
    res_margins: NDArray[np.float64],
    path: str | Path,
    title: str = "",
) -> None:
    """Plot determinant and resolvent margins side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, margins, label in zip(
        axes,
        [det_margins, res_margins],
        ["Determinant margin", "Resolvent margin"],
    ):
        ax.plot(theta, margins)
        ax.axhline(0.0, linestyle="--", linewidth=1, color="red")
        ax.set_xlabel(r"$\theta$")
        ax.set_ylabel("margin")
        ax.set_title(label)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_eigenvalues(
    matrices: dict[str, NDArray[np.complex128]],
    path: str | Path,
) -> None:
    """Plot eigenvalue locations for a set of labelled matrices."""
    theta = np.linspace(0, 2 * np.pi, 300)
    fig, axes = plt.subplots(1, len(matrices), figsize=(4 * len(matrices), 4), squeeze=False)
    for ax, (label, a) in zip(axes[0], matrices.items()):
        eigs = np.linalg.eigvals(a)
        ax.plot(np.cos(theta), np.sin(theta), "k--", linewidth=0.8)
        ax.scatter(eigs.real, eigs.imag, s=20, zorder=3)
        ax.set_aspect("equal")
        ax.set_title(label)
        ax.set_xlabel("Re")
        ax.set_ylabel("Im")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_discretisation_study(
    grid_sizes: list[int],
    sampled_cert: list[bool],
    rigorous_cert: list[bool],
    min_margins: list[float],
    criteria: list[float],
    path: str | Path,
) -> None:
    """Plot how certification changes with grid size N."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax1.plot(grid_sizes, min_margins, "o-", label="min sampled margin")
    ax1.plot(grid_sizes, criteria, "s--", label=r"$\pi L / N$")
    ax1.axhline(0.0, linestyle=":", linewidth=0.8, color="black")
    ax1.set_ylabel("value")
    ax1.legend()
    ax1.set_title("Margin vs grid size")
    ax2.plot(grid_sizes, [int(c) for c in sampled_cert], "o-", label="sampled certified")
    ax2.plot(grid_sizes, [int(c) for c in rigorous_cert], "s--", label="rigorous certified")
    ax2.set_ylabel("certified (0/1)")
    ax2.set_xlabel("N (grid points)")
    ax2.legend()
    ax2.set_title("Certification status vs grid size")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
