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
