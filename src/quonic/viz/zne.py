"""Zero-noise extrapolation plot: measured values vs. noise factor λ, with the
linear fit and the λ=0 extrapolated value."""

from __future__ import annotations

from typing import Any

from ..zne import ZNEResult
from ._mpl import _plt, finalize


def plot_zne(
    result: ZNEResult,
    ax: Any = None,
    show: bool = False,
    save: str | None = None,
    title: str | None = None,
) -> Any:
    """Plot a ZNEResult: scatter of (λ, value), the least-squares linear fit,
    and the extrapolated value at λ = 0 (a star marker).

    Parameters:
        result: the ZNEResult returned by zne().
        ax / show / save / title: same as plot_circuit.

    Returns: matplotlib Axes.
    """
    import numpy as np

    plt = _plt()
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4.2))
    else:
        fig = ax.figure

    lam = np.asarray(result.factors, dtype=float)
    val = np.asarray(result.values, dtype=float)

    ax.scatter(lam, val, color="#C44E52", zorder=3, label="measured")

    # least-squares linear fit y = a + b·λ, extrapolate to λ = 0
    b, a = np.polyfit(lam, val, 1)
    xs = np.array([0.0, float(max(lam)) + 0.5])
    ax.plot(xs, a + b * xs, "--", color="#4C72B0", alpha=0.7, label="linear fit")
    ax.scatter(
        [0.0], [result.extrapolated], marker="*", s=200, color="#C44E52",
        zorder=4, label=f"extrapolated {result.extrapolated:.4f}",
    )

    ax.axvline(0.0, color="gray", lw=0.8, alpha=0.5)
    ax.set_xlabel("Noise amplification factor λ")
    ax.set_ylabel(
        "Success probability" if result.metric == "success" else "Expectation value"
    )
    ax.legend(fontsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if title is None:
        title = "Zero-noise extrapolation"
    return finalize(fig, ax, show, save, title)
