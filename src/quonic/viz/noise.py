"""Noise cost visualization: timing heatmap for density-matrix simulation under depolarizing noise, and noise-overlaid circuit diagram."""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

from ..ir import Circuit
from ..noise import NoiseModel
from ._mpl import _plt, finalize
from .circuit import _draw_box, _draw_target, _gate_label


def _noisy_ghz_time(n: int, p: float, repeats: int) -> float | None:
    """Run an n-qubit GHZ (H + CX chain) on the density-matrix engine with depolarizing noise and return the shortest elapsed time."""
    from ..noise import depolarizing
    from ..simulators import DensityMatrixEngine

    best = None
    for _ in range(repeats):
        eng = DensityMatrixEngine(n, noise=depolarizing(p))
        t0 = time.perf_counter()
        eng.apply("h", (0,))
        for i in range(n - 1):
            eng.apply("cx", (i, i + 1))
        dt = time.perf_counter() - t0
        if best is None or dt < best:
            best = dt
    return best


def plot_noise_heatmap(
    n_values: Sequence[int] = (2, 4, 6, 8, 10),
    noise_rates: Sequence[float] = (0.0, 0.01, 0.05, 0.1, 0.5),
    budget: float = 1.0,
    repeats: int = 1,
    ax: Any = None,
    show: bool = False,
    save: str | None = None,
    title: str | None = None,
) -> Any:
    """Draw the timing heatmap of density-matrix simulation under depolarizing
    noise.

    The x-axis is the qubit count n, the y-axis is the depolarizing probability
    p, and each cell is log10(elapsed time / second). Time grows with n roughly
    as 4^n (the memory/time wall of density matrices); p itself barely affects
    timing — the depolarizing channel performs the same amount of matrix work
    for any p > 0, and only p = 0 (noise off) skips the noise layer.

    Cells exceeding budget seconds are outlined in red, indicating that (p, n)
    combination is infeasible within this budget.

    Parameters:
        n_values: sequence of qubit counts (density matrix costs 4^n, ≤ 10
            recommended).
        noise_rates: sequence of depolarizing probabilities (0.0 means no
            noise, still run on the density-matrix engine).
        budget: time budget (seconds); cells exceeding it are marked infeasible.
        repeats: number of repeats per cell, taking the minimum (to suppress
            timing jitter).
        ax / show / save / title: same as plot_circuit.

    Returns: matplotlib Axes.
    """
    import numpy as np

    plt = _plt()
    grid = []
    for p in noise_rates:
        row = [_noisy_ghz_time(n, p, repeats) for n in n_values]
        grid.append(row)

    times = np.array(grid)
    data = np.log10(times)
    infeasible = times > budget

    if ax is None:
        fig, ax = plt.subplots(
            figsize=(max(6.0, len(n_values) * 0.9 + 1.2),
                     max(3.0, len(noise_rates) * 0.7 + 1.0))
        )
    else:
        fig = ax.figure

    vmin = float(np.min(data))
    vmax = float(np.max(data))
    if vmax - vmin < 1e-12:
        vmax = vmin + 1.0

    im = ax.imshow(data, cmap="viridis", aspect="auto", vmin=vmin, vmax=vmax)

    from matplotlib.patches import Rectangle

    for i in range(len(noise_rates)):
        for j in range(len(n_values)):
            t = times[i][j]
            label = f"{t:.3g}"
            if infeasible[i][j]:
                ax.add_patch(
                    Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                              edgecolor="red", lw=2, zorder=3)
                )
                ax.text(j, i, label, ha="center", va="center", fontsize=8,
                        color="red", fontweight="bold", zorder=4)
            else:
                lum = (data[i][j] - vmin) / (vmax - vmin)
                color = "white" if lum > 0.5 else "black"
                ax.text(j, i, label, ha="center", va="center", fontsize=8,
                        color=color, zorder=4)

    ax.set_xticks(range(len(n_values)))
    ax.set_xticklabels(n_values)
    ax.set_yticks(range(len(noise_rates)))
    ax.set_yticklabels(noise_rates)
    ax.set_xlabel("Number of qubits n")
    ax.set_ylabel("Depolarizing probability p")
    fig.colorbar(im, ax=ax, label="log10(time/s)")
    fig.subplots_adjust(bottom=0.18)
    fig.text(
        0.5, 0.02,
        f"Red box = exceeds budget {budget}s; timing is dominated by n (4^n wall), p has little effect",
        ha="center", va="bottom", fontsize=8, color="0.4",
    )
    return finalize(fig, ax, show, save, title)


def plot_noisy_circuit(
    circuit: Circuit,
    noise: NoiseModel | float | None = None,
    ax: Any = None,
    show: bool = False,
    save: str | None = None,
    title: str | None = None,
) -> Any:
    """Overlay noise intensity on the circuit diagram: each gate's background
    color is the depolarizing probability it experiences.

    Single-qubit gates are colored with noise.single and two-or-more-qubit gates
    with noise.double (YlOrRd colormap); measurement gates are not colored.
    White gate boxes / target symbols are drawn on top of the color band for
    readability.

    Parameters:
        circuit: A Circuit object.
        noise: NoiseModel / numeric probability / None (None means no noise, all
            zero intensity).
        ax / show / save / title: same as plot_circuit.

    Returns: matplotlib Axes.
    """
    from matplotlib.colors import Normalize
    from matplotlib.patches import Rectangle

    from ..noise import resolve_noise

    plt = _plt()
    noise = resolve_noise(noise)
    n = circuit.num_qubits
    m = len(circuit.ops)
    cmap = plt.cm.YlOrRd
    vmax = max(noise.single, noise.double, 1e-6)
    norm = Normalize(vmin=0, vmax=vmax)

    if ax is None:
        fig, ax = plt.subplots(
            figsize=(max(6.0, m * 0.55 + 1.0), max(1.8, n * 0.55))
        )
    else:
        fig = ax.figure

    for q in range(n):
        ax.plot([-0.5, m - 0.5], [q, q], color="0.35", lw=1.0, zorder=1)

    for col, op in enumerate(circuit.ops):
        qs = list(op.qubits)
        if op.name == "measure":
            _draw_box(ax, col, qs[0], _gate_label(op))
            continue
        rate = noise.single if len(qs) == 1 else noise.double
        if len(qs) == 1:
            y = qs[0]
            ax.add_patch(
                Rectangle((col - 0.5, y - 0.42), 1.0, 0.84,
                          facecolor=cmap(norm(rate)), edgecolor="none",
                          alpha=0.55, zorder=1.5)
            )
            _draw_box(ax, col, y, _gate_label(op))
            continue
        ymin, ymax = min(qs), max(qs)
        ax.add_patch(
            Rectangle((col - 0.5, ymin - 0.42), 1.0, (ymax - ymin) + 0.84,
                      facecolor=cmap(norm(rate)), edgecolor="none",
                      alpha=0.55, zorder=1.5)
        )
        ax.plot([col, col], [ymin, ymax], color="0.35", lw=1.0, zorder=2)
        target = qs[-1]
        for q in qs:
            if q == target:
                if op.name in ("cx", "ccx"):
                    _draw_target(ax, col, q)
                else:
                    _draw_box(ax, col, q, _gate_label(op))
            else:
                ax.plot(col, q, "ko", ms=5, zorder=3)

    ax.set_xlim(-0.8, m - 0.2)
    ax.set_ylim(-0.6, n - 0.4)
    ax.invert_yaxis()
    ax.set_yticks(range(n))
    ax.set_yticklabels([f"q{q}" for q in range(n)])
    ax.set_xticks([])
    ax.set_ylabel("Qubit")
    for s in ("top", "right", "bottom"):
        ax.spines[s].set_visible(False)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, label="Depolarizing noise rate")
    cbar.set_ticks([0, vmax])
    cbar.set_ticklabels(["0", f"{vmax:.3g}"])
    if title is None:
        title = f"Noise overlay (single={noise.single}, double={noise.double})"
    return finalize(fig, ax, show, save, title)
