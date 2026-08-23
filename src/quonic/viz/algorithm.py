"""Algorithm visualization: VQE/QAOA energy convergence plot, Grover iteration amplitude plot."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from .._i18n import tr
from ..result import Result
from ._mpl import _plt, finalize

# ---------------------------------------------------------------------------
# 10. Energy convergence plot
# ---------------------------------------------------------------------------

def _as_energies(data: Result | Sequence[float]) -> list[float]:
    if isinstance(data, Result):
        history = data.metadata.get("history")
        if history is None:
            raise ValueError(tr("err.viz_history"))
        return list(history)
    return list(data)


def plot_energy_convergence(
    energies: Result | Sequence[float],
    ax: Any = None,
    show: bool = False,
    save: str | None = None,
    title: str | None = None,
) -> Any:
    """Draw the convergence curve of energy vs. optimization iteration for a
    variational algorithm (VQE/QAOA).

    Parameters:
        energies: a list of energies, or a Result with metadata["history"].
        ax / show / save / title: same as plot_circuit.

    Returns: matplotlib Axes.
    """
    plt = _plt()
    ys = _as_energies(energies)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.figure

    ax.plot(range(len(ys)), ys, marker="o", markersize=3, color="#4C72B0")
    ax.set_xlabel("Optimization iteration")
    ax.set_ylabel("Energy")
    ax.grid(True, alpha=0.3)
    return finalize(fig, ax, show, save, title)


# ---------------------------------------------------------------------------
# 11. Grover iteration amplitude plot
# ---------------------------------------------------------------------------

def _apply_oracle(eng: Any, marked: str) -> None:
    n = eng.n
    for q in range(n):
        if marked[n - 1 - q] == "0":
            eng.apply("x", (q,))
    eng.apply("mcz", tuple(range(n)))
    for q in range(n):
        if marked[n - 1 - q] == "0":
            eng.apply("x", (q,))


def _apply_diffusion(eng: Any) -> None:
    n = eng.n
    for q in range(n):
        eng.apply("h", (q,))
    for q in range(n):
        eng.apply("x", (q,))
    eng.apply("mcz", tuple(range(n)))
    for q in range(n):
        eng.apply("x", (q,))
    for q in range(n):
        eng.apply("h", (q,))


def _target_prob(eng: Any, marked: str) -> float:
    idx = int(marked, 2)
    return float(abs(eng.state[idx]) ** 2)


def plot_grover_amplitudes(
    n_qubits: int,
    marked: str,
    iterations: int | None = None,
    ax: Any = None,
    show: bool = False,
    save: str | None = None,
    title: str | None = None,
) -> Any:
    """Draw the probability of the target state in Grover search as a function
    of iteration count (simulated with the in-house engine, no backend
    dependency).

    Parameters:
        n_qubits: number of qubits.
        marked: the target bit string to mark (e.g. "11").
        iterations: number of iterations, default floor(π/4 · √(2^n)).
        ax / show / save / title: same as plot_circuit.

    Returns: matplotlib Axes.
    """
    from ..simulators import StatevectorEngine

    plt = _plt()
    marked = str(marked)
    if len(marked) != n_qubits or any(ch not in "01" for ch in marked):
        raise ValueError(tr("err.viz_marked", n_qubits=n_qubits, marked=marked))
    if iterations is None:
        iterations = int(math.pi / 4 * math.sqrt(2 ** n_qubits))

    eng = StatevectorEngine(n_qubits)
    for q in range(n_qubits):
        eng.apply("h", (q,))
    probs = [_target_prob(eng, marked)]
    for _ in range(iterations):
        _apply_oracle(eng, marked)
        _apply_diffusion(eng)
        probs.append(_target_prob(eng, marked))

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.figure

    ax.plot(range(len(probs)), probs, marker="o", color="#4C72B0")
    ax.set_xlabel("Iterations")
    ax.set_ylabel(f"Probability of target state |{marked}>")
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)
    return finalize(fig, ax, show, save, title)


# ---------------------------------------------------------------------------
# Problem graph (QAOA MaxCut)
# ---------------------------------------------------------------------------

def plot_problem_graph(
    edges: Sequence[tuple[int, int]],
    n_qubits: int | None = None,
    partition: Any = None,
    ax: Any = None,
    show: bool = False,
    save: str | None = None,
    title: str | None = None,
) -> Any:
    """Draw an optimization problem graph (e.g. MaxCut vertices and edges),
    optionally colored by cut.

    Parameters:
        edges: list of edges [(i, j), ...].
        n_qubits: number of vertices; None takes the largest vertex in the edges
            plus 1.
        partition: optional, the cut assignment of each vertex — dict
            {vertex: 0/1} or a length-n 0/1 sequence. When given, cut-crossing
            edges and the vertices on each side are highlighted in different
            colors.
        ax / show / save / title: same as plot_circuit.

    Returns: matplotlib Axes.
    """
    from matplotlib.patches import Circle

    plt = _plt()
    edges = [(int(u), int(v)) for u, v in edges]
    if n_qubits is None:
        n_qubits = max((max(u, v) for u, v in edges), default=-1) + 1

    side = None
    if partition is not None:
        if isinstance(partition, dict):
            side = {int(k): int(v) for k, v in partition.items()}
        else:
            side = {q: int(partition[q]) for q in range(n_qubits)}

    pos = {}
    for q in range(n_qubits):
        ang = 2 * math.pi * q / n_qubits - math.pi / 2
        pos[q] = (math.cos(ang), math.sin(ang))

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    else:
        fig = ax.figure

    for u, v in edges:
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        is_cut = side is not None and side.get(u) != side.get(v)
        ax.plot([x1, x2], [y1, y2], color="#DD8452" if is_cut else "#4C72B0",
                lw=2.5 if is_cut else 1.5, zorder=1)

    for q in range(n_qubits):
        x, y = pos[q]
        color = "#4C72B0"
        if side is not None:
            color = "#4C72B0" if side[q] == 0 else "#DD8452"
        ax.add_patch(Circle((x, y), 0.08, facecolor=color, edgecolor="black", zorder=2))
        ax.text(x, y + 0.14, str(q), ha="center", va="bottom", fontsize=9)

    pad = 0.25
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)
    ax.set_aspect("equal")
    ax.axis("off")
    return finalize(fig, ax, show, save, title)


# ---------------------------------------------------------------------------
# Hamiltonian visualization
# ---------------------------------------------------------------------------

_OP_COLORS = {"I": "#EEEEEE", "X": "#4C72B0", "Y": "#55A868", "Z": "#C44E52"}


def plot_hamiltonian(
    hamiltonian: Sequence[tuple[float, str]],
    n_qubits: int | None = None,
    ax: Any = None,
    show: bool = False,
    save: str | None = None,
    title: str | None = None,
) -> Any:
    """Draw a Pauli-term Hamiltonian: left = bar chart of term coefficients,
    right = heatmap of the operator structure.

    Parameters:
        hamiltonian: list [(coefficient, Pauli string), ...], Pauli string
            length = n_qubits.
        n_qubits: number of qubits; None takes the length of the first Pauli
            string.
        ax: optional length-2 sequence of Axes ([coefficients, operator
            structure]); a new one is created when omitted.
        show / save / title: same as plot_circuit.

    Returns: a length-2 sequence of Axes [ax_coeff, ax_ops].
    """
    import numpy as np
    from matplotlib.colors import ListedColormap

    plt = _plt()
    coeffs = [float(c) for c, _ in hamiltonian]
    paulis = [p for _, p in hamiltonian]
    if n_qubits is None:
        n_qubits = len(paulis[0]) if paulis else 0
    paulis = [p.ljust(n_qubits, "I") for p in paulis]

    op_map = {"I": 0, "X": 1, "Y": 2, "Z": 3}
    grid = np.zeros((len(paulis), n_qubits), dtype=int)
    for i, p in enumerate(paulis):
        for j, ch in enumerate(p):
            grid[i, j] = op_map.get(ch, 0)

    if ax is None:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4),
                                 gridspec_kw={"width_ratios": [1, 1.5]})
    else:
        axes = ax
        fig = axes[0].figure

    xs = list(range(len(coeffs)))
    colors = ["#4C72B0" if c >= 0 else "#C44E52" for c in coeffs]
    axes[0].bar(xs, coeffs, color=colors)
    axes[0].set_xticks(xs)
    axes[0].set_xticklabels(paulis)
    axes[0].axhline(0, color="0.5", lw=0.8)
    axes[0].set_xlabel("Pauli term")
    axes[0].set_ylabel("Coefficient")
    axes[0].set_title("Coefficients")
    for s in ("top", "right"):
        axes[0].spines[s].set_visible(False)

    cmap = ListedColormap([_OP_COLORS[k] for k in ("I", "X", "Y", "Z")])
    im = axes[1].imshow(grid, cmap=cmap, aspect="auto", vmin=0, vmax=3)
    axes[1].set_xticks(range(n_qubits))
    axes[1].set_xticklabels([f"q{q}" for q in range(n_qubits)])
    axes[1].set_yticks(range(len(paulis)))
    axes[1].set_yticklabels(paulis)
    axes[1].set_xlabel("Qubit")
    axes[1].set_title("Operator structure")
    for i in range(len(paulis)):
        for j in range(n_qubits):
            axes[1].text(j, i, paulis[i][j], ha="center", va="center",
                         fontsize=8, color="0.15")
    cbar = fig.colorbar(im, ax=axes[1], ticks=[0, 1, 2, 3], fraction=0.046, pad=0.04)
    cbar.ax.set_yticklabels(["I", "X", "Y", "Z"])

    if title is not None:
        fig.suptitle(title)
    if save:
        fig.savefig(save, bbox_inches="tight", dpi=120)
    if show:
        plt.show()
    return axes
