"""Routing visualization: circuit diagram with inserted SWAPs on the coupling map."""

from __future__ import annotations

from typing import Any

from ..compiler import route_swaps
from ..ir import Circuit
from ..topology import CouplingMap
from ._mpl import _plt, finalize
from .circuit import _draw_box, _draw_target, _gate_label


def _draw_swap(ax: Any, x: float, y0: float, y1: float) -> None:
    """Draw the SWAP cross (orange) between two adjacent qubit wires."""
    ax.plot([x, x], [y0, y1], color="0.35", lw=1.0, zorder=2)
    ax.plot([x - 0.28, x + 0.28], [y0, y1], color="#E07B00", lw=1.8, zorder=3)
    ax.plot([x - 0.28, x + 0.28], [y1, y0], color="#E07B00", lw=1.8, zorder=3)


def plot_routing(
    circuit: Circuit,
    coupling_map: CouplingMap,
    ax: Any = None,
    show: bool = False,
    save: str | None = None,
    title: str | None = None,
) -> Any:
    """Draw the circuit after greedy SWAP routing, with inserted SWAPs marked as
    orange crosses.

    The x-axis is the gate sequence (original gates + inserted SWAPs), with one
    physical qubit per row. Single-qubit gates move to their physical location
    along with the mapping, visually showing how logical qubits "walk" across
    the coupling map.

    Parameters:
        circuit: source Circuit.
        coupling_map: CouplingMap (connectivity constraints).
        ax / show / save / title: same as plot_circuit.

    Returns: matplotlib Axes.
    """
    plt = _plt()
    routed = route_swaps(circuit, coupling_map)
    n = routed.num_qubits
    m = len(routed.ops)
    n_swaps = sum(1 for op in routed.ops if op.name == "swap")

    if ax is None:
        fig, ax = plt.subplots(
            figsize=(max(6.0, m * 0.55 + 1.0), max(1.8, n * 0.55))
        )
    else:
        fig = ax.figure

    for q in range(n):
        ax.plot([-0.5, m - 0.5], [q, q], color="0.35", lw=1.0, zorder=1)

    for col, op in enumerate(routed.ops):
        qs = list(op.qubits)
        if op.name == "swap":
            _draw_swap(ax, col, qs[0], qs[1])
            continue
        if len(qs) == 1:
            _draw_box(ax, col, qs[0], _gate_label(op))
            continue
        ymin, ymax = min(qs), max(qs)
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
    ax.set_ylabel("Physical qubit")
    for s in ("top", "right", "bottom"):
        ax.spines[s].set_visible(False)
    if title is None:
        title = f"SWAP routing ({n_swaps} SWAPs inserted)"
    return finalize(fig, ax, show, save, title)
