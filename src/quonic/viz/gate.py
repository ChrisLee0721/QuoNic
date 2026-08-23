"""Gate visualization: real/imaginary two-panel heatmap of a single gate's unitary matrix."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from .._i18n import tr
from ..gates import Gate, resolve
from ..ir import GateOperation
from ._mpl import _plt


def _gate_unitary(name: str, qubits: Sequence[int], params: tuple[float, ...]) -> Any:
    """Build the gate's unitary matrix column by column using the in-house statevector engine (column = basis state after applying the gate)."""
    import numpy as np

    from ..simulators import StatevectorEngine

    n = max(qubits) + 1 if qubits else 1
    dim = 2 ** n
    u = np.zeros((dim, dim), dtype=complex)
    for col in range(dim):
        eng = StatevectorEngine(n)
        eng.state[col] = 1.0
        eng.apply(name, qubits, params)
        u[:, col] = eng.state
    return u


def _resolve_gate(gate: Any) -> tuple[str, list[int], tuple[float, ...]]:
    """Normalize the input into (name, qubits, params)."""
    if isinstance(gate, GateOperation):
        return gate.name, list(gate.qubits), gate.params
    if isinstance(gate, Gate):
        name = gate.name
        qubits = list(range(max(1, gate.num_qubits)))
        return name, qubits, gate.params
    if isinstance(gate, str):
        g = resolve(gate)
        return g.name, list(range(max(1, g.num_qubits))), g.params
    raise TypeError(tr("err.viz_gate_matrix"))


def plot_gate_matrix(
    gate: Any,
    ax: Any = None,
    show: bool = False,
    save: str | None = None,
    title: str | None = None,
) -> Any:
    """Draw a two-panel heatmap of a single gate's unitary matrix (real/imag).

    Parameters:
        gate: a Gate object / GateOperation / gate-name string (e.g. "cx", "h",
            "mcz").
        ax: optional length-2 sequence of Axes ([real, imag]); a new one is
            created when omitted.
        show / save / title: same as plot_circuit.

    Returns: a length-2 sequence of Axes [ax_real, ax_imag].
    """
    import numpy as np

    plt = _plt()
    name, qubits, params = _resolve_gate(gate)
    if name == "measure":
        raise ValueError(tr("err.viz_measure_unitary"))

    u = _gate_unitary(name, qubits, params)
    n = round(math.log2(u.shape[0]))
    labels = [f"|{format(i, '0%db' % n)}>" for i in range(u.shape[0])]

    if ax is None:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    else:
        axes = ax
        fig = axes[0].figure

    vmax = max(float(np.max(np.abs(u.real))), float(np.max(np.abs(u.imag))), 1e-12)

    im0 = axes[0].imshow(u.real, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    axes[0].set_title(f"Re({name})")
    im1 = axes[1].imshow(u.imag, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    axes[1].set_title(f"Im({name})")
    for a in axes:
        a.set_xticks(range(u.shape[0]))
        a.set_yticks(range(u.shape[0]))
        a.set_xticklabels(labels, rotation=90, fontsize=6)
        a.set_yticklabels(labels, fontsize=6)
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    if title is not None:
        fig.suptitle(title)
    if save:
        fig.savefig(save, bbox_inches="tight", dpi=120)
    if show:
        plt.show()
    return axes
