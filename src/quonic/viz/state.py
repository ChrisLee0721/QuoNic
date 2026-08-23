"""Quantum state visualization: Bloch sphere, density-matrix heatmap, entanglement spectrum, per-gate state evolution."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from .._i18n import tr
from ..ir import Circuit
from ._mpl import _plt, finalize
from .circuit import _to_statevector


def _bloch_vector(state: Any) -> tuple[float, float, float]:
    """Convert a single-qubit state (or a Bloch-vector triple) into (x, y, z)."""
    import numpy as np

    if isinstance(state, (list, tuple)) and len(state) == 3:
        x, y, z = (float(v) for v in state)
        if x * x + y * y + z * z > 1.0 + 1e-9:
            raise ValueError(tr("err.viz_bloch_norm"))
        return x, y, z

    sv = np.asarray(_to_statevector(state), dtype=complex)
    if sv.size != 2:
        raise ValueError(tr("err.viz_bloch_single"))
    sv = sv / np.linalg.norm(sv)
    a, b = sv[0], sv[1]
    x = 2 * (a.conjugate() * b).real
    y = 2 * (a.conjugate() * b).imag
    z = abs(a) ** 2 - abs(b) ** 2
    return float(x), float(y), float(z)


def _rho_bloch_vector(rho: Any) -> tuple[float, float, float]:
    """Compute the Bloch vector from a 2×2 density matrix (also works for mixed states)."""
    import numpy as np

    rho = np.asarray(rho, dtype=complex)
    x = 2 * rho[0, 1].real
    y = 2 * rho[1, 0].imag  # = -2 Im(rho01)
    z = (rho[0, 0] - rho[1, 1]).real
    return float(x), float(y), float(z)


def _draw_bloch_sphere(
    ax: Any, x: float, y: float, z: float, label: str | None = None
) -> None:
    """Draw a Bloch sphere on the given 3D Axes (sphere wireframe + coordinate axes + statevector arrow)."""
    import numpy as np

    u = np.linspace(0, 2 * np.pi, 48)
    v = np.linspace(0, np.pi, 24)
    sx = np.outer(np.cos(u), np.sin(v))
    sy = np.outer(np.sin(u), np.sin(v))
    sz = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(sx, sy, sz, color="0.85", linewidth=0.4, alpha=0.6)

    for (dx, dy, dz), t in (((1, 0, 0), "x"), ((0, 1, 0), "y"), ((0, 0, 1), "|0>")):
        ax.plot([-dx, dx], [-dy, dy], [-dz, dz], color="0.5", lw=0.8)
        ax.text(dx * 1.15, dy * 1.15, dz * 1.15, t, fontsize=9, color="0.3")
    ax.text(0, 0, -1.18, "|1>", fontsize=9, color="0.3")

    r = math.sqrt(x * x + y * y + z * z)
    color = "#4C72B0" if r > 0.99 else "#C44E52"  # pure state blue / mixed state orange-red
    ax.plot([0, x], [0, y], [0, z], color=color, lw=3)
    ax.scatter([x], [y], [z], color=color, s=80, zorder=5,
               edgecolor="white", linewidth=0.5)
    ax.scatter([0], [0], [0], color="0.4", s=12, zorder=4)  # sphere-center reference point
    if label is not None:
        ax.text2D(0.03, 0.97, label, transform=ax.transAxes, fontsize=10,
                  color="0.2", va="top", ha="left")

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])


def plot_bloch_sphere(
    state: Any,
    ax: Any = None,
    show: bool = False,
    save: str | None = None,
    title: str | None = None,
) -> Any:
    """Draw the Bloch sphere of a single-qubit state (3D, a point on the unit
    sphere plus an arrow from the origin to it).

    Parameters:
        state: a 2-element complex amplitude array / StatevectorEngine(1) / a
            single-qubit Circuit / a 3D Bloch vector.
        ax: optional, must have a 3D projection; a new one is created when
            omitted.
        show / save / title: same as plot_circuit.

    Returns: matplotlib Axes (3D).
    """
    plt = _plt()
    x, y, z = _bloch_vector(state)

    if ax is None:
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure

    _draw_bloch_sphere(ax, x, y, z)
    if title is not None:
        ax.set_title(title)
    if save:
        fig.savefig(save, bbox_inches="tight", dpi=120)
    if show:
        plt.show()
    return ax


def plot_bloch_multivector(
    state: Any,
    cols: int | None = None,
    annotate: bool = False,
    show: bool = False,
    save: str | None = None,
    title: str | None = None,
) -> Any:
    """Draw a Bloch sphere for each qubit of a multi-qubit state (grid layout).

    For each qubit q, compute its reduced single-qubit density matrix ρ_q (by
    tracing out the other qubits) and draw it as a grid of Bloch spheres. In an
    entangled state each qubit "shrinks" inside the sphere (mixed state), while
    product states stay on the surface.

    Parameters:
        state: an n-qubit state (1D statevector / 2D density matrix /
            StatevectorEngine / Circuit).
        cols: number of spheres per row; None uses min(n, 5).
        annotate: when True, annotate the exact Bloch vector (x, y, z) under
            each sphere; default False keeps the figure uncluttered (the arrow
            direction + |r| label already encode the same information).
        show / save / title: same as plot_circuit.

    Returns: a list of 3D Axes (one per qubit).
    """
    import math

    plt = _plt()
    rho = _to_density(state)
    n = round(math.log2(rho.shape[0]))

    cols = cols or min(n, 5)
    rows = math.ceil(n / cols)
    fig = plt.figure(figsize=(cols * 2.8, rows * 2.8))

    axes = []
    for q in range(n):
        rho_q = _partial_trace(rho, [q], n)
        x, y, z = _rho_bloch_vector(rho_q)
        r = math.sqrt(x * x + y * y + z * z)
        ax = fig.add_subplot(rows, cols, q + 1, projection="3d")
        _draw_bloch_sphere(ax, x, y, z, label=f"q{q}  |r|={r:.2f}")
        if annotate:
            ax.text2D(0.5, 0.02, f"({x:+.3f}, {y:+.3f}, {z:+.3f})",
                      transform=ax.transAxes, fontsize=8, color="0.3",
                      va="bottom", ha="center")
        axes.append(ax)

    if title is not None:
        fig.suptitle(title)
    if save:
        fig.savefig(save, bbox_inches="tight", dpi=120)
    if show:
        plt.show()
    return axes


# ---------------------------------------------------------------------------
# Density-matrix heatmap
# ---------------------------------------------------------------------------

def _to_density(state: Any) -> Any:
    """Normalize the input into a 2^n × 2^n complex density matrix (numpy array)."""
    import numpy as np

    from ..simulators import DensityMatrixEngine, StatevectorEngine

    if isinstance(state, DensityMatrixEngine):
        return np.asarray(state.rho)
    if isinstance(state, StatevectorEngine):
        sv = np.asarray(state.state)
        return np.outer(sv, sv.conjugate())
    if isinstance(state, Circuit):
        eng = DensityMatrixEngine(state.num_qubits)
        for op in state.ops:
            eng.apply(op.name, list(op.qubits), op.params)
        return np.asarray(eng.rho)
    arr = np.asarray(state, dtype=complex)
    if arr.ndim == 1:
        return np.outer(arr, arr.conjugate())
    if arr.ndim == 2:
        return arr
    raise TypeError(tr("err.viz_state_input"))


def plot_density_matrix(
    state: Any,
    ax: Any = None,
    show: bool = False,
    save: str | None = None,
    title: str | None = None,
) -> Any:
    """Draw a two-panel heatmap of a density matrix (real/imag).

    Parameters:
        state: DensityMatrixEngine / StatevectorEngine / Circuit / complex array.
        ax: optional length-2 sequence of Axes ([real, imag]); a new two-panel
            figure is created when omitted.
        show / save / title: same as plot_circuit.

    Returns: a length-2 sequence of Axes [ax_real, ax_imag].
    """
    import numpy as np

    plt = _plt()
    rho = _to_density(state)
    n = round(math.log2(rho.shape[0]))

    if ax is None:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    else:
        axes = ax
        fig = axes[0].figure

    vmax = max(float(np.max(np.abs(rho.real))), float(np.max(np.abs(rho.imag))), 1e-12)

    labels = [f"|{format(i, f'0{n}b')}>" for i in range(2 ** n)]

    im0 = axes[0].imshow(rho.real, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    axes[0].set_title("Re(ρ)")
    im1 = axes[1].imshow(rho.imag, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    axes[1].set_title("Im(ρ)")
    for a in axes:
        a.set_xticks(range(2 ** n))
        a.set_yticks(range(2 ** n))
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


# ---------------------------------------------------------------------------
# Entanglement visualization
# ---------------------------------------------------------------------------

def _partial_trace(rho: Any, keep: Sequence[int], n: int) -> Any:
    """Compute the reduced density matrix of subsystem A (the qubits in keep)
    for an n-qubit density matrix.

    Qubits that are traced out share the same einsum letter for their row and
    column indices (diagonal summation = partial trace).
    """
    import numpy as np

    rho = np.asarray(rho, dtype=complex).reshape([2] * (2 * n))
    letters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    row_axis = {q: n - 1 - q for q in range(n)}
    col_axis = {q: n + (n - 1 - q) for q in range(n)}

    keep = set(keep)
    sub = [None] * (2 * n)
    row_out, col_out = [], []
    idx = 0
    for q in sorted(keep):
        r, c = letters[idx], letters[idx + 1]
        idx += 2
        sub[row_axis[q]] = r
        sub[col_axis[q]] = c
        row_out.append(r)
        col_out.append(c)
    for q in range(n):
        if q in keep:
            continue
        s = letters[idx]
        idx += 1
        sub[row_axis[q]] = s
        sub[col_axis[q]] = s

    in_sub = "".join(sub)
    out_sub = "".join(row_out + col_out)
    result = np.einsum(in_sub + "->" + out_sub, rho)
    k = len(keep)
    return result.reshape(2 ** k, 2 ** k)


def _von_neumann_entropy(eigenvalues: Any) -> float:
    """Compute the von Neumann entropy (in bits) from the reduced density matrix eigenvalues."""
    import numpy as np

    lam = np.clip(np.real(eigenvalues), 0.0, None)
    lam = lam[lam > 1e-12]
    return float(-np.sum(lam * np.log2(lam)))


def _concurrence(rho: Any) -> float:
    """Wootters concurrence, valid for any two-qubit state (pure or mixed).

    For pure states it reduces to sqrt(2(1 - Tr(ρ_A²))); for mixed states (such
    as classically correlated states after measurement collapse) it correctly
    returns 0 — the pure-state formula would misjudge mixed states as non-zero.
    """
    import numpy as np

    rho = np.asarray(rho, dtype=complex)
    if rho.shape != (4, 4):
        raise ValueError(tr("err.viz_concurrence"))
    sy = np.array([[0.0, -1j], [1j, 0.0]], dtype=complex)
    m = np.kron(sy, sy)  # σ_y ⊗ σ_y
    rho_tilde = m @ rho.conj() @ m
    r = rho @ rho_tilde
    eig = np.linalg.eigvals(r)
    lam = np.sort(np.sqrt(np.clip(np.real(eig), 0.0, None)))[::-1]
    return float(max(0.0, lam[0] - lam[1] - lam[2] - lam[3]))


def plot_entanglement(
    state: Any,
    partition: Sequence[int] | None = None,
    ax: Any = None,
    show: bool = False,
    save: str | None = None,
    title: str | None = None,
) -> Any:
    """Draw the entanglement spectrum (reduced density matrix eigenvalues) plus
    the von Neumann entropy of a quantum state.

    Split the state by partition (the qubit indices of subsystem A; default is
    the first half), trace out B to obtain ρ_A, plot its eigenvalues (squared
    Schmidt coefficients) as a descending bar chart, and annotate the
    entanglement entropy. For two-qubit states also annotate the concurrence
    (Wootters formula, valid for both pure and mixed states).

    Parameters:
        state: 1D statevector / 2D density matrix / StatevectorEngine / Circuit.
        partition: list of qubit indices for subsystem A; None means the first
            floor(n/2) qubits.
        ax / show / save / title: same as plot_circuit.

    Returns: matplotlib Axes.
    """
    import numpy as np

    plt = _plt()
    rho = _to_density(state)
    n = round(math.log2(rho.shape[0]))

    if partition is None:
        partition = list(range(n // 2))
    partition = sorted(set(partition))
    if not partition or any(not 0 <= q < n for q in partition):
        raise ValueError(tr("err.viz_partition", n=n, partition=partition))

    rho_a = _partial_trace(rho, partition, n)
    eigvals = np.linalg.eigvalsh(rho_a)
    eigvals = np.sort(np.real(eigvals))[::-1]
    entropy = _von_neumann_entropy(eigvals)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.figure

    ax.bar(range(len(eigvals)), eigvals, color="#4C72B0")
    ax.set_xticks(range(len(eigvals)))
    ax.set_xticklabels([f"λ{i + 1}" for i in range(len(eigvals))])
    ax.set_ylabel("Reduced density matrix eigenvalues")
    ax.set_xlabel("Schmidt coefficients (descending)")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    label = f"Entanglement entropy S(ρ_A) = {entropy:.3f} bit"
    if n == 2 and len(partition) == 1:
        concurrence = _concurrence(rho)
        label += f"   Concurrence C = {concurrence:.3f}"
    ax.set_title(label)

    if title is not None:
        ax.set_title(f"{title}\n{label}")
    if save:
        fig.savefig(save, bbox_inches="tight", dpi=120)
    if show:
        plt.show()
    return ax


def plot_entanglement_profile(
    state: Any,
    ax: Any = None,
    show: bool = False,
    save: str | None = None,
    title: str | None = None,
) -> Any:
    """Draw the entanglement-entropy spectrum for every adjacent bipartition
    (0..k vs k+1..n-1).

    For each cut k compute the von Neumann entropy of the reduced density matrix
    ρ_{0..k} and plot it as a bar chart. GHZ states are 1 everywhere, product
    states are 0 everywhere, and low-entanglement chain states show a monotone
    distribution.

    Parameters:
        state: 1D statevector / 2D density matrix / StatevectorEngine / Circuit.
        ax / show / save / title: same as plot_circuit.

    Returns: matplotlib Axes.
    """
    import numpy as np

    plt = _plt()
    rho = _to_density(state)
    n = round(math.log2(rho.shape[0]))

    entropies = []
    for k in range(n - 1):
        rho_a = _partial_trace(rho, list(range(k + 1)), n)
        eigvals = np.linalg.eigvalsh(rho_a)
        entropies.append(_von_neumann_entropy(eigvals))

    if ax is None:
        fig, ax = plt.subplots(figsize=(max(5.0, n * 0.8), 4.0))
    else:
        fig = ax.figure

    cuts = list(range(n - 1))
    ax.bar(cuts, entropies, color="#4C72B0")
    ax.set_xticks(cuts)
    ax.set_xticklabels([f"k={k}" for k in cuts])
    ax.set_xlabel("Bipartition cut (0..k | k+1..n-1)")
    ax.set_ylabel("Entanglement entropy S (bit)")
    ax.set_ylim(0, max(1.0, (max(entropies) if entropies else 0) * 1.15))
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return finalize(fig, ax, show, save, title)


# ---------------------------------------------------------------------------
# Per-gate state evolution
# ---------------------------------------------------------------------------

def plot_state_evolution(
    circuit: Circuit,
    ax: Any = None,
    show: bool = False,
    save: str | None = None,
    title: str | None = None,
    top_k: int | None = 16,
) -> Any:
    """Draw the probability heatmap of the statevector evolving gate by gate
    through the circuit.

    The x-axis is the gate sequence (0 denotes the initial state), the y-axis is
    the basis states, and each cell is |amplitude|². When there are too many
    basis states, only the top_k with the largest probability peaks are kept
    (sorted by index).

    Parameters:
        circuit: A Circuit object.
        ax / show / save / title: same as plot_circuit.
        top_k: number of basis states to keep (the top k by peak probability
            over the whole run); None means all.

    Returns: matplotlib Axes.
    """
    import numpy as np

    from ..simulators import StatevectorEngine

    plt = _plt()
    n = circuit.num_qubits
    eng = StatevectorEngine(n)
    probs = [np.abs(eng.state) ** 2]
    for op in circuit.ops:
        if op.name == "measure":
            continue
        eng.apply(op.name, list(op.qubits), op.params)
        probs.append(np.abs(eng.state) ** 2)
    grid = np.array(probs).T  # shape (2^n, steps)

    shown = np.arange(2 ** n)
    if top_k is not None and grid.shape[0] > top_k:
        peak = grid.max(axis=1)
        shown = np.sort(np.argsort(peak)[::-1][:top_k])
        grid = grid[shown]

    if ax is None:
        fig, ax = plt.subplots(figsize=(max(6.0, grid.shape[1] * 0.4), max(3.0, grid.shape[0] * 0.3)))
    else:
        fig = ax.figure

    im = ax.imshow(grid, aspect="auto", cmap="Blues", interpolation="nearest", vmin=0, vmax=1)
    ax.set_yticks(range(grid.shape[0]))
    ax.set_yticklabels([f"|{format(i, f'0{n}b')}>" for i in shown], fontsize=7)
    ax.set_xlabel("Gate sequence (0 = initial state)")
    ax.set_ylabel("Basis state")
    fig.colorbar(im, ax=ax, label="|Amplitude|²")
    if title is None and grid.shape[0] != 2 ** n:
        title = f"State evolution (top {top_k} basis states by peak probability, {2 ** n} total)"
    return finalize(fig, ax, show, save, title)
