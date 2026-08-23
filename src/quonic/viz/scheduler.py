"""Scheduler visualization: method comparison, decision tree, selection heatmap, fallback chain, feature radar chart."""

from __future__ import annotations

from typing import Any

from .._i18n import tr
from ..ir import Circuit
from ..scheduler import circuit_features, load_measured_decision, recommend_method
from ..scheduler.registry import load_performance
from ._mpl import _plt, finalize

# ---------------------------------------------------------------------------
# 4. Method comparison line plot
# ---------------------------------------------------------------------------

def plot_method_comparison(
    cls: str = "clifford",
    ax: Any = None,
    show: bool = False,
    save: str | None = None,
    title: str | None = None,
) -> Any:
    """Draw the elapsed-time line for each simulation method vs. growing qubit
    count (log y-axis).

    Parameters:
        cls: decision class "clifford" or "low_tw" (the circuit family in the
            benchmark).
        ax / show / save / title: same as plot_circuit.

    Returns: matplotlib Axes.
    """
    plt = _plt()
    perf = [p for p in load_performance() if p.get("class") == cls]
    if not perf:
        raise ValueError(tr("err.viz_no_perf", cls=cls))

    ns = sorted({p["n"] for p in perf})
    methods = sorted({m for p in perf for m in p["timings"]})

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4.5))
    else:
        fig = ax.figure

    for method in methods:
        xs, ys = [], []
        for n in ns:
            row = next((p for p in perf if p["n"] == n), None)
            if row is not None and method in row["timings"]:
                xs.append(n)
                ys.append(row["timings"][method])
        ax.plot(xs, ys, marker="o", label=method)

    ax.set_xlabel("Number of qubits n")
    ax.set_ylabel("Time (s)")
    ax.set_yscale("log")
    ax.legend()
    ax.set_title(f"Decision class: {cls}")
    return finalize(fig, ax, show, save, title)


# ---------------------------------------------------------------------------
# 5. Scheduling decision tree
# ---------------------------------------------------------------------------

class _Node:
    __slots__ = ("children", "label", "x", "y")

    def __init__(
        self, label: str, children: list[tuple[str, _Node]] | None = None
    ) -> None:
        self.label: str = label
        self.children: list[tuple[str, _Node]] = children or []
        self.x: float = 0.0
        self.y: float = 0.0


_LAYER_SPACING = 1.6  # y spacing between adjacent layers (top to bottom)
_LEAF_SPACING = 2.0   # x spacing between adjacent leaves


def _assign(
    node: _Node, depth: int, next_leaf: list[int], max_depth: list[int]
) -> None:
    node.y = depth * _LAYER_SPACING
    max_depth[0] = max(max_depth[0], depth)
    if not node.children:
        node.x = next_leaf[0] * _LEAF_SPACING
        next_leaf[0] += 1
        return
    for _, child in node.children:
        _assign(child, depth + 1, next_leaf, max_depth)
    node.x = sum(c.x for _, c in node.children) / len(node.children)


def _build_decision_tree() -> _Node:
    decision = load_measured_decision() or {}
    cliff_n = decision.get("clifford", {}).get("above_n", 24)
    lowtw_n = decision.get("low_tw", {}).get("above_n", 24)

    root = _Node("Circuit")
    noise_yes = _Node("DM", [])
    classify = _Node("Decision class")

    general = _Node("SV", [])
    cliff = _Node(f"n ≥ {cliff_n}?")
    cliff.children = [
        ("Yes", _Node("Stab", [])),
        ("No", _Node("SV", [])),
    ]
    lowtw = _Node(f"n ≥ {lowtw_n}?")
    lowtw.children = [
        ("Yes", _Node("MPS", [])),
        ("No", _Node("SV", [])),
    ]

    classify.children = [
        ("general", general),
        ("clifford", cliff),
        ("low_tw", lowtw),
    ]
    root.children = [("Has noise", noise_yes), ("No noise", classify)]
    return root


def plot_decision_tree(
    ax: Any = None,
    show: bool = False,
    save: str | None = None,
    title: str | None = None,
) -> Any:
    """Draw the scheduling decision tree: noise → density_matrix, others choose
    the method by class and crossover point.

    Parameters:
        ax / show / save / title: same as plot_circuit.

    Returns: matplotlib Axes.
    """
    from matplotlib.patches import FancyArrowPatch, Rectangle

    plt = _plt()
    root = _build_decision_tree()
    next_leaf = [0]
    max_depth = [0]
    _assign(root, 0, next_leaf, max_depth)

    depth = max_depth[0]
    n_leaves = next_leaf[0]
    xmax = (n_leaves - 1) * _LEAF_SPACING
    ymax = depth * _LAYER_SPACING

    if ax is None:
        fig, ax = plt.subplots(
            figsize=(max(7.0, xmax * 0.8 + 1.6), max(4.0, ymax * 0.8 + 1.2))
        )
    else:
        fig = ax.figure

    def draw(node: _Node) -> None:
        # Draw child edges first, then the node itself, so nodes sit on top of edges
        for edge_label, child in node.children:
            ax.add_patch(
                FancyArrowPatch(
                    (node.x, node.y + 0.22),
                    (child.x, child.y - 0.22),
                    arrowstyle="-",
                    color="0.5",
                    lw=1.2,
                )
            )
            mx, my = (node.x + child.x) / 2, (node.y + child.y) / 2
            ax.text(
                mx,
                my,
                edge_label,
                ha="center",
                va="center",
                fontsize=8,
                color="0.25",
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5},
                zorder=5,
            )
            draw(child)
        ax.add_patch(
            Rectangle(
                (node.x - 0.5, node.y - 0.22),
                1.0,
                0.44,
                facecolor="#4C72B0",
                edgecolor="black",
                zorder=2,
            )
        )
        ax.text(node.x, node.y, node.label, ha="center", va="center",
                fontsize=8, color="white", zorder=3)

    draw(root)
    ax.set_xlim(-1.2, xmax + 1.2)
    ax.set_ylim(-0.6, ymax + 0.6)
    ax.invert_yaxis()
    ax.axis("off")
    fig.subplots_adjust(bottom=0.12)
    fig.text(0.5, 0.02, _method_legend(), ha="center", va="bottom", fontsize=8, color="0.4")
    return finalize(fig, ax, show, save, title)


# ---------------------------------------------------------------------------
# 6. Method selection heatmap
# ---------------------------------------------------------------------------

_METHODS = ["statevector", "stabilizer", "matrix_product_state", "density_matrix"]
_METHOD_COLORS = {
    "statevector": 0,
    "stabilizer": 1,
    "matrix_product_state": 2,
    "density_matrix": 3,
}
_METHOD_ABBREV = {
    "statevector": "SV",
    "stabilizer": "Stab",
    "matrix_product_state": "MPS",
    "density_matrix": "DM",
}


def _method_legend() -> str:
    return "   ".join(f"{abbr} = {name}" for name, abbr in _METHOD_ABBREV.items())


def _features_for_class(cls: str, n: int) -> dict[str, Any]:
    if cls == "clifford":
        return {"n": n, "gate_types": ["cx", "h"], "treewidth_ub": 1}
    if cls == "low_tw":
        return {"n": n, "gate_types": ["cx", "rz"], "treewidth_ub": 1}
    # general: high treewidth, non-Clifford
    return {"n": n, "gate_types": ["mcz"], "treewidth_ub": n - 1}


def plot_method_heatmap(
    ax: Any = None,
    show: bool = False,
    save: str | None = None,
    title: str | None = None,
) -> Any:
    """Draw the "decision class × qubit count → method chosen by the scheduler"
    heatmap.

    Parameters:
        ax / show / save / title: same as plot_circuit.

    Returns: matplotlib Axes.
    """
    plt = _plt()
    rows = ["clifford", "low_tw", "general", "noisy"]
    ns = [8, 12, 16, 20, 24, 28]

    grid = []
    for cls in rows:
        row = []
        for n in ns:
            if cls == "noisy":
                method = "density_matrix"
            else:
                method = recommend_method(_features_for_class(cls, n))
            row.append(method)
        grid.append(row)

    data = [[_METHOD_COLORS[m] for m in row] for row in grid]

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4.5))
    else:
        fig = ax.figure

    ax.imshow(data, cmap="viridis", aspect="auto", vmin=0, vmax=len(_METHODS) - 1)
    for i in range(len(rows)):
        for j in range(len(ns)):
            ax.text(j, i, _METHOD_ABBREV[grid[i][j]], ha="center", va="center", fontsize=7)

    ax.set_xticks(range(len(ns)))
    ax.set_xticklabels(ns)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows)
    ax.set_xlabel("Number of qubits n")
    ax.set_ylabel("Decision class")
    fig.subplots_adjust(bottom=0.18)
    fig.text(0.5, 0.02, _method_legend(), ha="center", va="bottom", fontsize=8, color="0.4")
    return finalize(fig, ax, show, save, title)


# ---------------------------------------------------------------------------
# 7. Fallback chain path diagram
# ---------------------------------------------------------------------------

def plot_fallback_chain(
    ax: Any = None,
    show: bool = False,
    save: str | None = None,
    title: str | None = None,
) -> Any:
    """Draw the backend fallback chain: qiskit → cirq → pennylane → native
    (in-house fallback).

    Parameters:
        ax / show / save / title: same as plot_circuit.

    Returns: matplotlib Axes.
    """
    from matplotlib.patches import FancyArrowPatch, Rectangle

    plt = _plt()
    chain = [
        ("qiskit", "Aer"),
        ("cirq", "Google"),
        ("pennylane", "Quantum ML"),
        ("native", "In-house engine fallback"),
    ]
    n = len(chain)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8.5, 2.6))
    else:
        fig = ax.figure

    for i, (name, desc) in enumerate(chain):
        ax.add_patch(
            Rectangle((i - 0.38, -0.22), 0.76, 0.44, facecolor="#4C72B0",
                      edgecolor="black", zorder=2)
        )
        ax.text(i, 0, name, ha="center", va="center", color="white", fontsize=9, zorder=3)
        ax.text(i, -0.42, desc, ha="center", va="center", fontsize=7, color="0.4")
        if i < n - 1:
            ax.add_patch(
                FancyArrowPatch((i + 0.42, 0), (i + 0.58, 0), arrowstyle="-|>",
                                mutation_scale=14, color="0.4", lw=1.2)
            )
            ax.text(i + 0.5, 0.16, "Not installed\n/ unsupported", ha="center", va="bottom",
                    fontsize=6.5, color="0.4")

    ax.set_xlim(-0.7, n - 0.3)
    ax.set_ylim(-0.7, 0.5)
    ax.axis("off")
    return finalize(fig, ax, show, save, title)


# ---------------------------------------------------------------------------
# 9. Circuit feature radar chart
# ---------------------------------------------------------------------------

def plot_feature_radar(
    circuit_or_features: Circuit | dict[str, Any],
    ax: Any = None,
    show: bool = False,
    save: str | None = None,
    title: str | None = None,
) -> Any:
    """Draw a multi-dimensional feature radar chart (polar) of a single circuit.

    Parameters:
        circuit_or_features: a Circuit or the dict from circuit_features(circuit).
        ax / show / save / title: same as plot_circuit.

    Returns: matplotlib Axes.
    """
    import numpy as np

    plt = _plt()
    if isinstance(circuit_or_features, Circuit):
        feats = circuit_features(circuit_or_features)
    else:
        feats = circuit_or_features

    dims = ["n", "depth", "gate_count", "treewidth_ub", "clifford"]
    labels = ["Qubits", "Depth", "Gates", "Treewidth", "Clifford"]
    raw = [
        feats.get("n", 0) / 30.0,
        feats.get("depth", 0) / 100.0,
        feats.get("gate_count", 0) / 200.0,
        feats.get("treewidth_ub", 0) / 10.0,
        1.0 if feats.get("is_clifford") else 0.0,
    ]
    values = [min(max(v, 0.0), 1.0) for v in raw]

    if ax is None:
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111, projection="polar")
    else:
        fig = ax.figure

    angles = np.linspace(0, 2 * np.pi, len(dims), endpoint=False).tolist()
    values_c = values + values[:1]
    angles_c = angles + angles[:1]

    ax.plot(angles_c, values_c, color="#4C72B0", lw=1.5)
    ax.fill(angles_c, values_c, color="#4C72B0", alpha=0.25)
    ax.set_xticks(angles)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"], fontsize=7)
    if title is not None:
        ax.set_title(title)
    if save:
        fig.savefig(save, bbox_inches="tight", dpi=120)
    if show:
        plt.show()
    return ax
