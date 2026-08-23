"""Result visualization: measurement histogram."""

from __future__ import annotations

from typing import Any

from .._i18n import tr
from ..result import Result
from ._mpl import _plt, finalize


def plot_counts(
    result: Result | dict[str, int],
    ax: Any = None,
    show: bool = False,
    save: str | None = None,
    title: str | None = None,
    top_k: int | None = 20,
) -> Any:
    """Draw the measurement histogram: x-axis is bit strings, y-axis is sample
    counts.

    Parameters:
        result: a Result (kind="counts") or a dict histogram.
        ax / show / save / title: same as plot_circuit.
        top_k: show only the top_k bit strings with the largest counts (sorted
            by bit string); None means show all. When a sampled histogram has
            many entries only the first 20 are shown by default, to avoid a
            cluttered bar chart.

    Returns: matplotlib Axes.
    """
    plt = _plt()
    if isinstance(result, Result):
        counts = result.counts or {}
    elif isinstance(result, dict):
        counts = result
    else:
        raise TypeError(tr("err.viz_counts"))

    truncated = top_k is not None and len(counts) > top_k
    if truncated:
        items = sorted(counts.items(), key=lambda kv: -kv[1])[:top_k]
        items.sort()
        labels = [k for k, _ in items]
    else:
        labels = sorted(counts)
    values = [counts[k] for k in labels]

    if ax is None:
        fig, ax = plt.subplots(figsize=(max(5.0, len(labels) * 0.8), 4.0))
    else:
        fig = ax.figure

    ax.bar(range(len(labels)), values, color="#4C72B0")
    ax.set_xticks(range(len(labels)))
    tick_labels = [f"|{k}>" for k in labels]
    max_len = max((len(t) for t in tick_labels), default=0)
    rotation = 90 if (len(labels) > 8 or max_len > 6) else 0
    fontsize = 9 if max_len <= 6 else (7 if max_len <= 10 else 6)
    ax.set_xticklabels(tick_labels, rotation=rotation, fontsize=fontsize)
    ax.set_ylabel("Counts")
    ax.set_xlabel("Bit string")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if title is None and truncated:
        title = f"Histogram (top {top_k} bit strings by count, {len(counts)} total)"
    return finalize(fig, ax, show, save, title)
