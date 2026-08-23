"""matplotlib lazy-loading + common plotting helpers.

Visualization is an optional capability: `import quonic` does not pull in
matplotlib; it is loaded only when a plot_* function is actually called.
Chinese font configuration and save/show finalization all live here.
"""

from __future__ import annotations

from typing import Any

from .._i18n import tr

_MPL = None


def _plt() -> Any:
    """Lazy-load matplotlib.pyplot; give a hint when it is not installed."""
    global _MPL
    if _MPL is None:
        try:
            import matplotlib.pyplot as plt

            _configure_chinese_font()
            _MPL = plt
        except ImportError as e:
            raise ImportError(tr("err.viz_matplotlib")) from e
    return _MPL


def _configure_chinese_font() -> None:
    """Try to enable a Chinese font (fall back silently to English if not found, without affecting plotting)."""
    import matplotlib
    from matplotlib import font_manager

    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in (
        "Microsoft YaHei",
        "SimHei",
        "PingFang SC",
        "Noto Sans CJK SC",
        "WenQuanYi Micro Hei",
    ):
        if name in available:
            matplotlib.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            break
    matplotlib.rcParams["axes.unicode_minus"] = False


def finalize(
    fig: Any,
    ax: Any = None,
    show: bool = False,
    save: str | None = None,
    title: str | None = None,
) -> Any:
    """Unified finalization: set title, save, show, return ax (return fig when no ax is provided)."""
    if title is not None and ax is not None:
        ax.set_title(title)
    if save:
        fig.savefig(save, bbox_inches="tight", dpi=120)
    if show:
        _plt().show()
    return ax if ax is not None else fig


def new_ax(figsize: tuple[float, float] = (6, 4)) -> tuple[Any, Any]:
    """Convenience entry point for creating a new figure + ax."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax
