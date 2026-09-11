"""
Visualization for decision boundary analysis.
"""

from typing import Optional, Any
import numpy as np

from .platforms import PlatformParams, get_platform
from .analyzer import analyze_circuit
from .boundary import Strategy, calculate_decision, compare_strategies


def qshow_decision(
    circuit: Any,
    platform: Any,
    backend: Any = None,
    show_comparison: bool = True,
):
    """Enhanced qshow with decision boundary analysis.

    Args:
        circuit: Quantum circuit
        platform: Platform name or PlatformParams instance
        backend: Optional backend for transpilation analysis
        show_comparison: Whether to show detailed comparison

    Example:
        >>> from quonic.decision import qshow_decision
        >>> qshow_decision(circuit, "ibm_heron")
    """
    # Get platform parameters
    if isinstance(platform, str):
        platform_params = get_platform(platform)
    else:
        platform_params = platform

    # Analyze circuit
    analysis = analyze_circuit(circuit, backend)

    # Calculate decision
    decision = calculate_decision(analysis, platform_params)

    # Print analysis
    print(analysis.summary())
    print()
    print(platform_params.summary())
    print()
    print(decision.summary())

    # Show detailed comparison if requested
    if show_comparison:
        print()
        print("=" * 60)
        print("Detailed Comparison")
        print("=" * 60)

        comparison = compare_strategies(analysis, platform_params)

        print(f"\nFor N = {comparison['N']} conditional operations:")
        if comparison['F_dynamic'] is not None:
            print(f"  Dynamic fidelity:  {comparison['F_dynamic']:.6f}")
        if comparison['F_unitary'] is not None:
            print(f"  Unitary fidelity:  {comparison['F_unitary']:.6f}")
        print(f"  Advantage:         {comparison['advantage']:.2f}x")
        print(f"  Recommended:       {comparison['recommended'].upper()}")


def plot_decision_boundary(
    platform: Any,
    c_range: tuple = (1, 10),
    d_range: tuple = (1, 100),
    n_points: int = 50,
    save_path: Optional[str] = None,
):
    """Plot the decision boundary in (c, d) space.

    Args:
        platform: Platform name or PlatformParams instance
        c_range: Range of compilation overhead factor c
        d_range: Range of bare unitary depth d
        n_points: Number of points in each dimension
        save_path: Optional path to save the plot

    Example:
        >>> from quonic.decision import plot_decision_boundary
        >>> plot_decision_boundary("ibm_heron")
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available. Install with: pip install matplotlib")
        return

    # Get platform parameters
    if isinstance(platform, str):
        platform_params = get_platform(platform)
    else:
        platform_params = platform

    threshold = platform_params.threshold

    if threshold is None:
        print("Cannot plot: insufficient platform parameters")
        return

    # Create grid
    c_vals = np.linspace(c_range[0], c_range[1], n_points)
    d_vals = np.linspace(d_range[0], d_range[1], n_points)
    C, D = np.meshgrid(c_vals, d_vals)

    # Calculate (c-1)*d
    CD = (C - 1) * D

    # Decision: unitary wins when (c-1)*d < threshold
    decision = (CD < threshold).astype(float)

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot decision regions
    contour = ax.contourf(C, D, decision, levels=[0, 0.5, 1],
                          colors=['#ff6b6b', '#51cf66'], alpha=0.3)
    ax.contour(C, D, CD, levels=[threshold], colors='black',
               linestyles='--', linewidths=2)

    # Labels and title
    ax.set_xlabel('Compilation Overhead Factor (c)', fontsize=12)
    ax.set_ylabel('Bare Unitary Depth (d)', fontsize=12)
    ax.set_title(
        f'Decision Boundary: {platform_params.name}\n'
        f'Threshold = {threshold:.1f}',
        fontsize=14
    )

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#51cf66', alpha=0.3, label='Unitary preferred'),
        Patch(facecolor='#ff6b6b', alpha=0.3, label='Dynamic preferred'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=11)

    # Add threshold line label
    ax.text(0.95, 0.95, f'(c-1)*d = {threshold:.0f}',
            transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()

    return fig


def plot_platform_comparison(
    circuit: Any,
    platform_names: list,
    backend: Any = None,
    save_path: Optional[str] = None,
):
    """Plot comparison of strategies across platforms.

    Args:
        circuit: Quantum circuit
        platform_names: List of platform names
        backend: Optional backend for transpilation analysis
        save_path: Optional path to save the plot
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available. Install with: pip install matplotlib")
        return

    from .selector import compare_platforms

    # Get comparison results
    results = compare_platforms(circuit, platform_names, backend)

    # Filter out errors
    valid_results = [r for r in results if r["strategy"] != "error"]

    if not valid_results:
        print("No valid results to plot")
        return

    # Extract data
    platforms = [r["platform"] for r in valid_results]
    advantages = [r["advantage"] for r in valid_results]
    thresholds = [r["threshold"] if r["threshold"] else 0 for r in valid_results]
    strategies = [r["strategy"] for r in valid_results]

    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot 1: Advantages
    colors = ['#51cf66' if s == 'unitary' else '#ff6b6b' for s in strategies]
    axes[0].barh(platforms, advantages, color=colors)
    axes[0].set_xlabel('Expected Advantage (x)')
    axes[0].set_title('Unitary Advantage')
    axes[0].axvline(x=1, color='black', linestyle='--', alpha=0.5)

    # Plot 2: Thresholds
    axes[1].barh(platforms, thresholds, color='#339af0')
    axes[1].set_xlabel('Decision Threshold')
    axes[1].set_title('Platform Threshold')

    # Plot 3: Strategy summary
    strategy_counts = {}
    for s in strategies:
        strategy_counts[s] = strategy_counts.get(s, 0) + 1

    axes[2].pie(
        strategy_counts.values(),
        labels=strategy_counts.keys(),
        autopct='%1.1f%%',
        colors=['#51cf66', '#ff6b66', '#ffd43b']
    )
    axes[2].set_title('Strategy Distribution')

    plt.suptitle(f'Platform Comparison', fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()

    return fig
