"""Gradient visualization for quantum machine learning.

Provides tools to visualize gradient flow, training convergence,
and circuit analysis for QML experiments.

Example::

    from quonic.ml.viz import plot_gradient_flow, plot_training_convergence

    plot_gradient_flow(result)
    plot_training_convergence(result)
"""

from __future__ import annotations

from typing import Any

import numpy as np


def plot_gradient_flow(
    grad_history: list[np.ndarray],
    title: str = "Gradient Flow",
    show: bool = True,
    save: str | None = None,
) -> Any:
    """Plot gradient flow over training iterations.

    Shows how gradients evolve during training, useful for diagnosing
    vanishing/exploding gradient problems.

    Args:
        grad_history: list of gradient vectors at each iteration
        title: plot title
        show: display plot
        save: save to file path

    Returns:
        matplotlib Axes object.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib required for visualization. Install with: pip install matplotlib")
        return None

    _fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Gradient norm over iterations
    norms = [np.linalg.norm(g) for g in grad_history]
    axes[0].plot(norms, 'b-', linewidth=2)
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Gradient Norm")
    axes[0].set_title("Gradient Norm over Training")
    axes[0].set_yscale('log')
    axes[0].grid(True, alpha=0.3)

    # Gradient distribution (last iteration)
    if grad_history:
        axes[1].hist(grad_history[-1], bins=30, edgecolor='black', alpha=0.7)
        axes[1].set_xlabel("Gradient Value")
        axes[1].set_ylabel("Frequency")
        axes[1].set_title("Gradient Distribution (Last Iteration)")
        axes[1].grid(True, alpha=0.3)

    plt.suptitle(title)
    plt.tight_layout()

    if save:
        plt.savefig(save, dpi=150, bbox_inches='tight')
    if show:
        plt.show()

    return axes


def plot_training_convergence(
    loss_history: list[float],
    title: str = "Training Convergence",
    show: bool = True,
    save: str | None = None,
) -> Any:
    """Plot training loss convergence.

    Shows how the loss function decreases over training iterations.

    Args:
        loss_history: loss values at each iteration
        title: plot title
        show: display plot
        save: save to file path

    Returns:
        matplotlib Axes object.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib required for visualization. Install with: pip install matplotlib")
        return None

    _fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss over iterations
    axes[0].plot(loss_history, 'b-', linewidth=2)
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss over Training")
    axes[0].grid(True, alpha=0.3)

    # Log scale loss
    axes[1].plot(loss_history, 'b-', linewidth=2)
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Loss (log scale)")
    axes[1].set_title("Loss over Training (Log Scale)")
    axes[1].set_yscale('log')
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(title)
    plt.tight_layout()

    if save:
        plt.savefig(save, dpi=150, bbox_inches='tight')
    if show:
        plt.show()

    return axes


def plot_parameter_distribution(
    params: np.ndarray,
    title: str = "Parameter Distribution",
    show: bool = True,
    save: str | None = None,
) -> Any:
    """Plot distribution of circuit parameters.

    Args:
        params: parameter vector
        title: plot title
        show: display plot
        save: save to file path

    Returns:
        matplotlib Axes object.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib required for visualization. Install with: pip install matplotlib")
        return None

    _fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Parameter values
    axes[0].bar(range(len(params)), params, alpha=0.7)
    axes[0].set_xlabel("Parameter Index")
    axes[0].set_ylabel("Value")
    axes[0].set_title("Parameter Values")
    axes[0].grid(True, alpha=0.3)

    # Parameter histogram
    axes[1].hist(params, bins=30, edgecolor='black', alpha=0.7)
    axes[1].set_xlabel("Parameter Value")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Parameter Distribution")
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(title)
    plt.tight_layout()

    if save:
        plt.savefig(save, dpi=150, bbox_inches='tight')
    if show:
        plt.show()

    return axes


def plot_circuit_analysis(
    circuit: Any,
    title: str = "Circuit Analysis",
    show: bool = True,
    save: str | None = None,
) -> Any:
    """Plot circuit structure analysis.

    Shows gate distribution, depth, and qubit usage.

    Args:
        circuit: quantum circuit
        title: plot title
        show: display plot
        save: save to file path

    Returns:
        matplotlib Axes object.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib required for visualization. Install with: pip install matplotlib")
        return None

    from collections import Counter

    # Count gates
    gate_counts = Counter()
    qubit_usage = Counter()
    for op in circuit.ops:
        if hasattr(op, 'name') and op.name != 'measure':
            gate_counts[op.name] += 1
            for q in op.qubits:
                qubit_usage[q] += 1

    _fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Gate distribution
    if gate_counts:
        names = list(gate_counts.keys())
        counts = list(gate_counts.values())
        axes[0].bar(names, counts, alpha=0.7)
        axes[0].set_xlabel("Gate Type")
        axes[0].set_ylabel("Count")
        axes[0].set_title("Gate Distribution")
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].grid(True, alpha=0.3)

    # Qubit usage
    if qubit_usage:
        qubits = sorted(qubit_usage.keys())
        usage = [qubit_usage[q] for q in qubits]
        axes[1].bar(qubits, usage, alpha=0.7)
        axes[1].set_xlabel("Qubit Index")
        axes[1].set_ylabel("Gate Count")
        axes[1].set_title("Qubit Usage")
        axes[1].grid(True, alpha=0.3)

    # Circuit depth (simplified)
    qubit_depths = {}
    for op in circuit.ops:
        if hasattr(op, 'name') and op.name != 'measure':
            for q in op.qubits:
                qubit_depths[q] = qubit_depths.get(q, 0) + 1
    if qubit_depths:
        max_depth = max(qubit_depths.values())
        axes[2].barh(list(qubit_depths.keys()), list(qubit_depths.values()), alpha=0.7)
        axes[2].set_xlabel("Depth")
        axes[2].set_ylabel("Qubit")
        axes[2].set_title(f"Circuit Depth (max={max_depth})")
        axes[2].grid(True, alpha=0.3)

    plt.suptitle(title)
    plt.tight_layout()

    if save:
        plt.savefig(save, dpi=150, bbox_inches='tight')
    if show:
        plt.show()

    return axes
