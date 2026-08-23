"""QuoNic full visualization suite — 12 chart types, using only matplotlib.

Visualization is an optional capability: `import quonic` does not pull in
matplotlib; it is lazy-loaded only when one of the plot_* functions is called.
Dependencies converge to the single `quonic[viz]` (matplotlib) extra, without
Graphviz / Seaborn / NetworkX.

    from quonic.viz import plot_circuit, plot_counts

    plot_circuit(circuit)          # gate-sequence circuit diagram
    plot_counts(result)            # measurement histogram

Every function accepts optional ax / show / save / title arguments and returns
matplotlib Axes (plot_statevector returns a list of Axes), making it easy to
embed into your own figure or export.
"""

from .algorithm import (
    plot_energy_convergence,
    plot_grover_amplitudes,
    plot_hamiltonian,
    plot_problem_graph,
)
from .circuit import (
    plot_circuit,
    plot_coupling_map,
    plot_qubit_activity,
    plot_statevector,
)
from .gate import plot_gate_matrix
from .noise import plot_noise_heatmap, plot_noisy_circuit
from .result import plot_counts
from .routing import plot_routing
from .scheduler import (
    plot_decision_tree,
    plot_fallback_chain,
    plot_feature_radar,
    plot_method_comparison,
    plot_method_heatmap,
)
from .state import (
    plot_bloch_multivector,
    plot_bloch_sphere,
    plot_density_matrix,
    plot_entanglement,
    plot_entanglement_profile,
    plot_state_evolution,
)
from .zne import plot_zne

__all__ = [
    "plot_bloch_multivector",
    "plot_bloch_sphere",
    "plot_circuit",
    "plot_counts",
    "plot_coupling_map",
    "plot_decision_tree",
    "plot_density_matrix",
    "plot_energy_convergence",
    "plot_entanglement",
    "plot_entanglement_profile",
    "plot_fallback_chain",
    "plot_feature_radar",
    "plot_gate_matrix",
    "plot_grover_amplitudes",
    "plot_hamiltonian",
    "plot_method_comparison",
    "plot_method_heatmap",
    "plot_noise_heatmap",
    "plot_noisy_circuit",
    "plot_problem_graph",
    "plot_qubit_activity",
    "plot_routing",
    "plot_state_evolution",
    "plot_statevector",
    "plot_zne",
]
