"""
Integration with QuoNic's Circuit IR.
"""

from dataclasses import dataclass, field
from typing import Any

from ..ir import Circuit, ClassicalIfOperation, ClassicalWhileOperation


@dataclass
class QuoNicCircuitAnalysis:
    """Analysis results for a QuoNic Circuit.

    Attributes:
        n_qubits: Number of qubits
        depth: Circuit depth
        n_gates: Total gate count
        n_conditional: Number of conditional operations (cif, cwhile)
        c: Compilation overhead factor
        d: Bare unitary depth
        has_midcircuit_measurement: Whether circuit has mid-circuit measurement
        control_structures: Detected control structures
        gate_counts: Dictionary of gate type -> count
    """

    n_qubits: int
    depth: int
    n_gates: int
    n_conditional: int
    c: float
    d: int
    has_midcircuit_measurement: bool
    control_structures: list = field(default_factory=list)
    gate_counts: dict = field(default_factory=dict)

    @property
    def cd(self) -> float:
        """(c-1)*d for decision boundary."""
        return (self.c - 1) * self.d

    def summary(self) -> str:
        """Return a summary string."""
        lines = [
            "Circuit Analysis:",
            f"  Qubits: {self.n_qubits}",
            f"  Depth: {self.depth}",
            f"  Gates: {self.n_gates}",
            f"  Conditional ops: {self.n_conditional}",
            f"  Compilation overhead c: {self.c:.2f}",
            f"  Bare unitary depth d: {self.d}",
            f"  (c-1)*d: {self.cd:.1f}",
            f"  Mid-circuit measurement: {self.has_midcircuit_measurement}",
        ]
        if self.control_structures:
            lines.append(f"  Control structures: {', '.join(self.control_structures)}")
        return "\n".join(lines)


def analyze_quonic_circuit(
    circuit: Circuit,
    coupling_map: Any = None,
) -> QuoNicCircuitAnalysis:
    """Analyze a QuoNic Circuit for decision boundary calculation.

    Args:
        circuit: QuoNic Circuit instance
        coupling_map: Optional CouplingMap for connectivity analysis

    Returns:
        QuoNicCircuitAnalysis instance
    """
    # Basic circuit properties
    n_qubits = circuit.num_qubits
    depth = circuit.depth()
    n_gates = circuit.gate_count()

    # Count gate types
    gate_counts = {}
    for op in circuit.ops:
        name = op.name if hasattr(op, 'name') else str(type(op).__name__)
        gate_counts[name] = gate_counts.get(name, 0) + 1

    # Detect control structures and conditional operations
    control_structures = []
    n_conditional = 0
    has_mcm = False

    for op in circuit.ops:
        # Check for classical control flow
        if isinstance(op, ClassicalIfOperation):
            control_structures.append("if")
            n_conditional += 1
        elif isinstance(op, ClassicalWhileOperation):
            control_structures.append("while")
            n_conditional += 1

        # Check for mid-circuit measurement
        if hasattr(op, 'name'):
            if op.name == 'cmeasure':
                has_mcm = True
            elif op.name == 'cif' and isinstance(getattr(op, 'control', None), int):
                # cif with qubit control implies measurement
                has_mcm = True

    # Remove duplicates
    control_structures = sorted(set(control_structures))

    # Estimate compilation overhead
    c = _estimate_c_from_coupling_map(circuit, coupling_map)

    # Get bare unitary depth
    d = _get_bare_depth(circuit, n_conditional)

    return QuoNicCircuitAnalysis(
        n_qubits=n_qubits,
        depth=depth,
        n_gates=n_gates,
        n_conditional=n_conditional,
        c=c,
        d=d,
        has_midcircuit_measurement=has_mcm,
        control_structures=control_structures,
        gate_counts=gate_counts,
    )


def _estimate_c_from_coupling_map(
    circuit: Circuit,
    coupling_map: Any,
) -> float:
    """Estimate compilation overhead factor from coupling map.

    For fully connected: c = 1
    For limited connectivity: c > 1 (SWAP overhead)
    """
    if coupling_map is None:
        return 1.0

    # Count two-qubit gates
    two_qubit_gates = [
        op for op in circuit.ops
        if hasattr(op, 'qubits') and len(op.qubits) == 2
    ]

    if not two_qubit_gates:
        return 1.0

    # Check how many need routing
    needs_routing = 0
    for op in two_qubit_gates:
        q0, q1 = op.qubits
        if not coupling_map.has_edge(q0, q1):
            needs_routing += 1

    # Estimate overhead: each routed gate needs ~2-4 SWAPs
    # This is a simplified estimate
    if needs_routing == 0:
        return 1.0

    # Rough estimate: 1 + 2 * (fraction of gates needing routing)
    return 1.0 + 2.0 * (needs_routing / len(two_qubit_gates))


def _get_bare_depth(circuit: Circuit, n_conditional: int) -> int:
    """Get bare unitary depth.

    For the decision boundary, d is the depth of the bare unitary U
    before controlled compilation.
    """
    # Simplified: use circuit depth
    # Real implementation would separate unitary and measurement parts
    return circuit.depth()


def select_strategy_for_quonic(
    circuit: Circuit,
    platform_name: str,
    coupling_map: Any = None,
):
    """Select compilation strategy for a QuoNic Circuit.

    Args:
        circuit: QuoNic Circuit instance
        platform_name: Platform identifier
        coupling_map: Optional CouplingMap

    Returns:
        Tuple of (Strategy, DecisionResult)
    """
    from .boundary import calculate_decision
    from .platforms import get_platform

    # Analyze circuit
    analysis = analyze_quonic_circuit(circuit, coupling_map)

    # Get platform parameters
    platform = get_platform(platform_name)

    # Convert to generic CircuitAnalysis for boundary calculation
    from .analyzer import CircuitAnalysis
    generic_analysis = CircuitAnalysis(
        n_qubits=analysis.n_qubits,
        depth=analysis.depth,
        n_conditional=analysis.n_conditional,
        c=analysis.c,
        d=analysis.d,
        has_midcircuit_measurement=analysis.has_midcircuit_measurement,
        control_structures=analysis.control_structures,
        n_gates=analysis.n_gates,
        gate_counts=analysis.gate_counts,
    )

    # Calculate decision
    decision = calculate_decision(generic_analysis, platform)

    return decision.strategy, decision
