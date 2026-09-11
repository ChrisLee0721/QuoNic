"""
Circuit analyzer for decision boundary calculation.
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class CircuitAnalysis:
    """Analysis results for a quantum circuit.

    Attributes:
        n_qubits: Number of qubits
        depth: Transpiled circuit depth
        n_conditional: Number of conditional operations
        c: Compilation overhead factor (c >= 1)
        d: Bare unitary depth
        has_midcircuit_measurement: Whether circuit has mid-circuit measurement
        control_structures: Detected control structures (for, if, switch, while)
        n_gates: Total number of gates
        gate_counts: Dictionary of gate type -> count
    """

    n_qubits: int
    depth: int
    n_conditional: int
    c: float
    d: int
    has_midcircuit_measurement: bool
    control_structures: List[str] = field(default_factory=list)
    n_gates: int = 0
    gate_counts: dict = field(default_factory=dict)

    @property
    def cd(self) -> float:
        """(c-1)*d for decision boundary."""
        return (self.c - 1) * self.d

    def summary(self) -> str:
        """Return a summary string of circuit analysis."""
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


def analyze_circuit(
    circuit: Any,
    backend: Any = None,
    transpiled: bool = False,
) -> CircuitAnalysis:
    """Analyze circuit for decision boundary calculation.

    Args:
        circuit: Quantum circuit (Qiskit QuantumCircuit or similar)
        backend: Optional backend for transpilation analysis
        transpiled: Whether circuit is already transpiled

    Returns:
        CircuitAnalysis instance
    """
    # Basic circuit properties
    n_qubits = _get_n_qubits(circuit)
    depth = _get_depth(circuit)
    n_gates = _get_n_gates(circuit)
    gate_counts = _get_gate_counts(circuit)

    # Detect control structures
    control_structures = _detect_control_structures(circuit)
    has_mcm = _has_midcircuit_measurement(circuit)

    # Count conditional operations
    n_conditional = _count_conditional_operations(circuit)

    # Estimate compilation overhead
    c = _estimate_compilation_overhead(circuit, backend, transpiled)

    # Get bare unitary depth
    d = _get_bare_depth(circuit, n_conditional)

    return CircuitAnalysis(
        n_qubits=n_qubits,
        depth=depth,
        n_conditional=n_conditional,
        c=c,
        d=d,
        has_midcircuit_measurement=has_mcm,
        control_structures=control_structures,
        n_gates=n_gates,
        gate_counts=gate_counts,
    )


def _get_n_qubits(circuit: Any) -> int:
    """Get number of qubits from circuit."""
    # Qiskit
    if hasattr(circuit, 'num_qubits'):
        return circuit.num_qubits
    # Generic
    if hasattr(circuit, 'n_qubits'):
        return circuit.n_qubits
    if hasattr(circuit, 'qubits'):
        return len(circuit.qubits)
    return 0


def _get_depth(circuit: Any) -> int:
    """Get circuit depth."""
    if hasattr(circuit, 'depth'):
        depth = circuit.depth
        if callable(depth):
            return depth()
        return depth
    return 0


def _get_n_gates(circuit: Any) -> int:
    """Get total number of gates."""
    if hasattr(circuit, 'size'):
        size = circuit.size
        if callable(size):
            return size()
        return size
    if hasattr(circuit, 'data'):
        return len(circuit.data)
    return 0


def _get_gate_counts(circuit: Any) -> dict:
    """Get gate type counts."""
    counts = {}

    # Qiskit
    if hasattr(circuit, 'count_ops'):
        return circuit.count_ops()

    # Generic: iterate through data
    if hasattr(circuit, 'data'):
        for instruction in circuit.data:
            gate_name = _get_gate_name(instruction)
            if gate_name:
                counts[gate_name] = counts.get(gate_name, 0) + 1

    return counts


def _get_gate_name(instruction: Any) -> Optional[str]:
    """Extract gate name from instruction."""
    # Qiskit CircuitInstruction
    if hasattr(instruction, 'operation'):
        return instruction.operation.name
    # Qiskit older format (instruction, qargs, cargs)
    if isinstance(instruction, tuple) and len(instruction) >= 1:
        return instruction[0].name
    # Generic
    if hasattr(instruction, 'name'):
        return instruction.name
    return None


def _detect_control_structures(circuit: Any) -> List[str]:
    """Detect classical control structures in circuit."""
    structures = set()

    if not hasattr(circuit, 'data'):
        return []

    for instruction in circuit.data:
        gate_name = _get_gate_name(instruction)

        if gate_name is None:
            continue

        # Detect if/switch (conditional gates)
        if _is_conditional(instruction):
            structures.add("if")

        # Detect while (loops - heuristic based on repeated patterns)
        # This is simplified; real detection would need higher-level IR

        # Detect for (unrolled loops - hard to detect from gates alone)

    return sorted(structures)


def _is_conditional(instruction: Any) -> bool:
    """Check if instruction is a conditional gate."""
    # Qiskit
    if hasattr(instruction, 'operation'):
        op = instruction.operation
        if hasattr(op, 'condition') and op.condition is not None:
            return True
        if hasattr(op, 'condition_bits') and op.condition_bits:
            return True
    return False


def _has_midcircuit_measurement(circuit: Any) -> bool:
    """Check if circuit has mid-circuit measurement."""
    if not hasattr(circuit, 'data'):
        return False

    # Find all measurements and their positions
    measurements = []
    total_instructions = len(circuit.data)

    for i, instruction in enumerate(circuit.data):
        gate_name = _get_gate_name(instruction)
        if gate_name == 'measure':
            measurements.append(i)

    # If measurement is not at the end, it's mid-circuit
    for pos in measurements:
        # Check if there are non-measurement gates after this measurement
        for j in range(pos + 1, total_instructions):
            gate_name = _get_gate_name(circuit.data[j])
            if gate_name and gate_name != 'measure':
                return True

    return False


def _count_conditional_operations(circuit: Any) -> int:
    """Count number of conditional operations."""
    count = 0

    if not hasattr(circuit, 'data'):
        return 0

    for instruction in circuit.data:
        if _is_conditional(instruction):
            count += 1

    return count


def _estimate_compilation_overhead(
    circuit: Any,
    backend: Any,
    transpiled: bool,
) -> float:
    """Estimate compilation overhead factor c.

    c = (compiled controlled-U depth) / (bare U depth)
    For native CNOT: c = 1
    With SWAP routing: c > 1
    """
    # If already transpiled and we have backend info, use actual depth
    if transpiled:
        # Simplified: assume c=1 for transpiled circuits
        # Real implementation would compare logical vs physical depth
        return 1.0

    # Default: estimate based on connectivity
    # This is a placeholder; real implementation needs backend topology
    if backend is not None:
        # Estimate SWAP overhead based on qubit connectivity
        # For linear connectivity: c ~ O(L) where L is distance
        # For all-to-all: c ~ 1
        return 1.0  # Placeholder

    # Conservative default
    return 1.0


def _get_bare_depth(circuit: Any, n_conditional: int) -> int:
    """Get bare unitary depth (excluding measurement overhead).

    For the decision boundary, d is the depth of the bare unitary U
    before controlled compilation.
    """
    # Simplified: use circuit depth as approximation
    # Real implementation would separate unitary and non-unitary parts
    depth = _get_depth(circuit)

    # If we have conditional ops, the bare depth is roughly depth/n_conditional
    # This is a very rough approximation
    if n_conditional > 0:
        return max(1, depth // max(1, n_conditional))

    return depth
