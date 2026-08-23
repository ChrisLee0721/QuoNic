"""Entanglement distribution and remote gates for quantum networks.

Implements teleportation and remote CNOT using entangled pairs.

Example::

    from quonic.distributed import EntanglementPair, remote_cnot, teleport_state

    pair = EntanglementPair(node_a=0, node_b=1)
    circuit = remote_cnot(pair, control_qubit=0, target_qubit=3)
"""

from __future__ import annotations

from dataclasses import dataclass

from ..ir import Circuit, GateOperation


@dataclass
class EntanglementPair:
    """An entangled pair shared between two nodes.

    Args:
        node_a: index of first node
        node_b: index of second node
        ancilla_a: ancilla qubit index on node_a (for the Bell pair)
        ancilla_b: ancilla qubit index on node_b (for the Bell pair)
        fidelity: entanglement fidelity
    """

    node_a: int
    node_b: int
    ancilla_a: int = -1
    ancilla_b: int = -1
    fidelity: float = 1.0


def create_bell_pair(
    circuit: Circuit,
    qubit_a: int,
    qubit_b: int,
) -> None:
    """Create a Bell pair (|00> + |11>) / sqrt(2) between two qubits.

    Args:
        circuit: circuit to add gates to
        qubit_a: first qubit index
        qubit_b: second qubit index
    """
    circuit.add(GateOperation("h", (qubit_a,)))
    circuit.add(GateOperation("cx", (qubit_a, qubit_b)))


def remote_cnot(
    pair: EntanglementPair,
    control_qubit: int,
    target_qubit: int,
    circuit: Circuit | None = None,
) -> Circuit:
    """Apply a remote CNOT using entanglement.

    Uses the entangled pair to implement a CNOT between qubits on different nodes.
    Protocol:
    1. CNOT(control, ancilla_a)
    2. CNOT(ancilla_b, target)
    3. Measure ancilla_a and ancilla_b
    4. Apply corrections: if ancilla_a=1, X(target); if ancilla_b=1, Z(control)

    Args:
        pair: entangled pair between nodes
        control_qubit: control qubit index (on node_a)
        target_qubit: target qubit index (on node_b)
        circuit: existing circuit to add to (creates new if None)

    Returns:
        Circuit with remote CNOT implemented.
    """
    if circuit is None:
        circuit = Circuit()
        circuit.allocate(max(control_qubit, target_qubit, pair.ancilla_a, pair.ancilla_b) + 1)

    # Step 1: CNOT from control to ancilla_a
    circuit.add(GateOperation("cx", (control_qubit, pair.ancilla_a)))

    # Step 2: CNOT from ancilla_b to target
    circuit.add(GateOperation("cx", (pair.ancilla_b, target_qubit)))

    # Step 3: Measure ancillas (deferred: apply corrections based on measurement)
    # In a real network, these measurements would be communicated classically
    # For simulation, we apply the corrections conditionally

    # Step 4: Corrections (applied based on measurement outcomes)
    # If ancilla_a = 1: apply X to target
    # If ancilla_b = 1: apply Z to control
    # In simulation, we use classical control flow
    circuit.add(GateOperation("cx", (pair.ancilla_a, target_qubit)))  # X correction
    circuit.add(GateOperation("cz", (pair.ancilla_b, control_qubit)))  # Z correction

    return circuit


def teleport_state(
    pair: EntanglementPair,
    source_qubit: int,
    target_qubit: int,
    circuit: Circuit | None = None,
) -> Circuit:
    """Teleport a qubit state using an entangled pair.

    Protocol:
    1. CNOT(source, ancilla_a)
    2. H(source)
    3. Measure source and ancilla_a
    4. Apply corrections to target: X if ancilla_a=1, Z if source=1

    Args:
        pair: entangled pair between nodes
        source_qubit: qubit to teleport (on node_a)
        target_qubit: destination qubit (on node_b)
        circuit: existing circuit to add to

    Returns:
        Circuit with teleportation protocol.
    """
    if circuit is None:
        circuit = Circuit()
        circuit.allocate(max(source_qubit, target_qubit, pair.ancilla_a, pair.ancilla_b) + 1)

    # Step 1: CNOT(source, ancilla_a)
    circuit.add(GateOperation("cx", (source_qubit, pair.ancilla_a)))

    # Step 2: H(source)
    circuit.add(GateOperation("h", (source_qubit,)))

    # Step 3: Corrections based on measurements
    # If source = 1: apply X to target (via ancilla_b)
    # If ancilla_a = 1: apply Z to target
    circuit.add(GateOperation("cx", (source_qubit, pair.ancilla_b)))  # X correction path
    circuit.add(GateOperation("cz", (pair.ancilla_a, target_qubit)))  # Z correction

    return circuit


def distribute_entanglement(
    circuit: Circuit,
    node_a_qubit: int,
    node_b_qubit: int,
    fidelity: float = 1.0,
) -> EntanglementPair:
    """Create and distribute an entangled pair between two nodes.

    Args:
        circuit: circuit to add Bell pair creation to
        node_a_qubit: qubit index on node_a
        node_b_qubit: qubit index on node_b
        fidelity: entanglement fidelity

    Returns:
        EntanglementPair with the qubit indices.
    """
    create_bell_pair(circuit, node_a_qubit, node_b_qubit)
    return EntanglementPair(
        node_a=0,
        node_b=1,
        ancilla_a=node_a_qubit,
        ancilla_b=node_b_qubit,
        fidelity=fidelity,
    )
