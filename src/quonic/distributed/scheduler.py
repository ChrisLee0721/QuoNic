"""Quantum task scheduler for distributed networks.

Schedules quantum circuits across network nodes, considering qubit allocation,
communication costs, and entanglement availability.

Example::

    from quonic.distributed import QuantumNetwork, schedule_task

    network = QuantumNetwork(n_nodes=3, topology="star")
    schedule = schedule_task(circuit, network)
"""

from __future__ import annotations

from dataclasses import dataclass

from ..ir import Circuit, GateOperation
from .network import QuantumNetwork


@dataclass
class ScheduleStep:
    """A single step in the execution schedule.

    Args:
        node: node name executing this step
        gates: list of gate operations to execute
        sync_after: if True, synchronize with other nodes after this step
    """

    node: str
    gates: list[GateOperation]
    sync_after: bool = False


@dataclass
class TaskSchedule:
    """A complete execution schedule for a distributed circuit.

    Args:
        steps: ordered list of schedule steps
        total_time: estimated total execution time
        entanglement_pairs: number of entanglement pairs needed
    """

    steps: list[ScheduleStep]
    total_time: float = 0.0
    entanglement_pairs: int = 0

    def __repr__(self) -> str:
        return (
            f"TaskSchedule(steps={len(self.steps)}, "
            f"time={self.total_time:.1f}, "
            f"pairs={self.entanglement_pairs})"
        )


def schedule_task(
    circuit: Circuit,
    network: QuantumNetwork,
    qubit_assignment: dict[int, str] | None = None,
) -> TaskSchedule:
    """Schedule a circuit across network nodes.

    Assigns qubits to nodes and schedules gates, inserting entanglement
    operations for cross-node gates.

    Args:
        circuit: the circuit to schedule
        network: the quantum network
        qubit_assignment: mapping of qubit index → node name (auto-assigned if None)

    Returns:
        TaskSchedule with ordered execution steps.
    """
    n = circuit.num_qubits
    nodes = network.nodes

    # Auto-assign qubits to nodes (round-robin)
    if qubit_assignment is None:
        qubit_assignment = {}
        for q in range(n):
            qubit_assignment[q] = nodes[q % len(nodes)].name

    # Group gates by which nodes they involve
    steps: list[ScheduleStep] = []
    current_gates: dict[str, list[GateOperation]] = {node.name: [] for node in nodes}
    entanglement_pairs = 0

    for op in circuit.ops:
        if isinstance(op, GateOperation):
            if len(op.qubits) == 1:
                # Single-qubit gate: assign to the node owning that qubit
                node = qubit_assignment.get(op.qubits[0], nodes[0].name)
                current_gates[node].append(op)
            elif len(op.qubits) == 2:
                # Two-qubit gate: check if both qubits are on the same node
                node_a = qubit_assignment.get(op.qubits[0], nodes[0].name)
                node_b = qubit_assignment.get(op.qubits[1], nodes[0].name)
                if node_a == node_b:
                    # Same node: local gate
                    current_gates[node_a].append(op)
                else:
                    # Cross-node: need entanglement
                    entanglement_pairs += 1
                    # Flush current gates before sync
                    for node_name, gates in current_gates.items():
                        if gates:
                            steps.append(ScheduleStep(node=node_name, gates=gates, sync_after=True))
                            current_gates[node_name] = []
                    # Add the cross-node gate as a step
                    steps.append(ScheduleStep(
                        node=f"{node_a}->{node_b}",
                        gates=[op],
                        sync_after=True,
                    ))

    # Flush remaining gates
    for node_name, gates in current_gates.items():
        if gates:
            steps.append(ScheduleStep(node=node_name, gates=gates))

    # Estimate total time (simplified: 1 unit per step + 10 units per entanglement)
    total_time = len(steps) + entanglement_pairs * 10

    return TaskSchedule(
        steps=steps,
        total_time=total_time,
        entanglement_pairs=entanglement_pairs,
    )
