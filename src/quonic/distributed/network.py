"""Quantum network topology and node management.

Example::

    from quonic.distributed import QuantumNetwork, Node

    network = QuantumNetwork(n_nodes=3)
    node0 = network.nodes[0]
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Node:
    """A node in a quantum network.

    Args:
        name: node name
        n_qubits: number of qubits at this node
        backend: backend to use for this node
    """

    name: str
    n_qubits: int
    backend: str = "native"


@dataclass
class QuantumNetwork:
    """A quantum network with multiple nodes.

    Args:
        n_nodes: number of nodes
        topology: network topology ("star", "ring", "linear")
    """

    n_nodes: int
    topology: str = "star"
    nodes: list[Node] = field(default_factory=list)
    connections: dict[str, list[str]] = field(default_factory=dict)

    def __post_init__(self):
        if not self.nodes:
            self.nodes = [Node(f"node_{i}", 4) for i in range(self.n_nodes)]
        if not self.connections:
            self._build_topology()

    def _build_topology(self):
        """Build network topology."""
        if self.topology == "star":
            hub = self.nodes[0].name
            self.connections[hub] = [n.name for n in self.nodes[1:]]
            for n in self.nodes[1:]:
                self.connections[n.name] = [hub]
        elif self.topology == "ring":
            for i, node in enumerate(self.nodes):
                next_node = self.nodes[(i + 1) % self.n_nodes]
                self.connections[node.name] = [next_node.name]
        elif self.topology == "linear":
            for i, node in enumerate(self.nodes):
                neighbors = []
                if i > 0:
                    neighbors.append(self.nodes[i - 1].name)
                if i < self.n_nodes - 1:
                    neighbors.append(self.nodes[i + 1].name)
                self.connections[node.name] = neighbors

    def get_neighbors(self, node_name: str) -> list[str]:
        """Get neighbors of a node."""
        return self.connections.get(node_name, [])
