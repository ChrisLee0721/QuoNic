"""Variational ansatz library — parameterized quantum circuits for QML.

Provides pre-built ansatz structures for variational quantum algorithms.

Example::

    from quonic.ml import Ansatz
    ansatz = Ansatz.hardware_efficient(n_qubits=4, layers=3)
    circuit = ansatz.build(params)
"""

from __future__ import annotations

import numpy as np

from ..ir import Circuit, GateOperation


class Ansatz:
    """Variational ansatz builder."""

    @staticmethod
    def hardware_efficient(
        n_qubits: int,
        layers: int = 1,
        entanglement: str = "linear",
    ) -> AnsatzBuilder:
        """Hardware-efficient ansatz: Ry rotations + entangling CX ladder.

        Args:
            n_qubits: number of qubits
            layers: number of rotation + entanglement layers
            entanglement: "linear" (nearest-neighbor CX) or "full" (all-pairs CX)

        Returns:
            AnsatzBuilder with build(params) method.
        """
        return _HardwareEfficient(n_qubits, layers, entanglement)

    @staticmethod
    def qaoa(n_qubits: int, p: int = 1) -> AnsatzBuilder:
        """QAOA ansatz: alternating mixer and problem unitaries.

        Args:
            n_qubits: number of qubits
            p: number of QAOA layers

        Returns:
            AnsatzBuilder with build(params) method.
        """
        return _QAOA(n_qubits, p)

    @staticmethod
    def uccsd(n_qubits: int) -> AnsatzBuilder:
        """UCCSD ansatz (simplified): singles + doubles excitations.

        Args:
            n_qubits: number of qubits

        Returns:
            AnsatzBuilder with build(params) method.
        """
        return _UCCSD(n_qubits)

    @staticmethod
    def strongly_entangling(
        n_qubits: int,
        layers: int = 1,
    ) -> AnsatzBuilder:
        """Strongly entangling ansatz: Ry, Rz rotations + all-pairs CX.

        Each layer applies Ry and Rz rotations, then CX between all qubit pairs.
        More expressive than hardware-efficient for entangled states.

        Args:
            n_qubits: number of qubits
            layers: number of layers

        Returns:
            AnsatzBuilder with build(params) method.
        """
        return _StronglyEntangling(n_qubits, layers)

    @staticmethod
    def random(
        n_qubits: int,
        depth: int = 3,
        seed: int = 42,
    ) -> AnsatzBuilder:
        """Random ansatz: random single-qubit gates + random entangling gates.

        Useful for benchmarking and exploring quantum advantage.

        Args:
            n_qubits: number of qubits
            depth: circuit depth
            seed: random seed for gate selection

        Returns:
            AnsatzBuilder with build(params) method.
        """
        return _Random(n_qubits, depth, seed)

    @staticmethod
    def data_reuploading(
        n_qubits: int,
        layers: int = 3,
    ) -> AnsatzBuilder:
        """Data re-uploading ansatz: re-encodes data at each layer.

        Each layer applies data-encoding rotations followed by entangling gates.
        Achieves universal approximation with enough layers.

        Args:
            n_qubits: number of qubits
            layers: number of re-uploading layers

        Returns:
            AnsatzBuilder with build(params) method.
        """
        return _DataReuploading(n_qubits, layers)

    @staticmethod
    def circuit_centric(
        n_qubits: int,
        layers: int = 2,
    ) -> AnsatzBuilder:
        """Circuit-centric ansatz: fixed entangling structure + trainable rotations.

        Uses a fixed pattern of entangling gates with trainable single-qubit rotations.

        Args:
            n_qubits: number of qubits
            layers: number of layers

        Returns:
            AnsatzBuilder with build(params) method.
        """
        return _CircuitCentric(n_qubits, layers)


class AnsatzBuilder:
    """Base class for ansatz builders."""

    n_params: int

    def build(self, params: list[float]) -> Circuit:
        """Build a circuit from parameters."""
        raise NotImplementedError


class _HardwareEfficient(AnsatzBuilder):
    """Hardware-efficient ansatz: Ry rotations + CX ladder."""

    def __init__(self, n_qubits: int, layers: int, entanglement: str):
        self.n_qubits = n_qubits
        self.layers = layers
        self.entanglement = entanglement
        # Each layer: n Ry rotations + (n-1) CX gates
        self.n_params = n_qubits * layers

    def build(self, params: list[float]) -> Circuit:
        c = Circuit()
        c.allocate(self.n_qubits)
        idx = 0
        for layer in range(self.layers):
            # Rotation layer
            for q in range(self.n_qubits):
                c.add(GateOperation("ry", (q,), (params[idx],)))
                idx += 1
            # Entanglement layer
            if self.entanglement == "linear":
                for q in range(self.n_qubits - 1):
                    c.add(GateOperation("cx", (q, q + 1)))
            elif self.entanglement == "full":
                for i in range(self.n_qubits):
                    for j in range(i + 1, self.n_qubits):
                        c.add(GateOperation("cx", (i, j)))
        return c


class _QAOA(AnsatzBuilder):
    """QAOA ansatz: alternating mixer and problem unitaries."""

    def __init__(self, n_qubits: int, p: int):
        self.n_qubits = n_qubits
        self.p = p
        # Each layer: n Rx rotations (mixer) + n CX + n Rz (problem)
        self.n_params = 2 * n_qubits * p

    def build(self, params: list[float]) -> Circuit:
        c = Circuit()
        c.allocate(self.n_qubits)
        # Initial superposition
        for q in range(self.n_qubits):
            c.add(GateOperation("h", (q,)))
        idx = 0
        for layer in range(self.p):
            # Problem unitary: ZZ interactions
            for q in range(self.n_qubits - 1):
                c.add(GateOperation("cx", (q, q + 1)))
                c.add(GateOperation("rz", (q + 1,), (params[idx],)))
                idx += 1
                c.add(GateOperation("cx", (q, q + 1)))
            # Mixer unitary: Rx rotations
            for q in range(self.n_qubits):
                c.add(GateOperation("rx", (q,), (params[idx],)))
                idx += 1
        return c


class _UCCSD(AnsatzBuilder):
    """Simplified UCCSD ansatz: singles + doubles excitations."""

    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        # Singles: n_qubits, Doubles: n_qubits*(n_qubits-1)/2
        self.n_params = n_qubits + n_qubits * (n_qubits - 1) // 2

    def build(self, params: list[float]) -> Circuit:
        c = Circuit()
        c.allocate(self.n_qubits)
        idx = 0
        # Singles excitations
        for q in range(self.n_qubits):
            c.add(GateOperation("ry", (q,), (params[idx],)))
            idx += 1
        # Doubles excitations
        for i in range(self.n_qubits):
            for j in range(i + 1, self.n_qubits):
                c.add(GateOperation("cx", (i, j)))
                c.add(GateOperation("ry", (j,), (params[idx],)))
                idx += 1
                c.add(GateOperation("cx", (i, j)))
        return c


class _StronglyEntangling(AnsatzBuilder):
    """Strongly entangling ansatz: Ry, Rz rotations + all-pairs CX."""

    def __init__(self, n_qubits: int, layers: int):
        self.n_qubits = n_qubits
        self.layers = layers
        # Each layer: n Ry + n Rz + n*(n-1)/2 CX
        self.n_params = 2 * n_qubits * layers

    def build(self, params: list[float]) -> Circuit:
        c = Circuit()
        c.allocate(self.n_qubits)
        idx = 0
        for layer in range(self.layers):
            # Rotation layer
            for q in range(self.n_qubits):
                c.add(GateOperation("ry", (q,), (params[idx],)))
                idx += 1
                c.add(GateOperation("rz", (q,), (params[idx],)))
                idx += 1
            # Entanglement layer (all pairs)
            for i in range(self.n_qubits):
                for j in range(i + 1, self.n_qubits):
                    c.add(GateOperation("cx", (i, j)))
        return c


class _Random(AnsatzBuilder):
    """Random ansatz: random single-qubit gates + random entangling gates."""

    def __init__(self, n_qubits: int, depth: int, seed: int):
        self.n_qubits = n_qubits
        self.depth = depth
        self.seed = seed
        # Each depth layer: n rotations + some entangling gates
        self.n_params = n_qubits * depth

    def build(self, params: list[float]) -> Circuit:
        rng = np.random.RandomState(self.seed)
        c = Circuit()
        c.allocate(self.n_qubits)
        idx = 0
        for d in range(self.depth):
            # Random rotation layer
            for q in range(self.n_qubits):
                gate = rng.choice(["ry", "rz", "rx"])
                c.add(GateOperation(gate, (q,), (params[idx],)))
                idx += 1
            # Random entangling layer
            for i in range(self.n_qubits - 1):
                if rng.random() > 0.5:
                    c.add(GateOperation("cx", (i, i + 1)))
        return c


class _DataReuploading(AnsatzBuilder):
    """Data re-uploading ansatz: re-encodes data at each layer."""

    def __init__(self, n_qubits: int, layers: int):
        self.n_qubits = n_qubits
        self.layers = layers
        # Each layer: n data params + n trainable params + entangling
        self.n_params = 2 * n_qubits * layers

    def build(self, params: list[float]) -> Circuit:
        c = Circuit()
        c.allocate(self.n_qubits)
        idx = 0
        for layer in range(self.layers):
            # Data encoding + trainable rotations
            for q in range(self.n_qubits):
                c.add(GateOperation("ry", (q,), (params[idx],)))  # data
                idx += 1
                c.add(GateOperation("rz", (q,), (params[idx],)))  # trainable
                idx += 1
            # Entangling
            for q in range(self.n_qubits - 1):
                c.add(GateOperation("cx", (q, q + 1)))
        return c


class _CircuitCentric(AnsatzBuilder):
    """Circuit-centric ansatz: fixed entangling structure + trainable rotations."""

    def __init__(self, n_qubits: int, layers: int):
        self.n_qubits = n_qubits
        self.layers = layers
        self.n_params = n_qubits * layers

    def build(self, params: list[float]) -> Circuit:
        c = Circuit()
        c.allocate(self.n_qubits)
        idx = 0
        for layer in range(self.layers):
            # Fixed entangling structure (circular)
            for q in range(self.n_qubits):
                c.add(GateOperation("cx", (q, (q + 1) % self.n_qubits)))
            # Trainable rotations
            for q in range(self.n_qubits):
                c.add(GateOperation("ry", (q,), (params[idx],)))
                idx += 1
        return c
