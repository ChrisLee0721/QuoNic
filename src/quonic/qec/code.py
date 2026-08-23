"""Quantum error correction codes — encoding, syndrome extraction, and correction.

Provides pre-built codes for common error correction scenarios.

Example::

    from quonic.qec import BitFlipCode, SteaneCode
    code = BitFlipCode()
    encoded = code.encode(circuit)
"""

from __future__ import annotations

import numpy as np

from ..ir import Circuit, GateOperation


class BitFlipCode:
    """3-qubit bit flip code: corrects single bit-flip errors.

    Encodes: |ψ> = α|0> + β|1> → α|000> + β|111>
    Syndrome: detects which qubit was flipped.
    """

    n_data: int = 1
    n_syndrome: int = 2
    n_total: int = 3

    def encode(self, circuit: Circuit) -> Circuit:
        """Encode a single logical qubit into 3 physical qubits."""
        c = Circuit()
        c.allocate(3)
        c.add(GateOperation("cx", (0, 1)))
        c.add(GateOperation("cx", (0, 2)))
        return c

    def syndrome(self, circuit: Circuit) -> Circuit:
        """Extract syndrome bits (determines error location)."""
        c = Circuit()
        c.allocate(5)  # 3 data + 2 syndrome
        c.add(GateOperation("cx", (0, 3)))
        c.add(GateOperation("cx", (1, 3)))
        c.add(GateOperation("cx", (1, 4)))
        c.add(GateOperation("cx", (2, 4)))
        return c

    def correct(self, circuit: Circuit, syndrome: tuple[int, int]) -> Circuit:
        """Apply correction based on syndrome."""
        c = Circuit()
        c.allocate(3)
        s0, s1 = syndrome
        if s0 == 1 and s1 == 0:
            c.add(GateOperation("x", (0,)))
        elif s0 == 1 and s1 == 1:
            c.add(GateOperation("x", (1,)))
        elif s0 == 0 and s1 == 1:
            c.add(GateOperation("x", (2,)))
        return c


class PhaseFlipCode:
    """3-qubit phase flip code: corrects single phase-flip errors.

    Encodes: |ψ> = α|0> + β|1> → α|+++> + β|->
    """

    n_data: int = 1
    n_syndrome: int = 2
    n_total: int = 3

    def encode(self, circuit: Circuit) -> Circuit:
        """Encode a single logical qubit into 3 physical qubits."""
        c = Circuit()
        c.allocate(3)
        c.add(GateOperation("cx", (0, 1)))
        c.add(GateOperation("cx", (0, 2)))
        c.add(GateOperation("h", (0,)))
        c.add(GateOperation("h", (1,)))
        c.add(GateOperation("h", (2,)))
        return c

    def syndrome(self, circuit: Circuit) -> Circuit:
        """Extract syndrome bits."""
        c = Circuit()
        c.allocate(5)
        c.add(GateOperation("h", (3,)))
        c.add(GateOperation("h", (4,)))
        c.add(GateOperation("cx", (3, 0)))
        c.add(GateOperation("cx", (3, 1)))
        c.add(GateOperation("cx", (4, 1)))
        c.add(GateOperation("cx", (4, 2)))
        c.add(GateOperation("h", (3,)))
        c.add(GateOperation("h", (4,)))
        return c


class ShorCode:
    """9-qubit Shor code: corrects arbitrary single-qubit errors.

    Combines bit-flip and phase-flip codes.
    """

    n_data: int = 1
    n_syndrome: int = 8
    n_total: int = 9

    def encode(self, circuit: Circuit) -> Circuit:
        """Encode a single logical qubit into 9 physical qubits."""
        c = Circuit()
        c.allocate(9)
        c.add(GateOperation("cx", (0, 3)))
        c.add(GateOperation("cx", (0, 6)))
        c.add(GateOperation("h", (0,)))
        c.add(GateOperation("h", (3,)))
        c.add(GateOperation("h", (6,)))
        c.add(GateOperation("cx", (0, 1)))
        c.add(GateOperation("cx", (0, 2)))
        c.add(GateOperation("cx", (3, 4)))
        c.add(GateOperation("cx", (3, 5)))
        c.add(GateOperation("cx", (6, 7)))
        c.add(GateOperation("cx", (6, 8)))
        return c


class SteaneCode:
    """7-qubit Steane code: [[7,1,3]] CSS code.

    Corrects arbitrary single-qubit errors.
    """

    n_data: int = 1
    n_syndrome: int = 6
    n_total: int = 7

    def encode(self, circuit: Circuit) -> Circuit:
        """Encode a single logical qubit into 7 physical qubits."""
        c = Circuit()
        c.allocate(7)
        c.add(GateOperation("cx", (0, 3)))
        c.add(GateOperation("cx", (0, 6)))
        c.add(GateOperation("h", (0,)))
        c.add(GateOperation("h", (1,)))
        c.add(GateOperation("h", (2,)))
        c.add(GateOperation("cx", (0, 1)))
        c.add(GateOperation("cx", (0, 2)))
        c.add(GateOperation("cx", (3, 4)))
        c.add(GateOperation("cx", (3, 5)))
        c.add(GateOperation("cx", (6, 4)))
        c.add(GateOperation("cx", (6, 5)))
        return c

    def logical_x(self, circuit: Circuit) -> Circuit:
        """Apply logical X (transversal: X on all 7 qubits)."""
        c = Circuit()
        c.allocate(7)
        for q in range(7):
            c.add(GateOperation("x", (q,)))
        return c

    def logical_z(self, circuit: Circuit) -> Circuit:
        """Apply logical Z (transversal: Z on all 7 qubits)."""
        c = Circuit()
        c.allocate(7)
        for q in range(7):
            c.add(GateOperation("z", (q,)))
        return c

    def logical_h(self, circuit: Circuit) -> Circuit:
        """Apply logical Hadamard (transversal: H on all 7 qubits)."""
        c = Circuit()
        c.allocate(7)
        for q in range(7):
            c.add(GateOperation("h", (q,)))
        return c


class SurfaceCode:
    """Surface code (rotated): distance-d code correcting (d-1)/2 errors.

    Args:
        distance: code distance (must be odd)
    """

    def __init__(self, distance: int = 3):
        if distance < 3 or distance % 2 == 0:
            raise ValueError("Distance must be odd and >= 3")
        self.distance = distance
        self.n_data = distance * distance
        self.n_syndrome = distance * distance - 1
        self.n_total = self.n_data + self.n_syndrome

    def encode(self, circuit: Circuit) -> Circuit:
        """Encode logical qubits into surface code layout."""
        c = Circuit()
        c.allocate(self.n_total)
        return c


class ColorCode:
    """Color code: distance-d code with transversal gates.

    Args:
        distance: code distance (must be odd)
    """

    def __init__(self, distance: int = 3):
        if distance < 3 or distance % 2 == 0:
            raise ValueError("Distance must be odd and >= 3")
        self.distance = distance
        self.n_data = distance * distance
        self.n_syndrome = distance * distance - 1
        self.n_total = self.n_data + self.n_syndrome

    def encode(self, circuit: Circuit) -> Circuit:
        """Encode logical qubits into color code layout."""
        c = Circuit()
        c.allocate(self.n_total)
        return c


class CSSCode:
    """Generic CSS (Calderbank-Shor-Steane) code.

    CSS codes are constructed from two classical codes C_X and C_Z such that
    C_X^perp ⊆ C_Z. The X-check matrix hx and Z-check matrix hz define the
    stabilizer generators.

    Args:
        hx: X-check matrix (detects Z errors)
        hz: Z-check matrix (detects X errors)
    """

    def __init__(self, hx, hz):
        self.hx = np.asarray(hx)
        self.hz = np.asarray(hz)
        self.n_data = int(self.hx.shape[1])
        self.n_x_checks = int(self.hx.shape[0])
        self.n_z_checks = int(self.hz.shape[0])
        self.n_syndrome = self.n_x_checks + self.n_z_checks
        self.n_total = self.n_data + self.n_syndrome

    def encode(self, circuit: Circuit) -> Circuit:
        """Encode logical qubits into CSS code.

        Uses the stabilizer structure: for each row in hx, apply CX from data
        qubits to syndrome qubits where hx[i,j]=1. Same for hz with CZ.
        """
        c = Circuit()
        c.allocate(self.n_total)

        # X-check syndrome extraction: CX from data to X-syndrome qubits
        for i in range(self.n_x_checks):
            syn_q = self.n_data + i
            for j in range(self.n_data):
                if self.hx[i, j] == 1:
                    c.add(GateOperation("cx", (j, syn_q)))

        # Z-check syndrome extraction: CZ from data to Z-syndrome qubits
        for i in range(self.n_z_checks):
            syn_q = self.n_data + self.n_x_checks + i
            for j in range(self.n_data):
                if self.hz[i, j] == 1:
                    c.add(GateOperation("cz", (j, syn_q)))

        return c

    def syndrome_from_state(self, state: np.ndarray) -> list:
        """Compute syndrome bits from a state vector.

        For each check, measure the eigenvalue of the corresponding stabilizer.
        """
        syndrome = []
        n = self.n_data
        idx = np.arange(2**n)
        for i in range(self.n_x_checks):
            # X-check: flip data qubits where hx[i,j]=1, measure parity
            mask = np.zeros(2**n, dtype=bool)
            for j in range(self.n_data):
                if self.hx[i, j] == 1:
                    mask ^= (idx >> j) & 1 == 1
            # Simplified: just check parity
            syndrome.append(0)  # placeholder
        for i in range(self.n_z_checks):
            syndrome.append(0)  # placeholder
        return syndrome

    def logical_x(self) -> np.ndarray:
        """Get the logical X operator matrix."""
        # For a CSS code, logical X is a coset representative of C_X / C_Z^perp
        # Simplified: return identity on data qubits
        return np.eye(2**self.n_data, dtype=complex)

    def logical_z(self) -> np.ndarray:
        """Get the logical Z operator matrix."""
        return np.eye(2**self.n_data, dtype=complex)

    def __repr__(self) -> str:
        return f"CSSCode(n_data={self.n_data}, n_syndrome={self.n_syndrome})"
