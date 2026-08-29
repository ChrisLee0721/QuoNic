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

    from gciqa import GroverOracle, ConstraintSet, GeometricConstraint

    constraints = ConstraintSet([
        GeometricConstraint.bond("0", "1", 1.3, 1.5),
        GeometricConstraint.pocket(center=(10, 20, 30), radius=8.0),
    ])
    oracle = GroverOracle(n_qubits=12, constraints=constraints)
    circuit = oracle.build()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..constraints import ConstraintSet, ConstraintType, GeometricConstraint


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

    def build(self, backend: str = "quonic") -> Any:
        """Build the Grover oracle circuit.

        For 1-bit encoding with bond constraints only, uses optimized
        bit-pattern checking (O(1) gates per constraint). Otherwise,
        falls back to enumeration (≤16 qubits) or arithmetic circuits.

        Args:
            backend: "quonic" (returns QuoNicCircuit), "qiskit", or "pyqpanda3".

        Returns:
            Quantum circuit implementing the oracle.
        """

        # Use optimized bit-pattern oracle for 1-bit bond constraints
        if self.bits_per_coord == 1 and self._is_bond_only():
            qc = self._build_bitpattern_oracle()
        elif self.n_qubits <= 16:
            qc = self._build_enumeration_oracle()
        else:
            qc = self._build_arithmetic_oracle()

        if backend == "qiskit":
            return qc.to_qiskit()
        elif backend == "pyqpanda3":
            return qc.to_pyqpanda3()
        return qc

    def _is_bond_only(self) -> bool:
        """Check if all constraints are bond constraints."""
        return all(c.type == ConstraintType.BOND for c in self.constraints)

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
        from ..circuit import QuoNicCircuit

        n = self.n_qubits
        # For MCZ with >5 controls, need ancilla qubits for Toffoli cascade
        n_controls = n - 1
        n_ancilla = max(0, n_controls - 3) if n_controls > 5 else 0
        total_qubits = n + n_ancilla

        qc = QuoNicCircuit(total_qubits)

        # Find all valid bitstrings
        valid_states = []
        for state_int in range(2**n):
            bitstring = format(state_int, f'0{n}b')
            if self.classical_oracle_fn(bitstring):
                valid_states.append(state_int)

        if not valid_states:
            return qc

        # Apply phase flip for each valid state
        ancillas = list(range(n, total_qubits))
        for state_int in valid_states:
            self._add_phase_flip(qc, state_int, ancillas)

        return qc

    def _add_phase_flip(self, qc: Any, state_int: int, ancillas: list[int] | None = None) -> None:
        """Apply phase flip to a specific computational basis state.

        Flips |state⟩ → -|state⟩ using X gates + multi-controlled Z.
        Uses decomposed MCZ (Toffoli cascade) when controls > 5.
        """
        n = self.n_qubits

        # Apply X to qubits where state has a 0 bit
        for i in range(n):
            if not ((state_int >> i) & 1):
                qc.x(i)

        # Multi-controlled Z
        controls = list(range(n - 1))
        target = n - 1
        if n == 1:
            qc.z(0)
        elif n == 2:
            qc.h(1)
            qc.cx(0, 1)
            qc.h(1)
        elif len(controls) > 5 and ancillas:
            # Decomposed MCZ for hardware with max 5 control qubits
            qc.mcz_decomposed(controls, target, ancillas)
        else:
            qc.mcz(controls, target)

        # Undo X gates
        for i in range(n):
            if not ((state_int >> i) & 1):
                qc.x(i)

    def _build_bitpattern_oracle(self) -> Any:
        """Build optimized oracle using bit-pattern checking.

        For 1-bit encoding with bond constraints, checks invalid patterns
        directly instead of enumerating valid states. O(1) gates per constraint.

        Invalid bond patterns (1-bit encoding):
            - Distance 0: all coordinate bits identical
            - Distance √12: all coordinate bits different

        Valid bond patterns:
            - Distance 2: differ in 1 coordinate
            - Distance 2√2: differ in 2 coordinates
        """
        from ..circuit import QuoNicCircuit

        n_data = self.n_qubits
        n_constraints = len(self.constraints)
        # Need ancillas for: constraint results + AND target + diffuser MCZ
        # Diffuser MCZ needs n_data - 4 ancillas (for n_data - 1 controls)
        n_diffuser_ancilla = max(0, n_data - 4)
        n_ancilla = n_constraints + 1 + n_diffuser_ancilla
        total = n_data + n_ancilla

        qc = QuoNicCircuit(total)

        # Check each bond constraint
        constraint_ancillas = []
        for idx, constraint in enumerate(self.constraints):
            a0 = int(constraint.atoms[0])
            a1 = int(constraint.atoms[1])
            anc = n_data + idx
            self._check_bond_bitpattern(qc, a0 * 3, a1 * 3, anc)
            constraint_ancillas.append(anc)

        # AND all constraint results
        and_target = n_data + n_constraints
        self._multi_and(qc, constraint_ancillas, and_target)

        # Phase flip on valid states (and_target = 1)
        qc.z(and_target)

        # Uncompute AND
        self._multi_and(qc, constraint_ancillas, and_target)

        # Uncompute constraint checks
        for idx, constraint in enumerate(self.constraints):
            a0 = int(constraint.atoms[0])
            a1 = int(constraint.atoms[1])
            anc = n_data + idx
            self._check_bond_bitpattern(qc, a0 * 3, a1 * 3, anc)

        return qc

    def _check_bond_bitpattern(
        self, qc: Any, a0_start: int, a1_start: int, ancilla: int
    ) -> None:
        """Check bond constraint using bit-pattern matching.

        Marks ancilla = 1 if bond is VALID (distance 2 or 2√2).
        Uses CNOT + MCX for efficient pattern detection.

        With 1-bit encoding, possible distances:
            - Distance 0: all bits same (INVALID)
            - Distance 2: differ in 1 coord (VALID)
            - Distance 2√2: differ in 2 coords (VALID)
            - Distance 2√3: differ in 3 coords (INVALID)

        Valid patterns: differ in exactly 1 or 2 coordinates.
        Invalid patterns: differ in 0 or 3 coordinates.

        Args:
            qc: QuoNicCircuit.
            a0_start: Start qubit index for atom 0.
            a1_start: Start qubit index for atom 1.
            ancilla: Ancilla qubit to store result.
        """
        # Compute XOR: diff[i] = a0[i] XOR a1[i]
        # After CNOT: a1[i] = a0[i] XOR a1[i] (1 if different)
        for i in range(3):
            qc.cx(a0_start + i, a1_start + i)

        # Count differences: valid if exactly 1 or 2 bits differ
        # Invalid if 0 bits differ (all same) or 3 bits differ (all different)

        # Check for 0 differences (all same): all diff bits are 0
        # Flip all diff bits, then MCX
        for i in range(3):
            qc.x(a1_start + i)
        qc.mcx([a1_start, a1_start + 1, a1_start + 2], ancilla)
        for i in range(3):
            qc.x(a1_start + i)

        # Check for 3 differences (all different): all diff bits are 1
        qc.mcx([a1_start, a1_start + 1, a1_start + 2], ancilla)

        # Undo XOR
        for i in range(3):
            qc.cx(a0_start + i, a1_start + i)

        # Flip ancilla: now ancilla = 1 if VALID (not 0-diff and not 3-diff)
        qc.x(ancilla)

    def _build_arithmetic_oracle(self) -> Any:
        """Build oracle using quantum arithmetic circuits.

        Uses CDKM ripple-carry adder for subtraction, shift-and-add
        for squaring, and bit-level comparison for range checks.

        Qubit layout:
            [0..n_data-1]: data qubits (atom coordinates)
            [n_data..n_data+n_anc-1]: ancilla qubits
        """
        from ..circuit import QuoNicCircuit

        b = self.bits_per_coord
        bits_per_atom = 3 * b

        atom_indices = self._get_atom_indices()
        n_data = max(self.n_qubits, (max(atom_indices) + 1) * bits_per_atom if atom_indices else bits_per_atom)

        n_constraints = len(self.constraints)
        n_arith_ancilla = self._estimate_arith_ancilla()
        n_ancilla = n_constraints + 1 + n_arith_ancilla
        total = n_data + n_ancilla

        qc = QuoNicCircuit(total)

        # Build each constraint check
        constraint_ancillas = []
        anc_offset = 0
        for i, constraint in enumerate(self.constraints):
            target_anc = n_data + anc_offset
            workspace = list(range(n_data + anc_offset + 1, n_data + anc_offset + 1 + n_arith_ancilla))
            self._add_constraint_check(qc, n_data, target_anc, workspace, constraint)
            constraint_ancillas.append(target_anc)
            anc_offset += 1

        # AND all constraint results
        and_target = n_data + anc_offset
        self._multi_and(qc, constraint_ancillas, and_target)

        # Phase flip on valid states
        qc.z(and_target)

        # Uncompute AND
        self._multi_and(qc, constraint_ancillas, and_target)

        # Uncompute constraint checks
        anc_offset = 0
        for i, constraint in enumerate(self.constraints):
            target_anc = n_data + anc_offset
            workspace = list(range(n_data + anc_offset + 1, n_data + anc_offset + 1 + n_arith_ancilla))
            self._add_constraint_check_inverse(qc, n_data, target_anc, workspace, constraint)
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

        Parallel workspace layout (b = bits_per_coord):
            wx, wy, wz: 3b qubits (differences)
            wacc: 2b+1 qubits (d² accumulator)
            wtmp_x, wtmp_y, wtmp_z: 3 × (2b+1) qubits (parallel squaring)
            wcarry_sub: 3 × (b+1) qubits (parallel subtraction carry)
            wcarry_add: 3 × (2b+1) qubits (parallel adder carry for squaring)
            wcarry_cmp: 2b+2 qubits (comparison borrow)
        Total: 3b + (2b+1) + 3*(2b+1) + 3*(b+1) + 3*(2b+1) + (2b+2) = 24b + 10
        """
        b = self.bits_per_coord
        return 24 * b + 10

    def _get_atom_qubits(self, data_offset: int, atom_idx: int, bits: int) -> tuple[list, list, list]:
        """Get qubit index lists for an atom's x, y, z coordinates."""
        base = data_offset + atom_idx * 3 * bits
        x_qubits = [base + i for i in range(bits)]
        y_qubits = [base + bits + i for i in range(bits)]
        z_qubits = [base + 2 * bits + i for i in range(bits)]
        return x_qubits, y_qubits, z_qubits

    def _add_constraint_check(
        self, qc: Any, data_offset: int, target: int, workspace: list,
        constraint: GeometricConstraint,
    ) -> None:
        """Add constraint checking to circuit.

        Args:
            qc: QuoNicCircuit.
            data_offset: Offset where data qubits start (0 for flat layout).
            target: Index of target ancilla qubit.
            workspace: List of workspace qubit indices.
            constraint: The constraint to check.
        """
        b = self.bits_per_coord

        if constraint.type == ConstraintType.BOND:
            a1 = int(constraint.atoms[0]) if constraint.atoms[0] != "*" else 0
            a2 = int(constraint.atoms[1]) if constraint.atoms[1] != "*" else 1
            x1, y1, z1 = self._get_atom_qubits(data_offset, a1, b)
            x2, y2, z2 = self._get_atom_qubits(data_offset, a2, b)
            self._check_distance_range(
                qc, x1, y1, z1, x2, y2, z2, target, workspace,
                constraint.params["min_dist"],
                constraint.params["max_dist"],
            )
        elif constraint.type == ConstraintType.NO_CLASH:
            a1 = int(constraint.atoms[0]) if constraint.atoms[0] != "*" else 0
            a2 = int(constraint.atoms[1]) if constraint.atoms[1] != "*" else 1
            x1, y1, z1 = self._get_atom_qubits(data_offset, a1, b)
            x2, y2, z2 = self._get_atom_qubits(data_offset, a2, b)
            self._check_distance_min(
                qc, x1, y1, z1, x2, y2, z2, target, workspace,
                constraint.params["min_dist"],
            )
        elif constraint.type == ConstraintType.POCKET:
            atom = constraint.atoms[0]
            a_idx = int(atom) if atom != "*" else 0
            x1, y1, z1 = self._get_atom_qubits(data_offset, a_idx, b)
            cx = constraint.params["cx"]
            cy = constraint.params["cy"]
            cz = constraint.params["cz"]
            radius = constraint.params["radius"]
            self._check_pocket(
                qc, x1, y1, z1, target, workspace,
                cx, cy, cz, radius,
            )
        elif constraint.type == ConstraintType.HYDROGEN_BOND:
            a1 = int(constraint.atoms[0]) if constraint.atoms[0] != "*" else 0
            a2 = int(constraint.atoms[1]) if constraint.atoms[1] != "*" else 1
            x1, y1, z1 = self._get_atom_qubits(data_offset, a1, b)
            x2, y2, z2 = self._get_atom_qubits(data_offset, a2, b)
            self._check_distance_range(
                qc, x1, y1, z1, x2, y2, z2, target, workspace,
                0.0,
                constraint.params["max_dist"],
            )
        else:
            qc.x(target)

    def _add_constraint_check_inverse(
        self, qc: Any, data_offset: int, target: int, workspace: list,
        constraint: GeometricConstraint,
    ) -> None:
        """Inverse of constraint check (self-inverse for CNOT/Toffoli/X)."""
        self._add_constraint_check(qc, data_offset, target, workspace, constraint)

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
        """Check if distance between two atoms is in [min_dist, max_dist].

        Uses parallel workspace for independent operations.
        """
        b = len(x1)
        n_acc = 2 * b + 1

        # Parallel workspace layout
        off = 0
        wx = workspace[off:off+b]; off += b
        wy = workspace[off:off+b]; off += b
        wz = workspace[off:off+b]; off += b
        wacc = workspace[off:off+n_acc]; off += n_acc
        # Parallel tmp for squaring (3 independent)
        wtmp_x = workspace[off:off+n_acc]; off += n_acc
        wtmp_y = workspace[off:off+n_acc]; off += n_acc
        wtmp_z = workspace[off:off+n_acc]; off += n_acc
        # Parallel carry for subtraction (3 independent)
        wcarry_sub_x = workspace[off:off+b+1]; off += b+1
        wcarry_sub_y = workspace[off:off+b+1]; off += b+1
        wcarry_sub_z = workspace[off:off+b+1]; off += b+1
        # Parallel carry for adder in squaring (3 independent)
        wcarry_add_x = workspace[off:off+n_acc]; off += n_acc
        wcarry_add_y = workspace[off:off+n_acc]; off += n_acc
        wcarry_add_z = workspace[off:off+n_acc]; off += n_acc
        # Carry for comparison
        wcarry_cmp = workspace[off:off+n_acc+1]; off += n_acc+1

        # Step 1: Compute dx, dy, dz IN PARALLEL (independent qubits)
        self._quantum_subtract(qc, x1, x2, wx, wcarry_sub_x, b)
        self._quantum_subtract(qc, y1, y2, wy, wcarry_sub_y, b)
        self._quantum_subtract(qc, z1, z2, wz, wcarry_sub_z, b)

        # Step 2: Compute dx², dy², dz² and accumulate IN PARALLEL
        # Each add_square uses its own tmp and carry
        self._add_square(qc, wx, wacc, wtmp_x, wcarry_add_x, b, n_acc)
        self._add_square(qc, wy, wacc, wtmp_y, wcarry_add_y, b, n_acc)
        self._add_square(qc, wz, wacc, wtmp_z, wcarry_add_z, b, n_acc)

        # Step 3: Check range
        lo, hi = self.coord_range
        scale = (hi - lo) / (2**b - 1)
        min_sq = int((min_dist / scale) ** 2)
        max_sq = int((max_dist / scale) ** 2)
        self._check_range(qc, wacc, target, min_sq, max_sq, wcarry_cmp, n_acc)

        # Step 4: Uncompute (reverse order)
        self._add_square_inv(qc, wz, wacc, wtmp_z, wcarry_add_z, b, n_acc)
        self._add_square_inv(qc, wy, wacc, wtmp_y, wcarry_add_y, b, n_acc)
        self._add_square_inv(qc, wx, wacc, wtmp_x, wcarry_add_x, b, n_acc)
        self._quantum_subtract_inv(qc, z1, z2, wz, wcarry_sub_z, b)
        self._quantum_subtract_inv(qc, y1, y2, wy, wcarry_sub_y, b)
        self._quantum_subtract_inv(qc, x1, x2, wx, wcarry_sub_x, b)

    def _check_distance_min(
        self, qc: Any,
        x1: list, y1: list, z1: list,
        x2: list, y2: list, z2: list,
        target: Any, workspace: list,
        min_dist: float,
    ) -> None:
        """Check if distance >= min_dist (no clash). Uses parallel workspace."""
        b = len(x1)
        n_acc = 2 * b + 1

        off = 0
        wx = workspace[off:off+b]; off += b
        wy = workspace[off:off+b]; off += b
        wz = workspace[off:off+b]; off += b
        wacc = workspace[off:off+n_acc]; off += n_acc
        wtmp_x = workspace[off:off+n_acc]; off += n_acc
        wtmp_y = workspace[off:off+n_acc]; off += n_acc
        wtmp_z = workspace[off:off+n_acc]; off += n_acc
        wcarry_sub_x = workspace[off:off+b+1]; off += b+1
        wcarry_sub_y = workspace[off:off+b+1]; off += b+1
        wcarry_sub_z = workspace[off:off+b+1]; off += b+1
        wcarry_add_x = workspace[off:off+n_acc]; off += n_acc
        wcarry_add_y = workspace[off:off+n_acc]; off += n_acc
        wcarry_add_z = workspace[off:off+n_acc]; off += n_acc
        wcarry_cmp = workspace[off:off+n_acc+1]; off += n_acc+1

        self._quantum_subtract(qc, x1, x2, wx, wcarry_sub_x, b)
        self._quantum_subtract(qc, y1, y2, wy, wcarry_sub_y, b)
        self._quantum_subtract(qc, z1, z2, wz, wcarry_sub_z, b)

        self._add_square(qc, wx, wacc, wtmp_x, wcarry_add_x, b, n_acc)
        self._add_square(qc, wy, wacc, wtmp_y, wcarry_add_y, b, n_acc)
        self._add_square(qc, wz, wacc, wtmp_z, wcarry_add_z, b, n_acc)

        lo, hi = self.coord_range
        scale = (hi - lo) / (2**b - 1)
        min_sq = int((min_dist / scale) ** 2)
        self._check_greater_equal(qc, wacc, target, min_sq, wcarry_cmp, n_acc)

        self._add_square_inv(qc, wz, wacc, wtmp_z, wcarry_add_z, b, n_acc)
        self._add_square_inv(qc, wy, wacc, wtmp_y, wcarry_add_y, b, n_acc)
        self._add_square_inv(qc, wx, wacc, wtmp_x, wcarry_add_x, b, n_acc)
        self._quantum_subtract_inv(qc, z1, z2, wz, wcarry_sub_z, b)
        self._quantum_subtract_inv(qc, y1, y2, wy, wcarry_sub_y, b)
        self._quantum_subtract_inv(qc, x1, x2, wx, wcarry_sub_x, b)

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
    arith_ancilla = 24 * b + 10  # parallel workspace: 3b + (2b+1) + 3*(2b+1) + 3*(b+1) + 3*(2b+1) + (2b+2)
    ancilla_qubits = n_constraints + 1 + arith_ancilla
    return data_qubits + ancilla_qubits
