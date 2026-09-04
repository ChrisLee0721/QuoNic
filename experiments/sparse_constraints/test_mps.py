"""Test MPS (Matrix Product State) simulator for larger qubit counts.

MPS simulator uses tensor networks instead of full state vector.
Memory: O(n * chi^2) instead of O(2^n), where chi is bond dimension.
Works well for circuits with limited entanglement.
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from gciqa import ConstraintSet, GeometricConstraint
from gciqa.oracle import GroverOracle, estimate_oracle_qubits


def test_mps_simulation(n_atoms, bits, n_constraints):
    """Test MPS simulator with given parameters."""
    n_qubits = n_atoms * 3 * bits
    total_qubits = estimate_oracle_qubits(n_atoms, bits, n_constraints)

    print(f"\n  Atoms: {n_atoms}, Bits: {bits}")
    print(f"  Data qubits: {n_qubits}")
    print(f"  Total qubits (with ancilla): {total_qubits}")

    # Build simple constraints
    cs = ConstraintSet([
        GeometricConstraint.bond(str(i), str(i+1), min_dist=1.0, max_dist=3.0)
        for i in range(n_constraints)
    ])

    oracle = GroverOracle(
        n_qubits=n_qubits,
        constraints=cs,
        bits_per_coord=bits,
    )

    # Build circuit
    from gciqa.search import _build_grover_circuit
    qc = _build_grover_circuit(oracle, n_qubits, n_iterations=1)

    print(f"  Circuit qubits: {qc.num_qubits}")
    print(f"  Circuit depth: {qc.depth()}")

    # Try MPS simulator
    from qiskit_aer import AerSimulator

    t0 = time.time()
    try:
        sim = AerSimulator(method='matrix_product_state')
        result = sim.run(qc, shots=100).result()
        counts = result.get_counts(qc)
        elapsed = time.time() - t0
        print(f"  MPS: OK, {len(counts)} states, {elapsed:.1f}s")
        return True
    except Exception as e:
        elapsed = time.time() - t0
        print(f"  MPS: FAILED ({e}), {elapsed:.1f}s")

    # Try automatic method
    t0 = time.time()
    try:
        sim = AerSimulator()
        result = sim.run(qc, shots=100).result()
        counts = result.get_counts(qc)
        elapsed = time.time() - t0
        print(f"  Auto: OK, {len(counts)} states, {elapsed:.1f}s")
        return True
    except Exception as e:
        elapsed = time.time() - t0
        print(f"  Auto: FAILED ({e}), {elapsed:.1f}s")

    return False


def main():
    print("="*70)
    print("MPS Simulator Test")
    print("="*70)

    # Test increasing sizes
    configs = [
        # (n_atoms, bits, n_constraints)
        (2, 3, 1),   # 18+31 = 49 qubits
        (3, 3, 2),   # 27+38 = 65 qubits
        (3, 2, 2),   # 18+22 = 40 qubits
        (4, 2, 3),   # 24+22 = 46 qubits
        (5, 2, 4),   # 30+22 = 52 qubits
    ]

    for n_atoms, bits, n_constraints in configs:
        test_mps_simulation(n_atoms, bits, n_constraints)


if __name__ == "__main__":
    main()
