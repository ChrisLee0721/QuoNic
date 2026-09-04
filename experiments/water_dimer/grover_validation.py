"""Quantum Grover search validation on water dimer.

Tests whether quantum Grover search (enumeration mode) can find
valid conformations satisfying geometric constraints, and compares
with classical search.

Parameters:
    - 2 super-atoms (one per water molecule)
    - 2 bits per coordinate
    - Range (-1.5, 1.5) Angstrom
    - 12 qubits (enumeration mode)

Achievable O-O distances with this encoding:
    Step = 3.0 / 3 = 1.0 Å
    Positions: -1.5, -0.5, 0.5, 1.5
    Possible O-O distances: 1.73, 2.45, 3.00, 3.46, 4.90 Å
    Closest to experimental 2.98 Å: 3.00 Å (0.67% error)

Usage:
    python grover_validation.py
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from gciqa import (
    ConstraintSet,
    GeometricConstraint,
    GroverOracle,
    grover_search,
)


def distance(c1, c2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(c1, c2)))


def decode_bitstring(bitstring, n_atoms, bits_per_coord, coord_range):
    """Decode a bitstring to atom coordinates."""
    b = bits_per_coord
    bits_per_atom = 3 * b
    lo, hi = coord_range
    scale = (hi - lo) / (2**b - 1)
    bits = bitstring[::-1]  # qiskit convention: reverse to get qubit order

    coords = {}
    for i in range(n_atoms):
        start = i * bits_per_atom
        # Reverse each coordinate's bits for int() (MSB first)
        x_bits = bits[start:start+b][::-1]
        y_bits = bits[start+b:start+2*b][::-1]
        z_bits = bits[start+2*b:start+3*b][::-1]
        x = lo + int(x_bits, 2) * scale
        y = lo + int(y_bits, 2) * scale
        z = lo + int(z_bits, 2) * scale
        coords[f"{i}"] = (x, y, z)
    return coords


def main():
    print("=" * 60)
    print("Quantum Grover Search Validation: Water Dimer")
    print("=" * 60)

    # Parameters
    n_atoms = 2  # 2 super-atoms (one per water molecule)
    bits_per_coord = 2
    coord_range = (-1.5, 1.5)
    n_qubits = n_atoms * 3 * bits_per_coord  # 12 qubits

    lo, hi = coord_range
    step = (hi - lo) / (2**bits_per_coord - 1)
    print(f"\nEncoding: {n_atoms} atoms, {bits_per_coord} bits/coord, {n_qubits} qubits")
    print(f"Range: ({lo}, {hi}) Å, step = {step:.3f} Å")
    print(f"Possible positions: {[lo + i * step for i in range(2**bits_per_coord)]}")

    # Constraints: O-O bond distance
    expected_oo = 2.98  # Experimental
    constraints = ConstraintSet([
        GeometricConstraint.bond("0", "1", min_dist=2.5, max_dist=3.5),
        GeometricConstraint.pocket(center=(0, 0, 0), radius=3.0),
    ])

    print("\nConstraints:")
    for c in constraints:
        print(f"  {c}")

    # Build oracle and enumerate valid states
    oracle = GroverOracle(
        n_qubits=n_qubits,
        constraints=constraints,
        bits_per_coord=bits_per_coord,
        coord_range=coord_range,
    )

    # Count valid states
    valid_states = []
    for state_int in range(2**n_qubits):
        bitstring = format(state_int, f'0{n_qubits}b')
        if oracle.classical_oracle_fn(bitstring):
            coords = decode_bitstring(bitstring, n_atoms, bits_per_coord, coord_range)
            oo_dist = distance(coords["0"], coords["1"])
            valid_states.append((state_int, bitstring, coords, oo_dist))

    print(f"\nValid states: {len(valid_states)} / {2**n_qubits} ({100*len(valid_states)/2**n_qubits:.1f}%)")
    print("\nAll valid conformations:")
    for state_int, bitstring, coords, oo_dist in valid_states:
        print(f"  |{bitstring}⟩ (={state_int:3d}): O1={coords['0']}, O2={coords['1']}, O-O={oo_dist:.3f} Å")

    if not valid_states:
        print("No valid states found! Constraints may be too tight.")
        return

    # Find best achievable distance
    best = min(valid_states, key=lambda x: abs(x[3] - expected_oo))
    print(f"\nBest achievable: O-O = {best[3]:.3f} Å (error: {abs(best[3] - expected_oo):.3f} Å, {100*abs(best[3] - expected_oo)/expected_oo:.1f}%)")

    # Run Grover search
    print(f"\n{'='*60}")
    print("Running Quantum Grover Search...")
    print(f"{'='*60}")

    n_solutions = len(valid_states)
    N = 2**n_qubits
    n_iterations = max(1, int(math.pi / 4 * math.sqrt(N / n_solutions)))
    print(f"Optimal Grover iterations: {n_iterations}")

    result = grover_search(
        oracle=oracle,
        n_qubits=n_qubits,
        n_iterations=n_iterations,
        n_shots=1000,
    )

    print("\nGrover search results:")
    print(f"  Unique states measured: {result.n_unique}")
    print(f"  Total shots: {result.n_shots}")

    # Decode top states
    print("\nTop 10 measured states:")
    for bitstring, count in result.top_states[:10]:
        coords = decode_bitstring(bitstring, n_atoms, bits_per_coord, coord_range)
        oo_dist = distance(coords["0"], coords["1"])
        is_valid = oracle.classical_oracle_fn(bitstring)
        marker = "✓" if is_valid else "✗"
        print(f"  {marker} |{bitstring}⟩: O-O={oo_dist:.3f} Å, count={count} ({100*count/result.n_shots:.1f}%)")

    # Count how many measurements are valid states
    valid_count = 0
    for bitstring, count in result.measurements.items():
        if oracle.classical_oracle_fn(bitstring):
            valid_count += count

    print(f"\nValid state measurements: {valid_count} / {result.n_shots} ({100*valid_count/result.n_shots:.1f}%)")
    print(f"Classical probability: {100*n_solutions/N:.1f}%")

    amplification = (valid_count / result.n_shots) / (n_solutions / N) if n_solutions > 0 else 0
    print(f"Grover amplification factor: {amplification:.1f}x")

    # Compare with classical random search
    print(f"\n{'='*60}")
    print("Classical Random Search (1000 samples)...")
    print(f"{'='*60}")

    import random
    classical_valid = 0
    classical_oo_dists = []
    for _ in range(1000):
        state_int = random.randint(0, N - 1)
        bitstring = format(state_int, f'0{n_qubits}b')
        if oracle.classical_oracle_fn(bitstring):
            classical_valid += 1
            coords = decode_bitstring(bitstring, n_atoms, bits_per_coord, coord_range)
            classical_oo_dists.append(distance(coords["0"], coords["1"]))

    print(f"Classical valid: {classical_valid} / 1000 ({100*classical_valid/1000:.1f}%)")
    if classical_oo_dists:
        mean_oo = sum(classical_oo_dists) / len(classical_oo_dists)
        print(f"Mean O-O distance: {mean_oo:.3f} Å")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Encoding: {n_atoms} atoms × {bits_per_coord} bits/coord = {n_qubits} qubits")
    print(f"Search space: {N} states, {n_solutions} valid ({100*n_solutions/N:.1f}%)")
    print(f"Best achievable O-O: {best[3]:.3f} Å (error {100*abs(best[3] - expected_oo)/expected_oo:.1f}%)")
    print(f"Grover amplification: {amplification:.1f}x")
    print(f"Grover valid rate: {100*valid_count/result.n_shots:.1f}% vs Classical {100*classical_valid/1000:.1f}%")

    if amplification > 1.5:
        print("\nCONCLUSION: Quantum Grover search successfully amplifies valid states!")
    else:
        print("\nCONCLUSION: Grover amplification is weak — may need more iterations or better encoding.")


if __name__ == "__main__":
    main()
