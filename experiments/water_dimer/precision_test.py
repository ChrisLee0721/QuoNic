"""Test higher encoding precision on water dimer.

Compares 2 bits/coord vs 3 bits/coord encoding:
- 2 bits: 4 positions per axis, step = 1.0 Å
- 3 bits: 8 positions per axis, step ≈ 0.429 Å

With 3 bits, the closest achievable O-O distance to 2.98 Å should be
much more accurate.

Usage:
    python precision_test.py
"""

import sys
import os
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from quonic.gciqa import (
    GroverOracle,
    ConstraintSet,
    GeometricConstraint,
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
    bits = bitstring[::-1]

    coords = {}
    for i in range(n_atoms):
        start = i * bits_per_atom
        x_bits = bits[start:start+b][::-1]
        y_bits = bits[start+b:start+2*b][::-1]
        z_bits = bits[start+2*b:start+3*b][::-1]
        x = lo + int(x_bits, 2) * scale
        y = lo + int(y_bits, 2) * scale
        z = lo + int(z_bits, 2) * scale
        coords[f"{i}"] = (x, y, z)
    return coords


def analyze_encoding(bits_per_coord, coord_range, expected_oo):
    """Analyze achievable distances for a given encoding."""
    b = bits_per_coord
    lo, hi = coord_range
    step = (hi - lo) / (2**b - 1)
    positions = [lo + i * step for i in range(2**b)]

    # Find all achievable O-O distances
    distances = set()
    for x1 in positions:
        for y1 in positions:
            for z1 in positions:
                for x2 in positions:
                    for y2 in positions:
                        for z2 in positions:
                            d = math.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)
                            distances.add(round(d, 6))

    distances = sorted(distances)
    closest = min(distances, key=lambda d: abs(d - expected_oo))
    error = abs(closest - expected_oo)

    return {
        'bits': b,
        'step': step,
        'positions': positions,
        'n_distances': len(distances),
        'distances': distances,
        'closest': closest,
        'error': error,
    }


def main():
    print("=" * 60)
    print("Encoding Precision Comparison: Water Dimer")
    print("=" * 60)

    expected_oo = 2.98
    coord_range = (-1.5, 1.5)

    # Analyze both encodings
    for b in [2, 3]:
        info = analyze_encoding(b, coord_range, expected_oo)
        print(f"\n--- {b} bits/coord ---")
        print(f"  Step: {info['step']:.4f} Å")
        print(f"  Positions: {[f'{p:.3f}' for p in info['positions']]}")
        print(f"  Achievable O-O distances: {info['n_distances']}")
        print(f"  Closest to {expected_oo} Å: {info['closest']:.3f} Å (error: {info['error']:.3f} Å, {100*info['error']/expected_oo:.1f}%)")

        # Show distances near expected
        near = [d for d in info['distances'] if abs(d - expected_oo) < 1.0]
        print(f"  Distances near {expected_oo} Å: {[f'{d:.3f}' for d in near]}")

    # Run Grover search with 3 bits/coord
    print(f"\n{'='*60}")
    print("Grover Search with 3 bits/coord")
    print(f"{'='*60}")

    b = 3
    n_atoms = 2
    n_qubits = n_atoms * 3 * b  # 18 qubits (arithmetic mode)

    constraints = ConstraintSet([
        GeometricConstraint.bond("0", "1", min_dist=2.5, max_dist=3.5),
        GeometricConstraint.pocket(center=(0, 0, 0), radius=3.0),
    ])

    oracle = GroverOracle(
        n_qubits=n_qubits,
        constraints=constraints,
        bits_per_coord=b,
        coord_range=coord_range,
    )

    # Count valid states
    valid_count = 0
    total = 2**n_qubits
    print(f"\nCounting valid states ({total} total)...")
    for state_int in range(total):
        bitstring = format(state_int, f'0{n_qubits}b')
        if oracle.classical_oracle_fn(bitstring):
            valid_count += 1

    print(f"Valid states: {valid_count} / {total} ({100*valid_count/total:.2f}%)")

    if valid_count == 0:
        print("No valid states found!")
        return

    # Find best achievable distance
    best_oo = None
    best_error = float('inf')
    for state_int in range(total):
        bitstring = format(state_int, f'0{n_qubits}b')
        if oracle.classical_oracle_fn(bitstring):
            coords = decode_bitstring(bitstring, n_atoms, b, coord_range)
            oo = distance(coords["0"], coords["1"])
            err = abs(oo - expected_oo)
            if err < best_error:
                best_error = err
                best_oo = oo

    print(f"Best achievable O-O: {best_oo:.3f} Å (error: {best_error:.3f} Å, {100*best_error/expected_oo:.1f}%)")

    # Note: Quantum simulation of arithmetic oracle requires too much memory
    # for statevector simulator (42+ qubits). On a real quantum computer,
    # Grover search would amplify valid states from 32.44% to ~100%.
    print(f"\nNote: Arithmetic oracle circuit is too large for statevector simulation.")
    print(f"On a real quantum computer, Grover would amplify valid states from {100*valid_count/total:.1f}% to ~100%.")

    # Comparison
    print(f"\n{'='*60}")
    print("COMPARISON: 2-bit vs 3-bit Encoding")
    print(f"{'='*60}")
    print(f"{'Encoding':<20} {'Qubits':>8} {'Best O-O':>10} {'Error':>10} {'Error%':>8}")
    print("-" * 60)

    # 2-bit result (from previous experiment)
    print(f"{'2 bits/coord':<20} {'12':>8} {'3.000':>9} Å {'0.020':>9} Å {'0.7':>7}%")
    print(f"{'3 bits/coord':<20} {n_qubits:>8} {best_oo:>9.3f} Å {best_error:>9.3f} Å {100*best_error/expected_oo:>7.1f}%")

    if best_error < 0.020:
        print(f"\n3-bit encoding achieves {0.020/best_error:.1f}x better precision")
    else:
        print(f"\n2-bit encoding already achieves sufficient precision for this system")


if __name__ == "__main__":
    main()
