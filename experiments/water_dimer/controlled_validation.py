"""Controlled validation: Physical vs Random constraints.

Tests whether geometric constraints actually guide the search toward
 physically meaningful conformations, or if any constraints would work.

Experiment:
    - Treatment: Physical constraints (correct atom pairs, known distances)
    - Control: Random constraints (wrong atom pairs, same distance ranges)
    - Both run on the same water dimer system
    - Compare: does O-O distance converge to ~2.98 Å in both cases?

If physical constraints are meaningful, treatment should outperform control.
If constraints don't matter, both should give similar results.

Usage:
    python controlled_validation.py
"""

import sys
import os
import random
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from gciqa import (
    GCIQA,
    GeometricConstraint,
    ConstraintSet,
)


def distance(c1, c2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(c1, c2)))


def run_experiment(name, constraints, n_trials=5):
    """Run GCIQA multiple times and collect O-O distances."""
    results = []
    for trial in range(n_trials):
        gciqa = GCIQA(
            n_super_atoms=6,
            constraints=constraints,
            coord_range=(-10.0, 10.0),
            bits_per_coord=4,
            alpha=0.7,
            convergence_threshold=1.0,
            use_quantum=False,
        )
        result = gciqa.run(max_iterations=5, n_shots=300, n_clusters=1)

        if result.best_conformation and "0" in result.best_conformation and "3" in result.best_conformation:
            o1 = result.best_conformation["0"]
            o2 = result.best_conformation["3"]
            oo_dist = distance(o1, o2)
            results.append(oo_dist)
        else:
            results.append(None)

    return results


def main():
    print("=" * 60)
    print("Controlled Validation: Physical vs Random Constraints")
    print("=" * 60)

    # Water dimer atoms
    # O1=0, H1=1, H2=2, O2=3, H3=4, H4=5

    # --- Treatment: Physical constraints ---
    physical = ConstraintSet([
        # O-O distance (known: 2.98 Å)
        GeometricConstraint.bond("0", "3", min_dist=2.5, max_dist=3.5),
        # H-bond: O1...H4 (known: ~1.95 Å)
        GeometricConstraint.bond("0", "4", min_dist=1.5, max_dist=2.5),
        # No clash
        GeometricConstraint.no_clash("0", "3", min_dist=2.0),
        # Pocket
        GeometricConstraint.pocket(center=(0, 0, 0), radius=6.0),
    ])

    # --- Control 1: Random constraints (wrong atom pairs) ---
    # Same distance ranges, but applied to wrong atoms
    random_constraints = ConstraintSet([
        # H-H "bond" (physically meaningless — H atoms don't bond at 3 Å)
        GeometricConstraint.bond("1", "4", min_dist=2.5, max_dist=3.5),
        # O-H at wrong distance
        GeometricConstraint.bond("2", "5", min_dist=1.5, max_dist=2.5),
        # No clash between H atoms (weak constraint)
        GeometricConstraint.no_clash("1", "4", min_dist=2.0),
        # Pocket
        GeometricConstraint.pocket(center=(0, 0, 0), radius=6.0),
    ])

    # --- Control 2: Wide constraints (correct atoms, loose ranges) ---
    # Tests whether constraint precision matters
    wide_constraints = ConstraintSet([
        # O-O: [1.0, 5.0] instead of [2.5, 3.5]
        GeometricConstraint.bond("0", "3", min_dist=1.0, max_dist=5.0),
        # H-bond: [1.0, 4.0] instead of [1.5, 2.5]
        GeometricConstraint.bond("0", "4", min_dist=1.0, max_dist=4.0),
        # No clash
        GeometricConstraint.no_clash("0", "3", min_dist=1.5),
        # Pocket
        GeometricConstraint.pocket(center=(0, 0, 0), radius=8.0),
    ])

    n_trials = 5
    expected = 2.98  # Experimental O-O distance

    print(f"\nRunning {n_trials} trials each...")
    print(f"Expected O-O distance: {expected} Å\n")

    experiments = [
        ("Physical (tight)", physical),
        ("Random (wrong atoms)", random_constraints),
        ("Wide (correct atoms, loose)", wide_constraints),
    ]

    all_results = {}
    for name, constraints in experiments:
        print(f"--- {name} ---")
        results = run_experiment(name, constraints, n_trials)
        all_results[name] = results
        for i, d in enumerate(results):
            if d:
                error = abs(d - expected)
                print(f"  Trial {i+1}: O-O = {d:.2f} Å (error: {error:.2f} Å, {100*error/expected:.1f}%)")
            else:
                print(f"  Trial {i+1}: No valid conformation")
        print()

    # Summary table
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Group':<30} {'Mean O-O':>10} {'Error':>10} {'Error%':>8}")
    print("-" * 60)

    for name, results in all_results.items():
        valid = [d for d in results if d is not None]
        if valid:
            mean_d = sum(valid) / len(valid)
            error = abs(mean_d - expected)
            print(f"{name:<30} {mean_d:>9.2f} Å {error:>9.2f} Å {100*error/expected:>7.1f}%")
        else:
            print(f"{name:<30} {'N/A':>10} {'N/A':>10} {'N/A':>8}")

    # Conclusion
    print("\n" + "-" * 60)
    phys_valid = [d for d in all_results["Physical (tight)"] if d is not None]
    rand_valid = [d for d in all_results["Random (wrong atoms)"] if d is not None]
    wide_valid = [d for d in all_results["Wide (correct atoms, loose)"] if d is not None]

    phys_error = abs(sum(phys_valid)/len(phys_valid) - expected) if phys_valid else float('inf')
    rand_error = abs(sum(rand_valid)/len(rand_valid) - expected) if rand_valid else float('inf')
    wide_error = abs(sum(wide_valid)/len(wide_valid) - expected) if wide_valid else float('inf')

    print("Key comparisons:")
    print(f"  Physical vs Random:  {phys_error:.2f} vs {rand_error:.2f} Å "
          f"({'Physical wins' if phys_error < rand_error else 'Random wins'})")
    print(f"  Physical vs Wide:    {phys_error:.2f} vs {wide_error:.2f} Å "
          f"({'Physical wins' if phys_error < wide_error else 'Wide wins'})")
    print(f"  Wide vs Random:      {wide_error:.2f} vs {rand_error:.2f} Å "
          f"({'Wide wins' if wide_error < rand_error else 'Random wins'})")

    if phys_error < wide_error < rand_error:
        print("\nCONCLUSION: Both atom identity AND constraint precision matter.")
        print("Physical constraints with tight ranges give best results.")
    elif phys_error < wide_error and wide_error >= rand_error:
        print("\nCONCLUSION: Atom identity matters, but constraint precision doesn't.")
        print("Wide constraints on correct atoms are as bad as random constraints.")
    elif phys_error >= wide_error:
        print("\nCONCLUSION: Wide constraints perform as well as tight ones.")
        print("Constraint precision is less important than atom identity.")
    print("=" * 60)


if __name__ == "__main__":
    main()
