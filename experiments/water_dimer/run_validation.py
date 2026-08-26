"""GCIQA validation: Water dimer binding geometry.

Water dimer (H2O...H2O) is the simplest hydrogen-bonded system.
Known equilibrium geometry:
    - O-O distance: 2.98 Angstrom
    - H-bond distance (O-H...O): 1.95 Angstrom
    - O-H...O angle: ~170 degrees

This validates GCIQA by searching for the binding geometry using
only geometric constraints (no energy function).

Usage:
    python run_validation.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from quonic.gciqa import (
    GCIQA,
    GeometricConstraint,
    ConstraintSet,
    coarse_grain,
)


def main():
    print("=" * 60)
    print("GCIQA Validation: Water Dimer Binding Geometry")
    print("=" * 60)

    # Water dimer: 2 water molecules = 6 atoms
    # Molecule 1 (donor): O1, H1, H2
    # Molecule 2 (acceptor): O2, H3, H4
    atoms = ["O", "H", "H", "O", "H", "H"]

    # Approximate initial coordinates (far apart, not optimized)
    # Molecule 1 centered at origin
    # Molecule 2 displaced along x-axis
    coords = [
        (0.0, 0.0, 0.0),      # O1
        (0.76, 0.59, 0.0),    # H1
        (0.76, -0.59, 0.0),   # H2
        (5.0, 0.0, 0.0),      # O2 (far away initially)
        (5.76, 0.59, 0.0),    # H3
        (5.76, -0.59, 0.0),   # H4
    ]

    print(f"\nSystem: {len(atoms)} atoms ({', '.join(atoms)})")
    print("Molecule 1 (donor): O1, H1, H2")
    print("Molecule 2 (acceptor): O2, H3, H4")

    # Define geometric constraints based on known water dimer geometry
    constraints = ConstraintSet([
        # O-O distance: 2.5 - 3.5 Angstrom (known: 2.98)
        GeometricConstraint.bond("0", "3", min_dist=2.5, max_dist=3.5),

        # H-bond distance: O1...H4 should be 1.5 - 2.5 Angstrom (known: ~1.95)
        GeometricConstraint.bond("0", "4", min_dist=1.5, max_dist=2.5),

        # No steric clash between oxygens
        GeometricConstraint.no_clash("0", "3", min_dist=2.0),

        # Pocket: all atoms within 6 Angstrom of origin
        GeometricConstraint.pocket(center=(0, 0, 0), radius=6.0),
    ])

    print(f"\nConstraints: {len(constraints)}")
    for c in constraints.constraints:
        print(f"  {c.type.value}: atoms={c.atoms}, params={c.params}")

    # Run GCIQA without coarse-graining (6 atoms is small enough)
    print("\n--- Running GCIQA ---")
    gciqa = GCIQA(
        n_super_atoms=6,  # Use all atoms directly
        constraints=constraints,
        coord_range=(-10.0, 10.0),
        bits_per_coord=4,
        alpha=0.7,
        convergence_threshold=1.0,
        use_quantum=False,
    )

    result = gciqa.run(max_iterations=8, n_shots=300, n_clusters=1)

    print(f"\nIterations: {result.n_iterations}")
    print(f"Converged: {result.converged}")
    print(f"Total time: {result.total_time:.3f} s")

    if result.coarse_graining:
        cg = result.coarse_graining
        print(f"\nCoarse-graining: {cg.n_full_atoms} atoms -> {cg.n_super_atoms} super-atoms")
        for i, (sa, sc) in enumerate(zip(cg.super_atoms, cg.super_coords)):
            members = cg.super_to_atoms[i]
            member_atoms = [atoms[j] for j in members]
            print(f"  {sa}: {sc[0]:.2f}, {sc[1]:.2f}, {sc[2]:.2f} "
                  f"(atoms: {member_atoms})")

    if result.best_conformation:
        print(f"\nBest conformation ({len(result.best_conformation)} atoms):")
        for name, coord in sorted(result.best_conformation.items(), key=lambda x: int(x[0])):
            print(f"  {atoms[int(name)]}{name}: ({coord[0]:.2f}, {coord[1]:.2f}, {coord[2]:.2f})")

        # Check O-O distance
        if "0" in result.best_conformation and "3" in result.best_conformation:
            o1 = result.best_conformation["0"]
            o2 = result.best_conformation["3"]
            import math
            oo_dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(o1, o2)))
            print(f"\nO-O distance: {oo_dist:.2f} Angstrom (expected: ~2.98)")

        # Convergence history
        if result.convergence_history:
            print("\nConvergence history:")
            for it, radius in result.convergence_history:
                print(f"  Iteration {it}: radius = {radius:.2f} Angstrom")
    else:
        print("\nNo valid conformation found!")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if result.best_conformation and "0" in result.best_conformation and "3" in result.best_conformation:
        o1 = result.best_conformation["0"]
        o2 = result.best_conformation["3"]
        import math
        oo_dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(o1, o2)))
        error = abs(oo_dist - 2.98)
        print(f"O-O distance: {oo_dist:.2f} Angstrom")
        print(f"Expected:     2.98 Angstrom")
        print(f"Error:        {error:.2f} Angstrom ({100*error/2.98:.1f}%)")
        if error < 0.5:
            print("PASS: Within 0.5 Angstrom of expected value")
        else:
            print("FAIL: More than 0.5 Angstrom from expected value")
    else:
        print("FAIL: No valid conformation found")
    print("=" * 60)
    return result


if __name__ == "__main__":
    main()
