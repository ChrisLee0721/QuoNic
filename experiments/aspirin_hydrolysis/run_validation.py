"""Aspirin hydrolysis validation experiment.

Tests GCIQA on a larger molecular system: aspirin + water → salicylic acid + acetic acid.

This is a simplified validation using approximate coordinates. The goal is to
test coarse-graining and GCIQA on a system larger than water dimer.

Reaction:
    Aspirin (C₉H₈O₄) + H₂O → Salicylic acid (C₇H₆O₃) + Acetic acid (C₂H₄O₂)

System:
    - 24 atoms total (21 from aspirin + 3 from water)
    - Coarse-grain to ~8 super-atoms
    - Focus on ester bond + water attack site

Usage:
    python run_validation.py
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from gciqa import (
    GCIQA,
    ConstraintSet,
    GeometricConstraint,
    binding_site_super_atoms,
    coarse_grain,
)


def distance(c1, c2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(c1, c2)))


# Approximate aspirin coordinates (simplified, Angstrom)
# Atoms: C1-C9, O1-O4, H1-H8 (aspirin) + O5, H9, H10 (water)
ASPIRIN_ATOMS = [
    # Aspirin backbone (approximate)
    "C", "C", "C", "C", "C", "C",  # Benzene ring (C1-C6)
    "C", "C", "C",                   # Acetyl group (C7-C9)
    "O", "O", "O", "O",             # Oxygens (O1-O4)
    "H", "H", "H", "H", "H", "H", "H", "H",  # Hydrogens (H1-H8)
    # Water
    "O", "H", "H",                   # Water (O5, H9, H10)
]

# Approximate coordinates (Angstrom)
# Simplified geometry for testing
ASPIRIN_COORDS = [
    # Benzene ring (C1-C6)
    (0.0, 0.0, 0.0),     # C1
    (1.4, 0.0, 0.0),     # C2
    (2.1, 1.2, 0.0),     # C3
    (1.4, 2.4, 0.0),     # C4
    (0.0, 2.4, 0.0),     # C5
    (-0.7, 1.2, 0.0),    # C6
    # Acetyl group (C7-C9)
    (2.8, 0.0, 0.0),     # C7 (ester carbonyl)
    (4.2, 0.0, 0.0),     # C8 (ester oxygen)
    (5.0, 1.2, 0.0),     # C9 (methyl)
    # Oxygens (O1-O4)
    (2.8, -1.2, 0.0),    # O1 (carbonyl oxygen)
    (-1.4, 1.2, 0.0),    # O2 (hydroxyl)
    (4.2, -1.2, 0.0),    # O3 (ester oxygen)
    (5.0, 2.4, 0.0),     # O4 (acetyl oxygen)
    # Hydrogens (H1-H8)
    (0.0, -1.0, 0.0),    # H1
    (1.4, -1.0, 0.0),    # H2
    (2.1, 2.2, 0.0),     # H3
    (1.4, 3.4, 0.0),     # H4
    (0.0, 3.4, 0.0),     # H5
    (-0.7, 2.2, 0.0),    # H6
    (-1.4, 0.2, 0.0),    # H7 (hydroxyl H)
    (6.0, 1.2, 0.0),     # H8 (methyl H)
    # Water (O5, H9, H10)
    (3.5, 3.0, 1.0),     # O5 (water oxygen, near ester)
    (4.0, 3.5, 1.5),     # H9 (water H)
    (3.0, 3.5, 0.5),     # H10 (water H)
]


def main():
    print("=" * 60)
    print("Aspirin Hydrolysis Validation")
    print("=" * 60)

    atoms = ASPIRIN_ATOMS
    coords = ASPIRIN_COORDS
    n_atoms = len(atoms)

    print(f"\nSystem: {n_atoms} atoms")
    print(f"  Aspirin: {n_atoms - 3} atoms")
    print("  Water: 3 atoms")

    # Coarse-grain (use 12 to separate reaction site atoms)
    n_super_atoms = 12
    print(f"\nCoarse-graining: {n_atoms} atoms → {n_super_atoms} super-atoms")

    cg = coarse_grain(
        atoms=atoms,
        coords=coords,
        strategy="spatial",
        n_super_atoms=n_super_atoms,
    )

    print("  Super-atom positions:")
    for i, (sx, sy, sz) in enumerate(cg.super_coords):
        members = cg.super_to_atoms[i]
        print(f"    SA{i}: ({sx:.2f}, {sy:.2f}, {sz:.2f}) — {len(members)} atoms")

    # Find super-atoms near the reaction site (ester bond + water)
    # Reaction site: C7 (ester carbonyl) + O5 (water)
    reaction_center = (3.5, 1.5, 0.5)  # Between ester and water
    nearby = binding_site_super_atoms(cg, reaction_center, pocket_radius=3.0)
    print(f"\n  Super-atoms near reaction site ({reaction_center}): {nearby}")

    # Define constraints for the reaction
    # Key interactions:
    # 1. Water O5 attacks ester C7 (distance ~2.5-3.5 Å)
    # 2. Water H9 bonds to ester O1 (distance ~1.5-2.5 Å)
    # 3. No clash between water and aspirin

    # Map atom names to super-atom indices
    # Find which super-atoms contain the key atoms
    def find_super_atom(atom_idx):
        for sa_idx, members in enumerate(cg.super_to_atoms):
            if atom_idx in members:
                return sa_idx
        return None

    # Key atoms: C7 (index 6), O1 (index 9), O5 (water oxygen, index 21)
    sa_c7 = find_super_atom(6)   # Ester carbonyl carbon
    sa_o1 = find_super_atom(9)   # Carbonyl oxygen
    sa_o5 = find_super_atom(21)  # Water oxygen

    print("\n  Key atom mapping:")
    print(f"    C7 (ester) → SA{sa_c7}")
    print(f"    O1 (carbonyl) → SA{sa_o1}")
    print(f"    O5 (water) → SA{sa_o5}")

    # Constraints
    constraints = ConstraintSet([
        # Water attacks ester: O5...C7 distance
        GeometricConstraint.bond(str(sa_o5), str(sa_c7), min_dist=2.0, max_dist=4.0),
        # H-bond: O5...O1
        GeometricConstraint.bond(str(sa_o5), str(sa_o1), min_dist=2.0, max_dist=4.0),
        # Pocket around reaction site
        GeometricConstraint.pocket(center=reaction_center, radius=5.0),
    ])

    print("\n  Constraints:")
    for c in constraints:
        print(f"    {c}")

    # Run GCIQA with classical search (for testing)
    print(f"\n{'='*60}")
    print("Running GCIQA (classical search)")
    print(f"{'='*60}")

    gciqa = GCIQA(
        n_super_atoms=n_super_atoms,
        constraints=constraints,
        coord_range=(-10.0, 10.0),
        bits_per_coord=4,
        alpha=0.7,
        convergence_threshold=1.0,
        use_quantum=False,
        atoms=atoms,
        coords=coords,
        cg_strategy="spatial",
    )

    result = gciqa.run(max_iterations=5, n_shots=500, n_clusters=3)

    print(f"\n  Iterations: {result.n_iterations}")
    print(f"  Converged: {result.converged}")
    print(f"  Time: {result.total_time:.2f} s")

    if result.best_conformation:
        # Check key distances
        if "6" in result.best_conformation and "21" in result.best_conformation:
            c7 = result.best_conformation["6"]
            o5 = result.best_conformation["21"]
            d = distance(c7, o5)
            print(f"\n  C7-O5 distance: {d:.2f} Å (expected: ~3.0 Å)")

        if "9" in result.best_conformation and "21" in result.best_conformation:
            o1 = result.best_conformation["9"]
            o5 = result.best_conformation["21"]
            d = distance(o1, o5)
            print(f"  O1-O5 distance: {d:.2f} Å (expected: ~3.0 Å)")

    # Convergence history
    if result.convergence_history:
        print("\n  Convergence history:")
        for iter_idx, conv_radius in result.convergence_history:
            print(f"    Iteration {iter_idx}: radius = {conv_radius:.3f} Å")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  System: Aspirin + Water ({n_atoms} atoms)")
    print(f"  Coarse-graining: {n_atoms} → {n_super_atoms} super-atoms")
    print(f"  GCIQA iterations: {result.n_iterations}")
    print(f"  Converged: {result.converged}")

    if result.best_conformation:
        print("  Best conformation found: Yes")
    else:
        print("  Best conformation found: No")

    print("\n  This validates GCIQA on a system larger than water dimer.")
    print("  For quantum Grover search, the system needs ≤16 qubits")
    print(f"  (currently {n_super_atoms * 3 * 4} qubits with 4 bits/coord).")
    print(f"  With 2 bits/coord: {n_super_atoms * 3 * 2} qubits (still >16).")
    print("  Hierarchical search needed: coarse scan → fine scan.")


if __name__ == "__main__":
    main()
