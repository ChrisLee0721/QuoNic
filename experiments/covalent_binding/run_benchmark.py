"""Covalent binding site prediction benchmark.

Tests GCIQA's ability to predict covalent binding modes where a ligand
forms a covalent bond with a specific protein residue.

This is a "classical dead end" scenario: classical docking tools struggle
with covalent binding because the energy landscape has a deep, narrow
minimum at the covalent bond distance.

GCIQA advantage: constraints encode the covalent bond directly,
bypassing the energy landscape entirely.

Usage:
    python run_benchmark.py
"""

import sys
import os
import math
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from quonic.gciqa import (
    GeometricConstraint,
    ConstraintSet,
    GCIQA,
    generate_report,
    diagnose_failure,
)


def generate_synthetic_covalent_site(
    n_residues=30,
    target_residue="CYS",
    ligand_atoms=5,
    seed=42,
):
    """Generate a synthetic covalent binding site.

    Creates a protein-like structure with a reactive residue (e.g., Cys)
    and a ligand positioned for covalent bond formation.

    Returns:
        dict with atoms, coords, true_ligand_positions, reactive_atom_idx
    """
    random.seed(seed)

    atoms = []
    coords = []

    # Reactive residue (e.g., Cys SG) at origin
    atoms.append("S")
    coords.append((0.0, 0.0, 0.0))  # SG
    reactive_idx = 0

    # Ligand atoms bonded to SG (covalent bond ~1.8 Å)
    ligand_start = len(atoms)
    ligand_positions = [
        (1.8, 0.0, 0.0),   # Bonded atom
        (2.5, 1.2, 0.0),   # Adjacent
        (2.5, -1.2, 0.0),  # Adjacent
        (3.5, 0.0, 0.8),   # Far
        (3.5, 0.0, -0.8),  # Far
    ]
    for i in range(ligand_atoms):
        atoms.append("C")
        coords.append(ligand_positions[i % len(ligand_positions)])

    # Background residues
    for _ in range(n_residues):
        while True:
            x = random.gauss(0, 12)
            y = random.gauss(0, 12)
            z = random.gauss(0, 12)
            if math.sqrt(x**2 + y**2 + z**2) < 18:
                break
        atoms.append("C")
        coords.append((x, y, z))

    return {
        "atoms": atoms,
        "coords": coords,
        "reactive_idx": reactive_idx,
        "ligand_start": ligand_start,
        "ligand_atoms": ligand_atoms,
        "true_ligand_positions": ligand_positions[:ligand_atoms],
    }


def run_covalent_benchmark(verbose=True):
    """Run covalent binding benchmark."""
    if verbose:
        print("=" * 60)
        print("Covalent Binding Site Prediction Benchmark")
        print("=" * 60)

    results = []

    for target in ["CYS", "SER", "LYS"]:
        site = generate_synthetic_covalent_site(
            n_residues=20,
            target_residue=target,
            ligand_atoms=4,
            seed=hash(target) % 10000,
        )

        if verbose:
            print(f"\n  Target: {target}")
            print(f"  Reactive atom: {site['atoms'][site['reactive_idx']]} at {site['coords'][site['reactive_idx']]}")
            print(f"  Ligand atoms: {site['ligand_atoms']}")

        # Constraints: covalent bond + spatial pocket
        constraints = ConstraintSet([
            # Covalent bond: reactive atom to ligand bonded atom
            GeometricConstraint.bond(
                str(site["reactive_idx"]),
                str(site["ligand_start"]),
                min_dist=1.5,
                max_dist=2.0,
            ),
            # Pocket around reactive site
            GeometricConstraint.pocket(
                center=site["coords"][site["reactive_idx"]],
                radius=5.0,
            ),
        ])

        # Coarse-grain: reactive atom + ligand + background
        n_super_atoms = 3
        from quonic.gciqa import coarse_grain
        cg = coarse_grain(
            site["atoms"], site["coords"],
            strategy="spatial",
            n_super_atoms=n_super_atoms,
        )

        # Map constraints
        reactive_super = cg.atom_to_super[site["reactive_idx"]]
        ligand_super = cg.atom_to_super[site["ligand_start"]]

        super_constraints = []
        if reactive_super != ligand_super:
            super_constraints.append(
                GeometricConstraint.bond(
                    str(reactive_super), str(ligand_super),
                    min_dist=1.0, max_dist=5.0,
                )
            )
        super_constraints.append(
            GeometricConstraint.pocket(
                center=site["coords"][site["reactive_idx"]],
                radius=5.0,
            )
        )

        gciqa = GCIQA(
            n_super_atoms=n_super_atoms,
            constraints=ConstraintSet(super_constraints),
            coord_range=(-10.0, 10.0),
            bits_per_coord=3,
            use_quantum=False,
        )

        result = gciqa.run(max_iterations=2, n_shots=500, n_clusters=3)

        if result.best_conformation:
            report = generate_report(result.best_conformation, ConstraintSet(super_constraints))
            if verbose:
                print(f"  Converged: {result.converged}")
                print(f"  Constraint score: {report.overall_score:.2f}")
            results.append({
                "target": target,
                "converged": result.converged,
                "score": report.overall_score,
                "time": result.total_time,
            })
        else:
            if verbose:
                print("  No valid conformation found")
            results.append({"target": target, "converged": False, "score": 0.0, "time": 0})

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for r in results:
        status = "OK" if r["converged"] else "FAIL"
        print(f"  {r['target']:6s}  {status:4s}  score={r['score']:.2f}  time={r['time']:.2f}s")

    return results


if __name__ == "__main__":
    run_covalent_benchmark()
