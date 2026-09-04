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

import math
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from gciqa import (
    GCIQA,
    ConstraintSet,
    GeometricConstraint,
    generate_report,
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


def _hybrid_coarse_grain_covalent(atoms, coords, reactive_idx, ligand_indices):
    """Coarse-grain preserving reactive site as separate super-atoms.

    Strategy:
    - Reactive atom → its own super-atom
    - Each ligand atom → its own super-atom
    - All background → 1 super-atom
    """
    from gciqa.coarsegrain import _ATOMIC_MASSES, _build_cg_from_groups

    n = len(atoms)
    masses = [_ATOMIC_MASSES.get(a, 12.0) for a in atoms]

    groups = {}
    group_idx = 0
    used = set()

    # Reactive atom → own group
    groups[group_idx] = [reactive_idx]
    used.add(reactive_idx)
    group_idx += 1

    # Each ligand atom → own group
    for idx in ligand_indices:
        groups[group_idx] = [idx]
        used.add(idx)
        group_idx += 1

    # Background → 1 group
    bg = [i for i in range(n) if i not in used]
    if bg:
        groups[group_idx] = bg

    return _build_cg_from_groups(atoms, coords, masses, groups)


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

        # Hybrid coarse-grain: preserve reactive site
        ligand_indices = list(range(site["ligand_start"], site["ligand_start"] + site["ligand_atoms"]))
        cg = _hybrid_coarse_grain_covalent(
            site["atoms"], site["coords"],
            site["reactive_idx"], ligand_indices,
        )

        if verbose:
            print(f"  Coarse-graining: {len(site['atoms'])} → {cg.n_super_atoms} super-atoms")
            print(f"  Reactive → SA{cg.atom_to_super[site['reactive_idx']]}")
            print(f"  Ligand[0] → SA{cg.atom_to_super[ligand_indices[0]]}")

        # Map constraints to super-atom level
        reactive_super = cg.atom_to_super[site["reactive_idx"]]
        ligand_super = cg.atom_to_super[ligand_indices[0]]

        super_constraints = []
        if reactive_super != ligand_super:
            # Covalent bond ~1.8Å, use tight range with buffer
            super_constraints.append(
                GeometricConstraint.bond(
                    str(reactive_super), str(ligand_super),
                    min_dist=1.3, max_dist=2.3,
                )
            )
            # Add bond to adjacent ligand atom to constrain direction
            if len(ligand_indices) > 1:
                adj_super = cg.atom_to_super[ligand_indices[1]]
                if adj_super != ligand_super and adj_super != reactive_super:
                    super_constraints.append(
                        GeometricConstraint.bond(
                            str(ligand_super), str(adj_super),
                            min_dist=1.0, max_dist=2.0,
                        )
                    )
        super_constraints.append(
            GeometricConstraint.pocket(
                center=site["coords"][site["reactive_idx"]],
                radius=2.0,
            )
        )

        gciqa = GCIQA(
            n_super_atoms=cg.n_super_atoms,
            constraints=ConstraintSet(super_constraints),
            coord_range=(-10.0, 10.0),
            bits_per_coord=3,
            convergence_threshold=5.0,  # Relax for covalent binding
            use_quantum=False,
        )

        result = gciqa.run(max_iterations=3, n_shots=1000, n_clusters=3)

        if result.best_conformation:
            report = generate_report(result.best_conformation, ConstraintSet(super_constraints))

            # Check RMSD: predicted vs true ligand position
            ligand_key = str(ligand_super)
            if ligand_key in result.best_conformation:
                predicted = result.best_conformation[ligand_key]
                true_pos = site["true_ligand_positions"][0]
                rmsd = math.sqrt(sum((a - b) ** 2 for a, b in zip(predicted, true_pos)))
                success = rmsd < 2.0  # 2.0 Å threshold for covalent
            else:
                rmsd = float('inf')
                success = False

            if verbose:
                print(f"  Converged: {result.converged}")
                print(f"  Constraint score: {report.overall_score:.2f}")
                print(f"  RMSD: {rmsd:.2f} Å  Success: {success}")
            results.append({
                "target": target,
                "success": success,
                "rmsd": rmsd,
                "converged": result.converged,
                "score": report.overall_score,
                "time": result.total_time,
            })
        else:
            if verbose:
                print("  No valid conformation found")
            results.append({"target": target, "success": False, "converged": False, "score": 0.0, "time": 0})

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for r in results:
        status = "OK" if r.get("success") else "FAIL"
        rmsd = r.get("rmsd", float('inf'))
        print(f"  {r['target']:6s}  {status:4s}  RMSD={rmsd:.2f}Å  score={r['score']:.2f}  time={r['time']:.2f}s")

    return results


if __name__ == "__main__":
    run_covalent_benchmark()
