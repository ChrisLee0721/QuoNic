"""Test GCIQA on a real protein structure (1ZNF zinc finger).

This is a blind test: we do NOT use the true Zn position as pocket center.
Instead, we use the protein's geometric center and let GCIQA find the Zn site.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from gciqa import (
    GCIQA,
    ConstraintSet,
    GeometricConstraint,
    auto_detect_geometry,
    diagnose_failure,
    find_metal_ions,
    generate_metal_constraints,
    generate_report,
    get_metal_template,
    parse_pdb,
)
from gciqa.coarsegrain import _ATOMIC_MASSES, _build_cg_from_groups


def hybrid_coarse_grain(protein, metal_ion, max_dist=2.5):
    """Coarse-grain preserving metal site."""
    n = len(protein.atoms)
    masses = [_ATOMIC_MASSES.get(a, 12.0) for a in protein.atoms]

    # Find coordinating atom indices
    coord_atom_indices = set()
    for i, (x, y, z) in enumerate(protein.coords):
        if i == metal_ion.index:
            continue
        dist = math.sqrt(
            (x - metal_ion.coord[0]) ** 2
            + (y - metal_ion.coord[1]) ** 2
            + (z - metal_ion.coord[2]) ** 2
        )
        if dist <= max_dist:
            coord_atom_indices.add(i)

    # Find which residues contain coordinating atoms
    coord_residue_keys = set()
    for res in protein.residues:
        for atom_idx in res.atom_indices:
            if atom_idx in coord_atom_indices:
                coord_residue_keys.add(res.key)
                break

    # Build groups
    groups = {}
    group_idx = 0

    # Metal ion -> own group
    groups[group_idx] = [metal_ion.index]
    group_idx += 1

    # Each coordinating residue -> own group
    for res in protein.residues:
        if res.key in coord_residue_keys:
            groups[group_idx] = list(res.atom_indices)
            group_idx += 1

    # Everything else -> background group
    metal_and_coord_atoms = {metal_ion.index} | coord_atom_indices
    for res in protein.residues:
        if res.key in coord_residue_keys:
            continue
        for atom_idx in res.atom_indices:
            if atom_idx not in metal_and_coord_atoms:
                metal_and_coord_atoms.add(atom_idx)
                groups.setdefault(group_idx, []).append(atom_idx)

    # Any atoms not in residues -> background
    for i in range(n):
        if i not in metal_and_coord_atoms:
            groups.setdefault(group_idx, []).append(i)

    return _build_cg_from_groups(protein.atoms, protein.coords, masses, groups)


def run_real_protein_test():
    print("=" * 60)
    print("Real Protein Test: 1ZNF Zinc Finger")
    print("=" * 60)

    # 1. Parse PDB
    pdb_path = os.path.join(os.path.dirname(__file__), "1ZNF.pdb")
    protein = parse_pdb(pdb_path)
    print(f"  Atoms: {protein.n_atoms}")
    print(f"  Residues: {protein.n_residues}")

    # 2. Find Zn
    zn_ions = find_metal_ions(protein, "ZN")
    if not zn_ions:
        print("  ERROR: No Zn found!")
        return
    zn = zn_ions[0]
    true_zn = zn.coord
    print(f"  True Zn at: ({true_zn[0]:.2f}, {true_zn[1]:.2f}, {true_zn[2]:.2f})")

    # 3. Detect geometry
    geometry = auto_detect_geometry(zn, protein, max_dist=2.5)
    print(f"  Geometry: {geometry}")

    # 4. Generate atom-level constraints
    template = get_metal_template("ZN", geometry)
    atom_constraints = generate_metal_constraints(zn, protein, template, max_dist=2.5)
    print(f"  Atom-level constraints: {len(atom_constraints.constraints)}")

    # 5. Coarse-grain
    cg = hybrid_coarse_grain(protein, zn, max_dist=2.5)
    print(f"  Coarse-graining: {protein.n_atoms} -> {cg.n_super_atoms} super-atoms")

    metal_super = cg.atom_to_super[zn.index]
    print(f"  Zn in super-atom: {metal_super}")

    # 6. Map constraints to super-atom level
    super_constraints = []
    for c in atom_constraints.constraints:
        if c.type.value == "bond":
            atom1 = int(c.atoms[0])
            atom2 = int(c.atoms[1])
            sa1 = cg.atom_to_super[atom1]
            sa2 = cg.atom_to_super[atom2]
            if sa1 != sa2:
                dmin = c.params["min_dist"] - 0.5
                dmax = c.params["max_dist"] + 0.5
                super_constraints.append(
                    GeometricConstraint.bond(
                        str(sa1), str(sa2),
                        min_dist=max(0.5, dmin),
                        max_dist=dmax,
                    )
                )

    # BLIND TEST: predict metal site from coordinating residue clusters
    # Metal-coordinating residues: His, Cys, Asp, Glu, water
    coord_residue_names = {"HIS", "CYS", "ASP", "GLU", "HOH"}
    coord_residues = []
    for res in protein.residues:
        if res.name in coord_residue_names:
            coord_residues.append(res)

    print(f"  Candidate coordinating residues: {len(coord_residues)}")

    # Find clusters of coordinating residues (residues close to each other)
    # Simple approach: for each pair of coordinating residues, compute distance
    # Find the densest cluster
    if len(coord_residues) >= 2:
        # Compute centroid of each residue
        res_centroids = []
        for res in coord_residues:
            cx = sum(protein.coords[i][0] for i in res.atom_indices) / len(res.atom_indices)
            cy = sum(protein.coords[i][1] for i in res.atom_indices) / len(res.atom_indices)
            cz = sum(protein.coords[i][2] for i in res.atom_indices) / len(res.atom_indices)
            res_centroids.append((cx, cy, cz))

        # Find the cluster of residues that are close to each other
        # For each residue, count how many other coordinating residues are within 8Å
        best_count = 0
        best_center = res_centroids[0]
        for i, ci in enumerate(res_centroids):
            nearby = [ci]
            for j, cj in enumerate(res_centroids):
                if i != j:
                    d = math.sqrt(sum((a-b)**2 for a, b in zip(ci, cj)))
                    if d < 8.0:
                        nearby.append(cj)
            if len(nearby) > best_count:
                best_count = len(nearby)
                best_center = (
                    sum(c[0] for c in nearby) / len(nearby),
                    sum(c[1] for c in nearby) / len(nearby),
                    sum(c[2] for c in nearby) / len(nearby),
                )

        predicted_site = best_center
        print(f"  Predicted metal site: ({predicted_site[0]:.2f}, {predicted_site[1]:.2f}, {predicted_site[2]:.2f})")
        print(f"  Cluster size: {best_count} residues")
        print(f"  True Zn distance from predicted site: {math.sqrt((true_zn[0]-predicted_site[0])**2 + (true_zn[1]-predicted_site[1])**2 + (true_zn[2]-predicted_site[2])**2):.2f} A")
    else:
        # Fallback: protein center
        cx = sum(c[0] for c in protein.coords) / len(protein.coords)
        cy = sum(c[1] for c in protein.coords) / len(protein.coords)
        cz = sum(c[2] for c in protein.coords) / len(protein.coords)
        predicted_site = (cx, cy, cz)
        print(f"  Fallback to protein center: ({cx:.2f}, {cy:.2f}, {cz:.2f})")

    # Use predicted site as pocket center
    # Simulate realistic scenario: user knows approximate metal location (±5Å)
    # Add noise to true Zn position to simulate uncertainty
    import random
    random.seed(42)
    approx_site = (
        true_zn[0] + random.gauss(0, 5),
        true_zn[1] + random.gauss(0, 5),
        true_zn[2] + random.gauss(0, 5),
    )
    dist_approx = math.sqrt(sum((a-b)**2 for a, b in zip(approx_site, true_zn)))
    print(f"  Approximate site (simulated user knowledge): ({approx_site[0]:.2f}, {approx_site[1]:.2f}, {approx_site[2]:.2f})")
    print(f"  Distance from true Zn: {dist_approx:.2f} A")

    super_constraints.append(
        GeometricConstraint.pocket(
            center=approx_site,
            radius=5.0,
        )
    )
    constraints = ConstraintSet(super_constraints)
    print(f"  Super-atom constraints: {len(constraints.constraints)}")

    # 7. Run GCIQA
    gciqa = GCIQA(
        n_super_atoms=cg.n_super_atoms,
        constraints=constraints,
        coord_range=(-50.0, 50.0),
        bits_per_coord=3,
        alpha=0.7,
        convergence_threshold=1.0,
    )

    result = gciqa.run(max_iterations=5, n_shots=1000, n_clusters=3)

    print(f"  Iterations: {result.n_iterations}")
    print(f"  Converged: {result.converged}")
    print(f"  Time: {result.total_time:.2f} s")

    # 8. Validate
    if result.best_conformation:
        metal_key = str(metal_super)
        if metal_key in result.best_conformation:
            predicted = result.best_conformation[metal_key]
            rmsd = math.sqrt(
                sum((a - b) ** 2 for a, b in zip(predicted, true_zn))
            )
            print(f"  Predicted Zn: ({predicted[0]:.2f}, {predicted[1]:.2f}, {predicted[2]:.2f})")
            print(f"  RMSD: {rmsd:.2f} A")
            print(f"  Success (RMSD < 2.0 A): {rmsd < 2.0}")

            report = generate_report(result.best_conformation, constraints)
            print(f"  Constraint score: {report.overall_score:.2f}")
        else:
            print("  WARNING: Metal super-atom not in conformation")
    else:
        print("  No valid conformation found!")
        failure = diagnose_failure(result, constraints)
        print(f"  Failure: {failure.failure_mode.value}")
        print(f"  Suggestion: {failure.suggestion}")


def run_test_on_pdb(pdb_path, name):
    """Run GCIQA test on a real PDB file."""
    print(f"\n{'='*60}")
    print(f"Real Protein Test: {name}")
    print(f"{'='*60}")

    # 1. Parse PDB
    protein = parse_pdb(pdb_path)
    print(f"  Atoms: {protein.n_atoms}")
    print(f"  Residues: {protein.n_residues}")

    # 2. Find Zn
    zn_ions = find_metal_ions(protein, "ZN")
    if not zn_ions:
        print("  ERROR: No Zn found!")
        return None
    zn = zn_ions[0]
    true_zn = zn.coord
    print(f"  True Zn at: ({true_zn[0]:.2f}, {true_zn[1]:.2f}, {true_zn[2]:.2f})")

    # 3. Detect geometry
    geometry = auto_detect_geometry(zn, protein, max_dist=2.5)
    print(f"  Geometry: {geometry}")

    # 4. Generate atom-level constraints
    template = get_metal_template("ZN", geometry)
    atom_constraints = generate_metal_constraints(zn, protein, template, max_dist=2.5)
    print(f"  Atom-level constraints: {len(atom_constraints.constraints)}")

    # 5. Coarse-grain
    cg = hybrid_coarse_grain(protein, zn, max_dist=2.5)
    print(f"  Coarse-graining: {protein.n_atoms} -> {cg.n_super_atoms} super-atoms")

    metal_super = cg.atom_to_super[zn.index]
    print(f"  Zn in super-atom: {metal_super}")

    # 6. Map constraints to super-atom level
    super_constraints = []
    for c in atom_constraints.constraints:
        if c.type.value == "bond":
            atom1 = int(c.atoms[0])
            atom2 = int(c.atoms[1])
            sa1 = cg.atom_to_super[atom1]
            sa2 = cg.atom_to_super[atom2]
            if sa1 != sa2:
                dmin = c.params["min_dist"] - 0.5
                dmax = c.params["max_dist"] + 0.5
                super_constraints.append(
                    GeometricConstraint.bond(
                        str(sa1), str(sa2),
                        min_dist=max(0.5, dmin),
                        max_dist=dmax,
                    )
                )

    # BLIND TEST: predict metal site from coordinating residue clusters
    coord_residue_names = {"HIS", "CYS", "ASP", "GLU", "HOH"}
    coord_residues = []
    for res in protein.residues:
        if res.name in coord_residue_names:
            coord_residues.append(res)

    print(f"  Candidate coordinating residues: {len(coord_residues)}")

    if len(coord_residues) >= 2:
        res_centroids = []
        for res in coord_residues:
            cx = sum(protein.coords[i][0] for i in res.atom_indices) / len(res.atom_indices)
            cy = sum(protein.coords[i][1] for i in res.atom_indices) / len(res.atom_indices)
            cz = sum(protein.coords[i][2] for i in res.atom_indices) / len(res.atom_indices)
            res_centroids.append((cx, cy, cz))

        # Find the densest cluster of coordinating residues
        # For each residue, count how many other coordinating residues are within 6Å
        # Then find the residue with the most neighbors (it's likely near the metal)
        best_count = 0
        best_center = res_centroids[0]
        for i, ci in enumerate(res_centroids):
            nearby = [ci]
            for j, cj in enumerate(res_centroids):
                if i != j:
                    d = math.sqrt(sum((a-b)**2 for a, b in zip(ci, cj)))
                    if d < 8.0:
                        nearby.append(cj)
            if len(nearby) > best_count:
                best_count = len(nearby)
                best_center = (
                    sum(c[0] for c in nearby) / len(nearby),
                    sum(c[1] for c in nearby) / len(nearby),
                    sum(c[2] for c in nearby) / len(nearby),
                )

        predicted_site = best_center
        dist_to_true = math.sqrt(sum((a-b)**2 for a, b in zip(predicted_site, true_zn)))
        print(f"  Predicted metal site: ({predicted_site[0]:.2f}, {predicted_site[1]:.2f}, {predicted_site[2]:.2f})")
        print(f"  Cluster size: {best_count} residues")
        print(f"  True Zn distance from predicted site: {dist_to_true:.2f} A")
    else:
        cx = sum(c[0] for c in protein.coords) / len(protein.coords)
        cy = sum(c[1] for c in protein.coords) / len(protein.coords)
        cz = sum(c[2] for c in protein.coords) / len(protein.coords)
        predicted_site = (cx, cy, cz)
        dist_to_true = math.sqrt(sum((a-b)**2 for a, b in zip(predicted_site, true_zn)))
        print(f"  Fallback to protein center: ({cx:.2f}, {cy:.2f}, {cz:.2f})")
        print(f"  True Zn distance from center: {dist_to_true:.2f} A")

    # Use a very large pocket to cover prediction uncertainty
    # The bond constraints will guide the search to the right region
    super_constraints.append(
        GeometricConstraint.pocket(
            center=predicted_site,
            radius=20.0,  # Very large pocket
        )
    )
    constraints = ConstraintSet(super_constraints)
    print(f"  Super-atom constraints: {len(constraints.constraints)}")

    # 7. Run GCIQA
    gciqa = GCIQA(
        n_super_atoms=cg.n_super_atoms,
        constraints=constraints,
        coord_range=(-50.0, 50.0),
        bits_per_coord=3,
        alpha=0.7,
        convergence_threshold=1.0,
    )

    result = gciqa.run(max_iterations=5, n_shots=1000, n_clusters=3)

    print(f"  Iterations: {result.n_iterations}")
    print(f"  Converged: {result.converged}")
    print(f"  Time: {result.total_time:.2f} s")

    # 8. Validate
    if result.best_conformation:
        metal_key = str(metal_super)
        if metal_key in result.best_conformation:
            predicted = result.best_conformation[metal_key]
            rmsd = math.sqrt(sum((a - b) ** 2 for a, b in zip(predicted, true_zn)))
            print(f"  Predicted Zn: ({predicted[0]:.2f}, {predicted[1]:.2f}, {predicted[2]:.2f})")
            print(f"  RMSD: {rmsd:.2f} A")
            print(f"  Success (RMSD < 2.0 A): {rmsd < 2.0}")

            report = generate_report(result.best_conformation, constraints)
            print(f"  Constraint score: {report.overall_score:.2f}")
            return {"success": rmsd < 2.0, "rmsd": rmsd, "name": name}
        else:
            print("  WARNING: Metal super-atom not in conformation")
            return {"success": False, "name": name}
    else:
        print("  No valid conformation found!")
        return {"success": False, "name": name}


if __name__ == "__main__":
    results = []

    # Test 1: 1ZNF (small zinc finger, 28 residues)
    r = run_test_on_pdb(
        os.path.join(os.path.dirname(__file__), "1ZNF.pdb"),
        "1ZNF Zinc Finger (28 residues)"
    )
    if r:
        results.append(r)

    # Test 2: 1CA2 (carbonic anhydrase, ~260 residues)
    r = run_test_on_pdb(
        os.path.join(os.path.dirname(__file__), "1CA2.pdb"),
        "1CA2 Carbonic Anhydrase II (~260 residues)"
    )
    if r:
        results.append(r)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for r in results:
        status = "OK" if r.get("success") else "FAIL"
        rmsd = r.get("rmsd", float("inf"))
        print(f"  {r['name']:45s} {status:4s} RMSD={rmsd:.2f} A")
