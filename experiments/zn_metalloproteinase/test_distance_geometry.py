"""Test distance geometry approach for metal position prediction.

Instead of using a pocket center to guide search, we compute the metal
position directly from the intersection of coordination spheres.

For each coordinating atom, the metal must be within [dmin, dmax] distance.
The intersection of all these spheres gives the metal position.
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
    auto_detect_geometry,
    find_metal_ions,
    generate_metal_constraints,
    get_metal_template,
    parse_pdb,
)
from gciqa.coarsegrain import _ATOMIC_MASSES, _build_cg_from_groups


def find_metal_by_distance_geometry(
    ligand_coords: list[tuple[float, float, float]],
    distance_ranges: list[tuple[float, float]],
    n_samples: int = 10000,
) -> tuple[tuple[float, float, float], float]:
    """Find metal position by sampling intersection of coordination spheres.

    Args:
        ligand_coords: Positions of coordinating atoms.
        distance_ranges: (dmin, dmax) for each ligand.
        n_samples: Number of random samples.

    Returns:
        (best_position, best_score) where score is number of satisfied constraints.
    """
    if not ligand_coords:
        return (0, 0, 0), 0

    # Compute bounding box of possible metal positions
    # For each ligand, the metal is within [dmin, dmax] sphere
    # The intersection is roughly centered between the ligands
    cx = sum(c[0] for c in ligand_coords) / len(ligand_coords)
    cy = sum(c[1] for c in ligand_coords) / len(ligand_coords)
    cz = sum(c[2] for c in ligand_coords) / len(ligand_coords)

    # Estimate search radius: max distance from center to any ligand + max dmax
    max_dmax = max(dmax for _, dmax in distance_ranges)
    search_radius = max(
        math.sqrt((c[0]-cx)**2 + (c[1]-cy)**2 + (c[2]-cz)**2)
        for c in ligand_coords
    ) + max_dmax

    best_pos = (cx, cy, cz)
    best_score = -1
    best_violation = float("inf")

    for _ in range(n_samples):
        # Sample a random point within the search sphere
        # Use rejection sampling
        while True:
            x = cx + random.gauss(0, search_radius / 2)
            y = cy + random.gauss(0, search_radius / 2)
            z = cz + random.gauss(0, search_radius / 2)
            d = math.sqrt((x-cx)**2 + (y-cy)**2 + (z-cz)**2)
            if d <= search_radius:
                break

        # Check how many constraints are satisfied
        satisfied = 0
        total_violation = 0.0
        for ligand, (dmin, dmax) in zip(ligand_coords, distance_ranges):
            dist = math.sqrt(
                (x - ligand[0])**2 + (y - ligand[1])**2 + (z - ligand[2])**2
            )
            if dmin <= dist <= dmax:
                satisfied += 1
            else:
                # Violation: how far from the nearest bound
                if dist < dmin:
                    total_violation += (dmin - dist) ** 2
                else:
                    total_violation += (dist - dmax) ** 2

        # Better if more constraints satisfied, less violation
        if satisfied > best_score or (satisfied == best_score and total_violation < best_violation):
            best_score = satisfied
            best_violation = total_violation
            best_pos = (x, y, z)

    return best_pos, best_score


def find_metal_by_intersection(
    ligand_coords: list[tuple[float, float, float]],
    distance_ranges: list[tuple[float, float]],
    n_samples: int = 100000,
) -> tuple[tuple[float, float, float], float]:
    """Find metal position by dense sampling in the intersection region.

    More efficient: sample within the first sphere, then filter by other constraints.
    """
    if not ligand_coords:
        return (0, 0, 0), 0

    # Use the first ligand's sphere as the primary search region
    center1 = ligand_coords[0]
    dmin1, dmax1 = distance_ranges[0]

    # Mean distance for the first sphere
    (dmin1 + dmax1) / 2

    best_pos = center1
    best_score = -1
    best_violation = float("inf")

    for _ in range(n_samples):
        # Sample on the surface of the first sphere (mean distance)
        # Random direction
        theta = random.uniform(0, 2 * math.pi)
        phi = math.acos(2 * random.random() - 1)
        r = random.uniform(dmin1, dmax1)

        x = center1[0] + r * math.sin(phi) * math.cos(theta)
        y = center1[1] + r * math.sin(phi) * math.sin(theta)
        z = center1[2] + r * math.cos(phi)

        # Check all constraints
        satisfied = 1  # First constraint is satisfied by construction
        total_violation = 0.0
        for ligand, (dmin, dmax) in zip(ligand_coords[1:], distance_ranges[1:]):
            dist = math.sqrt(
                (x - ligand[0])**2 + (y - ligand[1])**2 + (z - ligand[2])**2
            )
            if dmin <= dist <= dmax:
                satisfied += 1
            else:
                if dist < dmin:
                    total_violation += (dmin - dist) ** 2
                else:
                    total_violation += (dist - dmax) ** 2

        if satisfied > best_score or (satisfied == best_score and total_violation < best_violation):
            best_score = satisfied
            best_violation = total_violation
            best_pos = (x, y, z)

    return best_pos, best_score


def run_distance_geometry_test(pdb_path, name):
    """Test distance geometry approach on a real protein."""
    print(f"\n{'='*60}")
    print(f"Distance Geometry Test: {name}")
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

    # 3. Detect geometry and get template
    geometry = auto_detect_geometry(zn, protein, max_dist=2.5)
    print(f"  Geometry: {geometry}")

    template = get_metal_template("ZN", geometry)
    atom_constraints = generate_metal_constraints(zn, protein, template, max_dist=2.5)

    # 4. Extract ligand positions and distance ranges
    ligand_coords = []
    distance_ranges = []
    for c in atom_constraints.constraints:
        if c.type.value == "bond":
            atom1 = int(c.atoms[0])
            atom2 = int(c.atoms[1])
            # One of them is the metal (zn.index)
            if atom1 == zn.index:
                ligand_idx = atom2
            elif atom2 == zn.index:
                ligand_idx = atom1
            else:
                continue
            ligand_coords.append(protein.coords[ligand_idx])
            distance_ranges.append((c.params["min_dist"], c.params["max_dist"]))

    print(f"  Coordinating atoms: {len(ligand_coords)}")
    for i, (coord, (dmin, dmax)) in enumerate(zip(ligand_coords, distance_ranges)):
        print(f"    {i}: ({coord[0]:.2f}, {coord[1]:.2f}, {coord[2]:.2f}) dist=[{dmin:.1f}, {dmax:.1f}]")

    # 5. Find metal by distance geometry
    print("\n  --- Distance Geometry (dense sampling) ---")
    pos, score = find_metal_by_intersection(ligand_coords, distance_ranges, n_samples=100000)
    rmsd = math.sqrt(sum((a-b)**2 for a, b in zip(pos, true_zn)))
    print(f"  Predicted Zn: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
    print(f"  Constraints satisfied: {score}/{len(ligand_coords)}")
    print(f"  RMSD: {rmsd:.2f} A")
    print(f"  Success (RMSD < 2.0 A): {rmsd < 2.0}")

    # 6. Also try with GCIQA using the predicted position as pocket center
    print("\n  --- GCIQA with distance geometry pocket ---")
    cg = hybrid_coarse_grain(protein, zn, max_dist=2.5)
    metal_super = cg.atom_to_super[zn.index]

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

    # Use distance geometry result as pocket center
    super_constraints.append(
        GeometricConstraint.pocket(center=pos, radius=5.0)
    )
    constraints = ConstraintSet(super_constraints)

    gciqa = GCIQA(
        n_super_atoms=cg.n_super_atoms,
        constraints=constraints,
        coord_range=(-50.0, 50.0),
        bits_per_coord=3,
        use_quantum=False,
    )
    result = gciqa.run(max_iterations=5, n_shots=1000, n_clusters=3)

    if result.best_conformation and str(metal_super) in result.best_conformation:
        pred = result.best_conformation[str(metal_super)]
        rmsd_gciqa = math.sqrt(sum((a-b)**2 for a, b in zip(pred, true_zn)))
        print(f"  GCIQA predicted Zn: ({pred[0]:.2f}, {pred[1]:.2f}, {pred[2]:.2f})")
        print(f"  GCIQA RMSD: {rmsd_gciqa:.2f} A")
        print(f"  GCIQA success: {rmsd_gciqa < 2.0}")
    else:
        print("  GCIQA: no valid conformation")

    return {"name": name, "rmsd_dg": rmsd, "score": score}


def hybrid_coarse_grain(protein, metal_ion, max_dist=2.5):
    """Coarse-grain preserving metal site."""
    n = len(protein.atoms)
    masses = [_ATOMIC_MASSES.get(a, 12.0) for a in protein.atoms]

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

    coord_residue_keys = set()
    for res in protein.residues:
        for atom_idx in res.atom_indices:
            if atom_idx in coord_atom_indices:
                coord_residue_keys.add(res.key)
                break

    groups = {}
    group_idx = 0
    groups[group_idx] = [metal_ion.index]
    group_idx += 1

    for res in protein.residues:
        if res.key in coord_residue_keys:
            groups[group_idx] = list(res.atom_indices)
            group_idx += 1

    metal_and_coord_atoms = {metal_ion.index} | coord_atom_indices
    for res in protein.residues:
        if res.key in coord_residue_keys:
            continue
        for atom_idx in res.atom_indices:
            if atom_idx not in metal_and_coord_atoms:
                metal_and_coord_atoms.add(atom_idx)
                groups.setdefault(group_idx, []).append(atom_idx)

    for i in range(n):
        if i not in metal_and_coord_atoms:
            groups.setdefault(group_idx, []).append(i)

    return _build_cg_from_groups(protein.atoms, protein.coords, masses, groups)


if __name__ == "__main__":
    results = []

    # Test 1: 1ZNF (small zinc finger)
    r = run_distance_geometry_test(
        os.path.join(os.path.dirname(__file__), "1ZNF.pdb"),
        "1ZNF Zinc Finger (28 residues)"
    )
    if r:
        results.append(r)

    # Test 2: 1CA2 (carbonic anhydrase)
    r = run_distance_geometry_test(
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
        print(f"  {r['name']:45s} DG RMSD={r['rmsd_dg']:.2f} A  constraints={r['score']}")
