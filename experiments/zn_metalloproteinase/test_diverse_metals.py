"""Test distance geometry on diverse metalloproteins.

Tests 10+ proteins with different:
- Metals: Zn, Fe, Cu, Mg, Ca, Mn
- Coordination geometries: tetrahedral, octahedral, square planar
- Sizes: small (< 100 residues) to large (> 300 residues)
"""

import sys
import os
import math
import random
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from gciqa import (
    parse_pdb,
    find_metal_ions,
    auto_detect_geometry,
    get_metal_template,
    generate_metal_constraints,
)
from gciqa.coarsegrain import _ATOMIC_MASSES, _build_cg_from_groups


def download_pdb(pdb_id, dest_dir):
    """Download PDB file if not already present."""
    path = os.path.join(dest_dir, f"{pdb_id}.pdb")
    if os.path.exists(path):
        return path
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    try:
        urllib.request.urlretrieve(url, path)
        return path
    except Exception as e:
        print(f"  WARNING: Failed to download {pdb_id}: {e}")
        return None


def find_metal_by_distance_geometry(
    ligand_coords, distance_ranges, n_samples=100000
):
    """Find metal position by dense sampling in coordination sphere intersection."""
    if not ligand_coords:
        return (0, 0, 0), 0

    center1 = ligand_coords[0]
    dmin1, dmax1 = distance_ranges[0]

    best_pos = center1
    best_score = -1
    best_violation = float("inf")

    for _ in range(n_samples):
        theta = random.uniform(0, 2 * math.pi)
        phi = math.acos(2 * random.random() - 1)
        r = random.uniform(dmin1, dmax1)

        x = center1[0] + r * math.sin(phi) * math.cos(theta)
        y = center1[1] + r * math.sin(phi) * math.sin(theta)
        z = center1[2] + r * math.cos(phi)

        satisfied = 1
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


def test_protein(pdb_path, name, metal_element="ZN"):
    """Test distance geometry on a single protein."""
    print(f"\n  --- {name} ---")

    protein = parse_pdb(pdb_path)
    metal_ions = find_metal_ions(protein, metal_element)

    if not metal_ions:
        # Try all metals
        metal_ions = protein.metal_ions
        if not metal_ions:
            print(f"    ERROR: No metal ions found")
            return None
        metal_element = metal_ions[0].element
        print(f"    Note: Using {metal_element} instead of {metal_element}")

    # Test each metal ion
    results = []
    for zn in metal_ions:
        true_pos = zn.coord
        geometry = auto_detect_geometry(zn, protein, max_dist=2.5)
        try:
            template = get_metal_template(zn.element, geometry)
        except ValueError:
            print(f"    {zn.element}: unsupported metal, skipping")
            results.append({"element": zn.element, "rmsd": float("inf"), "ligands": 0})
            continue
        atom_constraints = generate_metal_constraints(zn, protein, template, max_dist=2.5)

        # Extract ligand positions and distance ranges
        ligand_coords = []
        distance_ranges = []
        for c in atom_constraints.constraints:
            if c.type.value == "bond":
                a1, a2 = int(c.atoms[0]), int(c.atoms[1])
                if a1 == zn.index:
                    ligand_idx = a2
                elif a2 == zn.index:
                    ligand_idx = a1
                else:
                    continue
                ligand_coords.append(protein.coords[ligand_idx])
                distance_ranges.append((c.params["min_dist"], c.params["max_dist"]))

        if len(ligand_coords) < 2:
            print(f"    {zn.element} at ({true_pos[0]:.1f}, {true_pos[1]:.1f}, {true_pos[2]:.1f}): "
                  f"only {len(ligand_coords)} ligands, need >= 2")
            results.append({"element": zn.element, "rmsd": float("inf"), "ligands": len(ligand_coords)})
            continue

        # Find metal by distance geometry
        pred_pos, score = find_metal_by_distance_geometry(
            ligand_coords, distance_ranges, n_samples=50000
        )
        rmsd = math.sqrt(sum((a-b)**2 for a, b in zip(pred_pos, true_pos)))

        print(f"    {zn.element} at ({true_pos[0]:.1f}, {true_pos[1]:.1f}, {true_pos[2]:.1f}): "
              f"geometry={geometry}, ligands={len(ligand_coords)}, "
              f"DG RMSD={rmsd:.2f}A, constraints={score}/{len(ligand_coords)}")

        results.append({
            "element": zn.element,
            "rmsd": rmsd,
            "ligands": len(ligand_coords),
            "score": score,
            "geometry": geometry,
        })

    return results


def main():
    print("=" * 60)
    print("Distance Geometry Test: Diverse Metalloproteins")
    print("=" * 60)

    data_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(data_dir, exist_ok=True)

    # Diverse metalloproteins
    proteins = [
        # Zinc proteins
        ("1ZNF", "Zinc Finger (Zn, tetrahedral, 28 res)", "ZN"),
        ("1CA2", "Carbonic Anhydrase II (Zn, tetrahedral, 260 res)", "ZN"),
        ("1LND", "Thermolysin (Zn, tetrahedral, 316 res)", "ZN"),
        # Iron proteins
        ("1MBN", "Myoglobin (Fe, octahedral, 153 res)", "FE"),
        ("2HHB", "Hemoglobin (Fe, octahedral, 574 res)", "FE"),
        ("1FHA", "Ferredoxin (Fe, tetrahedral, 54 res)", "FE"),
        # Copper proteins
        ("1AZU", "Azurin (Cu, tetrahedral, 128 res)", "CU"),
        ("1PCY", "Plastocyanin (Cu, tetrahedral, 99 res)", "CU"),
        # Magnesium proteins
        ("1AKE", "Adenylate Kinase (Mg, octahedral, 214 res)", "MG"),
        # Calcium proteins
        ("1CDP", "Calmodulin (Ca, octahedral, 148 res)", "CA"),
        # Manganese proteins
        ("1SOD", "Superoxide Dismutase (Mn, octahedral, 151 res)", "MN"),
        # Cobalt protein
        ("1CCM", "Cytochrome c (Co, octahedral, 103 res)", "CO"),
    ]

    all_results = []
    for pdb_id, name, metal in proteins:
        pdb_path = download_pdb(pdb_id, data_dir)
        if pdb_path is None:
            print(f"\n  --- {name} ---")
            print(f"    SKIPPED: download failed")
            continue

        results = test_protein(pdb_path, name, metal)
        if results:
            for r in results:
                r["pdb_id"] = pdb_id
                r["name"] = name
            all_results.extend(results)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    successful = [r for r in all_results if r["rmsd"] < 2.0]
    failed = [r for r in all_results if r["rmsd"] >= 2.0 and r["rmsd"] < float("inf")]
    skipped = [r for r in all_results if r["rmsd"] == float("inf")]

    print(f"\n  Total metals tested: {len(all_results)}")
    print(f"  Successful (RMSD < 2.0A): {len(successful)}")
    print(f"  Failed: {len(failed)}")
    print(f"  Skipped (too few ligands): {len(skipped)}")

    if successful:
        rmsds = [r["rmsd"] for r in successful]
        print(f"\n  RMSD statistics (successful):")
        print(f"    Mean: {sum(rmsds)/len(rmsds):.2f} A")
        print(f"    Min: {min(rmsds):.2f} A")
        print(f"    Max: {max(rmsds):.2f} A")

    print(f"\n  Per-metal results:")
    for r in all_results:
        if r["rmsd"] == float("inf"):
            status = "SKIP"
        elif r["rmsd"] < 2.0:
            status = "OK"
        else:
            status = "FAIL"
        print(f"    {r['pdb_id']:5s} {r['element']:2s} {r.get('geometry','?'):20s} "
              f"ligands={r['ligands']} RMSD={r['rmsd']:.2f}A {status}")


if __name__ == "__main__":
    main()
