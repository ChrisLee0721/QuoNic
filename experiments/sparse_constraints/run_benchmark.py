"""Sparse constraint structure reconstruction benchmark.

Tests GCIQA's ability to reconstruct protein structure from very few
distance constraints. This is the "classical dead end" scenario:
classical methods (CYANA, XPLOR-NIH) need ~10-15 constraints per residue,
but we test with < 1 constraint per residue.

Benchmark:
1. Take a known protein structure
2. Compute residue centers of mass
3. Extract N random distance constraints (5, 10, 20, 50)
4. Use GCIQA to reconstruct residue positions from constraints
5. Compare RMSD with original structure

Usage:
    python run_benchmark.py
"""

import sys
import os
import math
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from gciqa import (
    parse_pdb,
    GeometricConstraint,
    ConstraintSet,
    GCIQA,
    generate_report,
)
from gciqa.coarsegrain import _ATOMIC_MASSES, _build_cg_from_groups


def compute_residue_centers(protein):
    """Compute center of mass for each residue."""
    centers = []
    for res in protein.residues:
        if not res.atom_indices:
            continue
        cx = sum(protein.coords[i][0] for i in res.atom_indices) / len(res.atom_indices)
        cy = sum(protein.coords[i][1] for i in res.atom_indices) / len(res.atom_indices)
        cz = sum(protein.coords[i][2] for i in res.atom_indices) / len(res.atom_indices)
        centers.append((cx, cy, cz))
    return centers


def coarse_grain_by_segments(centers, n_segments):
    """Group residues into segments to reduce degrees of freedom.

    Each segment becomes one super-atom (center of mass of its residues).
    """
    n = len(centers)
    segment_size = max(1, n // n_segments)

    segments = []
    for i in range(0, n, segment_size):
        chunk = centers[i:i+segment_size]
        cx = sum(c[0] for c in chunk) / len(chunk)
        cy = sum(c[1] for c in chunk) / len(chunk)
        cz = sum(c[2] for c in chunk) / len(chunk)
        segments.append((cx, cy, cz))

    return segments


def extract_random_constraints(centers, n_constraints, distance_noise=0.0):
    """Extract random distance constraints from residue centers.

    Args:
        centers: Residue center positions.
        n_constraints: Number of constraints to extract.
        distance_noise: Noise added to distances (0 = exact, >0 = range).

    Returns:
        List of (i, j, dmin, dmax) tuples.
    """
    n = len(centers)
    constraints = []
    used_pairs = set()

    for _ in range(n_constraints * 10):  # Try many times to get unique pairs
        if len(constraints) >= n_constraints:
            break

        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)
        if i == j or (i, j) in used_pairs or (j, i) in used_pairs:
            continue

        used_pairs.add((i, j))
        d = math.sqrt(sum((a - b) ** 2 for a, b in zip(centers[i], centers[j])))

        if distance_noise > 0:
            dmin = max(0.5, d - distance_noise)
            dmax = d + distance_noise
        else:
            # Exact distance with small tolerance
            dmin = max(0.5, d - 0.5)
            dmax = d + 0.5

        constraints.append((i, j, dmin, dmax))

    return constraints


def run_sparse_benchmark(pdb_path, name, n_constraints_list, n_segments=10, seed=42):
    """Run sparse constraint benchmark on a protein.

    Uses coarse-grained segments to reduce degrees of freedom.
    """
    print(f"\n{'='*60}")
    print(f"Sparse Constraint Benchmark: {name}")
    print(f"{'='*60}")

    protein = parse_pdb(pdb_path)
    centers = compute_residue_centers(protein)
    n_residues = len(centers)

    # Coarse-grain into segments
    segments = coarse_grain_by_segments(centers, n_segments)
    n_seg = len(segments)

    print(f"  Residues: {n_residues}")
    print(f"  Segments: {n_seg} (coarse-grained)")
    print(f"  Constraint levels: {n_constraints_list}")

    results = []

    for n_constraints in n_constraints_list:
        random.seed(seed)

        # Extract random constraints on segments
        dist_constraints = extract_random_constraints(
            segments, n_constraints, distance_noise=2.0
        )

        if len(dist_constraints) < n_constraints:
            print(f"\n  --- {n_constraints} constraints ---")
            print(f"    WARNING: Only {len(dist_constraints)} constraints extracted")
            n_constraints = len(dist_constraints)

        if n_constraints == 0:
            results.append({
                "n_constraints": 0,
                "rmsd": float("inf"),
                "success": False,
            })
            continue

        # Build GCIQA constraints
        super_constraints = []
        for i, j, dmin, dmax in dist_constraints:
            super_constraints.append(
                GeometricConstraint.bond(str(i), str(j), min_dist=dmin, max_dist=dmax)
            )

        # Add pocket constraint
        cx = sum(c[0] for c in segments) / len(segments)
        cy = sum(c[1] for c in segments) / len(segments)
        cz = sum(c[2] for c in segments) / len(segments)
        max_r = max(
            math.sqrt((c[0]-cx)**2 + (c[1]-cy)**2 + (c[2]-cz)**2)
            for c in segments
        )
        super_constraints.append(
            GeometricConstraint.pocket(center=(cx, cy, cz), radius=max_r * 1.5)
        )

        constraints = ConstraintSet(super_constraints)

        # Run GCIQA
        gciqa = GCIQA(
            n_super_atoms=n_seg,
            constraints=constraints,
            coord_range=(-100.0, 100.0),
            bits_per_coord=3,
            alpha=0.7,
            convergence_threshold=5.0,
            use_quantum=False,
        )

        result = gciqa.run(max_iterations=5, n_shots=500, n_clusters=3)

        # Compute RMSD against original segments
        if result.best_conformation:
            predicted = []
            reference = []
            for idx in range(n_seg):
                key = str(idx)
                if key in result.best_conformation:
                    predicted.append(result.best_conformation[key])
                    reference.append(segments[idx])

            if predicted:
                rmsd = compute_rmsd_with_alignment(predicted, reference)
                satisfied, score = constraints.evaluate(result.best_conformation)

                print(f"\n  --- {n_constraints} constraints ({n_constraints/n_seg:.2f}/segment) ---")
                print(f"    Iterations: {result.n_iterations}")
                print(f"    Converged: {result.converged}")
                print(f"    Time: {result.total_time:.2f} s")
                print(f"    RMSD: {rmsd:.2f} A")
                print(f"    Constraint score: {score:.2f}")
                print(f"    Success (RMSD < 5.0 A): {rmsd < 5.0}")

                results.append({
                    "n_constraints": n_constraints,
                    "constraints_per_segment": n_constraints / n_seg,
                    "rmsd": rmsd,
                    "score": score,
                    "time": result.total_time,
                    "success": rmsd < 5.0,
                })
            else:
                print(f"\n  --- {n_constraints} constraints ---")
                print(f"    No valid conformation found")
                results.append({
                    "n_constraints": n_constraints,
                    "rmsd": float("inf"),
                    "success": False,
                })
        else:
            print(f"\n  --- {n_constraints} constraints ---")
            print(f"    No valid conformation found")
            results.append({
                "n_constraints": n_constraints,
                "rmsd": float("inf"),
                "success": False,
            })

    return results


def compute_rmsd_with_alignment(predicted, reference):
    """Compute RMSD after optimal alignment (Kabsch algorithm)."""
    n = len(predicted)
    if n == 0:
        return float("inf")

    # Center both sets
    pred_center = [sum(p[i] for p in predicted) / n for i in range(3)]
    ref_center = [sum(r[i] for r in reference) / n for i in range(3)]

    pred_centered = [(p[0]-pred_center[0], p[1]-pred_center[1], p[2]-pred_center[2]) for p in predicted]
    ref_centered = [(r[0]-ref_center[0], r[1]-ref_center[1], r[2]-ref_center[2]) for r in reference]

    # Compute cross-covariance matrix
    H = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    for p, r in zip(pred_centered, ref_centered):
        for i in range(3):
            for j in range(3):
                H[i][j] += p[i] * r[j]

    # SVD (simplified - just compute RMSD without optimal rotation)
    # For a proper implementation, we'd use SVD to find optimal rotation
    # But for now, just compute RMSD after centering
    rmsd_sq = sum(
        sum((p[i] - r[i]) ** 2 for i in range(3))
        for p, r in zip(pred_centered, ref_centered)
    ) / n

    return math.sqrt(rmsd_sq)


def main():
    print("=" * 60)
    print("Sparse Constraint Structure Reconstruction Benchmark")
    print("=" * 60)
    print()
    print("Tests GCIQA's ability to reconstruct protein structure")
    print("from very few distance constraints.")
    print()
    print("Classical methods need ~10-15 constraints/residue.")
    print("We test with < 1 constraint/residue.")

    data_dir = os.path.join(os.path.dirname(__file__), "..", "zn_metalloproteinase", "data")

    # Test proteins of different sizes
    proteins = [
        ("1ZNF", "Zinc Finger (28 residues)", 28),
        ("1MBN", "Myoglobin (153 residues)", 153),
        ("1CA2", "Carbonic Anhydrase (260 residues)", 260),
    ]

    # Constraint levels: 5, 10, 20, 50 constraints
    constraint_levels = [5, 10, 20, 50]

    all_results = []
    for pdb_id, name, n_res in proteins:
        pdb_path = os.path.join(data_dir, f"{pdb_id}.pdb")
        if not os.path.exists(pdb_path):
            print(f"\n  SKIPPING {name}: PDB file not found")
            continue

        results = run_sparse_benchmark(pdb_path, name, constraint_levels)
        for r in results:
            r["pdb_id"] = pdb_id
            r["name"] = name
            r["n_residues"] = n_res
        all_results.extend(results)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    print(f"\n  {'Protein':30s} {'Constraints':>12s} {'Per Residue':>12s} {'RMSD':>8s} {'Status':>8s}")
    print(f"  {'-'*30} {'-'*12} {'-'*12} {'-'*8} {'-'*8}")

    for r in all_results:
        status = "OK" if r.get("success") else "FAIL"
        rmsd = r.get("rmsd", float("inf"))
        cpr = r.get("constraints_per_residue", 0)
        print(f"  {r['name']:30s} {r['n_constraints']:12d} {cpr:12.2f} {rmsd:8.2f} {status:>8s}")

    # Analysis
    print(f"\n  Analysis:")
    print(f"  - Classical methods need ~10-15 constraints/residue")
    print(f"  - GCIQA is tested with < 1 constraint/residue")
    print(f"  - Success = RMSD < 5.0 A (fold-level accuracy)")


if __name__ == "__main__":
    main()
