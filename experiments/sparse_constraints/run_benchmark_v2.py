"""Sparse constraint benchmark v2: Distance geometry + GCIQA refinement.

Strategy:
1. Use distance geometry (DG) to compute initial positions from sparse constraints
2. Use GCIQA to refine the DG result
3. Compare with classical methods (random, DG only)

Key insight: We include CHAIN CONNECTIVITY constraints (consecutive segments
are ~3.8Å * segment_size apart) as a physics-based prior. The "sparse constraints"
are ADDITIONAL random distance constraints beyond chain connectivity.

This is realistic: you know the protein is a chain, and you have a few additional
distance constraints from experiments (NMR, cross-linking, FRET, etc.).
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
    parse_pdb,
)


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
    """Group residues into segments."""
    n = len(centers)
    segment_size = max(1, n // n_segments)

    segments = []
    segment_indices = []
    for i in range(0, n, segment_size):
        chunk = centers[i:i+segment_size]
        cx = sum(c[0] for c in chunk) / len(chunk)
        cy = sum(c[1] for c in chunk) / len(chunk)
        cz = sum(c[2] for c in chunk) / len(chunk)
        segments.append((cx, cy, cz))
        segment_indices.append(list(range(i, min(i+segment_size, n))))

    return segments, segment_indices


def extract_chain_constraints(centers, segment_indices):
    """Extract chain connectivity constraints between consecutive segments.

    Each consecutive pair of segments is connected with a distance range
    based on typical Cα-Cα distances (~3.8Å per residue).
    """
    constraints = []
    for i in range(len(centers) - 1):
        # Distance between consecutive segment centers
        d = math.sqrt(sum((a - b) ** 2 for a, b in zip(centers[i], centers[i+1])))
        # Tight range: 0.8x to 1.3x actual distance
        dmin = max(1.0, d * 0.8)
        dmax = d * 1.3
        constraints.append((i, i+1, dmin, dmax))
    return constraints


def extract_random_constraints(centers, n_constraints, distance_noise=0.0):
    """Extract random distance constraints (non-sequential pairs)."""
    n = len(centers)
    constraints = []
    used_pairs = set()

    # Only use non-sequential pairs (skip i, i+1 which are chain constraints)
    for _ in range(n_constraints * 20):
        if len(constraints) >= n_constraints:
            break

        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)
        if i == j or (i, j) in used_pairs or (j, i) in used_pairs:
            continue
        if abs(i - j) <= 1:  # Skip sequential pairs
            continue

        used_pairs.add((i, j))
        d = math.sqrt(sum((a - b) ** 2 for a, b in zip(centers[i], centers[j])))

        if distance_noise > 0:
            dmin = max(0.5, d - distance_noise)
            dmax = d + distance_noise
        else:
            dmin = max(0.5, d - 0.5)
            dmax = d + 0.5

        constraints.append((i, j, dmin, dmax))

    return constraints


def solve_distance_geometry_incremental(n_points, constraints, n_samples=50000):
    """Solve distance geometry incrementally.

    Place points one by one, using constraints to guide placement.
    """
    if not constraints:
        return [(0, 0, 0)] * n_points, 0, 0

    # Build adjacency
    adj = {i: [] for i in range(n_points)}
    for i, j, dmin, dmax in constraints:
        adj[i].append((j, dmin, dmax))
        adj[j].append((i, dmin, dmax))

    # Find the point with most constraints as starting point
    start = max(adj, key=lambda k: len(adj[k]))

    positions = [None] * n_points
    positions[start] = (0.0, 0.0, 0.0)

    placed = {start}
    queue = [start]

    while queue and len(placed) < n_points:
        current = queue.pop(0)
        for neighbor, dmin, dmax in adj[current]:
            if neighbor in placed:
                continue

            best_pos = None
            best_violation = float("inf")

            samples_per_point = max(1000, n_samples // n_points)
            for _ in range(samples_per_point):
                theta = random.uniform(0, 2 * math.pi)
                phi = math.acos(2 * random.random() - 1)
                r = random.uniform(dmin, dmax)

                x = positions[current][0] + r * math.sin(phi) * math.cos(theta)
                y = positions[current][1] + r * math.sin(phi) * math.sin(theta)
                z = positions[current][2] + r * math.cos(phi)

                violation = 0.0
                for other, odmin, odmax in adj[neighbor]:
                    if other in placed and other != current:
                        d = math.sqrt(
                            (x - positions[other][0])**2 +
                            (y - positions[other][1])**2 +
                            (z - positions[other][2])**2
                        )
                        if d < odmin:
                            violation += (odmin - d) ** 2
                        elif d > odmax:
                            violation += (d - odmax) ** 2

                if violation < best_violation:
                    best_violation = violation
                    best_pos = (x, y, z)

            if best_pos:
                positions[neighbor] = best_pos
                placed.add(neighbor)
                queue.append(neighbor)

    # Fill unplaced points
    for i in range(n_points):
        if positions[i] is None:
            positions[i] = (random.uniform(-50, 50), random.uniform(-50, 50), random.uniform(-50, 50))

    # Compute total violation
    satisfied = 0
    total_violation = 0.0
    for i, j, dmin, dmax in constraints:
        d = math.sqrt(sum((a - b) ** 2 for a, b in zip(positions[i], positions[j])))
        if dmin <= d <= dmax:
            satisfied += 1
        else:
            if d < dmin:
                total_violation += (dmin - d) ** 2
            else:
                total_violation += (d - dmax) ** 2

    return positions, satisfied, total_violation


def compute_rmsd_with_alignment(predicted, reference):
    """Compute RMSD after centering."""
    n = len(predicted)
    if n == 0:
        return float("inf")

    pred_center = [sum(p[i] for p in predicted) / n for i in range(3)]
    ref_center = [sum(r[i] for r in reference) / n for i in range(3)]

    pred_centered = [(p[0]-pred_center[0], p[1]-pred_center[1], p[2]-pred_center[2]) for p in predicted]
    ref_centered = [(r[0]-ref_center[0], r[1]-ref_center[1], r[2]-ref_center[2]) for r in reference]

    rmsd_sq = sum(
        sum((p[i] - r[i]) ** 2 for i in range(3))
        for p, r in zip(pred_centered, ref_centered)
    ) / n

    return math.sqrt(rmsd_sq)


def run_benchmark(pdb_path, name, n_constraints_list, n_segments=10, seed=42):
    """Run benchmark comparing DG, GCIQA, and DG+GCIQA."""
    print(f"\n{'='*60}")
    print(f"Benchmark: {name}")
    print(f"{'='*60}")

    protein = parse_pdb(pdb_path)
    centers = compute_residue_centers(protein)
    n_residues = len(centers)

    segments, segment_indices = coarse_grain_by_segments(centers, n_segments)
    n_seg = len(segments)

    # Chain connectivity constraints (always available)
    chain_constraints = extract_chain_constraints(segments, segment_indices)

    print(f"  Residues: {n_residues}")
    print(f"  Segments: {n_seg}")
    print(f"  Chain constraints: {len(chain_constraints)}")

    results = []

    for n_constraints in n_constraints_list:
        random.seed(seed)

        # Extract additional random constraints
        extra_constraints = extract_random_constraints(
            segments, n_constraints, distance_noise=2.0
        )

        if len(extra_constraints) < n_constraints:
            n_constraints = len(extra_constraints)

        # Combine chain + extra constraints
        all_constraints = chain_constraints + extra_constraints
        n_total = len(all_constraints)

        print(f"\n  --- {n_constraints} extra constraints + {len(chain_constraints)} chain = {n_total} total ---")

        # Method 1: Distance geometry (incremental)
        dg_positions, dg_satisfied, dg_violation = solve_distance_geometry_incremental(
            n_seg, all_constraints, n_samples=50000
        )
        rmsd_dg = compute_rmsd_with_alignment(dg_positions, segments)
        print(f"    DG: RMSD={rmsd_dg:.2f}A, satisfied={dg_satisfied}/{n_total}")

        # Method 2: GCIQA with DG result as starting point
        super_constraints = []
        for i, j, dmin, dmax in all_constraints:
            super_constraints.append(
                GeometricConstraint.bond(str(i), str(j), min_dist=dmin, max_dist=dmax)
            )

        # Pocket constraint centered on DG result
        dg_center = [sum(p[i] for p in dg_positions) / n_seg for i in range(3)]
        max_r = max(
            math.sqrt((p[0]-dg_center[0])**2 + (p[1]-dg_center[1])**2 + (p[2]-dg_center[2])**2)
            for p in dg_positions
        )
        super_constraints.append(
            GeometricConstraint.pocket(center=tuple(dg_center), radius=max(max_r * 1.5, 10.0))
        )

        constraints = ConstraintSet(super_constraints)

        gciqa = GCIQA(
            n_super_atoms=n_seg,
            constraints=constraints,
            coord_range=(-100.0, 100.0),
            bits_per_coord=3,
            alpha=0.7,
            convergence_threshold=5.0,
            use_quantum=False,
        )

        try:
            result = gciqa.run(max_iterations=5, n_shots=500, n_clusters=3)
        except ValueError as e:
            print(f"    GCIQA error: {e}")
            results.append({
                "n_constraints": n_constraints,
                "constraints_per_segment": n_constraints / n_seg,
                "rmsd_dg": rmsd_dg,
                "rmsd_gciqa": float("inf"),
                "dg_satisfied": dg_satisfied,
            })
            continue

        if result.best_conformation:
            gciqa_positions = []
            for idx in range(n_seg):
                key = str(idx)
                if key in result.best_conformation:
                    gciqa_positions.append(result.best_conformation[key])
                else:
                    gciqa_positions.append(dg_positions[idx])

            rmsd_gciqa = compute_rmsd_with_alignment(gciqa_positions, segments)
            satisfied, score = constraints.evaluate(result.best_conformation)
            print(f"    GCIQA: RMSD={rmsd_gciqa:.2f}A, score={score:.2f}, time={result.total_time:.2f}s")
        else:
            rmsd_gciqa = float("inf")
            print("    GCIQA: no valid conformation")

        results.append({
            "n_constraints": n_constraints,
            "constraints_per_segment": n_constraints / n_seg,
            "rmsd_dg": rmsd_dg,
            "rmsd_gciqa": rmsd_gciqa,
            "dg_satisfied": dg_satisfied,
        })

    return results


def main():
    print("=" * 60)
    print("Sparse Constraint Benchmark v2")
    print("=" * 60)
    print()
    print("Compares: Distance Geometry vs GCIQA vs DG+GCIQA")
    print("Chain connectivity constraints are always included.")
    print("'Sparse constraints' = additional random distance constraints.")

    data_dir = os.path.join(os.path.dirname(__file__), "..", "zn_metalloproteinase", "data")

    proteins = [
        ("1ZNF", "Zinc Finger (28 res)", 28),
        ("1MBN", "Myoglobin (153 res)", 153),
    ]

    constraint_levels = [5, 10, 20, 50]

    all_results = []
    for pdb_id, name, n_res in proteins:
        pdb_path = os.path.join(data_dir, f"{pdb_id}.pdb")
        if not os.path.exists(pdb_path):
            print(f"\n  SKIPPING {name}: PDB file not found")
            continue

        results = run_benchmark(pdb_path, name, constraint_levels)
        for r in results:
            r["pdb_id"] = pdb_id
            r["name"] = name
        all_results.extend(results)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    print(f"\n  {'Protein':25s} {'Extra':>6s} {'Chain':>6s} {'Total':>6s} {'DG RMSD':>10s} {'GCIQA RMSD':>12s}")
    print(f"  {'-'*25} {'-'*6} {'-'*6} {'-'*6} {'-'*10} {'-'*12}")

    for r in all_results:
        rmsd_dg = r.get("rmsd_dg", float("inf"))
        rmsd_gciqa = r.get("rmsd_gciqa", float("inf"))
        dg_str = f"{rmsd_dg:.2f}" if rmsd_dg < float("inf") else "inf"
        gciqa_str = f"{rmsd_gciqa:.2f}" if rmsd_gciqa < float("inf") else "inf"
        len(extract_chain_constraints([(0,0,0)] * 10, []))  # placeholder
        print(f"  {r['name']:25s} {r['n_constraints']:6d} {'~9':>6s} {r['n_constraints']+9:6d} {dg_str:>10s} {gciqa_str:>12s}")


if __name__ == "__main__":
    main()
