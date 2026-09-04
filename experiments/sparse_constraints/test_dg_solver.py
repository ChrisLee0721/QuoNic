"""Test: Distance Geometry as main solver + GCIQA refinement.

Strategy:
1. Use incremental DG to generate initial conformations
2. Use GCIQA to refine the best DG result
3. Compare DG-only vs DG+GCIQA

Key insight: DG reduces search space from 2^(3N*b) to N × 2^(3b).
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
)


def generate_chain(n_points, bond_length=2.0, angle=110.0):
    """Generate a simple chain of points."""
    positions = [(0.0, 0.0, 0.0)]
    for i in range(1, n_points):
        angle_rad = math.radians(angle + random.gauss(0, 15))
        dx = bond_length * math.cos(angle_rad * (i % 3))
        dy = bond_length * math.sin(angle_rad * (i % 3))
        dz = random.gauss(0, 0.3)
        positions.append((
            positions[-1][0] + dx,
            positions[-1][1] + dy,
            positions[-1][2] + dz,
        ))
    return positions


def extract_constraints(positions, n_extra, noise=1.0):
    """Extract constraints: chain connectivity + random extra."""
    n = len(positions)
    constraints = []

    # Chain connectivity
    for i in range(n - 1):
        d = math.sqrt(sum((a - b) ** 2 for a, b in zip(positions[i], positions[i+1])))
        constraints.append((i, i+1, max(0.5, d - noise), d + noise))

    # Random extra constraints
    used = set()
    for _ in range(n_extra * 20):
        if len(constraints) - (n - 1) >= n_extra:
            break
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)
        if i == j or abs(i - j) <= 1 or (i, j) in used or (j, i) in used:
            continue
        used.add((i, j))
        d = math.sqrt(sum((a - b) ** 2 for a, b in zip(positions[i], positions[j])))
        constraints.append((i, j, max(0.5, d - noise), d + noise))

    return constraints


def solve_dg_incremental(n_points, constraints, n_samples=5000):
    """Incremental distance geometry solver.

    Places points one by one, using constraints to guide placement.
    Much smarter than random sampling: reduces search space from 2^(3N*b) to N × 2^(3b).
    """
    # Build adjacency
    adj = {i: [] for i in range(n_points)}
    for i, j, dmin, dmax in constraints:
        adj[i].append((j, dmin, dmax))
        adj[j].append((i, dmin, dmax))

    # Find root (most connected point)
    root = max(adj, key=lambda k: len(adj[k]))

    best_positions = None
    best_violation = float("inf")

    for _ in range(n_samples):
        positions = [None] * n_points
        positions[root] = (0.0, 0.0, 0.0)
        placed = {root}
        queue = [root]
        total_violation = 0.0

        while queue and len(placed) < n_points:
            current = queue.pop(0)
            for neighbor, dmin, dmax in adj[current]:
                if neighbor in placed:
                    continue

                # Place neighbor at random position satisfying constraint to current
                best_pos = None
                best_local_violation = float("inf")

                for _ in range(100):
                    theta = random.uniform(0, 2 * math.pi)
                    phi = math.acos(2 * random.random() - 1)
                    r = random.uniform(dmin, dmax)

                    x = positions[current][0] + r * math.sin(phi) * math.cos(theta)
                    y = positions[current][1] + r * math.sin(phi) * math.sin(theta)
                    z = positions[current][2] + r * math.cos(phi)

                    # Check against all placed neighbors
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

                    if violation < best_local_violation:
                        best_local_violation = violation
                        best_pos = (x, y, z)

                if best_pos:
                    positions[neighbor] = best_pos
                    placed.add(neighbor)
                    queue.append(neighbor)
                    total_violation += best_local_violation

        # Fill unplaced
        for i in range(n_points):
            if positions[i] is None:
                positions[i] = (random.uniform(-10, 10), random.uniform(-10, 10), random.uniform(-10, 10))

        if total_violation < best_violation:
            best_violation = total_violation
            best_positions = positions

    return best_positions


def compute_rmsd(predicted, reference):
    """Compute RMSD after centering."""
    n = len(predicted)
    pc = [sum(p[i] for p in predicted) / n for i in range(3)]
    rc = [sum(r[i] for r in reference) / n for i in range(3)]
    pred_c = [(p[0]-pc[0], p[1]-pc[1], p[2]-pc[2]) for p in predicted]
    ref_c = [(r[0]-rc[0], r[1]-rc[1], r[2]-rc[2]) for r in reference]
    return math.sqrt(sum(sum((p[i]-r[i])**2 for i in range(3)) for p, r in zip(pred_c, ref_c)) / n)


def evaluate_constraints(positions, constraints):
    """Check how many constraints are satisfied."""
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
    return satisfied, total_violation


def run_test(n_points, n_extra, seed=42):
    """Run a single test: DG only vs DG+GCIQA."""
    random.seed(seed)

    true_positions = generate_chain(n_points)
    constraints = extract_constraints(true_positions, n_extra)
    n_total = len(constraints)

    # Method 1: DG only
    dg_positions = solve_dg_incremental(n_points, constraints, n_samples=1000)
    rmsd_dg = compute_rmsd(dg_positions, true_positions)
    sat_dg, viol_dg = evaluate_constraints(dg_positions, constraints)

    # Method 2: DG + GCIQA refinement
    # Build GCIQA constraints from DG result
    gciqa_constraints = []
    for i, j, dmin, dmax in constraints:
        gciqa_constraints.append(
            GeometricConstraint.bond(str(i), str(j), min_dist=dmin, max_dist=dmax)
        )

    # Pocket centered on DG result
    cx = sum(p[0] for p in dg_positions) / n_points
    cy = sum(p[1] for p in dg_positions) / n_points
    cz = sum(p[2] for p in dg_positions) / n_points
    max_r = max(math.sqrt((p[0]-cx)**2 + (p[1]-cy)**2 + (p[2]-cz)**2) for p in dg_positions)
    gciqa_constraints.append(
        GeometricConstraint.pocket(center=(cx, cy, cz), radius=max(max_r * 1.5, 3.0))
    )

    cs = ConstraintSet(gciqa_constraints)

    # Use DG result to narrow coordinate range
    xs = [p[0] for p in dg_positions]
    ys = [p[1] for p in dg_positions]
    zs = [p[2] for p in dg_positions]
    margin = 3.0
    coord_range = (
        (min(min(xs), min(ys), min(zs)) - margin, max(max(xs), max(ys), max(zs)) + margin)
    )

    # Convert DG result to GCIQA format (string keys)
    dg_conformation = {str(i): pos for i, pos in enumerate(dg_positions)}

    gciqa = GCIQA(
        n_super_atoms=n_points,
        constraints=cs,
        coord_range=coord_range,
        bits_per_coord=5,
        alpha=0.7,
        convergence_threshold=1.0,
        use_quantum=False,
        initial_conformation=dg_conformation,
    )

    try:
        result = gciqa.run(max_iterations=5, n_shots=2000, n_clusters=3)
    except Exception:
        result = None

    if result and result.best_conformation:
        gciqa_positions = []
        for idx in range(n_points):
            key = str(idx)
            if key in result.best_conformation:
                gciqa_positions.append(result.best_conformation[key])
            else:
                gciqa_positions.append(dg_positions[idx])

        rmsd_gciqa = compute_rmsd(gciqa_positions, true_positions)
        sat_gciqa, viol_gciqa = evaluate_constraints(gciqa_positions, constraints)
    else:
        rmsd_gciqa = float("inf")
        sat_gciqa = 0
        float("inf")

    return {
        "n_points": n_points,
        "n_extra": n_extra,
        "n_total": n_total,
        "rmsd_dg": rmsd_dg,
        "rmsd_gciqa": rmsd_gciqa,
        "sat_dg": sat_dg,
        "sat_gciqa": sat_gciqa,
        "improved": rmsd_gciqa < rmsd_dg,
    }


def main():
    print("=" * 70)
    print("Distance Geometry Solver + GCIQA Refinement")
    print("=" * 70)

    tests = [
        (3, 0), (3, 1), (3, 2),
        (4, 0), (4, 2), (4, 4),
        (5, 0), (5, 3), (5, 5),
        (6, 0), (6, 3), (6, 6),
        (8, 0), (8, 5), (8, 10),
        (10, 0), (10, 5), (10, 15),
    ]

    print(f"\n  {'Pts':>3s} {'Extra':>5s} {'Total':>5s} │ {'DG RMSD':>8s} {'Sat':>4s} │ {'GCIQA RMSD':>10s} {'Sat':>4s} │ {'Better':>6s}")
    print(f"  {'-'*3} {'-'*5} {'-'*5} │ {'-'*8} {'-'*4} │ {'-'*10} {'-'*4} │ {'-'*6}")

    for n_points, n_extra in tests:
        r = run_test(n_points, n_extra)
        dg_str = f"{r['rmsd_dg']:.2f}"
        gciqa_str = f"{r['rmsd_gciqa']:.2f}" if r['rmsd_gciqa'] < float("inf") else "inf"
        improved = "YES" if r['improved'] else "no"
        print(f"  {n_points:3d} {n_extra:5d} {r['n_total']:5d} │ {dg_str:>8s} {r['sat_dg']:4d} │ {gciqa_str:>10s} {r['sat_gciqa']:4d} │ {improved:>6s}")


if __name__ == "__main__":
    main()
