"""Small-scale sparse constraint test.

Tests GCIQA on a tiny system (5-6 points) with known ground truth.
This verifies whether GCIQA can solve sparse constraint problems at all.
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
    """Generate a simple chain of points within ~10A range."""
    positions = [(0.0, 0.0, 0.0)]
    for i in range(1, n_points):
        angle_rad = math.radians(angle + random.gauss(0, 15))
        dx = bond_length * math.cos(angle_rad * (i % 3))
        dy = bond_length * math.sin(angle_rad * (i % 3))
        dz = random.gauss(0, 0.3)
        x = positions[-1][0] + dx
        y = positions[-1][1] + dy
        z = positions[-1][2] + dz
        positions.append((x, y, z))
    return positions


def extract_constraints(positions, n_extra, noise=1.0):
    """Extract constraints: chain connectivity + random extra."""
    n = len(positions)
    constraints = []

    # Chain connectivity (always available)
    for i in range(n - 1):
        d = math.sqrt(sum((a - b) ** 2 for a, b in zip(positions[i], positions[i+1])))
        constraints.append((i, i+1, max(0.5, d - noise), d + noise))

    # Random extra constraints
    used = set()
    for _ in range(n_extra * 10):
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


def compute_rmsd(predicted, reference):
    """Compute RMSD after centering."""
    n = len(predicted)
    pc = [sum(p[i] for p in predicted) / n for i in range(3)]
    rc = [sum(r[i] for r in reference) / n for i in range(3)]
    pred_c = [(p[0]-pc[0], p[1]-pc[1], p[2]-pc[2]) for p in predicted]
    ref_c = [(r[0]-rc[0], r[1]-rc[1], r[2]-rc[2]) for r in reference]
    return math.sqrt(sum(sum((p[i]-r[i])**2 for i in range(3)) for p, r in zip(pred_c, ref_c)) / n)


def run_test(n_points, n_extra, seed=42, use_quantum=False, bits=5, coord_range=(-10.0, 10.0)):
    """Run a single test case."""
    random.seed(seed)

    # Generate ground truth
    true_positions = generate_chain(n_points)
    constraints = extract_constraints(true_positions, n_extra)

    n_points - 1
    n_total = len(constraints)

    # Build GCIQA constraints
    gciqa_constraints = []
    for i, j, dmin, dmax in constraints:
        gciqa_constraints.append(
            GeometricConstraint.bond(str(i), str(j), min_dist=dmin, max_dist=dmax)
        )

    # Pocket constraint centered on ground truth
    cx = sum(p[0] for p in true_positions) / n_points
    cy = sum(p[1] for p in true_positions) / n_points
    cz = sum(p[2] for p in true_positions) / n_points
    max_r = max(math.sqrt((p[0]-cx)**2 + (p[1]-cy)**2 + (p[2]-cz)**2) for p in true_positions)
    gciqa_constraints.append(
        GeometricConstraint.pocket(center=(cx, cy, cz), radius=max(max_r * 2, 5.0))
    )

    cs = ConstraintSet(gciqa_constraints)

    # Run GCIQA
    gciqa = GCIQA(
        n_super_atoms=n_points,
        constraints=cs,
        coord_range=coord_range,
        bits_per_coord=bits,
        alpha=0.7,
        convergence_threshold=1.0,
        use_quantum=use_quantum,
    )

    result = gciqa.run(max_iterations=15, n_shots=10000, n_clusters=3)

    if result.best_conformation:
        predicted = []
        for idx in range(n_points):
            key = str(idx)
            if key in result.best_conformation:
                predicted.append(result.best_conformation[key])
            else:
                predicted.append((0, 0, 0))

        rmsd = compute_rmsd(predicted, true_positions)
        satisfied, score = cs.evaluate(result.best_conformation)

        return {
            "n_points": n_points,
            "n_extra": n_extra,
            "n_total": n_total,
            "rmsd": rmsd,
            "score": score,
            "satisfied": satisfied,
            "time": result.total_time,
            "success": rmsd < 2.0,
        }
    else:
        return {
            "n_points": n_points,
            "n_extra": n_extra,
            "n_total": n_total,
            "rmsd": float("inf"),
            "success": False,
        }


def main():
    print("=" * 60)
    print("Small-Scale Sparse Constraint Test")
    print("=" * 60)

    # Test with different bit resolutions
    for bits, cr in [(4, (-5.0, 5.0)), (5, (-10.0, 10.0))]:
        resolution = (cr[1] - cr[0]) / (2**bits - 1)
        print(f"\n--- bits_per_coord={bits}, coord_range={cr}, resolution={resolution:.3f}A ---")

        tests = [
            (3, 0), (3, 1), (3, 2),
            (4, 0), (4, 2), (4, 4),
            (5, 0), (5, 3),
        ]

        print(f"  {'Points':>6s} {'Extra':>6s} {'Total':>6s} {'RMSD':>8s} {'Score':>8s} {'Time':>8s} {'Status':>8s}")
        print(f"  {'-'*6} {'-'*6} {'-'*6} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

        for n_points, n_extra in tests:
            r = run_test(n_points, n_extra, bits=bits, coord_range=cr)
            rmsd_str = f"{r['rmsd']:.2f}" if r['rmsd'] < float("inf") else "inf"
            score_str = f"{r.get('score', 0):.2f}" if r['rmsd'] < float("inf") else "-"
            time_str = f"{r.get('time', 0):.1f}" if r['rmsd'] < float("inf") else "-"
            status = "OK" if r.get("success") else "FAIL"
            print(f"  {n_points:6d} {n_extra:6d} {r['n_total']:6d} {rmsd_str:>8s} {score_str:>8s} {time_str:>8s} {status:>8s}")


if __name__ == "__main__":
    main()
