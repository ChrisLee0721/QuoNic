"""GCIQA performance benchmark.

Measures wall-clock time for each stage of the GCIQA pipeline:
1. PDB parsing
2. Metal detection + constraint generation
3. Coarse-graining
4. GCIQA search (classical simulation)
5. Clustering + validation

Tests scaling behavior with varying protein sizes.

Usage:
    python run_benchmark.py
"""

import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from gciqa import (
    GCIQA,
    ConstraintSet,
    GeometricConstraint,
    auto_detect_geometry,
    find_metal_ions,
    generate_metal_constraints,
    generate_report,
    get_metal_template,
    parse_pdb_string,
)


def generate_synthetic_protein(n_residues, seed=42):
    """Generate a synthetic protein with Zn at origin."""
    random.seed(seed)
    atoms = ["ZN"]
    coords = [(0.0, 0.0, 0.0)]

    # 4 coordinating residues at tetrahedral positions
    ligand_pos = [
        (2.0, 0.0, 0.0),
        (-1.0, 1.7, 0.0),
        (-1.0, -0.85, 1.5),
        (0.0, 0.0, 2.1),
    ]
    for pos in ligand_pos:
        atoms.append("N")
        coords.append(pos)

    # Background residues
    for _ in range(n_residues):
        while True:
            x = random.gauss(0, 15)
            y = random.gauss(0, 15)
            z = random.gauss(0, 15)
            if math.sqrt(x**2 + y**2 + z**2) < 20:
                break
        atoms.append("C")
        coords.append((x, y, z))

    # Build PDB string
    lines = ["HEADER    SYNTHETIC", "TITLE     PERF TEST"]
    for i, (atom, coord) in enumerate(zip(atoms, coords)):
        x, y, z = coord
        record = "HETATM" if atom == "ZN" else "ATOM  "
        name = "ZN" if atom == "ZN" else "CA"
        resname = "ZN" if atom == "ZN" else "ALA"
        resnum = i
        chain = "A"
        lines.append(
            f"{record}{i+1:5d} {name:<4s} {resname:3s} {chain:1s}{resnum:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}{1.0:6.2f}{0.0:6.2f}          {atom:>2s}  "
        )
    lines.append("END")
    return "\n".join(lines)


def benchmark_single(n_residues, verbose=False):
    """Run timing benchmark for a single protein size."""
    pdb_string = generate_synthetic_protein(n_residues)

    timings = {}

    # Stage 1: Parse PDB
    t0 = time.time()
    protein = parse_pdb_string(pdb_string)
    timings["parse"] = time.time() - t0

    # Stage 2: Metal detection + constraints
    t0 = time.time()
    zn_ions = find_metal_ions(protein, "ZN")
    zn = zn_ions[0]
    geometry = auto_detect_geometry(zn, protein, max_dist=2.5)
    template = get_metal_template("ZN", geometry)
    atom_constraints = generate_metal_constraints(zn, protein, template, max_dist=2.5)
    timings["constraints"] = time.time() - t0

    # Stage 3: Coarse-graining (spatial, 3 super-atoms)
    t0 = time.time()
    from gciqa import coarse_grain
    cg = coarse_grain(protein.atoms, protein.coords, strategy="spatial", n_super_atoms=3)
    timings["coarse_grain"] = time.time() - t0

    # Map constraints
    metal_super = cg.atom_to_super[zn.index]
    super_constraints = []
    for c in atom_constraints.constraints:
        if c.type.value == "bond":
            atom1 = int(c.atoms[0])
            atom2 = int(c.atoms[1])
            sa1 = cg.atom_to_super[atom1]
            sa2 = cg.atom_to_super[atom2]
            if sa1 != sa2:
                super_constraints.append(
                    GeometricConstraint.bond(str(sa1), str(sa2), 1.0, 5.0)
                )
    super_constraints.append(
        GeometricConstraint.pocket(center=(0, 0, 0), radius=5.0)
    )
    constraints = ConstraintSet(super_constraints)

    # Stage 4: GCIQA search
    t0 = time.time()
    gciqa = GCIQA(
        n_super_atoms=cg.n_super_atoms,
        constraints=constraints,
        coord_range=(-10.0, 10.0),
        bits_per_coord=3,
        use_quantum=False,
    )
    result = gciqa.run(max_iterations=2, n_shots=500, n_clusters=3)
    timings["gciqa_search"] = time.time() - t0

    # Stage 5: Validation
    t0 = time.time()
    if result.best_conformation:
        metal_key = str(metal_super)
        if metal_key in result.best_conformation:
            predicted = result.best_conformation[metal_key]
            math.sqrt(sum((a - b) ** 2 for a, b in zip(predicted, (0, 0, 0))))
            generate_report(result.best_conformation, constraints)
    timings["validation"] = time.time() - t0

    timings["total"] = sum(timings.values())
    timings["n_atoms"] = len(protein.atoms)
    timings["n_residues"] = n_residues

    if verbose:
        print(f"  n_residues={n_residues:4d}  n_atoms={len(protein.atoms):5d}  "
              f"total={timings['total']:.3f}s  "
              f"parse={timings['parse']:.3f}  cg={timings['coarse_grain']:.3f}  "
              f"search={timings['gciqa_search']:.3f}")

    return timings


def main():
    print("=" * 60)
    print("GCIQA Performance Benchmark")
    print("=" * 60)

    sizes = [10, 25, 50, 100, 200, 500]
    results = []

    print("\n  Size scaling test:")
    for n in sizes:
        t = benchmark_single(n, verbose=True)
        results.append(t)

    # Summary
    print(f"\n{'='*60}")
    print("SCALING SUMMARY")
    print(f"{'='*60}")
    print(f"\n  {'Residues':>8s}  {'Atoms':>6s}  {'Total':>8s}  {'Parse':>8s}  {'CG':>8s}  {'Search':>8s}")
    print(f"  {'--------':>8s}  {'------':>6s}  {'--------':>8s}  {'--------':>8s}  {'--------':>8s}  {'--------':>8s}")
    for r in results:
        print(f"  {r['n_residues']:8d}  {r['n_atoms']:6d}  {r['total']:8.3f}  "
              f"{r['parse']:8.3f}  {r['coarse_grain']:8.3f}  {r['gciqa_search']:8.3f}")

    # Check scaling
    if len(results) >= 2:
        t_small = results[0]["total"]
        t_large = results[-1]["total"]
        ratio = t_large / t_small if t_small > 0 else float("inf")
        size_ratio = results[-1]["n_atoms"] / results[0]["n_atoms"]
        print(f"\n  Scaling: {size_ratio:.1f}x atoms → {ratio:.1f}x time")
        if ratio < size_ratio * 2:
            print("  Scaling: GOOD (sub-quadratic)")
        else:
            print("  Scaling: POOR (super-quadratic)")


if __name__ == "__main__":
    main()
