#!/usr/bin/env python3
"""Pipeline validation with fully optimized small molecules.

Reaction: H2 + H2 → H2 + H2 (identity, ΔG ≈ 0)
Reaction: CH3COOH → CH3COOH (identity, ΔG ≈ 0)
Reaction: C6H5OH + H2O → C6H5OH + H2O (identity, ΔG ≈ 0)

These identity reactions validate that the pipeline gives ΔG ≈ 0
when reactants and products are the same.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from quonic.chem import Molecule, compute_dg


def main():
    print("=" * 60)
    print("PIPELINE VALIDATION: Identity Reactions (ΔG ≈ 0)")
    print("=" * 60)

    method, basis = "hf", "sto-3g"

    # Test 1: H2 + H2 → H2 + H2
    h2 = Molecule.from_xyz("""
    2
    H2
    H  0.0  0.0  0.0
    H  0.0  0.0  0.74
    """)
    print("\n--- Test 1: H2 + H2 → H2 + H2 ---")
    t0 = time.time()
    r1 = compute_dg(
        reaction={"reactants": [h2, h2], "products": [h2, h2]},
        method=method, basis=basis, optimize=True,
    )
    print(f"  ΔG = {r1.dg:.6f} kcal/mol (expected ≈ 0)")
    print(f"  ΔE = {r1.d_electronic:.6f}, ΔZPE = {r1.d_zpe:.6f}")
    print(f"  Time: {time.time()-t0:.1f}s")

    # Test 2: H2O + H2O → H2O + H2O
    h2o = Molecule.from_xyz("""
    3
    H2O
    O  0.0  0.0  0.0
    H  0.0  0.0  0.96
    H  0.0  0.96  0.0
    """)
    print("\n--- Test 2: H2O + H2O → H2O + H2O ---")
    t0 = time.time()
    r2 = compute_dg(
        reaction={"reactants": [h2o, h2o], "products": [h2o, h2o]},
        method=method, basis=basis, optimize=True,
    )
    print(f"  ΔG = {r2.dg:.6f} kcal/mol (expected ≈ 0)")
    print(f"  ΔE = {r2.d_electronic:.6f}, ΔZPE = {r2.d_zpe:.6f}")
    print(f"  ΔH_thermal = {r2.d_thermal:.6f}, Δsolv = {r2.d_solvation:.6f}")
    print(f"  Time: {time.time()-t0:.1f}s")

    # Test 3: Acetic acid identity
    acetic = Molecule.from_xyz("""
    8
    Acetic acid
    C   0.0000   0.0000   0.0000
    C   1.5000   0.0000   0.0000
    O   2.0700   1.1000   0.0000
    O   2.0700  -1.1000   0.0000
    H   3.0000  -1.0000   0.0000
    H  -0.4000   1.0200   0.0000
    H  -0.4000  -0.5100   0.8800
    H  -0.4000  -0.5100  -0.8800
    """)
    print("\n--- Test 3: CH3COOH → CH3COOH ---")
    t0 = time.time()
    r3 = compute_dg(
        reaction={"reactants": [acetic], "products": [acetic]},
        method=method, basis=basis, optimize=True,
    )
    print(f"  ΔG = {r3.dg:.6f} kcal/mol (expected ≈ 0)")
    print(f"  ΔE = {r3.d_electronic:.6f}, ΔZPE = {r3.d_zpe:.6f}")
    print(f"  Time: {time.time()-t0:.1f}s")

    print("\n" + "=" * 60)
    print("All identity reactions should give ΔG ≈ 0")
    print("This validates the pipeline is internally consistent.")
    print("=" * 60)


if __name__ == "__main__":
    main()
