#!/usr/bin/env python3
"""Build aspirin and salicylic acid from optimized phenol fragment.

Uses optimized phenol coordinates and adds functional groups
with correct bond lengths and angles.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def optimize_and_save(mol, name, method="hf", basis="sto-3g", max_steps=30):
    """Optimize molecule and save XYZ."""
    from quonic.chem import optimize_geometry

    print(f"\nOptimizing {name} ({mol.n_atoms} atoms)...")
    t0 = time.time()
    result = optimize_geometry(mol, method=method, basis=basis, max_steps=max_steps)
    elapsed = time.time() - t0

    print(f"  Energy: {result.energy:.6f} Hartree")
    print(f"  Converged: {result.converged}")
    print(f"  Gradient norm: {result.gradient_norm:.2e}")
    print(f"  Time: {elapsed:.1f}s")

    opt_mol = result.molecule
    lines = [str(opt_mol.n_atoms), f"{name} HF/STO-3G optimized"]
    for atom, (x, y, z) in zip(opt_mol.atoms, opt_mol.coords):
        lines.append(f"{atom:2s}  {x:12.6f}  {y:12.6f}  {z:12.6f}")

    out_path = Path(__file__).parent / f"{name.lower().replace(' ', '_')}.xyz"
    out_path.write_text("\n".join(lines) + "\n")
    print(f"  Saved to {out_path.name}")

    return result


def main():
    from quonic.chem import Molecule

    print("=" * 60)
    print("Building molecules from optimized phenol fragment")
    print("=" * 60)

    # Optimized phenol coordinates (from HF/STO-3G)
    # Ring in xy-plane, OH at C1 (top)
    # C1(0, 1.388), C2(1.204, 0.690), C3(1.198, -0.696),
    # C4(0.003, -1.394), C5(-1.199, -0.696), C6(-1.208, 0.686)
    # O(-0.071, 2.782), H_OH(0.868, 3.091)

    # Build salicylic acid: replace H on C2 with COOH
    # C2 is at (1.204, 0.690), H on C2 is at (2.139, 1.234)
    # COOH: C attached to C2, with =O and -OH
    # C-C bond ~1.48 A, C=O ~1.21 A, C-OH ~1.34 A
    salicylic_acid = Molecule.from_xyz("""
    16
    Salicylic acid 2-HO-C6H4-COOH
    C   0.0000   1.3880   0.0000
    C   1.2040   0.6900   0.0000
    C   1.1980  -0.6960   0.0000
    C   0.0030  -1.3940   0.0000
    C  -1.1990  -0.6960   0.0000
    C  -1.2080   0.6860   0.0000
    O  -0.0710   2.7820   0.0000
    H   0.8680   3.0910   0.0000
    C   2.5600   1.3000   0.0000
    O   2.5600   2.5200   0.0000
    O   3.6800   0.7000   0.0000
    H   4.4000   1.2000   0.0000
    H   2.1390  -1.2320   0.0000
    H   0.0030  -2.4750   0.0000
    H  -2.1370  -1.2370   0.0000
    H  -2.1380   1.2390   0.0000
    """)

    # Build aspirin: replace OH on C1 with OC(=O)CH3, add COOH on C2
    aspirin = Molecule.from_xyz("""
    21
    Aspirin acetylsalicylic acid
    C   0.0000   1.3880   0.0000
    C   1.2040   0.6900   0.0000
    C   1.1980  -0.6960   0.0000
    C   0.0030  -1.3940   0.0000
    C  -1.1990  -0.6960   0.0000
    C  -1.2080   0.6860   0.0000
    O  -0.0710   2.7820   0.0000
    C   1.1000   3.4000   0.0000
    O   2.1000   2.8000   0.0000
    C   1.1000   4.9000   0.0000
    H   0.5000   5.3000   0.8700
    H   2.1000   5.3000   0.0000
    H   0.5000   5.3000  -0.8700
    C   2.5600   1.3000   0.0000
    O   2.5600   2.5200   0.0000
    O   3.6800   0.7000   0.0000
    H   4.4000   1.2000   0.0000
    H   2.1390  -1.2320   0.0000
    H   0.0030  -2.4750   0.0000
    H  -2.1370  -1.2370   0.0000
    H  -2.1380   1.2390   0.0000
    """)

    # Optimize all molecules
    results = {}

    # Small molecules (fast)
    water = Molecule.from_xyz("""
    3
    Water
    O   0.0000   0.0000   0.0000
    H   0.0000   0.0000   0.9600
    H   0.0000   0.9600   0.0000
    """)

    acetic_acid = Molecule.from_xyz("""
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

    for name, mol in [("water", water), ("acetic_acid", acetic_acid)]:
        try:
            results[name] = optimize_and_save(mol, name, max_steps=20)
        except Exception as e:
            print(f"  FAILED: {e}")

    # Large molecules (slower)
    for name, mol in [("salicylic_acid", salicylic_acid), ("aspirin", aspirin)]:
        try:
            results[name] = optimize_and_save(mol, name, max_steps=20)
        except Exception as e:
            print(f"  FAILED: {e}")

    # Write combined reactants and products
    if "aspirin" in results and "water" in results:
        r = results["aspirin"].molecule
        w = results["water"].molecule
        lines = [str(r.n_atoms + w.n_atoms), "Aspirin + Water reactants"]
        for atom, (x, y, z) in zip(r.atoms, r.coords):
            lines.append(f"{atom:2s}  {x:12.6f}  {y:12.6f}  {z:12.6f}")
        for atom, (x, y, z) in zip(w.atoms, w.coords):
            lines.append(f"{atom:2s}  {x+6.0:12.6f}  {y:12.6f}  {z:12.6f}")
        (Path(__file__).parent / "reactants.xyz").write_text("\n".join(lines) + "\n")
        print(f"\n  reactants.xyz ({r.n_atoms + w.n_atoms} atoms)")

    if "salicylic_acid" in results and "acetic_acid" in results:
        s = results["salicylic_acid"].molecule
        a = results["acetic_acid"].molecule
        lines = [str(s.n_atoms + a.n_atoms), "Salicylic acid + Acetic acid products"]
        for atom, (x, y, z) in zip(s.atoms, s.coords):
            lines.append(f"{atom:2s}  {x:12.6f}  {y:12.6f}  {z:12.6f}")
        for atom, (x, y, z) in zip(a.atoms, a.coords):
            lines.append(f"{atom:2s}  {x+6.0:12.6f}  {y:12.6f}  {z:12.6f}")
        (Path(__file__).parent / "products.xyz").write_text("\n".join(lines) + "\n")
        print(f"  products.xyz ({s.n_atoms + a.n_atoms} atoms)")

    print("\nDone!")


if __name__ == "__main__":
    main()
