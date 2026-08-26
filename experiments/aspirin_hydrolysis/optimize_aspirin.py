#!/usr/bin/env python3
"""Quick aspirin optimization with limited steps."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from quonic.chem import Molecule, optimize_geometry

# Aspirin with better initial geometry based on optimized phenol
# Ester group: O-C(=O)-CH3 attached at C1 position
# Carboxylic acid: C(=O)-OH attached at C2 position
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

print("Optimizing aspirin (10 steps max)...")
t0 = time.time()
result = optimize_geometry(aspirin, method="hf", basis="sto-3g", max_steps=10)
elapsed = time.time() - t0

print(f"Energy: {result.energy:.6f} Hartree")
print(f"Converged: {result.converged}")
print(f"Gradient norm: {result.gradient_norm:.2e}")
print(f"Time: {elapsed:.1f}s")

# Save optimized coordinates
opt_mol = result.molecule
lines = [str(opt_mol.n_atoms), "Aspirin HF/STO-3G partial optimization"]
for atom, (x, y, z) in zip(opt_mol.atoms, opt_mol.coords):
    lines.append(f"{atom:2s}  {x:12.6f}  {y:12.6f}  {z:12.6f}")

out_path = Path(__file__).parent / "aspirin_opt.xyz"
out_path.write_text("\n".join(lines) + "\n")
print(f"Saved to {out_path.name}")

# Also update reactants.xyz with optimized aspirin + water
water = Molecule.from_xyz_file(Path(__file__).parent / "water.xyz")
lines = [str(opt_mol.n_atoms + water.n_atoms), "Aspirin + Water reactants"]
for atom, (x, y, z) in zip(opt_mol.atoms, opt_mol.coords):
    lines.append(f"{atom:2s}  {x:12.6f}  {y:12.6f}  {z:12.6f}")
for atom, (x, y, z) in zip(water.atoms, water.coords):
    lines.append(f"{atom:2s}  {x+6.0:12.6f}  {y:12.6f}  {z:12.6f}")
(Path(__file__).parent / "reactants.xyz").write_text("\n".join(lines) + "\n")
print(f"Updated reactants.xyz ({opt_mol.n_atoms + water.n_atoms} atoms)")
