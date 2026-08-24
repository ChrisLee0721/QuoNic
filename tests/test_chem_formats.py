"""Molecular format parser tests."""

from __future__ import annotations

import pytest

from quonic.chem import Molecule
from quonic.chem.formats import from_mol2, from_pdb

# ── PDB test data ──────────────────────────────────────────────

PDB_SIMPLE = """\
ATOM      1  O   HOH A   1       0.000   0.000   0.117  1.00  0.00           O
ATOM      2  H   HOH A   1       0.000   0.757  -0.469  1.00  0.00           H
ATOM      3  H   HOH A   1       0.000  -0.757  -0.469  1.00  0.00           H
END
"""

PDB_MULTI_MODEL = """\
MODEL        1
ATOM      1  C   MOL A   1       0.000   0.000   0.000  1.00  0.00           C
ATOM      2  H   MOL A   1       1.000   0.000   0.000  1.00  0.00           H
ENDMDL
MODEL        2
ATOM      1  C   MOL A   1       0.000   0.000   0.000  1.00  0.00           C
ATOM      2  H   MOL A   1       2.000   0.000   0.000  1.00  0.00           H
ENDMDL
"""


def test_from_pdb_simple(tmp_path):
    pdb_file = tmp_path / "water.pdb"
    pdb_file.write_text(PDB_SIMPLE)
    mol = from_pdb(pdb_file)
    assert mol.n_atoms == 3
    assert mol.atoms[0] == "O"
    assert mol.atoms[1] == "H"
    assert mol.atoms[2] == "H"


def test_from_pdb_coords(tmp_path):
    pdb_file = tmp_path / "water.pdb"
    pdb_file.write_text(PDB_SIMPLE)
    mol = from_pdb(pdb_file)
    assert abs(mol.coords[0][2] - 0.117) < 1e-6


def test_from_pdb_multi_model(tmp_path):
    pdb_file = tmp_path / "multi.pdb"
    pdb_file.write_text(PDB_MULTI_MODEL)
    mol = from_pdb(pdb_file)
    # Should only read first model
    assert mol.n_atoms == 2
    assert abs(mol.coords[1][0] - 1.0) < 1e-6


def test_from_pdb_empty(tmp_path):
    pdb_file = tmp_path / "empty.pdb"
    pdb_file.write_text("END\n")
    with pytest.raises(ValueError, match="PDB"):
        from_pdb(pdb_file)


# ── MOL2 test data ─────────────────────────────────────────────

MOL2_SIMPLE = """\
@<TRIPOS>MOLECULE
water
 3 2 0 0 0
SMALL
GASTEIGER

@<TRIPOS>ATOM
      1 O           0.0000    0.0000    0.1170 O.3         1  HOH    -0.4100
      2 H           0.0000    0.7570   -0.4690 H           1  HOH     0.2050
      3 H           0.0000   -0.7570   -0.4690 H           1  HOH     0.2050
@<TRIPOS>BOND
     1    1    2 1
     2    1    3 1
"""


def test_from_mol2_simple(tmp_path):
    mol2_file = tmp_path / "water.mol2"
    mol2_file.write_text(MOL2_SIMPLE)
    mol = from_mol2(mol2_file)
    assert mol.n_atoms == 3
    assert mol.atoms[0] == "O"


def test_from_mol2_coords(tmp_path):
    mol2_file = tmp_path / "water.mol2"
    mol2_file.write_text(MOL2_SIMPLE)
    mol = from_mol2(mol2_file)
    assert abs(mol.coords[1][1] - 0.757) < 1e-6


def test_from_mol2_empty(tmp_path):
    mol2_file = tmp_path / "empty.mol2"
    mol2_file.write_text("@<TRIPOS>MOLECULE\nempty\n")
    with pytest.raises(ValueError, match="MOL2"):
        from_mol2(mol2_file)


# ── Molecule.from_pdb / from_mol2 ──────────────────────────────

def test_molecule_from_pdb(tmp_path):
    pdb_file = tmp_path / "water.pdb"
    pdb_file.write_text(PDB_SIMPLE)
    mol = Molecule.from_pdb(pdb_file, charge=0, spin=0, basis="sto-3g")
    assert mol.n_atoms == 3
    assert mol.basis == "sto-3g"


def test_molecule_from_mol2(tmp_path):
    mol2_file = tmp_path / "water.mol2"
    mol2_file.write_text(MOL2_SIMPLE)
    mol = Molecule.from_mol2(mol2_file, charge=1)
    assert mol.n_atoms == 3
    assert mol.charge == 1
