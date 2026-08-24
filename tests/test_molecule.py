"""Molecule loading tests."""

from __future__ import annotations

import pytest

from quonic.chem import Molecule

H2_XYZ = """\
2
H2 at equilibrium
H  0.0  0.0  0.0
H  0.0  0.0  0.74
"""

H2O_XYZ = """\
3
Water molecule
O  0.000  0.000  0.117
H  0.000  0.757 -0.469
H  0.000 -0.757 -0.469
"""


def test_molecule_from_xyz_string():
    mol = Molecule.from_xyz(H2_XYZ)
    assert mol.n_atoms == 2
    assert mol.atoms == ("H", "H")


def test_molecule_from_xyz_h2o():
    mol = Molecule.from_xyz(H2O_XYZ)
    assert mol.n_atoms == 3
    assert mol.atoms == ("O", "H", "H")


def test_molecule_n_electrons():
    mol_h2 = Molecule.from_xyz(H2_XYZ)
    assert mol_h2.n_electrons == 2

    mol_h2o = Molecule.from_xyz(H2O_XYZ)
    assert mol_h2o.n_electrons == 10  # O(8) + H(1) + H(1)


def test_molecule_charge():
    mol = Molecule.from_xyz(H2_XYZ, charge=1)
    assert mol.n_electrons == 1  # H2+


def test_molecule_spin_multiplicity():
    mol = Molecule.from_xyz(H2_XYZ, spin=0)
    assert mol.spin_multiplicity == 1  # singlet

    mol2 = Molecule.from_xyz(H2_XYZ, spin=1)
    assert mol2.spin_multiplicity == 2  # doublet


def test_molecule_repr():
    mol = Molecule.from_xyz(H2_XYZ)
    r = repr(mol)
    assert "H2" in r
    assert "sto-3g" in r


def test_molecule_coords():
    mol = Molecule.from_xyz(H2_XYZ)
    assert len(mol.coords) == 2
    assert mol.coords[0] == (0.0, 0.0, 0.0)
    assert abs(mol.coords[1][2] - 0.74) < 1e-10


def test_molecule_from_xyz_file(tmp_path):
    xyz_file = tmp_path / "h2.xyz"
    xyz_file.write_text(H2_XYZ)
    mol = Molecule.from_xyz_file(xyz_file)
    assert mol.n_atoms == 2


def test_molecule_from_xyz_bad():
    with pytest.raises(ValueError, match="too few lines"):
        Molecule.from_xyz("1\n")


def test_molecule_from_smiles():
    pytest.importorskip("rdkit")
    mol = Molecule.from_smiles("O")  # water
    assert mol.n_atoms == 3  # O + 2H
    assert mol.n_electrons == 10


def test_molecule_frozen():
    mol = Molecule.from_xyz(H2_XYZ)
    with pytest.raises(AttributeError):
        mol.atoms = ("He",)  # type: ignore[misc]
