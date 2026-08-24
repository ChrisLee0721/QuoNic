"""Active space selection tests."""

from __future__ import annotations

import pytest

from quonic.chem import ActiveSpace, Molecule, select_active_space

H2_XYZ = """\
2
H2
H  0.0  0.0  0.0
H  0.0  0.0  0.74
"""


def test_manual_active_space():
    mol = Molecule.from_xyz(H2_XYZ)
    cas = select_active_space(mol, n_active_electrons=2, n_active_orbitals=2, method="manual")
    assert isinstance(cas, ActiveSpace)
    assert cas.n_electrons == 2
    assert cas.n_orbitals == 2
    assert cas.orbital_indices == (0, 1)


def test_active_space_validation_too_many_electrons():
    mol = Molecule.from_xyz(H2_XYZ)
    with pytest.raises(ValueError, match="invalid"):
        select_active_space(mol, n_active_electrons=100, n_active_orbitals=2, method="manual")


def test_active_space_validation_missing_args():
    mol = Molecule.from_xyz(H2_XYZ)
    with pytest.raises(ValueError, match="requires both"):
        select_active_space(mol, method="manual")


def test_full_valence_h2():
    mol = Molecule.from_xyz(H2_XYZ)
    cas = select_active_space(mol, method="full_valence")
    assert cas.n_electrons == 2
    assert cas.n_orbitals >= 1
