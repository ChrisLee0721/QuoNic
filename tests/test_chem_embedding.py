"""DMET embedding tests."""

from __future__ import annotations

import pytest

from quonic.chem import DMET, DMETResult, Fragment, Molecule, fragment_molecule

H4_XYZ = """\
4
Linear H4
H  0.0  0.0  0.0
H  0.0  0.0  0.74
H  0.0  0.0  2.0
H  0.0  0.0  2.74
"""


def test_fragment_molecule_auto():
    mol = Molecule.from_xyz(H4_XYZ)
    frags = fragment_molecule(mol, max_fragment_size=2)
    assert len(frags) == 2
    assert all(isinstance(f, Fragment) for f in frags)
    assert frags[0].n_atoms == 2  # type: ignore[attr-defined]
    assert frags[1].n_atoms == 2  # type: ignore[attr-defined]


def test_fragment_molecule_by_atom_count():
    mol = Molecule.from_xyz(H4_XYZ)
    frags = fragment_molecule(mol, method="by_atom_count", max_fragment_size=3)
    assert len(frags) == 2  # 4 atoms / 3 = 2 fragments


def test_fragment_fields():
    mol = Molecule.from_xyz(H4_XYZ)
    frags = fragment_molecule(mol, max_fragment_size=2)
    frag = frags[0]
    assert hasattr(frag, "atom_indices")
    assert hasattr(frag, "atoms")
    assert hasattr(frag, "coords")
    assert hasattr(frag, "charge")
    assert hasattr(frag, "spin")


def test_dmet_result_fields():
    """DMETResult should have all expected fields."""
    result = DMETResult(
        energy=-2.0,
        fragment_energies=[-1.0, -1.0],
        converged=True,
        n_iterations=5,
        chemical_potential=0.01,
    )
    assert result.energy == -2.0
    assert len(result.fragment_energies) == 2
    assert result.converged is True


def test_dmet_h4_linear():
    """DMET on linear H4 with fragment_size=2 should converge."""
    pytest.importorskip("pyscf")
    mol = Molecule.from_xyz(H4_XYZ, basis="sto-3g")
    dmet = DMET(mol, fragment_size=2, max_iter=10)
    result = dmet.solve()
    assert isinstance(result, DMETResult)
    assert result.energy < 0  # should be negative for bound system
    assert len(result.fragment_energies) == 2
