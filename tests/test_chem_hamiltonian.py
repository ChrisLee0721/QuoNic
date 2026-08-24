"""Molecular Hamiltonian generation tests."""

from __future__ import annotations

import pytest

from quonic.chem import Molecule, molecular_hamiltonian, select_active_space

H2_XYZ = """\
2
H2 at equilibrium
H  0.0  0.0  0.0
H  0.0  0.0  0.74
"""

LIH_XYZ = """\
2
LiH
Li  0.0  0.0  0.0
H   0.0  0.0  1.6
"""


def test_h2_hamiltonian_terms():
    """H2/STO-3G should produce a non-empty list of Pauli terms."""
    pytest.importorskip("pyscf")
    pytest.importorskip("openfermion")
    mol = Molecule.from_xyz(H2_XYZ, basis="sto-3g")
    result = molecular_hamiltonian(mol)
    terms = result.metadata["hamiltonian"]
    assert len(terms) > 0
    # Each term is (coeff, pauli_string)
    for coeff, pauli in terms:
        assert isinstance(coeff, float)
        assert isinstance(pauli, str)
        assert len(pauli) == result.metadata["n_qubits"]


def test_h2_hamiltonian_n_qubits():
    """H2/STO-3G has 2 spatial orbitals -> 4 spin orbitals -> 4 qubits."""
    pytest.importorskip("pyscf")
    pytest.importorskip("openfermion")
    mol = Molecule.from_xyz(H2_XYZ, basis="sto-3g")
    result = molecular_hamiltonian(mol)
    assert result.metadata["n_qubits"] == 4
    assert result.metadata["n_orbitals"] == 2
    assert result.metadata["n_electrons"] == 2


def test_h2_hamiltonian_vqe():
    """Full pipeline: Molecule -> Hamiltonian -> VQE. Energy should be near -1.137 Hartree."""
    pytest.importorskip("pyscf")
    pytest.importorskip("openfermion")
    pytest.importorskip("scipy")
    from quonic.algorithms import vqe

    mol = Molecule.from_xyz(H2_XYZ, basis="sto-3g")
    ham_result = molecular_hamiltonian(mol)
    vqe_result = vqe(
        ham_result.metadata["hamiltonian"],
        ham_result.metadata["n_qubits"],
        maxiter=500,
    )
    # H2 exact ground state energy at equilibrium is ~-1.137 Hartree
    assert vqe_result.value < -1.0
    assert vqe_result.value > -2.0


def test_lih_hamiltonian():
    """LiH/STO-3G should have 6 spatial orbitals -> 12 qubits."""
    pytest.importorskip("pyscf")
    pytest.importorskip("openfermion")
    mol = Molecule.from_xyz(LIH_XYZ, basis="sto-3g")
    result = molecular_hamiltonian(mol)
    assert result.metadata["n_orbitals"] == 6
    assert result.metadata["n_qubits"] == 12
    assert result.metadata["n_electrons"] == 4


def test_active_space_reduction():
    """CAS(2,2) on H2 should reduce to 2-qubit Hamiltonian."""
    pytest.importorskip("pyscf")
    pytest.importorskip("openfermion")
    mol = Molecule.from_xyz(H2_XYZ, basis="sto-3g")
    cas = select_active_space(mol, n_active_electrons=2, n_active_orbitals=2, method="manual")
    result = molecular_hamiltonian(mol, active_space=cas)
    assert result.metadata["n_qubits"] == 4  # 2 spatial * 2 spin
    assert result.metadata["n_orbitals"] == 2


def test_bravyi_kitaev_mapping():
    """Test BK mapping produces valid terms."""
    pytest.importorskip("pyscf")
    pytest.importorskip("openfermion")
    mol = Molecule.from_xyz(H2_XYZ, basis="sto-3g")
    result = molecular_hamiltonian(mol, mapping="bravyi_kitaev")
    terms = result.metadata["hamiltonian"]
    assert len(terms) > 0


def test_mf_energy_in_result():
    """SCF energy should be in metadata."""
    pytest.importorskip("pyscf")
    pytest.importorskip("openfermion")
    mol = Molecule.from_xyz(H2_XYZ, basis="sto-3g")
    result = molecular_hamiltonian(mol)
    # H2 RHF/STO-3G energy is ~-1.117 Hartree
    assert result.metadata["mf_energy"] < -1.0
    assert result.metadata["mf_energy"] > -1.5
