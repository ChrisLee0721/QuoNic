"""Tests for molecular_hamiltonian_from_integrals (PySCF-free path)."""

from __future__ import annotations

import numpy as np
import pytest

from quonic.chem import molecular_hamiltonian_from_integrals


def test_h2_from_integrals():
    """H2/STO-3G integrals → qubit Hamiltonian should produce valid terms."""
    pytest.importorskip("openfermion")

    # Known H2/STO-3G integrals (2 spatial orbitals)
    h1 = np.array([
        [-1.2533, -0.4759],
        [-0.4759, -0.4759],
    ])
    h2 = np.zeros((2, 2, 2, 2))
    # (00|00) = 0.6745
    h2[0, 0, 0, 0] = 0.6745
    # (00|11) = 0.6635
    h2[0, 0, 1, 1] = 0.6635
    h2[1, 1, 0, 0] = 0.6635
    # (11|11) = 0.6974
    h2[1, 1, 1, 1] = 0.6974
    # (01|01) = 0.1813
    h2[0, 1, 0, 1] = 0.1813
    h2[1, 0, 1, 0] = 0.1813
    # (01|10) = 0.6635
    h2[0, 1, 1, 0] = 0.6635
    h2[1, 0, 0, 1] = 0.6635

    result = molecular_hamiltonian_from_integrals(
        h1, h2, nuclear_repulsion=0.7199, n_electrons=2, n_orbitals=2
    )
    terms = result.metadata["hamiltonian"]
    assert len(terms) > 0
    assert result.metadata["n_qubits"] == 4
    assert result.metadata["n_orbitals"] == 2
    assert result.metadata["n_electrons"] == 2


def test_h2_terms_have_valid_paulis():
    """Each Pauli string should contain only valid characters."""
    pytest.importorskip("openfermion")

    h1 = np.array([[-1.0, 0.0], [0.0, -0.5]])
    h2 = np.zeros((2, 2, 2, 2))
    h2[0, 0, 0, 0] = 0.5
    h2[1, 1, 1, 1] = 0.5
    h2[0, 0, 1, 1] = 0.3
    h2[1, 1, 0, 0] = 0.3

    result = molecular_hamiltonian_from_integrals(
        h1, h2, nuclear_repulsion=0.5, n_electrons=2, n_orbitals=2
    )
    for coeff, pauli in result.metadata["hamiltonian"]:
        assert all(c in "IXYZ" for c in pauli)
        assert len(pauli) <= 4  # at most n_qubits


def test_bravyi_kitaev_mapping():
    pytest.importorskip("openfermion")

    h1 = np.array([[-1.0, 0.0], [0.0, -0.5]])
    h2 = np.zeros((2, 2, 2, 2))
    h2[0, 0, 0, 0] = 0.5

    result = molecular_hamiltonian_from_integrals(
        h1, h2, nuclear_repulsion=0.5, n_electrons=2, n_orbitals=2,
        mapping="bravyi_kitaev",
    )
    assert len(result.metadata["hamiltonian"]) > 0


def test_nuclear_repulsion_in_value():
    pytest.importorskip("openfermion")

    h1 = np.array([[-1.0]])
    h2 = np.zeros((1, 1, 1, 1))

    result = molecular_hamiltonian_from_integrals(
        h1, h2, nuclear_repulsion=1.234, n_electrons=1, n_orbitals=1,
    )
    assert abs(result.value - 1.234) < 1e-10
