"""Quantum Chemistry — molecular Hamiltonian generation for VQE.

Provides the full pipeline: molecule geometry → SCF → active space →
second-quantized Hamiltonian → qubit Hamiltonian → VQE.

Example::

    from quonic.chem import Molecule, molecular_hamiltonian
    from quonic.algorithms import vqe

    mol = Molecule.from_xyz('''
    2
    H2 at equilibrium
    H  0.0  0.0  0.0
    H  0.0  0.0  0.74
    ''')
    result = molecular_hamiltonian(mol)
    vqe_result = vqe(result.metadata["hamiltonian"], result.metadata["n_qubits"])
"""

from __future__ import annotations

from .active_space import ActiveSpace, select_active_space
from .basis import list_bases, validate_basis
from .embedding import DMET, DMETResult
from .formats import from_fcidump, from_mol2, from_pdb
from .fragment import Fragment, fragment_molecule
from .hamiltonian import molecular_hamiltonian, molecular_hamiltonian_from_integrals
from .molecule import Molecule

__all__ = [
    "DMET",
    "ActiveSpace",
    "DMETResult",
    "Fragment",
    "Molecule",
    "fragment_molecule",
    "from_fcidump",
    "from_mol2",
    "from_pdb",
    "list_bases",
    "molecular_hamiltonian",
    "molecular_hamiltonian_from_integrals",
    "select_active_space",
    "validate_basis",
]
