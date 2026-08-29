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
from .dg import DGResult, compute_dg, compute_dg_barrier
from .embedding import DMET, DMETResult
from .formats import from_fcidump, from_mol2, from_pdb
from .fragment import Fragment, fragment_molecule
from .hamiltonian import molecular_hamiltonian, molecular_hamiltonian_from_integrals
from .molecule import Molecule
from .optimize import OptimizationResult, optimize_geometry, optimize_transition_state
from .solvation import differential_solvation, solvation_correction
from .thermo import ThermoResult, gibbs_free_energy, thermochemistry

__all__ = [
    "DMET",
    "ActiveSpace",
    "DGResult",
    "DMETResult",
    "Fragment",
    "Molecule",
    "OptimizationResult",
    "ThermoResult",
    "compute_dg",
    "compute_dg_barrier",
    "differential_solvation",
    "fragment_molecule",
    "from_fcidump",
    "from_mol2",
    "from_pdb",
    "gibbs_free_energy",
    "list_bases",
    "molecular_hamiltonian",
    "molecular_hamiltonian_from_integrals",
    "optimize_geometry",
    "optimize_transition_state",
    "select_active_space",
    "solvation_correction",
    "thermochemistry",
    "validate_basis",
]
