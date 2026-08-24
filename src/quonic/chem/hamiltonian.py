"""Molecular Hamiltonian generation pipeline.

Converts a molecular geometry into a qubit Hamiltonian in QuoNic format
via PySCF (SCF + integrals) and OpenFermion (fermion-to-qubit mapping).

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

from typing import Any

from .._i18n import tr
from ..result import Result


def molecular_hamiltonian(
    molecule: Any,
    basis: str | None = None,
    active_space: Any | None = None,
    method: str = "rhf",
    mapping: str = "jordan_wigner",
) -> Result:
    """Compute the qubit Hamiltonian for a molecule.

    Pipeline:
        1. Run PySCF SCF (RHF/UHF based on spin).
        2. Optionally apply CAS active space reduction.
        3. Extract one-/two-electron integrals.
        4. Build OpenFermion ``FermionOperator``.
        5. Apply qubit mapping (Jordan-Wigner or Bravyi-Kitaev).
        6. Convert to QuoNic ``[(coeff, pauli_string), ...]`` format.

    Args:
        molecule: A ``quonic.chem.Molecule`` or PySCF ``Mole``.
        basis: Override basis set (``None`` = use molecule's basis).
        active_space: An :class:`ActiveSpace` descriptor (``None`` = full space).
        method: ``"rhf"`` or ``"uhf"``.
        mapping: ``"jordan_wigner"`` or ``"bravyi_kitaev"``.

    Returns:
        ``Result.from_value(nuclear_repulsion, ...)`` with metadata keys:

        - ``"hamiltonian"`` — ``[(coeff, pauli_string), ...]``
        - ``"n_qubits"`` — number of qubits
        - ``"mf_energy"`` — SCF energy
        - ``"n_orbitals"`` — number of spatial orbitals
        - ``"n_electrons"`` — number of electrons
    """
    # Lazy imports
    try:
        from pyscf import scf  # noqa: F401
    except ImportError as exc:
        raise ImportError(tr("err.chem.pyscf_missing")) from exc
    try:
        import openfermion  # noqa: F401
    except ImportError as exc:
        raise ImportError(tr("err.chem.openfermion_missing")) from exc

    # Resolve molecule to PySCF Mole
    from .molecule import Molecule

    if isinstance(molecule, Molecule):
        pyscf_mol = molecule.to_pyscf_mol(basis=basis)
    else:
        pyscf_mol = molecule

    # Step 1: SCF
    mf = _run_pyscf_scf(pyscf_mol, method=method)
    mf_energy = float(mf.e_tot)

    # Step 2-3: Integrals (optionally in active space)
    h1, h2, nuclear_repulsion, n_orbitals = _get_integrals(mf, active_space)

    # Step 4: Fermion operator
    fermion_op = _build_fermion_operator(h1, h2, nuclear_repulsion)

    # Step 5: Qubit mapping
    qubit_op = _map_to_qubits(fermion_op, mapping)

    # Step 6: Convert to QuoNic format
    n_qubits = 2 * n_orbitals  # spin orbitals
    terms = _to_quonic_format(qubit_op, n_qubits)

    n_electrons = pyscf_mol.nelectron
    if active_space is not None:
        n_electrons = active_space.n_electrons

    return Result.from_value(
        nuclear_repulsion,
        hamiltonian=terms,
        n_qubits=n_qubits,
        mf_energy=mf_energy,
        n_orbitals=n_orbitals,
        n_electrons=n_electrons,
    )


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _run_pyscf_scf(mol: Any, method: str = "rhf") -> Any:
    """Run PySCF SCF and return the mean-field object."""
    from pyscf import scf

    if method == "uhf":
        mf = scf.UHF(mol)
    else:
        mf = scf.RHF(mol)
    mf.verbose = 0
    conv = mf.kernel()
    if not conv:
        raise RuntimeError(tr("err.chem.scf_converge", max_cycle=mf.max_cycle))
    return mf


def _get_integrals(
    mf: Any,
    active_space: Any | None = None,
) -> tuple[Any, Any, float, int]:
    """Extract (h1, h2, nuclear_repulsion, n_orbitals) from mean-field.

    If *active_space* is provided, uses CASCI to obtain active-space integrals.
    """
    mol = mf.mol

    if active_space is not None:
        return _get_casci_integrals(mf, active_space)

    # Full space — get integrals from the mean-field object
    h1 = mf.get_hcore()
    h2_ao = mol.intor("int2e", aosym="s1")

    # Transform to MO basis
    C = mf.mo_coeff
    import numpy as np

    h1_mo = C.T @ h1 @ C
    # Two-electron integrals: (ij|kl) in chemist's notation
    n = C.shape[1]
    h2_mo = np.einsum("pqrs,pi,qj,rk,sl->ijkl", h2_ao, C, C, C, C)

    nuclear_repulsion = float(mol.energy_nuc())
    return h1_mo, h2_mo, nuclear_repulsion, n


def _get_casci_integrals(
    mf: Any,
    active_space: Any,
) -> tuple[Any, Any, float, int]:
    """Get integrals in the active space via CASCI."""
    import numpy as np
    from pyscf import mcscf

    n_e = active_space.n_electrons
    n_o = active_space.n_orbitals
    orb_indices = list(active_space.orbital_indices)

    mc = mcscf.CASCI(mf, n_o, n_e)
    mc.verbose = 0

    # Get the active-space integrals
    mo_active = mf.mo_coeff[:, orb_indices]
    h1_active = mo_active.T @ mf.get_hcore() @ mo_active

    mol = mf.mol
    h2_ao = mol.intor("int2e", aosym="s1")
    h2_active = np.einsum("pqrs,pi,qj,rk,sl->ijkl", h2_ao, mo_active, mo_active, mo_active, mo_active)

    nuclear_repulsion = float(mol.energy_nuc())
    return h1_active, h2_active, nuclear_repulsion, n_o


def _build_fermion_operator(
    h1: Any,
    h2: Any,
    nuclear_repulsion: float,
) -> Any:
    """Build OpenFermion FermionOperator from one-/two-electron integrals."""
    import numpy as np
    import openfermion

    n_orbitals = h1.shape[0]
    _n_spin_orbitals = 2 * n_orbitals

    # Build molecular Hamiltonian using OpenFermion's MolecularData conventions
    # h1 is (n_orb, n_orb) one-electron integrals in spatial MO basis
    # h2 is (n_orb, n_orb, n_orb, n_orb) two-electron integrals in chemist's notation (ij|kl)

    # Convert to physicist's notation for OpenFermion
    # (ij|kl) -> <ik|jl>
    h2_phys = np.einsum("ijkl->ikjl", h2)

    # Build the second-quantized operator
    fermion_op = openfermion.FermionOperator((), nuclear_repulsion)

    # One-electron terms: sum_pq h1[p,q] a†_p a_q (spin-orbital)
    for p in range(n_orbitals):
        for q in range(n_orbitals):
            coeff = h1[p, q]
            if abs(coeff) < 1e-12:
                continue
            # Alpha spin
            pa, qa = 2 * p, 2 * q
            fermion_op += openfermion.FermionOperator(
                ((pa, 1), (qa, 0)), coeff
            )
            # Beta spin
            pb, qb = 2 * p + 1, 2 * q + 1
            fermion_op += openfermion.FermionOperator(
                ((pb, 1), (qb, 0)), coeff
            )

    # Two-electron terms: 0.5 * sum_pqrs h2_phys[p,q,r,s] a†_p a†_q a_r a_s
    for p in range(n_orbitals):
        for q in range(n_orbitals):
            for r in range(n_orbitals):
                for s in range(n_orbitals):
                    coeff = h2_phys[p, q, r, s]
                    if abs(coeff) < 1e-12:
                        continue
                    # Alpha-Alpha
                    pa, qa, ra, sa = 2*p, 2*q, 2*r, 2*s
                    fermion_op += openfermion.FermionOperator(
                        ((pa, 1), (qa, 1), (ra, 0), (sa, 0)),
                        0.5 * coeff,
                    )
                    # Beta-Beta
                    pb, qb, rb, sb = 2*p+1, 2*q+1, 2*r+1, 2*s+1
                    fermion_op += openfermion.FermionOperator(
                        ((pb, 1), (qb, 1), (rb, 0), (sb, 0)),
                        0.5 * coeff,
                    )
                    # Alpha-Beta
                    fermion_op += openfermion.FermionOperator(
                        ((pa, 1), (qb, 1), (rb, 0), (sa, 0)),
                        0.5 * coeff,
                    )
                    # Beta-Alpha
                    fermion_op += openfermion.FermionOperator(
                        ((pb, 1), (qa, 1), (ra, 0), (sb, 0)),
                        0.5 * coeff,
                    )

    return fermion_op


def _map_to_qubits(fermion_op: Any, mapping: str = "jordan_wigner") -> Any:
    """Apply qubit mapping to a FermionOperator."""
    import openfermion

    if mapping == "jordan_wigner":
        return openfermion.jordan_wigner(fermion_op)
    elif mapping == "bravyi_kitaev":
        return openfermion.bravyi_kitaev(fermion_op)
    else:
        raise ValueError(tr("err.chem.mapping_unknown", mapping=mapping))


def _to_quonic_format(qubit_op: Any, n_qubits: int) -> list[tuple[float, str]]:
    """Convert OpenFermion QubitOperator to QuoNic format.

    Delegates to the existing ``from_openfermion()`` from
    ``quonic.algorithms.hamiltonians_ext``.
    """
    from ..algorithms.hamiltonians_ext import from_openfermion

    return from_openfermion(qubit_op)


# ------------------------------------------------------------------
# Path B: from pre-computed integrals (no PySCF needed)
# ------------------------------------------------------------------

def molecular_hamiltonian_from_integrals(
    h1: Any,
    h2: Any,
    nuclear_repulsion: float,
    n_electrons: int,
    n_orbitals: int,
    mapping: str = "jordan_wigner",
) -> Result:
    """Build a qubit Hamiltonian from pre-computed integrals.

    This is the PySCF-free path — users provide integrals computed by
    any quantum chemistry software (Gaussian, ORCA, NWChem, PSI4, etc.).

    Args:
        h1: One-electron integrals, shape ``(n_orb, n_orb)``.
        h2: Two-electron integrals in chemist's notation ``(ij|kl)``,
            shape ``(n_orb, n_orb, n_orb, n_orb)``.
        nuclear_repulsion: Nuclear repulsion energy.
        n_electrons: Total number of electrons.
        n_orbitals: Number of spatial orbitals.
        mapping: ``"jordan_wigner"`` or ``"bravyi_kitaev"``.

    Returns:
        ``Result.from_value(nuclear_repulsion, hamiltonian=terms,
        n_qubits=..., n_orbitals=..., n_electrons=...)``

    Example::

        import numpy as np
        from quonic.chem import molecular_hamiltonian_from_integrals
        from quonic.algorithms import vqe

        # Load integrals from your QC software
        h1 = np.load("h1.npy")
        h2 = np.load("h2.npy")
        result = molecular_hamiltonian_from_integrals(
            h1, h2, nuclear_repulsion=0.7199, n_electrons=2, n_orbitals=2
        )
        vqe_result = vqe(result.metadata["hamiltonian"], result.metadata["n_qubits"])
    """
    try:
        import openfermion  # noqa: F401
    except ImportError as exc:
        raise ImportError(tr("err.chem.openfermion_missing")) from exc

    import numpy as np

    h1 = np.asarray(h1)
    h2 = np.asarray(h2)

    fermion_op = _build_fermion_operator(h1, h2, nuclear_repulsion)
    qubit_op = _map_to_qubits(fermion_op, mapping)

    n_qubits = 2 * n_orbitals
    terms = _to_quonic_format(qubit_op, n_qubits)

    return Result.from_value(
        nuclear_repulsion,
        hamiltonian=terms,
        n_qubits=n_qubits,
        n_orbitals=n_orbitals,
        n_electrons=n_electrons,
    )
