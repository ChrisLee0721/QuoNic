"""Active space selection for CAS calculations.

Provides manual and automatic active space selection for
complete active space (CAS) methods.

Example::

    from quonic.chem import Molecule, select_active_space

    mol = Molecule.from_xyz('''4\\nLiH\\nLi 0 0 0\\nH 0 0 1.6''')
    cas = select_active_space(mol, n_active_electrons=2, n_active_orbitals=2)
    print(cas)  # ActiveSpace(n_electrons=2, n_orbitals=2, ...)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .._i18n import tr


@dataclass(frozen=True)
class ActiveSpace:
    """Describes a CAS(n,m) active space.

    Attributes:
        n_electrons: Number of active electrons.
        n_orbitals: Number of active spatial orbitals.
        orbital_indices: Indices of selected orbitals (0-based, spatial).
    """

    n_electrons: int
    n_orbitals: int
    orbital_indices: tuple[int, ...]


def select_active_space(
    molecule: Any,
    n_active_electrons: int | None = None,
    n_active_orbitals: int | None = None,
    method: str = "manual",
    n_qubits: int | None = None,
) -> ActiveSpace:
    """Select an active space for CAS calculation.

    Args:
        molecule: A ``quonic.chem.Molecule`` or PySCF ``Mole`` object.
        n_active_electrons: Number of active electrons (required for ``"manual"``).
        n_active_orbitals: Number of active spatial orbitals (required for ``"manual"``).
        method: Selection strategy:

            - ``"manual"`` — use *n_active_electrons* and *n_active_orbitals* directly.
            - ``"full_valence"`` — all valence electrons and orbitals.
            - ``"avas"`` — Automated Valence Active Space (requires PySCF).

        n_qubits: Maximum number of qubits (active orbitals × 2).
            Used by ``"avas"`` to constrain active space size.

    Returns:
        An :class:`ActiveSpace` descriptor.

    Raises:
        ValueError: If parameters are invalid.
    """
    if method == "manual":
        return _manual(molecule, n_active_electrons, n_active_orbitals)
    elif method == "full_valence":
        return _full_valence(molecule)
    elif method == "avas":
        return _avas(molecule, n_active_orbitals, n_qubits)
    else:
        raise ValueError(
            tr("err.chem.active_space_auto")
            + f" Unknown method: {method!r}"
        )


def _manual(
    molecule: Any,
    n_e: int | None,
    n_o: int | None,
) -> ActiveSpace:
    if n_e is None or n_o is None:
        raise ValueError(
            "Manual active space requires both n_active_electrons and n_active_orbitals."
        )
    n_electrons = _get_n_electrons(molecule)
    if n_e <= 0 or n_o <= 0 or n_e > n_electrons or 2 * n_o < n_e:
        n_orbitals = _get_n_orbitals_safe(molecule)
        raise ValueError(
            tr(
                "err.chem.active_space_invalid",
                n_e=n_e,
                n_o=n_o,
                elec=n_electrons,
                orb=n_orbitals,
            )
        )
    indices = tuple(range(n_o))
    return ActiveSpace(n_electrons=n_e, n_orbitals=n_o, orbital_indices=indices)


def _avas(
    molecule: Any,
    n_active_orbitals: int | None = None,
    n_qubits: int | None = None,
) -> ActiveSpace:
    """Automated Valence Active Space (AVAS) selection.

    Uses PySCF's AVAS method to automatically select the active space
    based on orbital character (valence vs core).

    Args:
        molecule: A ``quonic.chem.Molecule`` or PySCF ``Mole`` object.
        n_active_orbitals: Target number of active orbitals (optional).
        n_qubits: Maximum qubits (constrains n_active_orbitals = n_qubits // 2).

    Returns:
        An :class:`ActiveSpace` descriptor.

    Raises:
        ImportError: If PySCF is not installed.
    """
    try:
        from pyscf import scf
        from pyscf.mcscf import avas
    except ImportError as exc:
        raise ImportError(tr("err.chem.pyscf_missing")) from exc

    from .molecule import Molecule as QuoNicMolecule

    # Get PySCF molecule
    if isinstance(molecule, QuoNicMolecule):
        pyscf_mol = molecule.to_pyscf_mol()
    else:
        pyscf_mol = molecule

    # Run HF first
    if pyscf_mol.spin == 0:
        mf = scf.RHF(pyscf_mol)
    else:
        mf = scf.UHF(pyscf_mol)
    mf.kernel()

    # Determine target active orbitals
    if n_qubits is not None:
        target_orbs = n_qubits // 2
    elif n_active_orbitals is not None:
        target_orbs = n_active_orbitals
    else:
        target_orbs = None

    # Run AVAS
    if target_orbs is not None:
        n_elec, n_orb, orbs = avas.avas(mf, "valence", ncas=target_orbs)
    else:
        n_elec, n_orb, orbs = avas.avas(mf, "valence")

    # Get orbital indices (AVAS returns MO coefficients, need to determine which are active)
    # The orbs array contains the AVAS-selected orbital coefficients
    # We need to find which original orbitals are selected
    import numpy as np

    # Project AVAS orbitals back to original basis
    mo_coeff = mf.mo_coeff
    if mo_coeff.ndim == 2:
        # RHF case
        overlap = pyscf_mol.intor("int1e_ovlp")
        # Find which original orbitals have significant overlap with AVAS space
        proj = mo_coeff.T @ overlap @ orbs
        orbital_importance = np.sum(proj ** 2, axis=1)
        active_indices = np.argsort(orbital_importance)[-n_orb:]
        active_indices = tuple(sorted(active_indices.tolist()))
    else:
        # UHF case - use alpha spin
        overlap = pyscf_mol.intor("int1e_ovlp")
        proj = mo_coeff[0].T @ overlap @ orbs
        orbital_importance = np.sum(proj ** 2, axis=1)
        active_indices = np.argsort(orbital_importance)[-n_orb:]
        active_indices = tuple(sorted(active_indices.tolist()))

    return ActiveSpace(
        n_electrons=int(n_elec),
        n_orbitals=int(n_orb),
        orbital_indices=active_indices,
    )


def _full_valence(molecule: Any) -> ActiveSpace:
    """Approximate full-valence active space.

    Uses a simple heuristic: count valence electrons based on the
    periodic table block (s/p/d) and select the same number of orbitals.
    """
    from .molecule import _ATOMIC_NUMBERS

    atoms = _get_atoms(molecule)
    n_valence = 0
    for atom in atoms:
        z = _ATOMIC_NUMBERS.get(atom, 0)
        if z <= 2:
            n_valence += z  # H, He — all electrons are valence
        elif z <= 10:
            n_valence += z - 2  # Li–Ne — 2 core electrons
        elif z <= 18:
            n_valence += z - 10  # Na–Ar — 10 core electrons
        elif z <= 36:
            n_valence += z - 18  # K–Kr — 18 core electrons
        else:
            n_valence += z - 28  # rough fallback

    n_electrons = _get_n_electrons(molecule)
    # Each spatial orbital holds 2 electrons; need at least as many orbitals
    # as half the valence electrons
    n_orbitals = max(n_valence // 2, 1)
    # Cap at total available orbitals (approximate: n_electrons // 2 for RHF)
    total_orbs = n_electrons // 2 + n_electrons % 2
    n_orbitals = min(n_orbitals, total_orbs)
    indices = tuple(range(n_orbitals))
    return ActiveSpace(
        n_electrons=min(n_valence, n_electrons),
        n_orbitals=n_orbitals,
        orbital_indices=indices,
    )


# ------------------------------------------------------------------
# Helpers to extract info from Molecule or PySCF Mole
# ------------------------------------------------------------------

def _get_atoms(mol: Any) -> tuple[str, ...]:
    if hasattr(mol, "atoms"):
        return mol.atoms
    # PySCF Mole
    if hasattr(mol, "atom_charges"):
        from .molecule import _ATOMIC_NUMBERS
        _Z_TO_SYM = {v: k for k, v in _ATOMIC_NUMBERS.items()}
        return tuple(_Z_TO_SYM.get(z, "?") for z in mol.atom_charges())
    raise TypeError("Unsupported molecule type")


def _get_n_electrons(mol: Any) -> int:
    if hasattr(mol, "n_electrons"):
        return mol.n_electrons
    if hasattr(mol, "nelectron"):
        return mol.nelectron
    raise TypeError("Cannot determine electron count")


def _get_n_orbitals(mol: Any) -> int:
    if hasattr(mol, "n_orbitals"):
        return mol.n_orbitals
    # PySCF Mole — approximate from nao_nr()
    if hasattr(mol, "nao_nr"):
        return mol.nao_nr()
    raise TypeError("Cannot determine orbital count")


def _get_n_orbitals_safe(mol: Any) -> int | str:
    """Like _get_n_orbitals but returns '?' on failure."""
    try:
        return _get_n_orbitals(mol)
    except (TypeError, AttributeError):
        return "?"
