"""Density Matrix Embedding Theory (DMET) solver.

Provides a simplified DMET implementation that partitions a molecule into
fragments, solves each fragment in an embedding environment, and iterates
to self-consistency.

Example::

    from quonic.chem import Molecule, DMET

    mol = Molecule.from_xyz('''
    4
    Linear H4
    H  0.0  0.0  0.0
    H  0.0  0.0  0.74
    H  0.0  0.0  2.0
    H  0.0  0.0  2.74
    ''')
    dmet = DMET(mol, fragment_size=2)
    result = dmet.solve()
    print(result.energy)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .._i18n import tr
from .fragment import Fragment, fragment_molecule


@dataclass
class DMETResult:
    """Result of a DMET calculation.

    Attributes:
        energy: Total DMET energy.
        fragment_energies: Per-fragment energies.
        converged: Whether self-consistency was achieved.
        n_iterations: Number of DMET iterations performed.
        chemical_potential: Fitted chemical potential.
    """

    energy: float
    fragment_energies: list[float]
    converged: bool
    n_iterations: int
    chemical_potential: float = 0.0


class DMET:
    """Density Matrix Embedding Theory solver.

    Args:
        molecule: A ``quonic.chem.Molecule``.
        fragment_size: Maximum atoms per fragment.
        basis: Override basis set (``None`` = use molecule's basis).
        solver: Fragment solver — ``"fci"`` or ``"casscf"``.
        max_iter: Maximum DMET iterations.
        tol: Convergence tolerance on the chemical potential residual.
    """

    def __init__(
        self,
        molecule: Any,
        fragment_size: int = 2,
        basis: str | None = None,
        solver: str = "fci",
        max_iter: int = 50,
        tol: float = 1e-6,
    ) -> None:
        self.molecule = molecule
        self.fragment_size = fragment_size
        self.basis = basis or molecule.basis
        self.solver = solver
        self.max_iter = max_iter
        self.tol = tol

    def solve(self) -> DMETResult:
        """Run the self-consistent DMET calculation.

        Returns:
            A :class:`DMETResult` with the total energy and convergence info.
        """
        try:
            from pyscf import fci, scf  # noqa: F401
        except ImportError as exc:
            raise ImportError(tr("err.chem.pyscf_missing")) from exc

        # Build full molecule in PySCF
        pyscf_mol = self.molecule.to_pyscf_mol(basis=self.basis)

        # Step 1: Mean-field on the full system
        mf = scf.RHF(pyscf_mol)
        mf.verbose = 0
        mf.kernel()

        # Step 2: Fragment the molecule
        fragments = fragment_molecule(
            self.molecule,
            max_fragment_size=self.fragment_size,
        )

        # Step 3: Self-consistent DMET loop
        mu = 0.0  # chemical potential
        frag_energies: list[float] = []
        converged = False

        for iteration in range(self.max_iter):
            total_energy = 0.0
            frag_energies = []
            total_electron_diff = 0.0

            for frag in fragments:
                # Build embedding Hamiltonian for this fragment
                h_emb, n_emb_elec = self._build_embedding_hamiltonian(
                    frag, mf, pyscf_mol, mu
                )

                # Solve the fragment
                e_frag, dm_frag = self._solve_fragment(h_emb, n_emb_elec)
                frag_energies.append(e_frag)
                total_energy += e_frag

                # Track electron count mismatch for chemical potential fitting
                n_frag_elec_expected = self._count_fragment_electrons(frag)
                n_frag_elec_actual = np.trace(dm_frag).real
                total_electron_diff += n_frag_elec_actual - n_frag_elec_expected

            # Update chemical potential
            mu += total_electron_diff * 0.5  # simple mixing

            # Check convergence
            if abs(total_electron_diff) < self.tol:
                converged = True
                break

        return DMETResult(
            energy=total_energy,
            fragment_energies=frag_energies,
            converged=converged,
            n_iterations=iteration + 1,
            chemical_potential=mu,
        )

    def _build_embedding_hamiltonian(
        self,
        fragment: Fragment,
        mf: Any,
        pyscf_mol: Any,
        mu: float,
    ) -> tuple[Any, int]:
        """Construct the embedding Hamiltonian for a fragment.

        Returns (h_emb, n_emb_electrons).
        """
        import numpy as np

        # Get AO indices for fragment atoms
        frag_atoms = set(fragment.atom_indices)
        ao_labels = pyscf_mol.ao_labels()
        ao_indices = []
        for i, label in enumerate(ao_labels):
            # PySCF labels are like "0 H 1s"
            atom_idx = int(label.split()[0])
            if atom_idx in frag_atoms:
                ao_indices.append(i)

        ao_indices = np.array(ao_indices)
        n_frag_ao = len(ao_indices)

        # Get the full system MO coefficients
        C = mf.mo_coeff
        n_orb = C.shape[1]

        # Project fragment AOs onto the full MO space
        # Fragment occupied orbitals
        n_occ = pyscf_mol.nelectron // 2
        C_occ = C[:, :n_occ]
        _C_virt = C[:, n_occ:]

        # Fragment projection
        S = pyscf_mol.intor("int1e_ovlp")
        P_frag = np.zeros((n_orb, n_orb))
        for i in ao_indices:
            for j in ao_indices:
                P_frag += S[i, j] * np.outer(C[i], C[j])

        # Embedding space: fragment + bath (from SVD of occupied fragment)
        C_frag_occ = C_occ[ao_indices]
        _U, _sigma, Vt = np.linalg.svd(C_frag_occ, full_matrices=True)

        # Number of significant bath orbitals
        n_bath = min(n_frag_ao, n_occ)
        n_emb = n_frag_ao + n_bath

        # Build embedding orbitals (fragment + bath)
        C_emb_frag = np.eye(pyscf_mol.nao_nr())[:, ao_indices]  # fragment AOs
        C_bath = C_occ @ Vt[:n_bath].T  # bath orbitals from occupied space

        # Combine into embedding space
        C_emb = np.hstack([C_emb_frag, C_bath])

        # Build embedding integrals
        h_core = mf.get_hcore()
        h1_emb = C_emb.T @ h_core @ C_emb

        h2_ao = pyscf_mol.intor("int2e", aosym="s1")
        h2_emb = np.einsum("pqrs,pi,qj,rk,sl->ijkl", h2_ao, C_emb, C_emb, C_emb, C_emb)

        # Add chemical potential to fragment part
        for i in range(n_frag_ao):
            h1_emb[i, i] -= mu

        n_emb_elec = pyscf_mol.nelectron  # total electrons in embedding space
        # Adjust for the fact that we're in a reduced space
        n_emb_elec = min(n_emb_elec, 2 * n_emb)

        return (h1_emb, h2_emb), n_emb_elec

    def _solve_fragment(
        self,
        h_emb: tuple[Any, Any],
        n_electrons: int,
    ) -> tuple[float, Any]:
        """Solve the embedding Hamiltonian with FCI.

        Returns (energy, one_particle_density_matrix).
        """
        import numpy as np
        from pyscf import fci

        h1_emb, h2_emb = h_emb
        n_orb = h1_emb.shape[0]
        n_e = min(n_electrons, 2 * n_orb)
        n_alpha = n_e // 2
        n_beta = n_e - n_alpha

        # Ensure we have valid electron counts
        if n_alpha < 0 or n_beta < 0 or n_alpha > n_orb or n_beta > n_orb:
            return 0.0, np.zeros((n_orb, n_orb))

        # FCI solver
        cisolver = fci.FCI()
        cisolver.verbose = 0
        e, ci = cisolver.kernel(h1_emb, h2_emb, n_orb, (n_alpha, n_beta))

        # One-particle density matrix
        dm1 = cisolver.make_rdm1(ci, n_orb, (n_alpha, n_beta))

        return float(e), dm1

    def _count_fragment_electrons(self, fragment: Fragment) -> float:
        """Count expected electrons in a fragment."""
        from .molecule import _ATOMIC_NUMBERS

        return sum(_ATOMIC_NUMBERS.get(a, 0) for a in fragment.atoms)
