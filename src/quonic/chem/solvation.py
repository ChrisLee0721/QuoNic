"""Solvation corrections using implicit solvent models.

Provides solvation free energy calculations using PySCF's PCM
(Polarizable Continuum Model) implementation.

Example::

    from quonic.chem import Molecule, solvation_correction

    mol = Molecule.from_xyz('''
    3
    H2O
    O  0.0  0.0  0.0
    H  0.0  0.0  0.96
    H  0.0  0.96  0.0
    ''')
    dg_solv = solvation_correction(mol, solvent="water", method="b3lyp")
    print(dg_solv)  # Solvation free energy in Hartree
"""

from __future__ import annotations

from typing import Any

from .._i18n import tr
from .molecule import Molecule

# Common solvent dielectric constants
SOLVENT_DIELECTRIC: dict[str, float] = {
    "water": 78.3553,
    "h2o": 78.3553,
    "ethanol": 24.852,
    "etoh": 24.852,
    "methanol": 32.613,
    "meoh": 32.613,
    "acetone": 20.493,
    "dmso": 46.826,
    "chloroform": 4.7113,
    "chcl3": 4.7113,
    "dichloromethane": 8.93,
    "dcm": 8.93,
    "toluene": 2.3741,
    "hexane": 1.8819,
    "octanol": 9.8629,
    "vacuum": 1.0,
    "gas": 1.0,
}


def solvation_correction(
    molecule: Molecule,
    solvent: str = "water",
    method: str = "hf",
    basis: str | None = None,
) -> float:
    """Compute solvation free energy correction.

    Uses PySCF's PCM (Polarizable Continuum Model) to compute the
    solvation free energy difference between solution and gas phase.

    Args:
        molecule: Molecular geometry.
        solvent: Solvent name (water, ethanol, octanol, etc.) or
            dielectric constant as string (e.g., "78.3553").
        method: Electronic structure method.
        basis: Basis set name (defaults to molecule's basis).

    Returns:
        Solvation free energy ΔG_solv in Hartree (negative = stabilizing).

    Raises:
        ImportError: If PySCF is not installed.
        ValueError: If solvent is not recognized.
    """
    try:
        from pyscf import solvent as pyscf_solvent
    except ImportError as exc:
        raise ImportError(tr("err.chem.pyscf_missing")) from exc

    # Build PySCF molecule
    pyscf_mol = molecule.to_pyscf_mol(basis=basis)

    # Get dielectric constant
    eps = _get_dielectric(solvent)

    # Compute gas phase energy
    mf_gas = _create_solver(pyscf_mol, method)
    mf_gas.kernel()
    if not mf_gas.converged:
        raise RuntimeError(
            tr("err.chem.scf_not_converged", method=method)
        )
    e_gas = mf_gas.e_tot

    # Compute solution phase energy with PCM
    mf_sol = _create_solver(pyscf_mol, method)
    mf_sol = pyscf_solvent.PCM(mf_sol)
    mf_sol.with_solvent.eps = eps
    mf_sol.kernel()
    if not mf_sol.converged:
        raise RuntimeError(
            tr("err.chem.scf_not_converged", method=method)
        )
    e_sol = mf_sol.e_tot

    # Solvation free energy
    dg_solv = e_sol - e_gas

    return float(dg_solv)


def solvation_correction_custom(
    molecule: Molecule,
    dielectric: float,
    method: str = "hf",
    basis: str | None = None,
) -> float:
    """Compute solvation correction with custom dielectric constant.

    Args:
        molecule: Molecular geometry.
        dielectric: Solvent dielectric constant (ε).
        method: Electronic structure method.
        basis: Basis set name.

    Returns:
        Solvation free energy ΔG_solv in Hartree.
    """
    try:
        from pyscf import solvent as pyscf_solvent
    except ImportError as exc:
        raise ImportError(tr("err.chem.pyscf_missing")) from exc

    pyscf_mol = molecule.to_pyscf_mol(basis=basis)

    # Gas phase
    mf_gas = _create_solver(pyscf_mol, method)
    mf_gas.kernel()
    if not mf_gas.converged:
        raise RuntimeError(tr("err.chem.scf_not_converged", method=method))
    e_gas = mf_gas.e_tot

    # Solution phase
    mf_sol = _create_solver(pyscf_mol, method)
    mf_sol = pyscf_solvent.PCM(mf_sol)
    mf_sol.with_solvent.eps = dielectric
    mf_sol.kernel()
    if not mf_sol.converged:
        raise RuntimeError(tr("err.chem.scf_not_converged", method=method))
    e_sol = mf_sol.e_tot

    return float(e_sol - e_gas)


def differential_solvation(
    reactants: list[Molecule],
    products: list[Molecule],
    solvent: str = "water",
    method: str = "hf",
    basis: str | None = None,
) -> float:
    """Compute differential solvation correction for a reaction.

    ΔΔG_solv = Σ ΔG_solv(products) - Σ ΔG_solv(reactants)

    Args:
        reactants: List of reactant molecules.
        products: List of product molecules.
        solvent: Solvent name.
        method: Electronic structure method.
        basis: Basis set name.

    Returns:
        Differential solvation correction in Hartree.
    """
    dg_reactants = sum(
        solvation_correction(mol, solvent, method, basis) for mol in reactants
    )
    dg_products = sum(
        solvation_correction(mol, solvent, method, basis) for mol in products
    )

    return dg_products - dg_reactants


def _get_dielectric(solvent: str) -> float:
    """Get dielectric constant from solvent name.

    Args:
        solvent: Solvent name or dielectric constant string.

    Returns:
        Dielectric constant.

    Raises:
        ValueError: If solvent is not recognized.
    """
    solvent_lower = solvent.lower().strip()

    # Check if it's a known solvent name
    if solvent_lower in SOLVENT_DIELECTRIC:
        return SOLVENT_DIELECTRIC[solvent_lower]

    # Try to parse as float
    try:
        eps = float(solvent)
        if eps < 1.0:
            raise ValueError(
                tr("err.chem.solvent_invalid", solvent=solvent)
            )
        return eps
    except ValueError:
        pass

    raise ValueError(
        tr("err.chem.solvent_unknown", solvent=solvent)
        + f" Known solvents: {', '.join(sorted(SOLVENT_DIELECTRIC.keys()))}"
    )


def _create_solver(mol: Any, method: str):
    """Create PySCF SCF solver for given method.

    Args:
        mol: PySCF Mole object.
        method: Method name (hf, b3lyp, mp2, etc.).

    Returns:
        PySCF solver object.
    """
    from pyscf import dft, scf

    method_lower = method.lower()

    if method_lower == "hf":
        if mol.spin == 0:
            return scf.RHF(mol)
        else:
            return scf.UHF(mol)
    elif method_lower in ("b3lyp", "pbe", "pbe0", "m06", "m06-2x", "wb97x"):
        if mol.spin == 0:
            mf = dft.RKS(mol)
        else:
            mf = dft.UKS(mol)
        mf.xc = method_lower
        return mf
    else:
        # Try as DFT functional
        if mol.spin == 0:
            mf = dft.RKS(mol)
        else:
            mf = dft.UKS(mol)
        mf.xc = method_lower
        return mf
