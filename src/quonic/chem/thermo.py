"""Thermochemical corrections from frequency calculations.

Computes zero-point energy (ZPE), enthalpy, entropy, and Gibbs free
energy using the harmonic oscillator / rigid rotor approximation.

Example::

    from quonic.chem import Molecule, thermochemistry

    mol = Molecule.from_xyz('''
    3
    H2O
    O  0.0  0.0  0.0
    H  0.0  0.0  0.96
    H  0.0  0.96  0.0
    ''')
    result = thermochemistry(mol, method="b3lyp", basis="6-31g*")
    print(result.gibbs)  # Gibbs free energy in Hartree
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .._i18n import tr
from .molecule import Molecule

# Physical constants
HARTREE_TO_KCAL = 627.509  # 1 Hartree = 627.509 kcal/mol
KB = 3.166811563e-6  # Boltzmann constant in Hartree/K
PLANCK = 1.054571817e-34  # Reduced Planck constant in J·s
C = 2.99792458e10  # Speed of light in cm/s
AMU_TO_KG = 1.66053906660e-27  # Atomic mass unit in kg
BOHR_TO_M = 5.29177210903e-11  # Bohr radius in meters


@dataclass
class ThermoResult:
    """Result of thermochemical analysis.

    All energies are in Hartree, entropy in Hartree/K, frequencies in cm⁻¹.

    Attributes:
        zpe: Zero-point energy.
        enthalpy: Total enthalpy H = E_elec + ZPE + H_thermal.
        entropy: Total entropy S.
        gibbs: Gibbs free energy G = H - T·S.
        electronic_energy: Electronic energy from SCF.
        temperature: Temperature used (K).
        pressure: Pressure used (atm).
        frequencies: Vibrational frequencies (cm⁻¹).
        n_imaginary: Number of imaginary frequencies.
    """

    zpe: float
    enthalpy: float
    entropy: float
    gibbs: float
    electronic_energy: float
    temperature: float
    pressure: float
    frequencies: list[float]
    n_imaginary: int


def thermochemistry(
    molecule: Molecule,
    method: str = "hf",
    basis: str | None = None,
    temperature: float = 298.15,
    pressure: float = 1.0,
) -> ThermoResult:
    """Compute thermochemical corrections from frequency calculation.

    Uses the harmonic oscillator / rigid rotor approximation to compute
    ZPE, enthalpy, entropy, and Gibbs free energy.

    Args:
        molecule: Molecular geometry (should be at optimized structure).
        method: Electronic structure method.
        basis: Basis set name (defaults to molecule's basis).
        temperature: Temperature in Kelvin.
        pressure: Pressure in atm.

    Returns:
        A :class:`ThermoResult` with all thermodynamic quantities.

    Raises:
        ImportError: If PySCF is not installed.
        RuntimeError: If frequency calculation fails.
    """
    try:
        from pyscf import gto, hessian, scf
        from pyscf.hessian import thermo
    except ImportError as exc:
        raise ImportError(tr("err.chem.pyscf_missing")) from exc

    import numpy as np

    # Build PySCF molecule
    pyscf_mol = molecule.to_pyscf_mol(basis=basis)

    # Create SCF solver
    mf = _create_solver(pyscf_mol, method)

    # Run SCF
    mf.kernel()
    if not mf.converged:
        raise RuntimeError(
            tr("err.chem.scf_not_converged", method=method)
        )

    # Compute Hessian
    hess = mf.Hessian()
    hess_matrix = hess.kernel()

    # Get vibrational frequencies and normal modes
    freq_result = thermo.harmonic_analysis(pyscf_mol, hess_matrix)
    freqs = freq_result["freq_au"]  # atomic units (Hartree)
    freq_result["norm_mode"]

    # Count imaginary frequencies
    n_imaginary = int(np.sum(np.array(freqs) < -10))  # threshold for imaginary

    # Get atom masses
    np.array(pyscf_mol.atom_mass_list())

    # Compute thermochemical corrections
    # Using PySCF's built-in thermochemistry module
    # Note: pressure in PySCF is in Pascal (1 atm = 101325 Pa)
    pressure_pa = pressure * 101325
    thermo_result = thermo.thermo(
        mf,
        freqs,
        temperature,
        pressure_pa,
    )

    # Extract results from dictionary
    e_tot = mf.e_tot
    zpe = thermo_result['ZPE'][0]
    h_tot = thermo_result['H_tot'][0]
    g_tot = thermo_result['G_tot'][0]
    s_tot = thermo_result['S_tot'][0]

    return ThermoResult(
        zpe=float(zpe),
        enthalpy=float(h_tot),
        entropy=float(s_tot),
        gibbs=float(g_tot),
        electronic_energy=float(e_tot),
        temperature=temperature,
        pressure=pressure,
        frequencies=[float(f) for f in freqs if f > -10],  # exclude imaginary
        n_imaginary=n_imaginary,
    )


def gibbs_free_energy(
    molecule: Molecule,
    method: str = "hf",
    basis: str | None = None,
    temperature: float = 298.15,
    pressure: float = 1.0,
) -> float:
    """Compute Gibbs free energy at given temperature and pressure.

    Convenience wrapper around :func:`thermochemistry` that returns
    only the Gibbs free energy.

    Args:
        molecule: Molecular geometry (should be at optimized structure).
        method: Electronic structure method.
        basis: Basis set name.
        temperature: Temperature in Kelvin.
        pressure: Pressure in atm.

    Returns:
        Gibbs free energy in Hartree.
    """
    result = thermochemistry(molecule, method, basis, temperature, pressure)
    return result.gibbs


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
    elif method_lower == "mp2":
        if mol.spin == 0:
            return scf.RHF(mol).run().MP2()
        else:
            return scf.UHF(mol).run().UMP2()
    else:
        # Try as DFT functional
        if mol.spin == 0:
            mf = dft.RKS(mol)
        else:
            mf = dft.UKS(mol)
        mf.xc = method_lower
        return mf
