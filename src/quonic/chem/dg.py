"""Gibbs free energy (ΔG) calculation pipeline.

Computes reaction free energies by combining electronic structure
calculations with thermochemical corrections and solvation effects.

Example::

    from quonic.chem import Molecule, compute_dg

    # Define reaction: H2 + O2 -> H2O
    h2 = Molecule.from_xyz('''2\\nH2\\nH 0 0 0\\nH 0 0 0.74''')
    o2 = Molecule.from_xyz('''2\\nO2\\nO 0 0 0\\nO 0 0 1.21''')
    h2o = Molecule.from_xyz('''3\\nH2O\\nO 0 0 0\\nH 0 0 0.96\\nH 0 0.96 0''')

    result = compute_dg(
        reaction={"reactants": [h2, o2], "products": [2 * h2o]},
        method="b3lyp", basis="6-31g*",
        solvent="water", temperature=298.15,
    )
    print(result.dg)  # ΔG in kcal/mol
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .._i18n import tr
from .molecule import Molecule
from .optimize import OptimizationResult, optimize_geometry
from .solvation import differential_solvation, solvation_correction
from .thermo import ThermoResult, thermochemistry

# Conversion factor
HARTREE_TO_KCAL = 627.509  # 1 Hartree = 627.509 kcal/mol


@dataclass
class DGResult:
    """Result of Gibbs free energy calculation.

    All energies are in kcal/mol unless noted.

    Attributes:
        dg: Total Gibbs free energy change ΔG.
        d_electronic: Electronic energy change ΔE.
        d_zpe: Zero-point energy change ΔZPE.
        d_thermal: Thermal correction change (ΔH_thermal - TΔS).
        d_solvation: Solvation correction ΔΔG_solv.
        temperature: Temperature used (K).
        solvent: Solvent name.
        reactant_thermo: Thermochemistry results for reactants.
        product_thermo: Thermochemistry results for products.
    """

    dg: float
    d_electronic: float
    d_zpe: float
    d_thermal: float
    d_solvation: float
    temperature: float
    solvent: str
    reactant_thermo: list[ThermoResult] = field(default_factory=list)
    product_thermo: list[ThermoResult] = field(default_factory=list)


@dataclass
class Reaction:
    """Defines a chemical reaction.

    Attributes:
        reactants: List of (stoichiometry, Molecule) pairs for reactants.
        products: List of (stoichiometry, Molecule) pairs for products.
    """

    reactants: list[tuple[float, Molecule]]
    products: list[tuple[float, Molecule]]

    @classmethod
    def from_dict(cls, reaction: dict[str, list]) -> Reaction:
        """Create Reaction from dictionary format.

        Args:
            reaction: Dict with 'reactants' and 'products' keys.
                Each value is a list of Molecule objects or (coeff, Molecule) tuples.

        Returns:
            A Reaction object.
        """
        reactants = _parse_species(reaction.get("reactants", []))
        products = _parse_species(reaction.get("products", []))
        return cls(reactants=reactants, products=products)


def compute_dg(
    reaction: dict[str, list] | Reaction,
    method: str = "hf",
    basis: str = "sto-3g",
    solvent: str = "vacuum",
    temperature: float = 298.15,
    pressure: float = 1.0,
    optimize: bool = True,
    optimize_basis: str | None = None,
) -> DGResult:
    """Compute Gibbs free energy change for a reaction.

    ΔG = Σ ν_i G_i(products) - Σ ν_i G_i(reactants)

    where G_i = E_elec + ZPE + H_thermal - T·S + ΔG_solv

    Args:
        reaction: Reaction definition (dict or Reaction object).
        method: Electronic structure method for energy calculation.
        basis: Basis set for energy calculation.
        solvent: Solvent name for solvation correction.
        temperature: Temperature in Kelvin.
        pressure: Pressure in atm.
        optimize: Whether to optimize geometries before computing energies.
        optimize_basis: Basis set for optimization (defaults to energy basis).

    Returns:
        A :class:`DGResult` with all energy components.

    Raises:
        ImportError: If PySCF is not installed.
        ValueError: If reaction definition is invalid.
    """
    if isinstance(reaction, dict):
        reaction = Reaction.from_dict(reaction)

    if not reaction.reactants or not reaction.products:
        raise ValueError(tr("err.chem.reaction_empty"))

    # Optimize geometries if requested
    reactant_mols = []
    for coeff, mol in reaction.reactants:
        if optimize:
            opt_result = optimize_geometry(
                mol, method=method, basis=optimize_basis or basis
            )
            reactant_mols.append((coeff, opt_result.molecule))
        else:
            reactant_mols.append((coeff, mol))

    product_mols = []
    for coeff, mol in reaction.products:
        if optimize:
            opt_result = optimize_geometry(
                mol, method=method, basis=optimize_basis or basis
            )
            product_mols.append((coeff, opt_result.molecule))
        else:
            product_mols.append((coeff, mol))

    # Compute thermochemistry for each species
    reactant_thermo = []
    for coeff, mol in reactant_mols:
        thermo_result = thermochemistry(
            mol, method=method, basis=basis,
            temperature=temperature, pressure=pressure
        )
        reactant_thermo.append(thermo_result)

    product_thermo = []
    for coeff, mol in product_mols:
        thermo_result = thermochemistry(
            mol, method=method, basis=basis,
            temperature=temperature, pressure=pressure
        )
        product_thermo.append(thermo_result)

    # Compute electronic energy change
    e_reactants = sum(
        coeff * thermo.electronic_energy
        for (coeff, _), thermo in zip(reactant_mols, reactant_thermo)
    )
    e_products = sum(
        coeff * thermo.electronic_energy
        for (coeff, _), thermo in zip(product_mols, product_thermo)
    )
    d_electronic = (e_products - e_reactants) * HARTREE_TO_KCAL

    # Compute ZPE change
    zpe_reactants = sum(
        coeff * thermo.zpe
        for (coeff, _), thermo in zip(reactant_mols, reactant_thermo)
    )
    zpe_products = sum(
        coeff * thermo.zpe
        for (coeff, _), thermo in zip(product_mols, product_thermo)
    )
    d_zpe = (zpe_products - zpe_reactants) * HARTREE_TO_KCAL

    # Compute thermal correction change
    # H_thermal = H - E_elec - ZPE, so ΔH_thermal = ΔH - ΔE - ΔZPE
    h_reactants = sum(
        coeff * thermo.enthalpy
        for (coeff, _), thermo in zip(reactant_mols, reactant_thermo)
    )
    h_products = sum(
        coeff * thermo.enthalpy
        for (coeff, _), thermo in zip(product_mols, product_thermo)
    )
    d_enthalpy = (h_products - h_reactants) * HARTREE_TO_KCAL

    # TΔS term
    ts_reactants = sum(
        coeff * thermo.entropy * temperature
        for (coeff, _), thermo in zip(reactant_mols, reactant_thermo)
    )
    ts_products = sum(
        coeff * thermo.entropy * temperature
        for (coeff, _), thermo in zip(product_mols, product_thermo)
    )
    d_ts = (ts_products - ts_reactants) * HARTREE_TO_KCAL

    # Thermal correction = ΔH_thermal - TΔS
    d_thermal = (d_enthalpy - d_zpe) - d_ts

    # Compute solvation correction
    if solvent.lower() not in ("vacuum", "gas", "1.0"):
        dg_solv = differential_solvation(
            [mol for _, mol in reactant_mols],
            [mol for _, mol in product_mols],
            solvent=solvent, method=method, basis=basis,
        )
        d_solvation = dg_solv * HARTREE_TO_KCAL
    else:
        d_solvation = 0.0

    # Total ΔG
    dg = d_electronic + d_zpe + d_thermal + d_solvation

    return DGResult(
        dg=dg,
        d_electronic=d_electronic,
        d_zpe=d_zpe,
        d_thermal=d_thermal,
        d_solvation=d_solvation,
        temperature=temperature,
        solvent=solvent,
        reactant_thermo=reactant_thermo,
        product_thermo=product_thermo,
    )


def compute_dg_barrier(
    ts: Molecule,
    reactant: Molecule,
    method: str = "hf",
    basis: str = "sto-3g",
    solvent: str = "vacuum",
    temperature: float = 298.15,
    pressure: float = 1.0,
    optimize: bool = False,
) -> float:
    """Compute activation free energy ΔG‡.

    ΔG‡ = G(TS) - G(reactant)

    Args:
        ts: Transition state geometry.
        reactant: Reactant geometry.
        method: Electronic structure method.
        basis: Basis set name.
        solvent: Solvent name.
        temperature: Temperature in Kelvin.
        pressure: Pressure in atm.
        optimize: Whether to optimize geometries.

    Returns:
        Activation free energy ΔG‡ in kcal/mol.
    """
    # Optimize if requested
    if optimize:
        ts_opt = optimize_geometry(ts, method=method, basis=basis)
        reactant_opt = optimize_geometry(reactant, method=method, basis=basis)
        ts_mol = ts_opt.molecule
        reactant_mol = reactant_opt.molecule
    else:
        ts_mol = ts
        reactant_mol = reactant

    # Compute thermochemistry
    ts_thermo = thermochemistry(
        ts_mol, method=method, basis=basis,
        temperature=temperature, pressure=pressure
    )
    reactant_thermo = thermochemistry(
        reactant_mol, method=method, basis=basis,
        temperature=temperature, pressure=pressure
    )

    # Compute solvation correction
    if solvent.lower() not in ("vacuum", "gas", "1.0"):
        ts_solv = solvation_correction(ts_mol, solvent, method, basis)
        reactant_solv = solvation_correction(reactant_mol, solvent, method, basis)
        d_solvation = (ts_solv - reactant_solv) * HARTREE_TO_KCAL
    else:
        d_solvation = 0.0

    # ΔG‡ = G(TS) - G(reactant)
    dg_barrier = (ts_thermo.gibbs - reactant_thermo.gibbs) * HARTREE_TO_KCAL + d_solvation

    return float(dg_barrier)


def _parse_species(species_list: list) -> list[tuple[float, Molecule]]:
    """Parse species list into (coefficient, Molecule) pairs.

    Accepts:
    - List of Molecule objects (coefficient defaults to 1)
    - List of (coefficient, Molecule) tuples
    - List of (coefficient, Molecule) lists

    Args:
        species_list: List of species.

    Returns:
        List of (coefficient, Molecule) tuples.
    """
    result = []
    for item in species_list:
        if isinstance(item, Molecule):
            result.append((1.0, item))
        elif isinstance(item, (tuple, list)) and len(item) == 2:
            coeff, mol = item
            if isinstance(mol, Molecule):
                result.append((float(coeff), mol))
            else:
                raise ValueError(
                    tr("err.chem.reaction_invalid_species")
                    + f" Expected Molecule, got {type(mol).__name__}"
                )
        else:
            raise ValueError(
                tr("err.chem.reaction_invalid_species")
                + f" Expected Molecule or (coeff, Molecule), got {type(item).__name__}"
            )
    return result
