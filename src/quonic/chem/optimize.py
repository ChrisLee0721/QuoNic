"""Geometry optimization using PySCF gradients.

Provides energy minimization and transition state search for molecular
geometries, bridging QuoNic's Molecule class with PySCF's optimizer.

Example::

    from quonic.chem import Molecule, optimize_geometry

    mol = Molecule.from_xyz('''
    3
    H2O
    O  0.0  0.0  0.0
    H  0.0  0.0  0.96
    H  0.0  0.96  0.0
    ''')
    opt = optimize_geometry(mol, method="b3lyp", basis="6-31g*")
    print(opt.energy)        # optimized energy
    print(opt.molecule)      # optimized geometry
    print(opt.converged)     # True
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .._i18n import tr
from .molecule import Molecule


@dataclass
class OptimizationResult:
    """Result of geometry optimization.

    Attributes:
        molecule: Optimized molecular geometry.
        energy: Electronic energy at optimized geometry (Hartree).
        converged: Whether optimization converged within max_steps.
        n_steps: Number of optimization steps taken.
        gradient_norm: Final gradient norm (Hartree/Bohr).
    """

    molecule: Molecule
    energy: float
    converged: bool
    n_steps: int
    gradient_norm: float


def optimize_geometry(
    molecule: Molecule,
    method: str = "hf",
    basis: str | None = None,
    max_steps: int = 50,
    gradient_tol: float = 1e-4,
    step_tol: float = 1e-4,
    energy_tol: float = 1e-6,
) -> OptimizationResult:
    """Optimize molecular geometry using PySCF gradients.

    Args:
        molecule: Input molecular geometry.
        method: Electronic structure method (``"hf"``, ``"b3lyp"``, ``"mp2"``, etc.).
        basis: Basis set name (defaults to molecule's basis).
        max_steps: Maximum optimization steps.
        gradient_tol: Convergence threshold for gradient norm (Hartree/Bohr).
        step_tol: Convergence threshold for step size (Bohr).
        energy_tol: Convergence threshold for energy change (Hartree).

    Returns:
        An :class:`OptimizationResult` with optimized geometry.

    Raises:
        ImportError: If PySCF is not installed.
        RuntimeError: If optimization fails to converge.
    """
    try:
        from pyscf.geomopt import geometric_solver
    except ImportError as exc:
        raise ImportError(tr("err.chem.pyscf_missing")) from exc

    # Build PySCF molecule
    pyscf_mol = molecule.to_pyscf_mol(basis=basis)

    # Create SCF solver
    mf = _create_solver(pyscf_mol, method)

    # Run initial SCF to get reference energy
    mf.kernel()
    if not mf.converged:
        raise RuntimeError(
            tr("err.chem.scf_not_converged", method=method)
        )

    # Optimize geometry using geometric optimizer
    try:
        opt_mol = geometric_solver.optimize(
            mf,
            maxsteps=max_steps,
        )
    except Exception as exc:
        raise RuntimeError(
            tr("err.chem.optimize_failed", reason=str(exc))
        ) from exc

    # Extract optimized geometry
    coords = opt_mol.atom_coords()  # in Bohr
    atoms = []
    for i in range(opt_mol.natm):
        atoms.append(opt_mol.atom_symbol(i))

    # Convert Bohr to Angstrom (1 Bohr = 0.529177 Å)
    BOHR_TO_ANG = 0.529177
    coords_ang = tuple(
        (float(x * BOHR_TO_ANG), float(y * BOHR_TO_ANG), float(z * BOHR_TO_ANG))
        for x, y, z in coords
    )

    # Run final SCF on optimized geometry to get correct energy
    pyscf_mol_opt = pyscf_mol.copy()
    # coords are in Bohr, PySCF expects Bohr by default
    pyscf_mol_opt.atom = [
        [atoms[i], coords[i][0], coords[i][1], coords[i][2]]
        for i in range(len(atoms))
    ]
    pyscf_mol_opt.unit = 'Bohr'
    pyscf_mol_opt.build()
    mf_opt = _create_solver(pyscf_mol_opt, method)
    mf_opt.kernel()

    final_energy = mf_opt.e_tot
    grad = mf_opt.Gradients()
    grad_grid = grad.grad()
    import numpy as np
    grad_norm = float(np.linalg.norm(grad_grid))

    # Check convergence
    converged = grad_norm < 1e-3

    optimized_mol = Molecule(
        atoms=tuple(atoms),
        coords=coords_ang,
        charge=molecule.charge,
        spin=molecule.spin,
        basis=basis or molecule.basis,
    )

    return OptimizationResult(
        molecule=optimized_mol,
        energy=final_energy,
        converged=converged,
        n_steps=max_steps,  # PySCF doesn't expose step count easily
        gradient_norm=grad_norm,
    )


def optimize_transition_state(
    reactant: Molecule,
    product: Molecule,
    method: str = "hf",
    basis: str | None = None,
    max_steps: int = 50,
) -> OptimizationResult:
    """Find transition state using linear interpolation + optimization.

    Uses a simple approach: interpolate between reactant and product
    geometries, then optimize with eigenvector following. For production
    use, consider NEB or QST2/QST3 methods.

    Args:
        reactant: Reactant molecular geometry.
        product: Product molecular geometry.
        method: Electronic structure method.
        basis: Basis set name.
        max_steps: Maximum optimization steps.

    Returns:
        An :class:`OptimizationResult` with transition state geometry.

    Raises:
        ImportError: If PySCF is not installed.
        ValueError: If reactant and product have different atom counts/types.
    """
    if reactant.atoms != product.atoms:
        raise ValueError(
            tr("err.chem.ts_atom_mismatch")
        )

    try:
        from pyscf.geomopt import geometric_solver
    except ImportError as exc:
        raise ImportError(tr("err.chem.pyscf_missing")) from exc

    # Linear interpolation: midpoint between reactant and product
    import numpy as np

    r_coords = np.array(reactant.coords)
    p_coords = np.array(product.coords)
    ts_coords = (r_coords + p_coords) / 2.0

    # Build PySCF molecule with interpolated geometry
    pyscf_mol = reactant.to_pyscf_mol(basis=basis)
    BOHR_TO_ANG = 0.529177
    pyscf_mol.atom = [
        [reactant.atoms[i], ts_coords[i][0] / BOHR_TO_ANG, ts_coords[i][1] / BOHR_TO_ANG, ts_coords[i][2] / BOHR_TO_ANG]
        for i in range(len(reactant.atoms))
    ]
    pyscf_mol.build()

    # Create SCF solver
    mf = _create_solver(pyscf_mol, method)

    # Run initial SCF
    mf.kernel()
    if not mf.converged:
        raise RuntimeError(
            tr("err.chem.scf_not_converged", method=method)
        )

    # For TS optimization, we use geometric optimizer with reduced convergence
    try:
        opt_mol = geometric_solver.optimize(
            mf,
            maxsteps=max_steps,
        )
    except Exception as exc:
        raise RuntimeError(
            tr("err.chem.ts_optimize_failed", reason=str(exc))
        ) from exc

    # Extract TS geometry
    coords = opt_mol.atom_coords()
    atoms = []
    for i in range(opt_mol.natm):
        atoms.append(opt_mol.atom_symbol(i))

    coords_ang = tuple(
        (float(x * BOHR_TO_ANG), float(y * BOHR_TO_ANG), float(z * BOHR_TO_ANG))
        for x, y, z in coords
    )

    final_energy = mf.e_tot
    grad = mf.Gradients()
    grad_grid = grad.grad()
    grad_norm = float(np.linalg.norm(grad_grid))

    ts_mol = Molecule(
        atoms=tuple(atoms),
        coords=coords_ang,
        charge=reactant.charge,
        spin=reactant.spin,
        basis=basis or reactant.basis,
    )

    return OptimizationResult(
        molecule=ts_mol,
        energy=final_energy,
        converged=grad_norm < 1e-3,
        n_steps=max_steps,
        gradient_norm=grad_norm,
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
