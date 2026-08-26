#!/usr/bin/env python3
"""Aspirin Hydrolysis ΔG Validation Experiment.

This script validates the QuoNic quantum chemistry pipeline by computing
the Gibbs free energy change for aspirin hydrolysis and comparing with
experimental and DFT reference values.

Reaction: Aspirin + H2O -> Salicylic acid + Acetic acid

Usage:
    python run_experiment.py [--quantum] [--method METHOD] [--basis BASIS]
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Add parent directory to path for quonic import
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from quonic.chem import (
    Molecule,
    compute_dg,
    optimize_geometry,
    thermochemistry,
    solvation_correction,
)


def load_reference_data():
    """Load experimental and DFT reference data."""
    ref_path = Path(__file__).parent / "reference_data.json"
    with open(ref_path) as f:
        return json.load(f)


def load_molecules():
    """Load reactant and product molecules from XYZ files."""
    reactant_path = Path(__file__).parent / "reactants.xyz"
    product_path = Path(__file__).parent / "products.xyz"

    reactant = Molecule.from_xyz_file(reactant_path)
    product = Molecule.from_xyz_file(product_path)

    return reactant, product


def run_classical_pipeline(reactant, product, method, basis, solvent, temperature, skip_optimization=False):
    """Run the classical (DFT) pipeline."""
    print("=" * 60)
    print("CLASSICAL PIPELINE (DFT)")
    print("=" * 60)

    HARTREE_TO_KCAL = 627.509

    if skip_optimization:
        # Skip optimization, use provided geometries directly
        print("\n[1/3] Computing electronic energies (skipping optimization)...")
        t0 = time.time()

        # Compute gas phase energies
        from pyscf import gto, scf as pyscf_scf, dft as pyscf_dft

        # Reactant energy
        mol_r = reactant.to_pyscf_mol(basis=basis)
        if method.lower() == "hf":
            mf_r = pyscf_scf.RHF(mol_r) if mol_r.spin == 0 else pyscf_scf.UHF(mol_r)
        else:
            mf_r = pyscf_dft.RKS(mol_r) if mol_r.spin == 0 else pyscf_dft.UKS(mol_r)
            mf_r.xc = method.lower()
        mf_r.kernel()
        reactant_energy = mf_r.e_tot

        # Product energy
        mol_p = product.to_pyscf_mol(basis=basis)
        if method.lower() == "hf":
            mf_p = pyscf_scf.RHF(mol_p) if mol_p.spin == 0 else pyscf_scf.UHF(mol_p)
        else:
            mf_p = pyscf_dft.RKS(mol_p) if mol_p.spin == 0 else pyscf_dft.UKS(mol_p)
            mf_p.xc = method.lower()
        mf_p.kernel()
        product_energy = mf_p.e_tot

        t_electronic = time.time() - t0
        print(f"  Reactant energy: {reactant_energy:.6f} Hartree")
        print(f"  Product energy: {product_energy:.6f} Hartree")
        print(f"  Time: {t_electronic:.1f}s")

        # Step 2: Compute solvation correction
        print("\n[2/3] Computing solvation correction...")
        t0 = time.time()
        reactant_solv = solvation_correction(
            reactant, solvent=solvent, method=method, basis=basis
        )
        product_solv = solvation_correction(
            product, solvent=solvent, method=method, basis=basis
        )
        d_solvation = (product_solv - reactant_solv) * HARTREE_TO_KCAL
        t_solv = time.time() - t0
        print(f"  Reactant ΔG_solv: {reactant_solv:.6f} Hartree")
        print(f"  Product ΔG_solv: {product_solv:.6f} Hartree")
        print(f"  ΔΔG_solv: {d_solvation:.2f} kcal/mol")
        print(f"  Time: {t_solv:.1f}s")

        # Compute ΔG (without ZPE/thermal corrections)
        d_electronic = (product_energy - reactant_energy) * HARTREE_TO_KCAL
        dg_total = d_electronic + d_solvation

        print(f"\n[3/3] Summary (no ZPE/thermal corrections)")
        print(f"  ΔE(electronic) = {d_electronic:.2f} kcal/mol")
        print(f"  ΔΔG(solvation) = {d_solvation:.2f} kcal/mol")
        print(f"  ΔG(partial)    = {dg_total:.2f} kcal/mol")

        return {
            "dg": dg_total,
            "d_electronic": d_electronic,
            "d_zpe": 0.0,
            "d_thermal": 0.0,
            "d_solvation": d_solvation,
            "reactant_energy": reactant_energy,
            "product_energy": product_energy,
            "time_total": t_electronic + t_solv,
        }

    # Full pipeline with optimization
    # Step 1: Optimize geometries
    print("\n[1/4] Optimizing reactant geometry...")
    t0 = time.time()
    reactant_opt = optimize_geometry(reactant, method=method, basis=basis)
    t_reactant = time.time() - t0
    print(f"  Converged: {reactant_opt.converged}")
    print(f"  Energy: {reactant_opt.energy:.6f} Hartree")
    print(f"  Gradient norm: {reactant_opt.gradient_norm:.2e}")
    print(f"  Time: {t_reactant:.1f}s")

    print("\n[2/4] Optimizing product geometry...")
    t0 = time.time()
    product_opt = optimize_geometry(product, method=method, basis=basis)
    t_product = time.time() - t0
    print(f"  Converged: {product_opt.converged}")
    print(f"  Energy: {product_opt.energy:.6f} Hartree")
    print(f"  Gradient norm: {product_opt.gradient_norm:.2e}")
    print(f"  Time: {t_product:.1f}s")

    # Step 2: Compute thermochemistry
    print("\n[3/4] Computing thermochemistry...")
    t0 = time.time()
    reactant_thermo = thermochemistry(
        reactant_opt.molecule, method=method, basis=basis,
        temperature=temperature
    )
    product_thermo = thermochemistry(
        product_opt.molecule, method=method, basis=basis,
        temperature=temperature
    )
    t_thermo = time.time() - t0
    print(f"  Reactant ZPE: {reactant_thermo.zpe:.6f} Hartree")
    print(f"  Product ZPE: {product_thermo.zpe:.6f} Hartree")
    print(f"  Time: {t_thermo:.1f}s")

    # Step 3: Compute solvation correction
    print("\n[4/4] Computing solvation correction...")
    t0 = time.time()
    reactant_solv = solvation_correction(
        reactant_opt.molecule, solvent=solvent, method=method, basis=basis
    )
    product_solv = solvation_correction(
        product_opt.molecule, solvent=solvent, method=method, basis=basis
    )
    d_solvation = (product_solv - reactant_solv) * HARTREE_TO_KCAL
    t_solv = time.time() - t0
    print(f"  Reactant ΔG_solv: {reactant_solv:.6f} Hartree")
    print(f"  Product ΔG_solv: {product_solv:.6f} Hartree")
    print(f"  ΔΔG_solv: {d_solvation:.2f} kcal/mol")
    print(f"  Time: {t_solv:.1f}s")

    # Compute total ΔG
    d_electronic = (product_opt.energy - reactant_opt.energy) * HARTREE_TO_KCAL
    d_zpe = (product_thermo.zpe - reactant_thermo.zpe) * HARTREE_TO_KCAL
    d_enthalpy = (product_thermo.enthalpy - reactant_thermo.enthalpy) * HARTREE_TO_KCAL
    d_ts = (product_thermo.entropy - reactant_thermo.entropy) * temperature * HARTREE_TO_KCAL
    d_thermal = (d_enthalpy - d_zpe) - d_ts
    dg_total = d_electronic + d_zpe + d_thermal + d_solvation

    return {
        "dg": dg_total,
        "d_electronic": d_electronic,
        "d_zpe": d_zpe,
        "d_thermal": d_thermal,
        "d_solvation": d_solvation,
        "reactant_energy": reactant_opt.energy,
        "product_energy": product_opt.energy,
        "time_total": t_reactant + t_product + t_thermo + t_solv,
    }


def run_quantum_pipeline(reactant, product, method, basis, solvent, temperature, n_qubits):
    """Run the quantum (VQE) pipeline for active space."""
    print("\n" + "=" * 60)
    print("QUANTUM PIPELINE (VQE)")
    print("=" * 60)

    from quonic.chem import molecular_hamiltonian, select_active_space

    # Step 1: Select active space
    print(f"\n[1/3] Selecting active space ({n_qubits} qubits)...")
    active_space = select_active_space(
        reactant, method="avas", n_qubits=n_qubits
    )
    print(f"  Active electrons: {active_space.n_electrons}")
    print(f"  Active orbitals: {active_space.n_orbitals}")
    print(f"  Orbital indices: {active_space.orbital_indices}")

    # Step 2: Build Hamiltonian
    print("\n[2/3] Building molecular Hamiltonian...")
    result = molecular_hamiltonian(
        reactant,
        active_space=active_space,
        mapping="jordan_wigner",
    )
    hamiltonian = result.metadata["hamiltonian"]
    n_qubits_actual = result.metadata["n_qubits"]
    print(f"  Qubits: {n_qubits_actual}")
    print(f"  Hamiltonian terms: {len(hamiltonian)}")

    # Step 3: Run VQE
    print("\n[3/3] Running VQE...")
    from quonic.algorithms import vqe

    t0 = time.time()
    vqe_result = vqe(
        hamiltonian,
        n_qubits_actual,
        ansatz="uccsd",
        maxiter=100,
    )
    t_vqe = time.time() - t0
    print(f"  VQE energy: {vqe_result.value:.6f} Hartree")
    print(f"  Converged: {vqe_result.converged}")
    print(f"  Time: {t_vqe:.1f}s")

    return {
        "vqe_energy": vqe_result.value,
        "n_qubits": n_qubits_actual,
        "converged": vqe_result.converged,
        "time": t_vqe,
    }


def print_results(classical_result, quantum_result, reference_data):
    """Print comparison with reference data."""
    print("\n" + "=" * 60)
    print("RESULTS COMPARISON")
    print("=" * 60)

    exp = reference_data["experimental"]
    dft_ref = reference_data["dft_reference"]

    print(f"\n{'Method':<25} {'ΔG (kcal/mol)':<15} {'Error':<15}")
    print("-" * 55)
    print(f"{'QuoNic DFT':<25} {classical_result['dg']:<15.2f} "
          f"{abs(classical_result['dg'] - exp['delta_g']):<15.2f}")
    print(f"{'Reference DFT':<25} {dft_ref['delta_g']:<15.2f} "
          f"{abs(dft_ref['delta_g'] - exp['delta_g']):<15.2f}")
    print(f"{'Experimental':<25} {exp['delta_g']:<15.2f} {'(reference)':<15}")

    print(f"\n{'Component':<25} {'Value (kcal/mol)':<20}")
    print("-" * 45)
    print(f"{'ΔE(electronic)':<25} {classical_result['d_electronic']:<20.2f}")
    print(f"{'ΔZPE':<25} {classical_result['d_zpe']:<20.2f}")
    print(f"{'ΔH(thermal) - TΔS':<25} {classical_result['d_thermal']:<20.2f}")
    print(f"{'ΔΔG(solvation)':<25} {classical_result['d_solvation']:<20.2f}")
    print(f"{'─' * 45}")
    print(f"{'ΔG(total)':<25} {classical_result['dg']:<20.2f}")

    if quantum_result:
        print(f"\n{'Quantum Result':<25} {'Value':<20}")
        print("-" * 45)
        print(f"{'VQE energy':<25} {quantum_result['vqe_energy']:<20.6f}")
        print(f"{'Qubits':<25} {quantum_result['n_qubits']:<20}")
        print(f"{'Converged':<25} {str(quantum_result['converged']):<20}")

    print(f"\n{'Timing':<25} {'Time (s)':<20}")
    print("-" * 45)
    print(f"{'Classical pipeline':<25} {classical_result['time_total']:<20.1f}")
    if quantum_result:
        print(f"{'Quantum pipeline':<25} {quantum_result['time']:<20.1f}")


def main():
    parser = argparse.ArgumentParser(
        description="Aspirin hydrolysis ΔG validation experiment"
    )
    parser.add_argument(
        "--quantum", action="store_true",
        help="Run quantum (VQE) pipeline in addition to classical"
    )
    parser.add_argument(
        "--method", default="b3lyp",
        help="DFT functional (default: b3lyp)"
    )
    parser.add_argument(
        "--basis", default="sto-3g",
        help="Basis set (default: sto-3g)"
    )
    parser.add_argument(
        "--solvent", default="water",
        help="Solvent for solvation correction (default: water)"
    )
    parser.add_argument(
        "--temperature", type=float, default=298.15,
        help="Temperature in K (default: 298.15)"
    )
    parser.add_argument(
        "--qubits", type=int, default=8,
        help="Number of qubits for quantum pipeline (default: 8)"
    )
    parser.add_argument(
        "--skip-optimization", action="store_true",
        help="Skip geometry optimization (faster, uses provided geometries)"
    )
    args = parser.parse_args()

    # Load data
    print("Loading molecules...")
    reactant, product = load_molecules()
    print(f"  Reactant: {reactant.n_atoms} atoms")
    print(f"  Product: {product.n_atoms} atoms")

    reference_data = load_reference_data()
    print(f"  Reference ΔG: {reference_data['experimental']['delta_g']} kcal/mol")

    # Run classical pipeline
    classical_result = run_classical_pipeline(
        reactant, product,
        method=args.method,
        basis=args.basis,
        solvent=args.solvent,
        temperature=args.temperature,
        skip_optimization=args.skip_optimization,
    )

    # Run quantum pipeline if requested
    quantum_result = None
    if args.quantum:
        quantum_result = run_quantum_pipeline(
            reactant, product,
            method=args.method,
            basis=args.basis,
            solvent=args.solvent,
            temperature=args.temperature,
            n_qubits=args.qubits,
        )

    # Print results
    print_results(classical_result, quantum_result, reference_data)

    # Save results
    output = {
        "classical": classical_result,
        "quantum": quantum_result,
        "reference": reference_data,
        "parameters": {
            "method": args.method,
            "basis": args.basis,
            "solvent": args.solvent,
            "temperature": args.temperature,
        },
    }
    output_path = Path(__file__).parent / "results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
