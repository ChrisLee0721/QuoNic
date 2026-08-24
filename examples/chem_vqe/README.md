# Molecular VQE / 分子 VQE

Complete quantum chemistry pipeline using `quonic.chem`:

Molecule geometry → PySCF SCF → Hamiltonian → Jordan-Wigner → VQE

## Requirements

```bash
pip install 'quonic[chem]'
```

## Run

```bash
python chem_vqe.py
```

## Expected Output

```
Molecule: Molecule(H2, charge=0, spin=0, basis=sto-3g)
Electrons: 2

Number of qubits:  4
SCF energy:        -1.117349 Hartree
Hamiltonian terms: 15

VQE energy:   -1.137270 Hartree
```

The VQE energy should be close to the exact FCI energy of **-1.13727 Hartree** for H2 at equilibrium bond length (0.74 A) with STO-3G basis.
