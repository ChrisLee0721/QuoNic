# GCIQA: Geometric Constraint Iterative Quantum Amplitude Amplification

## Overview

GCIQA is a quantum algorithm framework for conformational search of large molecular systems (drug-enzyme, enzyme-enzyme). It uses Grover search with **geometric constraints** (not energy thresholds) and iterative clustering feedback to converge on the most-probable binding conformations.

**Key innovation**: No pre-computed energy values needed. The oracle encodes geometric constraints (bond lengths, angles, pocket boundaries, surface distance), not energy.

## Algorithm (5 Stages)

### Stage 0: Preprocessing (Classical)
- Get rough structure from experiment or AlphaFold
- Define active pocket / interaction interface
- Build initial constraint set C₀ = {c₁, c₂, ..., cₖ}
- Design coarse-graining mapping: full atoms → super-atoms

### Stage 1: Coarse Global Scan (Quantum)
- Coarse-grain to N_CG super-atoms (N_CG << N_full)
- Encode C₀ as Oracle O₀: marks states satisfying all constraints
- Run Grover search: R₁ ≈ √(N_CG/M₁) iterations
- Output: K high-probability coarse regions A₁ = {A₁⁽¹⁾, ..., A_K⁽¹⁾}

### Stage 2: Local Fine Sampling (Quantum)
- For each region A_j⁽¹⁾, switch to full-atom model
- Define local constraints C₁ ⊃ C₀ (add H-bond geometry, hydrophobic exposure)
- Build new Oracle O₁
- Run Grover search → K' high-probability conformations X₁

### Stage 3: Geometric Clustering (Classical)
- Decode quantum measurements to atomic coordinates
- K-means clustering → cluster centers {c₁, c₂, ..., c_m}
- Select largest cluster center c* as representative
- Compute convergence radius r = max‖xᵢ - c*‖

### Stage 4: Oracle Update & Iteration
- New constraint: C_new = C_old ∪ {‖x - c*‖ < r_new}, r_new = α·r (0 < α < 1)
- Build new Oracle O_new
- Return to Stage 2

**Termination**: Δc < ε (0.1 Å), or r < δ (0.1-0.5 Å), or T_max iterations (5-10)

## Assumptions

| ID | Assumption | Basis |
|----|-----------|-------|
| A1 | Valid conformations are sparse in full space | PDB: native structures << theoretical possibilities |
| A2 | Ground state has few non-zero amplitudes | Coupled cluster / tensor network methods |
| A3 | High-probability conformations cluster geometrically | MD simulations: metastable conformational clusters |

## Module Structure

```
src/quonic/gciqa/
├── README.md          # This file
├── __init__.py        # Public API
├── constraints.py     # Geometric constraint encoding
├── oracle.py          # Quantum oracle construction
├── search.py          # Grover search implementation
├── clustering.py      # K-means geometric clustering
└── iterative.py       # Main iterative loop (Stage 0-4)
```

## Oracle Implementation

The oracle marks valid conformations: O|x⟩ = (-1)^{f(x)}|x⟩ where f(x)=1 if all constraints are satisfied.

### Two modes:

**1. Enumeration mode (n_qubits ≤ 16)**
- Classically enumerate all 2^n states
- Check each against constraints using `classical_oracle_fn()`
- Apply phase flip to valid states using X gates + multi-controlled Z
- Correct by construction, used for small search spaces

**2. Arithmetic mode (n_qubits > 16)**
- Uses CDKM ripple-carry adder for subtraction (two's complement)
- Shift-and-add squaring for distance² computation
- Bit-level comparison for range checks
- Workspace: 7b+3 ancilla qubits (b = bits per coordinate)
- Scalable but approximate (XOR-based squaring, bit-level comparison)

### Coordinate encoding
- Binary, little-endian per coordinate
- Layout: atom_i = [x_0..x_{b-1}, y_0..y_{b-1}, z_0..z_{b-1}]
- Physical value: lo + int(bits, 2) × (hi - lo) / (2^b - 1)

### Verified behavior
- 8/64 valid states (pocket radius=40, 2 bits/coord): Grover amplifies to ~77% (from 12.5% classical)
- 0 valid states: uniform distribution (no amplification)
- 64/64 valid states: all states measured equally

## Coarse-Graining

Maps full molecular systems to super-atoms for quantum encoding.

Strategies:
- `spatial`: Cluster atoms by proximity (greedy seed selection)
- `residue`: Group by residue ID (center of mass per group)
- `fragment`: Connected components by distance cutoff

```python
from quonic.gciqa import coarse_grain

cg = coarse_grain(atoms, coords, strategy="spatial", n_super_atoms=50)
print(cg.super_coords)  # 50 super-atom positions
print(cg.atom_to_super)  # mapping from full to coarse
```

## Validation

### Classical search (basic)
Water dimer (H2O...H2O) binding geometry:
- O-O distance: 3.27 Å (expected 2.98, error 9.7%)
- 4 geometric constraints: O-O bond, H-bond, no-clash, pocket
- Converges in 2 iterations

### Controlled validation
Three groups prove constraints encode physical information:
- Physical (tight): 5.9% error
- Wide (correct atoms, loose): 11.8% error
- Random (wrong atoms): 32.3% error

### Quantum Grover search
Enumeration mode (12 qubits, 2 super-atoms, 2 bits/coord):
- 99.1% valid state rate (vs 27.2% classical)
- 3.5x Grover amplification
- Best achievable O-O: 3.000 Å (0.7% error)

See `experiments/water_dimer/` for all scripts.

## Usage

```python
from quonic.gciqa import GCIQA

# Define system
system = GCIQA(
    receptor_pdb="1abc.pdb",
    ligand_xyz="ligand.xyz",
    pocket_center=(10.0, 20.0, 30.0),
    pocket_radius=8.0,
)

# Run iterative search
result = system.run(
    n_super_atoms=50,
    n_qubits=100,
    max_iterations=5,
    convergence_threshold=0.1,  # Angstrom
)

print(result.best_conformation)  # Atomic coordinates
print(result.convergence_history)
```

## Relationship to ΔG Pipeline

GCIQA finds the conformation. The classical ΔG pipeline (PySCF) evaluates the energy:

```
GCIQA (quantum) → best conformation → PySCF (classical) → ΔG
```

They are separate components. GCIQA replaces geometry optimization, not energy calculation.
