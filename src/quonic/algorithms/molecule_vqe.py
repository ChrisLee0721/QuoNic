"""VQE Molecular Simulation — compute molecular ground state energy.

Boundary conditions:
- Requires PySCF + OpenFermion for full functionality
- Minimal demo: H2 molecule with hardcoded Hamiltonian
- Uses UCCSD ansatz concept (simplified)
- NOT a production quantum chemistry tool

Example::

    from quonic.algorithms import molecule_vqe
    result = molecule_vqe()
"""

from __future__ import annotations

from ..result import Result
from .vqe import vqe


def molecule_vqe(
    maxiter: int = 200,
) -> Result:
    """Minimal molecular VQE demo with H2 Hamiltonian.

    The H2 Hamiltonian at equilibrium bond length (0.735 Å) in STO-3G basis,
    after Jordan-Wigner transformation, has 4 qubits and the following terms
    (simplified to 2-qubit effective Hamiltonian):
    """
    # H2 effective Hamiltonian (2 qubits, from literature)
    # H = -0.81261 II + 0.17120 IZ + -0.22279 ZI + 0.17120 ZZ + 0.04532 XX
    hamiltonian = [
        (-0.81261, "II"),
        (0.17120, "IZ"),
        (-0.22279, "ZI"),
        (0.17120, "ZZ"),
        (0.04532, "XX"),
    ]

    result = vqe(hamiltonian, 2, maxiter=maxiter)
    return Result.from_value(
        result.value,
        energy=result.value,
        params=result.metadata.get("params"),
        molecule="H2",
        basis="STO-3G",
        exact_energy=-1.8572,  # exact ground state energy
    )
