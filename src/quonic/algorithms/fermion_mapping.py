"""Fermion-to-qubit mapping — Jordan-Wigner and Bravyi-Kitaev transforms.

Boundary conditions:
- Jordan-Wigner: O(n) Pauli weight per operator
- Bravyi-Kitaev: O(log n) Pauli weight per operator
- Requires openfermion for full functionality
- Minimal demonstration with 2-site Hubbard model

Example::

    from quonic.algorithms import jordan_wigner_2site
    result = jordan_wigner_2site(t=1.0, U=2.0)
"""

from __future__ import annotations

from ..result import Result


def jordan_wigner_2site(
    t: float = 1.0,
    U: float = 2.0,
) -> Result:
    """Compute Jordan-Wigner transformation for 2-site Hubbard model.

    H = -t * (c†_0 c_1 + c†_1 c_0) + U * n_0 * n_1

    Under JW mapping:
    c†_0 = (X_0 - iY_0)/2
    c†_1 = Z_0 (X_1 - iY_1)/2
    n_i = (I - Z_i)/2

    Returns:
        Result with Pauli Hamiltonian terms.
    """
    # H = -t/2 * (X_0 X_1 + Y_0 Y_1) + U/4 * (I - Z_0 - Z_1 + Z_0 Z_1)
    terms: list[tuple[float, str]] = []
    terms.append((-t / 2, "XX"))
    terms.append((-t / 2, "YY"))
    terms.append((U / 4, "II"))
    terms.append((-U / 4, "ZI"))
    terms.append((-U / 4, "IZ"))
    terms.append((U / 4, "ZZ"))

    return Result.from_value(0.0, hamiltonian=terms)
