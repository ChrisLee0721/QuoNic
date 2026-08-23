"""Hamiltonian import from multiple sources.

Boundary conditions:
- Pauli string convention: "ZZXI" means Z⊗Z⊗X⊗I (qubit 0 = rightmost)
- All coefficients must be real (imaginary raises ValueError)
- OpenFermion requires: pip install openfermion
- PennyLane requires: pip install pennylane

Example::

    from quonic.algorithms import from_pauli_string, from_openfermion
    terms = from_pauli_string("1.0*ZZ + 0.5*XI - 0.3*IX")
"""

from __future__ import annotations

from typing import Any

from .._i18n import tr


def from_pauli_string(expr: str) -> list[tuple[float, str]]:
    """Parse a Pauli Hamiltonian string into [(coeff, pauli), ...].

    Format: "1.0*ZZ + 0.5*XI - 0.3*IX"
    Supports: +, -, *, spaces.
    """
    terms = []
    # Split by + or - (keeping the sign)
    import re
    tokens = re.findall(r'[+-]?\s*[\d.eE+-]+\s*\*\s*[A-Z]+', expr)
    for token in tokens:
        token = token.strip()
        # Extract coefficient and pauli
        match = re.match(r'([+-]?\s*[\d.eE+-]+)\s*\*\s*([A-Z]+)', token)
        if match:
            coeff = float(match.group(1).replace(' ', ''))
            pauli = match.group(2)
            terms.append((coeff, pauli))
    return terms


def from_openfermion(op: Any) -> list[tuple[float, str]]:
    """Convert an OpenFermion QubitOperator to [(coeff, pauli), ...].

    Requires: pip install openfermion
    """
    terms = []
    for term in op.terms:
        pauli = ["I"] * (max(q for q, _ in term) + 1 if term else 0)
        for q, p in term:
            pauli[q] = p
        coeff = op.terms[term]
        if abs(coeff.imag) > 1e-8:
            raise ValueError(tr("err.hamiltonian_imag", coeff=coeff))
        terms.append((float(coeff.real), "".join(reversed(pauli))))
    return terms


def from_pennylane(op: Any) -> list[tuple[float, str]]:
    """Convert a PennyLane Hamiltonian to [(coeff, pauli), ...].

    Requires: pip install pennylane
    """
    terms = []
    coeffs = op.coeffs
    ops = op.ops
    for coeff, pauli_op in zip(coeffs, ops):
        label = pauli_op.label()
        # PennyLane uses "Z(0)@Z(1)" format, convert to "ZZ"
        label = label.replace("@", "").replace("(", "").replace(")", "")
        # Remove qubit indices, keep only Pauli letters
        import re
        pauli = re.sub(r'\d+', '', label)
        terms.append((float(coeff), pauli))
    return terms
