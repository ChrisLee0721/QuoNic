"""Hamiltonian helpers: import a Pauli Hamiltonian from an external quantum
chemistry library.

QuoNic does not ship a chemistry database; users must generate the electronic
structure Hamiltonian of a molecule with tools such as Qiskit Nature /
OpenFermion, then use this module's adapter to convert it into the
[(coefficient, Pauli string), ...] format required by vqe().
"""


from __future__ import annotations

from typing import Any

from .._i18n import tr


def from_qiskit_nature(op: Any) -> list[tuple[float, str]]:
    """Convert a Qiskit Nature (or Qiskit) SparsePauliOp into [(coeff, pauli), ...].

    op must have .coeffs and .paulis attributes (satisfied by
    qiskit.quantum_info.SparsePauliOp).

    Example (illustrative):
        from qiskit_nature.second_q.drivers import PySCFDriver
        from qiskit_nature.second_q.mappers import JordanWignerMapper
        # ... use PySCFDriver to get an ElectronicStructureProblem, apply the JW mapping ...
        qubit_op = problem.hamiltonian.second_q_op()  # a SparsePauliOp after mapping
        terms = from_qiskit_nature(qubit_op)
        vqe(terms, n_qubits)

    Notes:
        - The Pauli string ordering already matches QuoNic's convention (the first
          character from the left = qubit 0), consistent with Qiskit, so no reversal
          is needed.
        - Under JW mapping the molecular Hamiltonian has real coefficients; an error is
          raised if a non-negligible imaginary part appears.
    """
    terms: list[tuple[float, str]] = []
    for coeff, pauli in zip(op.coeffs, op.paulis):
        if abs(coeff.imag) > 1e-8:
            raise ValueError(tr("err.hamiltonian_imag", coeff=coeff))
        label = pauli.to_label() if hasattr(pauli, "to_label") else str(pauli)
        terms.append((float(coeff.real), label))
    return terms
