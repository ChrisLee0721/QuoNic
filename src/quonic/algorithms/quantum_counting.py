"""Quantum counting template.

Estimate the number M of solutions satisfying an oracle using Grover iteration
plus Quantum Phase Estimation (QPE).

Principle: the Grover operator G rotates by angle 2θ in the {|non-solution>,
|solution>} subspace, where sin²θ = M / N (N = 2^n). QPE estimates G's eigenphase
θ/π, from which M is recovered.

Example: count how many states in 3 bits (N=8) satisfy a predicate

    from quonic.algorithms import oracle, quantum_counting

    @oracle(3)
    def f(x):
        return x & 1 == 0            # even numbers: 4 solutions in total

    result = quantum_counting(f, 3)
    print(result.value)              # close to 4
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from .._i18n import tr
from ..backends import run_circuit
from ..ir import Circuit, GateOperation
from ..result import Result
from .qpe import _add_iqft


def _marked_states(oracle: Any, n_qubits: int) -> list[int]:
    if isinstance(oracle, str):
        if len(oracle) != n_qubits:
            raise ValueError(tr("err.oracle_len", oracle=oracle, n=len(oracle), n_qubits=n_qubits))
        return [int(oracle, 2)]
    if hasattr(oracle, "marked"):  # output of the @oracle decorator
        if oracle.n_qubits != n_qubits:
            raise ValueError(tr("err.oracle_n_qubits", n=oracle.n_qubits, n_qubits=n_qubits))
        return list(oracle.marked)
    if callable(oracle):  # bare predicate f(x) -> bool
        states = [x for x in range(2 ** n_qubits) if oracle(x)]
        if not states:
            raise ValueError(tr("err.oracle_empty"))
        return states
    raise TypeError(tr("err.oracle_type"))


def _add_controlled_oracle(
    circuit: Circuit, control: int, search: Sequence[int], marked: Sequence[int]
) -> None:
    n = len(search)
    for x in marked:
        bits = format(x, f"0{n}b")
        for q in range(n):
            if bits[n - 1 - q] == "0":
                circuit.add(GateOperation("x", (search[q],)))
        circuit.add(GateOperation("mcz", tuple([control] + list(search))))
        for q in range(n):
            if bits[n - 1 - q] == "0":
                circuit.add(GateOperation("x", (search[q],)))


def _add_controlled_diffusion(
    circuit: Circuit, control: int, search: Sequence[int]
) -> None:
    for q in search:
        circuit.add(GateOperation("h", (q,)))
    for q in search:
        circuit.add(GateOperation("x", (q,)))
    circuit.add(GateOperation("mcz", tuple([control] + list(search))))
    for q in search:
        circuit.add(GateOperation("x", (q,)))
    for q in search:
        circuit.add(GateOperation("h", (q,)))


def _add_controlled_grover(
    circuit: Circuit, control: int, search: Sequence[int], marked: Sequence[int]
) -> None:
    _add_controlled_oracle(circuit, control, search, marked)
    _add_controlled_diffusion(circuit, control, search)


def quantum_counting(
    oracle: Any,
    n_qubits: int,
    t: int | None = None,
    backend: str = "auto",
    shots: int = 1024,
) -> Result:
    """Estimate the number M of solutions satisfying the oracle.

    Args:
        oracle: A marking bitstring, the output of the @oracle decorator, or a
            predicate f(x)->bool.
        n_qubits: Number of search-space qubits (N = 2**n_qubits).
        t: Number of counting qubits, default n_qubits + 1 (larger gives a more
            accurate estimate).
        backend / shots: Sampling parameters.

    Returns: Result (kind="value"); result.value is the estimate of M.
    """
    marked = _marked_states(oracle, n_qubits)
    if t is None:
        t = n_qubits + 1

    search = list(range(t, t + n_qubits))
    circuit = Circuit()
    for q in range(t + n_qubits):
        circuit.add(GateOperation("h", (q,)))

    for j in range(t):
        for _ in range(2 ** (t - 1 - j)):
            _add_controlled_grover(circuit, j, search, marked)

    _add_iqft(circuit, t)

    result = run_circuit(circuit, backend=backend, shots=shots)
    best = max(result.counts, key=result.counts.get)
    j = int(best[-t:], 2)  # the rightmost t bits are the counting qubits
    m = 2 ** n_qubits * math.sin(math.pi * abs(j / 2 ** t - 0.5)) ** 2
    return Result.from_value(m, j=j, t=t, counts=result.counts)
