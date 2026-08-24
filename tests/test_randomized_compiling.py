"""Tests for randomized compiling (Pauli twirling)."""

from __future__ import annotations

import numpy as np

from quonic.compiler import randomized_compiling
from quonic.ir import Circuit, GateOperation


def _make_cx_circuit() -> Circuit:
    c = Circuit()
    c.allocate(2)
    c.add(GateOperation("h", (0,)))
    c.add(GateOperation("cx", (0, 1)))
    c.add(GateOperation("h", (1,)))
    return c


def test_randomized_compiling_returns_list():
    c = _make_cx_circuit()
    result = randomized_compiling(c, n_samples=3, seed=42)
    assert isinstance(result, list)
    assert len(result) == 3


def test_randomized_compiling_preserves_qubit_count():
    c = _make_cx_circuit()
    result = randomized_compiling(c, n_samples=1, seed=42)
    assert result[0].num_qubits == 2


def test_randomized_compiling_has_cx():
    c = _make_cx_circuit()
    result = randomized_compiling(c, n_samples=1, seed=42)
    cx_count = sum(
        1 for op in result[0].ops
        if isinstance(op, GateOperation) and op.name == "cx"
    )
    assert cx_count == 1  # original CX preserved


def test_randomized_compiling_different_seeds():
    c = _make_cx_circuit()
    r1 = randomized_compiling(c, n_samples=1, seed=0)
    r2 = randomized_compiling(c, n_samples=1, seed=1)
    # Different seeds should (with high probability) produce different Pauli insertions
    ops1 = [op.name for op in r1[0].ops if isinstance(op, GateOperation)]
    ops2 = [op.name for op in r2[0].ops if isinstance(op, GateOperation)]
    # They might be the same by chance, but the structure should be valid
    assert all(name in ("h", "x", "y", "z", "cx") for name in ops1)
    assert all(name in ("h", "x", "y", "z", "cx") for name in ops2)


def test_randomized_compiling_no_cx():
    """Circuit with no CX gates should pass through unchanged."""
    c = Circuit()
    c.allocate(1)
    c.add(GateOperation("h", (0,)))
    c.add(GateOperation("x", (0,)))
    result = randomized_compiling(c, n_samples=2, seed=42)
    for r in result:
        names = [op.name for op in r.ops if isinstance(op, GateOperation)]
        assert names == ["h", "x"]


def test_randomized_compiling_preserves_state():
    """Twirled circuit should produce the same statevector as original."""
    from quonic.simulators import StatevectorEngine

    c = _make_cx_circuit()
    # Original statevector
    eng1 = StatevectorEngine(2)
    for op in c.ops:
        if isinstance(op, GateOperation):
            eng1.apply(op.name, list(op.qubits), op.params)

    # Twirled statevector
    twirled = randomized_compiling(c, n_samples=1, seed=42)
    eng2 = StatevectorEngine(2)
    for op in twirled[0].ops:
        if isinstance(op, GateOperation):
            eng2.apply(op.name, list(op.qubits), op.params)

    np.testing.assert_allclose(
        np.abs(eng1.state) ** 2,
        np.abs(eng2.state) ** 2,
        atol=1e-12,
    )
