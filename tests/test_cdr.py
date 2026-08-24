"""Clifford Data Regression tests."""

from __future__ import annotations

import pytest

from quonic.ir import Circuit, GateOperation
from quonic.mitigation import CDRResult, cdr


def _make_x_circuit() -> Circuit:
    """Simple X gate circuit."""
    c = Circuit()
    c.add(GateOperation("x", (0,)))
    return c


def test_cdr_result_fields():
    result = CDRResult(value=-0.9, r2_score=0.95, n_training_circuits=10)
    assert result.value == -0.9
    assert result.r2_score == 0.95
    assert result.n_training_circuits == 10


def test_cdr_requires_noise():
    c = _make_x_circuit()
    with pytest.raises(ValueError, match="requires noise"):
        cdr(c, noise=None, observable="Z")


def test_cdr_runs():
    c = _make_x_circuit()
    result = cdr(c, noise=0.01, observable="Z", n_training=5, seed=42)
    assert isinstance(result, CDRResult)
    assert result.n_training_circuits == 5
    assert -2.0 < result.value < 2.0


def test_cdr_two_qubit():
    c = Circuit()
    c.add(GateOperation("h", (0,)))
    c.add(GateOperation("cx", (0, 1)))
    result = cdr(c, noise=0.02, observable="ZZ", n_training=8, seed=42)
    assert isinstance(result, CDRResult)
    assert result.r2_score <= 1.0
