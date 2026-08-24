"""Probabilistic Error Cancellation tests."""

from __future__ import annotations

import pytest

from quonic.ir import Circuit, GateOperation
from quonic.mitigation import PECResult, pec


def _make_x_circuit() -> Circuit:
    c = Circuit()
    c.add(GateOperation("x", (0,)))
    return c


def test_pec_result_fields():
    result = PECResult(value=-0.9, variance=0.01, n_samples=100, overhead=1.5)
    assert result.value == -0.9
    assert result.variance == 0.01
    assert result.n_samples == 100
    assert result.overhead == 1.5


def test_pec_requires_noise():
    c = _make_x_circuit()
    with pytest.raises(ValueError, match="requires noise"):
        pec(c, noise=None)


def test_pec_runs():
    c = _make_x_circuit()
    result = pec(c, noise=0.01, observable="Z", n_samples=100, seed=42)
    assert isinstance(result, PECResult)
    assert result.n_samples == 100
    assert result.overhead > 1.0
    assert -2.0 < result.value < 2.0


def test_pec_overhead_increases_with_noise():
    c = _make_x_circuit()
    r1 = pec(c, noise=0.01, observable="Z", n_samples=50, seed=42)
    r2 = pec(c, noise=0.1, observable="Z", n_samples=50, seed=42)
    assert r2.overhead > r1.overhead
