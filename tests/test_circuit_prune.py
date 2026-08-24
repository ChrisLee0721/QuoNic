"""Tests for circuit pruning optimization pass."""

from __future__ import annotations

from quonic.compiler import optimize, optimize_prune
from quonic.ir import Circuit, GateOperation


def test_prune_identity_gates():
    c = Circuit()
    c.allocate(2)
    c.add(GateOperation("i", (0,)))
    c.add(GateOperation("h", (1,)))
    c.add(GateOperation("i", (1,)))
    result = optimize_prune(c)
    names = [op.name for op in result.ops if isinstance(op, GateOperation)]
    assert names == ["h"]


def test_prune_zero_angle_rotation():
    c = Circuit()
    c.allocate(1)
    c.add(GateOperation("rx", (0,), (0.0,)))
    c.add(GateOperation("ry", (0,), (0.0,)))
    c.add(GateOperation("rz", (0,), (0.0,)))
    c.add(GateOperation("h", (0,)))
    result = optimize_prune(c)
    names = [op.name for op in result.ops if isinstance(op, GateOperation)]
    assert names == ["h"]


def test_prune_nonzero_rotation_kept():
    c = Circuit()
    c.allocate(1)
    c.add(GateOperation("rx", (0,), (0.5,)))
    c.add(GateOperation("rz", (0,), (0.0,)))
    c.add(GateOperation("ry", (0,), (0.1,)))
    result = optimize_prune(c)
    names = [op.name for op in result.ops if isinstance(op, GateOperation)]
    assert names == ["rx", "ry"]


def test_prune_cancels_adjacent_self_inverse():
    c = Circuit()
    c.allocate(1)
    c.add(GateOperation("x", (0,)))
    c.add(GateOperation("x", (0,)))
    result = optimize_prune(c)
    names = [op.name for op in result.ops if isinstance(op, GateOperation)]
    assert names == []


def test_prune_preserves_measurements():
    c = Circuit()
    c.allocate(1)
    c.add(GateOperation("i", (0,)))
    c.add(GateOperation("measure", (0,)))
    result = optimize_prune(c)
    names = [op.name for op in result.ops if isinstance(op, GateOperation)]
    assert "measure" in names


def test_prune_in_optimize_pipeline():
    c = Circuit()
    c.allocate(1)
    c.add(GateOperation("i", (0,)))
    c.add(GateOperation("x", (0,)))
    c.add(GateOperation("x", (0,)))
    c.add(GateOperation("i", (0,)))
    result = optimize(c, passes=("prune",))
    names = [op.name for op in result.ops if isinstance(op, GateOperation)]
    assert names == []


def test_prune_cp_zero_angle():
    c = Circuit()
    c.allocate(2)
    c.add(GateOperation("cp", (0, 1), (0.0,)))
    c.add(GateOperation("cx", (0, 1)))
    result = optimize_prune(c)
    names = [op.name for op in result.ops if isinstance(op, GateOperation)]
    assert names == ["cx"]


def test_prune_tolerance():
    c = Circuit()
    c.allocate(1)
    c.add(GateOperation("rz", (0,), (1e-15,)))
    c.add(GateOperation("rz", (0,), (0.5,)))
    result = optimize_prune(c, tol=1e-12)
    names = [op.name for op in result.ops if isinstance(op, GateOperation)]
    assert names == ["rz"]
