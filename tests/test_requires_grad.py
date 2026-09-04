"""Tests for requires_grad feature — autodiff-aware scheduling."""

from __future__ import annotations

from quonic import qgate, reset
from quonic.gates import H
from quonic.ir import Circuit
from quonic.scheduler import (
    circuit_features,
    recommend_backend_autodiff,
    recommend_method,
)
from quonic.stack import current_circuit

# ---------------------------------------------------------------------------
# 1. Circuit.requires_grad attribute
# ---------------------------------------------------------------------------


def test_requires_grad_default_false():
    c = Circuit()
    assert c.requires_grad is False


def test_requires_grad_set_true():
    c = Circuit()
    c.requires_grad = True
    assert c.requires_grad is True


# ---------------------------------------------------------------------------
# 2. circuit_features propagation
# ---------------------------------------------------------------------------


def test_features_includes_requires_grad_false():
    reset()
    qgate(H, 0)
    feats = circuit_features(current_circuit())
    assert "requires_grad" in feats
    assert feats["requires_grad"] is False


def test_features_includes_requires_grad_true():
    reset()
    qgate(H, 0)
    current_circuit().requires_grad = True
    feats = circuit_features(current_circuit())
    assert feats["requires_grad"] is True


# ---------------------------------------------------------------------------
# 3. recommend_method with requires_grad
# ---------------------------------------------------------------------------


def test_recommend_method_grad_returns_statevector():
    feats = {"n": 4, "gate_types": ["h", "cx"], "requires_grad": True}
    rec = recommend_method(feats)
    assert rec.method == "statevector"


def test_recommend_method_no_grad_normal():
    feats = {"n": 4, "gate_types": ["h", "cx"], "is_clifford": True,
             "treewidth_ub": 0, "requires_grad": False}
    rec = recommend_method(feats)
    # Clifford with low treewidth: benchmarks show density_matrix is fastest
    assert rec.method in ("statevector", "stabilizer", "density_matrix")


# ---------------------------------------------------------------------------
# 4. recommend_backend_autodiff
# ---------------------------------------------------------------------------


def test_autodiff_small_circuit():
    feats = {"n": 10}
    rec = recommend_backend_autodiff(feats)
    assert rec.backend == "pennylane"
    assert rec.method == "statevector"


def test_autodiff_large_circuit():
    feats = {"n": 25}
    rec = recommend_backend_autodiff(feats)
    assert rec.backend == "tensorcircuit"
    assert rec.method == "statevector"


# ---------------------------------------------------------------------------
# 5. qshow integration
# ---------------------------------------------------------------------------


def test_qshow_sets_requires_grad():
    reset()
    qgate(H, 0)
    circ = current_circuit()
    assert circ.requires_grad is False
    # After setting manually
    circ.requires_grad = True
    assert circ.requires_grad is True
    feats = circuit_features(circ)
    assert feats["requires_grad"] is True
