"""Static capability matrix for simulation methods: the "hard constraint" layer of scheduling decisions.

A capability mismatch is a hard constraint (the method is excluded outright);
performance data is a "soft selection" (pick the fastest among the remaining
methods). Only static, machine-independent facts are recorded here:

- Which gates each method can handle (basic Clifford / full Clifford / arbitrary non-Clifford)
- Whether each method supports depolarizing noise

Performance data (timings) is measured in benchmark.py, keeping the two
separate: capabilities are stable and free to ship, while performance drifts
with the machine and needs re-calibration.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

BASIC_CLIFFORD: set[str] = {"h", "x", "y", "z", "cx", "cz"}
"""Aer's stabilizer method only accepts this set of basic Clifford gates (no mcz / cp / ccx)."""

CLIFFORD_GATES: set[str] = BASIC_CLIFFORD | {"mcz"}
"""The full Clifford gate set (including multi-controlled Z). Used for the is_clifford check."""


METHOD_CAPABILITIES: dict[str, dict[str, Any]] = {
    "statevector": {
        "clifford": True,
        "nonclifford": True,
        "noise": False,
        "gates": "all",
    },
    "stabilizer": {
        "clifford": True,  # basic Clifford only (no mcz)
        "nonclifford": False,
        "noise": False,
        "gates": "basic_clifford",
    },
    "matrix_product_state": {
        "clifford": True,
        "nonclifford": True,
        "noise": False,
        "gates": "all",
    },
    "density_matrix": {
        "clifford": True,
        "nonclifford": True,
        "noise": True,
        "gates": "all",
    },
    "gpu": {
        "clifford": True,
        "nonclifford": True,
        "noise": True,
        "gates": "all",
    },
}


BACKEND_CAPABILITIES: dict[str, set[str]] = {
    # Which simulation methods each backend supports (including v2 upgrades).
    "qiskit": {"statevector", "stabilizer", "matrix_product_state", "density_matrix", "gpu"},
    "cirq": {"statevector"},
    "pennylane": {"statevector", "autodiff"},
    "native": {"statevector", "stabilizer", "matrix_product_state", "density_matrix"},
    "qi": {"statevector"},
    "qulacs": {"statevector", "density_matrix", "gpu"},
    "tensorcircuit": {"statevector", "density_matrix", "gpu", "autodiff"},
    "cudaq": {"statevector", "density_matrix", "gpu"},
    "mindquantum": {"statevector", "density_matrix", "gpu"},
    "qpanda": {"statevector", "density_matrix", "gpu"},
    "cqlib": {"statevector", "density_matrix"},
    "cupy": {"statevector", "density_matrix", "gpu"},
}


def eligible_methods(gate_types: Iterable[str], noise: bool = False) -> set[str]:
    """Return the set of methods that can run this circuit (capability hard constraints).

    - noise -> only density_matrix supports it
    - basic Clifford -> statevector / stabilizer / matrix_product_state
    - otherwise (mcz / arbitrary-angle rotations, etc.) -> statevector / matrix_product_state
    """
    if noise:
        return {"density_matrix"}
    gs = set(gate_types)
    methods = {"statevector", "matrix_product_state"}
    if gs <= BASIC_CLIFFORD:
        methods.add("stabilizer")
    return methods


def decision_class(features: dict[str, Any]) -> str:
    """Classify circuit features into three decision classes, matching benchmark circuit families one-to-one.

    - "clifford"  -- pure basic Clifford, where stabilizer shines
    - "low_tw"    -- non-basic Clifford but low treewidth, where MPS shines
    - "general"   -- high entanglement / high treewidth, only statevector runs efficiently
    """
    gs = set(features["gate_types"])
    if gs and gs <= BASIC_CLIFFORD:
        return "clifford"
    if features["treewidth_ub"] <= 4:
        return "low_tw"
    return "general"


__all__ = [
    "BACKEND_CAPABILITIES",
    "BASIC_CLIFFORD",
    "CLIFFORD_GATES",
    "METHOD_CAPABILITIES",
    "decision_class",
    "eligible_methods",
]
