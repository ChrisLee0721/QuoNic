"""Generic tests for engine backends (qulacs / tensorcircuit / cudaq / ...).

Every available engine backend is exercised with the same benchmark circuits.
Backends whose SDK is not installed are skipped via ``pytest.importorskip``.
"""

from __future__ import annotations

import math

import pytest

from quonic import qgate, reset
from quonic.backends import get_backend
from quonic.gates import CCX, CX, SWAP, H, Ry, X, Y, Z
from quonic.stack import current_circuit

BACKENDS = [
    "qulacs",
    "tensorcircuit",
    "cudaq",
    "mindquantum",
    "qpanda",
    "cqlib",
]

# Backend name -> importable module name (when they differ)
_MODULE_MAP = {
    "qpanda": "pyqpanda3",
}


def _import_backend(backend: str):
    # Apply compat patches before importing backends
    if backend == "tensorcircuit":
        from quonic.backends.tensorcircuit import _ensure_tc_numpy_compat

        _ensure_tc_numpy_compat()
    mod = _MODULE_MAP.get(backend, backend)
    return pytest.importorskip(mod)


def _run(backend: str, shots: int = 256):
    return get_backend(backend).run(current_circuit(), shots=shots)


# ---------------------------------------------------------------------------
# 1. Bell state — should produce ~50/50 |00> and |11>
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_bell(backend):
    _import_backend(backend)
    reset()
    qgate(H, 0)
    qgate(CX, 0, 1)
    result = _run(backend, shots=1024)
    p00 = result.counts.get("00", 0) / 1024
    p11 = result.counts.get("11", 0) / 1024
    assert p00 > 0.3 and p11 > 0.3
    assert p00 + p11 > 0.9


# ---------------------------------------------------------------------------
# 2. GHZ-3 — should produce ~50/50 |000> and |111>
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_ghz3(backend):
    _import_backend(backend)
    reset()
    qgate(H, 0)
    qgate(CX, 0, 1)
    qgate(CX, 1, 2)
    result = _run(backend, shots=1024)
    p000 = result.counts.get("000", 0) / 1024
    p111 = result.counts.get("111", 0) / 1024
    assert p000 > 0.3 and p111 > 0.3
    assert p000 + p111 > 0.9


# ---------------------------------------------------------------------------
# 3. Single-qubit rotation — Ry(pi) flips |0> to |1>
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_ry_pi(backend):
    _import_backend(backend)
    reset()
    qgate(Ry(math.pi), 0)
    result = _run(backend, shots=100)
    assert result.counts.get("1", 0) > 95


# ---------------------------------------------------------------------------
# 4. Toffoli (CCX) — |110> -> |111>
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_ccx(backend):
    _import_backend(backend)
    reset()
    qgate(X, 0)
    qgate(X, 1)
    qgate(CCX, 0, 1, 2)
    result = _run(backend, shots=100)
    assert result.counts.get("111", 0) == 100


# ---------------------------------------------------------------------------
# 5. Pauli gates — X, Y, Z on |0>
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_pauli_x(backend):
    _import_backend(backend)
    reset()
    qgate(X, 0)
    result = _run(backend, shots=100)
    assert result.counts.get("1", 0) == 100


@pytest.mark.parametrize("backend", BACKENDS)
def test_pauli_y(backend):
    _import_backend(backend)
    reset()
    qgate(Y, 0)
    result = _run(backend, shots=100)
    assert result.counts.get("1", 0) == 100


@pytest.mark.parametrize("backend", BACKENDS)
def test_pauli_z(backend):
    _import_backend(backend)
    reset()
    qgate(Z, 0)
    result = _run(backend, shots=100)
    assert result.counts.get("0", 0) == 100


# ---------------------------------------------------------------------------
# 6. SWAP — |10> -> |01>
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_swap(backend):
    _import_backend(backend)
    reset()
    qgate(X, 0)
    qgate(SWAP, 0, 1)
    result = _run(backend, shots=100)
    assert result.counts.get("01", 0) == 100


# ---------------------------------------------------------------------------
# 7. Noise — backends with DM support should produce leakage
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_bell_noisy(backend):
    """Noise should produce leakage in a Bell state."""
    _import_backend(backend)
    reset()
    qgate(H, 0)
    qgate(CX, 0, 1)
    be = get_backend(backend)
    shots = 512  # reduced for slow backends (TensorCircuit DMCircuit)
    try:
        result = be.run(current_circuit(), shots=shots, noise=0.05)
    except NotImplementedError:
        pytest.skip(f"{backend} does not support noise yet")
    leakage = result.counts.get("01", 0) + result.counts.get("10", 0)
    assert leakage / shots > 0.003  # noise produced some leakage
    assert leakage / shots < 0.45   # not too much


# ---------------------------------------------------------------------------
# 8. Density matrix method — should produce same results as statevector
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_density_matrix_method(backend):
    """Density matrix method should produce correct Bell state."""
    _import_backend(backend)
    reset()
    qgate(H, 0)
    qgate(CX, 0, 1)
    be = get_backend(backend)
    if "density_matrix" not in be.methods:
        pytest.skip(f"{backend} does not support density_matrix method")
    result = be.run(current_circuit(), shots=1024, method="density_matrix")
    p00 = result.counts.get("00", 0) / 1024
    p11 = result.counts.get("11", 0) / 1024
    assert p00 > 0.3 and p11 > 0.3
    assert p00 + p11 > 0.9
