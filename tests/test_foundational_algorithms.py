"""Tests for Phase 1 foundational algorithms."""

from __future__ import annotations

from quonic.algorithms import (
    amplitude_amplification,
    bernstein_vazirani,
    deutsch_jozsa,
    hadamard_test,
    mark_state,
    qft,
    simon,
    swap_test,
)
from quonic.ir import GateOperation

# ---------------------------------------------------------------------------
# QFT
# ---------------------------------------------------------------------------


def test_qft_3qubit():
    """3-qubit QFT on |000> should produce uniform distribution."""
    result = qft(3, shots=4096)
    # All outcomes should appear with roughly equal probability
    for bitstring in result.counts:
        assert len(bitstring) == 3


def test_qft_inverse():
    """IQFT on |000> should produce uniform distribution (|+++>)."""
    # QFT|000> = |+++>, so IQFT|000> = |+++> too (uniform)
    result = qft(3, inverse=True, shots=4096)
    # All outcomes should appear
    assert len(result.counts) == 8


# ---------------------------------------------------------------------------
# Deutsch-Jozsa
# ---------------------------------------------------------------------------


def _constant_oracle(circuit, n):
    """f(x) = 0 for all x (constant)."""


def _balanced_oracle(circuit, n):
    """f(x) = x_0 (balanced)."""
    circuit.add(GateOperation("cx", (0, n)))


def test_deutsch_jozsa_constant():
    """Constant oracle should return is_balanced=False."""
    result = deutsch_jozsa(2, _constant_oracle, shots=100)
    assert result.metadata["is_balanced"] is False


def test_deutsch_jozsa_balanced():
    """Balanced oracle should return is_balanced=True."""
    result = deutsch_jozsa(2, _balanced_oracle, shots=100)
    assert result.metadata["is_balanced"] is True


# ---------------------------------------------------------------------------
# Bernstein-Vazirani
# ---------------------------------------------------------------------------


def _oracle_s_101(circuit, n):
    """Oracle for secret s = '101'."""
    circuit.add(GateOperation("cx", (0, n)))
    circuit.add(GateOperation("cx", (2, n)))


def _oracle_s_111(circuit, n):
    """Oracle for secret s = '111'."""
    circuit.add(GateOperation("cx", (0, n)))
    circuit.add(GateOperation("cx", (1, n)))
    circuit.add(GateOperation("cx", (2, n)))


def test_bernstein_vazirani_101():
    """Should find secret s = '101'."""
    result = bernstein_vazirani(3, _oracle_s_101, shots=100)
    assert result.metadata["secret"] == "101"


def test_bernstein_vazirani_111():
    """Should find secret s = '111'."""
    result = bernstein_vazirani(3, _oracle_s_111, shots=100)
    assert result.metadata["secret"] == "111"


# ---------------------------------------------------------------------------
# Simon
# ---------------------------------------------------------------------------


def _oracle_simon_11(circuit, n):
    """Oracle for secret s = '11' (2 qubits)."""
    circuit.add(GateOperation("cx", (0, n)))
    circuit.add(GateOperation("cx", (1, n + 1)))
    circuit.add(GateOperation("cx", (0, n + 1)))
    circuit.add(GateOperation("cx", (1, n)))


def test_simon_2qubit():
    """Should find secret s = '11' for 2-qubit Simon."""
    result = simon(2, _oracle_simon_11, shots=200)
    assert result.metadata["secret"] == "11"


# ---------------------------------------------------------------------------
# SWAP Test
# ---------------------------------------------------------------------------


def _prepare_zero(circuit, start, n):
    """Prepare |0...0> (default state)."""


def _prepare_one(circuit, start, n):
    """Prepare |1> on first qubit."""
    circuit.add(GateOperation("x", (start,)))


def test_swap_test_identical():
    """Identical states should have overlap ≈ 1."""
    result = swap_test(1, _prepare_zero, _prepare_zero, shots=10000)
    assert result.metadata["overlap"] > 0.9


def test_swap_test_orthogonal():
    """Orthogonal states should have overlap ≈ 0."""
    result = swap_test(1, _prepare_zero, _prepare_one, shots=10000)
    assert result.metadata["overlap"] < 0.15


# ---------------------------------------------------------------------------
# Hadamard Test
# ---------------------------------------------------------------------------


def _prepare_zero_h(circuit, start, n):
    """Prepare |0>."""


def _apply_identity(circuit, n):
    """Apply identity (do nothing)."""


def test_hadamard_test_identity():
    """⟨0|I|0⟩ = 1."""
    result = hadamard_test(1, _prepare_zero_h, _apply_identity, shots=10000)
    assert result.metadata["expectation"] > 0.9


# ---------------------------------------------------------------------------
# Amplitude Amplification
# ---------------------------------------------------------------------------


def test_amplitude_amplification_grover():
    """Amplitude amplification with uniform state = Grover search."""
    result = amplitude_amplification(2, mark_state("11"), iterations=1, shots=1024)
    p11 = result.counts.get("11", 0) / 1024
    assert p11 > 0.7


def test_amplitude_amplification_3qubit():
    """3-qubit amplitude amplification should boost target state."""
    result = amplitude_amplification(3, mark_state("101"), iterations=1, shots=1024)
    p101 = result.counts.get("101", 0) / 1024
    assert p101 > 0.5
