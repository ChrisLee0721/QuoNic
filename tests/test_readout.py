"""Readout error mitigation: confusion-matrix ordering, inversion, and end-to-end recovery."""

import numpy as np
import pytest

from quonic import NoiseModel, calibrate
from quonic.ir import Circuit, GateOperation
from quonic.readout import ReadoutCalibration

# ---------------------------------------------------------------------------
# matrix ordering
# ---------------------------------------------------------------------------

def test_matrix_ordering_single_qubit():
    m = np.array([[[0.9, 0.1], [0.2, 0.8]]])
    cal = ReadoutCalibration(m, 1)
    assert cal.matrix.shape == (2, 2)
    assert np.allclose(cal.matrix, m[0])


def test_matrix_ordering_two_qubit():
    # qubit 0 matrix, qubit 1 matrix (A = A1 ⊗ A0, index i = bit1*2 + bit0)
    m = np.array([
        [[0.9, 0.1], [0.2, 0.8]],  # qubit 0
        [[0.8, 0.2], [0.3, 0.7]],  # qubit 1
    ])
    A = ReadoutCalibration(m, 2).matrix
    # A[i,j]: i = true state, j = read state
    assert A[0, 0] == pytest.approx(0.8 * 0.9)  # both stay 00
    assert A[1, 2] == pytest.approx(0.2 * 0.2)  # 01 -> 10 (q1 0->1, q0 1->0)
    assert A[3, 3] == pytest.approx(0.7 * 0.8)  # 11 -> 11
    # each row (fixed true state) sums to 1 over its read distribution
    assert np.allclose(A.sum(axis=1), 1.0)


# ---------------------------------------------------------------------------
# inversion
# ---------------------------------------------------------------------------

def test_apply_recovers_asymmetric_readout():
    # single qubit with asymmetric misread: P(1|0)=0.1, P(0|1)=0.3
    m = np.array([[[0.9, 0.1], [0.3, 0.7]]])
    cal = ReadoutCalibration(m, 1)
    # true: always |1>; measured: 70% "1", 30% "0"
    measured = {"1": 700, "0": 300}
    corrected = cal.apply(measured, 1000)
    p1 = corrected.get("1", 0) / sum(corrected.values())
    assert p1 > 0.95


def test_apply_is_identity_without_readout_error():
    m = np.array([[[1.0, 0.0], [0.0, 1.0]]])
    cal = ReadoutCalibration(m, 1)
    corrected = cal.apply({"1": 1000}, 1000)
    assert corrected == {"1": 1000}


def test_apply_singular_uses_regularization():
    # p = 0.5 readout is a degenerate (uninformative) confusion matrix
    # With Tikhonov regularization, this should not raise but return a result
    m = np.array([[[0.5, 0.5], [0.5, 0.5]]])
    cal = ReadoutCalibration(m, 1)
    result = cal.apply({"1": 1000}, 1000)
    # Regularized result should still be a valid counts dict
    assert isinstance(result, dict)
    assert sum(result.values()) > 0


# ---------------------------------------------------------------------------
# noise model readout field
# ---------------------------------------------------------------------------

def test_noise_model_readout_enabled():
    assert NoiseModel(readout=0.1).enabled
    assert not NoiseModel().enabled


def test_noise_model_readout_validation():
    with pytest.raises(ValueError):
        NoiseModel(readout=1.5)


# ---------------------------------------------------------------------------
# calibrate + apply end-to-end (native backend injects readout error)
# ---------------------------------------------------------------------------

def test_calibrate_identity_without_noise():
    cal = calibrate(1, backend="native", shots=4096)
    # no readout error -> confusion matrix ~ identity
    assert np.allclose(cal.matrix, np.eye(2), atol=1e-6)


def test_calibrate_and_apply_recovers():
    # X|0> = |1> on qubit 0; with 15% readout error the measured "1" is diluted.
    p = 0.15
    nm = NoiseModel(readout=p)
    c = Circuit()
    c.add(GateOperation("x", (0,)))
    c.add(GateOperation("measure", (0,)))

    from quonic.backends import get_backend

    raw = get_backend("native").run(c, shots=16384, noise=nm)
    raw_p1 = raw.counts.get("1", 0) / raw.shots
    # readout error flips ~15% of the |1> outcomes to "0"
    assert raw_p1 < 1.0 - p / 2

    cal = calibrate(1, backend="native", shots=16384, noise=nm)
    corrected = cal.apply(raw.counts, raw.shots)
    corr_p1 = corrected.get("1", 0) / sum(corrected.values())
    # mitigated success probability is much closer to the true value 1
    assert corr_p1 > raw_p1 + 0.05
    assert corr_p1 > 0.9


def test_calibrate_two_qubit_recovers():
    # Bell + readout error on both qubits; the corrected ZZ-like correlation recovers
    p = 0.1
    nm = NoiseModel(readout=p)
    c = Circuit()
    c.add(GateOperation("h", (0,)))
    c.add(GateOperation("cx", (0, 1)))
    c.add(GateOperation("measure", (0,)))
    c.add(GateOperation("measure", (1,)))

    from quonic.backends import get_backend

    raw = get_backend("native").run(c, shots=16384, noise=nm)
    cal = calibrate(2, backend="native", shots=16384, noise=nm)
    corrected = cal.apply(raw.counts, raw.shots)
    total = sum(corrected.values())
    # Bell state |00>+|11>: recovered parity-correlated states dominate
    corr_same = (corrected.get("00", 0) + corrected.get("11", 0)) / total
    assert corr_same > 0.95


def test_calibrate_rejects_bad_n():
    with pytest.raises(ValueError):
        calibrate(0, backend="native")
    with pytest.raises(ValueError):
        calibrate(-1, backend="native")


def test_calibrate_and_apply_qiskit_recovers():
    # qiskit backend injects readout error via ReadoutError; calibration should recover.
    pytest.importorskip("qiskit_aer")
    p = 0.1
    nm = NoiseModel(readout=p)
    c = Circuit()
    c.add(GateOperation("x", (0,)))
    c.add(GateOperation("measure", (0,)))

    from quonic.backends import get_backend

    raw = get_backend("qiskit").run(c, shots=16384, noise=nm)
    cal = calibrate(1, backend="qiskit", shots=16384, noise=nm)
    corrected = cal.apply(raw.counts, raw.shots)
    corr_p1 = corrected.get("1", 0) / sum(corrected.values())
    assert corr_p1 > 0.9


# ---------------------------------------------------------------------------
# correlated (full 2^n matrix) calibration
# ---------------------------------------------------------------------------

def test_correlated_matrix_returns_full():
    m = np.array([[[0.9, 0.1], [0.2, 0.8]]])
    full = np.array([[0.7, 0.3], [0.1, 0.9]])
    cal = ReadoutCalibration(m, 1, full)
    assert np.allclose(cal.matrix, full)


def test_correlated_recovers_crosstalk():
    # 2-qubit full matrix with crosstalk: qubit 1's misread depends on qubit 0.
    # Rows sum to 1; the tensor-product model cannot express row 1's asymmetry.
    A = np.array([
        [0.8, 0.1, 0.1, 0.0],
        [0.0, 0.7, 0.05, 0.25],
        [0.15, 0.05, 0.8, 0.0],
        [0.0, 0.2, 0.05, 0.75],
    ])
    cal = ReadoutCalibration(np.zeros((2, 2, 2)), 2, A)
    # true state always |11>: measured distribution is row 3
    shots = 1000
    measured = {format(j, "02b"): round(A[3, j] * shots) for j in range(4)}
    measured = {bs: c for bs, c in measured.items() if c}
    corrected = cal.apply(measured, shots)
    total = sum(corrected.values())
    assert corrected.get("11", 0) / total > 0.9


def test_marginals_from_full():
    from quonic.readout import _marginals_from_full

    m0 = np.array([[0.9, 0.1], [0.2, 0.8]])
    m1 = np.array([[0.8, 0.2], [0.3, 0.7]])
    full = np.kron(m1, m0)  # A = A1 ⊗ A0 (qubit 0 LSB)
    mat = _marginals_from_full(full, 2)
    assert np.allclose(mat[0], m0)
    assert np.allclose(mat[1], m1)


def test_calibrate_correlated_end_to_end():
    # native backend with independent readout noise: correlated calibration recovers.
    p = 0.1
    nm = NoiseModel(readout=p)
    c = Circuit()
    c.add(GateOperation("h", (0,)))
    c.add(GateOperation("cx", (0, 1)))
    c.add(GateOperation("measure", (0,)))
    c.add(GateOperation("measure", (1,)))

    from quonic.backends import get_backend

    raw = get_backend("native").run(c, shots=16384, noise=nm)
    cal = calibrate(2, backend="native", shots=16384, noise=nm, correlated=True)
    assert cal.full is not None
    assert cal.full.shape == (4, 4)
    corrected = cal.apply(raw.counts, raw.shots)
    total = sum(corrected.values())
    corr_same = (corrected.get("00", 0) + corrected.get("11", 0)) / total
    assert corr_same > 0.95


def test_calibrate_correlated_rejects_large_n():
    with pytest.raises(ValueError):
        calibrate(13, backend="native", correlated=True)
