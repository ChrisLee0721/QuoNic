"""Data encoding — encode classical data into quantum states.

Example::

    from quonic.ml import angle_encode, amplitude_encode
    circuit = angle_encode([0.5, 1.0, 1.5])
    circuit = amplitude_encode([0.7, 0.3, 0.5, 0.3])
"""

from __future__ import annotations

import math

from ..ir import Circuit, GateOperation


def angle_encode(features: list[float]) -> Circuit:
    """Angle encoding: each feature encoded as a Ry rotation.

    Ry(θ)|0> = cos(θ/2)|0> + sin(θ/2)|1>

    Args:
        features: list of feature values (angles in radians)

    Returns:
        Circuit with Ry rotations on successive qubits.
    """
    c = Circuit()
    c.allocate(len(features))
    for i, theta in enumerate(features):
        c.add(GateOperation("ry", (i,), (float(theta),)))
    return c


def amplitude_encode(data: list[float]) -> Circuit:
    """Amplitude encoding: encode a real vector into quantum amplitudes.

    Uses a recursive decomposition to prepare the state.
    The data vector is normalized internally.

    Args:
        data: real-valued data vector (length must be a power of 2)

    Returns:
        Circuit that prepares the encoded state.
    """
    import numpy as np

    vec = np.array(data, dtype=float)
    vec = vec / np.linalg.norm(vec)
    n = int(np.log2(len(vec)))
    if 2**n != len(vec):
        raise ValueError(f"Data length {len(vec)} is not a power of 2")

    c = Circuit()
    c.allocate(n)
    _prepare_state(c, vec, list(range(n)))
    return c


def iqp_encode(features: list[float]) -> Circuit:
    """IQP encoding: encode features as diagonal unitaries.

    Applies Rz(feature) to each qubit, then CZ between all pairs.

    Args:
        features: list of feature values

    Returns:
        Circuit with IQP encoding.
    """
    n = len(features)
    c = Circuit()
    c.allocate(n)
    # Hadamard layer
    for q in range(n):
        c.add(GateOperation("h", (q,)))
    # Rz rotations
    for q in range(n):
        c.add(GateOperation("rz", (q,), (float(features[q]),)))
    # CZ entangling layer
    for i in range(n):
        for j in range(i + 1, n):
            c.add(GateOperation("cz", (i, j)))
    # Another Hadamard layer
    for q in range(n):
        c.add(GateOperation("h", (q,)))
    return c


def _prepare_state(circuit: Circuit, vec, qubits: list[int]) -> None:
    """Recursively prepare a state vector using controlled rotations."""
    import numpy as np

    n = len(qubits)
    if n == 0:
        return

    if n == 1:
        prob0 = abs(vec[0]) ** 2
        if prob0 > 0.999:
            return
        theta = 2 * math.acos(min(1.0, abs(vec[0])))
        circuit.add(GateOperation("ry", (qubits[0],), (theta,)))
        return

    mid = len(vec) // 2
    upper = vec[:mid]
    lower = vec[mid:]

    norm_upper = np.linalg.norm(upper)
    norm_lower = np.linalg.norm(lower)

    if norm_upper + norm_lower > 0:
        theta = 2 * math.acos(min(1.0, norm_upper / (norm_upper + norm_lower)))
        circuit.add(GateOperation("ry", (qubits[0],), (theta,)))

    if norm_upper > 1e-10:
        _prepare_state(circuit, upper / norm_upper, qubits[1:])
    if norm_lower > 1e-10:
        _prepare_state(circuit, lower / norm_lower, qubits[1:])
