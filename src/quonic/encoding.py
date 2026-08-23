"""Data encoding — encode classical data into quantum states.

Example::

    from quonic.encoding import angle_encode
    circuit = angle_encode([0.5, 1.0, 1.5])
"""

from __future__ import annotations

import math

from .ir import Circuit, GateOperation


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

    # Use state preparation via uniformly controlled rotations
    # This is a simplified version; a full implementation would use
    # the Shende-Bullock-Markov decomposition
    _prepare_state(c, vec, list(range(n)))
    return c


def _prepare_state(circuit: Circuit, vec, qubits: list[int]) -> None:
    """Recursively prepare a state vector using controlled rotations."""
    import numpy as np

    n = len(qubits)
    if n == 0:
        return

    if n == 1:
        # Single qubit: find rotation angle
        prob0 = abs(vec[0]) ** 2
        if prob0 > 0.999:
            return  # already |0>
        theta = 2 * math.acos(min(1.0, abs(vec[0])))
        circuit.add(GateOperation("ry", (qubits[0],), (theta,)))
        return

    # Split into upper and lower halves
    mid = len(vec) // 2
    upper = vec[:mid]
    lower = vec[mid:]

    norm_upper = np.linalg.norm(upper)
    norm_lower = np.linalg.norm(lower)

    # Rotate to split amplitude between upper and lower
    if norm_upper + norm_lower > 0:
        theta = 2 * math.acos(min(1.0, norm_upper / (norm_upper + norm_lower)))
        circuit.add(GateOperation("ry", (qubits[0],), (theta,)))

    # Recurse on each half
    if norm_upper > 1e-10:
        _prepare_state(circuit, upper / norm_upper, qubits[1:])
    if norm_lower > 1e-10:
        _prepare_state(circuit, lower / norm_lower, qubits[1:])
