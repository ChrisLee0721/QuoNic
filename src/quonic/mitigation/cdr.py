"""Clifford Data Regression (CDR) — learn noise via near-Clifford circuits.

Generates near-Clifford training circuits whose ideal results can be computed
efficiently, runs them on the noisy backend, fits a linear regression model
(noisy → ideal), and applies it to the target circuit.

Example::

    from quonic.ir import Circuit, GateOperation
    from quonic.mitigation import cdr

    circ = Circuit()
    circ.add(GateOperation("h", (0,)))
    circ.add(GateOperation("cx", (0, 1)))
    circ.add(GateOperation("rz", (0,), (0.5,)))

    result = cdr(circ, noise=0.01, observable="ZZ", n_training=10)
    print(result.value)           # mitigated expectation
    print(result.r2_score)        # regression quality
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..ir import Circuit, GateOperation
from ..noise import NoiseModel, resolve_noise


@dataclass(frozen=True)
class CDRResult:
    """Result of Clifford Data Regression.

    Attributes:
        value: Mitigated expectation value.
        r2_score: R² of the linear fit (1.0 = perfect).
        n_training_circuits: Number of training circuits used.
        slope: Regression slope.
        intercept: Regression intercept.
    """

    value: float
    r2_score: float
    n_training_circuits: int
    slope: float = 1.0
    intercept: float = 0.0


def cdr(
    circuit: Circuit,
    noise: NoiseModel | float | None = None,
    observable: str = "Z" * 99,
    n_training: int = 10,
    backend: str = "native",
    shots: int = 4096,
    seed: int | None = None,
) -> CDRResult:
    """Run Clifford Data Regression on a circuit.

    Args:
        circuit: Target circuit to mitigate.
        noise: Noise model (required for simulator backends).
        observable: Pauli string for expectation value (e.g. ``"ZZ"``).
        n_training: Number of near-Clifford training circuits.
        backend: Backend name (``"native"`` or ``"qiskit"``).
        shots: Shots per circuit.
        seed: Random seed for reproducibility.

    Returns:
        A :class:`CDRResult` with the mitigated value and fit quality.
    """
    nm = resolve_noise(noise)
    if not nm.enabled and backend == "native":
        raise ValueError("CDR requires noise. Pass noise=NoiseModel(...) or noise=float.")

    n_qubits = circuit.num_qubits
    obs = _validate_observable(observable, n_qubits)

    rng = np.random.default_rng(seed)

    # Generate near-Clifford training circuits
    training_circuits = [
        _make_near_clifford(circuit, rng) for _ in range(n_training)
    ]

    # Compute ideal expectations (statevector, no noise)
    ideal_values = np.array([
        _expectation(c, obs, backend="native", noise=None, shots=shots)
        for c in training_circuits
    ])

    # Compute noisy expectations
    noisy_values = np.array([
        _expectation(c, obs, backend=backend, noise=nm, shots=shots)
        for c in training_circuits
    ])

    # Fit linear model: ideal = slope * noisy + intercept
    if len(noisy_values) < 2:
        slope, intercept = 1.0, 0.0
        r2 = 0.0
    else:
        A = np.vstack([noisy_values, np.ones(len(noisy_values))]).T
        result = np.linalg.lstsq(A, ideal_values, rcond=None)
        slope, intercept = result[0]
        ss_res = np.sum((ideal_values - (slope * noisy_values + intercept)) ** 2)
        ss_tot = np.sum((ideal_values - np.mean(ideal_values)) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Apply to target circuit
    target_noisy = _expectation(circuit, obs, backend=backend, noise=nm, shots=shots)
    mitigated = slope * target_noisy + intercept

    return CDRResult(
        value=float(mitigated),
        r2_score=float(r2),
        n_training_circuits=n_training,
        slope=float(slope),
        intercept=float(intercept),
    )


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

_CLIFFORD_SINGLE = ["h", "s", "x", "y", "z"]
_CLIFFORD_DOUBLE = ["cx", "cz"]


def _make_near_clifford(original: Circuit, rng: np.random.Generator) -> Circuit:
    """Create a near-Clifford version of a circuit.

    Replaces non-Clifford rotation gates (rx, ry, rz, p) with random
    Clifford rotations, preserving the circuit structure.
    """
    new = Circuit()
    for op in original.ops:
        if not isinstance(op, GateOperation):
            new.add(op)
            continue
        name = op.name.lower()
        if name in ("rx", "ry", "rz", "p"):
            # Replace with a random Clifford single-qubit gate
            cliff = rng.choice(_CLIFFORD_SINGLE)
            new.add(GateOperation(cliff, op.qubits))
        elif name in _CLIFFORD_DOUBLE or name in _CLIFFORD_SINGLE:
            new.add(op)
        else:
            # Unknown gate — keep as-is
            new.add(op)
    return new


def _expectation(
    circuit: Circuit,
    observable: str,
    backend: str = "native",
    noise: NoiseModel | None = None,
    shots: int = 4096,
) -> float:
    """Compute <observable> for a circuit."""
    from ..backends import get_backend

    be = get_backend(backend)
    result = be.run(circuit, shots=shots, noise=noise)
    if result.counts is None:
        return 0.0
    return _expectation_from_counts(result.counts, observable, shots)


def _expectation_from_counts(
    counts: dict[str, int],
    observable: str,
    shots: int,
) -> float:
    """Estimate <O> from measurement counts."""
    exp = 0.0
    for bitstring, count in counts.items():
        bits = bitstring.replace(" ", "")
        sign = 1.0
        for i, p in enumerate(observable):
            if p == "Z" and i < len(bits) and bits[-(i + 1)] == "1":
                sign *= -1.0
        exp += sign * count
    return exp / shots


def _validate_observable(observable: str, n_qubits: int) -> str:
    obs = observable.upper()
    valid = set("IXYZ")
    if not all(c in valid for c in obs):
        raise ValueError(f"Invalid Pauli character in '{observable}'. Use I, X, Y, Z.")
    if len(obs) < n_qubits:
        obs = "I" * (n_qubits - len(obs)) + obs
    return obs
