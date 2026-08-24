"""Probabilistic Error Cancellation (PEC) — quasi-probability decomposition.

Estimates the ideal expectation value by decomposing each noisy gate into a
quasi-probability distribution over inverse channels and sampling.

Example::

    from quonic.ir import Circuit, GateOperation
    from quonic.mitigation import pec

    circ = Circuit()
    circ.add(GateOperation("x", (0,)))

    result = pec(circ, noise=0.01, observable="Z", n_samples=1000)
    print(result.value)       # mitigated expectation
    print(result.variance)    # estimator variance
    print(result.overhead)    # sampling overhead (gamma^2)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..ir import Circuit, GateOperation
from ..noise import NoiseModel, resolve_noise
from ..simulator import StatevectorSimulator


@dataclass(frozen=True)
class PECResult:
    """Result of Probabilistic Error Cancellation.

    Attributes:
        value: Mitigated expectation value.
        variance: Estimator variance.
        n_samples: Number of Monte Carlo samples used.
        overhead: Sampling overhead (gamma^2).  Higher = more shots needed.
    """

    value: float
    variance: float
    n_samples: int
    overhead: float


def pec(
    circuit: Circuit,
    noise: NoiseModel | float | None = None,
    observable: str = "Z" * 99,
    n_samples: int = 1000,
    backend: str = "native",
    shots: int = 1,
    seed: int | None = None,
) -> PECResult:
    """Run Probabilistic Error Cancellation on a circuit.

    Args:
        circuit: Target circuit to mitigate.
        noise: Noise model (required).
        observable: Pauli string for expectation value.
        n_samples: Number of Monte Carlo samples.
        backend: Backend name (``"native"`` only for now).
        shots: Shots per sample (1 = exact statevector).
        seed: Random seed.

    Returns:
        A :class:`PECResult` with the mitigated value and variance.
    """
    nm = resolve_noise(noise)
    if not nm.enabled:
        raise ValueError("PEC requires noise. Pass noise=NoiseModel(...) or noise=float.")

    n_qubits = circuit.num_qubits
    obs = _validate_observable(observable, n_qubits)
    rng = np.random.default_rng(seed)

    # Build quasi-probability decomposition for each gate
    # For depolarizing noise, the inverse channel has a known form
    gamma = _compute_gamma(nm)
    overhead = gamma ** (2 * _count_gates(circuit))

    # Monte Carlo sampling
    estimates = np.zeros(n_samples)
    for i in range(n_samples):
        # Sample a sign-corrected circuit
        sampled_circuit, sign = _sample_inverse_circuit(circuit, nm, rng)
        # Compute expectation on the ideal (noiseless) simulator
        exp_val = _ideal_expectation(sampled_circuit, obs)
        estimates[i] = sign * exp_val

    value = float(np.mean(estimates))
    variance = float(np.var(estimates) / n_samples)

    return PECResult(
        value=value,
        variance=variance,
        n_samples=n_samples,
        overhead=float(overhead),
    )


# ------------------------------------------------------------------
# Quasi-probability decomposition helpers
# ------------------------------------------------------------------

def _compute_gamma(noise: NoiseModel) -> float:
    """Compute the gamma factor for depolarizing noise.

    For a depolarizing channel with parameter p:
        gamma = (1 + 3p/4) / (1 - p)  for single-qubit gates
        gamma = (1 + 15p/4) / (1 - p) for two-qubit gates (approx)

    We use the single-qubit gamma as the dominant cost.
    """
    p = max(noise.single, noise.double, 1e-10)
    if p >= 1.0:
        return float("inf")
    # Simplified: gamma = 1/(1-p) for depolarizing
    return 1.0 / (1.0 - p)


def _count_gates(circuit: Circuit) -> int:
    return sum(1 for op in circuit.ops if isinstance(op, GateOperation))


def _sample_inverse_circuit(
    circuit: Circuit,
    noise: NoiseModel,
    rng: np.random.Generator,
) -> tuple[Circuit, float]:
    """Sample a sign-corrected inverse circuit for PEC.

    For each gate, with probability p_i apply the inverse (correcting) channel
    with the appropriate sign.  This is a simplified version that works with
    the depolarizing noise model.
    """
    new = Circuit()
    total_sign = 1.0

    for op in circuit.ops:
        if not isinstance(op, GateOperation):
            new.add(op)
            continue

        p = noise.double if len(op.qubits) > 1 else noise.single
        if p < 1e-12:
            new.add(op)
            continue

        # Quasi-probability: {ideal: (1+3p/4)/(1-p), X-error: -p/(4(1-p)), ...}
        # Simplified: with prob (1-p) apply identity (correct), with prob p apply correction
        gamma = 1.0 / (1.0 - p)
        q_ideal = gamma * (1 - p + 3 * p / 4)  # weight for "do nothing"
        q_fix = -gamma * p / 4  # weight for applying correction

        # Sample: choose correction with probability |q_fix| / (|q_ideal| + |q_fix|)
        p_fix = abs(q_fix) / (abs(q_ideal) + abs(q_fix))
        if rng.random() < p_fix:
            # Apply a random Pauli correction
            pauli = rng.choice(["x", "y", "z"])
            new.add(GateOperation(pauli, op.qubits))
            total_sign *= np.sign(q_fix)
        else:
            total_sign *= np.sign(q_ideal)

        new.add(op)

    return new, total_sign


def _ideal_expectation(circuit: Circuit, observable: str) -> float:
    """Compute <observable> on a noiseless statevector simulator."""
    sim = StatevectorSimulator(circuit.num_qubits)
    for op in circuit.ops:
        if isinstance(op, GateOperation):
            sim.apply(op.name, op.qubits, op.params)
    return sim.expectation(observable)


def _validate_observable(observable: str, n_qubits: int) -> str:
    obs = observable.upper()
    valid = set("IXYZ")
    if not all(c in valid for c in obs):
        raise ValueError(f"Invalid Pauli character in '{observable}'.")
    if len(obs) < n_qubits:
        obs = "I" * (n_qubits - len(obs)) + obs
    return obs
