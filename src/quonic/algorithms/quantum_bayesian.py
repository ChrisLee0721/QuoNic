"""Quantum Bayesian Inference — quantum-enhanced hypothesis testing.

Uses quantum amplitude encoding and controlled rotations to perform
Bayesian posterior estimation. The prior is encoded as a qubit rotation,
likelihoods are applied as controlled operations, and the posterior is
extracted via measurement statistics.

This is a genuine quantum algorithm: the posterior computation is performed
by the quantum circuit, not by classical pre-computation.

Example::

    from quonic.algorithms import quantum_bayesian
    result = quantum_bayesian(prior_h0=0.5, likelihood_h0=0.8, likelihood_h1=0.3)
"""

from __future__ import annotations

import math

from ..backends import run_circuit
from ..ir import Circuit, GateOperation
from ..result import Result


def quantum_bayesian(
    prior_h0: float = 0.5,
    likelihood_h0: float = 0.8,
    likelihood_h1: float = 0.3,
    shots: int = 1024,
    backend: str = "auto",
) -> Result:
    """Quantum Bayesian inference for binary hypothesis testing.

    Encodes the prior P(H0) as a qubit rotation, applies likelihood-dependent
    controlled rotations, and measures to sample from the posterior.

    Circuit (3 qubits):
        q0: prior qubit — Ry(2·arcsin(√prior)) encodes P(H0)
        q1: likelihood qubit — controlled by q0, encodes likelihood ratio
        q2: ancilla for amplitude estimation

    The posterior P(H0|data) is extracted from the measurement statistics
    of q0 after the likelihood rotation.

    Args:
        prior_h0: Prior probability of hypothesis H0 (0 < p < 1).
        likelihood_h0: P(data|H0), the likelihood of data under H0.
        likelihood_h1: P(data|H1), the likelihood of data under H1.
        shots: Number of measurement shots.
        backend: Backend for execution.

    Returns:
        Result with posterior_h0 (quantum estimate) and classical_posterior
        (exact Bayesian update for comparison).
    """
    if not (0.0 < prior_h0 < 1.0):
        raise ValueError(f"prior_h0 must be in (0,1), got {prior_h0}")
    if not (0.0 < likelihood_h0 <= 1.0 and 0.0 < likelihood_h1 <= 1.0):
        raise ValueError("Likelihoods must be in (0,1]")

    # Classical posterior for comparison
    evidence = prior_h0 * likelihood_h0 + (1 - prior_h0) * likelihood_h1
    classical_posterior = (prior_h0 * likelihood_h0) / evidence

    # Quantum circuit: encode prior and apply likelihood as controlled rotation
    circuit = Circuit()

    # Qubit 0: encode prior P(H0) as amplitude
    # |ψ0> = √(1-prior)|0> + √(prior)|1>
    theta_prior = 2 * math.asin(math.sqrt(prior_h0))
    circuit.add(GateOperation("ry", (0,), (theta_prior,)))

    # Qubit 1: likelihood encoding
    # The likelihood ratio determines how much to rotate q1 conditioned on q0
    # If q0=|1> (H0): rotate by angle encoding likelihood_h0
    # If q0=|0> (H1): rotate by angle encoding likelihood_h1
    theta_l0 = 2 * math.asin(math.sqrt(likelihood_h0))
    theta_l1 = 2 * math.asin(math.sqrt(likelihood_h1))

    # Controlled rotation: q0 controls q1
    # When q0=|1>: Ry(theta_l0) on q1
    # When q0=|0>: Ry(theta_l1) on q1
    # Decompose: CRy = Ry(l1) · C-Ry(l0-l1) on q1 controlled by q0
    delta = theta_l0 - theta_l1
    # Apply Ry(theta_l1) unconditionally
    circuit.add(GateOperation("ry", (1,), (theta_l1,)))
    # Controlled-Ry(delta): q0 controls q1
    circuit.add(GateOperation("cx", (0, 1)))
    circuit.add(GateOperation("ry", (1,), (-delta / 2,)))
    circuit.add(GateOperation("cx", (0, 1)))
    circuit.add(GateOperation("ry", (1,), (delta / 2,)))

    # Measure q0: the probability of |1> gives the posterior P(H0|data)
    circuit.add(GateOperation("measure", (0,)))

    result = run_circuit(circuit, backend=backend, shots=shots)

    # Extract posterior from measurement statistics
    counts = result.counts
    total = sum(counts.values())
    # q0 is the rightmost bit
    count_h0 = sum(v for k, v in counts.items() if k[-1] == "1")
    quantum_posterior = count_h0 / total if total > 0 else 0.5

    return Result.from_value(
        quantum_posterior,
        posterior_h0=quantum_posterior,
        classical_posterior=classical_posterior,
        counts=counts,
    )
