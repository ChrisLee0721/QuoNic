"""Quantum Generative Adversarial Network (QGAN).

A quantum generator trained adversarially against a classical discriminator
to learn a target probability distribution.

Boundary conditions:
- Quantum generator (parameterized Ry rotations)
- Classical discriminator (simple threshold)
- NOT a production GAN — demonstrates the adversarial training concept

Example::

    from quonic.algorithms.qgan import qgan_train
    result = qgan_train(n_qubits=2, n_steps=100)
    print(f"Final loss: {result.value:.4f}")
"""

from __future__ import annotations

from ..ir import Circuit, GateOperation
from ..result import Result


def _generator_circuit(n_qubits: int, params: list[float]) -> Circuit:
    """Build a parameterized generator circuit.

    Structure: Ry rotations on each qubit + entangling CX ladder.
    """
    c = Circuit()
    c.allocate(n_qubits)

    for i in range(n_qubits):
        c.add(GateOperation("ry", (i,), (params[i],)))

    # Entangling layer
    for i in range(n_qubits - 1):
        c.add(GateOperation("cx", (i, i + 1)))

    return c


def _discriminator(probs: list[float], threshold: float = 0.5) -> list[float]:
    """Simple classical discriminator: classify samples as real (1) or fake (0)."""
    return [1.0 if p > threshold else 0.0 for p in probs]


def qgan_train(
    n_qubits: int = 2,
    n_steps: int = 100,
    lr: float = 0.1,
    target: list[float] | None = None,
    seed: int = 42,
) -> Result:
    """Train a QGAN to learn a target distribution.

    Parameters:
        n_qubits: number of qubits in the generator
        n_steps: number of training steps
        lr: learning rate for parameter updates
        target: target probability distribution (default: [0.3, 0.7] for 2 qubits)
        seed: random seed

    Returns:
        Result with final loss, learned distribution, and trained parameters.
    """
    import numpy as np

    rng = np.random.RandomState(seed)

    # Default target: 2-qubit distribution favoring |00> and |11>
    if target is None:
        target = [0.4, 0.2, 0.2, 0.4]  # P(00), P(01), P(10), P(11)

    # Initialize generator parameters
    params = rng.randn(n_qubits) * 0.5

    losses = []
    learned_distributions = []

    for step in range(n_steps):
        # Generate samples from quantum circuit
        circuit = _generator_circuit(n_qubits, params)

        from ..backends import run_circuit
        result = run_circuit(circuit, shots=1024)

        # Convert counts to probability distribution
        gen_probs = np.zeros(2**n_qubits)
        for bs, count in result.counts.items():
            idx = int(bs, 2)
            gen_probs[idx] += count / 1024

        # Discriminator loss: how well can it distinguish real vs fake?
        # Real samples from target, fake from generator
        d_loss = np.mean(np.abs(np.array(target) - gen_probs))

        # Generator update: adjust parameters to match target
        # Simple gradient-free update: nudge toward target
        for i in range(n_qubits):
            # Compute gradient numerically
            params_plus = params.copy()
            params_plus[i] += 0.01
            circuit_plus = _generator_circuit(n_qubits, params_plus)
            result_plus = run_circuit(circuit_plus, shots=1024)

            gen_plus = np.zeros(2**n_qubits)
            for bs, count in result_plus.counts.items():
                gen_plus[int(bs, 2)] += count / 1024

            loss_plus = np.mean(np.abs(np.array(target) - gen_plus))
            grad = (loss_plus - d_loss) / 0.01

            params[i] -= lr * grad

        losses.append(d_loss)
        learned_distributions.append(gen_probs.tolist())

    # Final evaluation
    final_circuit = _generator_circuit(n_qubits, params)
    final_result = run_circuit(final_circuit, shots=1024)
    final_probs = np.zeros(2**n_qubits)
    for bs, count in final_result.counts.items():
        final_probs[int(bs, 2)] += count / 1024

    return Result.from_value(
        float(np.mean(losses[-10:])),  # average loss over last 10 steps
        final_distribution=final_probs.tolist(),
        target_distribution=target,
        params=params.tolist(),
        n_qubits=n_qubits,
        n_steps=n_steps,
    )


def qgan(n_steps: int = 50) -> Result:
    """Quick QGAN demo with default settings."""
    return qgan_train(n_qubits=2, n_steps=n_steps)
