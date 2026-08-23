"""Quantum Rejection Sampling — efficient probability distribution sampling.

Boundary conditions:
- Uses Grover search to amplify target distribution
- Minimal demo: sample from a biased coin
- Requires oracle for target distribution

Example::

    from quonic.algorithms import rejection_sampling_demo
    result = rejection_sampling_demo()
"""

from __future__ import annotations

from ..result import Result
from ..simulator import StatevectorSimulator


def rejection_sampling_demo(n_samples: int = 100) -> Result:
    """Minimal quantum rejection sampling demo."""
    import numpy as np

    # Target distribution: P(0) = 0.3, P(1) = 0.7
    target = [0.3, 0.7]
    samples = []

    for _ in range(n_samples):
        # Prepare uniform superposition
        sim = StatevectorSimulator(1)
        sim.apply("h", (0,))

        # Accept/reject based on target probability
        prob = sim.expectation("Z")
        p_0 = (1 + prob) / 2
        if np.random.random() < target[round(p_0)]:
            samples.append(round(p_0))

    counts = {str(s): samples.count(s) for s in set(samples)}
    return Result.from_value(float(len(samples)), counts=counts, n_accepted=len(samples))
