"""Quantum Reinforcement Learning — variational policy for simple environment.

Boundary conditions:
- 2-state environment (toy problem)
- Variational circuit as policy network
- Reward signal used to update parameters
- NOT a production RL agent — demonstrates the concept

Example::

    from quonic.algorithms import qrl
    result = qrl(n_episodes=10)
"""

from __future__ import annotations

import numpy as np

from ..result import Result
from ..simulator import StatevectorSimulator


def qrl(n_episodes: int = 10) -> Result:
    """Minimal quantum RL demo."""
    params = np.array([0.5, 0.5])
    rewards = []

    for _ in range(n_episodes):
        # Policy: Ry(params[0]) on state encoding, measure
        sim = StatevectorSimulator(1)
        sim.apply("ry", (0,), (float(params[0]),))
        prob_1 = 1 - sim.expectation("Z") ** 2  # P(action=1)

        # Environment: reward = 1 if action matches target
        action = 1 if np.random.random() < prob_1 else 0
        reward = 1.0 if action == 1 else 0.0
        rewards.append(reward)

        # Simple parameter update
        params[0] += 0.1 * (reward - 0.5)

    return Result.from_value(float(np.mean(rewards)), rewards=rewards)
