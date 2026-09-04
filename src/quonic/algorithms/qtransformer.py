"""Quantum Transformer — minimal demo of quantum self-attention.

Boundary conditions:
- 2-token self-attention using quantum circuits
- NOT a production transformer — demonstrates the concept
- Uses parameterized circuits for Q, K, V projections

Example::

    from quonic.algorithms import qtransformer
    result = qtransformer()
"""

from __future__ import annotations

from ..result import Result
from ..simulator import StatevectorSimulator


def qtransformer() -> Result:
    """Minimal quantum transformer demo with 2 tokens."""
    import numpy as np

    # 2 tokens, 1-qubit embedding each
    params_q = np.array([0.5, 0.3])
    params_k = np.array([0.7, 0.2])
    params_v = np.array([0.4, 0.6])

    # Q projection
    sim_q = StatevectorSimulator(1)
    sim_q.apply("ry", (0,), (float(params_q[0]),))

    # K projection
    sim_k = StatevectorSimulator(1)
    sim_k.apply("ry", (0,), (float(params_k[0]),))

    # Attention score: <Q|K>
    score = abs(sim_q.state @ sim_k.state.conj()) ** 2

    # V projection
    sim_v = StatevectorSimulator(1)
    sim_v.apply("ry", (0,), (float(params_v[0]),))

    output = score * sim_v.expectation("Z")
    return Result.from_value(float(output), attention_score=float(score))
