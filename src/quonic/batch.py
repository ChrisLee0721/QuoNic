"""Batch execution — run multiple circuits efficiently.

Example::

    from quonic.batch import run_batch
    results = run_batch([circuit1, circuit2], backend="qulacs", shots=1024)
"""

from __future__ import annotations

from .ir import Circuit
from .noise import NoiseModel
from .result import Result


def run_batch(
    circuits: list[Circuit],
    backend: str = "native",
    shots: int = 1024,
    noise: NoiseModel | float | None = None,
) -> list[Result]:
    """Run multiple circuits on the same backend.

    Args:
        circuits: list of circuits to run
        backend: backend name
        shots: shots per circuit
        noise: optional noise model

    Returns:
        List of Result objects, one per circuit.
    """
    from .backends import get_backend

    be = get_backend(backend)
    return [be.run(c, shots=shots, noise=noise) for c in circuits]
