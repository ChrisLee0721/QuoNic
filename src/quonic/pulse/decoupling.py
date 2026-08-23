"""Decoupling sequences for noise suppression.

Example::

    from quonic.pulse import cpmg_sequence, xy4_sequence
    sequence = cpmg_sequence(n_pulses=4, delay=100)
"""

from __future__ import annotations

import numpy as np


def cpmg_sequence(n_pulses: int, delay: float) -> list[float]:
    """CPMG (Carr-Purcell-Meiboom-Gill) decoupling sequence.

    Args:
        n_pulses: number of refocusing pulses
        delay: delay between pulses (ns)

    Returns:
        List of pulse times.
    """
    times = []
    for i in range(n_pulses):
        times.append(delay * (i + 0.5))
    return times


def xy4_sequence(delay: float) -> list[float]:
    """XY-4 decoupling sequence.

    Args:
        delay: delay between pulses (ns)

    Returns:
        List of pulse times.
    """
    return [delay * 0.25, delay * 0.75, delay * 1.25, delay * 1.75]


def uhrig_sequence(n_pulses: int, total_time: float) -> list[float]:
    """Uhrig dynamic decoupling sequence.

    Args:
        n_pulses: number of pulses
        total_time: total sequence time (ns)

    Returns:
        List of pulse times.
    """
    times = []
    for j in range(1, n_pulses + 1):
        t = total_time * np.sin(np.pi * j / (2 * n_pulses + 2))**2
        times.append(t)
    return times
