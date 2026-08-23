"""Calibration routines for quantum hardware.

Example::

    from quonic.pulse import rabi_calibration, t1_calibration

    result = rabi_calibration(qubit=0, amplitudes=[0.1, 0.2, 0.3, 0.4, 0.5])
    print(f"Pi pulse amplitude: {result.pi_amplitude:.3f}")
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RabiResult:
    """Result of a Rabi oscillation calibration."""

    amplitudes: list[float]
    populations: list[float]
    pi_amplitude: float
    frequency: float


@dataclass
class T1Result:
    """Result of a T1 relaxation measurement."""

    delays: list[float]
    populations: list[float]
    t1: float


@dataclass
class T2Result:
    """Result of a T2 dephasing measurement."""

    delays: list[float]
    populations: list[float]
    t2: float


def rabi_calibration(
    qubit: int,
    amplitudes: list[float],
    backend: str = "native",
    shots: int = 1024,
) -> RabiResult:
    """Run a Rabi oscillation experiment.

    Measures the population of |1> as a function of pulse amplitude.

    Args:
        qubit: qubit to calibrate
        amplitudes: list of pulse amplitudes to sweep
        backend: simulation backend
        shots: number of shots per amplitude

    Returns:
        RabiResult with pi_amplitude and frequency.
    """
    from .. import qgate, reset
    from ..backends import get_backend
    from ..gates import Ry
    from ..stack import current_circuit

    populations = []
    for amp in amplitudes:
        reset()
        qgate(Ry(float(amp)), 0)
        result = get_backend(backend).run(current_circuit(), shots=shots)
        p1 = result.counts.get("1", 0) / shots
        populations.append(p1)

    # Find pi amplitude (where population is maximum)
    pi_amp = amplitudes[np.argmax(populations)]

    return RabiResult(
        amplitudes=amplitudes,
        populations=populations,
        pi_amplitude=pi_amp,
        frequency=1.0,  # placeholder
    )


def t1_calibration(
    qubit: int,
    delays: list[float],
    backend: str = "native",
    shots: int = 1024,
) -> T1Result:
    """Run a T1 relaxation experiment.

    Measures the population of |1> after waiting for different delays.

    Args:
        qubit: qubit to calibrate
        delays: list of delays in ns
        backend: simulation backend
        shots: number of shots per delay

    Returns:
        T1Result with T1 time.
    """
    from .. import qgate, reset
    from ..backends import get_backend
    from ..gates import X
    from ..stack import current_circuit

    populations = []
    for delay in delays:
        reset()
        qgate(X, 0)  # prepare |1>
        # Delay is simulated by identity gates (placeholder)
        result = get_backend(backend).run(current_circuit(), shots=shots)
        p1 = result.counts.get("1", 0) / shots
        populations.append(p1)

    # Fit exponential decay
    t1 = delays[-1]  # placeholder

    return T1Result(
        delays=delays,
        populations=populations,
        t1=t1,
    )


def t2_calibration(
    qubit: int,
    delays: list[float],
    backend: str = "native",
    shots: int = 1024,
) -> T2Result:
    """Run a T2 dephasing experiment.

    Args:
        qubit: qubit to calibrate
        delays: list of delays in ns
        backend: simulation backend
        shots: number of shots per delay

    Returns:
        T2Result with T2 time.
    """
    from .. import qgate, reset
    from ..backends import get_backend
    from ..gates import H
    from ..stack import current_circuit

    populations = []
    for delay in delays:
        reset()
        qgate(H, 0)
        # Delay is simulated by identity gates (placeholder)
        qgate(H, 0)
        result = get_backend(backend).run(current_circuit(), shots=shots)
        p0 = result.counts.get("0", 0) / shots
        populations.append(p0)

    t2 = delays[-1]  # placeholder

    return T2Result(
        delays=delays,
        populations=populations,
        t2=t2,
    )
