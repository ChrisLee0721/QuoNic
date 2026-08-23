"""Quantum control theory — pulse-level control, calibration, and optimization.

Example::

    from quonic.pulse import GaussianPulse, DragPulse
    from quonic.pulse import rabi_calibration, t1_calibration

    pulse = GaussianPulse(duration=20, sigma=5, amplitude=0.5)
    result = rabi_calibration(qubit=0, amplitudes=[0.1, 0.2, 0.3, 0.4, 0.5])
"""

from .calibration import rabi_calibration, t1_calibration, t2_calibration
from .decoupling import cpmg_sequence, xy4_sequence
from .optimal import GRAPEResult, grape_optimize, krotov_optimize
from .pulse import CrossResonancePulse, DragPulse, GaussianPulse

__all__ = [
    "CrossResonancePulse",
    "DragPulse",
    "GRAPEResult",
    "GaussianPulse",
    "cpmg_sequence",
    "grape_optimize",
    "krotov_optimize",
    "rabi_calibration",
    "t1_calibration",
    "t2_calibration",
    "xy4_sequence",
]
