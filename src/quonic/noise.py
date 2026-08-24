"""Noise models: depolarizing, amplitude damping, phase damping, thermal relaxation.

Usage:
    from quonic import qshow, depolarizing, NoiseModel

    qshow(backend="qiskit", shots=1024, noise=0.05)          # 5% depolarizing per gate
    qshow(noise=depolarizing(0.05))                          # equivalent
    qshow(noise=NoiseModel(single=0.01, double=0.05))        # single/two-qubit gates separately
    qshow(noise=amplitude_damping(0.01))                     # T1 decay
    qshow(noise=phase_damping(0.01))                         # T2 dephasing
    qshow(noise=thermal_relaxation(t1=50e-6, t2=70e-6, gate_time=20e-9))  # physical T1/T2
"""

from __future__ import annotations

from dataclasses import dataclass

from ._i18n import tr


@dataclass(frozen=True)
class NoiseModel:
    """Configurable noise model supporting multiple channels.

    Parameters:
        single: depolarizing probability p applied after each single-qubit gate.
        double: depolarizing probability p applied after each two-qubit gate.
        readout: probability p that a measured bit is misread (flipped) at the end.
        amplitude_damping: probability gamma of |1> -> |0> decay (T1 process).
        phase_damping: probability of pure dephasing without energy loss (T2 process).
        t1: T1 relaxation time in seconds (used with thermal_relaxation factory).
        t2: T2 dephasing time in seconds (used with thermal_relaxation factory).
        gate_time: gate duration in seconds (used with thermal_relaxation factory).
    """

    single: float = 0.0
    double: float = 0.0
    readout: float = 0.0
    amplitude_damping: float = 0.0
    phase_damping: float = 0.0
    t1: float = 0.0
    t2: float = 0.0
    gate_time: float = 0.0

    def __post_init__(self) -> None:
        for name in ("single", "double", "readout", "amplitude_damping", "phase_damping"):
            p = getattr(self, name)
            if not 0.0 <= p <= 1.0:
                raise ValueError(tr("err.noise_prob", name=name, p=p))
        if self.t1 < 0.0:
            raise ValueError(tr("err.noise_prob", name="t1", p=self.t1))
        if self.t2 < 0.0:
            raise ValueError(tr("err.noise_prob", name="t2", p=self.t2))
        if self.gate_time < 0.0:
            raise ValueError(tr("err.noise_prob", name="gate_time", p=self.gate_time))
        if self.t2 > 2.0 * self.t1 > 0.0:
            raise ValueError(tr("err.noise_t2_t1"))

    @property
    def enabled(self) -> bool:
        return (
            self.single > 0.0
            or self.double > 0.0
            or self.readout > 0.0
            or self.amplitude_damping > 0.0
            or self.phase_damping > 0.0
            or self._has_thermal
        )

    @property
    def _has_thermal(self) -> bool:
        return self.t1 > 0.0 and self.t2 > 0.0 and self.gate_time > 0.0


def depolarizing(p: float) -> NoiseModel:
    """Construct a noise model with depolarizing probability p for both single- and two-qubit gates."""
    return NoiseModel(single=float(p), double=float(p))


def amplitude_damping(gamma: float) -> NoiseModel:
    """Construct a noise model with amplitude damping (T1 decay) probability gamma.

    Models energy relaxation: |1> -> |0> with probability gamma.
    gamma = 1 - exp(-gate_time / T1).
    """
    return NoiseModel(amplitude_damping=float(gamma))


def phase_damping(gamma: float) -> NoiseModel:
    """Construct a noise model with phase damping (pure dephasing) probability gamma.

    Models loss of quantum coherence without energy loss.
    gamma = 1 - exp(-gate_time / T2_star).
    """
    return NoiseModel(phase_damping=float(gamma))


def thermal_relaxation(
    t1: float, t2: float, gate_time: float
) -> NoiseModel:
    """Construct a noise model from physical T1, T2, and gate time.

    Computes amplitude damping and phase damping probabilities from:
        gamma_amp = 1 - exp(-gate_time / T1)
        gamma_phase = 1 - exp(-gate_time / T2) - gamma_amp / 2

    Constraint: T2 <= 2 * T1.

    Parameters:
        t1: T1 relaxation time in seconds.
        t2: T2 dephasing time in seconds (must be <= 2 * T1).
        gate_time: gate duration in seconds.
    """
    import math

    if t1 <= 0.0:
        raise ValueError(tr("err.noise_prob", name="t1", p=t1))
    if t2 <= 0.0:
        raise ValueError(tr("err.noise_prob", name="t2", p=t2))
    if gate_time <= 0.0:
        raise ValueError(tr("err.noise_prob", name="gate_time", p=gate_time))
    if t2 > 2.0 * t1:
        raise ValueError(tr("err.noise_t2_t1"))

    gamma_amp = 1.0 - math.exp(-gate_time / t1)
    gamma_phase = 1.0 - math.exp(-gate_time / t2) - gamma_amp / 2.0
    gamma_phase = max(0.0, gamma_phase)

    return NoiseModel(amplitude_damping=gamma_amp, phase_damping=gamma_phase)


def resolve_noise(noise: NoiseModel | float | None) -> NoiseModel:
    """Normalize the noise parameter into a NoiseModel (None means no noise)."""
    if noise is None:
        return NoiseModel()
    if isinstance(noise, NoiseModel):
        return noise
    if isinstance(noise, (int, float)):
        return depolarizing(noise)
    raise TypeError(tr("err.noise_arg"))
