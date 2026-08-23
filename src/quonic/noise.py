"""Depolarizing noise model.

Usage:
    from quonic import qshow, depolarizing, NoiseModel

    qshow(backend="qiskit", shots=1024, noise=0.05)          # 5% depolarizing per gate
    qshow(noise=depolarizing(0.05))                          # equivalent
    qshow(noise=NoiseModel(single=0.01, double=0.05))        # single/two-qubit gates separately
"""

from __future__ import annotations

from dataclasses import dataclass

from ._i18n import tr


@dataclass(frozen=True)
class NoiseModel:
    """Depolarizing + readout noise model.

    Parameters:
        single: depolarizing probability p applied after each single-qubit gate.
        double: depolarizing probability p applied after each two-qubit gate.
        readout: probability p that a measured bit is misread (flipped) at the end.
    """

    single: float = 0.0
    double: float = 0.0
    readout: float = 0.0

    def __post_init__(self) -> None:
        for name in ("single", "double", "readout"):
            p = getattr(self, name)
            if not 0.0 <= p <= 1.0:
                raise ValueError(tr("err.noise_prob", name=name, p=p))

    @property
    def enabled(self) -> bool:
        return self.single > 0.0 or self.double > 0.0 or self.readout > 0.0


def depolarizing(p: float) -> NoiseModel:
    """Construct a noise model with depolarizing probability p for both single- and two-qubit gates."""
    return NoiseModel(single=float(p), double=float(p))


def resolve_noise(noise: NoiseModel | float | None) -> NoiseModel:
    """Normalize the noise parameter into a NoiseModel (None means no noise)."""
    if noise is None:
        return NoiseModel()
    if isinstance(noise, NoiseModel):
        return noise
    if isinstance(noise, (int, float)):
        return depolarizing(noise)
    raise TypeError(tr("err.noise_arg"))
