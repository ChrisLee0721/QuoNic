"""numpy matrix construction for gates (shared by the in-house engines).

Conventions:
- qubit 0 is the least-significant bit (rightmost in the bitstring), consistent
  with the three sampling backends.
- single-qubit matrix matrix[out, in].
- multi-qubit gates (cx/ccx/cz/cp/mcz) are implemented in each engine using the
  "diagonal phase + H" trick, so only single-qubit matrices are provided here to
  avoid ambiguity in the index ordering of multi-qubit matrices.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .._i18n import tr

_SQRT_HALF = 1.0 / np.sqrt(2.0)

_I = np.eye(2, dtype=complex)
_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
_Y = np.array([[0.0, -1j], [1j, 0.0]], dtype=complex)
_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
_H = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex) * _SQRT_HALF


def rotation(axis: str, theta: float) -> Any:
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    if axis == "x":
        return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)
    if axis == "y":
        return np.array([[c, -s], [s, c]], dtype=complex)
    if axis == "z":
        return np.array(
            [[np.exp(-1j * theta / 2.0), 0.0], [0.0, np.exp(1j * theta / 2.0)]],
            dtype=complex,
        )
    raise ValueError(tr("err.unknown_axis", axis=axis))


def phase_shift(theta: float) -> Any:
    """Phase gate P(θ) = diag(1, e^{iθ})."""
    return np.array([[1.0, 0.0], [0.0, np.exp(1j * theta)]], dtype=complex)


def single(name: str, params: tuple[float, ...] = ()) -> Any:
    """Return the single-qubit gate matrix; name is the lowercase gate name."""
    name = name.lower()
    if name == "i":
        return _I
    if name == "h":
        return _H
    if name == "x":
        return _X
    if name == "y":
        return _Y
    if name == "z":
        return _Z
    if name == "rx":
        return rotation("x", params[0])
    if name == "ry":
        return rotation("y", params[0])
    if name == "rz":
        return rotation("z", params[0])
    if name == "p":
        return phase_shift(params[0])
    raise ValueError(tr("err.self_gate", name=name))


# single-qubit Clifford gate set (usable by the stabilizer engine)
SINGLE_GATES: set[str] = {"i", "h", "x", "y", "z", "rx", "ry", "rz", "p"}
