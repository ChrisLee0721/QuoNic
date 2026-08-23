"""Readout (measurement) error mitigation via confusion-matrix inversion.

Real hardware misreads a measured |0> as |1> (and vice versa) with some
probability — the *readout* (or measurement/assignment) error. This module
calibrates that error and corrects it:

1. ``calibrate`` prepares each qubit in |0> and |1>, measures it, and builds a
   per-qubit confusion matrix ``A_q[i][j] = P(read j | true i)``.
2. ``ReadoutCalibration.apply`` inverts the full tensor-product matrix
   ``A = ⊗ A_q`` and maps the measured counts back to the (approximately)
   noise-free counts: ``p_true = A⁻¹ · p_meas``.

The per-qubit (tensor-product) model assumes readout errors are uncorrelated
across qubits — the standard, scalable choice (2n calibration circuits instead
of 2ⁿ). Inverting the full 2ⁿ × 2ⁿ matrix makes ``apply`` exponential in the
qubit count, so it is intended for small circuits (≲ 10 qubits).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ._i18n import tr
from .ir import Circuit, GateOperation
from .noise import NoiseModel

if TYPE_CHECKING:
    import numpy as np


@dataclass
class ReadoutCalibration:
    """Readout confusion matrices ``A[i][j] = P(read j | true i)``.

    Two representations are supported:

    - per-qubit (``matrices``, shape ``(n, 2, 2)``): the standard tensor-product
      model ``A = A₀ ⊗ A₁ ⊗ … ⊗ Aₙ₋₁``, calibrated from 2n circuits.
    - correlated (``full``, shape ``(2ⁿ, 2ⁿ)``): the complete confusion matrix,
      calibrated from 2ⁿ circuits. Captures readout crosstalk between qubits that
      the tensor-product model cannot represent.
    """

    matrices: np.ndarray  # shape (n, 2, 2); per-qubit marginals (always populated)
    n: int
    full: np.ndarray | None = None  # shape (2**n, 2**n); correlated model

    @property
    def matrix(self) -> np.ndarray:
        """The full 2ⁿ × 2ⁿ confusion matrix (LSB = qubit 0).

        Returns the correlated ``full`` matrix when present, otherwise the
        tensor product ``A = A₀ ⊗ A₁ ⊗ … ⊗ Aₙ₋₁`` of the per-qubit matrices.
        """
        import numpy as np

        if self.full is not None:
            return self.full
        a = self.matrices[self.n - 1].copy()
        for q in range(self.n - 2, -1, -1):
            a = np.kron(a, self.matrices[q])
        return a

    def apply(self, counts: dict[str, int], shots: int) -> dict[str, int]:
        """Correct a measured counts histogram by inverting the confusion matrix.

        Returns integer counts (rounded, so the total may differ slightly from
        ``shots``). Negative entries from inversion noise are clipped to zero and
        the result renormalized to a probability distribution.
        """
        import numpy as np

        if self.n > 20:
            import warnings
            warnings.warn(
                f"Readout calibration with {self.n} qubits requires "
                f"{2**self.n * 2**self.n * 8 / 2**30:.1f}GB for the confusion matrix. "
                f"Consider using fewer qubits or per-qubit mode.",
                stacklevel=2,
            )

        p_meas = np.zeros(2 ** self.n)
        for bs, cnt in counts.items():
            p_meas[int(bs, 2)] += cnt / shots

        # A[i,j] = P(read j | true i)  =>  p_meas = Aᵀ · p_true, so solve for p_true.
        try:
            p_true = np.linalg.solve(self.matrix.T, p_meas)
        except np.linalg.LinAlgError:
            # Near-singular matrix — fall back to pseudo-inverse with regularization
            lam = 1e-6  # Tikhonov regularization
            reg = self.matrix.T @ self.matrix + lam * np.eye(2 ** self.n)
            p_true = np.linalg.solve(reg, self.matrix.T @ p_meas)

        p_true = np.clip(p_true, 0.0, None)
        total = float(p_true.sum())
        if total > 0.0:
            p_true = p_true / total

        fmt = f"0{self.n}b"
        corrected: dict[str, int] = {}
        for i in range(2 ** self.n):
            c = round(p_true[i] * shots)
            if c:
                corrected[format(i, fmt)] = c
        return corrected


def _marginals_from_full(full: np.ndarray, n: int) -> np.ndarray:
    """Reduce a full confusion matrix to per-qubit marginals (other qubits in |0>).

    ``matrices[q, i, j] = P(read_q = j | true_q = i, other qubits |0>)``, the same
    definition the per-qubit calibration measures directly.
    """
    import numpy as np

    matrices = np.zeros((n, 2, 2))
    for q in range(n):
        for i in (0, 1):
            for j in (0, 1):
                total = 0.0
                for jj in range(2 ** n):
                    if ((jj >> q) & 1) == j:
                        total += full[i << q, jj]
                matrices[q, i, j] = total
    return matrices


def calibrate(
    num_qubits: int,
    backend: str = "native",
    shots: int = 1024,
    noise: NoiseModel | float | None = None,
    device: str | None = None,
    correlated: bool = False,
) -> ReadoutCalibration:
    """Build a readout calibration by preparing computational-basis states.

    By default (``correlated=False``) runs 2·``num_qubits`` circuits — each
    prepares one qubit in |0> and |1> and measures all — and estimates per-qubit
    confusion matrices (tensor-product model).

    With ``correlated=True`` it runs all 2ⁿ circuits (every computational basis
    state), building the full 2ⁿ × 2ⁿ confusion matrix to capture readout
    crosstalk between qubits. This scales exponentially, so it is limited to
    ``num_qubits <= 12``.

    Parameters:
        num_qubits: number of qubits to calibrate.
        backend: "native", "qiskit", or "qi". For simulator backends the readout
            error must be injected via ``noise`` (e.g. ``NoiseModel(readout=0.05)``);
            for backend="qi" real hardware has intrinsic readout error, so
            ``noise`` must be None.
        shots: samples per calibration circuit.
        noise: readout/depolarizing noise for simulator backends (None for qi).
        device: hardware device for backend="qi" (e.g. "tuna17"); ignored otherwise.
        correlated: build the full 2ⁿ × 2ⁿ confusion matrix instead of per-qubit.
    """
    import numpy as np

    if not isinstance(num_qubits, int) or num_qubits <= 0:
        raise ValueError(tr("err.readout_n", n=num_qubits))

    from .backends import get_backend

    be = get_backend(backend, device=device)

    if correlated:
        if num_qubits > 12:
            raise ValueError(tr("err.readout_correlated_n", max_n=12, n=num_qubits))
        full = np.zeros((2 ** num_qubits, 2 ** num_qubits))
        for i in range(2 ** num_qubits):
            c = Circuit()
            c.allocate(num_qubits)
            for q in range(num_qubits):
                if (i >> q) & 1:
                    c.add(GateOperation("x", (q,)))
            for qq in range(num_qubits):
                c.add(GateOperation("measure", (qq,)))
            result = be.run(c, shots=shots, noise=noise)
            for bs, cnt in (result.counts or {}).items():
                full[i, int(bs, 2)] += cnt / shots
        matrices = _marginals_from_full(full, num_qubits)
        return ReadoutCalibration(matrices, num_qubits, full)

    matrices = np.zeros((num_qubits, 2, 2))
    for q in range(num_qubits):
        for prep in (0, 1):
            c = Circuit()
            c.allocate(num_qubits)
            if prep == 1:
                c.add(GateOperation("x", (q,)))
            for qq in range(num_qubits):
                c.add(GateOperation("measure", (qq,)))
            result = be.run(c, shots=shots, noise=noise)
            for bs, cnt in (result.counts or {}).items():
                m = int(bs[num_qubits - 1 - q])
                matrices[q, prep, m] += cnt / shots
    return ReadoutCalibration(matrices, num_qubits)


__all__ = ["ReadoutCalibration", "calibrate"]
