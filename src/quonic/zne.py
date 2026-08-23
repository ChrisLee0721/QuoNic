"""Zero-noise extrapolation (ZNE): unitary folding + linear extrapolation.

Backend-independent error mitigation. Global unitary folding C → C(C†C)^k
amplifies the noise scale to λ = 2k+1; running at several λ and linearly
extrapolating back to λ = 0 yields a noise-free estimate.

Two metrics are supported:

- **success probability** (``target``): fold → sample under noise → the hit
  probability of the target bitstring set.
- **expectation value** (``observable``): fold → evolve the noisy density
  matrix → ⟨O⟩ = Tr(ρ·O) for a Pauli string, computed with the in-house
  DensityMatrixEngine.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ._i18n import tr
from .compiler import _adjoint
from .ir import Circuit, GateOperation
from .noise import NoiseModel, resolve_noise

if TYPE_CHECKING:
    from .readout import ReadoutCalibration

_PAULI_CHARS = frozenset("IXYZ")


@dataclass
class ZNEResult:
    """Outcome of a ZNE run: per-λ measured values and the λ=0 extrapolation."""

    factors: tuple[float, ...]
    values: tuple[float, ...]
    extrapolated: float
    metric: str  # "success" or "expectation"


def fold(circuit: Circuit, k: int) -> Circuit:
    """Globally fold the circuit C → C(C†C)^k, amplifying noise by λ = 2k+1.

    Only unitary gates are folded; trailing measurements are preserved at the end.
    Dynamic ops (cmeasure / cif / cwhile) are not foldable and raise ValueError.
    The original circuit is not modified.
    """
    if k < 0:
        raise ValueError(tr("err.zne_fold_k", k=k))

    unitary: list = []
    measure: list = []
    for op in circuit.ops:
        if op.name == "measure":
            measure.append(op)
        elif op.name in ("cmeasure", "cif", "cwhile"):
            raise ValueError(tr("err.zne_fold_unitary", name=op.name))
        else:
            unitary.append(op)

    dag = [_adjoint(op) for op in reversed(unitary)]
    out = Circuit()
    out.allocate(circuit.num_qubits)
    for op in unitary:
        out.add(op)
    for _ in range(k):
        for op in dag:
            out.add(op)
        for op in unitary:
            out.add(op)
    for op in measure:
        out.add(op)
    return out


def _validate_factors(factors: Sequence[float]) -> tuple[int, ...]:
    ks: list = []
    prev: float | None = None
    for lam in factors:
        k = (float(lam) - 1.0) / 2.0
        if k < 0 or k != int(k):
            raise ValueError(tr("err.zne_factors", lam=lam))
        if prev is not None and float(lam) <= prev:
            raise ValueError(tr("err.zne_factors_order", lam=lam, prev=prev))
        ks.append(int(k))
        prev = float(lam)
    return tuple(ks)


def _linear_extrap(factors: Sequence[float], values: Sequence[float]) -> float:
    import numpy as np

    lam = np.asarray(factors, dtype=float)
    y = np.asarray(values, dtype=float)
    intercept = float(np.polyfit(lam, y, 1)[1])
    return intercept


def _exponential_extrap(factors: Sequence[float], values: Sequence[float]) -> float:
    """Fit ``y = a·e^(-bλ) + c`` and extrapolate to λ=0 (``a + c``).

    Depolarizing noise decays a metric exponentially toward an asymptote, so this
    three-parameter model is a closer match than linear when gate noise dominates.
    With fewer than three λ points the fit is underdetermined, and if the
    nonlinear fit fails we fall back to linear extrapolation.
    """
    if len(factors) < 3:
        return _linear_extrap(factors, values)
    try:
        from scipy.optimize import OptimizeWarning, curve_fit
    except ImportError as exc:  # pragma: no cover - exercised only without scipy
        raise ImportError(tr("err.zne_scipy")) from exc
    import warnings

    import numpy as np

    lam = np.asarray(factors, dtype=float)
    y = np.asarray(values, dtype=float)

    def _model(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
        return a * np.exp(-b * x) + c

    p0 = (y[0] - y[-1], 0.5, y[-1])
    bounds = ([-np.inf, 0.0, -np.inf], [np.inf, np.inf, np.inf])
    try:
        with warnings.catch_warnings():
            # a perfect exponential fit leaves the covariance unestimated; harmless
            warnings.simplefilter("ignore", OptimizeWarning)
            popt, _ = curve_fit(_model, lam, y, p0=p0, bounds=bounds, maxfev=20000)
    except (ImportError, RuntimeError, ValueError):  # pragma: no cover - defensive fallback
        return _linear_extrap(factors, values)
    return float(popt[0] + popt[2])


def _extrapolate(factors: Sequence[float], values: Sequence[float], method: str) -> float:
    if method == "exponential":
        return _exponential_extrap(factors, values)
    return _linear_extrap(factors, values)


def _success_value(counts: dict, targets: set, shots: int) -> float:
    hits = sum(counts.get(bs, 0) for bs in targets)
    return hits / shots


def _expectation_value(circuit: Circuit, nm: NoiseModel, observable: str) -> float:
    from .simulators import DensityMatrixEngine

    engine = DensityMatrixEngine(circuit.num_qubits, noise=nm)
    for op in circuit.ops:
        engine.apply(op.name, list(op.qubits), op.params)
    return engine.expectation(observable)


def _validate_observable(observable: object, n_qubits: int) -> str:
    """Validate and normalize a Pauli string to length ``n_qubits``."""
    if not isinstance(observable, str) or not observable:
        raise ValueError(tr("err.zne_observable", observable=observable))
    obs = observable.upper()
    if set(obs) - _PAULI_CHARS:
        raise ValueError(tr("err.zne_observable", observable=observable))
    if len(obs) != n_qubits:
        raise ValueError(tr("err.pauli_len", actual=len(obs), expected=n_qubits))
    return obs


def _pauli_rotation(observable: str) -> list:
    """Basis-rotation gates mapping each Pauli (X/Y/Z) into the Z basis (I untouched)."""
    gates = []
    for q, p in enumerate(observable):
        if p == "X":
            gates.append(GateOperation("h", (q,)))
        elif p == "Y":
            gates.append(GateOperation("rz", (q,), (-math.pi / 2,)))
            gates.append(GateOperation("h", (q,)))
    return gates


def _observable_circuit(circuit: Circuit, k: int, observable: str) -> Circuit:
    """Fold the unitary part, append basis rotations, and measure every qubit."""
    folded = fold(circuit, k)
    out = Circuit()
    out.allocate(folded.num_qubits)
    for op in folded.ops:
        if op.name == "measure":
            continue
        out.add(op)
    for g in _pauli_rotation(observable):
        out.add(g)
    for q in range(folded.num_qubits):
        out.add(GateOperation("measure", (q,)))
    return out


def _expectation_from_counts(counts: dict, observable: str, shots: int) -> float:
    """Estimate ⟨O⟩ = mean over samples of ∏ sign(bit_i) over non-identity Paulis.

    Qubit 0 is the least-significant bit (rightmost character), matching the
    native / qiskit / qi counts convention.
    """
    n = len(observable)
    total = 0.0
    for bs, cnt in counts.items():
        prod = 1.0
        for q, p in enumerate(observable):
            if p == "I":
                continue
            if int(bs[n - 1 - q]) == 1:
                prod = -prod
        total += prod * cnt
    return total / shots


def zne(
    circuit: Circuit,
    noise: NoiseModel | float | None = None,
    factors: Sequence[float] = (1, 3, 5),
    target: str | Iterable[str] | None = None,
    observable: str | None = None,
    backend: str = "native",
    shots: int = 1024,
    device: str | None = None,
    calibration: ReadoutCalibration | None = None,
    extrapolation: str = "linear",
) -> ZNEResult:
    """Estimate a noise-free quantity by zero-noise extrapolation.

    Exactly one of ``target`` (success-probability metric) or ``observable``
    (expectation-value metric) must be provided.

    Parameters:
        circuit: the unitary circuit (with optional trailing measurements).
        noise: depolarizing noise (NoiseModel or probability) for the simulator
            backends. Must be None for backend="qi" — real hardware has intrinsic
            noise, and ZNE amplifies it by folding the circuit instead of injecting.
        factors: noise amplification factors λ (odd integers ≥ 1), default (1, 3, 5).
        target: bitstring or iterable of bitstrings to hit (success metric).
        observable: a Pauli string such as "ZZ" (expectation metric).
        backend: "native", "qiskit", or "qi".
        shots: samples per λ (success metric; also the qi expectation metric).
        device: hardware device for backend="qi" (e.g. "tuna17"); ignored otherwise.
        calibration: an optional readout calibration (from ``calibrate``). When given,
            the measured counts at each λ are corrected through it before the metric
            is evaluated — stacking readout mitigation on top of ZNE. Its qubit count
            must match ``circuit.num_qubits``.
        extrapolation: "linear" (default) fits a straight line through (λ, metric);
            "exponential" fits ``a·e^(-bλ) + c``, a better match when depolarizing
            gate noise dominates. "exponential" requires scipy.
    """
    if extrapolation not in ("linear", "exponential"):
        raise ValueError(tr("err.zne_extrap", method=extrapolation))

    if (target is None) == (observable is None):
        raise ValueError(tr("err.zne_metric"))

    if calibration is not None and calibration.n != circuit.num_qubits:
        raise ValueError(
            tr("err.zne_calib_n", calib=calibration.n, qubits=circuit.num_qubits)
        )

    def _correct(counts: dict) -> dict:
        if calibration is None:
            return counts
        return calibration.apply(counts, shots)

    ks = _validate_factors(factors)

    # quantum-inspire: folding amplifies the hardware's intrinsic noise (no injection)
    if backend == "qi":
        if noise is not None:
            raise ValueError(tr("err.zne_qi_noise"))
        from .backends import get_backend

        be = get_backend("qi", device=device)
        if observable is not None:
            obs = _validate_observable(observable, circuit.num_qubits)
            values = tuple(
                _expectation_from_counts(
                    _correct(
                        be.run(_observable_circuit(circuit, k, obs), shots=shots).counts
                        or {}
                    ),
                    obs,
                    shots,
                )
                for k in ks
            )
            return ZNEResult(
                factors=tuple(float(f) for f in factors),
                values=values,
                extrapolated=_extrapolate(factors, values, extrapolation),
                metric="expectation",
            )
        targets = {target} if isinstance(target, str) else set(target)
        values = [
            _success_value(
                _correct(be.run(fold(circuit, k), shots=shots).counts or {}),
                targets,
                shots,
            )
            for k in ks
        ]
        return ZNEResult(
            factors=tuple(float(f) for f in factors),
            values=tuple(values),
            extrapolated=_extrapolate(factors, values, extrapolation),
            metric="success",
        )

    # simulator backends: inject depolarizing noise explicitly
    nm = resolve_noise(noise)
    if not nm.enabled:
        raise ValueError(tr("err.zne_noise"))

    if observable is not None:
        obs = _validate_observable(observable, circuit.num_qubits)
        values = tuple(
            _expectation_value(fold(circuit, k), nm, obs) for k in ks
        )
        return ZNEResult(
            factors=tuple(float(f) for f in factors),
            values=values,
            extrapolated=_extrapolate(factors, values, extrapolation),
            metric="expectation",
        )

    # success metric
    if backend not in ("native", "qiskit"):
        raise ValueError(tr("err.zne_backend", backend=backend))
    from .backends import get_backend

    be = get_backend(backend)
    targets = {target} if isinstance(target, str) else set(target)
    values = []
    for k in ks:
        folded = fold(circuit, k)
        result = be.run(folded, shots=shots, noise=nm)
        values.append(
            _success_value(_correct(result.counts or {}), targets, shots)
        )
    extrap = _extrapolate(factors, values, extrapolation)
    # Success probability is bounded [0, 1]; clamp extrapolation overshoot
    extrap = max(0.0, min(1.0, extrap))
    return ZNEResult(
        factors=tuple(float(f) for f in factors),
        values=tuple(values),
        extrapolated=extrap,
        metric="success",
    )


__all__ = ["ZNEResult", "fold", "zne"]
