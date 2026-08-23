"""CuPy GPU engine — universal fallback for backends without native GPU support.

Uses CuPy (numpy GPU drop-in) for statevector simulation.  Falls back to numpy
when CuPy or GPU hardware is unavailable, so the engine always works.

Gate logic mirrors ``EngineBackend._sv_*`` but uses the ``xp`` (array module)
abstraction so the same code runs on both CPU (numpy) and GPU (CuPy).
"""

from __future__ import annotations

from typing import Any, ClassVar

from ..ir import Circuit, CRegCondition
from ..noise import NoiseModel
from ..result import Result
from .engine import EngineBackend


def _xp():
    """Return CuPy if available, otherwise numpy."""
    try:
        import cupy
        if cupy.cuda.runtime.getDeviceCount() > 0:
            return cupy
    except (ImportError, RuntimeError, ValueError):
        pass
    import numpy
    return numpy


def _check_gpu_memory(n: int, xp: Any) -> None:
    """Check if GPU has enough memory for a 2^n statevector.  Raises if not."""
    if xp.__name__ != "cupy":
        return  # numpy — no GPU memory to check
    try:
        import cupy
        free, _total = cupy.cuda.Device().mem_info
        needed = 2**n * 16  # complex128 = 16 bytes
        if needed > free:
            raise MemoryError(
                f"GPU memory insufficient for {n}-qubit statevector: "
                f"need {needed / 2**30:.1f}GB, free {free / 2**30:.1f}GB"
            )
    except MemoryError:
        raise
    except (ImportError, RuntimeError, ValueError):
        pass  # memory check unavailable, proceed anyway


class _CuPyState:
    """Mutable container for CuPy statevector + qubit count."""

    __slots__ = ("n", "sv")

    def __init__(self, sv, n):
        self.sv = sv
        self.n = n


class CupyEngineBackend(EngineBackend):
    name = "cupy"
    _MISSING_ERR = "err.gpu_missing"
    _GATE_ERR = "err.gpu_gate"
    methods = frozenset({"statevector", "density_matrix", "gpu"})
    _CAPABILITIES: ClassVar[dict[str, bool]] = {"noise": True, "ctrl": True, "mid_measure": True, "gpu": True}

    def _create(self, n: int) -> Any:
        xp = _xp()
        sv = xp.zeros(2**n, dtype=complex)
        sv[0] = 1.0
        return _CuPyState(sv, n)

    def _apply_one(
        self, engine: Any, name: str, qubits: list[int], params: tuple[float, ...]
    ) -> None:
        engine.sv = _sv_apply_gate(engine.sv, name, qubits, params, engine.n)

    def _sample(self, engine: Any, shots: int, n: int) -> dict[str, int]:
        return _sv_sample(engine.sv, shots, engine.n, _xp())

    def _get_statevector(self, engine: Any, n: int) -> Any:
        import numpy as np
        sv = engine.sv
        if hasattr(sv, "get"):
            return sv.get()
        return np.asarray(sv)

    def _run_gpu(self, circuit: Circuit, shots: int, nm: NoiseModel) -> Result:
        """GPU execution: build statevector, apply gates, sample."""
        xp = _xp()
        n = circuit.num_qubits
        _check_gpu_memory(n, xp)

        if any(op.name in ("cif", "cmeasure", "cwhile") for op in circuit.ops):
            return self._run_gpu_dynamic(circuit, shots, nm, xp)

        sv = xp.zeros(2**n, dtype=complex)
        sv[0] = 1.0

        for op in circuit.ops:
            if op.name == "measure":
                continue
            sv = _sv_apply_gate(sv, op.name, list(op.qubits), op.params, n)
            if nm.enabled:
                sv = _sv_apply_noise(sv, op.qubits, nm, n, xp)

        counts = _sv_sample(sv, shots, n, xp)
        if nm.readout > 0:
            counts = self._apply_readout_noise(counts, n, nm.readout)
        return Result.from_counts(counts, shots)

    def _run_dynamic(self, circuit, shots, nm, method, return_state=False):
        """Override: use CuPy statevector for classical control flow."""
        xp = _xp()
        _check_gpu_memory(circuit.num_qubits, xp)
        return self._run_gpu_dynamic(circuit, shots, nm, xp)

    def _run_gpu_dynamic(
        self, circuit: Circuit, shots: int, nm: NoiseModel, xp: Any
    ) -> Result:
        """Per-shot GPU execution with classical control flow."""
        n = circuit.num_qubits
        counts: dict[str, int] = {}

        for _ in range(shots):
            sv = xp.zeros(2**n, dtype=complex)
            sv[0] = 1.0
            cregs: dict[str, int] = {}
            sv = _sv_execute(sv, circuit.ops, cregs, n, xp)
            shot_counts = _sv_sample(sv, 1, n, xp)
            for bs, c in shot_counts.items():
                counts[bs] = counts.get(bs, 0) + c

        if nm.readout > 0:
            counts = self._apply_readout_noise(counts, n, nm.readout)
        return Result.from_counts(counts, shots)


# ------------------------------------------------------------------ #
#  Gate logic (xp-agnostic: works with both numpy and cupy)
# ------------------------------------------------------------------ #

def _sv_apply_gate(sv, name, qubits, params, n):
    """Apply a gate to a state vector."""
    xp = _xp()
    if name in ("measure", "identity", "i"):
        return sv
    u = _gate_matrix(name, params, xp)
    if u is not None and len(qubits) == 1:
        return _sv_apply_single(sv, u, qubits[0], n, xp)
    if name == "cx" and len(qubits) == 2:
        return _sv_apply_cx(sv, qubits[0], qubits[1], n, xp)
    if name == "cz" and len(qubits) == 2:
        return _sv_apply_cz(sv, qubits[0], qubits[1], n, xp)
    if name == "cp" and len(qubits) == 2:
        return _sv_apply_cp(sv, qubits[0], qubits[1], params[0], n, xp)
    if name == "ccx" and len(qubits) == 3:
        return _sv_apply_ccx(sv, qubits[0], qubits[1], qubits[2], n, xp)
    if name == "swap" and len(qubits) == 2:
        return _sv_apply_swap(sv, qubits[0], qubits[1], n, xp)
    if name == "mcz":
        return _sv_apply_mcz(sv, qubits, n, xp)
    return sv


def _gate_matrix(name, params, xp):
    """Return 2x2 unitary matrix for a single-qubit gate."""
    if name == "h":
        return xp.array([[1, 1], [1, -1]], dtype=complex) / xp.sqrt(2)
    if name == "x":
        return xp.array([[0, 1], [1, 0]], dtype=complex)
    if name == "y":
        return xp.array([[0, -1j], [1j, 0]], dtype=complex)
    if name == "z":
        return xp.array([[1, 0], [0, -1]], dtype=complex)
    if name == "rx":
        t = params[0]
        return xp.array(
            [[xp.cos(t / 2), -1j * xp.sin(t / 2)], [-1j * xp.sin(t / 2), xp.cos(t / 2)]],
            dtype=complex,
        )
    if name == "ry":
        t = params[0]
        return xp.array(
            [[xp.cos(t / 2), -xp.sin(t / 2)], [xp.sin(t / 2), xp.cos(t / 2)]],
            dtype=complex,
        )
    if name == "rz":
        t = params[0]
        return xp.array(
            [[xp.exp(-1j * t / 2), 0], [0, xp.exp(1j * t / 2)]], dtype=complex
        )
    if name == "p":
        t = params[0]
        return xp.array([[1, 0], [0, xp.exp(1j * t)]], dtype=complex)
    return None


def _sv_apply_single(sv, u, qubit, n, xp):
    """Apply a single-qubit gate via reshape + tensordot.

    Reshape gives axes in MSB-first order: axis 0 = qubit n-1, axis n-1 = qubit 0.
    So qubit q maps to axis (n - 1 - q).
    """
    sv2 = sv.reshape([2] * n)
    axis = n - 1 - qubit
    result = xp.tensordot(u, sv2, axes=([1], [axis]))
    result = xp.moveaxis(result, 0, axis)
    return result.ravel()


def _sv_apply_cx(sv, control, target, n, xp):
    """Apply CX (CNOT) gate — vectorized."""
    idx = xp.arange(2**n)
    mask_ctrl = (idx >> control) & 1 == 1
    # Swap amplitudes between states where control=1 and their target-flipped pairs
    new_sv = sv.copy()
    pairs = idx[mask_ctrl] ^ (1 << target)
    new_sv[mask_ctrl] = sv[pairs]
    new_sv[pairs] = sv[mask_ctrl]
    return new_sv


def _sv_apply_cz(sv, q0, q1, n, xp):
    """Apply CZ gate — vectorized."""
    idx = xp.arange(2**n)
    mask = ((idx >> q0) & 1 == 1) & ((idx >> q1) & 1 == 1)
    new_sv = sv.copy()
    new_sv[mask] = -sv[mask]
    return new_sv


def _sv_apply_cp(sv, q0, q1, theta, n, xp):
    """Apply controlled-phase gate — vectorized."""
    idx = xp.arange(2**n)
    mask = ((idx >> q0) & 1 == 1) & ((idx >> q1) & 1 == 1)
    new_sv = sv.copy()
    new_sv[mask] = xp.exp(1j * theta) * sv[mask]
    return new_sv


def _sv_apply_ccx(sv, c0, c1, target, n, xp):
    """Apply Toffoli (CCX) gate — vectorized."""
    idx = xp.arange(2**n)
    mask_ctrl = ((idx >> c0) & 1 == 1) & ((idx >> c1) & 1 == 1)
    new_sv = sv.copy()
    pairs = idx[mask_ctrl] ^ (1 << target)
    new_sv[mask_ctrl] = sv[pairs]
    new_sv[pairs] = sv[mask_ctrl]
    return new_sv


def _sv_apply_swap(sv, q0, q1, n, xp):
    """Apply SWAP gate — vectorized."""
    idx = xp.arange(2**n)
    b0 = (idx >> q0) & 1
    b1 = (idx >> q1) & 1
    mask = b0 != b1
    new_sv = sv.copy()
    # Only swap once per pair (avoid double-swap)
    swap_mask = mask & ((idx >> q0) & 1 == 0)  # only where qubit q0 = 0
    partners = idx[swap_mask] ^ (1 << q0) ^ (1 << q1)
    new_sv[swap_mask] = sv[partners]
    new_sv[partners] = sv[swap_mask]
    return new_sv


def _sv_apply_mcz(sv, qubits, n, xp):
    """Apply multi-controlled Z gate — vectorized."""
    idx = xp.arange(2**n)
    mask = xp.ones(2**n, dtype=bool)
    for q in qubits:
        mask = mask & ((idx >> q) & 1 == 1)
    new_sv = sv.copy()
    new_sv[mask] = -sv[mask]
    return new_sv


def _sv_measure(sv, qubit, n, xp):
    """Measure a qubit, return 0 or 1."""
    import numpy as np
    idx = xp.arange(2**n)
    bit = (idx >> qubit) & 1
    p0 = float(xp.sum(xp.abs(sv[bit == 0]) ** 2))
    return 0 if np.random.random() < p0 else 1


def _sv_collapse(sv, qubit, n, outcome, xp):
    """Collapse state vector after measurement."""
    idx = xp.arange(2**n)
    bit = (idx >> qubit) & 1
    new_sv = sv.copy()
    new_sv[bit != outcome] = 0.0
    norm = xp.linalg.norm(new_sv)
    if norm > 0:
        new_sv /= norm
    return new_sv


def _sv_execute(sv, ops, cregs, n, xp):
    """Execute ops on a state vector with classical control flow."""
    for op in ops:
        name = op.name
        if name == "cmeasure":
            outcome = _sv_measure(sv, op.qubit, n, xp)
            sv = _sv_collapse(sv, op.qubit, n, outcome, xp)
            v = cregs.get(op.creg, 0)
            cregs[op.creg] = (v & ~(1 << op.bit)) | (outcome << op.bit)
        elif name == "cif":
            if isinstance(op.control, int):
                outcome = _sv_measure(sv, op.control, n, xp)
                sv = _sv_collapse(sv, op.control, n, outcome, xp)
                hit = outcome == 1
            elif isinstance(op.control, CRegCondition):
                hit = cregs.get(op.control.creg, 0) == op.control.value
            else:
                hit = cregs.get(op.control, 0) == 1
            branch = op.then_op if hit else op.else_op
            sv = _sv_apply_gate(sv, branch.name, list(branch.qubits), branch.params, n)
        elif name == "cwhile":
            iters = 0
            while cregs.get(op.creg, 0) != op.until:
                sv = _sv_execute(sv, op.body, cregs, n, xp)
                iters += 1
                if iters > 100000:
                    raise RuntimeError("cwhile limit exceeded")
        elif name == "measure":
            pass
        else:
            sv = _sv_apply_gate(sv, name, list(op.qubits), op.params, n)
    return sv


def _sv_apply_noise(sv, qubits, nm, n, xp):
    """Apply depolarizing noise after a gate."""
    import numpy as np
    nq = len(qubits)
    p = nm.single if nq == 1 else nm.double
    if p <= 0:
        return sv
    # Simple depolarizing: with probability p, replace state with maximally mixed
    if np.random.random() < p:
        # Apply random Pauli error
        for q in qubits:
            r = np.random.random()
            if r < 1 / 3:
                sv = _sv_apply_gate(sv, "x", [q], (), n)
            elif r < 2 / 3:
                sv = _sv_apply_gate(sv, "y", [q], (), n)
            else:
                sv = _sv_apply_gate(sv, "z", [q], (), n)
    return sv


def _sv_sample(sv, shots, n, xp):
    """Sample from state vector."""
    probs = xp.abs(sv) ** 2
    probs = probs / probs.sum()
    if hasattr(probs, 'get'):  # CuPy array
        probs_cpu = probs.get()
    else:
        probs_cpu = probs
    import numpy as np
    indices = np.random.choice(2**n, size=shots, p=probs_cpu)
    counts: dict[str, int] = {}
    fmt = f"0{n}b"
    for idx in indices:
        bs = format(int(idx), fmt)[::-1]
        counts[bs] = counts.get(bs, 0) + 1
    return counts
