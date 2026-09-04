"""Backend registry."""

from __future__ import annotations

from .._i18n import tr
from .azure import AzureBackend
from .base import Backend
from .braket import BraketBackend
from .cirq import CirqBackend
from .cqlib import CqlibBackend
from .cudaq import CudaQBackend
from .cupy_engine import CupyEngineBackend
from .engine import EngineBackend
from .ibm import IBMBackend
from .ionq import IonQBackend
from .mindquantum import MindQuantumBackend
from .native import NativeBackend
from .originq import OriginQBackend
from .pennylane import PennyLaneBackend
from .qi import QuantumInspireBackend
from .qiskit import QiskitBackend
from .qpanda import QPandaBackend
from .quera import QuEraBackend
from .qulacs import QulacsBackend
from .rigetti import RigettiBackend
from .tensorcircuit import TensorCircuitBackend
from .xanadu import XanaduBackend

# Engine registry: the backend argument only recognizes these five engine names
# (local simulators plus the qi cloud entry point). Specific real-hardware devices
# (Tuna-9 / Tuna-17 / QX emulator) are selected via the device argument, not
# registered here as independent backend names.
_REGISTRY: dict[str, Backend] = {
    "qiskit": QiskitBackend(),
    "cirq": CirqBackend(),
    "pennylane": PennyLaneBackend(),
    "native": NativeBackend(),
    "qi": QuantumInspireBackend(),
    "qulacs": QulacsBackend(),
    "tensorcircuit": TensorCircuitBackend(),
    "cudaq": CudaQBackend(),
    "mindquantum": MindQuantumBackend(),
    "qpanda": QPandaBackend(),
    "cqlib": CqlibBackend(),
    "cupy": CupyEngineBackend(),
    "ibm": IBMBackend(),
    "braket": BraketBackend(),
    "azure": AzureBackend(),
    "ionq": IonQBackend(),
    "rigetti": RigettiBackend(),
    "xanadu": XanaduBackend(),
    "quera": QuEraBackend(),
    "originq": OriginQBackend(),
}

# Backward-compatible aliases for the legacy one-shot device shortcuts: backend="tuna9" is equivalent to backend="qi", device="tuna9".
_BACKEND_ALIASES: dict[str, tuple[str, str]] = {
    "tuna9": ("qi", "tuna9"),
    "tuna17": ("qi", "tuna17"),
    "qx": ("qi", "qx"),
}


def resolve_target(
    backend: str, device: str | None = None
) -> tuple[str, str | None]:
    """Normalize (backend, device) into (engine name, device name).

    - If backend is a legacy device shortcut (tuna9/tuna17/qx), translate it into ("qi", device alias);
    - device is only meaningful when backend="qi"; passing device to any other engine raises a Chinese error;
    - when backend="auto", device must not be passed (auto only probes local simulators).
    """
    if backend in _BACKEND_ALIASES:
        alias_engine, alias_device = _BACKEND_ALIASES[backend]
        if device is not None and str(device).lower() != alias_device:
            raise ValueError(
                tr(
                    "err.device_alias_conflict",
                    backend=backend,
                    alias_device=alias_device,
                    device=device,
                )
            )
        return alias_engine, alias_device
    # Backends that accept device parameter
    _DEVICE_BACKENDS = {"qi", "braket", "ibm", "azure", "ionq", "rigetti", "xanadu", "quera", "originq"}
    if device is not None and backend not in _DEVICE_BACKENDS:
        raise ValueError(
            tr("err.device_only_qi", backend=backend)
            + f"\nBackends that support device: {', '.join(sorted(_DEVICE_BACKENDS))}"
        )
    return backend, device


def _detect_available() -> str:
    """Probe installed backends in priority order (qiskit -> cirq -> pennylane -> native)."""
    import importlib.util

    candidates = (
        ("qiskit", ("qiskit", "qiskit_aer")),
        ("cirq", ("cirq",)),
        ("pennylane", ("pennylane",)),
    )
    for name, modules in candidates:
        if all(importlib.util.find_spec(m) is not None for m in modules):
            return name
    return "native"  # Fallback: the in-house engine, which only needs numpy


def get_backend(name: str, device: str | None = None) -> Backend:
    """Get a backend by name. Supports legacy device-shortcut aliases (returns a qi instance carrying the device)."""
    name, device = resolve_target(name, device)
    if name == "auto":
        name = _detect_available()
    if name not in _REGISTRY:
        # Fuzzy match
        import difflib
        matches = difflib.get_close_matches(name, _REGISTRY.keys(), n=1, cutoff=0.6)
        hint = f" Did you mean '{matches[0]}'?" if matches else ""
        raise ValueError(
            tr("err.unknown_backend", name=name, engines=", ".join(sorted(_REGISTRY)))
            + hint
        )
    if device is not None:
        # Backends that accept device parameter
        if name == "qi":
            return QuantumInspireBackend(device)
        elif name == "originq":
            return OriginQBackend(device)
        elif name == "braket":
            from .braket import BraketBackend
            return BraketBackend(device)
        elif name in ("ibm", "azure", "ionq", "rigetti", "xanadu", "quera"):
            cls = _REGISTRY[name]
            return cls(device=device)
    return _REGISTRY[name]


def get_backend_for_method(
    name: str, method: str, device: str | None = None
) -> Backend:
    """Resolve a backend by method: fall back to native (the in-house engine) when the target backend does not support the method.

    Users can use stabilizer / MPS and other methods on any backend — when the
    capabilities do not match, uniformly fall back to the QuoNic in-house engine
    rather than forcing statevector.
    """
    be = get_backend(name, device=device)
    if be.supports(method):
        return be
    native = _REGISTRY["native"]
    if native.supports(method):
        return native
    raise ValueError(tr("err.no_method_support", method=method))


def available_backends() -> list[str]:
    return sorted(_REGISTRY)


_EXPLORE_TIMEOUT = 10  # seconds


def _explore_subprocess(args: tuple) -> float:
    """Module-level function for ProcessPoolExecutor (must be picklable)."""
    explore_backend, ops, n, shots, explore_method = args
    from .ir import Circuit as _Circuit

    c = _Circuit()
    c.num_qubits = n
    c.ops = list(ops)
    be = get_backend(explore_backend)
    import time as _time
    t0 = _time.time()
    be.run(c, shots=shots, method=explore_method)
    return _time.time() - t0


def _background_explore(circuit, explore_backend, explore_method, profiles, feats):
    """Run an alternative backend in a subprocess with a hard timeout.

    Uses ProcessPoolExecutor so the subprocess can be terminated on timeout.
    """
    from concurrent.futures import ProcessPoolExecutor
    from concurrent.futures import TimeoutError as _Timeout

    ops = list(circuit.ops)
    n = circuit.num_qubits
    args = (explore_backend, ops, n, 1024, explore_method)

    try:
        with ProcessPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_explore_subprocess, args)
            elapsed = future.result(timeout=_EXPLORE_TIMEOUT)
        profiles.report_result(feats, f"{explore_backend}/{explore_method}", elapsed, None)
        profiles._save()
    except (_Timeout, Exception):
        pass


def _pick_alternative(chosen_rec, profiles, feats):
    """Pick an alternative backend/method for exploration.

    Uses a two-tier strategy:
    1. Prefer alternatives predicted within 10x of the chosen backend (fast exploration).
    2. If none found, pick any eligible backend (broad exploration).
    """
    import random as _random

    from ..scheduler.capabilities import eligible_methods

    n = feats["n"]
    gate_count = feats.get("gate_count", n)
    chosen_key = f"{chosen_rec.backend}/{chosen_rec.method}"

    gate_types = feats.get("gate_types", [])
    eligible = eligible_methods(gate_types) if gate_types else None

    candidates = []
    broad_candidates = []
    chosen_time = profiles.predict_time(chosen_key, n, gate_count)

    for bm_key in profiles.profiles.profiles:
        if bm_key == chosen_key:
            continue
        method = bm_key.split("/", 1)[1] if "/" in bm_key else bm_key
        if eligible is not None and method not in eligible:
            continue
        broad_candidates.append(bm_key)
        if chosen_time is not None and chosen_time > 0:
            t = profiles.predict_time(bm_key, n, gate_count)
            if t is not None and 0 < t < chosen_time * 10:
                candidates.append(bm_key)

    # Prefer close alternatives, fall back to any eligible
    pool = candidates if candidates else broad_candidates
    if not pool:
        return None

    pick = _random.choice(pool)
    if "/" in pick:
        backend, method = pick.split("/", 1)
        from ..scheduler.registry import Recommendation
        return Recommendation(backend, method)
    return None


def run_circuit(circuit, backend: str = "auto", shots: int = 1024, **kwargs):
    """Run a circuit with automatic backend/method selection via the scheduler.

    When backend="auto", uses the scheduler to pick the optimal backend and method
    based on circuit features (n, gate_count, depth, is_clifford, treewidth, etc.).
    Falls back to statevector if no measured data is available.

    With 5% probability, runs an alternative backend in the background to gather
    timing data for the learning scheduler (does not block the user).

    This is the recommended entry point for algorithm templates and user code.
    """
    if backend == "auto":
        import random as _random
        import threading as _threading
        import time as _time

        from ..scheduler import default_profiles, schedule
        from ..scheduler.features import circuit_features as _cf

        profiles = default_profiles()
        rec = schedule(circuit, profiles=profiles)
        be = get_backend(rec.backend)
        feats = _cf(circuit)

        t0 = _time.time()
        result = be.run(circuit, shots=shots, method=rec.method, **kwargs)
        elapsed = _time.time() - t0

        if profiles is not None:
            try:
                profiles.report_result(feats, f"{rec.backend}/{rec.method}", elapsed, None)
                profiles._save()  # persist immediately
            except Exception:
                pass

            # 5% exploration: run an alternative backend in background
            if _random.random() < 0.05:
                explore = _pick_alternative(rec, profiles, feats)
                if explore is not None:
                    _threading.Thread(
                        target=_background_explore,
                        args=(circuit, explore.backend, explore.method, profiles, feats),
                        daemon=True,
                    ).start()

        return result
    return get_backend(backend).run(circuit, shots=shots, **kwargs)


__all__ = [
    "Backend",
    "EngineBackend",
    "available_backends",
    "get_backend",
    "get_backend_for_method",
    "resolve_target",
    "run_circuit",
]
