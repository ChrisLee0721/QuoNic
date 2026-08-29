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


__all__ = [
    "Backend",
    "EngineBackend",
    "available_backends",
    "get_backend",
    "get_backend_for_method",
    "resolve_target",
]
