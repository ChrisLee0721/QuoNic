"""Origin Quantum cloud backend (本源量子云平台).

Submits circuits to Origin Quantum's cloud quantum computers via pyqpanda3.qcloud.

Prerequisites (one-time):
    1. Install:  pip install pyqpanda3
    2. Set API key:  export ORIGINGQ_API_KEY="your-key-here"
       Get your key from: https://console.originqc.com.cn/zh/apikey

Usage:
    from quonic.backends import get_backend

    # Default: WK_C180
    be = get_backend("originq")
    result = be.run(circuit, shots=1024)

    # Specify backend
    be = get_backend("originq", device="WK_C180")
    be = get_backend("originq", device="full_amplitude")  # simulator
    result = be.run(circuit, shots=1024)
"""

from __future__ import annotations

import os
from typing import Any, ClassVar

from ..ir import Circuit
from ..noise import NoiseModel
from ..result import Result
from .base import Backend

_ENV_KEY = "ORIGINGQ_API_KEY"


class OriginQBackend(Backend):
    """Origin Quantum cloud backend via pyqpanda3.qcloud."""

    name = "originq"
    methods = frozenset({"statevector"})

    setup: ClassVar[dict[str, Any]] = {
        "name": "Origin Quantum",
        "sdk": {
            "package": "pyqpanda3",
            "pip": "pyqpanda3",
            "install": "pyqpanda3",
        },
        "auth": {
            "kind": "env_var",
            "env_var": _ENV_KEY,
        },
        "devices": ["WK_C180", "PQPUMESH8", "full_amplitude", "partial_amplitude", "single_amplitude"],
        "billing": True,
    }

    def __init__(self, device: str | None = None, api_key: str | None = None) -> None:
        self.device = device or "WK_C180"
        self._api_key = api_key or os.environ.get(_ENV_KEY, "")
        self._service: Any = None
        self._backend: Any = None

    def supports(self, method: str) -> bool:
        return True

    def _get_service(self) -> Any:
        if self._service is not None:
            return self._service
        if not self._api_key:
            raise RuntimeError(
                f"Origin Quantum API key not set.\n"
                f"Get one at: https://console.originqc.com.cn/zh/apikey\n"
                f"Then set: set {_ENV_KEY}=your-key"
            )
        try:
            from pyqpanda3.qcloud.qcloud import QCloudService
        except ImportError:
            raise ImportError("pyqpanda3 not installed. Run: pip install pyqpanda3")
        self._service = QCloudService(self._api_key)
        return self._service

    def _get_backend(self) -> Any:
        if self._backend is not None:
            return self._backend
        service = self._get_service()
        available = service.backends()
        if self.device not in available:
            raise ValueError(
                f"Backend '{self.device}' not available. "
                f"Available: {list(available.keys())}"
            )
        self._backend = service.backend(self.device)
        return self._backend

    def run(
        self,
        circuit: Circuit,
        shots: int = 1024,
        noise: NoiseModel | float | None = None,
        method: str = "statevector",
    ) -> Result:
        if noise is not None:
            raise ValueError("Cannot inject noise on real hardware.")
        backend = self._get_backend()
        prog = self._to_qprog(circuit)
        job = backend.run(prog, shots)
        raw = job.result()
        return self._parse_result(raw, circuit.num_qubits, shots)

    def _to_qprog(self, circuit: Circuit) -> Any:
        """Convert QuoNic IR circuit to pyqpanda3 QProg."""
        import math

        from pyqpanda3.core import (
            CNOT,
            CZ,
            RX,
            RY,
            RZ,
            SWAP,
            TOFFOLI,
            H,
            I,
            QCircuit,
            QProg,
            S,
            T,
            X,
            Y,
            Z,
            measure,
        )

        n = circuit.num_qubits
        qc = QCircuit(n)

        # SD = S† = RZ(-π/2), TD = T† = RZ(-π/4)
        def _sd(q):
            return RZ(q, -math.pi / 2)

        def _td(q):
            return RZ(q, -math.pi / 4)

        gate_map = {
            "h": lambda op: H(op.qubits[0]),
            "x": lambda op: X(op.qubits[0]),
            "y": lambda op: Y(op.qubits[0]),
            "z": lambda op: Z(op.qubits[0]),
            "s": lambda op: S(op.qubits[0]),
            "sd": lambda op: _sd(op.qubits[0]),
            "t": lambda op: T(op.qubits[0]),
            "td": lambda op: _td(op.qubits[0]),
            "i": lambda op: I(op.qubits[0]),
            "rx": lambda op: RX(op.qubits[0], op.params[0]),
            "ry": lambda op: RY(op.qubits[0], op.params[0]),
            "rz": lambda op: RZ(op.qubits[0], op.params[0]),
            "cx": lambda op: CNOT(op.qubits[0], op.qubits[1]),
            "cz": lambda op: CZ(op.qubits[0], op.qubits[1]),
            "swap": lambda op: SWAP(op.qubits[0], op.qubits[1]),
            "ccx": lambda op: TOFFOLI(op.qubits[0], op.qubits[1], op.qubits[2]),
        }

        for op in circuit.ops:
            if op.name == "measure":
                continue
            handler = gate_map.get(op.name)
            if handler is None:
                raise ValueError(f"Gate '{op.name}' not supported by pyqpanda3 backend.")
            qc << handler(op)

        prog = QProg()
        prog << qc
        for q in range(n):
            prog << measure(q, q)

        return prog

    def _parse_result(self, raw: Any, n_qubits: int, shots: int) -> Result:
        """Parse pyqpanda3 qcloud result into QuoNic Result."""
        if hasattr(raw, "get_counts"):
            raw_counts = raw.get_counts()
            if not raw_counts and hasattr(raw, "get_probs"):
                # full_amplitude simulator returns probabilities, not counts
                probs = raw.get_probs()
                if probs:
                    counts: dict[str, int] = {}
                    for key, prob in probs.items():
                        bs = str(key)
                        if bs.startswith("0x"):
                            bs = format(int(bs, 16), f"0{n_qubits}b")
                        elif bs.startswith("0b"):
                            bs = bs[2:].zfill(n_qubits)
                        # pyqpanda3 returns MSB-first, reverse for qubit-0-is-LSB
                        counts[bs[::-1]] = max(1, round(prob * shots))
                    return Result.from_counts(counts, shots)
        elif isinstance(raw, dict):
            raw_counts = raw
        else:
            raise ValueError(f"Unexpected result type: {type(raw)}")

        counts = {}
        for key, count in raw_counts.items():
            bs = str(key)
            if bs.startswith("0x"):
                bs = format(int(bs, 16), f"0{n_qubits}b")
            elif bs.startswith("0b"):
                bs = bs[2:].zfill(n_qubits)
            # pyqpanda3 returns MSB-first, reverse for qubit-0-is-LSB
            counts[bs[::-1]] = counts.get(bs[::-1], 0) + int(count)

        return Result.from_counts(counts, shots)

    def query_backends(self) -> dict[str, bool]:
        """Query available backends and their online status."""
        service = self._get_service()
        return service.backends()

    def chip_info(self) -> Any:
        """Get chip information for the current backend."""
        backend = self._get_backend()
        return backend.chip_info()
