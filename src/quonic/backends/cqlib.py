"""Cqlib backend adapter — circuit construction + native simulation.

Cqlib is a circuit construction library for cloud execution (TianYanPlatform /
GuoDunPlatform), not a local simulator.  This backend accumulates gate ops
during ``_apply_one`` and delegates simulation to the native statevector engine.
The cqlib.Circuit is built in parallel for QASM/QCIS export via ``engine``.
"""

from __future__ import annotations

from typing import Any

from .._i18n import tr
from .engine import EngineBackend


class CqlibBackend(EngineBackend):
    name = "cqlib"
    _MISSING_ERR = "err.cqlib_missing"
    _GATE_ERR = "err.cqlib_gate"
    methods = frozenset({"statevector"})
    _CAPABILITIES = {"noise": False, "ctrl": False, "mid_measure": False}

    def _create(self, n: int) -> Any:
        try:
            from cqlib import Circuit
        except ImportError as e:
            raise ImportError(tr(self._MISSING_ERR)) from e
        self._n = n
        self._ops: list[tuple[str, list, tuple]] = []
        return Circuit(n)

    def _apply_one(
        self, engine: Any, name: str, qubits: list[int], params: tuple[float, ...]
    ) -> None:
        if name in ("identity", "i", "measure"):
            pass
        else:
            # Accumulate for native replay
            self._ops.append((name, list(qubits), params))

    def _sample(self, engine: Any, shots: int, n: int) -> dict[str, int]:
        """Delegate simulation to the native statevector engine.

        The native engine uses qubit-0-is-MSB convention for bitstrings, while
        QuoNic's engine backends use qubit-0-is-LSB.  Reverse each bitstring.
        """
        from ..simulators import StatevectorEngine

        sv = StatevectorEngine(n)
        for name, qubits, params in self._ops:
            sv.apply(name, qubits, params)
        return {bs[::-1]: c for bs, c in sv.sample(shots).items()}
