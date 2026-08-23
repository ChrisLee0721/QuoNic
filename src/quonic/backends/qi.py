"""Quantum Inspire real-hardware backend.

Compiles a QuoNic circuit to cQASM 3.0 via qiskit-quantuminspire and submits it to
Quantum Inspire's superconducting real hardware (Tuna-9 / Tuna-17) or the QX emulator
(a 10-qubit cloud simulator for pre-submission validation).

Unlike the simulator backends (qiskit/cirq/pennylane/native):
  - real hardware has no "simulation method" distinction; run() ignores method;
  - hardware has intrinsic noise, so depolarizing noise cannot be injected; run() rejects noise;
  - classical control flow (cif/cwhile) is not supported: superconducting hardware has no mid-circuit measurement feedback.

Prerequisites (one-time):
    1. Install the dependency:   pip install 'qiskit-quantuminspire'
       (note: qiskit-quantuminspire 0.18.x requires qiskit<2.4.0; the current pyproject
        allows qiskit>=1.0; if 2.5.x is already installed you must temporarily downgrade to qiskit==2.3.1)
    2. Log in:       qi login   # OAuth device flow, browser authorization
       The token is stored at ~/.quantuminspire/config.json; do not hardcode or paste it.

Engine vs device: backend only selects the engine (qiskit/cirq/pennylane/native/qi);
the specific real-hardware chip is chosen via the device argument. Bare qi defaults
to the QX cloud simulator (safe, fast, for pre-submission validation); pass
device="tuna9"/"tuna17" explicitly to target real hardware.

Usage:
    from quonic.backends import get_backend
    get_backend("qi").run(circuit, shots=1024)                  # default: QX cloud simulator
    get_backend("qi", device="tuna9").run(circuit, shots=1024)  # real hardware Tuna-9
    get_backend("qi", device="tuna17").run(circuit, shots=1024) # real hardware Tuna-17
    # The legacy device shortcuts are still supported (equivalent to the device forms above):
    get_backend("tuna9").run(circuit, shots=1024)
    # Or construct explicitly:
    from quonic.backends.qi import QuantumInspireBackend
    QuantumInspireBackend("Tuna-9").run(circuit, shots=1024)
"""

from __future__ import annotations

from typing import Any

from .._i18n import tr
from ..ir import Circuit
from ..noise import NoiseModel
from ..result import Result
from .base import Backend
from .setup_guide import ensure_ready
from .translators import TRANSLATORS

# Device alias → official QI device name. Lets qshow(backend="tuna9") / get_backend("qx")
# work in one step without remembering the exact Tuna-9 / QX emulator spelling.
DEVICE_ALIASES: dict[str, str] = {
    "tuna9": "Tuna-9",
    "tuna17": "Tuna-17",
    "qx": "QX emulator",
}

# Default target when using bare qi (no device specified): the QX cloud simulator.
# Real hardware (Tuna-9/Tuna-17) queues, consumes quota, and requires login, so it is
# not the default — to avoid hitting the heaviest path when the user just wants a quick run.
DEFAULT_DEVICE: str = "QX emulator"


def resolve_device(device: str | None) -> str | None:
    """Map a device alias (tuna9 / tuna17 / qx) to the official QI device name; unknown names pass through unchanged."""
    if device is None:
        return None
    return DEVICE_ALIASES.get(str(device).lower(), device)


class QuantumInspireBackend(Backend):
    name = "qi"
    # Real hardware has no "simulation method" distinction; the full method list is
    # only for tooling/docs reference. Capability matching is overridden by supports()
    # to avoid get_backend_for_method falling back to native.
    methods = frozenset(
        {"statevector", "stabilizer", "matrix_product_state", "density_matrix"}
    )

    # Declarative description of the setup guide (setup_guide uses it to generate a
    # "press Enter to continue" interactive guide). When integrating IBM / AWS Braket /
    # domestic hardware in the future, each fills in its own setup while reusing the guide engine.
    setup: dict[str, Any] = {
        "name": "Quantum Inspire",
        "sdk": {
            "package": "qiskit_quantuminspire",       # import name, used to detect whether it is installed
            "pip": "qiskit-quantuminspire",           # PyPI name, used for messages / version queries
            "install": "quonic[quantum-inspire]",     # recommended install command (with extras)
        },
        "auth": {
            "kind": "oauth_cli",
            "command": ["qi", "login"],
            "token_file": "~/.quantuminspire/config.json",
        },
        "conflicts": [{"package": "qiskit", "constraint": "<2.4.0"}],
        "devices": ["Tuna-9", "Tuna-17", "QX emulator"],
        "billing": True,                               # real hardware consumes quota → confirm before submission
    }

    def __init__(self, device: str | None = None) -> None:
        # device=None defaults to the QX cloud simulator (safe, fast, for pre-submission
        # validation); you may also pass the aliases tuna9/tuna17/qx or the official
        # device names "Tuna-9"/"Tuna-17"/"QX emulator"
        self.device: str | None = resolve_device(device)

    def supports(self, method: str) -> bool:
        # Hardware backends do not participate in method capability matching: any method runs directly on hardware
        return True

    def run(
        self,
        circuit: Circuit,
        shots: int = 1024,
        noise: NoiseModel | float | None = None,
        method: str = "statevector",
    ) -> Result:
        if noise is not None:
            raise ValueError(tr("err.qi_noise"))
        self._check_supported(circuit)

        # Preflight checks: dependency / version conflicts / login. Interactive guide under a TTY, otherwise raise a Chinese error.
        ensure_ready(self.setup)

        # Deferred import: the dependency is guaranteed ready at this point
        from qiskit import QuantumCircuit, transpile
        from qiskit_quantuminspire.qi_provider import QIProvider

        qc = QuantumCircuit(circuit.num_qubits, circuit.num_qubits)
        for op in circuit.ops:
            if op.name == "cmeasure":
                # A named classical bit with no feedback semantics is equivalent to an ordinary measurement, mapped to that bit's own classical bit
                qc.measure(op.qubit, op.qubit)
            else:
                TRANSLATORS[op.name].to_qiskit(qc, op, {})

        # Auto-complete: qubits that are not explicitly measured are all measured at the end
        for q in circuit.unmeasured_qubits():
            qc.measure(q, q)

        provider = QIProvider()
        backend = provider.get_backend(self.device or DEFAULT_DEVICE)
        # level 3 aggressively reduces depth/SWAP count (matters for deep Grover/QFT circuits on real hardware)
        qc_compiled = transpile(qc, backend, optimization_level=3)

        job = backend.run(qc_compiled, shots=shots)
        # Real hardware queues (Tuna-17 can exceed 30 min under load), so the timeout
        # is relaxed to 60 minutes; the QX emulator usually returns in seconds.
        result = job.result(timeout=3600)
        counts_hex = result.get_counts()
        counts = {
            _hex_to_bitstring(k, circuit.num_qubits): v
            for k, v in counts_hex.items()
        }
        return Result.from_counts(counts, shots)

    @staticmethod
    def _check_supported(circuit: Circuit) -> None:
        for op in circuit.ops:
            if op.name == "cwhile":
                raise NotImplementedError(tr("err.qi_cwhile"))
            if op.name == "cif":
                raise NotImplementedError(tr("err.qi_cif"))


def _hex_to_bitstring(key: str, n_qubits: int) -> str:
    """Quantum Inspire returns counts keys in 0x.. form; convert them back to an MSB-first bitstring.

    QI's raw cQASM result string places qubit 0 at the rightmost position (standard
    binary), matching the QuoNic native / qiskit backend convention (qubit 0 = LSB =
    rightmost character).
    """
    key = str(key)
    if key.startswith("0x"):
        val = int(key, 16)
    else:  # Fallback: if it is already a binary string
        val = int(key, 2)
    return format(val, f"0{n_qubits}b")
