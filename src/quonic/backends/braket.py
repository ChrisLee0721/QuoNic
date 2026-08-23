"""AWS Braket backend adapter.

Submits circuits to Amazon Braket simulators or real quantum hardware.

Supported devices:
    - SV1: State vector simulator (up to 34 qubits)
    - TN1: Tensor network simulator (up to 50 qubits, low entanglement)
    - DM1: Density matrix simulator (up to 17 qubits, with noise)
    - Local: Local simulator (no AWS account needed)
    - Real hardware: IonQ, Rigetti, QuEra, Oxford Quantum

Prerequisites:
    pip install 'quonic[braket]'
    # or: pip install amazon-braket-sdk

Usage:
    # Cloud simulator
    qshow(backend='braket', device='arn:aws:braket:...:device/quantum-simulator/amazon/sv1')

    # Local simulator (no AWS account needed)
    qshow(backend='braket', device='local')

    # Real hardware
    qshow(backend='braket', device='arn:aws:braket:us-east-1:...:device/qpu/ionq/Aria-1')

Credentials:
    Set AWS credentials via environment variables or ~/.aws/credentials:
        export AWS_ACCESS_KEY_ID=...
        export AWS_SECRET_ACCESS_KEY=...
        export AWS_DEFAULT_REGION=us-east-1
"""

from __future__ import annotations

from typing import Any, ClassVar

from .._i18n import tr
from ..ir import Circuit
from ..noise import NoiseModel
from ..result import Result
from .base import Backend

# Default device ARNs for Braket cloud simulators
_BRAKET_DEVICES = {
    "sv1": "arn:aws:braket:::device/quantum-simulator/amazon/sv1",
    "tn1": "arn:aws:braket:::device/quantum-simulator/amazon/tn1",
    "dm1": "arn:aws:braket:::device/quantum-simulator/amazon/dm1",
}


class BraketBackend(Backend):
    name = "braket"
    methods = frozenset({"statevector", "density_matrix"})
    _CAPABILITIES: ClassVar[dict[str, bool]] = {"noise": True, "ctrl": False, "mid_measure": False, "gpu": False}

    def __init__(self, device: str = "sv1") -> None:
        """Initialize Braket backend.

        Args:
            device: device identifier. Can be:
                - "sv1", "tn1", "dm1" (cloud simulators)
                - "local" (local simulator, no AWS account needed)
                - Full ARN string (e.g. "arn:aws:braket:...:device/qpu/ionq/Aria-1")
        """
        # Resolve shorthand names
        self.device = _BRAKET_DEVICES.get(device.lower(), device)

    def run(
        self,
        circuit: Circuit,
        shots: int = 1024,
        noise: NoiseModel | float | None = None,
        method: str = "statevector",
        return_state: bool = False,
    ) -> Any:
        """Run a circuit on AWS Braket.

        Args:
            circuit: the circuit to run
            shots: number of measurement shots
            noise: noise model (only supported on DM1)
            method: simulation method
            return_state: if True, return the state vector (only on SV1/local)

        Returns:
            Result with measurement counts.
        """
        try:
            from braket.aws import AwsDevice
            from braket.circuits import Circuit as BraketCircuit
            from braket.circuits import gates as BG
        except ImportError as e:
            raise ImportError(
                tr("err.braket_missing")
                + "\n    pip install 'quonic[braket]"
                + "\n    or: pip install amazon-braket-sdk"
            ) from e

        # Decompose high-level gates (mcz, ccx, cp) into basic gates
        from ..compiler import decompose
        decomposed = decompose(circuit)

        # Build Braket circuit
        bc = BraketCircuit()
        for op in decomposed.ops:
            if op.name == "measure":
                continue
            _translate_gate(bc, BG, op)

        # Auto-measure all qubits
        for q in range(decomposed.num_qubits):
            bc.measure(q)

        # Handle local simulator
        if self.device == "local":
            from braket.devices.local_simulator import LocalSimulator

            device = LocalSimulator()
            result = device.run(bc, shots=shots).result()
            counts = result.measurement_counts
            return Result.from_counts(counts, shots)

        # Handle noise (only on DM1)
        if noise is not None:
            if "dm1" not in self.device:
                raise ValueError(
                    "Noise simulation requires DM1 device. "
                    "Use: qshow(backend='braket', device='dm1')"
                )
            nm = _resolve_noise(noise)
            bc = _apply_noise(bc, BG, nm)

        # Submit to cloud device
        device = AwsDevice(self.device)
        task = device.run(bc, shots=shots)
        result = task.result()
        counts = result.measurement_counts

        return Result.from_counts(counts, shots)

    def __repr__(self) -> str:
        return f"BraketBackend(device={self.device})"


def _translate_gate(bc, BG, op):
    """Translate a QuoNic gate to a Braket gate."""
    name = op.name
    q = op.qubits

    gate_map = {
        "h": lambda: bc.h(q[0]),
        "x": lambda: bc.x(q[0]),
        "y": lambda: bc.y(q[0]),
        "z": lambda: bc.z(q[0]),
        "i": lambda: bc.i(q[0]),
        "rx": lambda: bc.rx(q[0], op.params[0]),
        "ry": lambda: bc.ry(q[0], op.params[0]),
        "rz": lambda: bc.rz(q[0], op.params[0]),
        "p": lambda: bc.phaseshift(q[0], op.params[0]),
        "cx": lambda: bc.cnot(q[0], q[1]),
        "cnot": lambda: bc.cnot(q[0], q[1]),
        "cz": lambda: bc.cz(q[0], q[1]),
        "swap": lambda: bc.swap(q[0], q[1]),
        "ccx": lambda: bc.ccnot(q[0], q[1], q[2]),
        "cswap": lambda: bc.cswap(q[0], q[1], q[2]),
        "measure": lambda: None,
    }

    handler = gate_map.get(name)
    if handler:
        handler()
    else:
        # Try custom gate registry
        try:
            from ..gates import _GATE_REGISTRY
            if name in _GATE_REGISTRY:
                gate = _GATE_REGISTRY[name]
                if gate.matrix is not None:
                    import numpy as np
                    bc.unitary(matrix=np.array(gate.matrix, dtype=complex), targets=list(q))
                    return
        except (ImportError, AttributeError):
            pass
        raise ValueError(
            tr("err.braket_gate", name=name)
            + f"\nSupported: {', '.join(sorted(gate_map.keys()))}"
        )


def _resolve_noise(noise):
    """Convert noise parameter to NoiseModel."""
    from ..noise import resolve_noise
    return resolve_noise(noise)


def _apply_noise(bc, BG, nm):
    """Apply noise model to Braket circuit (DM1 only)."""
    from braket.circuits import Noise as BraketNoise

    # Apply depolarizing noise after each gate
    if nm.single > 0:
        bc.apply_gate_noise(BraketNoise.Depolarizing(nm.single))
    if nm.double > 0:
        bc.apply_gate_noise(BraketNoise.Depolarizing(nm.double), target_gates=[BG.CNot, BG.CZ])

    return bc
