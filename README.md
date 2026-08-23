# QuoNic — Quantum programming, as simple as writing Python

[![CI](https://github.com/ChrisLee0721/QuoNic/actions/workflows/ci.yml/badge.svg)](https://github.com/ChrisLee0721/QuoNic/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![Version](https://img.shields.io/badge/version-0.11.0-purple.svg)](CHANGELOG.md)

[![Qiskit](https://img.shields.io/badge/Qiskit-1.0+-green.svg)](https://qiskit.org/)
[![Cirq](https://img.shields.io/badge/Cirq-1.0+-orange.svg)](https://quantumai.google/cirq)
[![PennyLane](https://img.shields.io/badge/PennyLane-0.36+-yellow.svg)](https://pennylane.ai/)
[![Qulacs](https://img.shields.io/badge/Qulacs-0.6+-blue.svg)](https://qulacs.org/)
[![TensorCircuit](https://img.shields.io/badge/TensorCircuit-0.12+-red.svg)](https://github.com/tencent-quantum-lab/tensorcircuit)
[![CUDA-Q](https://img.shields.io/badge/CUDA--Q-0.8+-green.svg)](https://developer.nvidia.com/cuda-quantum)
[![MindQuantum](https://img.shields.io/badge/MindQuantum-0.9+-blue.svg)](https://gitee.com/mindspore/mindquantum)
[![QPanda3](https://img.shields.io/badge/QPanda3-3.0+-orange.svg)](https://qcloud.originqc.com.cn/)
[![77 Algorithms](https://img.shields.io/badge/algorithms-77-blueviolet.svg)](src/quonic/algorithms/)
[![771 Tests](https://img.shields.io/badge/tests-771%20passed-brightgreen.svg)](tests/)
[![3 Hardware](https://img.shields.io/badge/hardware-3%20verified-orange.svg)](#real-hardware)

```python
from quonic import qgate, qshow
from quonic.gates import H, CX

qgate(H, 0)      # Hadamard
qgate(CX, 0, 1)  # CNOT
qshow()           # Bell state: |00⟩ + |11⟩
```

**3 lines. 12+ backends. 77 algorithms.** No `QuantumCircuit`, no `backend`, no `measure`.

[中文文档](README.zh-CN.md) · [Quickstart](docs/quickstart.md) · [Examples](examples/)

---

## Install

```bash
pip install quonic
```

## Switch backend — one argument

```python
qshow(backend='qiskit')    # IBM
qshow(backend='cirq')      # Google
qshow(backend='qulacs')    # C++ fast
qshow(backend='gpu')       # GPU
```

## Real hardware

| Platform | Device | Status |
|----------|--------|--------|
| Origin Quantum | WK_C180 | ✅ Verified |
| AWS Braket | Rigetti Cepheus | ✅ Verified |
| Quantum Inspire | Tuna-9/17 | ✅ Verified |

## 77 algorithm templates

```python
from quonic.algorithms import grover, vqe, qft, qaoa_maxcut

grover("11", 2)                    # Search
vqe(hamiltonian, 2)                # Chemistry
qft(n_qubits=4)                    # Fourier Transform
qaoa_maxcut(edges, n_qubits=3)     # Optimization
```

## License

[Apache 2.0](LICENSE) — friendly to commercial use, with patent protection.
