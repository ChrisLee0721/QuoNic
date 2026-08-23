<div align="center">

<h1 style="font-size: 3em; margin-bottom: 0.2em;">QuoNic</h1>

<p style="font-size: 1.3em; font-weight: bold;">Quantum programming, as simple as writing Python.</p>

[![CI](https://img.shields.io/github/actions/workflow/status/ChrisLee0721/QuoNic/ci.yml?label=CI&logo=github)](https://github.com/ChrisLee0721/QuoNic/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Version](https://img.shields.io/badge/Version-0.11.0-purple.svg)](CHANGELOG.md)

[![Qiskit](https://img.shields.io/badge/Qiskit-1.0+-6929C4?logo=qiskit&logoColor=white)](https://qiskit.org/)
[![Cirq](https://img.shields.io/badge/Cirq-1.0+-FB8C00)](https://quantumai.google/cirq)
[![Qulacs](https://img.shields.io/badge/Qulacs-0.6+-00599C)](https://qulacs.org/)
[![CUDA-Q](https://img.shields.io/badge/CUDA--Q-0.8+-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-quantum)

[![77 Algorithms](https://img.shields.io/badge/Algorithms-77-7C3AED)](src/quonic/algorithms/)
[![771 Tests](https://img.shields.io/badge/Tests-771%20passed-22C55E)](tests/)
[![3 Hardware](https://img.shields.io/badge/Hardware-3%20verified-F59E0B)](#real-hardware)

[Quick Start](#quick-start) · [Features](#features) · [Backends](#backends) · [Algorithms](#algorithms) · [Docs](#docs)

</div>

---

## Quick Start

```python
from quonic import qgate, qshow
from quonic.gates import H, CX

qgate(H, 0)
qgate(CX, 0, 1)
qshow()
```

```
backend: native | shots: 1024
Result:
  |00>     512  ( 50.0%)  ####################
  |11>     512  ( 50.0%)  ####################
```

```bash
pip install quonic
```

---

## Features

| Feature | Description |
|---------|-------------|
| **3-line syntax** | No `QuantumCircuit`, no `backend`, no `measure` — just `qgate` and `qshow` |
| **12+ backends** | One argument switches between Qiskit, Cirq, Qulacs, GPU, real hardware |
| **77 algorithms** | Grover, Shor, VQE, QAOA, QFT, quantum error correction, quantum ML |
| **Smart scheduler** | Auto-selects fastest simulation method (statevector / stabilizer / MPS / density matrix) |
| **GPU acceleration** | `qshow(backend='gpu')` — 10x speedup on large circuits |
| **Real hardware** | Verified on Origin Quantum, AWS Braket, Quantum Inspire |
| **Noise simulation** | Depolarizing, bit-flip, phase-flip, decoherence models |
| **Error mitigation** | ZNE (zero-noise extrapolation), readout calibration |
| **23 visualizations** | Circuit diagrams, Bloch sphere, histograms — only Matplotlib |

---

## Backends

| Backend | Status | Notes |
|---------|--------|-------|
| Qiskit | ✅ stable | IBM ecosystem, all 4 methods, noise, classical control |
| Cirq | ✅ stable | Google ecosystem, statevector, noise |
| Qulacs | ✅ stable | High-performance C++, statevector + density matrix |
| TensorCircuit | ✅ stable | JAX/TensorFlow/PyTorch, statevector + density matrix |
| CUDA-Q | ✅ stable | NVIDIA GPU-accelerated |
| MindQuantum | ✅ stable | Huawei, statevector + density matrix |
| QPanda3 | ✅ stable | Origin Quantum, statevector + density matrix |
| Quantum Inspire | ✅ connected | Real hardware: Tuna-9 / Tuna-17 |
| Native | ✅ stable | In-house numpy engine, fallback |

> **Hardware verified:** Origin Quantum (WK\_C180), AWS Braket (Rigetti Cepheus), Quantum Inspire (Tuna-9/17).

---

## Algorithms

```python
from quonic.algorithms import grover, vqe, qft, qaoa_maxcut

grover("11", 2)                    # Search
vqe(hamiltonian, 2)                # Chemistry
qft(n_qubits=4)                    # Fourier Transform
qaoa_maxcut(edges, n_qubits=3)     # Optimization
```

| Domain | Algorithms |
|--------|-----------|
| **Foundational** | QFT, Deutsch-Jozsa, Bernstein-Vazirani, Simon, QPE |
| **Search & Optimization** | Grover, QAOA (MaxCut/TSP/MIS/Knapsack), quantum annealing |
| **Chemistry** | VQE, Hamiltonian simulation, Trotter, Jordan-Wigner |
| **Machine Learning** | QNN, QSVM, QGAN, QCNN, QGNN, QPCA, QRL |
| **Error Correction** | Bit/phase flip, Shor code, Steane code, surface code, color code |
| **Communication** | Teleportation, BB84, E91, superdense coding |

---

## Real Hardware

| Platform | Device | Status |
|----------|--------|--------|
| Origin Quantum | WK\_C180 | ✅ Verified |
| AWS Braket | Rigetti Cepheus-1-108Q | ✅ Verified |
| Quantum Inspire | Tuna-9 / Tuna-17 | ✅ Verified |

```python
qshow(backend='qpanda', device='WK_C180')
qshow(backend='braket', device='arn:aws:braket:...')
qshow(backend='qi', device='tuna9')
```

---

## Docs

- [Quick Start](docs/quickstart.md) — 5 minutes
- [Examples](docs/examples/) — 92 examples, bilingual
- [API Reference](docs/api/) — all modules
- [Tutorials](docs/tutorials/) — step-by-step guides

---

## Contributing

Fork → branch → PR.

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup and code style.

---

## License

[Apache 2.0](LICENSE) — friendly to commercial use, with patent protection.

[Lee LapYuen](https://github.com/ChrisLee0721) · [中文文档](README.zh-CN.md)
