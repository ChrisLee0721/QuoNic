<p align="center">
  <img src="Readme_logo.png" width="300" alt="QuoNic Banner" />
</p>

<p align="center">
  <b>Quantum programming, as simple as writing Python.</b>
</p>
<p align="center">
  No QuantumCircuit to learn, no backend to understand, no manual measure.<br>
  If you can write Python, you can write quantum programs.
</p>

<div align="center">

  <img src="https://img.shields.io/badge/Version-0.11.0-purple?style=for-the-badge" alt="Version" />
  <img src="https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/License-Apache%202.0-blue?style=for-the-badge" alt="License" />
  <img src="https://img.shields.io/badge/Tests-771%20passed-22C55E?style=for-the-badge" alt="Tests" />

</div>

<div align="center">

  <img src="https://img.shields.io/badge/Qiskit-1.0+-6929C4?style=for-the-badge&logo=qiskit&logoColor=white" alt="Qiskit" />
  <img src="https://img.shields.io/badge/Cirq-1.0+-FB8C00?style=for-the-badge" alt="Cirq" />
  <img src="https://img.shields.io/badge/Qulacs-0.6+-00599C?style=for-the-badge" alt="Qulacs" />
  <img src="https://img.shields.io/badge/CUDA--Q-0.8+-76B900?style=for-the-badge&logo=nvidia&logoColor=white" alt="CUDA-Q" />
  <img src="https://img.shields.io/badge/Algorithms-77-7C3AED?style=for-the-badge" alt="Algorithms" />
  <img src="https://img.shields.io/badge/Hardware-3%20verified-F59E0B?style=for-the-badge" alt="Hardware" />

</div>

<br>

## The Problem

Quantum programming today is unnecessarily complex. Writing a simple Bell state in Qiskit requires 10+ lines, understanding circuit objects, backend selection, and manual measurement. Switching frameworks means rewriting everything.

<table width="100%">
  <tr>
    <td width="33%" valign="top">
      <h3 align="center">Too Many Concepts</h3>
      <p align="center">QuantumCircuit, backend, transpile, measure_all — 8+ new concepts before writing a single gate.</p>
    </td>
    <td width="33%" valign="top">
      <h3 align="center">Framework Lock-in</h3>
      <p align="center">Code written for Qiskit can't run on Cirq. Switching frameworks means rewriting everything.</p>
    </td>
    <td width="33%" valign="top">
      <h3 align="center">No Smart Defaults</h3>
      <p align="center">Choosing the wrong simulation method can be 1000x slower. Users shouldn't need to know internals.</p>
    </td>
  </tr>
</table>

<br>

## The Solution

QuoNic abstracts away the complexity. Three lines of code, any backend, any hardware.

<p align="center">
  <img src="docs/local/preview_terminal.png" width="500" alt="QuoNic Terminal Preview" />
</p>

```python
from quonic import qgate, qshow
from quonic.gates import H, CX

qgate(H, 0)
qgate(CX, 0, 1)
qshow()
```

```bash
pip install quonic
```

<br>

## Features

| Feature | Description |
|---------|-------------|
| **3-line syntax** | `qgate` + `qshow` — that's it |
| **12+ backends** | One argument: `qshow(backend='qiskit')` |
| **77 algorithms** | Grover, Shor, VQE, QAOA, QFT, error correction, quantum ML |
| **Smart scheduler** | Auto-selects fastest method (statevector / stabilizer / MPS / density matrix) |
| **GPU acceleration** | `qshow(backend='gpu')` — 10x speedup |
| **Real hardware** | Origin Quantum, AWS Braket, Quantum Inspire verified |
| **Noise simulation** | Depolarizing, bit-flip, phase-flip, decoherence |
| **Error mitigation** | ZNE, readout calibration |
| **23 visualizations** | Circuit diagrams, Bloch sphere, histograms |

<br>

## Tech Stack

| Component | Technology | Description |
|-----------|-----------|-------------|
| **Core** | Python 3.9+ | IR, scheduler, compiler, noise models |
| **Backends** | Qiskit · Cirq · Qulacs · TensorCircuit · CUDA-Q · MindQuantum · QPanda3 | 12+ quantum backends |
| **GPU** | CuPy · Qulacs GPU · CUDA-Q | GPU-accelerated simulation |
| **Hardware** | Origin Quantum · AWS Braket · Quantum Inspire | Real quantum hardware |
| **Visualization** | Matplotlib | 23 chart types, lazy-loaded |

<br>

## Real Hardware

| Platform | Device | Status |
|----------|--------|--------|
| Origin Quantum | WK\_C180 | ✅ Verified |
| AWS Braket | Rigetti Cepheus-1-108Q | ✅ Verified |
| Quantum Inspire | Tuna-9 / Tuna-17 | ✅ Verified |

```python
qshow(backend='qpanda', device='WK_C180')
qshow(backend='qi', device='tuna9')
```

<br>

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

<br>

## Roadmap

- [x] **Core API:** `qgate`, `qshow`, `reset` — minimal syntax
- [x] **12+ backends:** Qiskit, Cirq, Qulacs, TensorCircuit, CUDA-Q, MindQuantum, QPanda3
- [x] **77 algorithm templates:** From Grover to quantum ML
- [x] **Smart scheduler:** Auto-select fastest simulation method
- [x] **GPU acceleration:** CuPy, Qulacs GPU, CUDA-Q
- [x] **Real hardware:** Origin Quantum, AWS Braket, Quantum Inspire
- [x] **Noise simulation:** Depolarizing, bit-flip, phase-flip, decoherence
- [x] **Error mitigation:** ZNE, readout calibration
- [x] **Visualization:** 23 chart types with Matplotlib
- [x] **Documentation:** 92 examples, bilingual (EN/ZH)
- [ ] **More backends:** IonQ, Rigetti, Xanadu, QuEra
- [ ] **Quantum networking:** Multi-node quantum communication
- [ ] **Fault-tolerant computing:** Logical qubit operations

<br>

## Docs

- [Quick Start](docs/quickstart.md) — 5 minutes
- [Examples](docs/examples/) — 92 examples, bilingual
- [API Reference](docs/api/) — all modules
- [Tutorials](docs/tutorials/) — step-by-step guides

<br>

## Contributing

Fork → branch → PR.

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup and code style.

<br>

## License

[Apache 2.0](LICENSE) — friendly to commercial use, with patent protection.

<br>

<p align="center">
  <sub>Built with ❤️ by <a href="https://github.com/ChrisLee0721">Lee LapYuen</a> · <a href="README.zh-CN.md">中文文档</a></sub>
</p>
