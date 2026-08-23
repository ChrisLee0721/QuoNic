<div align="center">

# QuoNic

**量子编程，像写 Python 一样简单。**

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

[快速开始](#quick-start) · [特性](#features) · [后端](#backends) · [算法](#algorithms) · [文档](#docs)

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

| 特性 | 说明 |
|------|------|
| **3 行代码** | 不需要 `QuantumCircuit`、`backend`、`measure` — 只需 `qgate` 和 `qshow` |
| **12+ 后端** | 一个参数切换 Qiskit、Cirq、Qulacs、GPU、真实硬件 |
| **77 个算法** | Grover、Shor、VQE、QAOA、QFT、量子纠错、量子机器学习 |
| **智能调度** | 自动选择最快模拟方法（statevector / stabilizer / MPS / density matrix） |
| **GPU 加速** | `qshow(backend='gpu')` — 大电路 10 倍加速 |
| **真实硬件** | 已在本源量子、AWS Braket、Quantum Inspire 上验证 |
| **噪声模拟** | 退极化、比特翻转、相位翻转、退相干模型 |
| **误差缓解** | ZNE（零噪声外推）、读出校准 |
| **23 种可视化** | 电路图、Bloch 球、直方图 — 只需 Matplotlib |

---

## Backends

| 后端 | 状态 | 说明 |
|------|------|------|
| Qiskit | ✅ 稳定 | IBM 生态，4 种模拟方法，噪声，经典控制流 |
| Cirq | ✅ 稳定 | Google 生态，statevector，噪声 |
| Qulacs | ✅ 稳定 | 高性能 C++，statevector + 密度矩阵 |
| TensorCircuit | ✅ 稳定 | JAX/TensorFlow/PyTorch，statevector + 密度矩阵 |
| CUDA-Q | ✅ 稳定 | NVIDIA GPU 加速 |
| MindQuantum | ✅ 稳定 | 华为，statevector + 密度矩阵 |
| QPanda3 | ✅ 稳定 | 本源量子，statevector + 密度矩阵 |
| Quantum Inspire | ✅ 已接入 | 真实硬件：Tuna-9 / Tuna-17 |
| Native | ✅ 稳定 | 自研 numpy 引擎，兜底方案 |

> **已验证硬件：** 本源量子（WK\_C180）、AWS Braket（Rigetti Cepheus）、Quantum Inspire（Tuna-9/17）。

---

## Algorithms

```python
from quonic.algorithms import grover, vqe, qft, qaoa_maxcut

grover("11", 2)                    # 搜索
vqe(hamiltonian, 2)                # 量子化学
qft(n_qubits=4)                    # 傅里叶变换
qaoa_maxcut(edges, n_qubits=3)     # 优化
```

| 领域 | 算法 |
|------|------|
| **基础** | QFT、Deutsch-Jozsa、Bernstein-Vazirani、Simon、QPE |
| **搜索优化** | Grover、QAOA（MaxCut/TSP/MIS/背包）、量子退火 |
| **量子化学** | VQE、哈密顿量模拟、Trotter、Jordan-Wigner |
| **机器学习** | QNN、QSVM、QGAN、QCNN、QGNN、QPCA、QRL |
| **纠错** | 比特/相位翻转码、Shor 码、Steane 码、表面码、颜色码 |
| **通信** | 隐形传态、BB84、E91、超密编码 |

---

## Real Hardware

| 平台 | 设备 | 状态 |
|------|------|------|
| 本源量子 | WK\_C180 | ✅ 已验证 |
| AWS Braket | Rigetti Cepheus-1-108Q | ✅ 已验证 |
| Quantum Inspire | Tuna-9 / Tuna-17 | ✅ 已验证 |

```python
qshow(backend='qpanda', device='WK_C180')
qshow(backend='braket', device='arn:aws:braket:...')
qshow(backend='qi', device='tuna9')
```

---

## Docs

- [快速开始](docs/quickstart.md) — 5 分钟上手
- [示例](docs/examples/) — 92 个示例，中英双语
- [API 文档](docs/api/) — 所有模块
- [教程](docs/tutorials/) — 分步指南

---

## Contributing

Fork → branch → PR。

详见 [CONTRIBUTING.md](CONTRIBUTING.md)。

---

## License

[Apache 2.0](LICENSE) — 对商用友好，提供专利保护。

[Lee LapYuen](https://github.com/ChrisLee0721) · [English](README.md)
