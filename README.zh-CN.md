# QuoNic — 量子编程，像写 Python 一样简单

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
qshow()           # 贝尔态：|00⟩ + |11⟩
```

**3 行代码。12+ 后端。77 个算法。** 不需要 `QuantumCircuit`，不需要 `backend`，不需要 `measure`。

[English](README.md) · [快速开始](docs/quickstart.md) · [示例](examples/)

---

## 安装

```bash
pip install quonic
```

## 切换后端 — 一个参数

```python
qshow(backend='qiskit')    # IBM
qshow(backend='cirq')      # Google
qshow(backend='qulacs')    # C++ 高性能
qshow(backend='gpu')       # GPU 加速
```

## 真实硬件

| 平台 | 设备 | 状态 |
|------|------|------|
| 本源量子 | WK_C180 | ✅ 已验证 |
| AWS Braket | Rigetti Cepheus | ✅ 已验证 |
| Quantum Inspire | Tuna-9/17 | ✅ 已验证 |

## 77 个算法模板

```python
from quonic.algorithms import grover, vqe, qft, qaoa_maxcut

grover("11", 2)                    # 搜索
vqe(hamiltonian, 2)                # 量子化学
qft(n_qubits=4)                    # 傅里叶变换
qaoa_maxcut(edges, n_qubits=3)     # 优化
```

## 许可证

[Apache 2.0](LICENSE) — 对商用友好，提供专利保护。
