<p align="center">
  <h1 align="center">QuoNic</h1>
  <p align="center">
    <b>量子编程，像写 Python 一样简单。</b>
  </p>
  <p align="center">
    不需要学 QuantumCircuit，不需要理解 backend，不需要手动 measure。<br>
    你会写 Python，就会写量子程序。
  </p>
</p>

<div align="center">

  <img src="https://img.shields.io/badge/版本-0.11.0-purple?style=for-the-badge" alt="版本" />
  <img src="https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/许可证-Apache%202.0-blue?style=for-the-badge" alt="许可证" />
  <img src="https://img.shields.io/badge/测试-771%20通过-22C55E?style=for-the-badge" alt="测试" />

</div>

<div align="center">

  <img src="https://img.shields.io/badge/Qiskit-1.0+-6929C4?style=for-the-badge&logo=qiskit&logoColor=white" alt="Qiskit" />
  <img src="https://img.shields.io/badge/Cirq-1.0+-FB8C00?style=for-the-badge" alt="Cirq" />
  <img src="https://img.shields.io/badge/Qulacs-0.6+-00599C?style=for-the-badge" alt="Qulacs" />
  <img src="https://img.shields.io/badge/CUDA--Q-0.8+-76B900?style=for-the-badge&logo=nvidia&logoColor=white" alt="CUDA-Q" />
  <img src="https://img.shields.io/badge/算法-77-7C3AED?style=for-the-badge" alt="算法" />
  <img src="https://img.shields.io/badge/硬件-3%20已验证-F59E0B?style=for-the-badge" alt="硬件" />

</div>

<br>

## 问题

今天的量子编程 unnecessarily 复杂。在 Qiskit 中写一个简单的贝尔态需要 10+ 行代码，理解电路对象、后端选择和手动测量。切换框架意味着重写所有代码。

<table width="100%">
  <tr>
    <td width="33%" valign="top">
      <h3 align="center">概念太多</h3>
      <p align="center">QuantumCircuit、backend、transpile、measure_all — 写一个门之前要学 8+ 个新概念。</p>
    </td>
    <td width="33%" valign="top">
      <h3 align="center">框架锁定</h3>
      <p align="center">为 Qiskit 写的代码不能在 Cirq 上运行。切换框架意味着重写所有代码。</p>
    </td>
    <td width="33%" valign="top">
      <h3 align="center">没有智能默认</h3>
      <p align="center">选错模拟方法可能慢 1000 倍。用户不应该需要了解内部实现。</p>
    </td>
  </tr>
</table>

<br>

## 解决方案

QuoNic 抽象掉了复杂性。三行代码，任何后端，任何硬件。

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

<br>

## 特性

| 特性 | 说明 |
|------|------|
| **3 行语法** | `qgate` + `qshow` — 就这么简单 |
| **12+ 后端** | 一个参数：`qshow(backend='qiskit')` |
| **77 个算法** | Grover、Shor、VQE、QAOA、QFT、纠错、量子机器学习 |
| **智能调度** | 自动选择最快方法（statevector / stabilizer / MPS / density matrix） |
| **GPU 加速** | `qshow(backend='gpu')` — 10 倍加速 |
| **真实硬件** | 本源量子、AWS Braket、Quantum Inspire 已验证 |
| **噪声模拟** | 退极化、比特翻转、相位翻转、退相干 |
| **误差缓解** | ZNE、读出校准 |
| **23 种可视化** | 电路图、Bloch 球、直方图 |

<br>

## 技术栈

| 组件 | 技术 | 说明 |
|------|------|------|
| **核心** | Python 3.9+ | IR、调度器、编译器、噪声模型 |
| **后端** | Qiskit · Cirq · Qulacs · TensorCircuit · CUDA-Q · MindQuantum · QPanda3 | 12+ 量子后端 |
| **GPU** | CuPy · Qulacs GPU · CUDA-Q | GPU 加速模拟 |
| **硬件** | 本源量子 · AWS Braket · Quantum Inspire | 真实量子硬件 |
| **可视化** | Matplotlib | 23 种图表，懒加载 |

<br>

## 真实硬件

| 平台 | 设备 | 状态 |
|------|------|------|
| 本源量子 | WK\_C180 | ✅ 已验证 |
| AWS Braket | Rigetti Cepheus-1-108Q | ✅ 已验证 |
| Quantum Inspire | Tuna-9 / Tuna-17 | ✅ 已验证 |

```python
qshow(backend='qpanda', device='WK_C180')
qshow(backend='qi', device='tuna9')
```

<br>

## 算法

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

<br>

## 路线图

- [x] **核心 API：** `qgate`、`qshow`、`reset` — 极简语法
- [x] **12+ 后端：** Qiskit、Cirq、Qulacs、TensorCircuit、CUDA-Q、MindQuantum、QPanda3
- [x] **77 个算法模板：** 从 Grover 到量子机器学习
- [x] **智能调度：** 自动选择最快模拟方法
- [x] **GPU 加速：** CuPy、Qulacs GPU、CUDA-Q
- [x] **真实硬件：** 本源量子、AWS Braket、Quantum Inspire
- [x] **噪声模拟：** 退极化、比特翻转、相位翻转、退相干
- [x] **误差缓解：** ZNE、读出校准
- [x] **可视化：** 23 种图表，Matplotlib
- [x] **文档：** 92 个示例，中英双语
- [ ] **更多后端：** IonQ、Rigetti、Xanadu、QuEra
- [ ] **量子网络：** 多节点量子通信
- [ ] **容错计算：** 逻辑量子比特操作

<br>

## 文档

- [快速开始](docs/quickstart.md) — 5 分钟
- [示例](docs/examples/) — 92 个示例，中英双语
- [API 文档](docs/api/) — 所有模块
- [教程](docs/tutorials/) — 分步指南

<br>

## 贡献

Fork → branch → PR。

详见 [CONTRIBUTING.md](CONTRIBUTING.md)。

<br>

## 许可证

[Apache 2.0](LICENSE) — 对商用友好，提供专利保护。

<br>

<p align="center">
  <sub>由 <a href="https://github.com/ChrisLee0721">Lee LapYuen</a> 用 ❤️ 构建 · <a href="README.md">English</a></sub>
</p>
