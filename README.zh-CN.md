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

**QuoNic 是一个让量子编程变得像写 Python 一样简单的工具。**

不需要学 `QuantumCircuit`，不需要理解 `backend`，不需要手动 `measure`。你会写 Python，就会用量子计算。

[English](README.md)

---

##  30 秒快速开始

```python
from quonic import qgate, qshow
from quonic.gates import H, CX

qgate(H, 0)
qgate(CX, 0, 1)
qshow()
```

**这是量子计算中最经典的贝尔态（Bell State）。** 同样的功能，用 Qiskit 原生代码需要 10+ 行。QuoNic 只需要 3 行。运行结果会直接显示在终端或 Jupyter 中。

更多**复制即跑**的示例（GHZ、`qif`、`QInt`、Grover、VQE、QAOA、噪声、GPU 加速、误差缓解）见 [`examples/`](examples/)。

---

##  5 分钟上手

**第 1 步：安装**
```bash
pip install quonic
```

**第 2 步：第一个量子电路（贝尔态）**
```python
from quonic import qgate, qshow
from quonic.gates import H, CX

qgate(H, 0)      # 对量子比特 0 施加 Hadamard 门
qgate(CX, 0, 1)  # CNOT：纠缠量子比特 0 和 1
qshow()           # 运行并显示结果
```

**第 3 步：切换后端（同一段代码，不同模拟器）**
```python
qshow(backend='qiskit')      # IBM Qiskit
qshow(backend='cirq')        # Google Cirq
qshow(backend='qulacs')      # Qulacs（高性能 C++）
qshow(backend='tensorcircuit') # TensorCircuit（JAX）
```

**第 4 步：加噪声**
```python
qshow(noise=0.05)  # 5% 去极化噪声
```

**第 5 步：GPU 加速**
```python
qshow(method='gpu')  # 自动选最优 GPU 后端
```

**就这么简单。** 同一段代码，任何后端，任何硬件。这就是 QuoNic 的方式。

---

##  安装

```bash
pip install quonic
```

后端是可选依赖，按需安装。要一键装齐所有后端（含算法模板的 numpy/scipy）：

```bash
pip install 'quonic[qiskit,cirq,pennylane,algorithms,all-sim]'
```

只装某一个后端，例如只用 Cirq：`pip install 'quonic[cirq]'`。未安装的后端在调用时会给出明确的英文提示（设置 `QUONIC_LANG=zh` 可切换为中文）。

额外模拟器后端（Qulacs / TensorCircuit / CUDA-Q / MindQuantum / QPanda3 / CqLib）：`pip install 'quonic[all-sim]'` 或单独安装，如 `pip install 'quonic[qulacs]'`。

硬件/云后端（IBM Quantum / AWS Braket / Azure Quantum / IonQ / Rigetti / Xanadu / QuEra）：`pip install 'quonic[all-hw]'` 或单独安装。

> **硬件状态：** Origin Quantum（WK\_C180）、AWS Braket（Rigetti Cepheus）、Quantum Inspire（Tuna-9/17）已通过**真实硬件验证**。IBM Quantum、Azure Quantum、IonQ、Xanadu、QuEra 后端按原样提供，未经真实硬件验证。本地模拟器（Qulacs / TensorCircuit / QPanda3 / CuPy / native）已完整测试。

GPU 加速：`pip install 'quonic[gpu]'`（CuPy）。

可视化是独立可选依赖：`pip install 'quonic[viz]'`（仅 matplotlib，不引入 Graphviz / Seaborn / NetworkX）。

---

##  核心特性

### 1. 极简语法：3 行代码跑通贝尔态

你不需要理解“量子电路对象”，不需要选择“后端模拟器”，不需要手动“测量”。QuoNic 替你处理一切。

### 2. 一个参数切换所有后端

```python
# 使用 Qiskit 模拟器（默认）
qshow(backend='qiskit')

# 切换到 Cirq
qshow(backend='cirq')

# 切换到 Qulacs（高性能 C++）
qshow(backend='qulacs')

# 切换到 TensorCircuit（JAX/TensorFlow/PyTorch）
qshow(backend='tensorcircuit')

# 噪声模拟
qshow(backend='qiskit', noise=0.05)

# 真实硬件（Quantum Inspire）已接入，需登录
qshow(backend='qi')                    # QX 云模拟器（默认，提交前验证）
qshow(backend='qi', device='tuna9')    # Tuna-9 真机
qshow(backend='qi', device='tuna17')   # Tuna-17 真机
qshow(backend='qi', device='qx')       # QX 云模拟器
```

**同一段代码，不加修改，跑在任何后端上。** 极简语法 + 后端无关，是 QuoNic 的组合差异化。

### 3. 条件门与”if = 叠加态”

QuoNic 用 `qif` 实现量子叠加控制，并严格区分两种概念：

- **量子叠加控制（`qif`，已实现）**：控制比特处于叠加态时**不测量**，两个分支
  相干叠加，产生真纠缠——这是”两种分支同时发生”，不是先测量再二选一。
  ```python
  from quonic import qgate, qif, qshow
  from quonic.gates import H, X, I

  qgate(H, 0)                       # 控制比特进入叠加态
  qif(0).then(X, 1).else_(I, 1)     # q0==1 翻转 q1，否则不动（= 受控 X）
  qshow()
  ```
  `else_(I, ...)` 里的 `I` 是恒等门，让「受控门 = qif 特例」写得自然。
- **条件门（经典控制，规划中）**：先测量、再按结果选择分支，这是”坍缩之后的经典分支”。
  ```python
  # 规划中：基于测量结果的条件门
  # qgate(H, 0)
  # if qgate(MEASURE, 0) == 0:
  #     qgate(X, 1)
  # else:
  #     qgate(Z, 1)
  ```

我们不把”测量后的经典分支”包装成”叠加态”——教错物理，比不教更糟。

### 4. 真正的“技术惠普”

- **清晰的错误信息**（默认英文，设置 `QUONIC_LANG=zh` 切换中文）：报错时告诉你“哪里错了、为什么错、怎么改”
- **自动补全**：在 VS Code / Jupyter 中自动提示门名称和参数
- **自动测量**：忘记写 `measure`？`qshow()` 自动补全

### 5. 智能调度器：自动挑最快的方法

量子模拟有四种方法，快慢差几个数量级，选错直接撞墙：

| 方法 | 复杂度 | 适合 |
|------|--------|------|
| `statevector` | 2^n | 通用默认 |
| `stabilizer` | 多项式 | 纯 Clifford 电路（如纠错码） |
| `matrix_product_state` | 随树宽增长 | 低树宽电路（如 QAOA） |
| `density_matrix` | 4^n | 噪声模拟 |

QuoNic 的调度器根据电路特征（门类型、树宽、是否含噪声）自动选择，不用你手动
指定方法。实测证据：**GHZ(24) 快 36 倍、QAOA(24) 快 19 倍**，Grover 的
`mcz` 只有 `statevector` 能跑，调度器会自动绕开会崩溃的方法。

```python
from quonic.scheduler import schedule
rec = schedule(circuit)   # -> Recommendation(backend='qiskit', method='stabilizer')
```

详见 [调度器基准与实测数据](docs/benchmarks.md)。

### 6. GPU 加速 — 一个参数

```python
qshow(method='gpu')                              # 当前后端 GPU
qshow(backend='qulacs', method='gpu')            # qulacs GPU（兜底 CuPy）
```

调度器也可自动选最优 GPU 后端：

```python
from quonic.scheduler import recommend_backend_gpu, circuit_features

rec = recommend_backend_gpu(circuit_features(circuit))
# -> Recommendation(backend='qulacs', method='gpu')
```

| 电路类型 | 最优 GPU 后端 | 原因 |
|---|---|---|
| 高纠缠、小电路 | qulacs | 最快状态向量 GPU |
| 低纠缠、大电路 | tensorcircuit | 张量网络 GPU |
| 有经典控制流 | qulacs | 支持状态塌缩 |
| 兜底 | cupy | 通用 GPU 引擎 |

安装：`pip install 'quonic[gpu]'`（CuPy）或 `pip install 'quonic[qulacs]'`（原生 GPU）。

### 7. 全量可视化套件：23 类图，只用 Matplotlib

```python
from quonic.viz import plot_circuit, plot_counts, plot_decision_tree

plot_circuit(circuit)        # 门序列电路图
plot_counts(result)          # 测量直方图
plot_decision_tree()         # 调度决策树
```

23 类图覆盖四层：**用户刚需**（电路图 / 直方图 / 拓扑图）、**调度器证据**
（方法对比 / 决策树 / 热力图 / 降级链 / 特征雷达图）、**算法教学**（能量收敛
/ Grover 振幅 / 态向量 / 布洛赫球）、**量子态**（密度矩阵 / 纠缠 / 门矩阵 /
路由 / 逐门态演化 / 噪声成本）。全部只用 matplotlib 一个依赖，懒加载，
`import quonic` 零开销。详见 [可视化套件](docs/visualization.md)。

---

##  对比：QuoNic vs Qiskit

| 场景 | Qiskit | QuoNic |
|------|--------|-------|
| **跑通第一个量子程序** | 需要理解 5-8 个新概念 | 只需要 2 个概念：`qgate` 和 `qshow` |
| **代码行数（贝尔态）** | 8-12 行 | **3 行** |
| **从安装到看到结果** | 30-60 分钟 | **2-3 分钟** |
| **切换后端** | 重写全部代码 | **改一个参数** |

---

##  为什么叫 QuoNic？

QuoNic 是 **Quantum Unified Operation Native Interface Core** 的首字母缩写：

| 字母 | 词 | 含义 |
|------|-----|------|
| Q | Quantum | 量子 |
| U | Unified | 统一 —— 一个参数切换所有后端 |
| O | Operation | 操作 —— `qgate` / `qshow` |
| N | Native | 原生 —— 像写 Python 一样自然 |
| I | Interface | 接口 —— 后端适配层 |
| C | Core | 核心 —— IR / 调度器 / 编译 |

读作 /ˈkwɑnɪk/（“阔尼克”）。

---

##  当前支持的后端

| 后端 | 状态 | 说明 |
|------|------|------|
| Qiskit | ✅ 稳定 | IBM 生态 · 4 种模拟方法 · 噪声 · 经典控制流 |
| Cirq | ✅ 稳定 | Google 生态 · statevector · 噪声 |
| PennyLane | ✅ 稳定 | 量子机器学习 · statevector · 噪声 |
| Qulacs | ✅ 稳定 | 高性能 C++ 模拟器 · statevector + 密度矩阵 · 噪声 |
| TensorCircuit | ✅ 稳定 | JAX/TensorFlow/PyTorch 后端 · statevector + 密度矩阵 · 噪声 |
| CUDA-Q | ✅ 稳定 | NVIDIA GPU 加速 · statevector · 全局噪声模型 |
| MindQuantum | ✅ 稳定 | 华为 · statevector + 密度矩阵 · 噪声（仅 Linux/macOS） |
| QPanda3 | ✅ 稳定 | 本源量子 · statevector + 密度矩阵 |
| CqLib | ⚠️ 仅云端 | 中电信量子 · 无本地模拟器 |
| Quantum Inspire | ✅ 已接入 | 真实硬件 Tuna-9 / Tuna-17 + QX 模拟器 |
| Native | ✅ 稳定 | 自研 numpy 引擎 · 4 种模拟方法 · 噪声 · fallback |

> **注意**：Qiskit / Cirq / PennyLane / Qulacs / TensorCircuit / QPanda3 运行在**本地模拟器**上。
> CUDA-Q 需要 NVIDIA CUDA 环境。MindQuantum 需要 Linux/macOS。CqLib 仅支持云端执行（天衍平台）。
> Quantum Inspire 真机通过 `qshow(backend="qi", device="tuna9")` 接入。

为硬件铺路，QuoNic 已内置 `CouplingMap`（耦合图）、`compile()` 编译 seam，以及 `decompose()` 门分解——把高阶门（`cp` / `ccx` / `mcz`）展开成基础门集。后者是 QuoNic 自己拥有的「可移植核心」：用户不被某个后端的电路形状绑住，Grover 的 `mcz` 分解成 `cx / h / p` 后能跑通所有后端方法。已内置 `route_swaps()` 贪心 SWAP 路由（配合 `plot_routing` 可视化），将来接 IBM / 国产引擎时，只需在编译层接入，无需改动 IR 或调度器。

### 并行执行

```python
from quonic import qgate, qshow_all, run_circuits
from quonic.gates import H, CX, X

# 同一电路在多个后端上并行执行
qgate(H, 0)
qgate(CX, 0, 1)
results = qshow_all(['qiskit', 'cirq', 'qulacs'])

# 不同电路并行执行
def bell(): qgate(H, 0); qgate(CX, 0, 1)
def flip(): qgate(X, 0)
results = run_circuits([bell, flip], backend='qiskit')
```

---

##  77 个算法模板

QuoNic 内置 77 个算法模板，覆盖 10 大领域——从基础量子计算到前沿研究演示。每个算法均有边界条件和使用案例说明。详见[完整算法报告](docs/QuoNic_Algorithm_Report.pdf)。

### 使用方式

```python
from quonic.algorithms import grover, vqe, qaoa_maxcut, deutsch_jozsa

# Grover 搜索 |11>
result = grover("11", 2, shots=1024)

# VQE 基态能量
hamiltonian = [(1.0, "ZZ"), (1.0, "XI"), (1.0, "IX")]
result = vqe(hamiltonian, 2)

# QAOA MaxCut
result = qaoa_maxcut([(0,1), (1,2), (0,2)], 3, p=2)

# Deutsch-Jozsa
result = deutsch_jozsa(2, my_oracle, shots=100)
```

### 算法目录

| 领域 | 算法 | 类型 |
|------|------|------|
| **基石算法** (9) | QFT、Deutsch-Jozsa、Bernstein-Vazirani、Simon、SWAP 测试、Hadamard 测试、振幅放大/估计、QPE | 完整 |
| **搜索优化** (7) | QAOA（通用/TSP/MIS/背包）、Grover、量子计数、量子行走、量子退火 | 完整 + 演示 |
| **量子化学** (8) | VQE、哈密顿量导入（OpenFermion/PennyLane/字符串）、Trotter、哈密顿量模拟、动态模拟、费米子映射、QSP、分子 VQE | 完整 + 演示 |
| **线性代数** (6) | HHL、矩阵求逆、特征值求解、PDE/ODE 求解、数据拟合 | 演示 |
| **通信密码** (6) | 隐形传态、BB84、E91、超密编码、Shor、离散对数 | 完整 + 演示 |
| **混合算法** (7) | VQC、量子核方法、QNG、VQR、QNN、QSVM、量子退火混合 | 完整 + 演示 |
| **量子纠错** (9) | 比特/相位翻转码、Shor 9 比特、Steane 7 比特、稳定子、syndrome、表面码、颜色码、容错门 | 演示 |
| **统计采样** (3) | 量子蒙特卡洛、拒绝采样、贝叶斯推理 | 演示 |
| **代数** (3) | 隐藏子群、格问题 SVP、椭圆曲线 | 演示 |
| **前沿演示** (10) | QCNN、QGNN、分布式 QAOA、QTransformer、QRL、QTDA、QPCA、聚类、QGAN、QBM | 演示 |

> **完整** = 标准算法，能在模拟器上跑，结果有意义。**演示** = 最小化核心概念展示，非生产级。

---

##  文档与教程

- [快速入门](docs/quickstart.md) — 5 分钟上手 QuoNic
- [Jupyter 教程](docs/tutorial.ipynb) — 可运行的交互单元
- [调度器基准与实测数据](docs/benchmarks.md) — 自动选最快方法的护城河
- [可视化套件](docs/visualization.md) — 23 类图，只用 Matplotlib
- [国产硬件调研](docs/domestic-hardware.md) — QPanda3 / CqLib 接入评估

---

##  贡献指南

QuoNic 是一个开源项目（Apache 2.0），欢迎任何形式的贡献：

- 报告 Bug
- 提出新功能建议
- 提交代码（新后端适配器、新门、新功能）
- 完善文档和教程

开发环境、代码风格与约定见 [CONTRIBUTING.md](CONTRIBUTING.md)。

---

##  许可证

QuoNic 使用 [Apache License 2.0](LICENSE)，对商用和闭源友好，同时提供专利保护。

---

##  给项目加星

如果 QuoNic 对你有帮助，请在 GitHub 上给我们一个 ⭐️。你的支持是我们持续改进的动力。
