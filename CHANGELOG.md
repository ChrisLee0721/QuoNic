# Changelog

本项目所有重要变更都记录于此。All notable changes to this project are documented here.

## [0.12.1] — 2026-08-26

本源量子云平台接入 + GCIQA 适配器框架完成。
OriginQ Cloud integration + GCIQA adapter framework complete.

### 修复 Fixed

- **originq 依赖修正**：`cqlib` → `pyqpanda3`，本源量子云使用 pyqpanda3.qcloud
  **originq dependency fix**: `cqlib` → `pyqpanda3`, OriginQ Cloud uses pyqpanda3.qcloud

### 新增 Added

- **本源量子云后端**：通过 pyqpanda3.qcloud 接入 WK_C180 / PQPUMESH8 / 全幅度模拟器
  **OriginQ Cloud backend**: WK_C180 / PQPUMESH8 / full_amplitude simulator via pyqpanda3.qcloud

- **GCIQA 适配器框架**：PDB 解析、金属配位模板、蛋白质粗粒化、结合位点检测、验证、报告
  **GCIQA adapter framework**: PDB parsing, metal templates, protein coarse-graining, binding site detection, validation, reporting

- **GCIQA 基准实验**：Zn²⁺金属蛋白酶、共价结合、NMR 稀疏约束、经典方法对比
  **GCIQA benchmarks**: Zn²⁺ metalloproteinase, covalent binding, NMR sparse constraints, classical comparison

## [0.11.0] — 2026-08-20

穷举测试 + 真机验证 + API 锁定。
Exhaustive testing + real hardware verification + API lock.

### 新增 Added

- **穷举测试**：跨后端一致性 + 边界情况 + 功能组合，771 测试全绿
  **Exhaustive tests**: cross-backend consistency + edge cases + integration, 771 tests passing

- **真机验证**：本源量子 WK_C180 + AWS Rigetti Cepheus + Quantum Inspire Tuna
  **Real hardware**: Origin Quantum WK_C180 + AWS Rigetti Cepheus + Quantum Inspire Tuna

- **ML 框架完整**：伴随微分 + GPU 加速 + 批处理 + 混合模型 + 可视化
  **ML framework complete**: adjoint diff + GPU + batch + hybrid model + visualization

- **MPS 张量网络**：期望值 + 正则化 + DMRG + 噪声 + 自定义门
  **MPS tensor network**: expectation + canonicalize + DMRG + noise + custom gates

- **ZX-calculus**：7 种重写规则 + 电路提取 + 模式匹配
  **ZX-calculus**: 7 rewrite rules + circuit extraction + pattern matching

### API 冻结 / API Freeze

v1.0.0 后遵循 semver 2.0.0：
- Patch (1.0.x): bug fix, 不改 API
- Minor (1.x.0): 新功能, 不破坏旧 API
- Major (2.0.0): 破坏性变更, 需要迁移指南

### 统计 Stats

- 771 passed, 61 skipped, 0 failed
- 19 backends (12 simulators + 7 hardware)
- 75 algorithm templates with examples
- 25 visualization functions
- 106 examples (all bilingual)
- 5/5 tutorial notebooks
- 14 API documentation pages
- 3 real hardware platforms verified

## [0.10.0] — 2026-08-20

进阶案例 + 论文复现 + 双语文档 + FPAA 优化。Advanced examples, paper reproductions, bilingual docs, FPAA optimization.

### 新增 Added

- **5 个进阶工作流案例**：
  - 量子化学工作流（VQE + 噪声 + ZNE + 梯度）
  - 量子优化工作流（QAOA + 多后端对比 + 电路优化）
  - 量子纠错工作流（QEC + 稳定子 + 解码器）
  - 硬件感知编译（分解 + 优化 + SWAP 路由 + 门融合）
  - 量子机器学习（VQC + 参数偏移 + 训练循环）

- **3 个论文复现**：
  - Peruzzo et al. (2014)：VQE 计算 H₂ 基态能量
  - Farhi et al. (2014)：QAOA 求解 MaxCut
  - Grover (1996)：量子搜索算法

- **5 个 example 加双语文档**：bell/grover/vqe/qft/teleportation

- **FPAA 优化**：找最优迭代次数，成功率从 70% 提升到 100%

- **Vale et al. (2024) MCX 分解**：14 CX（原 18 CX，-22%）

- **CP 门**：controlled-phase 参数化门

### 修复 Fixed

- 26 个测试补 importorskip（qiskit/cirq/pennylane/scipy）
- native 后端在 coupling_matrix 测试中误跳过
- test_backend_auto 不接受 native
- Notebook 02/03/04 多个执行错误

### 统计 Stats

- 740 passed, 61 skipped, 0 failed
- 19 backends, 75 algorithms, 25 visualizations
- 80+ examples (including 5 advanced + 3 papers)
- 5/5 tutorial notebooks working

## [0.9.0] — 2026-08-20

MPS 张量网络 + ZX-calculus 编译器 + 全模块 100% 完成。MPS tensor network + ZX-calculus compiler + all modules 100% complete.

### 新增 Added

- **MPS 张量网络模拟器**：完整实现，支持 100+ 比特低纠缠电路。
  **MPS tensor network simulator**: full implementation, supports 100+ qubit low-entanglement circuits.
  - `expectation(pauli)` — 任意 Pauli 字符串期望值 / expectation value for any Pauli string
  - `to_statevector()` — MPS 收缩为态向量 / contract MPS to statevector
  - `entropy(site)` — 二分纠缠熵 / bipartite entanglement entropy
  - `canonicalize()` — 左/右正则化形式 / left/right canonical form
  - `dmrg_sweep()` — Lanczos 2-site DMRG 基态优化 / Lanczos 2-site DMRG ground state optimization
  - `apply_noise()` — 去极化噪声 / depolarizing noise
  - 自定义门支持 / custom gate support

- **ZX-calculus 电路优化**：完整实现，支持电路↔ZX 图互转。
  **ZX-calculus circuit optimization**: full implementation, circuit ↔ ZX-graph conversion.
  - `circuit_to_zx()` — 电路转 ZX 图 / circuit to ZX-graph
  - `optimize_zx()` — 7 种重写规则 / 7 rewrite rules
  - `extract_circuit()` — ZX 图转回电路 / ZX-graph back to circuit
  - 蜘蛛融合 + 恒等消除 + H 边消除 + 补充规则 + 相位复制 + 双代数 + 模式匹配
  - Spider fusion + identity + H-edge + supplementarity + phase copy + bialgebra + pattern matching

- **QEC 模块增强**：
  - CSS code 编码 + Steane 逻辑 X/Z/H 门 / CSS code encoding + Steane logical gates
  - `qec_round_trip()` — 端到端纠错流程 / end-to-end error correction
  - `UnionFindDecoder` — Union-Find 解码器

- **Pulse 模块增强**：
  - `grape_optimize()` — GRAPE 脉冲优化 / GRAPE pulse optimization
  - `krotov_optimize()` — Krotov 脉冲优化 / Krotov pulse optimization

- **Distributed 模块增强**：
  - `schedule_task()` — 量子任务调度 / quantum task scheduling
  - `create_bell_pair()` / `teleport_state()` / `remote_cnot()` — 完整协议

- **ML 模块增强**：
  - `QMLPipeline` — 端到端 QML 流水线 / end-to-end QML pipeline
  - `param_shift_grad()` — 参数偏移梯度 / parameter-shift gradient

- **Compiler 增强**：
  - `optimize_fuse()` — 门融合优化 / gate fusion optimization
  - `optimize_zx_circuit()` — ZX 优化 pass / ZX optimization pass
  - `optimize()` 支持 "zx" 和 "fuse" pass

- **75 个算法 example**：全覆盖 / all algorithms have examples
- **25 种绘图函数**：全功能 / all visualization functions working
- **7 个硬件后端 mock 测试** / 7 hardware backend mock tests

### 修复 Fixed

- `quantum_kernel.py` GateOperation 元组语法 / tuple syntax
- `qbm.py` 能量字典去重 / dedup causing size mismatch
- `qnn.py` Pauli 长度不匹配 / Pauli length mismatch
- `test_coupling_matrix.py` native 后端误跳过 / native backend incorrectly skipped
- ZX graph `neighbors()` 跳过墓碑边 / skip tombstoned edges
- ZX graph `remove_id_spider()` KeyError / KeyError on tombstoned edges

### 统计 Stats

- 735 passed, 61 skipped, 0 failed
- 19 backends (12 simulators + 7 hardware)
- 75 algorithm templates with examples
- 25 visualization functions
- 5/5 tutorial notebooks working

## [0.8.3] — 2026-08-20

测试补齐 + 算法 example 全覆盖 + tutorial notebook 修复。Test coverage boost, full algorithm examples, tutorial fixes.

### 新增 Added

- **56 个算法 example**：覆盖全部 75 个算法函数（之前只有 15 个有 example）。
  **56 algorithm examples**: all 75 algorithm functions now have runnable examples (was 15).

- **CP 门**：新增 controlled-phase 参数化门 `CP(theta)`。
  **CP gate**: new controlled-phase parametric gate `CP(theta)`.

- **测试覆盖提升**：安装可选依赖后 passed 从 431 提升到 615（+184），skipped 从 190 降到 61。
  **Test coverage boost**: passed 431→615 (+184), skipped 190→61 with optional deps installed.

### 修复 Fixed

- **quantum_kernel.py**：`GateOperation("ry", (start + i), ...)` 元组语法错误 → `(start + i,)`。
- **qbm.py**：能量字典去重导致概率数组大小不匹配。
- **qnn.py**：`expectation("Z")` Pauli 长度与 qubit 数不匹配。
- **test_coupling_matrix.py**：native 后端被误当作 Python module 跳过。
- **test_backends.py**：`test_backend_auto` 不接受 native 后端。
- **Notebook 02**：QFT 电路使用不存在的 CP 门。
- **Notebook 03**：readout 校准比特数（2）与电路比特数（1）不匹配。
- **Notebook 04**：`qshow()` 使用不存在的 `method` 参数。

## [0.8.2] — 2026-08-19

学习型调度器 + qif Gray code 分解 + CI 修复。Learning scheduler, qif Gray code decomposition, CI fixes.

### 新增 Added

- **学习型调度器**：`LocalCacheRegistry` 记录每次运行的 `(backend, duration)`，`get_best_backend()` 按平均耗时选最优后端。
  **Learning scheduler**: `LocalCacheRegistry` records per-run timing data, `get_best_backend()` picks fastest backend by average duration.

- **qif Gray code 分解**：`_qif_multi_decompose` 支持通用受控多比特酉分解（4×4 对角块分解）。
  **qif Gray code decomposition**: `_qif_multi_decompose` supports general controlled multi-qubit unitary decomposition (4×4 diagonal block).

- **多比特门酉矩阵**：`_unitary_multi()` 构建 CX/CZ/SWAP/CCX 的酉矩阵。
  **Multi-qubit unitary matrices**: `_unitary_multi()` builds unitary matrices for CX/CZ/SWAP/CCX.

### 修复 Fixed

- **CI lint**：`scripts/benchmark_suite.py` 缩进修复，移除未使用 imports。
  **CI lint**: `benchmark_suite.py` indentation fix, remove unused imports.

- **CI tests**：`tests/test_integration.py` 添加 `pytest` import + qulacs 测试加 `importorskip`。
  **CI tests**: `test_integration.py` add pytest import + qulacs importorskip.

- **.gitignore**：添加 `.venv_test/`。
  **.gitignore**: add `.venv_test/`.

---

## [0.8.0] — 2026-08-19

全量教学 + 7 个新模块 + 7 个硬件后端 + 22-Phase 战略落地。Full tutorials, 7 new modules, 7 hardware backends, and 22-phase roadmap execution.

### 新增 Added

- **量子机器学习框架** `ml/`：ansatz（硬件高效/QAOA/UCCSD）、encoding（振幅/角度/IQP）、optimizer（SPSA/Adam/QNG）、loss（期望值/保真度/交叉熵）、trainer 训练循环。
  **Quantum ML framework** `ml/`: ansatz, encoding, optimizer, loss, trainer.

- **量子纠错框架** `qec/`：BitFlip/PhaseFlip/Shor/Steane/Surface/Color/CSS 码、稳定子形式、MWPM/lookup 解码器。
  **Quantum error correction** `qec/`: stabilizer codes, syndrome extraction, decoders.

- **插件系统** `plugins/`：BackendPlugin/PassPlugin/AlgorithmPlugin 基类 + 注册表。
  **Plugin system** `plugins/`: custom backend/pass/algorithm plugins with registry.

- **脉冲控制** `pulse/`：Gaussian/DRAG/CR 脉冲、Rabi/T1/T2 校准、CPMG/XY-4 解耦序列。
  **Quantum control** `pulse/`: pulse definitions, calibration routines, decoupling sequences.

- **分布式量子计算** `distributed/`：QuantumNetwork 拓扑（star/ring/linear）、EntanglementPair、remote_cnot。
  **Distributed quantum computing** `distributed/`: network topology, entanglement, remote gates.

- **7 个硬件后端骨架**：IBM Quantum / AWS Braket / Azure Quantum / IonQ / Rigetti / Xanadu / QuEra。代码已写好，未经真机测试（已声明 ⚠️ UNTESTED）。
  **7 hardware backend skeletons**: IBM Quantum / AWS Braket / Azure Quantum / IonQ / Rigetti / Xanadu / QuEra. Code written, untested on real hardware (⚠️ UNTESTED disclaimer added).

- **5 个新 example**：teleportation / bb84 / bit_flip_code / vqc / trotter。
  **5 new examples**: quantum teleportation, BB84, bit flip code, VQC, Trotter simulation.

- **QuoNic vs Qiskit benchmark**：`scripts/benchmark_vs_qiskit.py` 代码量 + 速度对比。
  **QuoNic vs Qiskit benchmark**: code size and speed comparison.

- **端到端 benchmark 套件**：`scripts/benchmark_suite.py`（Quantum Volume / 交叉熵 / 算法 benchmark）。
  **End-to-end benchmark suite**: Quantum Volume, cross-entropy, algorithm benchmarks.

- **智能错误提示**：`resolve()` 和 `get_backend()` 加 fuzzy matching（"Did you mean..."）。
  **Smart error messages**: fuzzy matching in `resolve()` and `get_backend()`.

- **README「5 分钟上手」教程**：中英双语 5 步教程。
  **README 5-minute tutorial**: bilingual 5-step guide.

- **22-Phase 战略文档**：`docs/roadmap.md`、`docs/paper-outline.md`、`docs/community.md`、`docs/quonic-hub.md`、`docs/performance-optimization.md`、`docs/scheduler-enhancement.md`、`docs/tech-debt.md`。
  **22-phase strategy docs**: roadmap, paper outline, community plan, hub design, performance, scheduler, tech debt.

- **模块测试**：`test_ml.py`（8 个）、`test_qec.py`（9 个）、`test_plugins.py`（7 个）、`test_pulse.py`（7 个）、`test_distributed.py`（7 个）。
  **Module tests**: QML (8), QEC (9), plugins (7), pulse (7), distributed (7).

### 变更 Changed

- **硬件后端声明**：7 个硬件后端加 `⚠️ UNTESTED` 警告，README 加未测试声明。
  **Hardware backends**: 7 backends marked with ⚠️ UNTESTED disclaimer.

- **PyPI 页面**：description/keywords/classifiers/urls 全面优化。
  **PyPI page**: description, keywords, classifiers, URLs optimized.

### 修复 Fixed

- **QEC decoder**：`n_data` → `n_total`（物理比特数 vs 逻辑比特数）。
  **QEC decoder**: `n_data` → `n_total` (physical vs logical qubit count).

- **pulse calibration**：加缺失的 gate imports（Ry / X / H）。
  **pulse calibration**: added missing gate imports.

---

## [0.7.0] — 2026-08-19

高阶用户功能 + 不兼容修复 + 功能耦合测试。Advanced user features, incompatibility fixes, and feature coupling tests.

### 新增 Added

- **自定义门** `Gate.from_matrix(name, matrix)`：任意酉矩阵定义自定义门，支持 native/qulacs/qiskit/cirq/pennylane 全后端。
  **Custom gates** `Gate.from_matrix(name, matrix)`: define gates from arbitrary unitary matrices, works on all backends.

- **态矢量访问** `run(circuit, return_state=True)`：返回 `StateVector` 对象，支持 `amplitude()`/`probabilities()`/`expectation()`/`fidelity()`。
  **Statevector access** `run(circuit, return_state=True)`: returns `StateVector` with `amplitude()`/`probabilities()`/`expectation()`/`fidelity()`.

- **混合态** `MixedState`：噪声路径返回密度矩阵封装，支持 `probabilities()`/`expectation()`/`purity()`。
  **MixedState**: noise path returns density-matrix wrapper with `probabilities()`/`expectation()`/`purity()`.

- **梯度 API** `param_shift()` / `numerical_gradient()`：参数平移法和有限差分梯度计算。
  **Gradient API** `param_shift()` / `numerical_gradient()`: parameter-shift and finite-difference gradients.

- **电路内省**：`Circuit` 支持 `__iter__`/`__len__`/`__repr__`/`copy()`/`filter()`/`slice()`/`inverse()`/`__add__`。
  **Circuit introspection**: `Circuit` supports iteration, repr, copy, filter, slice, inverse, concatenation.

- **电路序列化**：`to_json()`/`from_json()`/`to_qasm3()`/`to_dict()`/`from_dict()`。
  **Circuit serialization**: JSON and OpenQASM 3.0 import/export.

- **电路分析** `analyze(circuit)`：返回 `CircuitReport`（depth/gate_count/cx_count/fidelity_estimate）。
  **Circuit analysis** `analyze(circuit)`: returns `CircuitReport` with depth, gate counts, fidelity estimate.

- **自定义优化 pass**：`optimize()` 接受 callable 函数作为 pass。
  **Custom optimization passes**: `optimize()` accepts callable functions.

- **参数化电路** `Parameter` + `bind_params()` / `bind_batch()`：符号参数绑定。
  **Parameterized circuits** `Parameter` + `bind_params()` / `bind_batch()`.

- **批量执行** `run_batch(circuits, backend)`：多电路批量运行。
  **Batch execution** `run_batch(circuits, backend)`.

- **数据编码** `angle_encode()` / `amplitude_encode()`：经典数据编码为量子态。
  **Data encoding** `angle_encode()` / `amplitude_encode()`.

- **单步执行** `StepExecutor(circuit)`：逐步执行，每步返回态矢量。
  **Step-by-step execution** `StepExecutor(circuit)`: execute gate-by-gate, inspect state at each step.

- **cwhile + GPU 自动 groverize**：`run(method="gpu")` 自动将 cwhile 编译成静态电路再跑 GPU。
  **cwhile + GPU auto-groverize**: `run(method="gpu")` auto-compiles cwhile to static circuit for GPU.

- **翻译器自定义门**：qiskit/cirq/pennylane 翻译器遇到自定义门时自动查 `_GATE_REGISTRY` 翻译。
  **Translator custom gate support**: qiskit/cirq/pennylane translators look up `_GATE_REGISTRY` for custom gates.

- **功能耦合测试**：27 个集成测试 + 27 个耦合矩阵测试 + 15 个嵌套测试 = 69 个跨功能测试。
  **Feature coupling tests**: 27 integration + 27 coupling matrix + 15 nesting = 69 cross-feature tests.

### 修复 Fixed

- **`_run_dynamic` 签名统一**：所有后端的 `_run_dynamic` 统一接受 `return_state` 参数。
  **`_run_dynamic` signature unification**: all backends accept `return_state` parameter.

- **密度矩阵自定义门**：`_density.py` 支持 `_apply_custom`（自定义门在密度矩阵引擎上执行）。
  **Density matrix custom gates**: `_density.py` supports `_apply_custom`.

- **qif 自定义门**：`_unitary()` 支持从 `Gate.matrix` 取矩阵。
  **qif custom gates**: `_unitary()` reads from `Gate.matrix`.

---

## [0.6.0] — 2026-08-19

电路优化 + qif 多比特/嵌套 + API 文档 + GPU 基准 + 算法模板深化。Circuit
optimization, qif multi-qubit/nesting, API docs, GPU benchmarks, and algorithm
template upgrades.

### 新增 Added

- **电路优化 pass** `optimize()`：门消减（X·X=I）、交换重排（H·CX·H→CX）、
  模式替换（CX·CX·CX→SWAP）。统一入口 `optimize(circuit, passes=("cancel","commute","peephole"))`。
  **Circuit optimization** `optimize()`: gate cancellation, commutation reordering,
  peephole pattern replacement.

- **qif 多比特门**：`qif(0).then(CX, 1, 2)` = Toffoli，`qif(0).then(SWAP, 1, 2)` = Fredkin。
  `controlled()` 同步支持多比特门。
  **qif multi-qubit gates**: controlled-CX (Toffoli), controlled-SWAP (Fredkin), controlled-CZ (MCZ).

- **qif 嵌套**：`then_ops(inner).else_ops(outer)` 接受子电路操作列表，支持嵌套 qif。
  **qif nesting**: `then_ops()`/`else_ops()` accept sub-circuit operation lists for nested qif.

- **`requires_grad`**：`Circuit.requires_grad` 属性，调度器自动选 pennylane/tensorcircuit。
  `recommend_backend_autodiff()` 自动选最优 autodiff 后端。
  **`requires_grad`**: Circuit attribute for autodiff-aware scheduling.

- **MkDocs API 文档**：`mkdocs.yml` + `docs/api/`（9 个 API 页面）+ `docs/tutorials/`（5 个教程）。
  **MkDocs API documentation**: full docs site with API reference and tutorials.

- **Jupyter 教程**：`01_basics.ipynb` 到 `05_advanced.ipynb`，5 个教程 notebook。
  **Jupyter tutorials**: 5 notebooks from basics to advanced features.

- **算法模板深化**：QCNN（可配置层数 + 训练循环）、QGAN（对抗训练 + 梯度更新）、
  表面码（可配置距离 + syndrome 解码）。
  **Algorithm template upgrades**: QCNN, QGAN, surface code upgraded from minimal demos to
  complete implementations.

- **GPU 基准数据**：RTX 2070 上 qulacs + cupy benchmark 数据写入 `benchmarks.json`。
  **GPU benchmark data**: measured on RTX 2070, stored in `benchmarks.json`.

- **`pyproject.toml`**：新增 `docs` 和 `all-sim` optional dependencies。
  **pyproject.toml**: new `docs` and `all-sim` optional dependencies.

### 变更 Changed

- **`recommend_backend_gpu()`**：读 measured 数据，fallback 硬编码阈值。
  **`recommend_backend_gpu()`**: reads measured data from benchmarks.json, falls back to hardcoded thresholds.

- **`scheduler/capabilities.py`**：pennylane/tensorcircuit 标记 `autodiff` 能力。
  **Scheduler capabilities**: pennylane/tensorcircuit tagged with `autodiff` capability.

### 修复 Fixed

- **qif `_check_branch`**：cif 仍限制单比特门，qif 放开到多比特。
  **qif `_check_branch`**: cif still restricts to single-qubit, qif allows multi-qubit.

- **`controlled()` 目标数量校验**：`controlled(CX, 0, 1)` 现在抛清晰错误（CX 需要 2 个目标）。
  **`controlled()` target count validation**: clear error when target count doesn't match gate.

---

## [0.5.0] — 2026-08-19

GPU 智能调度 + 7 个新后端 + 误差缓解增强 + 大量 bug 修复。GPU smart scheduling,
7 new backends, error mitigation enhancements, and numerous bug fixes.

### 新增 Added

- **GPU 智能调度** `method="gpu"` + `recommend_backend_gpu()`：根据电路特征（纠缠级别 /
  经典控制流 / 电路大小）自动选择最优 GPU 后端。CuPy 通用引擎作为兜底。
  **GPU smart scheduling** `method="gpu"` + `recommend_backend_gpu()`: automatically selects
  the best GPU backend based on circuit features (entanglement / classical control / size).
  CuPy universal engine as fallback.

- **CuPy 通用引擎** `backends/cupy_engine.py`：基于 CuPy（numpy GPU drop-in）的状态向量
  模拟器，支持噪声注入和经典控制流，自动检测 CUDA/ROCm，无 GPU 时 fallback numpy。
  **CuPy universal engine**: CuPy-based statevector simulator with noise and classical control
  flow support, auto-detects CUDA/ROCm, falls back to numpy when no GPU.

- **7 个后端 GPU 变体**：qulacs / tensorcircuit / pennylane / qiskit / mindquantum / qpanda /
  cudaq 各自接入原生 GPU（QuantumStateGpu / JAX / lightning.gpu / Aer GPU 等）。
  **7 backend GPU variants**: each backend connects to its native GPU implementation.

- **能力矩阵** `_CAPABILITIES`：每个后端声明支持的特性（noise / ctrl / mid_measure / gpu），
  `run()` 入口检查，不支持时抛统一错误。
  **Capability matrix**: each backend declares supported features, `run()` checks at entry.

- **电路特征扩展** `scheduler/features.py`：新增 `entanglement`（纠缠级别）和 `has_ctrl`
  （经典控制流）特征，用于 GPU 调度决策。
  **Circuit features**: new `entanglement` and `has_ctrl` features for GPU scheduling.

- **cirq / pennylane 多比特 creg**：`cmeasure` 支持 `bit > 0`，`cif` 支持 `CRegCondition`
  （多比特寄存器相等判据）。
  **cirq / pennylane multi-bit creg**: `cmeasure` supports `bit > 0`, `cif` supports
  `CRegCondition` (multi-bit register equality).

- **新 example**：`gpu_demo/`（GPU 加速演示）、`error_mitigation/`（误差缓解演示）。
  **New examples**: `gpu_demo/` (GPU acceleration), `error_mitigation/` (ZNE + readout calibration).

- **新测试**：`test_gpu.py`（GPU 分发 / CuPy 引擎 / 能力矩阵 / 智能调度）。
  **New tests**: GPU dispatch, CuPy engine, capability matrix, smart scheduling.

### 变更 Changed

- **TensorCircuit numpy patch 隔离**：`_tc_compat()` 上下文管理器，patch 入口 restore 出口，
  不再全局污染 numpy。
  **TensorCircuit numpy patch isolation**: context manager patches on entry, restores on exit.

- **向量化 `_apply_readout_noise`**：从逐 shot Python 循环改为 numpy tensordot 张量收缩。
  **Vectorized readout noise**: numpy tensordot instead of per-shot Python loop.

- **CuPy 多比特门向量化**：CX / CCX / CZ / CP / SWAP / MCZ 全部用 numpy 索引替代
  Python 循环。
  **CuPy multi-qubit gate vectorization**: all gates use numpy indexing instead of Python loops.

- **`_i18n.py`**：新增 `err.no_gpu` / `err.gpu_missing` / `err.gpu_fallback_failed` 错误消息。
  **i18n**: new GPU error messages.

- **`pyproject.toml`**：新增 `gpu = ["cupy-cuda12x"]` 可选依赖。
  **pyproject.toml**: new `gpu` optional dependency.

### 修复 Fixed

- **ZNE success metric 外推被忽略**：`zne.py` success metric 路径硬编码线性外推，现改为
  使用 `extrapolation` 参数 + 成功概率 clamp 到 [0,1]。
  **ZNE success metric extrapolation ignored**: now uses the `extrapolation` parameter
  and clamps success probability to [0,1].

- **读出校准混淆矩阵奇异**：`readout.py` 捕获 `LinAlgError`，fallback Tikhonov 正则化。
  **Readout calibration singular matrix**: falls back to Tikhonov regularization.

- **读出校准无比特数校验**：`apply()` 加 n>20 时显存用量 warning。
  **Readout calibration no qubit limit**: `apply()` warns when n>20.

- **qshow_all 多进程 + CUDA**：加 warning（CUDA context 不可跨进程继承）。
  **qshow_all multiprocessing + CUDA**: added warning about CUDA context.

- **GPU 显存预检查**：CuPy 引擎在分配前检查可用显存，不足时抛 MemoryError。
  **GPU memory pre-check**: CuPy engine checks available memory before allocation.

- **CuPy fallback 错误信息**：捕获 CuPy 异常，重新抛出带原始后端名的错误。
  **CuPy fallback error messages**: re-raises with original backend name.

引擎后端全面升级、77 个算法模板、并行执行支持。Major release: engine backend upgrades,
77 algorithm templates, and parallel execution support.

### 新增 Added

- **6 个引擎后端升级为完整后端**：Qulacs / TensorCircuit / CUDA-Q / MindQuantum / QPanda3 / Cqlib 现支持密度矩阵模拟、噪声注入（退极化通道）、经典控制流（cif/cmeasure/cwhile 逐 shot 动态执行）。
  **6 engine backends upgraded to full-featured**: Qulacs / TensorCircuit / CUDA-Q / MindQuantum / QPanda3 / Cqlib now support density matrix simulation, noise injection (depolarizing channels), and classical control flow (cif/cmeasure/cwhile via per-shot dynamic execution).

- **EngineBackend v2 架构**：`run()` 三路分发（clean SV / noisy DM / dynamic per-shot），新增可选钩子 `_create_dm` / `_apply_noise_after_gate` / `_measure_qubit`。
  **EngineBackend v2 architecture**: `run()` three-way dispatch with optional hooks for density matrix, noise, and mid-circuit measurement.

- **`qshow_all(backends)`**：同一电路在多个后端上并行执行，每个后端独立进程。
  **`qshow_all(backends)`**: run the same circuit on multiple backends in parallel (each in a separate process).

- **`run_circuits(builders, backend)`**：多个不同电路并行执行，每个构建函数独立进程。
  **`run_circuits(builders, backend)`**: run different circuits in parallel (each builder in a separate process).

- **77 个算法模板**（从 6 个扩展），覆盖 10 大领域：
  **77 algorithm templates** (up from 6), covering 10 domains:
  - 基石算法（9）：QFT、Deutsch-Jozsa、Bernstein-Vazirani、Simon、SWAP 测试、Hadamard 测试、振幅放大、振幅估计、QPE
    Foundational (9): QFT, Deutsch-Jozsa, Bernstein-Vazirani, Simon, SWAP test, Hadamard test, amplitude amplification, amplitude estimation, QPE
  - 搜索优化（7）：QAOA 通用/TSP/MIS/背包、Grover、量子计数、量子随机行走、量子退火
    Search & Optimization (7): QAOA generic/TSP/MIS/Knapsack, Grover, quantum counting, quantum walk, quantum annealing
  - 量子化学（8）：VQE、哈密顿量导入（OpenFermion/PennyLane/手写）、Trotter、哈密顿量模拟、动态模拟、费米子映射、QSP、分子 VQE
    Quantum Chemistry (8): VQE, Hamiltonian import (OpenFermion/PennyLane/string), Trotter, Hamiltonian simulation, dynamics simulation, fermion mapping, QSP, molecular VQE
  - 线性代数（6）：HHL、矩阵求逆、特征值求解、PDE/ODE 求解、数据拟合
    Linear Algebra (6): HHL, matrix inversion, eigenvalue solver, PDE/ODE solver, data fitting
  - 通信密码（6）：隐形传态、BB84、E91、超密编码、Shor、离散对数
    Communication & Crypto (6): teleportation, BB84, E91, superdense coding, Shor, discrete log
  - 混合算法（7）：VQC、量子核方法、QNG、VQR、QNN、QSVM、量子退火混合
    Hybrid (7): VQC, quantum kernel, QNG, VQR, QNN, QSVM, quantum annealing hybrid
  - 量子纠错（9）：比特/相位翻转码、Shor 9 比特、Steane 7 比特、稳定子、syndrome、表面码、颜色码、容错门
    Error Correction (9): bit/phase flip code, Shor 9-qubit, Steane 7-qubit, stabilizer, syndrome, surface code, color code, FT gates
  - 统计采样（3）：量子蒙特卡洛、拒绝采样、贝叶斯推理
    Statistical (3): quantum Monte Carlo, rejection sampling, Bayesian inference
  - 代数（3）：隐藏子群、格问题、椭圆曲线
    Algebraic (3): hidden subgroup, lattice SVP, elliptic curve
  - 前沿演示（10）：QCNN、QGNN、分布式 QAOA、QTransformer、QRL、QTDA、QPCA、聚类、QGAN、QBM
    Cutting-edge (10): QCNN, QGNN, distributed QAOA, QTransformer, QRL, QTDA, QPCA, clustering, QGAN, QBM

- **cswap 门**：新增原生 CSWAP（Fredkin）门到 IR + `translators/cswap.py` 翻译器。
  **cswap gate**: native CSWAP (Fredkin) gate added to IR + translator.

- **新增测试**：`test_engine_backends.py`、`test_engine_noise.py`、`test_engine_ctrl.py`、`test_foundational_algorithms.py`、`test_search_algorithms.py`。
  **New tests**: engine backends, noise injection, classical control flow, foundational algorithms, search algorithms.

### 变更 Changed

- **TensorCircuit numpy 兼容**：monkey-patch `np.reshape` 和 `np.ComplexWarning` 以兼容 numpy 2.x。
  **TensorCircuit numpy compat**: monkey-patch `np.reshape` and `np.ComplexWarning` for numpy 2.x.

- **QPanda3 API 适配**：`CCX` → `TOFFOLI`、`directly_run` → `CPUQVM` + `QProg` + `measure()`。
  **QPanda3 API adaptation**: `CCX` → `TOFFOLI`, `directly_run` → `CPUQVM` + `QProg` + `measure()`.

- **Cqlib 后端**：确认无本地模拟器，`_sample` 抛出清晰错误。
  **Cqlib backend**: confirmed no local simulator; `_sample` raises clear error.

- **Scheduler 能力矩阵**：新增 `BACKEND_CAPABILITIES` 映射。
  **Scheduler capabilities**: new `BACKEND_CAPABILITIES` mapping.

- **.gitignore**：新增 `.venv312` 和 `.mimocode/`。
  **.gitignore**: added `.venv312` and `.mimocode/`.

### 移除 Removed

- **Paddle Quantum 后端**：因 paddle 3.x 依赖冲突（不支持 complex matmul）彻底移除。
  **Paddle Quantum backend**: removed entirely due to paddle 3.x dependency conflicts (no complex matmul support).

### 修复 Fixed

- **Qulacs `cp` 门**：从错误的 CZ+U1 近似改为正确的 CNOT+P+CNOT 分解。
  **Qulacs `cp` gate**: fixed incorrect CZ+U1 approximation to proper CNOT+P+CNOT decomposition.

- **CUDA-Q `p`/`cp` 门**：修复 `p` 错误近似为 `rz`、`cp` 错误近似为单比特 `rz`。
  **CUDA-Q `p`/`cp` gates**: fixed incorrect `p` ≈ `rz` and `cp` ≈ single-qubit `rz`.

- **QPanda3 `p` 门**：修复 `p` 错误近似为 `rz`。
  **QPanda3 `p` gate**: fixed incorrect `p` ≈ `rz`.

- **读出噪声**：修复 `_apply_readout_noise` 对同一比特串所有 shot 应用相同翻转的 bug。
  **Readout noise**: fixed bug where all shots with the same bitstring got the same flip.

---

## [0.3.0] — 2026-08-17

补齐五块缺口并加入误差缓解：多比特经典寄存器、路由编译、ZNE、读出校准，以及四个新
example。This release fills five gaps and adds error mitigation: multi-bit classical
registers, route-aware compilation, ZNE, readout calibration, and four new examples.

### 新增 Added

- **多比特经典寄存器** `creg(name, width=N)`；`cwhile(reg, until=v)` / `cif(reg, v)`
  支持整数值或比特串判据；`groverize()` 推广到 N 比特成功态。
  **Multi-bit classical registers** `creg(name, width=N)`; `cwhile`/`cif` accept integer
  or bitstring criteria; `groverize()` generalizes to N-bit success states.

- **路由编译** `compile(circuit, coupling_map, route=True)` 自动 `decompose` + `route_swaps`。
  **Route-aware compilation** `compile(..., route=True)` decomposes then routes onto the
  coupling map.

- **ZNE 误差缓解** `zne()` / `fold()`：全局酉折叠 + 线性或指数（三参数）外推，成功率与
  期望值两种指标；`plot_zne()` 可视化。
  **ZNE** `zne()` / `fold()`: global unitary folding with linear or exponential (3-param)
  extrapolation, success and expectation metrics; `plot_zne()` visualization.

- **读出校准** `calibrate()` / `ReadoutCalibration`：逐比特（张量积）与关联（完整 2ⁿ 矩阵）
  两种混淆矩阵模型。
  **Readout calibration** `calibrate()` / `ReadoutCalibration`: per-qubit (tensor-product)
  and correlated (full 2ⁿ matrix) confusion-matrix models.

- **`DensityMatrixEngine.expectation()`** 计算泡利串可观测量期望值。
  **`DensityMatrixEngine.expectation()`** for Pauli-string observables.

- **cmeasure 支持 cirq / pennylane** 后端翻译。
  **cmeasure translation** for the cirq and pennylane backends.

- **新 example**：`creg_multi`、`groverize`、`hardware_compile`、`qi_hardware`。
  **New examples**: `creg_multi`, `groverize`, `hardware_compile`, `qi_hardware`.

### 变更 Changed

- 后端门翻译重构为共享 `translators/` 模块。
  Backend gate translation factored into a shared `translators/` module.

- `NoiseModel` 新增 `readout` 字段（测量比特翻转）。
  `NoiseModel` gained a `readout` (measurement bit-flip) field.

- qi 后端 job 超时 30 → 60 分钟（Tuna-17 排队）。
  qi backend job timeout relaxed 30 → 60 minutes (Tuna-17 queue).

- qi 依赖冲突指南默认改走 venv 方案。
  qi dependency-conflict guide now defaults to the venv workaround.
