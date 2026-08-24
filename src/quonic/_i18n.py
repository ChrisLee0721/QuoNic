"""User-facing string localization (English default, Chinese optional).

Runtime messages — error text, terminal reports, the interactive setup guide —
are centralized here. Source-code docstrings and comments are NOT localized
(they stay English; see the surrounding docs).

Language selection, in priority order:

1. ``set_language("zh")`` called at runtime.
2. The ``QUONIC_LANG`` environment variable (``"en"`` or ``"zh"``).

``tr(key, **fmt)`` looks up the current language's template and interpolates
the given keyword arguments via ``str.format``. Missing keys fall back to
English, then to the raw key, so an untranslated message never crashes.
"""

from __future__ import annotations

import os
from typing import Any

_LANGUAGES = ("en", "zh")

# fmt: off
_MESSAGES: dict[str, dict[str, str]] = {
    # --------------------------------------------------------- setup guide
    "setup.default_name": {
        "en": "this backend",
        "zh": "该后端",
    },
    "setup.login_fallback": {
        "en": "login",
        "zh": "登录",
    },
    "setup.configuring": {
        "en": "Configuring {name} backend (one-time, ~1 min)...",
        "zh": "正在配置 {name} 后端（一次性，约 1 分钟）...",
    },
    "setup.missing_dep": {
        "en": "\n[1/3] Missing dependency {pkg}",
        "zh": "\n[1/3] 缺少依赖 {pkg}",
    },
    "setup.will_run": {
        "en": "      Will run: pip install '{install}'",
        "zh": "      将执行：pip install '{install}'",
    },
    "setup.press_enter_install": {
        "en": "      Press Enter to install, type n to skip",
        "zh": "      回车开始安装，输入 n 跳过",
    },
    "setup.conflict_detected": {
        "en": "\n[2/3] Detected {pkg} {installed}, conflicts with requirement {constraint}",
        "zh": "\n[2/3] 检测到 {pkg} {installed}，与要求 {constraint} 冲突",
    },
    "setup.need_login": {
        "en": "\n[3/3] Login required (runs {cmd}, browser authorization)",
        "zh": "\n[3/3] 需要登录（运行 {cmd}，浏览器授权）",
    },
    "setup.press_enter_login": {
        "en": "      Press Enter to log in, type n to skip",
        "zh": "      回车开始登录，输入 n 跳过",
    },
    "setup.logged_in": {
        "en": "      OK: logged in",
        "zh": "      ✓ 已登录",
    },
    "setup.login_incomplete": {
        "en": "      Login incomplete, please run {cmd} manually",
        "zh": "      登录未完成，请手动运行 {cmd}",
    },
    "setup.ready": {
        "en": "\n{name} backend is ready.",
        "zh": "\n✓ {name} 后端已就绪。",
    },
    "setup.not_ready": {
        "en": "\nConfiguration incomplete, retry the guide later.",
        "zh": "\n配置未完成，可稍后重试引导。",
    },
    "setup.how_to_handle": {
        "en": "      How to handle:",
        "zh": "      处理方式：",
    },
    "setup.opt_venv": {
        "en": "        Enter = create isolated venv (recommended, avoids conflicts)",
        "zh": "        回车 = 创建独立虚拟环境（推荐，隔离冲突）",
    },
    "setup.opt_downgrade": {
        "en": "        2    = downgrade {pkg} to {constraint} (affects current env)",
        "zh": "        2    = 回退 {pkg} 到 {constraint}（影响当前环境）",
    },
    "setup.opt_skip": {
        "en": "        3    = skip, I'll handle it myself",
        "zh": "        3    = 跳过，我自行处理",
    },
    "setup.prompt_input": {
        "en": "      Please enter",
        "zh": "      请输入",
    },
    "setup.will_run_pip": {
        "en": "      Will run: pip install '{pkg}{constraint}'",
        "zh": "      执行：pip install '{pkg}{constraint}'",
    },
    "setup.venv_create": {
        "en": "      Create isolated venv (avoids conflicts):",
        "zh": "      创建独立虚拟环境（隔离冲突）：",
    },
    "setup.venv_rerun": {
        "en": "      Then rerun your program in the activated environment.",
        "zh": "      然后在激活的环境里重新运行你的程序。",
    },
    "setup.confirm_hint": {
        "en": " [Enter=yes / n=no] ",
        "zh": " [回车=是 / n=否] ",
    },
    "setup.menu_suffix": {
        "en": ": ",
        "zh": "：",
    },

    # ------------------------------------------------------------- qshow
    "show.empty_circuit": {
        "en": "(current circuit is empty; build it first with qgate(...))",
        "zh": "（当前电路为空，请先用 qgate(...) 构建电路）",
    },
    "show.noise_cost": {
        "en": "Note: depolarizing noise uses density_matrix (4^n resources); "
              "reference machine exceeds budget at n>={infeasible}. "
              "Current n={n}, may be slow or run out of memory.",
        "zh": "提示：去极化噪声走 density_matrix（4^n 资源），"
              "参考机实测 n>={infeasible} 时已超预算。当前 n={n}，可能很慢或内存不足。",
    },
    "show.circuit_resources": {
        "en": "Circuit resources:",
        "zh": "电路资源:",
    },
    "show.gate_count": {
        "en": "  gates: {n}",
        "zh": "  门数: {n}",
    },
    "show.depth": {
        "en": "  depth: {n}",
        "zh": "  深度: {n}",
    },
    "show.qubit_count": {
        "en": "  qubits: {n}",
        "zh": "  量子比特: {n}",
    },
    "show.backend_header": {
        "en": "backend: {name} | ",
        "zh": "后端: {name} | ",
    },
    "show.shots": {
        "en": "shots: {shots}",
        "zh": "shots: {shots}",
    },
    "show.result": {
        "en": "Result:",
        "zh": "结果:",
    },

    # ---------------------------------------------------------- benchmark
    "bench.capabilities": {
        "en": "Capability matrix:",
        "zh": "能力矩阵：",
    },
    "bench.performance": {
        "en": "\nPerformance data:",
        "zh": "\n性能数据：",
    },
    "bench.decision": {
        "en": "\nDerived decision table:",
        "zh": "\n推导的决策表：",
    },
    "bench.general": {
        "en": "\nHigh-treewidth non-Clifford (statevector checkpoints):",
        "zh": "\n高树宽非 Clifford（statevector 验证点）：",
    },
    "bench.noise": {
        "en": "\nNoise (density_matrix, 4^n cost):",
        "zh": "\n噪声（density_matrix，4^n 成本）：",
    },
    "bench.infeasible": {
        "en": "  Infeasible threshold (>{budget}s): {infeasible_n}",
        "zh": "  不可行阈值（>{budget}s）：{infeasible_n}",
    },
    "bench.written": {
        "en": "\nWritten to {output}",
        "zh": "\n已写入 {output}",
    },

    # -------------------------------------------------------- setup errors
    "err.parse_constraint": {
        "en": "Cannot parse version constraint '{constraint}'",
        "zh": "无法解析版本约束 '{constraint}'",
    },
    "err.need_install": {
        "en": "Using {name} backend requires installing {pkg}:\n"
              "    pip install '{install}'\n"
              "or run python -m quonic.setup for guided setup",
        "zh": "使用 {name} 后端需要安装 {pkg}：\n"
              "    pip install '{install}'\n"
              "或运行 python -m quonic.setup 一键引导配置",
    },
    "err.conflict": {
        "en": "{name} backend has version conflict: {pkg} currently {installed}, "
              "requires {constraint}.\n"
              "Run python -m quonic.setup to resolve (downgrade / venv / skip)",
        "zh": "{name} 后端存在版本冲突：{pkg} 当前 {installed}，要求 {constraint}。\n"
              "可运行 python -m quonic.setup 引导处理（回退 / 建虚拟环境 / 跳过）",
    },
    "err.need_login": {
        "en": "Using {name} backend requires one-time login:\n"
              "    run {cmd}\n"
              "or run python -m quonic.setup for guided setup",
        "zh": "使用 {name} 后端需要先登录（一次性）：\n"
              "    运行 {cmd}\n"
              "或运行 python -m quonic.setup 一键引导配置",
    },

    # ------------------------------------------------------- backend errors
    "err.unknown_backend": {
        "en": "Unknown backend '{name}'. Available engines: {engines}",
        "zh": "未知的后端 '{name}'。当前可用引擎：{engines}",
    },
    "err.no_method_support": {
        "en": "No backend supports method '{method}'",
        "zh": "没有任何后端支持方法 '{method}'",
    },
    "err.device_alias_conflict": {
        "en": "backend='{backend}' already specifies device '{alias_device}', "
              "cannot also pass device='{device}'",
        "zh": "backend='{backend}' 已指定设备 '{alias_device}'，"
              "不能同时再传 device='{device}'",
    },
    "err.device_only_qi": {
        "en": "device is only valid for backend='qi' (Quantum Inspire hardware/cloud "
              "simulator); current backend='{backend}' is a local simulator",
        "zh": "device 参数仅对 backend='qi'（Quantum Inspire 真机/云模拟器）有效；"
              "当前 backend='{backend}' 是本地模拟器，无需指定设备",
    },
    "err.qi_noise": {
        "en": "qi backend runs real hardware and cannot inject noise; use the qiskit "
              "backend (Aer density_matrix) to simulate depolarizing noise",
        "zh": "qi 后端运行真实硬件，无法注入噪声 noise；"
              "请用 qiskit 后端（Aer density_matrix）模拟去极化噪声",
    },
    "err.qi_cwhile": {
        "en": "qi backend does not support cwhile (classical feedback loop); real "
              "hardware cannot read back per-shot, use the native backend",
        "zh": "qi 后端不支持 cwhile（经典反馈循环）；"
              "真实硬件无法逐 shot 动态回读，请用 native 后端",
    },
    "err.qi_cif": {
        "en": "qi backend does not support cif (mid-circuit measurement + classical "
              "branch); superconducting hardware has no mid-circuit feedback, use "
              "qiskit or native backend",
        "zh": "qi 后端不支持 cif（中段测量 + 经典分支）；"
              "超导真机无中段测量反馈，请用 qiskit 或 native 后端",
    },
    "err.native_method": {
        "en": "native backend does not support method '{method}', available: {engines}",
        "zh": "native 后端不支持方法 '{method}'，可用：{engines}",
    },
    "err.native_ctrl": {
        "en": "native backend classical control flow (cif/creg/cwhile) only supports "
              "statevector / density_matrix methods, current method='{method}'",
        "zh": "native 后端的经典控制流（cif/creg/cwhile）仅支持 "
              "statevector / density_matrix 方法，当前 method='{method}'",
    },
    "err.cwhile_limit": {
        "en": "cwhile loop exceeded safety limit (100000 iterations); condition "
              "creg={creg!r} may never be satisfied",
        "zh": "cwhile 循环超过安全上限（100000 次），条件 creg={creg!r} 可能一直未满足",
    },
    "err.cirq_missing": {
        "en": "cirq backend requires cirq:\n"
              "    pip install 'quonic[cirq]'\n"
              "or: pip install cirq",
        "zh": "使用 cirq 后端需要安装 cirq：\n"
              "    pip install 'quonic[cirq]'\n"
              "或： pip install cirq",
    },
    "err.cirq_ctrl": {
        "en": "cirq backend does not support cwhile (classical feedback loop); "
              "use native backend",
        "zh": "cirq 后端暂不支持 cwhile（经典反馈循环）；请用 native 后端",
    },
    "err.cirq_gate": {
        "en": "Cirq backend does not support gate '{name}'",
        "zh": "Cirq 后端暂不支持门 '{name}'",
    },
    "err.qiskit_missing": {
        "en": "qiskit backend requires qiskit and qiskit-aer:\n"
              "    pip install 'quonic[qiskit]'\n"
              "or: pip install qiskit qiskit-aer",
        "zh": "使用 qiskit 后端需要安装 qiskit 和 qiskit-aer：\n"
              "    pip install 'quonic[qiskit]'\n"
              "或： pip install qiskit qiskit-aer",
    },
    "err.qiskit_cwhile": {
        "en": "qiskit backend does not support cwhile (classical feedback loop); "
              "use native backend",
        "zh": "qiskit 后端暂不支持 cwhile（经典反馈循环）；请用 native 后端",
    },
    "err.qiskit_gate": {
        "en": "Qiskit backend does not support gate '{name}'",
        "zh": "Qiskit 后端暂不支持门 '{name}'",
    },
    "err.pennylane_missing": {
        "en": "pennylane backend requires pennylane:\n"
              "    pip install 'quonic[pennylane]'\n"
              "or: pip install pennylane",
        "zh": "使用 pennylane 后端需要安装 pennylane：\n"
              "    pip install 'quonic[pennylane]'\n"
              "或： pip install pennylane",
    },
    "err.pennylane_ctrl": {
        "en": "pennylane backend does not support cwhile (classical feedback loop); "
              "use native backend",
        "zh": "pennylane 后端暂不支持 cwhile（经典反馈循环）；请用 native 后端",
    },
    "err.pennylane_gate": {
        "en": "PennyLane backend does not support gate '{name}'",
        "zh": "PennyLane 后端暂不支持门 '{name}'",
    },

    # -- engine backend errors (shared across qulacs / tensorcircuit / cudaq / ...) --
    "err.engine_ctrl": {
        "en": "{name} backend does not support classical control flow "
              "(cif/cmeasure/cwhile); use native or qiskit backend",
        "zh": "{name} 后端暂不支持经典控制流（cif/cmeasure/cwhile）；"
              "请用 native 或 qiskit 后端",
    },
    "err.engine_noise": {
        "en": "{name} backend does not yet support noise injection; "
              "use qiskit (Aer density_matrix) or native backend for noise simulation",
        "zh": "{name} 后端暂不支持噪声注入；"
              "请用 qiskit（Aer density_matrix）或 native 后端模拟噪声",
    },
    "err.engine_no_dm": {
        "en": "{name} backend does not support density matrix simulation",
        "zh": "{name} 后端不支持密度矩阵模拟",
    },
    "err.engine_no_measure": {
        "en": "{name} backend does not support mid-circuit measurement",
        "zh": "{name} 后端不支持中段测量",
    },
    "err.engine_no_sv": {
        "en": "{name} backend does not support statevector extraction",
        "zh": "{name} 后端不支持态矢量提取",
    },
    "err.no_gpu": {
        "en": "{name} backend does not support GPU acceleration; "
              "use qulacs, tensorcircuit, or cupy backend",
        "zh": "{name} 后端不支持 GPU 加速；"
              "请用 qulacs、tensorcircuit 或 cupy 后端",
    },
    "err.gpu_missing": {
        "en": "GPU engine requires cupy and a CUDA/ROCm GPU:\n"
              "    pip install 'quonic[gpu]'\nor: pip install cupy-cuda12x",
        "zh": "GPU 引擎需要 cupy 和 CUDA/ROCm GPU：\n"
              "    pip install 'quonic[gpu]'\n或： pip install cupy-cuda12x",
    },
    "err.gpu_fallback_failed": {
        "en": "{backend} GPU failed, CuPy fallback also failed: {error}",
        "zh": "{backend} GPU 失败，CuPy 兜底也失败：{error}",
    },
    "err.qulacs_missing": {
        "en": "qulacs backend requires qulacs:\n"
              "    pip install 'quonic[qulacs]'\nor: pip install qulacs",
        "zh": "使用 qulacs 后端需要安装 qulacs：\n"
              "    pip install 'quonic[qulacs]'\n或： pip install qulacs",
    },
    "err.qulacs_gate": {
        "en": "qulacs backend does not support gate '{name}'",
        "zh": "qulacs 后端暂不支持门 '{name}'",
    },
    "err.tensorcircuit_missing": {
        "en": "tensorcircuit backend requires tensorcircuit:\n"
              "    pip install 'quonic[tensorcircuit]'\nor: pip install tensorcircuit",
        "zh": "使用 tensorcircuit 后端需要安装 tensorcircuit：\n"
              "    pip install 'quonic[tensorcircuit]'\n或： pip install tensorcircuit",
    },
    "err.tensorcircuit_gate": {
        "en": "tensorcircuit backend does not support gate '{name}'",
        "zh": "tensorcircuit 后端暂不支持门 '{name}'",
    },
    "err.cudaq_missing": {
        "en": "cudaq backend requires cuda-quantum:\n"
              "    pip install 'quonic[cudaq]'\nor: pip install cuda-quantum",
        "zh": "使用 cudaq 后端需要安装 cuda-quantum：\n"
              "    pip install 'quonic[cudaq]'\n或： pip install cuda-quantum",
    },
    "err.cudaq_gate": {
        "en": "cudaq backend does not support gate '{name}'",
        "zh": "cudaq 后端暂不支持门 '{name}'",
    },
    "err.mindquantum_missing": {
        "en": "mindquantum backend requires mindquantum:\n"
              "    pip install 'quonic[mindquantum]'\nor: pip install mindquantum",
        "zh": "使用 mindquantum 后端需要安装 mindquantum：\n"
              "    pip install 'quonic[mindquantum]'\n或： pip install mindquantum",
    },
    "err.mindquantum_gate": {
        "en": "mindquantum backend does not support gate '{name}'",
        "zh": "mindquantum 后端暂不支持门 '{name}'",
    },
    "err.qpanda_missing": {
        "en": "qpanda backend requires pyqpanda3:\n"
              "    pip install 'quonic[qpanda]'\nor: pip install pyqpanda3",
        "zh": "使用 qpanda 后端需要安装 pyqpanda3：\n"
              "    pip install 'quonic[qpanda]'\n或： pip install pyqpanda3",
    },
    "err.qpanda_gate": {
        "en": "qpanda backend does not support gate '{name}'",
        "zh": "qpanda 后端暂不支持门 '{name}'",
    },
    "err.cqlib_missing": {
        "en": "cqlib backend requires cqlib:\n"
              "    pip install 'quonic[cqlib]'\nor: pip install cqlib",
        "zh": "使用 cqlib 后端需要安装 cqlib：\n"
              "    pip install 'quonic[cqlib]'\n或： pip install cqlib",
    },
    "err.cqlib_gate": {
        "en": "cqlib backend does not support gate '{name}'",
        "zh": "cqlib 后端暂不支持门 '{name}'",
    },


    # -- hardware backend errors --
    "err.ibm_missing": {
        "en": "IBM Quantum backend requires qiskit-ibm-runtime:\n"
              "    pip install 'quonic[ibm]'\nor: pip install qiskit-ibm-runtime",
        "zh": "使用 IBM Quantum 后端需要安装 qiskit-ibm-runtime：\n"
              "    pip install 'quonic[ibm]'\n或： pip install qiskit-ibm-runtime",
    },
    "err.braket_missing": {
        "en": "AWS Braket backend requires amazon-braket-sdk:\n"
              "    pip install 'quonic[braket]'\nor: pip install amazon-braket-sdk",
        "zh": "使用 AWS Braket 后端需要安装 amazon-braket-sdk：\n"
              "    pip install 'quonic[braket]'\n或： pip install amazon-braket-sdk",
    },
    "err.braket_noise": {
        "en": "Braket backend does not support noise injection; "
              "use the Braket simulator's native noise model",
        "zh": "Braket 后端不支持噪声注入；请使用 Braket 模拟器的原生噪声模型",
    },
    "err.braket_gate": {
        "en": "Braket backend does not support gate '{name}'",
        "zh": "Braket 后端不支持门 '{name}'",
    },
    "err.azure_missing": {
        "en": "Azure Quantum backend requires azure-quantum:\n"
              "    pip install 'quonic[azure]'\nor: pip install azure-quantum",
        "zh": "使用 Azure Quantum 后端需要安装 azure-quantum：\n"
              "    pip install 'quonic[azure]'\n或： pip install azure-quantum",
    },
    "err.azure_noise": {
        "en": "Azure Quantum backend does not support noise injection",
        "zh": "Azure Quantum 后端不支持噪声注入",
    },
    "err.ionq_missing": {
        "en": "IonQ backend requires ionq-cirq:\n"
              "    pip install 'quonic[ionq]'\nor: pip install ionq-cirq",
        "zh": "使用 IonQ 后端需要安装 ionq-cirq：\n"
              "    pip install 'quonic[ionq]'\n或： pip install ionq-cirq",
    },
    "err.ionq_noise": {
        "en": "IonQ backend does not support noise injection",
        "zh": "IonQ 后端不支持噪声注入",
    },
    "err.ionq_gate": {
        "en": "IonQ backend does not support gate '{name}'",
        "zh": "IonQ 后端不支持门 '{name}'",
    },
    "err.rigetti_missing": {
        "en": "Rigetti backend requires pyquil:\n"
              "    pip install 'quonic[rigetti]'\nor: pip install pyquil",
        "zh": "使用 Rigetti 后端需要安装 pyquil：\n"
              "    pip install 'quonic[rigetti]'\n或： pip install pyquil",
    },
    "err.rigetti_noise": {
        "en": "Rigetti backend does not support noise injection",
        "zh": "Rigetti 后端不支持噪声注入",
    },
    "err.rigetti_gate": {
        "en": "Rigetti backend does not support gate '{name}'",
        "zh": "Rigetti 后端不支持门 '{name}'",
    },
    "err.xanadu_missing": {
        "en": "Xanadu backend requires strawberryfields:\n"
              "    pip install 'quonic[xanadu]'\nor: pip install strawberryfields",
        "zh": "使用 Xanadu 后端需要安装 strawberryfields：\n"
              "    pip install 'quonic[xanadu]'\n或： pip install strawberryfields",
    },
    "err.xanadu_noise": {
        "en": "Xanadu backend does not support noise injection",
        "zh": "Xanadu 后端不支持噪声注入",
    },
    "err.xanadu_gate": {
        "en": "Xanadu backend does not support gate '{name}'",
        "zh": "Xanadu 后端不支持门 '{name}'",
    },
    "err.quera_missing": {
        "en": "QuEra backend requires qurry:\n"
              "    pip install 'quonic[quera]'\nor: pip install qurry",
        "zh": "使用 QuEra 后端需要安装 qurry：\n"
              "    pip install 'quonic[quera]'\n或： pip install qurry",
    },
    "err.quera_noise": {
        "en": "QuEra backend does not support noise injection",
        "zh": "QuEra 后端不支持噪声注入",
    },
    "err.quera_gate": {
        "en": "QuEra backend does not support gate '{name}'",
        "zh": "QuEra 后端不支持门 '{name}'",
    },

    # -------------------------------------------------------- core errors
    "err.gate_angle": {
        "en": "parameterized gate rotation angle must be a number (radians), "
              "got {theta!r} ({type})",
        "zh": "参数化门的旋转角必须是数字（弧度），收到 {theta!r}（{type}）",
    },
    "err.unknown_gate": {
        "en": "Unknown gate '{gate}'. Available: {gates}",
        "zh": "未知的量子门 '{gate}'。可用门：{gates}",
    },
    "err.qgate_arg": {
        "en": "qgate's first argument must be a Gate object or gate name string, "
              "got {type}",
        "zh": "qgate 的第一个参数必须是门对象或门名字符串，收到 {type}",
    },
    "err.qgate_arity": {
        "en": "gate {name} requires {expected} qubits but got {actual}: {qubits}",
        "zh": "门 {name} 需要 {expected} 个量子比特，但给了 {actual} 个：{qubits}",
    },
    "err.unknown_axis": {
        "en": "unknown rotation axis '{axis}'",
        "zh": "未知旋转轴 '{axis}'",
    },
    "err.sim_unsupported_gate": {
        "en": "statevector simulator does not support gate '{name}'",
        "zh": "态矢量模拟器暂不支持门 '{name}'",
    },
    "err.pauli_len": {
        "en": "Pauli string length {actual} does not match qubit count {expected}",
        "zh": "泡利串长度 {actual} 与量子比特数 {expected} 不一致",
    },
    "err.noise_prob": {
        "en": "depolarizing probability {name} must be in [0, 1], got {p}",
        "zh": "去极化概率 {name} 需在 [0, 1] 内，收到 {p}",
    },
    "err.noise_arg": {
        "en": "noise must be a NoiseModel, a probability in [0,1], or None",
        "zh": "noise 参数必须是 NoiseModel、一个 [0,1] 内的概率数值，或 None",
    },
    "err.noise_t2_t1": {
        "en": "T2 must be <= 2 * T1 (causality constraint)",
        "zh": "T2 必须 <= 2 * T1（因果性约束）",
    },
    "err.compare_qint": {
        "en": "comparator requires a QInt register, got {type}",
        "zh": "比较器需要 QInt 寄存器，收到 {type}",
    },
    "err.qshow_arg": {
        "en": "qshow's first argument must be a Result object (construct with "
              "Result.from_counts / Result.from_value), or leave empty to run the "
              "current circuit",
        "zh": "qshow 的第一个参数必须是 Result 对象（可用 Result.from_counts / "
              "Result.from_value 构造），或留空以运行当前电路",
    },
    "err.unknown_result_kind": {
        "en": "Unknown Result kind '{kind}'",
        "zh": "未知的 Result 类型 '{kind}'",
    },

    # --------------------------------------------------------- qif errors
    "err.qif_single_bit": {
        "en": "{kind} branch only supports single-qubit gates, {which} got {name}",
        "zh": "MVP 的 {kind} 分支只支持单比特门，{which} 收到 {name}",
    },
    "err.qif_unitary": {
        "en": "{kind} branch requires a unitary gate, cannot be measurement gate "
              "'measure'",
        "zh": "{kind} 分支需要酉门，不能是测量门 'measure'",
    },
    "err.qif_missing_then": {
        "en": "qif missing then branch (call .then(...) before .else_(...))",
        "zh": "qif 缺少 then 分支（先 .then(...) 再 .else_(...)）",
    },
    "err.qif_missing_else": {
        "en": "qif missing else branch (call .else_(...))",
        "zh": "qif 缺少 else 分支（请调用 .else_(...)）",
    },
    "err.qif_same_target": {
        "en": "MVP qif requires then/else branches on the same target qubit, "
              "got {tt} and {ft}",
        "zh": "MVP 的 qif 要求 then/else 分支作用在同一目标比特，收到 {tt} 与 {ft}",
    },
    "err.qif_ctrl_eq_target": {
        "en": "qif control and target qubits cannot be the same",
        "zh": "qif 的控制比特与目标比特不能相同",
    },
    "err.qif_nested_too_large": {
        "en": "nested qif sub-circuit too large ({n} qubits, max 4); "
              "consider decomposing manually",
        "zh": "嵌套 qif 子电路过大（{n} 比特，最多 4 比特）；请手动分解",
    },
    "err.qif_general_cu": {
        "en": "general multi-qubit controlled-U decomposition not yet implemented "
              "(then={then_name}, else={else_name}); use identity else branch or "
              "known gates (CX/CZ/SWAP/CCX)",
        "zh": "通用多比特受控 U 分解尚未实现（then={then_name}, else={else_name}）；"
              "请用恒等 else 分支或已知门（CX/CZ/SWAP/CCX）",
    },
    "err.controlled_target_count": {
        "en": "gate '{name}' requires {expected} target qubit(s), got {got}",
        "zh": "门 '{name}' 需要 {expected} 个目标比特，收到 {got} 个",
    },
    "err.controlled_single": {
        "en": "controlled target gate must be single-qubit, got {name}",
        "zh": "controlled 的目标门必须是单比特门，收到 {name}",
    },
    "err.controlled_unitary": {
        "en": "controlled requires a unitary gate, cannot be measurement gate "
              "'measure'",
        "zh": "controlled 需要酉门，不能是测量门 'measure'",
    },
    "err.controlled_ctrl_eq_target": {
        "en": "controlled control and target qubits cannot be the same",
        "zh": "controlled 的控制比特与目标比特不能相同",
    },
    "err.creg_name": {
        "en": "creg name must be a non-empty string, got {name!r}",
        "zh": "creg 名必须是非空字符串，收到 {name!r}",
    },
    "err.cif_missing_then": {
        "en": "cif missing then branch (call .then(...) before .else_(...))",
        "zh": "cif 缺少 then 分支（先 .then(...) 再 .else_(...)）",
    },
    "err.cif_missing_else": {
        "en": "cif missing else branch (call .else_(...))",
        "zh": "cif 缺少 else 分支（请调用 .else_(...)）",
    },
    "err.cif_same_target": {
        "en": "MVP cif requires then/else branches on the same target qubit, "
              "got {tt} and {ft}",
        "zh": "MVP 的 cif 要求 then/else 分支作用在同一目标比特，收到 {tt} 与 {ft}",
    },
    "err.cif_ctrl_eq_target": {
        "en": "cif control and target qubits cannot be the same",
        "zh": "cif 的控制比特与目标比特不能相同",
    },
    "err.creg_width": {
        "en": "creg width must be >= 1, got {width}",
        "zh": "creg 位宽必须 >= 1，收到 {width}",
    },
    "err.creg_bit": {
        "en": "creg bit index {bit} out of range [0, {width})",
        "zh": "creg 位索引 {bit} 超出范围 [0, {width})",
    },
    "err.creg_bitstring": {
        "en": "register value must be an int or a bitstring of 0/1, got {value!r}",
        "zh": "寄存器值必须是 int 或只含 0/1 的比特串，收到 {value!r}",
    },
    "err.cif_value": {
        "en": "cif register value must be in [0, {max}), got {value}",
        "zh": "cif 的寄存器值必须在 [0, {max}) 内，收到 {value}",
    },
    "err.cwhile_cond": {
        "en": "cwhile condition must be a classical bit declared with creg(), "
              "got {cond!r}",
        "zh": "cwhile 的条件必须是 creg() 声明的经典位，收到 {cond!r}",
    },
    "err.cwhile_until": {
        "en": "cwhile until must be a register value in [0, {max}), got {until}",
        "zh": "cwhile 的 until 必须是 [0, {max}) 内的寄存器值，收到 {until}",
    },
    "err.cwhile_max_iters": {
        "en": "cwhile max_iters must be >= 1, got {max_iters}",
        "zh": "cwhile 的 max_iters 必须 >= 1，收到 {max_iters}",
    },

    # -------------------------------------------------------- topology
    "err.topology_nonneg": {
        "en": "qubit count must be non-negative, got {n}",
        "zh": "量子比特数需非负，收到 {n}",
    },
    "err.topology_self_loop": {
        "en": "self-loop edge ({u}, {v}) is invalid",
        "zh": "自环边 ({u}, {v}) 不合法",
    },
    "err.topology_out_of_range": {
        "en": "edge ({u}, {v}) is out of qubit range [0, {n})",
        "zh": "边 ({u}, {v}) 超出量子比特范围 [0, {n})",
    },

    # --------------------------------------------------------- compiler
    "err.routing": {
        "en": "circuit cannot be mapped to coupling map ({map}): the following "
              "gates have disconnected qubit pairs — {detail}",
        "zh": "电路无法映射到耦合图（{map}）：以下门的量子比特对不相连 —— {detail}",
    },
    "err.routing_cwhile": {
        "en": "SWAP routing does not support cwhile (classical feedback loop); "
              "groverize() it into a static circuit first, then compile(route=True)",
        "zh": "SWAP 路由暂不支持 cwhile（经典反馈循环）；"
              "请先用 groverize() 把它编译成静态电路，再 compile(route=True)",
    },
    "err.routing_disconnected": {
        "en": "coupling map is disconnected, cannot route {name}{qubits}",
        "zh": "耦合图不连通，无法路由 {name}{qubits}",
    },
    "err.routing_etc": {
        "en": " and {n} more gates",
        "zh": " 等 {n} 个门",
    },
    "err.grover_type": {
        "en": "groverize expects a cwhile operation (ClassicalWhileOperation), got {type}",
        "zh": "groverize 需要 cwhile 操作（ClassicalWhileOperation），收到 {type}",
    },
    "err.grover_prob": {
        "en": "success_prob must be in (0, 1), got {p}",
        "zh": "success_prob 需在 (0, 1) 内，收到 {p}",
    },
    "err.grover_no_op": {
        "en": "groverize() must be called after the `with cwhile(...)` block has "
              "finished (no loop body captured yet)",
        "zh": "groverize() 必须在 `with cwhile(...)` 块结束后调用（尚未捕获循环体）",
    },
    "err.grover_body_unitary": {
        "en": "cwhile body must be a sequence of unitary gates ending with "
              "creg.measure(...) ops for the loop creg; measure/cif/cwhile are not "
              "supported by groverize (fall back to the native backend)",
        "zh": "cwhile 循环体必须是纯酉门序列并以循环 creg 的 creg.measure(...) 结尾；"
              "groverize 不支持 measure/cif/cwhile（请回退 native 后端）",
    },
    "err.grover_body_bits": {
        "en": "cwhile body must measure each register bit exactly once for groverize",
        "zh": "groverize 要求 cwhile 循环体恰好测量寄存器每个位各一次",
    },

    # ------------------------------------------------------- simulators
    "err.density_gate": {
        "en": "density matrix engine does not support gate '{name}'",
        "zh": "密度矩阵引擎暂不支持门 '{name}'",
    },
    "err.stabilizer_gate": {
        "en": "stabilizer engine does not support gate '{name}'",
        "zh": "稳定子引擎暂不支持门 '{name}'",
    },
    "err.stabilizer_measure": {
        "en": "deterministic measurement failed: Z_q not in stabilizer group",
        "zh": "确定性测量失败：Z_q 不在稳定子群内",
    },
    "err.mps_swap": {
        "en": "MPS engine only supports swap on adjacent qubits",
        "zh": "MPS 引擎仅支持相邻量子比特的 swap 门",
    },
    "err.mps_gate": {
        "en": "MPS engine does not support gate '{name}'",
        "zh": "MPS 引擎暂不支持门 '{name}'",
    },
    "err.self_gate": {
        "en": "native engine does not support single-qubit gate '{name}'",
        "zh": "自研引擎暂不支持单比特门 '{name}'",
    },

    # ------------------------------------------------------- algorithms
    "err.vqe_scipy": {
        "en": "VQE requires scipy:\n"
              "    pip install 'quonic[algorithms]'\n"
              "or: pip install scipy",
        "zh": "使用 VQE 需要安装 scipy：\n"
              "    pip install 'quonic[algorithms]'\n"
              "或： pip install scipy",
    },
    "err.qaoa_scipy": {
        "en": "QAOA requires scipy:\n"
              "    pip install 'quonic[algorithms]'\n"
              "or: pip install scipy",
        "zh": "使用 QAOA 需要安装 scipy：\n"
              "    pip install 'quonic[algorithms]'\n"
              "或： pip install scipy",
    },
    "err.oracle_len": {
        "en": "marked bitstring '{oracle}' length {n} does not match qubit count "
              "{n_qubits}",
        "zh": "标记比特串 '{oracle}' 长度 {n} 与量子比特数 {n_qubits} 不一致",
    },
    "err.oracle_n_qubits": {
        "en": "oracle qubit count {n} does not match n_qubits={n_qubits}",
        "zh": "神谕的量子比特数 {n} 与 n_qubits={n_qubits} 不一致",
    },
    "err.oracle_empty": {
        "en": "oracle marks no states, cannot count",
        "zh": "神谕没有标记任何状态，无法计数",
    },
    "err.oracle_type": {
        "en": "oracle must be a marked bitstring, @oracle-decorated object, or "
              "predicate function",
        "zh": "oracle 必须是标记比特串、@oracle 装饰器产物或谓词函数",
    },
    "err.mark_state_bitstring": {
        "en": "mark_state requires a bitstring of only 0/1, got {bitstring!r}",
        "zh": "mark_state 需要只含 0/1 的比特串，收到 {bitstring!r}",
    },
    "err.oracle_n_qubits_positive": {
        "en": "n_qubits must be a positive integer, got {n_qubits!r}",
        "zh": "n_qubits 必须是正整数，收到 {n_qubits!r}",
    },
    "err.hamiltonian_imag": {
        "en": "Hamiltonian coefficient {coeff} has non-negligible imaginary part; "
              "current VQE only supports real coefficients",
        "zh": "哈密顿量系数 {coeff} 含不可忽略的虚部，当前 VQE 仅支持实系数",
    },
    "err.shor_n": {
        "en": "N must be >= 2, got {N}",
        "zh": "N 必须 >= 2，收到 {N}",
    },
    "err.shor_failed": {
        "en": "Shor's algorithm failed to find a factor of {N}; increase shots / "
              "attempts, or try a different N",
        "zh": "Shor 算法未能找到 {N} 的因子；请增加 shots / attempts，或更换 N",
    },

    # ------------------------------------------------------- quantum chemistry
    "err.chem.pyscf_missing": {
        "en": "Molecular SCF requires pyscf:\n"
              "    pip install 'quonic[chem]'\nor: pip install pyscf",
        "zh": "分子 SCF 计算需要安装 pyscf：\n"
              "    pip install 'quonic[chem]'\n或： pip install pyscf",
    },
    "err.chem.openfermion_missing": {
        "en": "Molecular Hamiltonian requires openfermion:\n"
              "    pip install 'quonic[chem]'\nor: pip install openfermion",
        "zh": "分子哈密顿量需要安装 openfermion：\n"
              "    pip install 'quonic[chem]'\n或： pip install openfermion",
    },
    "err.chem.rdkit_missing": {
        "en": "SMILES/SDF parsing requires rdkit:\n"
              "    pip install 'quonic[chem-rdkit]'\nor: pip install rdkit",
        "zh": "SMILES/SDF 解析需要安装 rdkit：\n"
              "    pip install 'quonic[chem-rdkit]'\n或： pip install rdkit",
    },
    "err.chem.unknown_basis": {
        "en": "Unknown basis set '{basis}'. Use list_bases() to see available options.",
        "zh": "未知基组 '{basis}'。使用 list_bases() 查看可用选项。",
    },
    "err.chem.xyz_parse": {
        "en": "Failed to parse XYZ string: {reason}",
        "zh": "解析 XYZ 字符串失败：{reason}",
    },
    "err.chem.smiles_convert": {
        "en": "Failed to convert SMILES '{smiles}' to 3D geometry: {reason}",
        "zh": "将 SMILES '{smiles}' 转换为 3D 几何结构失败：{reason}",
    },
    "err.chem.scf_converge": {
        "en": "SCF did not converge after {max_cycle} iterations",
        "zh": "SCF 在 {max_cycle} 次迭代后未收敛",
    },
    "err.chem.mapping_unknown": {
        "en": "Unknown qubit mapping '{mapping}'. Supported: jordan_wigner, bravyi_kitaev",
        "zh": "未知量子比特映射 '{mapping}'。支持：jordan_wigner, bravyi_kitaev",
    },
    "err.chem.active_space_invalid": {
        "en": "Active space CAS({n_e},{n_o}) is invalid for molecule with {elec} electrons and {orb} orbitals",
        "zh": "活性空间 CAS({n_e},{n_o}) 对于含 {elec} 个电子和 {orb} 个轨道的分子无效",
    },
    "err.chem.active_space_auto": {
        "en": "Automatic active space selection requires a PySCF mean-field calculation first",
        "zh": "自动活性空间选择需要先运行 PySCF 平均场计算",
    },
    "err.chem.fragment_empty": {
        "en": "Fragmentation produced no fragments",
        "zh": "分片未产生任何片段",
    },
    "err.chem.fragment_large": {
        "en": "Fragment {idx} has {n} atoms, exceeds max_fragment_size={max}",
        "zh": "片段 {idx} 含 {n} 个原子，超过 max_fragment_size={max}",
    },
    "err.chem.dmet_converge": {
        "en": "DMET did not converge after {max_iter} iterations (residual={residual:.2e})",
        "zh": "DMET 在 {max_iter} 次迭代后未收敛（残差={residual:.2e}）",
    },
    "err.chem.dmet_solver": {
        "en": "Unknown DMET solver '{solver}'. Supported: fci, ccsd",
        "zh": "未知 DMET 求解器 '{solver}'。支持：fci, ccsd",
    },
    "err.chem.pdb_parse": {
        "en": "Failed to parse PDB file '{path}': no ATOM/HETATM records found",
        "zh": "解析 PDB 文件 '{path}' 失败：未找到 ATOM/HETATM 记录",
    },
    "err.chem.mol2_parse": {
        "en": "Failed to parse MOL2 file '{path}': no @<TRIPOS>ATOM section found",
        "zh": "解析 MOL2 文件 '{path}' 失败：未找到 @<TRIPOS>ATOM 段",
    },
    "err.chem.fcidump_parse": {
        "en": "Failed to parse FCIDUMP file: {reason}",
        "zh": "解析 FCIDUMP 文件失败：{reason}",
    },

    # ------------------------------------------------------------- zne
    "err.zne_fold_k": {
        "en": "fold k must be a non-negative integer, got {k}",
        "zh": "fold 的 k 必须是非负整数，收到 {k}",
    },
    "err.zne_fold_unitary": {
        "en": "cannot fold op '{name}': only unitary gates with trailing "
              "measurements are foldable",
        "zh": "无法折叠操作 '{name}'：只能折叠酉门与末尾测量",
    },
    "err.zne_factors": {
        "en": "ZNE factor λ must be an odd integer >= 1, got {lam}",
        "zh": "ZNE 噪声倍数 λ 必须是 >= 1 的奇数，收到 {lam}",
    },
    "err.zne_factors_order": {
        "en": "ZNE factors must be strictly increasing, got {lam} after {prev}",
        "zh": "ZNE 噪声倍数必须严格递增，收到 {lam}（前一个为 {prev}）",
    },
    "err.zne_metric": {
        "en": "zne() requires exactly one of target (success metric) or "
              "observable (expectation metric)",
        "zh": "zne() 需要 target（成功率指标）或 observable（期望值指标）二选一",
    },
    "err.zne_noise": {
        "en": "zne() requires a non-zero noise model (a probability in (0, 1] "
              "or a NoiseModel)",
        "zh": "zne() 需要非零噪声模型（一个 (0, 1] 内的概率或 NoiseModel）",
    },
    "err.zne_observable": {
        "en": "observable must be a Pauli string of I/X/Y/Z, got {observable!r}",
        "zh": "observable 必须是 I/X/Y/Z 构成的泡利串，收到 {observable!r}",
    },
    "err.zne_backend": {
        "en": "ZNE success metric supports only 'native', 'qiskit', or 'qi' "
              "backend, got '{backend}'",
        "zh": "ZNE 成功率指标仅支持 'native'、'qiskit' 或 'qi' 后端，收到 '{backend}'",
    },
    "err.zne_qi_noise": {
        "en": "backend='qi' runs on real hardware with intrinsic noise; do not "
              "pass a noise model (ZNE folds the circuit to amplify it instead)",
        "zh": "backend='qi' 跑在真机上，自带本征噪声；请勿传入噪声模型（ZNE 通过折叠电路来放大本征噪声）",
    },
    "err.zne_calib_n": {
        "en": "readout calibration has {calib} qubits but the circuit has "
              "{qubits}; they must match",
        "zh": "读出校准覆盖 {calib} 个量子比特，但电路有 {qubits} 个，二者必须一致",
    },
    "err.zne_extrap": {
        "en": "extrapolation must be 'linear' or 'exponential', got {method!r}",
        "zh": "外推方法必须是 'linear' 或 'exponential'，收到 {method!r}",
    },
    "err.zne_scipy": {
        "en": "exponential extrapolation requires scipy:\n"
              "    pip install 'quonic[algorithms]'",
        "zh": "指数外推需要 scipy：\n"
              "    pip install 'quonic[algorithms]'",
    },

    # ------------------------------------------------------------ readout
    "err.readout_n": {
        "en": "num_qubits must be a positive integer, got {n!r}",
        "zh": "num_qubits 必须是正整数，收到 {n!r}",
    },
    "err.readout_singular": {
        "en": "readout confusion matrix is singular (readout error p=0.5 or "
              "degenerate); it cannot be inverted",
        "zh": "读出混淆矩阵奇异（读出误差 p=0.5 或退化），无法求逆",
    },
    "err.readout_correlated_n": {
        "en": "correlated readout calibration needs 2^n circuits and is limited "
              "to n <= {max_n}; got n={n}",
        "zh": "关联读出校准需要 2^n 个电路，仅支持 n <= {max_n}；收到 n={n}",
    },

    # ------------------------------------------------------------ misc core
    "err.stack_empty": {
        "en": "circuit stack is already at the bottom, cannot pop further",
        "zh": "电路栈已到底层，无法继续 pop",
    },
    "err.qint_n_bits": {
        "en": "n_bits must be a positive integer, got {n_bits!r}",
        "zh": "n_bits 必须是正整数，收到 {n_bits!r}",
    },
    "err.qint_value_range": {
        "en": "value out of range for a {n_bits}-bit integer [0, {max}), got {value}",
        "zh": "value 超出 {n_bits} 位整数范围 [0, {max})，收到 {value}",
    },
    "err.qint_superposition": {
        "en": "quantum integer is in superposition; cannot convert to int directly "
              "— run qshow() to measure first",
        "zh": "量子整数处于叠加态，无法直接转成 int；请先 qshow() 测量后读取结果",
    },
    "err.statevector_gate": {
        "en": "statevector engine does not support gate '{name}'",
        "zh": "态矢量引擎暂不支持门 '{name}'",
    },

    # ------------------------------------------------------------- viz
    "err.viz_matplotlib": {
        "en": "visualization requires matplotlib:\n"
              "    pip install 'quonic[viz]'\n"
              "or: pip install matplotlib",
        "zh": "使用可视化需要安装 matplotlib：\n"
              "    pip install 'quonic[viz]'\n"
              "或： pip install matplotlib",
    },
    "err.viz_history": {
        "en": "Result has no convergence history. Run with "
              "vqe(..., record_history=True) or qaoa_maxcut(..., record_history=True), "
              "or pass an energy list directly.",
        "zh": "Result 里没有收敛轨迹。请用 vqe(..., record_history=True) 或 "
              "qaoa_maxcut(..., record_history=True) 运行，或直接传入能量列表。",
    },
    "err.viz_marked": {
        "en": "marked must be a 0/1 bitstring of length {n_qubits}, got {marked!r}",
        "zh": "marked 需为长度 {n_qubits} 的 0/1 比特串，收到 {marked!r}",
    },
    "err.viz_gate_matrix": {
        "en": "plot_gate_matrix requires a Gate, GateOperation, or gate name string",
        "zh": "plot_gate_matrix 需要 Gate / GateOperation / 门名字符串",
    },
    "err.viz_measure_unitary": {
        "en": "measurement gate has no unitary matrix",
        "zh": "测量门没有酉矩阵",
    },
    "err.viz_counts": {
        "en": "plot_counts requires a Result (counts) or a dict histogram",
        "zh": "plot_counts 需要 Result（counts）或 dict 直方图",
    },
    "err.viz_no_perf": {
        "en": "no measured data for class '{cls}'; run the benchmark calibration first",
        "zh": "没有 '{cls}' 类别的实测数据，请先运行基准校准",
    },
    "err.viz_bloch_norm": {
        "en": "Bloch vector norm must be ≤ 1",
        "zh": "布洛赫向量模长需 ≤ 1",
    },
    "err.viz_bloch_single": {
        "en": "Bloch sphere only accepts a single-qubit state (2 complex amplitudes) "
              "or a 3D Bloch vector",
        "zh": "布洛赫球只接受单比特态（2 个复振幅）或 3 维布洛赫向量",
    },
    "err.viz_state_input": {
        "en": "unrecognized quantum state input (need 1D statevector / 2D density "
              "matrix / engine / Circuit)",
        "zh": "无法识别的量子态输入（需 1D 态矢量 / 2D 密度矩阵 / 引擎 / Circuit）",
    },
    "err.viz_concurrence": {
        "en": "concurrence is only defined for 2-qubit states (needs a 4×4 density "
              "matrix)",
        "zh": "并发度只对 2 比特态定义（需 4×4 密度矩阵）",
    },
    "err.viz_partition": {
        "en": "partition must be a non-empty subset of qubit indices [0, {n}), got "
              "{partition}",
        "zh": "partition 需为 [0, {n}) 的非空比特下标子集，收到 {partition}",
    },
}
# fmt: on

_current = os.environ.get("QUONIC_LANG", "en").strip().lower()
if _current not in _LANGUAGES:
    _current = "en"


def get_language() -> str:
    """Return the current language code ("en" or "zh")."""
    return _current


def set_language(lang: str) -> None:
    """Switch the runtime message language ("en" or "zh")."""
    global _current
    key = lang.strip().lower()
    if key not in _LANGUAGES:
        raise ValueError(
            f"unknown language '{lang}' (supported: {', '.join(sorted(_LANGUAGES))})"
        )
    _current = key


def tr(key: str, **fmt: Any) -> str:
    """Translate *key* for the current language and interpolate *fmt*.

    Falls back to English, then to the raw key, so a missing translation never
    raises at runtime.
    """
    entry = _MESSAGES.get(key)
    if entry is None:
        return key
    template = entry.get(_current) or entry.get("en")
    if template is None:
        return key
    return template.format(**fmt) if fmt else template
