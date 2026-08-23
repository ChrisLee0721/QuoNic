"""Generate Chinese versions of tutorial notebooks."""


TUTORIALS = {
    "01_basics": {
        "title": "教程 01：基础",
        "desc": "学习 QuoNic 的基础：构建电路、运行电路、读取结果。",
        "sections": [
            ("最简单的电路", "from quonic import qgate, qshow\nfrom quonic.gates import H, CX\n\nqgate(H, 0)      # 对量子比特 0 施加 Hadamard 门\nqgate(CX, 0, 1)  # CNOT：控制=0，目标=1\nqshow()           # 运行并显示结果\n\n# 这创建了一个 Bell 态。qshow() 在最佳可用后端上运行电路并打印结果。"),
            ("理解量子比特", "在 QuoNic 中，量子比特就是数字。qgate(H, 0) 对量子比特 0 施加 Hadamard 门。不需要创建电路对象，不需要管理量子比特寄存器。\n\nqgate(X, 0)      # Pauli-X（比特翻转）作用于量子比特 0\nqgate(H, 1)      # Hadamard 作用于量子比特 1\nqgate(CX, 0, 1)  # CNOT：控制=0，目标=1"),
            ("读取结果", "qshow() 返回一个 Result 对象，包含测量计数：\n\nresult = qshow()\nprint(result.counts)  # {'00': 512, '11': 512}\nprint(result.shots)   # 1024"),
            ("切换后端", "同一个电路可以在不同后端上运行：\n\nqshow(backend='qiskit')      # IBM Qiskit\nqshow(backend='cirq')        # Google Cirq\nqshow(backend='qulacs')      # Qulacs（高性能 C++）\nqshow(backend='native')      # QuoNic 原生引擎"),
            ("添加噪声", "模拟真实硬件的噪声：\n\nqshow(noise=0.05)  # 5% 去极化噪声"),
        ],
    },
    "02_algorithms": {
        "title": "教程 02：算法",
        "desc": "探索 QuoNic 的 75 个算法模板。",
        "sections": [
            ("Grover 搜索", "from quonic.algorithms import grover\n\nresult = grover('11', 2, shots=1024)\nprint(result.counts)  # {'11': ~1000}\n\n# Grover 搜索在无序数据库中找到目标，比经典方法快 √N 倍。"),
            ("量子傅里叶变换", "from quonic.algorithms import qft\n\nresult = qft(n_qubits=3, shots=1024)\nprint(result.counts)\n\n# QFT 是许多量子算法的基础，包括 Shor 算法和量子相位估计。"),
            ("VQE 变分量子本征求解器", "from quonic.algorithms import vqe\n\nhamiltonian = [(1.0, 'ZZ'), (0.5, 'XI'), (0.5, 'IX')]\nresult = vqe(hamiltonian, n_qubits=2, maxiter=200)\nprint(f'基态能量: {result.value}')\n\n# VQE 用于量子化学，计算分子的基态能量。"),
            ("QAOA 量子近似优化", "from quonic.algorithms import qaoa_maxcut\n\nedges = [(0, 1), (1, 2), (0, 2)]\nresult = qaoa_maxcut(edges, 3, p=1, maxiter=200)\nprint(f'MaxCut: {result.value}')\n\n# QAOA 用于组合优化问题，如 MaxCut、旅行商问题等。"),
        ],
    },
    "03_noise_mitigation": {
        "title": "教程 03：噪声缓解",
        "desc": "真实量子硬件有噪声。QuoNic 提供两种错误缓解技术。",
        "sections": [
            ("添加噪声", "from quonic import qgate, qshow, reset\nfrom quonic.gates import CX, H\n\nreset()\nqgate(H, 0)\nqgate(CX, 0, 1)\n\n# 无噪声：理想 Bell 态\nprint('无噪声：')\nqshow()\n\n# 5% 去极化噪声\nprint('有噪声：')\nqshow(noise=0.05)"),
            ("ZNE 零噪声外推", "from quonic import zne\nfrom quonic.ir import Circuit, GateOperation\n\nc = Circuit()\nc.add(GateOperation('x', (0,)))\nc.add(GateOperation('measure', (0,)))\n\n# ZNE 线性外推\nres = zne(c, noise=0.05, target='1', shots=4096, extrapolation='linear')\nprint(f'原始: {res.values[0]:.3f}')\nprint(f'ZNE: {res.extrapolated:.3f}')\nprint('理想: 1.000')"),
            ("读出校准", "from quonic import calibrate\nfrom quonic.noise import NoiseModel\n\nn = 1\nnoise = NoiseModel(readout=0.05)\n\n# 构建校准矩阵\ncal = calibrate(n, backend='native', shots=4096, noise=noise)\n\n# 运行含噪声电路\nreset()\nqgate(X, 0)\nraw = qshow(noise=noise, shots=4096)\n\n# 应用校准\ncorrected = cal.apply(raw.counts, 4096)\nprint(f'原始: {raw.counts}')\nprint(f'校准后: {corrected}')"),
        ],
    },
    "04_gpu_acceleration": {
        "title": "教程 04：GPU 加速",
        "desc": "使用 GPU 加速量子电路模拟。",
        "sections": [
            ("标准执行", "from quonic import qgate, qshow, reset\nfrom quonic.gates import CX, H\n\n# 标准执行（自动选择最佳后端）\nreset()\nqgate(H, 0)\nqgate(CX, 0, 1)\nresult = qshow()\nprint(f'后端: native, 结果: {result.counts}')"),
            ("智能调度", "from quonic.scheduler import circuit_features, recommend_backend_gpu\nfrom quonic.stack import current_circuit\n\nreset()\nqgate(H, 0)\nqgate(CX, 0, 1)\n\nfeats = circuit_features(current_circuit())\nrec = recommend_backend_gpu(feats)\nprint(f'电路特征: n={feats[\"n\"]}, 纠缠={feats[\"entanglement\"]}')\nprint(f'最佳 GPU 后端: {rec.backend} ({rec.method})')"),
            ("CuPy GPU 引擎", "# CuPy GPU 引擎（需要安装 CuPy）\n# pip install cupy-cuda12x\ntry:\n    result = qshow(backend='cupy')\n    print(f'CuPy GPU 结果: {result.counts}')\nexcept ImportError:\n    print('CuPy 未安装 — 安装: pip install cupy-cuda12x')"),
        ],
    },
    "05_advanced": {
        "title": "教程 05：高级功能",
        "desc": "探索 QuoNic 的高级功能：qif、creg、cwhile、优化。",
        "sections": [
            ("量子条件语句 qif", "from quonic import qif, qgate, reset\nfrom quonic.gates import H, X\n\nreset()\nqgate(H, 0)  # 叠加态\nqif(0).then(X, 1)  # 如果 qubit 0 是 |1>，翻转 qubit 1\nqshow()"),
            ("经典条件语句 cif", "from quonic import cif, creg, qgate, reset\nfrom quonic.gates import H, X\n\nreset()\nqgate(H, 0)\nflag = creg('flag')\nflag.measure(0)\ncif(flag, 1).then(X, 1)  # 如果测量结果是 1，翻转 qubit 1\nqshow()"),
            ("电路优化", "from quonic.compiler import optimize\nfrom quonic.ir import Circuit, GateOperation\n\nc = Circuit()\nc.allocate(1)\nc.add(GateOperation('h', (0,)))\nc.add(GateOperation('h', (0,)))  # H·H = I\n\noptimized = optimize(c, passes=('cancel',))\nprint(f'优化前: {len(c.ops)} 门')\nprint(f'优化后: {len(optimized.ops)} 门')"),
            ("自定义门", "from quonic.gates import Gate\nimport numpy as np\n\n# 创建自定义 T 门\nt_mat = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)\nGate.from_matrix('my_t', t_mat)\n\n# 使用自定义门\nqgate('my_t', 0)\nqshow()"),
        ],
    },
}


def gen_notebook(name, info):
    """Generate a Jupyter notebook for a tutorial."""
    cells = []

    # Title cell
    cells.append(f'{{"cell_type": "markdown", "metadata": {{}}, "source": ["# {info["title"]}\\n", "\\n", "{info["desc"]}"]}}')

    # Section cells
    for title, code in info["sections"]:
        cells.append(f'{{"cell_type": "markdown", "metadata": {{}}, "source": ["## {title}"]}}')
        cells.append(f'{{"cell_type": "code", "metadata": {{}}, "source": ["{code.replace(chr(10), chr(10) + '", "')}"], "outputs": [], "execution_count": null}}')

    # Build notebook JSON
    nb = f'''{{
 "cells": [{",".join(cells)}],
 "metadata": {{
  "kernelspec": {{
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  }},
  "language_info": {{
   "name": "python",
   "version": "3.12.0"
  }}
 }},
 "nbformat": 4,
 "nbformat_minor": 5
}}'''

    return nb


def main():
    for name, info in TUTORIALS.items():
        nb_path = f"docs/tutorials/{name}/{name}_zh.ipynb"
        nb_content = gen_notebook(name, info)
        with open(nb_path, "w", encoding="utf-8") as f:
            f.write(nb_content)
        print(f"Generated: {nb_path}")


if __name__ == "__main__":
    main()
