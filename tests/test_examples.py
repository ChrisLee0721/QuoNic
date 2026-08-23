"""examples/ 下的示例必须能独立跑通（returncode == 0）。

用 subprocess 隔离运行，确保示例永远是最新可用状态——示例过时比没有更糟。
"""

import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"

# 仅需 numpy（native 后端兜底），不依赖 scipy
CORE = [
    "bell/bell.py",
    "ghz/ghz.py",
    "qif/qif.py",
    "cif/cif.py",
    "controlled/controlled.py",
    "diffusion/diffusion.py",
    "qint/qint.py",
    "grover/grover.py",
    "noise/noise.py",
    "basic_gates/basic_gates.py",
    "noise_model/noise_model.py",
    "decompose/decompose.py",
    "coupling_map/coupling_map.py",
    "schedule/schedule.py",
    "mark_state/mark_state.py",
    "oracle/oracle.py",
    "qpe/qpe.py",
    "quantum_counting/quantum_counting.py",
    "shor/shor.py",
]
# 需要 scipy（vqe / qaoa 用 scipy.optimize.minimize）
SCIPY = ["vqe/vqe.py", "qaoa/qaoa.py"]
# 需要 qiskit + scipy（SparsePauliOp 导入 + VQE 变分）
QISKIT = ["from_qiskit_nature/from_qiskit_nature.py"]


def _run_example(name):
    path = EXAMPLES_DIR / name
    assert path.exists(), f"示例缺失：{path}"
    r = subprocess.run(
        [sys.executable, str(path)],
        capture_output=True, check=False,
        text=True,
        encoding="utf-8",
        timeout=180,
    )
    assert r.returncode == 0, f"{name} 运行失败：\n{r.stderr}"


@pytest.mark.parametrize("name", CORE)
def test_core_example_runs(name):
    _run_example(name)


@pytest.mark.parametrize("name", SCIPY)
def test_scipy_example_runs(name):
    pytest.importorskip("scipy")
    _run_example(name)


@pytest.mark.parametrize("name", QISKIT)
def test_qiskit_example_runs(name):
    pytest.importorskip("qiskit")
    pytest.importorskip("scipy")
    _run_example(name)
