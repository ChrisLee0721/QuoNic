"""qif —— 量子叠加 if 的 MVP 测试。

核心验证点：把 if/else 两个分支编译成受控酉再分解成基础门后，
其整体矩阵必须严格等于 |0><0|⊗F + |1><1|⊗T（F=else 门，T=then 门）。
"""

import numpy as np
import pytest

from quonic import controlled, qgate, qif, reset
from quonic.gates import CX, MEASURE, H, I, Ry, X, Z
from quonic.stack import current_circuit


def _circuit_matrix(ops, n):
    """用 StatevectorEngine 逐基态演化，拼出电路的完整酉矩阵。"""
    from quonic.simulators._statevector import StatevectorEngine

    cols = []
    for i in range(2 ** n):
        e = StatevectorEngine(n)
        e.state = np.zeros(2 ** n, dtype=complex)
        e.state[i] = 1.0
        for op in ops:
            e.apply(op.name, op.qubits, op.params)
        cols.append(e.state.copy())
    return np.column_stack(cols)


def _single_matrix(name, params=()):
    from quonic.simulators._gates import single

    return single(name, params)


# ---------------------------------------------------------------------------
# 1. ZYZ 分解回环：任意单比特酉 U 都能被 (α, β, γ, δ) 重构
# ---------------------------------------------------------------------------

def test_zyz_roundtrip():
    from quonic.qif import _zyz
    from quonic.simulators._gates import rotation

    rng = np.random.default_rng(42)
    for _ in range(30):
        m = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
        U, _ = np.linalg.qr(m)  # Q 酉（含任意全局相位）
        alpha, beta, gamma, delta = _zyz(U)
        recon = np.exp(1j * alpha) * (
            rotation("z", beta) @ rotation("y", gamma) @ rotation("z", delta)
        )
        assert np.allclose(U, recon, atol=1e-9), (
            f"ZYZ 回环失败：\nU={U}\nrecon={recon}"
        )


# ---------------------------------------------------------------------------
# 2. 整体矩阵验证：编译结果 == |0><0|⊗F + |1><1|⊗T
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("c,t", [(0, 1), (1, 0)])
def test_qif_matrix_matches_block(c, t):
    reset()
    F = Z  # else 分支
    T = X  # then 分支
    qif(c).then(T, t).else_(F, t)
    ops = current_circuit().ops

    got = _circuit_matrix(ops, 2)

    P0 = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex)
    P1 = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=complex)
    Fm = _single_matrix("z")
    Tm = _single_matrix("x")
    if c == 0:  # 控制是最低位 → 控制投影在 kron 右侧
        expected = np.kron(Fm, P0) + np.kron(Tm, P1)
    else:  # 控制是最高位 → 控制投影在 kron 左侧
        expected = np.kron(P0, Fm) + np.kron(P1, Tm)

    assert np.allclose(got, expected, atol=1e-9), (
        f"qif(c={c}, t={t}) 编译矩阵与 |0><0|⊗F + |1><1|⊗T 不一致\n"
        f"got=\n{np.round(got, 4)}\nexpected=\n{np.round(expected, 4)}"
    )


def test_qif_general_gates():
    # 非平凡参数化门组合：else=Ry(0.7)，then=X
    reset()
    F = Ry(0.7)
    T = X
    qif(0).then(T, 1).else_(F, 1)
    ops = current_circuit().ops

    got = _circuit_matrix(ops, 2)
    P0 = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex)
    P1 = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=complex)
    expected = np.kron(_single_matrix("ry", (0.7,)), P0) + np.kron(
        _single_matrix("x"), P1
    )
    assert np.allclose(got, expected, atol=1e-9)


def test_qif_else_identity():
    # else_(I, ...) 是「受控酉 = qif 特例」的自然写法，应编译成纯 CX
    reset()
    qif(0).then(X, 1).else_(I, 1)
    ops = current_circuit().ops
    # 恒等分支不残留无用的 I 门
    assert all(op.name != "i" for op in ops)

    got = _circuit_matrix(ops, 2)
    P0 = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex)
    P1 = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=complex)
    expected = np.kron(_single_matrix("i"), P0) + np.kron(_single_matrix("x"), P1)
    assert np.allclose(got, expected, atol=1e-9)


# ---------------------------------------------------------------------------
# 3. 物理语义：叠加控制 → 真纠缠（贝尔态），不是经典混合
# ---------------------------------------------------------------------------

def test_qif_bell_state():
    reset()
    qgate(H, 0)
    qif(0).then(X, 1).else_(Z, 1)
    circ = current_circuit()
    mat = _circuit_matrix(circ.ops, circ.num_qubits)
    # 输入 |00>（矩阵第 0 列）→ (|00> + |11>)/√2，即贝尔态（真纠缠，
    # 不是「先测量再二选一」的经典混合态）
    out = mat[:, 0]
    expected = np.zeros(4, dtype=complex)
    expected[0] = 1.0
    expected[3] = 1.0
    expected /= np.sqrt(2)
    assert np.allclose(out, expected, atol=1e-9)

    # 纠缠判据：整体态是纯态 Tr(ρ²)=1（经典 if 会给出 Tr(ρ²)=1/2 的混合态）
    rho = np.outer(out, out.conj())
    assert np.real(np.trace(rho @ rho)) == pytest.approx(1.0, abs=1e-9)
    # 且约化密度矩阵是最大混合（纯度 1/2）——这正是贝尔态的纠缠签名
    rho_t = np.trace(rho.reshape(2, 2, 2, 2), axis1=1, axis2=3)
    assert np.real(np.trace(rho_t @ rho_t)) == pytest.approx(0.5, abs=1e-9)


# ---------------------------------------------------------------------------
# 4. 编译结果只含基础门（可被任意后端执行）
# ---------------------------------------------------------------------------

def test_qif_uses_only_basic_gates():
    reset()
    qif(0).then(X, 1).else_(Z, 1)
    basic = {"h", "x", "y", "z", "rx", "ry", "rz", "p", "cx", "cz"}
    for op in current_circuit().ops:
        assert op.name in basic, f"qif 编译出了非基础门 '{op.name}'"


# ---------------------------------------------------------------------------
# 5. 错误分支
# ---------------------------------------------------------------------------

def test_qif_requires_both_branches():
    reset()
    with pytest.raises(ValueError, match="missing then"):
        qif(0).else_(Z, 1)


def test_qif_rejects_different_targets():
    reset()
    with pytest.raises(ValueError, match="same target"):
        qif(0).then(X, 1).else_(Z, 2)


def test_qif_rejects_multi_qubit_branch():
    reset()
    with pytest.raises(ValueError, match="single-qubit"):
        qif(0).then(CX, 1).else_(Z, 1)


def test_qif_rejects_measure_branch():
    reset()
    with pytest.raises(ValueError, match="unitary"):
        qif(0).then(MEASURE, 1).else_(Z, 1)


def test_qif_then_equals_else():
    # 两分支相同 → 退化为无条件门，不残留 CX/旋转
    reset()
    qif(0).then(X, 1).else_(X, 1)
    ops = current_circuit().ops
    assert [op.name for op in ops] == ["x"]
    assert ops[0].qubits == (1,)


def test_qif_both_identity():
    # then=I 且 else=I → 整体为空
    reset()
    qif(0).then(I, 1).else_(I, 1)
    assert current_circuit().ops == []


def test_qif_rejects_control_equals_target():
    reset()
    with pytest.raises(ValueError, match="cannot be the same"):
        qif(0).then(X, 0).else_(Z, 0)


# ---------------------------------------------------------------------------
# 6. controlled —— 通用受控单比特门（编译成 |0><0|⊗I + |1><1|⊗U）
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("c,t", [(0, 1), (1, 0)])
def test_controlled_x_matches_block(c, t):
    reset()
    controlled(X, c, t)
    ops = current_circuit().ops

    got = _circuit_matrix(ops, 2)
    P0 = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex)
    P1 = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=complex)
    Im = _single_matrix("i")
    Xm = _single_matrix("x")
    if c == 0:  # 控制是最低位 → 控制投影在 kron 右侧
        expected = np.kron(Im, P0) + np.kron(Xm, P1)
    else:  # 控制是最高位 → 控制投影在 kron 左侧
        expected = np.kron(P0, Im) + np.kron(P1, Xm)
    assert np.allclose(got, expected, atol=1e-9)


def test_controlled_ry():
    reset()
    theta = 0.7
    controlled(Ry(theta), 0, 1)
    ops = current_circuit().ops

    got = _circuit_matrix(ops, 2)
    P0 = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex)
    P1 = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=complex)
    expected = np.kron(_single_matrix("i"), P0) + np.kron(
        _single_matrix("ry", (theta,)), P1
    )
    assert np.allclose(got, expected, atol=1e-9)


def test_controlled_uses_only_basic_gates():
    reset()
    controlled(Ry(0.7), 0, 1)
    basic = {"h", "x", "y", "z", "rx", "ry", "rz", "p", "cx", "cz"}
    for op in current_circuit().ops:
        assert op.name in basic, f"controlled 编译出了非基础门 '{op.name}'"


def test_controlled_rejects_wrong_target_count():
    reset()
    with pytest.raises(ValueError, match="requires 2 target"):
        controlled(CX, 0, 1)  # CX needs 2 targets, only 1 given


def test_controlled_rejects_measure():
    reset()
    with pytest.raises(ValueError, match="unitary"):
        controlled(MEASURE, 0, 1)


def test_controlled_rejects_control_equals_target():
    reset()
    with pytest.raises(ValueError, match="cannot be the same"):
        controlled(X, 0, 0)


# ---------------------------------------------------------------------------
# 7. 懒加载：import quonic 不引入 numpy / matplotlib
# ---------------------------------------------------------------------------

def test_import_quonic_does_not_load_numpy():
    import subprocess
    import sys

    code = (
        "import sys, quonic; "
        "print('numpy' in sys.modules, 'matplotlib' in sys.modules)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, check=False, text=True, timeout=60
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "False False"
