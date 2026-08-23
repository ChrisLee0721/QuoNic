"""比较器（qlt / qeq / qgt）与乘法（mul）测试。"""

import pytest

from quonic import QInt, mul, qeq, qgt, qlt, reset
from quonic.backends import get_backend
from quonic.stack import current_circuit


def _run(shots=256, backend="native"):
    return get_backend(backend).run(current_circuit(), shots=shots)


def _bit(bs, q):
    """bs 为 MSB 在前的比特串，返回量子比特 q（q=0 为最低位）的取值。"""
    return int(bs[len(bs) - 1 - q])


def _reg(bs, qubits):
    """寄存器取值：qubits[0] 是寄存器最低位。"""
    return sum(_bit(bs, q) << i for i, q in enumerate(qubits))


# ---------------------------------------------------------------------------
# qeq：x == k
# ---------------------------------------------------------------------------

def test_qeq_equal():
    reset()
    x = QInt(3, value=5)
    flag = qeq(x, 5)
    bs = next(iter(_run().counts))
    assert _bit(bs, flag) == 1
    assert _reg(bs, x.qubits) == 5


def test_qeq_not_equal():
    reset()
    x = QInt(3, value=5)
    flag = qeq(x, 3)
    bs = next(iter(_run().counts))
    assert _bit(bs, flag) == 0
    assert _reg(bs, x.qubits) == 5


# ---------------------------------------------------------------------------
# qlt：x < k
# ---------------------------------------------------------------------------

def test_qlt_less():
    reset()
    x = QInt(3, value=2)
    flag = qlt(x, 5)
    bs = next(iter(_run().counts))
    assert _bit(bs, flag) == 1
    assert _reg(bs, x.qubits) == 2


def test_qlt_equal_not_less():
    reset()
    x = QInt(3, value=5)
    flag = qlt(x, 5)
    bs = next(iter(_run().counts))
    assert _bit(bs, flag) == 0


def test_qlt_greater_not_less():
    reset()
    x = QInt(3, value=6)
    flag = qlt(x, 5)
    bs = next(iter(_run().counts))
    assert _bit(bs, flag) == 0


# ---------------------------------------------------------------------------
# qgt：x > k
# ---------------------------------------------------------------------------

def test_qgt_greater():
    reset()
    x = QInt(3, value=6)
    flag = qgt(x, 5)
    bs = next(iter(_run().counts))
    assert _bit(bs, flag) == 1


def test_qgt_equal_not_greater():
    reset()
    x = QInt(3, value=5)
    flag = qgt(x, 5)
    bs = next(iter(_run().counts))
    assert _bit(bs, flag) == 0


# ---------------------------------------------------------------------------
# 比较器在叠加态下与 x 正确关联
# ---------------------------------------------------------------------------

def test_qlt_superposition_consistent():
    reset()
    x = QInt(3)
    x.h()  # 均匀叠加 |0..7>
    flag = qlt(x, 4)
    result = _run(shots=4096)
    total = sum(result.counts.values())
    # flag=1 的比例应约 4/8 = 50%
    flag1 = sum(c for bs, c in result.counts.items() if _bit(bs, flag) == 1)
    assert 0.4 < flag1 / total < 0.6
    # 每个 flag=1 的样本，x 都在 [0,4) 内；flag=0 的样本 x >= 4
    for bs in result.counts:
        xv = _reg(bs, x.qubits)
        assert (_bit(bs, flag) == 1) == (xv < 4), f"x={xv} flag={_bit(bs, flag)}"


# ---------------------------------------------------------------------------
# mul：乘法（结果寄存器）
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("k", [1, 2, 3, 5, 6, 7])
def test_mul_preserves_x_and_multiplies(k):
    reset()
    x = QInt(3, value=3)
    p = mul(x, k)
    bs = next(iter(_run().counts))
    assert _reg(bs, x.qubits) == 3
    assert _reg(bs, p.qubits) == (3 * k) % 8


def test_mul_even_k_uses_result_register():
    # 偶数 k 就地乘法不可逆，但结果寄存器干净，因此仍成立
    reset()
    x = QInt(3, value=3)
    p = mul(x, 2)
    bs = next(iter(_run().counts))
    assert _reg(bs, p.qubits) == 6  # 3*2 mod 8 = 6


def test_mul_qint_method():
    reset()
    x = QInt(2, value=3)
    p = x.mul(3)
    bs = next(iter(_run().counts))
    assert _reg(bs, x.qubits) == 3
    assert _reg(bs, p.qubits) == 1  # 9 mod 4 = 1


# ---------------------------------------------------------------------------
# 错误分支
# ---------------------------------------------------------------------------

def test_compare_rejects_non_qint():
    reset()
    with pytest.raises(TypeError, match="QInt"):
        qlt([0, 1, 2], 5)
