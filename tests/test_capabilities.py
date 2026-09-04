"""能力矩阵 + 实测数据驱动决策 + 噪声硬约束测试。"""


from quonic.ir import Circuit, GateOperation
from quonic.scheduler import (
    METHOD_CAPABILITIES,
    circuit_features,
    decision_class,
    eligible_methods,
    load_measured_decision,
    recommend_method,
    schedule,
)


def _clifford(n):
    c = Circuit()
    c.add(GateOperation("h", (0,)))
    for i in range(n - 1):
        c.add(GateOperation("cx", (i, i + 1)))
    return c


def _bell():
    c = Circuit()
    c.add(GateOperation("h", (0,)))
    c.add(GateOperation("cx", (0, 1)))
    return c


# ---------------------------------------------------------------------------
# 能力矩阵（静态硬约束）
# ---------------------------------------------------------------------------

def test_method_capabilities_structure():
    for m in ("statevector", "stabilizer", "matrix_product_state", "density_matrix"):
        cap = METHOD_CAPABILITIES[m]
        assert "noise" in cap and "gates" in cap
        assert isinstance(cap["noise"], bool)


def test_eligible_methods_basic_clifford():
    gs = {"h", "cx", "z"}
    assert eligible_methods(gs) == {"statevector", "stabilizer", "matrix_product_state"}


def test_eligible_methods_nonclifford_excludes_stabilizer():
    # 任意角旋转非 Clifford，stabilizer 不支持
    assert "stabilizer" not in eligible_methods({"rx", "cx"})


def test_eligible_methods_mcz_excludes_stabilizer():
    # mcz 是 Clifford 但 Aer 的 stabilizer 不吃它
    assert "stabilizer" not in eligible_methods({"mcz"})


def test_eligible_methods_noise_only_density():
    assert eligible_methods({"h", "cx"}, noise=True) == {"density_matrix"}


def test_decision_class():
    assert decision_class(circuit_features(_clifford(8))) == "clifford"
    # 低树宽 + 非 Clifford -> low_tw
    c = Circuit()
    c.add(GateOperation("rz", (0,), (0.3,)))
    for i in range(7):
        c.add(GateOperation("cx", (i, i + 1)))
    assert decision_class(circuit_features(c)) == "low_tw"


# ---------------------------------------------------------------------------
# 噪声硬约束
# ---------------------------------------------------------------------------

def test_recommend_method_noise_forces_density():
    # 大 Clifford 电路，噪声开启时方法必须是 density_matrix（而非 stabilizer）
    c = _clifford(24)
    assert recommend_method(circuit_features(c), noise=True).method == "density_matrix"


def test_schedule_threads_noise():
    rec = schedule(_bell(), noise=True)
    assert rec.method == "density_matrix"


# ---------------------------------------------------------------------------
# 实测数据驱动决策（覆盖冷启动规则）
# ---------------------------------------------------------------------------

def test_recommend_method_uses_measured_decision(monkeypatch):
    import quonic.scheduler.registry as reg

    # 实测表说 clifford 交叉点在 n=10（比默认 24 更激进）
    monkeypatch.setattr(
        reg, "load_measured_decision",
        lambda: {"clifford": {"method": "stabilizer", "above_n": 10}},
    )
    c = _clifford(12)  # n=12 >= 10，应路由到 stabilizer
    assert recommend_method(circuit_features(c)).method == "stabilizer"


def test_recommend_method_fallback_when_no_data(monkeypatch):
    import quonic.scheduler.registry as reg

    # 无实测数据时回退默认阈值（n=20）
    monkeypatch.setattr(reg, "load_measured_decision", dict)
    assert recommend_method(circuit_features(_clifford(24))).method == "stabilizer"
    assert recommend_method(circuit_features(_clifford(20))).method == "stabilizer"
    assert recommend_method(circuit_features(_clifford(16))).method == "statevector"


def test_load_measured_decision_returns_dict():
    d = load_measured_decision()
    assert isinstance(d, dict)
    # 随包附带的参考表若存在，应含 clifford / low_tw 两类且阈值合法
    for cls in ("clifford", "low_tw"):
        if cls in d:
            assert d[cls]["above_n"] > 0
            assert d[cls]["method"] in ("stabilizer", "matrix_product_state")
