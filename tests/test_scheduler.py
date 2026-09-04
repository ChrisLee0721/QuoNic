"""调度器测试：特征提取 + 解析电路选 method + 查表链 + 本地缓存闭环。"""

import json

from quonic.ir import Circuit, GateOperation
from quonic.scheduler import (
    FileRegistry,
    LocalCacheRegistry,
    MemoryRegistry,
    circuit_features,
    recommend_method,
    schedule,
)


def _bell():
    c = Circuit()
    c.add(GateOperation("h", (0,)))
    c.add(GateOperation("cx", (0, 1)))
    return c


# ---------------------------------------------------------------------------
# 特征提取
# ---------------------------------------------------------------------------

def test_features_bell():
    f = circuit_features(_bell())
    assert f["n"] == 2
    assert f["depth"] == 2
    assert f["gate_types"] == ["cx", "h"]
    assert f["is_clifford"] is True
    assert f["treewidth_ub"] == 1
    assert "n<8|clifford|tw<4|" in f["key"]


def test_clifford_detection():
    # 任意角旋转非 Clifford
    c = Circuit()
    c.add(GateOperation("rz", (0,), (0.3,)))
    assert circuit_features(c)["is_clifford"] is False
    # 多控制 Z 是 Clifford
    c = Circuit()
    c.add(GateOperation("mcz", (0, 1, 2)))
    assert circuit_features(c)["is_clifford"] is True
    # Toffoli 非 Clifford
    c = Circuit()
    c.add(GateOperation("ccx", (0, 1, 2)))
    assert circuit_features(c)["is_clifford"] is False


def test_treewidth_chain():
    c = Circuit()
    c.add(GateOperation("cx", (0, 1)))
    c.add(GateOperation("cx", (1, 2)))
    assert circuit_features(c)["treewidth_ub"] == 1


# ---------------------------------------------------------------------------
# 解析电路选 method（冷启动能力）
# ---------------------------------------------------------------------------

def test_recommend_method_small_is_statevector():
    # 小电路（n<20）statevector 反而最快，stabilizer 固定开销不划算
    assert recommend_method(circuit_features(_bell())).method == "statevector"


def test_recommend_method_stabilizer():
    # n>=20 且基础 Clifford 门集 -> stabilizer
    c = Circuit()
    c.add(GateOperation("h", (0,)))
    for i in range(23):
        c.add(GateOperation("cx", (i, i + 1)))
    assert recommend_method(circuit_features(c)).method == "stabilizer"


def test_recommend_method_nonclifford():
    # 含任意角旋转 -> statevector
    c = Circuit()
    c.add(GateOperation("rz", (0,), (0.3,)))
    assert recommend_method(circuit_features(c)).method == "statevector"
    # mcz 是 Clifford 但 stabilizer 不支持 -> statevector
    c = Circuit()
    c.add(GateOperation("mcz", (0, 1, 2)))
    assert recommend_method(circuit_features(c)).method == "statevector"


def test_recommend_method_mps():
    # 低树宽 + 大比特数（n>=20）+ 非 Clifford -> matrix_product_state
    c = Circuit()
    c.add(GateOperation("rz", (0,), (0.1,)))
    for i in range(23):
        c.add(GateOperation("cx", (i, i + 1)))
    assert recommend_method(circuit_features(c)).method == "matrix_product_state"


# ---------------------------------------------------------------------------
# 查表链
# ---------------------------------------------------------------------------

def test_memory_registry():
    # Bell circuit is detected as "ghz" family, so key includes family prefix
    table = MemoryRegistry({"ghz|n<8|clifford|tw<4|d0": "qiskit"})
    assert schedule(_bell(), table=table).backend == "qiskit"


def test_file_registry(tmp_path):
    p = tmp_path / "params.json"
    p.write_text(json.dumps({"ghz|n<8|clifford|tw<4|d0": "pennylane"}), encoding="utf-8")
    table = FileRegistry(str(p))
    assert schedule(_bell(), table=table).backend == "pennylane"


def test_local_cache_roundtrip(tmp_path):
    p = tmp_path / "cache.json"
    cache = LocalCacheRegistry(str(p))
    c = _bell()
    # 第一次查不到 -> 回退规则（QPanda 是默认后端）
    assert schedule(c, cache=cache).backend == "qpanda"
    # 记录一次运行结果 -> 写入缓存
    cache.report_result(circuit_features(c), "cirq", 0.1, 100)
    assert schedule(c, cache=cache).backend == "cirq"
    # 持久化后重新加载仍命中
    cache2 = LocalCacheRegistry(str(p))
    assert schedule(c, cache=cache2).backend == "cirq"


def test_schedule_priority(tmp_path):
    p = tmp_path / "cache.json"
    cache = LocalCacheRegistry(str(p))
    table = MemoryRegistry({"ghz|n<8|clifford|tw<4|d0": "pennylane"})
    c = _bell()
    # 无 cache 记录时，静态表生效
    assert schedule(c, cache=cache, table=table).backend == "pennylane"
    # cache 记录后，cache 优先于静态表
    cache.report_result(circuit_features(c), "cirq", 0.1, 100)
    assert schedule(c, cache=cache, table=table).backend == "cirq"


def test_schedule_includes_method():
    # schedule 的 method 来自解析电路（Bell n=2 <20 -> statevector）
    rec = schedule(_bell())
    assert rec.backend == "qpanda"
    assert rec.method == "statevector"


def test_micro_tuning_stable_key():
    # 微调（加几个单比特门，depth 增加）不应改变分桶 key，从而命中缓存
    c1 = _bell()
    c2 = _bell()
    for _ in range(5):
        c2.add(GateOperation("h", (0,)))
    assert circuit_features(c1)["key"] == circuit_features(c2)["key"]


def test_qshow_cache_integration(tmp_path, capsys):
    # qshow 传入 cache 后：第一次跑走规则兜底，跑完写回缓存
    from quonic import qgate, qshow, reset
    from quonic.gates import CX, H

    p = tmp_path / "cache.json"
    cache = LocalCacheRegistry(str(p))

    reset()
    qgate(H, 0)
    qgate(CX, 0, 1)
    # Use native backend explicitly to avoid qiskit dependency
    qshow(backend="native", cache=cache, shots=64)
    capsys.readouterr()

    # 跑完应把「特征 -> 后端:method」写入缓存文件
    cache2 = LocalCacheRegistry(str(p))
    assert len(cache2.table) == 1
    assert any(v.startswith("native") for v in cache2.table.values())
