"""噪声成本曲线 + QFT/Grover 验证点的基准与查询测试。"""

from quonic.scheduler import benchmark as bench
from quonic.scheduler import (
    circuit_features,
    decision_class,
    load_noise_cost,
    recommend_method,
)


def test_load_noise_cost_returns_dict():
    d = load_noise_cost()
    assert isinstance(d, dict)
    # 若随包附带参考表，噪声成本结构应合法
    if d:
        assert d.get("method") == "density_matrix"
        assert "performance" in d
        for r in d["performance"]:
            assert r["n"] > 0 and r["time"] > 0
        if d.get("infeasible_n") is not None:
            assert d["infeasible_n"] > 0


def test_qft_and_grover_are_general():
    # 高树宽非 Clifford：分类应为 general，方法应留在 statevector
    for fn in (bench._qft, bench._grover):
        feats = circuit_features(fn(6))
        assert decision_class(feats) == "general"
        assert recommend_method(feats).method == "statevector"


def test_benchmark_noise_infeasible_threshold(monkeypatch):
    # 用假耗时曲线验证 infeasible_n 推导：n=6 首次超预算
    fake = {2: 0.01, 4: 0.05, 6: 1.5, 8: 6.0}
    monkeypatch.setattr(
        bench, "_timed_run",
        lambda circuit, backend, method, shots=256, noise=None: fake[circuit.num_qubits],
    )
    result = bench.benchmark_noise((2, 4, 6, 8), budget=1.0, repeats=1)
    assert result["method"] == "density_matrix"
    assert result["infeasible_n"] == 6
    assert [r["n"] for r in result["performance"]] == [2, 4, 6, 8]


def test_benchmark_general_structure(monkeypatch):
    monkeypatch.setattr(
        bench, "_timed_run",
        lambda circuit, backend, method, shots=256, noise=None: 0.01,
    )
    result = bench.benchmark_general((8, 12), repeats=1)
    names = {(r["circuit"], r["n"]) for r in result}
    assert ("qft", 8) in names
    assert ("grover", 12) in names
