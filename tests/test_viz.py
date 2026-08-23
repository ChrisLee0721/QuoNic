"""全量可视化套件测试：12 类图都能跑、返回正确类型、关键数据正确。"""

import numpy as np
import pytest

pytest.importorskip("matplotlib")
import matplotlib

matplotlib.use("Agg")

from quonic import CouplingMap, Result, viz
from quonic.ir import Circuit, GateOperation


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    import matplotlib.pyplot as plt

    plt.close("all")


@pytest.fixture
def bell():
    c = Circuit()
    c.add(GateOperation("h", (0,)))
    c.add(GateOperation("cx", (0, 1)))
    c.add(GateOperation("measure", (0,)))
    c.add(GateOperation("measure", (1,)))
    return c


# ---------------------------------------------------------------------------
# 1. 门序列电路图
# ---------------------------------------------------------------------------

def test_plot_circuit_returns_axes(bell):
    ax = viz.plot_circuit(bell)
    assert ax is not None


def test_plot_circuit_multi_qubit():
    c = Circuit()
    c.add(GateOperation("ccx", (0, 1, 2)))
    c.add(GateOperation("mcz", (0, 1, 2, 3)))
    ax = viz.plot_circuit(c)
    assert ax is not None


# ---------------------------------------------------------------------------
# 2. 测量直方图
# ---------------------------------------------------------------------------

def test_plot_counts_result():
    ax = viz.plot_counts(Result.from_counts({"00": 400, "11": 600}, 1000))
    assert len(ax.patches) == 2


def test_plot_counts_dict():
    ax = viz.plot_counts({"0": 1, "1": 3})
    assert len(ax.patches) == 2


def test_plot_counts_invalid():
    with pytest.raises(TypeError):
        viz.plot_counts(123)


# ---------------------------------------------------------------------------
# 3. 耦合拓扑图
# ---------------------------------------------------------------------------

def test_plot_coupling_map_line():
    ax = viz.plot_coupling_map(CouplingMap.from_line(4))
    assert ax is not None


def test_plot_coupling_map_grid():
    ax = viz.plot_coupling_map(CouplingMap.from_grid(2, 2))
    assert ax is not None


# ---------------------------------------------------------------------------
# 4. 方法对比折线图
# ---------------------------------------------------------------------------

def test_plot_method_comparison_clifford():
    ax = viz.plot_method_comparison("clifford")
    assert len(ax.lines) >= 2


def test_plot_method_comparison_low_tw():
    ax = viz.plot_method_comparison("low_tw")
    assert ax is not None


def test_plot_method_comparison_invalid():
    with pytest.raises(ValueError):
        viz.plot_method_comparison("nope")


# ---------------------------------------------------------------------------
# 5. 调度决策树
# ---------------------------------------------------------------------------

def test_plot_decision_tree():
    ax = viz.plot_decision_tree()
    assert ax is not None


# ---------------------------------------------------------------------------
# 6. 方法选择热力图
# ---------------------------------------------------------------------------

def test_plot_method_heatmap():
    ax = viz.plot_method_heatmap()
    assert ax is not None


# ---------------------------------------------------------------------------
# 7. 降级链路径图
# ---------------------------------------------------------------------------

def test_plot_fallback_chain():
    ax = viz.plot_fallback_chain()
    assert ax is not None


# ---------------------------------------------------------------------------
# 8. 量子比特活跃度热力图
# ---------------------------------------------------------------------------

def test_plot_qubit_activity(bell):
    ax = viz.plot_qubit_activity(bell)
    assert ax is not None


# ---------------------------------------------------------------------------
# 9. 电路特征雷达图
# ---------------------------------------------------------------------------

def test_plot_feature_radar_circuit(bell):
    ax = viz.plot_feature_radar(bell)
    assert ax is not None


def test_plot_feature_radar_features(bell):
    from quonic.scheduler import circuit_features

    ax = viz.plot_feature_radar(circuit_features(bell))
    assert ax is not None


# ---------------------------------------------------------------------------
# 10. 能量收敛图
# ---------------------------------------------------------------------------

def test_plot_energy_convergence_list():
    ax = viz.plot_energy_convergence([1.0, 0.5, 0.3])
    assert ax is not None


def test_plot_energy_convergence_result():
    r = Result.from_value(0.2, history=[1.0, 0.5, 0.2])
    ax = viz.plot_energy_convergence(r)
    assert ax is not None


def test_plot_energy_convergence_missing_history():
    r = Result.from_value(0.2)
    with pytest.raises(ValueError):
        viz.plot_energy_convergence(r)


# ---------------------------------------------------------------------------
# 11. Grover 迭代振幅图
# ---------------------------------------------------------------------------

def test_plot_grover_amplitudes_peak():
    ax = viz.plot_grover_amplitudes(2, "11", iterations=1)
    probs = ax.lines[0].get_ydata()
    assert probs[-1] == pytest.approx(1.0, abs=1e-6)


def test_plot_grover_amplitudes_invalid():
    with pytest.raises(ValueError):
        viz.plot_grover_amplitudes(2, "111")


# ---------------------------------------------------------------------------
# 12. 态向量可视化
# ---------------------------------------------------------------------------

def test_plot_statevector_array():
    sv = np.array([1, 0, 0, 0], dtype=complex)
    axes = viz.plot_statevector(sv)
    assert len(axes) == 2


def test_plot_statevector_circuit():
    c = Circuit()
    c.add(GateOperation("h", (0,)))
    axes = viz.plot_statevector(c)
    assert len(axes) == 2


def test_plot_statevector_topk():
    # 10 比特态向量（1024 个基态）默认只画前 32 个
    sv = np.zeros(1024, dtype=complex)
    sv[0] = 1.0
    axes = viz.plot_statevector(sv)
    assert len(axes[0].patches) == 32


def test_plot_counts_topk():
    # 30 个比特串默认只画次数最多的前 20 个
    counts = {f"{i:04b}": i + 1 for i in range(30)}
    ax = viz.plot_counts(counts)
    assert len(ax.patches) == 20


# ---------------------------------------------------------------------------
# 能量收敛图与算法的集成（record_history）
# ---------------------------------------------------------------------------

def test_vqe_record_history():
    from quonic.algorithms import vqe

    hamiltonian = [(1.0, "ZZ"), (1.0, "XI"), (1.0, "IX")]
    r = vqe(hamiltonian, 2, maxiter=10, record_history=True)
    assert "history" in r.metadata
    assert len(r.metadata["history"]) > 0


def test_qaoa_record_history():
    from quonic.algorithms import qaoa_maxcut

    r = qaoa_maxcut([(0, 1), (1, 2), (0, 2)], 3, maxiter=10, record_history=True)
    assert "history" in r.metadata
    assert len(r.metadata["history"]) > 0


# ---------------------------------------------------------------------------
# 13. 噪声成本热力图
# ---------------------------------------------------------------------------

def test_plot_noise_heatmap():
    ax = viz.plot_noise_heatmap(n_values=(2, 4), noise_rates=(0.0, 0.1))
    assert ax is not None


def test_plot_noise_heatmap_infeasible():
    ax = viz.plot_noise_heatmap(n_values=(2, 4), noise_rates=(0.0, 0.1), budget=1e-9)
    assert ax is not None


# ---------------------------------------------------------------------------
# 14. 布洛赫球
# ---------------------------------------------------------------------------

def test_plot_bloch_sphere_array():
    ax = viz.plot_bloch_sphere(np.array([1, 0], dtype=complex))
    assert ax is not None


def test_plot_bloch_sphere_circuit():
    c = Circuit()
    c.add(GateOperation("h", (0,)))
    ax = viz.plot_bloch_sphere(c)
    assert ax is not None


def test_bloch_vector_plus_state():
    from quonic.viz.state import _bloch_vector

    x, y, z = _bloch_vector(np.array([1, 1], dtype=complex) / np.sqrt(2))
    assert x == pytest.approx(1.0)
    assert y == pytest.approx(0.0, abs=1e-9)
    assert z == pytest.approx(0.0, abs=1e-9)


def test_plot_bloch_multivector():
    # 10 比特 GHZ 态 → 每个比特的约化态都是完全混合态（球心）
    ghz = np.zeros(1024, dtype=complex)
    ghz[0] = ghz[-1] = 1.0
    ghz /= np.sqrt(2)
    axes = viz.plot_bloch_multivector(ghz, cols=5)
    assert len(axes) == 10


def test_plot_bloch_multivector_annotate():
    # annotate=True 时每个球下方多一个 (x,y,z) 标注文本
    ghz = np.zeros(1024, dtype=complex)
    ghz[0] = ghz[-1] = 1.0
    ghz /= np.sqrt(2)
    axes = viz.plot_bloch_multivector(ghz, cols=5, annotate=True)
    assert len(axes) == 10
    # 球内文本：左上是 label，下方是 (x,y,z) 标注
    assert len(axes[0].texts) >= 2


def test_plot_bloch_multivector_product_state():
    # 10 比特直积态 |0...0> → 每个比特都指向 +z（|0>）
    sv = np.zeros(1024, dtype=complex)
    sv[0] = 1.0
    from quonic.viz.state import _partial_trace, _rho_bloch_vector

    rho = np.outer(sv, sv.conj())
    for q in range(10):
        rho_q = _partial_trace(rho, [q], 10)
        _x, _y, z = _rho_bloch_vector(rho_q)
        assert z == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# 15. 密度矩阵热力图
# ---------------------------------------------------------------------------

def test_plot_density_matrix_array():
    rho = np.array([[0.5, 0], [0, 0.5]], dtype=complex)
    axes = viz.plot_density_matrix(rho)
    assert len(axes) == 2


def test_plot_density_matrix_circuit():
    c = Circuit()
    c.add(GateOperation("h", (0,)))
    axes = viz.plot_density_matrix(c)
    assert len(axes) == 2


# ---------------------------------------------------------------------------
# 16. 纠缠可视化
# ---------------------------------------------------------------------------

def test_plot_entanglement_bell():
    sv = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
    ax = viz.plot_entanglement(sv, partition=[0])
    assert ax is not None


def test_bell_state_entropy_is_one():
    from quonic.viz.state import _partial_trace, _von_neumann_entropy

    sv = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
    rho = np.outer(sv, sv.conj())
    rho_a = _partial_trace(rho, [0], 2)
    eigvals = np.linalg.eigvalsh(rho_a)
    assert _von_neumann_entropy(eigvals) == pytest.approx(1.0, abs=1e-9)


def test_product_state_entropy_is_zero():
    from quonic.viz.state import _partial_trace, _von_neumann_entropy

    sv = np.array([1, 0, 0, 0], dtype=complex)
    rho = np.outer(sv, sv.conj())
    rho_a = _partial_trace(rho, [0], 2)
    eigvals = np.linalg.eigvalsh(rho_a)
    assert _von_neumann_entropy(eigvals) == pytest.approx(0.0, abs=1e-9)


def test_concurrence_bell_is_one():
    from quonic.viz.state import _concurrence

    sv = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
    rho = np.outer(sv, sv.conj())
    assert _concurrence(rho) == pytest.approx(1.0, abs=1e-9)


def test_concurrence_classical_mixed_is_zero():
    # 测量坍缩后的经典关联态 1/2|00><00| + 1/2|11><11|：无量子纠缠
    from quonic.viz.state import _concurrence

    rho = np.diag([0.5, 0.0, 0.0, 0.5]).astype(complex)
    assert _concurrence(rho) == pytest.approx(0.0, abs=1e-9)


def test_concurrence_product_is_zero():
    from quonic.viz.state import _concurrence

    rho = np.diag([1.0, 0.0, 0.0, 0.0]).astype(complex)  # |00>
    assert _concurrence(rho) == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# 17. 门矩阵可视化
# ---------------------------------------------------------------------------

def test_plot_gate_matrix_h():
    axes = viz.plot_gate_matrix("h")
    assert len(axes) == 2


def test_plot_gate_matrix_cx():
    axes = viz.plot_gate_matrix("cx")
    assert len(axes) == 2


def test_plot_gate_matrix_gate_object():
    from quonic import gates

    axes = viz.plot_gate_matrix(gates.CX)
    assert len(axes) == 2


def test_plot_gate_matrix_measure_invalid():
    with pytest.raises(ValueError):
        viz.plot_gate_matrix("measure")


# ---------------------------------------------------------------------------
# 18. SWAP 路由可视化
# ---------------------------------------------------------------------------

def test_route_swaps_inserts_swap():
    from quonic.compiler import route_swaps

    c = Circuit()
    c.add(GateOperation("cx", (0, 2)))
    routed = route_swaps(c, CouplingMap.from_line(3))
    assert any(op.name == "swap" for op in routed.ops)


def test_plot_routing():
    c = Circuit()
    c.add(GateOperation("cx", (0, 2)))
    ax = viz.plot_routing(c, CouplingMap.from_line(3))
    assert ax is not None


# ---------------------------------------------------------------------------
# 19. 逐门态演化
# ---------------------------------------------------------------------------

def test_plot_state_evolution(bell):
    ax = viz.plot_state_evolution(bell)
    assert ax is not None


# ---------------------------------------------------------------------------
# 20. 问题图（QAOA MaxCut）
# ---------------------------------------------------------------------------

def test_plot_problem_graph():
    ax = viz.plot_problem_graph([(0, 1), (1, 2), (0, 2)])
    assert ax is not None


def test_plot_problem_graph_with_partition():
    ax = viz.plot_problem_graph([(0, 1), (1, 2), (0, 2)], partition={0: 0, 1: 1, 2: 1})
    assert ax is not None


# ---------------------------------------------------------------------------
# 21. 哈密顿量可视化
# ---------------------------------------------------------------------------

def test_plot_hamiltonian():
    axes = viz.plot_hamiltonian([(1.0, "ZZ"), (1.0, "XI"), (1.0, "IX")])
    assert len(axes) == 2


# ---------------------------------------------------------------------------
# 22. 纠缠熵谱
# ---------------------------------------------------------------------------

def test_plot_entanglement_profile():
    sv = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
    ax = viz.plot_entanglement_profile(sv)
    assert ax is not None


def test_ghz_entanglement_profile_all_one():
    from quonic.viz.state import _partial_trace, _to_density, _von_neumann_entropy

    sv = np.array([1, 0, 0, 0, 0, 0, 0, 1], dtype=complex) / np.sqrt(2)
    rho = _to_density(sv)
    for k in range(2):
        rho_a = _partial_trace(rho, list(range(k + 1)), 3)
        eig = np.linalg.eigvalsh(rho_a)
        assert _von_neumann_entropy(eig) == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# 23. 噪声叠加电路图
# ---------------------------------------------------------------------------

def test_plot_noisy_circuit(bell):
    ax = viz.plot_noisy_circuit(bell, noise=0.05)
    assert ax is not None
