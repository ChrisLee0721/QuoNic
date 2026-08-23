"""Quantum Optimization Workflow / 量子优化工作流

Complete workflow for combinatorial optimization with QAOA.
QAOA 组合优化的完整工作流。

## Application / 应用场景
- Logistics (物流)
- Scheduling (调度)
- Network design (网络设计)

## Output / 输出
Optimized solution with multi-backend comparison.
多后端对比的优化解。"""

from quonic import qgate, qshow, reset
from quonic.algorithms import qaoa_maxcut
from quonic.compiler import decompose, optimize
from quonic.gates import CX, H
from quonic.ir import Circuit, GateOperation

# Define graph: triangle with edges (0,1), (1,2), (0,2)
edges = [(0, 1), (1, 2), (0, 2)]
n_qubits = 3

print("=== Quantum Optimization: MaxCut ===")
print(f"Graph: {n_qubits} vertices, edges = {edges}")
print("MaxCut = 2 (partition {0} vs {1,2})")
print()

# Method 1: QAOA
print("--- Method 1: QAOA ---")
result = qaoa_maxcut(edges, n_qubits, p=1, maxiter=200)
print(f"QAOA MaxCut value: {result.value:.2f}")
print()

# Method 2: Multi-backend comparison
print("--- Method 2: Multi-backend Comparison ---")
backends = ["native", "qulacs", "qiskit", "cirq"]
for backend in backends:
    try:
        reset()
        qgate(H, 0)
        qgate(CX, 0, 1)
        qgate(CX, 1, 2)
        result = qshow(backend=backend, shots=1024)
        print(f"{backend}: {result.counts}")
    except (ImportError, RuntimeError, ValueError) as e:
        print(f"{backend}: {e}")
print()

# Method 3: Circuit optimization
print("--- Method 3: Circuit Optimization ---")

circuit = Circuit()
circuit.allocate(3)
circuit.add(GateOperation("h", (0,)))
circuit.add(GateOperation("cx", (0, 1)))
circuit.add(GateOperation("cx", (1, 2)))
circuit.add(GateOperation("h", (0,)))
circuit.add(GateOperation("h", (1,)))
circuit.add(GateOperation("h", (2,)))

optimized = optimize(decompose(circuit))
print(f"Original ops: {len([op for op in circuit.ops if op.name != 'measure'])}")
print(f"Optimized ops: {len([op for op in optimized.ops if op.name != 'measure'])}")
print()

print("=== Summary ===")
print("QuoNic provides complete optimization workflow:")
print("1. QAOA for combinatorial optimization")
print("2. Multi-backend comparison")
print("3. Circuit optimization for better performance")
print("4. Smart scheduling for automatic backend selection")
