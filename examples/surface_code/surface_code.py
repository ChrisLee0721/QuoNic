"""Surface Code / 表面码

Leading candidate for fault-tolerant quantum computing.
容错量子计算的主要候选方案。

## Application / 应用场景
- Fault tolerance (容错)
- Quantum memory (量子存储)
- Logical qubits (逻辑比特)

## Output / 输出
Logical qubit with error protection.
具有错误保护的逻辑比特。"""

from quonic.algorithms import surface_code

result = surface_code(distance=3, shots=100)
print(result.counts)
