"""Stabilizer Formalism / 稳定子形式

Clifford group simulation via stabilizer tableau.
通过稳定子表模拟 Clifford 群。

## Application / 应用场景
- Error correction (纠错)
- Clifford simulation (Clifford 模拟)
- Quantum circuits (量子电路)

## Output / 输出
Stabilizer state measurements.
稳定子态测量。"""

from quonic.algorithms import stabilizer

result = stabilizer(n_qubits=3, shots=100)
print(result.counts)
