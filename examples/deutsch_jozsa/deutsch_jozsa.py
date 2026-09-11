"""Constant or balanced function? / 常数还是平衡函数？

Determine if f is constant or balanced in one query.
一次查询确定 f 是常数还是平衡函数。

## Application / 应用场景
- Oracle complexity (预言机复杂度)
- Quantum advantage (量子优势)
- Function analysis (函数分析)

## Output / 输出
All zeros = constant, anything else = balanced.
全零 = 常数，其他 = 平衡。"""

from quonic.algorithms import deutsch_jozsa
from quonic.ir import GateOperation

N = 3

def balanced_oracle(circuit, n):
    """Balanced oracle: flip last qubit if first qubit is |1>."""
    circuit.add(GateOperation("cx", (0, n)))

result = deutsch_jozsa(N, balanced_oracle, shots=100)
print(f"Is balanced: {result.metadata.get('is_balanced', 'unknown')}")
print(f"Counts: {result.metadata.get('counts', {})}")
