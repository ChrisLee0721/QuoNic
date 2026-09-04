"""Color code error correction / 颜色码纠错

Topological error correction code with transversal gates.
具有横向门的拓扑纠错码。

## Application / 应用场景
- Fault-tolerant quantum computing (容错量子计算)
- Topological codes (拓扑码)
- Quantum memory (量子存储)

## Output / 输出
Encoded logical qubit with error protection.
具有错误保护的编码逻辑比特。"""

from quonic.algorithms import color_code

result = color_code(shots=100)
print(result.counts)
