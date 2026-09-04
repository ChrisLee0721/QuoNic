"""Quantum Topological Data Analysis / 量子拓扑数据分析

Quantum algorithm for persistent homology.
持续同调的量子算法。

## Application / 应用场景
- Data analysis (数据分析)
- Shape recognition (形状识别)
- Topology (拓扑学)

## Output / 输出
Topological features.
拓扑特征。"""

from quonic.algorithms import qtda

result = qtda()
print(result.counts)
