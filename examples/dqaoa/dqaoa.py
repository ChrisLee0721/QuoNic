"""Dynamic QAOA / 动态 QAOA

Adaptive layer QAOA that adds layers until convergence.
自适应层 QAOA，添加层直到收敛。

## Application / 应用场景
- Combinatorial optimization (组合优化)
- MaxCut (最大割)
- Scheduling (调度)

## Output / 输出
Approximate optimal solution.
近似最优解。"""

from quonic.algorithms import dqaoa

result = dqaoa()
print(result.counts)
