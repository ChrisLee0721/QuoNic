"""Quantum Annealing / 量子退火

Hybrid classical-quantum annealing for optimization.
用于优化的混合经典-量子退火。

## Application / 应用场景
- Optimization (优化)
- Combinatorial problems (组合问题)
- Sampling (采样)

## Output / 输出
Approximate ground state.
近似基态。"""

from quonic.algorithms import quantum_annealing_hybrid

result = quantum_annealing_hybrid(n_spins=4, n_steps=100)
print(result.counts)
