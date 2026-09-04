"""Estimate success probability / 估计成功概率

Quantum algorithm to estimate the amplitude of a marked state.
量子算法估计标记态的振幅。

## Application / 应用场景
- Monte Carlo integration (蒙特卡洛积分)
- Risk analysis (风险分析)
- Option pricing (期权定价)

## Output / 输出
Estimated amplitude with quadratic speedup over classical.
估计振幅，相比经典有二次加速。"""

from quonic.algorithms import amplitude_estimation

result = amplitude_estimation(n_qubits=2, n_precision=3, shots=1024)
print(result.counts)
