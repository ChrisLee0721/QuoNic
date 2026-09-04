"""Quantum Monte Carlo / 量子蒙特卡洛

Quantum speedup for Monte Carlo methods.
蒙特卡洛方法的量子加速。

## Application / 应用场景
- Integration (积分)
- Risk analysis (风险分析)
- Finance (金融)

## Output / 输出
Estimated integral value.
估计积分值。"""

from quonic.algorithms import quantum_monte_carlo

result = quantum_monte_carlo(n_qubits=2, shots=1024)
print(f"Estimated value: {result.value}")
