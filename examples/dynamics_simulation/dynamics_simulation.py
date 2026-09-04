"""Quantum dynamics simulation / 量子动力学模拟

Simulate time evolution of quantum systems.
模拟量子系统的时间演化。

## Application / 应用场景
- Quantum chemistry (量子化学)
- Material science (材料科学)
- Condensed matter (凝聚态)

## Output / 输出
Evolved state after time t.
时间 t 后的演化态。"""

from quonic.algorithms import dynamics_simulation

result = dynamics_simulation(n_steps=10, shots=1024)
print(result.counts)
