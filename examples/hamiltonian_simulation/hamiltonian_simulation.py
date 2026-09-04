"""Hamiltonian simulation / 哈密顿量模拟

Simulate e^{-iHt} for given Hamiltonian.
模拟给定哈密顿量的 e^{-iHt}。

## Application / 应用场景
- Quantum chemistry (量子化学)
- Material science (材料科学)
- Quantum simulation (量子模拟)

## Output / 输出
Evolved state under Hamiltonian evolution.
哈密顿量演化下的演化态。"""

from quonic.algorithms import hamiltonian_simulation

result = hamiltonian_simulation()
print(result.counts)
