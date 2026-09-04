"""Eigenvalue Estimation / 特征值估计

Estimate eigenvalues of unitary operators.
估计酉算子的特征值。

## Application / 应用场景
- Quantum chemistry (量子化学)
- Physics (物理学)
- Linear algebra (线性代数)

## Output / 输出
Eigenvalue estimates.
特征值估计。"""

from quonic.algorithms import quantum_eigenvalue

result = quantum_eigenvalue()
print(result.counts)
