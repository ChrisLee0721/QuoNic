"""Quantum Signal Processing / 量子信号处理

Core subroutine for quantum singular value transformation.
量子奇异值变换的核心子程序。

## Application / 应用场景
- Quantum algorithms (量子算法)
- Hamiltonian simulation (哈密顿量模拟)
- Eigenvalue problems (特征值问题)

## Output / 输出
Transformed signal.
变换后的信号。"""

from quonic.algorithms import qsp

result = qsp(angle=0.785)
print(result.counts)
