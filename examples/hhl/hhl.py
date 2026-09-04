"""Linear system solver / 线性方程组求解器

Quantum algorithm for Ax = b, exponential speedup.
量子算法求解 Ax = b，指数加速。

## Application / 应用场景
- Machine learning (机器学习)
- Optimization (优化)
- Differential equations (微分方程)

## Output / 输出
Quantum state proportional to x = A^{-1}b.
与 x = A^{-1}b 成正比的量子态。"""

from quonic.algorithms import hhl

result = hhl()
print(result.counts)
