"""Quantum Natural Gradient / 量子自然梯度

Uses quantum Fisher information for better optimization.
使用量子 Fisher 信息进行更好的优化。

## Application / 应用场景
- Variational algorithms (变分算法)
- VQE optimization (VQE 优化)
- Quantum ML (量子机器学习)

## Output / 输出
Optimized parameters with faster convergence.
更快收敛的优化参数。"""

from quonic.algorithms import qng

result = qng(n_params=2, maxiter=50)
print(f"Final loss: {result.value}")
