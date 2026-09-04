"""Matrix Inversion / 矩阵求逆

HHL-based matrix inversion for linear systems.
基于 HHL 的线性系统矩阵求逆。

## Application / 应用场景
- Linear systems (线性系统)
- Machine learning (机器学习)
- Optimization (优化)

## Output / 输出
Solution vector.
解向量。"""

from quonic.algorithms import quantum_matrix_inversion

result = quantum_matrix_inversion()
print(result.counts)
