"""Quantum PCA / 量子 PCA

Exponentially faster PCA for density matrices.
密度矩阵的指数加速 PCA。

## Application / 应用场景
- Dimensionality reduction (降维)
- Data analysis (数据分析)
- Feature extraction (特征提取)

## Output / 输出
Principal eigenvalues.
主特征值。"""

from quonic.algorithms import qpca

result = qpca()
print(result.counts)
