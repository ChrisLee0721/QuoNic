"""Quantum Clustering / 量子聚类

Quantum k-means clustering using SWAP test for distance estimation.
Distances between data points and centroids are computed by quantum circuits,
not classically.

量子k-means聚类，使用SWAP测试估计距离。数据点与质心之间的距离由量子电路计算，
而非经典计算。

## Application / 应用场景
- Data analysis (数据分析)
- Customer segmentation (客户细分)
- Anomaly detection (异常检测)

## Output / 输出
Cluster assignments and final centroids.
聚类分配和最终质心。"""

from quonic.algorithms import quantum_clustering

points = [[0.0, 1.0], [1.0, 0.0], [0.5, 0.5], [0.1, 0.9], [0.9, 0.1]]
centroids = [[0.0, 0.0], [1.0, 1.0]]

result = quantum_clustering(points, centroids, max_iter=3)
print(f"Assignments: {result.metadata['assignments']}")
print(f"Final centroids: {result.metadata['centroids']}")
