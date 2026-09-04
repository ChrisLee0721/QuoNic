"""Quantum Graph Neural Network / 量子图神经网络

Quantum GNN for graph-structured data.
量子 GNN 用于图结构数据。

## Application / 应用场景
- Graph classification (图分类)
- Molecular property prediction (分子性质预测)
- Social networks (社交网络)

## Output / 输出
Graph/node embeddings.
图/节点嵌入。"""

from quonic.algorithms import qgnn

result = qgnn()
print(result.counts)
