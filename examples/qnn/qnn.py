"""Quantum Neural Network / 量子神经网络

Variational quantum circuit as neural network.
变分量子电路作为神经网络。

## Application / 应用场景
- Classification (分类)
- Regression (回归)
- Function approximation (函数逼近)

## Output / 输出
Trained model predictions.
训练模型预测。"""

from quonic.algorithms import qnn

result = qnn(n_qubits=2, depth=2)
print(result.counts)
