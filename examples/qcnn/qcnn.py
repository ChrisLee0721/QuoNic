"""Quantum Convolutional Neural Network / 量子卷积神经网络

Quantum CNN for classification tasks.
量子 CNN 用于分类任务。

## Application / 应用场景
- Image classification (图像分类)
- Pattern recognition (模式识别)
- Quantum ML (量子机器学习)

## Output / 输出
Classification accuracy.
分类准确率。"""

from quonic.algorithms import qcnn

result = qcnn(maxiter=50)
print(f"Accuracy: {result.value}")
