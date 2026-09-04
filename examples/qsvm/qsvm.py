"""Quantum Support Vector Machine / 量子支持向量机

SVM with quantum kernel for classification.
使用量子核的 SVM 进行分类。

## Application / 应用场景
- Classification (分类)
- Pattern recognition (模式识别)
- Quantum ML (量子机器学习)

## Output / 输出
Classification accuracy.
分类准确率。"""

from quonic.algorithms import qsvm

result = qsvm()
print(result.counts)
