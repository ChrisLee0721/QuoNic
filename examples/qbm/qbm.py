"""Quantum Boltzmann Machine / 量子玻尔兹曼机

Quantum version of Boltzmann machine for generative modeling.
量子版玻尔兹曼机用于生成建模。

## Application / 应用场景
- Generative models (生成模型)
- Sampling (采样)
- Machine learning (机器学习)

## Output / 输出
Learned probability distribution.
学习到的概率分布。"""

from quonic.algorithms import qbm

result = qbm(temperature=1.0)
print(result.counts)
