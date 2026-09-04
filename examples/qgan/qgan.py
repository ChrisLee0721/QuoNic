"""Quantum GAN / 量子 GAN

Quantum generator + classical discriminator.
量子生成器 + 经典判别器。

## Application / 应用场景
- Data generation (数据生成)
- Image synthesis (图像合成)
- Quantum ML (量子机器学习)

## Output / 输出
Generated data distribution.
生成的数据分布。"""

from quonic.algorithms import qgan

result = qgan(n_steps=50)
print(result.counts)
