"""Hidden Subgroup Problem / 隐藏子群问题

General framework for Simon, Shor, and other HSP algorithms.
Simon、Shor 和其他 HSP 算法的通用框架。

## Application / 应用场景
- Factoring (因式分解)
- Discrete log (离散对数)
- Graph isomorphism (图同构)

## Output / 输出
Subgroup generators.
子群生成元。"""

from quonic.algorithms import hsp

result = hsp()
print(result.counts)
