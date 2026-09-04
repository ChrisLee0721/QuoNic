"""Shortest Vector Problem / 最短向量问题

Quantum approach to lattice-based cryptography.
格密码的量子方法。

## Application / 应用场景
- Post-quantum cryptography (后量子密码学)
- Lattice-based crypto (格密码)
- Security analysis (安全分析)

## Output / 输出
Approximate shortest vector.
近似最短向量。"""

from quonic.algorithms import lattice_svp

result = lattice_svp()
print(result.counts)
