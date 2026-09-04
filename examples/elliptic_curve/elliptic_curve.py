"""Elliptic curve quantum algorithm / 椭圆曲线量子算法

Quantum approach to elliptic curve discrete log.
椭圆曲线离散对数的量子方法。

## Application / 应用场景
- Post-quantum cryptography (后量子密码学)
- Blockchain security (区块链安全)
- Digital signatures (数字签名)

## Output / 输出
Approximate solution to ECDLP.
ECDLP 的近似解。"""

from quonic.algorithms import elliptic_curve

result = elliptic_curve()
print(result.counts)
