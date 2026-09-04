"""Discrete logarithm / 离散对数

Find x such that a^x = b mod p.
找到 x 使得 a^x = b mod p。

## Application / 应用场景
- Cryptography (密码学)
- RSA breaking (RSA 破解)
- Key exchange (密钥交换)

## Output / 输出
The discrete logarithm x.
离散对数 x。"""

from quonic.algorithms import discrete_log

result = discrete_log(a=2, b=8, p=11)
print(result.counts)
