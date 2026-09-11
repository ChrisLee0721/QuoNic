"""Find hidden bitstring / 找到隐藏比特串

Find secret s in f(x) = s·x mod 2. One query suffices.
在 f(x) = s·x mod 2 中找到秘密 s。一次查询即可。

## Application / 应用场景
- Oracle problems (预言机问题)
- Cryptography (密码学)
- Learning parity (学习奇偶性)

## Output / 输出
All shots give the hidden string s.
所有测量结果给出隐藏串 s。"""

from quonic.algorithms import bernstein_vazirani
from quonic.ir import GateOperation

# Hidden string s = "1010" (decimal 10)
S = 10
N = 4

def bv_oracle(circuit, n):
    """Phase oracle for f(x) = s·x mod 2."""
    for i in range(n):
        if (S >> i) & 1:
            circuit.add(GateOperation("cz", (i, n)))

result = bernstein_vazirani(N, bv_oracle, shots=1024)
print(f"Secret: {result.metadata.get('secret', 'unknown')}")
print(f"Counts: {result.metadata.get('counts', {})}")
