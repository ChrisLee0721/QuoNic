"""Simon's Algorithm / Simon 算法

Find hidden period of 2-to-1 function. Precursor to Shor.
找到 2-to-1 函数的隐藏周期。Shor 的前身。

## Application / 应用场景
- Cryptography (密码学)
- Period finding (周期查找)
- Quantum advantage (量子优势)

## Output / 输出
Hidden period string.
隐藏周期串。"""

from quonic.algorithms import simon
from quonic.ir import GateOperation

# Hidden period s = "101" (decimal 5)
S = 5
N = 3

def simon_oracle(circuit, n):
    """Oracle for f(x) = f(x XOR s)."""
    for i in range(n):
        circuit.add(GateOperation("cx", (i, i + n)))
    for i in range(n):
        if (S >> i) & 1:
            circuit.add(GateOperation("cx", (0, i + n)))

result = simon(N, simon_oracle, shots=200)
print(f"Secret: {result.metadata.get('secret', 'unknown')}")
print(f"Value: {result.value}")
