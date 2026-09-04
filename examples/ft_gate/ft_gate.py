"""Fault-tolerant gates / 容错门

Gates implemented with error detection/correction.
带有错误检测/纠正的门实现。

## Application / 应用场景
- Fault-tolerant computing (容错计算)
- Quantum error correction (量子纠错)
- Logical gates (逻辑门)

## Output / 输出
Logically encoded state with error protection.
具有错误保护的逻辑编码态。"""

from quonic.algorithms import ft_gate

result = ft_gate(shots=100)
print(result.counts)
