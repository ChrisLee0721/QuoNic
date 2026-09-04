"""Syndrome Measurement / Syndrome 测量

Extract error syndromes without disturbing encoded state.
提取错误 syndrome 而不扰动态。

## Application / 应用场景
- Error detection (错误检测)
- QEC decoding (QEC 解码)
- Fault tolerance (容错)

## Output / 输出
Syndrome bits indicating error location.
指示错误位置的 syndrome 比特。"""

from quonic.algorithms import syndrome

result = syndrome(n_data=3, shots=100)
print(result.counts)
