# GCIQA Failure Log

## [2026-08-28] mcz_decomposed 漏掉最后一个控制比特
- 症状: 9 qubit diffuser 不工作，Grover 搜索不放大有效态
- 原因: Toffoli 级联 `CZ(last_ancilla, target)` 没用到 `controls[-1]`
- 解决方案: 改为 `H(target); CCX(controls[-1], last_ancilla, target); H(target)`
- 文件: `src/quonic/circuit.py:139`
- 状态: 已解决

## [2026-08-28] _decode_bitstring 比特序错误
- 症状: 经典 oracle 和量子 oracle 标记不同的有效态
- 原因: 反转全 bitstring 后，组内比特仍按 MSB-first 读取
- 解决方案: `int(bits[::-1], 2)` — 组内再反转一次
- 文件: `src/quonic/gciqa/angle_oracle.py:282`
- 状态: 已解决

## [2026-08-28] encode_distance 比特序错误
- 症状: encode 和 decode 结果不一致
- 原因: `format(val, '03b')` 输出 MSB-first，qiskit 要 LSB-first
- 解决方案: `format(val, f'0{b}b')[::-1]`
- 文件: `src/quonic/gciqa/angle_oracle.py:308`
- 状态: 已解决

## [2026-08-28] pyqpanda3 在系统 Python 中找不到
- 症状: `ModuleNotFoundError: No module named 'pyqpanda3'`
- 原因: pyqpanda3 装在 .venv 里，系统 Python 3.13 没有
- 解决方案: 用 `.venv/Scripts/python.exe` 跑硬件测试
- 状态: 已解决

## [2026-08-28] NAA 51x 是 bug 造成的假象
- 症状: 硬件结果显示 51x 噪声放大
- 原因: 3 个 bug 导致量子/经典 oracle 标记了错误的状态
- 解决方案: 修复 3 个 bug 后 noiseless Grover 精确匹配理论
- 硬件验证: WK_C180 结果 NAA=0x，有效态概率从 1.77% 降到 0.00%
- Job ID: 37BB63754564D76105081875BC8D9044
- 结果文件: experiments/angle_oracle_hardware_fixed_20260828.json
- 状态: 已解决，NAA 不存在
