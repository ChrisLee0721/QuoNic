"""Quantum Inspire hardware / Quantum Inspire 硬件

Quantum Inspire hardware / Quantum Inspire 硬件

## Application / 应用场景
- Quantum computing (量子计算)
- Algorithm demonstration (算法演示)
- Educational (教学)

## Output / 输出
See code comments for output explanation.
参见代码注释了解输出说明。"""

import math
import sys

from quonic import creg, cwhile, qgate, qif, reset
from quonic.backends import get_backend
from quonic.gates import CX, H, I, Ry, X
from quonic.stack import current_circuit


def _take():
    """Grab the current circuit from the stack and clear it."""
    c = current_circuit()
    reset()
    return c


def bell():
    qgate(H, 0)
    qgate(CX, 0, 1)
    return _take()


def ghz3():
    qgate(H, 0)
    qgate(CX, 0, 1)
    qgate(CX, 1, 2)
    return _take()


def qif_bell():
    # coherent superposition control: qif(0).then(X,1).else_(I,1) == CNOT(0,1)
    qgate(H, 0)
    qif(0).then(X, 1).else_(I, 1)
    return _take()


def qif_controlled_ry():
    # a genuinely non-CNOT qif: |0> branch applies Ry(-π/2), |1> branch Ry(π/2)
    qgate(H, 0)
    qif(0).then(Ry(math.pi / 2), 1).else_(Ry(-math.pi / 2), 1)
    return _take()


def groverize_single():
    # Ry(2π/3) then measure q0 succeeds (q0==0) with p=1/4; groverize lifts it to ~1
    flag = creg("flag")
    with cwhile(flag, until=0) as loop:
        qgate(Ry(2 * math.pi / 3), 0)
        flag.measure(0)
    static = loop.groverize()
    reset()
    return static


def groverize_multi():
    # 2-bit register, H+H, until reg == 2 ("10"), p=1/4 -> static, ideal "1010"
    reg = creg("reg", width=2)
    with cwhile(reg, until=2) as loop:
        qgate(H, 0)
        qgate(H, 1)
        reg.measure(0, bit=0)
        reg.measure(1, bit=1)
    static = loop.groverize()
    reset()
    return static


CASES = [
    ("Bell (参考)", bell(), "|00>,|11> 各约 50%"),
    ("GHZ-3 (参考)", ghz3(), "|000>,|111> 各约 50%"),
    ("qif → CNOT", qif_bell(), "|00>,|11> 各约 50%"),
    ("qif → ctrl-Ry", qif_controlled_ry(), "|00>,|01>,|10>,|11> 各约 25%"),
    ("cwhile→groverize (单比特)", groverize_single(), "|00> 约 100%"),
    ("cwhile→groverize (多比特)", groverize_multi(), "|1010> 约 100%"),
]


def main() -> None:
    device = sys.argv[1] if len(sys.argv) > 1 else "qx"
    # optional: run a subset of cases by index, e.g. "0,1" (default: all)
    if len(sys.argv) > 2:
        idxs = [int(x) for x in sys.argv[2].split(",")]
        cases = [CASES[i] for i in idxs]
    else:
        cases = CASES
    shots = 1024
    be = get_backend("qi", device=device)
    with open("qi_hardware_results.txt", "a", encoding="utf-8") as log:
        log.write(f"=== {device} shots={shots} ===\n")
        log.flush()
    print(f"目标: Quantum Inspire {device}  shots={shots}  共 {len(cases)} 个电路\n", flush=True)
    for name, circuit, ideal in cases:
        print(f"提交 [{name}] ...", flush=True)
        result = be.run(circuit, shots=shots)
        line = f"[{name}]  门数={circuit.gate_count()}  理想={ideal}\n  counts = {result.counts}\n"
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()


if __name__ == "__main__":
    main()
