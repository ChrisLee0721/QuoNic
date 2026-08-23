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

from quonic import calibrate, creg, cwhile, qgate, reset, zne
from quonic.backends import get_backend
from quonic.gates import H, Ry
from quonic.stack import current_circuit


def _take():
    c = current_circuit()
    reset()
    return c


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
    ("groverize 单比特", groverize_single(), "00"),
    ("groverize 多比特", groverize_multi(), "1010"),
]


def _success(counts, target, shots):
    return sum(counts.get(bs, 0) for bs in (target,)) / shots


def main() -> None:
    device = sys.argv[1] if len(sys.argv) > 1 else "qx"
    if len(sys.argv) > 2:
        cases = [CASES[int(i)] for i in sys.argv[2].split(",")]
    else:
        cases = CASES
    shots = 1024
    be = get_backend("qi", device=device)

    with open("qi_mitigation_results.txt", "a", encoding="utf-8") as log:
        log.write(f"=== {device} shots={shots} ===\n")
        log.flush()
    print(f"目标: Quantum Inspire {device}  shots={shots}  共 {len(cases)} 个电路\n", flush=True)

    for name, circuit, target in cases:
        n = circuit.num_qubits
        print(f"\n[{name}]  n={n}  门数={circuit.gate_count()}  理想={target} 约 100%", flush=True)

        # raw
        raw = be.run(circuit, shots=shots)
        raw_p = _success(raw.counts or {}, target, shots)
        print(f"  raw           = {raw.counts}", flush=True)
        print(f"  raw 成功率     = {raw_p:.3f}", flush=True)

        # per-qubit readout calibration + apply
        cal = calibrate(n, backend="qi", device=device, shots=shots)
        corrected = cal.apply(raw.counts or {}, shots)
        corr_p = _success(corrected, target, shots)
        print(f"  读出校准(逐比特)后 = {corrected}", flush=True)
        print(f"  逐比特读出成功率   = {corr_p:.3f}", flush=True)

        # correlated (full 2^n matrix) readout calibration + apply
        cal_corr = calibrate(n, backend="qi", device=device, shots=shots, correlated=True)
        corrected_corr = cal_corr.apply(raw.counts or {}, shots)
        corr_corr_p = _success(corrected_corr, target, shots)
        print(f"  读出校准(关联)后   = {corrected_corr}", flush=True)
        print(f"  关联读出成功率     = {corr_corr_p:.3f}", flush=True)

        # ZNE success metric (folding amplifies hardware noise)
        res = zne(circuit, target=target, backend="qi", device=device,
                  factors=(1, 3, 5), shots=shots)
        res_exp = zne(circuit, target=target, backend="qi", device=device,
                      factors=(1, 3, 5), shots=shots, extrapolation="exponential")
        print(f"  ZNE λ 成功率   = {[f'{v:.3f}' for v in res.values]}", flush=True)
        print(f"  ZNE 线性外推   = {res.extrapolated:.3f}", flush=True)
        print(f"  ZNE 指数外推   = {res_exp.extrapolated:.3f}", flush=True)

        # stacked: apply the readout calibration to each folded λ, then exponential fit
        res_stacked = zne(circuit, target=target, backend="qi", device=device,
                          factors=(1, 3, 5), shots=shots, calibration=cal,
                          extrapolation="exponential")
        print(f"  ZNE+校准 λ     = {[f'{v:.3f}' for v in res_stacked.values]}", flush=True)
        print(f"  ZNE+校准 指数外推 = {res_stacked.extrapolated:.3f}", flush=True)

        line = (
            f"[{name}] n={n} gates={circuit.gate_count()} target={target}\n"
            f"  raw={raw_p:.3f}  cal={corr_p:.3f}  cal_corr={corr_corr_p:.3f}"
            f"  zne_lin={res.extrapolated:.3f}  zne_exp={res_exp.extrapolated:.3f}"
            f"  zne+cal_exp={res_stacked.extrapolated:.3f}\n"
        )
        log.write(line + "\n")
        log.flush()


if __name__ == "__main__":
    main()
