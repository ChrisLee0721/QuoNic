"""
Diagnose 本源 WK_C180 transpiler qubit mapping issue.
Measure BOTH qubits for k=1,2,3 to see what's happening.
"""

import os
os.environ['ORIGINGQ_API_KEY'] = 'c5e8b82fef2a8ae457191c9bb88c0a03884cb9be4bf96d0f90ad1691b7406652d1a811063fec7dd8750192907010cad769477863464d33346366764364717168'

import pyqpanda3 as pq
from pyqpanda3.qcloud.qcloud import QCloudService
from pyqpanda3.transpilation import Transpiler

service = QCloudService(os.environ['ORIGINGQ_API_KEY'])
backend = service.backend('WK_C180')
chip_backend = backend.chip_info().get_chip_backend()
transpiler = Transpiler()

def parse_probs(probs, n_qubits=2):
    """Convert hex/decimal keys to binary string keys."""
    counts = {}
    for key, prob in probs.items():
        bs = str(key)
        if bs.startswith('0x'):
            bs = format(int(bs, 16), f'0{n_qubits}b')
        elif bs.startswith('0b'):
            bs = bs[2:].zfill(n_qubits)
        counts[bs] = prob
    return counts

print("=== 本源 WK_C180 CZ+H Diagnostic ===\n")

for k in [1, 2, 3, 5]:
    q0 = pq.core.Qubit(0)
    q1 = pq.core.Qubit(1)
    c0 = pq.core.CBit(0)
    c1 = pq.core.CBit(1)

    prog = pq.core.QProg()
    prog << pq.core.H(q0)
    for _ in range(k):
        prog << pq.core.CZ(q0, q1)
        prog << pq.core.H(q0)
        prog << pq.core.H(q1)
    prog << pq.core.measure(q0, c0)
    prog << pq.core.measure(q1, c1)

    prog_t = transpiler.transpile(prog, chip_backend)

    job = backend.run(prog_t, 200)
    result = job.result()
    probs_raw = result.get_probs()
    probs = parse_probs(probs_raw)

    # Marginals: q0 is LSB (rightmost bit), q1 is MSB
    p_q0_0 = sum(v for bs, v in probs.items() if bs[-1] == '0')  # q0=0
    p_q0_1 = sum(v for bs, v in probs.items() if bs[-1] == '1')  # q0=1
    p_q1_0 = sum(v for bs, v in probs.items() if bs[0] == '0')   # q1=0
    p_q1_1 = sum(v for bs, v in probs.items() if bs[0] == '1')   # q1=1

    ideal_q0 = "P(0)≈1" if k % 2 == 1 else "P(0)≈0.5"
    print(f"k={k}: raw_probs={dict((k,round(v,4)) for k,v in probs.items())}")
    print(f"  q0: P(0)={p_q0_0:.4f}, P(1)={p_q0_1:.4f}  ideal: {ideal_q0}")
    print(f"  q1: P(0)={p_q1_0:.4f}, P(1)={p_q1_1:.4f}")
    print()
