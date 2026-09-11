"""
Test: swap qubit roles on 本源 to match IBM behavior.
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

def run_circuit(k, control_qubit=0, target_qubit=1):
    q0 = pq.core.Qubit(control_qubit)
    q1 = pq.core.Qubit(target_qubit)
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
    probs = result.get_probs()
    
    # Parse
    counts = {}
    for key, prob in probs.items():
        bs = str(key)
        if bs.startswith('0x'):
            bs = format(int(bs, 16), '02b')
        elif bs.startswith('0b'):
            bs = bs[2:].zfill(2)
        counts[bs] = prob
    return counts

# Original: q0=control, q1=target
print("=== Original: q0=control, q1=target ===")
for k in [1, 3]:
    probs = run_circuit(k, 0, 1)
    p_q0_0 = sum(v for bs, v in probs.items() if bs[-1] == '0')
    p_q0_1 = sum(v for bs, v in probs.items() if bs[-1] == '1')
    print(f"k={k}: q0 P(0)={p_q0_0:.4f}, P(1)={p_q0_1:.4f}  probs={dict((k,round(v,3)) for k,v in probs.items())}")

# Swapped: q1=control, q0=target
print("\n=== Swapped: q1=control, q0=target ===")
for k in [1, 3]:
    probs = run_circuit(k, 1, 0)
    p_q1_0 = sum(v for bs, v in probs.items() if bs[-1] == '0')
    p_q1_1 = sum(v for bs, v in probs.items() if bs[-1] == '1')
    print(f"k={k}: q1 P(0)={p_q1_0:.4f}, P(1)={p_q1_1:.4f}  probs={dict((k,round(v,3)) for k,v in probs.items())}")

print("\n=== IBM reference ===")
print("k=1: P(0)=0.9938 (qubit should be |0>)")
print("k=3: P(0)=0.9898 (qubit should be |0>)")
