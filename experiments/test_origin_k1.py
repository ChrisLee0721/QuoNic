"""
Minimal diagnostic: k=1 CZ+H on 本源 WK_C180, measure both qubits.
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

q0 = pq.core.Qubit(0)
q1 = pq.core.Qubit(1)
c0 = pq.core.CBit(0)
c1 = pq.core.CBit(1)

prog = pq.core.QProg()
prog << pq.core.H(q0)
prog << pq.core.CZ(q0, q1)
prog << pq.core.H(q0)
prog << pq.core.H(q1)
prog << pq.core.measure(q0, c0)
prog << pq.core.measure(q1, c1)

print("Transpiling k=1...", flush=True)
prog_t = transpiler.transpile(prog, chip_backend)
print("Submitting...", flush=True)
job = backend.run(prog_t, 200)
result = job.result()
probs = result.get_probs()

print(f"Raw probs: {probs}")

# Parse to binary
for key, prob in probs.items():
    bs = str(key)
    if bs.startswith('0x'):
        bs = format(int(bs, 16), '02b')
    elif bs.startswith('0b'):
        bs = bs[2:].zfill(2)
    print(f"  |{bs}>: {prob:.4f}")

# Marginals
def marginal(probs, bit_idx):
    """bit_idx: 0=q0(LSB), 1=q1(MSB)"""
    p0 = 0
    p1 = 0
    for key, prob in probs.items():
        bs = str(key)
        if bs.startswith('0x'):
            bs = format(int(bs, 16), '02b')
        elif bs.startswith('0b'):
            bs = bs[2:].zfill(2)
        if bs[-(bit_idx+1)] == '0':
            p0 += prob
        else:
            p1 += prob
    return p0, p1

p_q0_0, p_q0_1 = marginal(probs, 0)
p_q1_0, p_q1_1 = marginal(probs, 1)
print(f"\nq0: P(0)={p_q0_0:.4f}, P(1)={p_q0_1:.4f}  ideal: P(0)≈1")
print(f"q1: P(0)={p_q1_0:.4f}, P(1)={p_q1_1:.4f}")
