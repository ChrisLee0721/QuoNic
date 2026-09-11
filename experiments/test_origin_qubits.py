"""
Test different qubit pairs on 本源 to isolate the transpiler bug.
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

# Test k=1 on different qubit pairs
# First, get chip topology to find connected pairs
chip_info = backend.chip_info()
topology = chip_info.get_chip_backend()
print(f"Topology type: {type(topology)}")
print(f"First few edges: {topology[:5] if hasattr(topology, '__getitem__') else 'N/A'}")

# Try a few different logical qubit pairs
test_pairs = [(0, 1), (2, 3), (4, 5), (0, 2), (1, 3)]

for q_ctrl, q_tgt in test_pairs:
    try:
        q0 = pq.core.Qubit(q_ctrl)
        q1 = pq.core.Qubit(q_tgt)
        c0 = pq.core.CBit(0)
        c1 = pq.core.CBit(1)

        prog = pq.core.QProg()
        prog << pq.core.H(q0)
        prog << pq.core.CZ(q0, q1)
        prog << pq.core.H(q0)
        prog << pq.core.H(q1)
        prog << pq.core.measure(q0, c0)
        prog << pq.core.measure(q1, c1)

        prog_t = transpiler.transpile(prog, chip_backend)
        job = backend.run(prog_t, 100)
        result = job.result()
        probs = result.get_probs()

        # Compute q0 marginal
        p_q0_0 = 0
        p_q0_1 = 0
        for key, prob in probs.items():
            bs = str(key)
            if bs.startswith('0x'):
                bs = format(int(bs, 16), '02b')
            elif bs.startswith('0b'):
                bs = bs[2:].zfill(2)
            if bs[-1] == '0':
                p_q0_0 += prob
            else:
                p_q0_1 += prob

        status = "CORRECT" if p_q0_0 > 0.5 else "INVERTED"
        print(f"qubits({q_ctrl},{q_tgt}): q0 P(0)={p_q0_0:.4f}, P(1)={p_q0_1:.4f}  {status}")
    except Exception as e:
        print(f"qubits({q_ctrl},{q_tgt}): ERROR - {e}")
