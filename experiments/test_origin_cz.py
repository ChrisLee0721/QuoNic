"""
Test CZ+H circuit on 本源 WK_C180 to verify no transpiler bug.
k=1: ideal P(0) = high (odd → |1⟩)
k=2: ideal P(0) = 0.5 (even → equal superposition)
k=3: ideal P(0) = high (odd → |1⟩)
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

for k in [1, 2, 3]:
    q0 = pq.core.Qubit(0)
    q1 = pq.core.Qubit(1)
    c0 = pq.core.CBit(0)

    prog = pq.core.QProg()
    prog << pq.core.H(q0)
    for _ in range(k):
        prog << pq.core.CZ(q0, q1)
        prog << pq.core.H(q0)
        prog << pq.core.H(q1)
    prog << pq.core.measure(q0, c0)

    prog_t = transpiler.transpile(prog, chip_backend)
    job = backend.run(prog_t, 100)
    result = job.result()
    probs = result.get_probs()
    
    p1 = probs.get('1', 0)
    ideal = "P(1)≈1.0" if k % 2 == 1 else "P(0)≈P(1)≈0.5"
    print(f"k={k}: P(0)={probs.get('0',0):.4f}, P(1)={p1:.4f}  ideal: {ideal}")
