"""
Test: run k=1,3,5,7 with optimization_level=0 on 本源 WK_C180
"""
import os
os.environ['ORIGINGQ_API_KEY'] = 'c5e8b82fef2a8ae457191c9bb88c0a03884cb9be4bf96d0f90ad1691b7406652d1a811063fec7dd8750192907010cad769477863464d33346366764364717168'

import pyqpanda3 as pq
from pyqpanda3.qcloud.qcloud import QCloudService
from pyqpanda3.transpilation import Transpiler, TranspilationOptions

service = QCloudService(os.environ['ORIGINGQ_API_KEY'])
backend = service.backend('WK_C180')
chip_backend = backend.chip_info().get_chip_backend()

options = TranspilationOptions()
options.optimization_level = 0
transpiler = Transpiler()

print("=== 本源 WK_C180: opt_level=0 ===")
print(f"{'k':>4} {'P(0)':>8} {'P(1)':>8} {'expected':>12} {'status':>10}")

for k in [1, 2, 3, 5, 7, 10, 20]:
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

    result = transpiler.transpile(prog, chip_backend, options)
    prog_t = result.program
    job = backend.run(prog_t, 200)
    hw_result = job.result()
    probs = hw_result.get_probs()

    p0 = probs.get('0', 0)
    p1 = probs.get('1', 0)
    expected = "P(0)≈1" if k % 2 == 1 else "P(0)≈0.5"
    status = "OK" if (k % 2 == 1 and p0 > 0.5) or (k % 2 == 0 and 0.3 < p0 < 0.7) else "WRONG"
    print(f"{k:>4} {p0:>8.4f} {p1:>8.4f} {expected:>12} {status:>10}")
