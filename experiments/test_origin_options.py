"""
Test TranspilationOptions(opt_level=0) + CPUQVM simulation
"""

import os
os.environ['ORIGINGQ_API_KEY'] = 'c5e8b82fef2a8ae457191c9bb88c0a03884cb9be4bf96d0f90ad1691b7406652d1a811063fec7dd8750192907010cad769477863464d33346366764364717168'

import pyqpanda3 as pq
from pyqpanda3.qcloud.qcloud import QCloudService
from pyqpanda3.transpilation import Transpiler, TranspilationOptions

service = QCloudService(os.environ['ORIGINGQ_API_KEY'])
backend = service.backend('WK_C180')
chip_backend = backend.chip_info().get_chip_backend()

# === Part A: Check TranspilationResult ===
print("=== Part A: TranspilationResult ===")
options = TranspilationOptions()
options.optimization_level = 0
print(f"optimization_level = {options.optimization_level}")

transpiler = Transpiler()

for k in [1, 3, 5, 7]:
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
    print(f"\nk={k} (opt_level=0):")
    print(f"  Type: {type(result)}")
    print(f"  Attrs: {[a for a in dir(result) if not a.startswith('_')]}")
    # Try to get the transpiled program
    if hasattr(result, 'program'):
        print(f"  program: {result.program}")
    elif hasattr(result, 'get_program'):
        print(f"  get_program: {result.get_program()}")
    elif hasattr(result, 'result'):
        print(f"  result: {result.result()}")
