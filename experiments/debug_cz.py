"""Test: build CNOT from CZ manually, bypass transpiler decomposition."""

import os, sys
sys.path.insert(0, r"C:/Users/26427/AppData/Local/Programs/Python/Python312")

import pyqpanda3 as pq
from pyqpanda3.qcloud.qcloud import QCloudService
from pyqpanda3.transpilation import Transpiler
import numpy as np

api_key = os.environ.get("ORIGINGQ_API_KEY", "")
service = QCloudService(api_key)
backend = service.backend("WK_C180")
chip_backend = backend.chip_info().get_chip_backend()
transpiler = Transpiler()

CONTROL, TARGET = 102, 111

# Manual CNOT = H(target) → CZ(control, target) → H(target)
# In RPHI terms: H = RPHI(pi, pi/2) approximately
# But let's just use H and CZ directly if available

print("=== Test: manual CNOT via H-CZ-H ===")
prog = pq.core.QProg()
# Put control in |1>
prog << pq.core.X(pq.core.Qubit(0))
# Manual CNOT: H on target, CZ, H on target
prog << pq.core.H(pq.core.Qubit(1))
prog << pq.core.CZ(pq.core.Qubit(0), pq.core.Qubit(1))
prog << pq.core.H(pq.core.Qubit(1))
# Measure
prog << pq.core.measure(pq.core.Qubit(0), 0)
prog << pq.core.measure(pq.core.Qubit(1), 1)

prog_t = transpiler.transpile(prog, chip_backend, init_mapping={0: CONTROL, 1: TARGET}, optimization_level=0)
print(f"Transpiled:\n{prog_t}")

job = backend.run(prog_t, 4000)
raw = job.result()
probs = raw.get_probs()
counts = {}
for key, prob in probs.items():
    bs = str(key)
    if bs.startswith("0x"):
        bs = format(int(bs, 16), "02b")
    elif bs.startswith("0b"):
        bs = bs[2:].zfill(2)
    counts[bs] = max(1, round(prob * 4000))
print(f"Counts: {counts}")
print(f"Bit order: q111 q102")
print(f"Expect |11> (both 1) after CNOT flips target")

# Also test: just CZ without H (should NOT flip target)
print("\n=== Test: CZ only (no H) ===")
prog2 = pq.core.QProg()
prog2 << pq.core.X(pq.core.Qubit(0))
prog2 << pq.core.CZ(pq.core.Qubit(0), pq.core.Qubit(1))
prog2 << pq.core.measure(pq.core.Qubit(0), 0)
prog2 << pq.core.measure(pq.core.Qubit(1), 1)

prog_t2 = transpiler.transpile(prog2, chip_backend, init_mapping={0: CONTROL, 1: TARGET}, optimization_level=0)
job2 = backend.run(prog_t2, 4000)
raw2 = job2.result()
probs2 = raw2.get_probs()
counts2 = {}
for key, prob in probs2.items():
    bs = str(key)
    if bs.startswith("0x"):
        bs = format(int(bs, 16), "02b")
    elif bs.startswith("0b"):
        bs = bs[2:].zfill(2)
    counts2[bs] = max(1, round(prob * 4000))
print(f"Counts: {counts2}")
print(f"CZ only: |1> control → phase flip, no target flip. Expect |10>")
