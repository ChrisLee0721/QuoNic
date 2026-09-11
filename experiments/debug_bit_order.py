"""Test bit ordering: put qubit 102 in |1>, qubit 111 in |0>, measure without any entangling gate."""

import os, sys
sys.path.insert(0, r"C:/Users/26427/AppData/Local/Programs/Python/Python312")

import pyqpanda3 as pq
from pyqpanda3.qcloud.qcloud import QCloudService
from pyqpanda3.transpilation import Transpiler

api_key = os.environ.get("ORIGINGQ_API_KEY", "")
service = QCloudService(api_key)
backend = service.backend("WK_C180")
chip_backend = backend.chip_info().get_chip_backend()
transpiler = Transpiler()

# Test 1: q0=|1>, q1=|0> → expect |10> if bit order is q0 q1, |01> if q1 q0
print("=== Test 1: X(q0) only → expect |10> or |01> ===")
prog = pq.core.QProg()
prog << pq.core.X(pq.core.Qubit(0))
prog << pq.core.measure(pq.core.Qubit(0), 0)
prog << pq.core.measure(pq.core.Qubit(1), 1)
prog_t = transpiler.transpile(prog, chip_backend, init_mapping={0: 102, 1: 111}, optimization_level=0)
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
print(f"If bit order q102 q111: expect |10>")
print(f"If bit order q111 q102: expect |01>")

# Test 2: q0=|0>, q1=|1> → expect |01> or |10>
print("\n=== Test 2: X(q1) only → expect |01> or |10> ===")
prog2 = pq.core.QProg()
prog2 << pq.core.X(pq.core.Qubit(1))
prog2 << pq.core.measure(pq.core.Qubit(0), 0)
prog2 << pq.core.measure(pq.core.Qubit(1), 1)
prog_t2 = transpiler.transpile(prog2, chip_backend, init_mapping={0: 102, 1: 111}, optimization_level=0)
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
print(f"If bit order q102 q111: expect |01>")
print(f"If bit order q111 q102: expect |10>")

# Test 3: X(q0) + X(q1) → expect |11>
print("\n=== Test 3: X(q0)+X(q1) → expect |11> ===")
prog3 = pq.core.QProg()
prog3 << pq.core.X(pq.core.Qubit(0))
prog3 << pq.core.X(pq.core.Qubit(1))
prog3 << pq.core.measure(pq.core.Qubit(0), 0)
prog3 << pq.core.measure(pq.core.Qubit(1), 1)
prog_t3 = transpiler.transpile(prog3, chip_backend, init_mapping={0: 102, 1: 111}, optimization_level=0)
job3 = backend.run(prog_t3, 4000)
raw3 = job3.result()
probs3 = raw3.get_probs()
counts3 = {}
for key, prob in probs3.items():
    bs = str(key)
    if bs.startswith("0x"):
        bs = format(int(bs, 16), "02b")
    elif bs.startswith("0b"):
        bs = bs[2:].zfill(2)
    counts3[bs] = max(1, round(prob * 4000))
print(f"Counts: {counts3}")
print(f"Expect |11> in either bit order")
