"""Test RPhi gate behavior."""

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

def run_and_measure(prog, n_qubits=1):
    prog_t = transpiler.transpile(prog, chip_backend, init_mapping={0: 102}, optimization_level=0)
    job = backend.run(prog_t, 4000)
    raw = job.result()
    probs = raw.get_probs()
    counts = {}
    for key, prob in probs.items():
        bs = str(key)
        if bs.startswith("0x"):
            bs = format(int(bs, 16), f"0{n_qubits}b")
        elif bs.startswith("0b"):
            bs = bs[2:].zfill(n_qubits)
        counts[bs] = max(1, round(prob * 4000))
    return counts

# Test 1: X gate alone
print("=== X(q0) alone ===")
prog = pq.core.QProg()
prog << pq.core.X(pq.core.Qubit(0))
prog << pq.core.measure(pq.core.Qubit(0), 0)
print(run_and_measure(prog))

# Test 2: RPhi(pi, -pi/2) on |0> (what transpiler applied to q102)
print("\n=== RPhi(pi, -pi/2) on |0> ===")
prog = pq.core.QProg()
prog << pq.core.RPhi(0, 3.14159265, -1.57079633)
prog << pq.core.measure(pq.core.Qubit(0), 0)
print(run_and_measure(prog))

# Test 3: X then RPhi(pi, -pi/2) (what transpiler does to q102)
print("\n=== X → RPhi(pi, -pi/2) ===")
prog = pq.core.QProg()
prog << pq.core.X(pq.core.Qubit(0))
prog << pq.core.RPhi(0, 3.14159265, -1.57079633)
prog << pq.core.measure(pq.core.Qubit(0), 0)
print(run_and_measure(prog))

# Test 4: what is RPhi(pi/2, -pi/2)? (applied to q111 before CZ)
print("\n=== RPhi(pi/2, -pi/2) on |0> ===")
prog = pq.core.QProg()
prog << pq.core.RPhi(0, 1.57079633, -1.57079633)
prog << pq.core.measure(pq.core.Qubit(0), 0)
print(run_and_measure(prog))

# Test 5: The full transpiled CNOT sequence on 2 qubits
# q102: RPhi(pi, -pi/2) → CZ → M
# q111: RPhi(pi/2, -pi/2) → CZ → RPhi(pi/2, -8.46) → M
print("\n=== Full transpiled CNOT sequence ===")
prog = pq.core.QProg()
# No X - just the transpiler's decomposition starting from |00>
prog << pq.core.RPhi(0, 3.14159265, -1.57079633)  # q102
prog << pq.core.RPhi(1, 1.57079633, -1.57079633)  # q111
prog << pq.core.CZ(pq.core.Qubit(0), pq.core.Qubit(1))
prog << pq.core.RPhi(1, 1.57079633, -8.45639268)  # q111
prog << pq.core.measure(pq.core.Qubit(0), 0)
prog << pq.core.measure(pq.core.Qubit(1), 1)
c = run_and_measure(prog, 2)
print(f"From |00>: {c}")

# Now with X on q0 first (to simulate control=|1>)
print("\n=== X(q0) → transpiled CNOT sequence ===")
prog = pq.core.QProg()
prog << pq.core.X(pq.core.Qubit(0))
prog << pq.core.RPhi(0, 3.14159265, -1.57079633)
prog << pq.core.RPhi(1, 1.57079633, -1.57079633)
prog << pq.core.CZ(pq.core.Qubit(0), pq.core.Qubit(1))
prog << pq.core.RPhi(1, 1.57079633, -8.45639268)
prog << pq.core.measure(pq.core.Qubit(0), 0)
prog << pq.core.measure(pq.core.Qubit(1), 1)
c = run_and_measure(prog, 2)
print(f"From |10> (control=1): {c}")
print(f"Bit order: q111 q102")
