"""Debug: run k=1 and k=2 to check circuit behavior."""

import os, sys, time
sys.path.insert(0, r"C:/Users/26427/AppData/Local/Programs/Python/Python312")

import pyqpanda3 as pq
from pyqpanda3.qcloud.qcloud import QCloudService
from pyqpanda3.transpilation import Transpiler

api_key = os.environ.get("ORIGINGQ_API_KEY", "")
service = QCloudService(api_key)
backend = service.backend("WK_C180")
chip_backend = backend.chip_info().get_chip_backend()
transpiler = Transpiler()

CONTROL, TARGET = 102, 111

for k in [1, 2]:
    print(f"\n=== k={k} ===")

    # Build circuit
    prog = pq.core.QProg()
    prog << pq.core.X(pq.core.Qubit(0))
    for _ in range(k):
        prog << pq.core.CNOT(pq.core.Qubit(0), pq.core.Qubit(1))
    prog << pq.core.measure(pq.core.Qubit(0), 0)
    prog << pq.core.measure(pq.core.Qubit(1), 1)

    # Transpile
    prog_t = transpiler.transpile(
        prog, chip_backend,
        init_mapping={0: CONTROL, 1: TARGET},
        optimization_level=0,
    )

    # Print transpiled program
    print(f"Transpiled program:\n{prog_t}")

    # Run
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
    print(f"Expected: k={k} odd→|11>, even→|10>")
