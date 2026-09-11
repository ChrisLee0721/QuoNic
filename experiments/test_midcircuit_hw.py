"""Test WK_C180 mid-circuit measurement support.

Circuit A (control): RPHI(41) → measure(41) → measure(44)
  Expected: q41=1, q44=0

Circuit B (mid-circuit): RPHI(41) → measure(41,c) → if c==1: RPHI(44) → measure(44)
  Expected if mid-circuit works: q44=1
  Expected if mid-circuit ignored: q44=0
"""

import os
import sys
from pyqpanda3.core import *
from pyqpanda3.qcloud import QCloudService, QCloudOptions

api_key = os.environ.get("ORIGINGQ_API_KEY", "")
if not api_key:
    print("ERROR: ORIGINGQ_API_KEY not set")
    sys.exit(1)

options = QCloudOptions()
options.set_amend(True)
options.set_mapping(True)
options.set_optimization(True)

service = QCloudService(api_key=api_key)
backend = service.backend("WK_C180")

# --- Circuit A: control (no mid-circuit) ---
print("=== Circuit A: control ===")
prog_a = QProg()
prog_a << RPHI(41, 3.14159265, 4.71238898)
prog_a << BARRIER([41])
prog_a << measure(41, 1)
prog_a << measure(44, 0)

print("Submitting Circuit A...")
job_a = backend.run(prog_a, 1000, options)
result_a = job_a.result()
probs_a = result_a.get_probs()
print("Circuit A probs:", probs_a)

# --- Circuit B: mid-circuit measurement ---
print("\n=== Circuit B: mid-circuit measurement ===")
prog_b = QProg()
prog_b << RPHI(41, 3.14159265, 4.71238898)  # flip q41 to |1>
prog_b << BARRIER([41])
prog_b << measure(41, 1)  # mid-circuit: measure q41 → classical bit 1
prog_b << BARRIER([41, 44])

# if classical bit 1 == 1, apply RPHI to q44
then_block = QProg() << RPHI(44, 3.14159265, 4.71238898)
prog_b << qif([1]).then(then_block).qendif()

prog_b << BARRIER([44])
prog_b << measure(44, 0)

print("Submitting Circuit B...")
job_b = backend.run(prog_b, 1000, options)
result_b = job_b.result()
probs_b = result_b.get_probs()
print("Circuit B probs:", probs_b)

# --- Interpretation ---
print("\n=== RESULT ===")
print(f"Circuit A (control):      {probs_a}")
print(f"Circuit B (mid-circuit):  {probs_b}")

# Check q44 state in Circuit B
# key format: "q41, q44" where bit 1 = q41, bit 0 = q44
if probs_b:
    p_q44_1 = sum(v for k, v in probs_b.items() if k[0] == '1')  # q44=1
    p_q44_0 = sum(v for k, v in probs_b.items() if k[0] == '0')  # q44=0
    print(f"\nCircuit B - P(q44=0) = {p_q44_0:.4f}, P(q44=1) = {p_q44_1:.4f}")
    if p_q44_1 > 0.5:
        print("→ Mid-circuit measurement IS supported (q44 flipped by conditional gate)")
    else:
        print("→ Mid-circuit measurement NOT supported (conditional gate ignored)")
