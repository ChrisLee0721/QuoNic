"""Test whether WK_C180 supports mid-circuit measurement via pyqpanda3.

Circuit A (control): RPHI(41) → measure(41) → measure(44)
  Expected: q41=1, q44=0

Circuit B (mid-circuit): RPHI(41) → measure(41, c) → if c==1: RPHI(44) → measure(44)
  Expected if mid-circuit supported: q44=1
  Expected if NOT supported: q44=0 (or error)
"""

import os
import sys
import json

# Use Python 3.12 for pyqpanda3
PY312 = "C:/Users/26427/AppData/Local/Programs/Python/Python312/python.exe"

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

# --- Circuit A: control (no mid-circuit measurement) ---
print("=== Circuit A: control (no mid-circuit) ===")
prog_a = QProg()
prog_a << RPHI(41, 3.14159265, 4.71238898)
prog_a << BARRIER([41])
prog_a << measure(41, 1)  # q41 → classical bit 1
prog_a << measure(44, 0)  # q44 → classical bit 0

job_a = backend.run(prog_a, 1000, options)
result_a = job_a.result()
probs_a = result_a.get_probs()
print("Circuit A probs:", probs_a)

# --- Circuit B: test mid-circuit measurement ---
print("\n=== Circuit B: mid-circuit measurement test ===")
prog_b = QProg()
prog_b << RPHI(41, 3.14159265, 4.71238898)  # flip q41 to |1>
prog_b << BARRIER([41])

# Try mid-circuit: measure q41, then conditionally flip q44
try:
    # Attempt mid-circuit measurement + classical feedback
    # pyqpanda3 syntax may vary - trying common patterns
    c = prog_b.allocate_cbit()  # or similar
    prog_b << measure(41, c)
    prog_b << BARRIER([41, 44])

    # Conditional gate: if c == 1, apply RPHI to q44
    # This is the critical test - does pyqpanda3 support this?
    prog_b << (RPHI(44, 3.14159265, 4.71238898)).control(c)  # controlled by classical bit

    prog_b << BARRIER([44])
    prog_b << measure(44, 0)  # q44 → classical bit 0

    job_b = backend.run(prog_b, 1000, options)
    result_b = job_b.result()
    probs_b = result_b.get_probs()
    print("Circuit B probs:", probs_b)

    # Interpretation
    print("\n=== Interpretation ===")
    print("If q44 ≈ |1⟩: mid-circuit measurement IS supported")
    print("If q44 ≈ |0⟩: mid-circuit measurement NOT supported (conditional gate ignored)")
    print("If error: mid-circuit measurement NOT supported (API rejected)")

except Exception as e:
    print(f"Mid-circuit measurement FAILED: {type(e).__name__}: {e}")
    print("\nThis means WK_C180 does NOT support mid-circuit measurement via pyqpanda3.")
    print("The conditional gate syntax is not available in this API.")

# --- Alternative syntax attempts ---
print("\n=== Trying alternative pyqpanda3 syntax ===")

alt_approaches = [
    ("c_if syntax", lambda p: p << RPHI(44, 3.14159265, 4.71238898).c_if(c, 1)),
    ("if_test syntax", lambda p: p << If(c == 1, RPHI(44, 3.14159265, 4.71238898))),
]

for name, fn in alt_approaches:
    try:
        prog_c = QProg()
        prog_c << RPHI(41, 3.14159265, 4.71238898)
        c2 = prog_c.allocate_cbit()
        prog_c << measure(41, c2)
        fn(prog_c)
        prog_c << measure(44, 0)
        print(f"  {name}: syntax accepted (not submitted)")
    except Exception as e2:
        print(f"  {name}: FAILED - {type(e2).__name__}: {e2}")
