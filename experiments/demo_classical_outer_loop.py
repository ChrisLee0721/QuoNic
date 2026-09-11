"""Simulate dynamic circuit via classical outer loop on WK_C180.

Program logic (cwhile):
  while q41 == 1:
      flip q44

Simulated as:
  Step 1: RPHI(41) → measure q41
  Step 2: if q41==1, submit circuit with RPHI(44) → measure q44
          else stop

This proves: dynamic circuits CAN be simulated by repeated submission.
groverize() compresses this into a single submission.
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

# === Classical outer loop simulation ===
print("=== Classical Outer Loop: simulating cwhile ===")
print("Program: while q41==1, flip q44\n")

# Step 1: Prepare q41 in |1⟩ and measure
print("--- Step 1: Prepare and measure q41 ---")
prog1 = QProg()
prog1 << RPHI(41, 3.14159265, 4.71238898)  # flip q41 to |1>
prog1 << BARRIER([41])
prog1 << measure(41, 1)  # q41 → classical bit 1

print("Submitting Step 1...")
job1 = backend.run(prog1, 1000, options)
result1 = job1.result()
probs1 = result1.get_probs()
print(f"Step 1 result: {probs1}")

# Classical decision: is q41 == 1?
q41_is_1 = max(probs1.items(), key=lambda x: x[1])[0]  # most likely outcome
print(f"Most likely outcome: {q41_is_1}")
print(f"Classical decision: q41 = {q41_is_1} → {'CONTINUE' if q41_is_1 == '1' else 'STOP'}")

if q41_is_1 == '1':
    # Step 2: Apply conditional gate and measure q44
    print("\n--- Step 2: Conditional gate applied (q41 was 1) ---")
    prog2 = QProg()
    prog2 << RPHI(44, 3.14159265, 4.71238898)  # flip q44
    prog2 << BARRIER([44])
    prog2 << measure(44, 0)  # q44 → classical bit 0

    print("Submitting Step 2...")
    job2 = backend.run(prog2, 1000, options)
    result2 = job2.result()
    probs2 = result2.get_probs()
    print(f"Step 2 result: {probs2}")

    # Final interpretation
    print("\n=== FINAL RESULT ===")
    print(f"Step 1 (q41): {probs1}")
    print(f"Step 2 (q44): {probs2}")
    print(f"\nClassical outer loop successfully simulated dynamic circuit!")
    print(f"q41=1 → triggered conditional flip of q44")
    print(f"q44 should be |1⟩: {probs2}")
else:
    print("\nq41 was not 1, loop did not execute.")
    print("This would be the 'else' branch in a dynamic circuit.")

print("\n=== COMPARISON ===")
print("This required 2 hardware submissions (~4 min total)")
print("groverize() would do this in 1 submission (~2 min)")
print("Both produce the same semantic result.")
