"""Test if pyqpanda3 API accepts mid-circuit measurement syntax.
Does NOT submit to hardware - only checks if the syntax is valid."""

from pyqpanda3.core import *

print("=== Testing pyqpanda3 mid-circuit measurement syntax ===\n")

# Approach 1: allocate_cbit + conditional
print("--- Approach 1: allocate_cbit + control ---")
try:
    prog = QProg()
    prog << RPHI(41, 3.14159265, 4.71238898)
    c = prog.allocate_cbit()
    prog << measure(41, c)
    prog << (RPHI(44, 3.14159265, 4.71238898)).control(c)
    prog << measure(44, 0)
    print("  ACCEPTED")
except Exception as e:
    print(f"  FAILED: {type(e).__name__}: {e}")

# Approach 2: c_if syntax
print("\n--- Approach 2: c_if ---")
try:
    prog = QProg()
    prog << RPHI(41, 3.14159265, 4.71238898)
    c = prog.allocate_cbit()
    prog << measure(41, c)
    prog << RPHI(44, 3.14159265, 4.71238898).c_if(c, 1)
    prog << measure(44, 0)
    print("  ACCEPTED")
except Exception as e:
    print(f"  FAILED: {type(e).__name__}: {e}")

# Approach 3: If statement
print("\n--- Approach 3: If statement ---")
try:
    prog = QProg()
    prog << RPHI(41, 3.14159265, 4.71238898)
    c = prog.allocate_cbit()
    prog << measure(41, c)
    prog << If(c == 1, RPHI(44, 3.14159265, 4.71238898))
    prog << measure(44, 0)
    print("  ACCEPTED")
except Exception as e:
    print(f"  FAILED: {type(e).__name__}: {e}")

# Approach 4: ClassicalRegister
print("\n--- Approach 4: ClassicalRegister ---")
try:
    prog = QProg()
    cr = ClassicalRegister(1)
    prog << RPHI(41, 3.14159265, 4.71238898)
    prog << measure(41, cr[0])
    prog << (RPHI(44, 3.14159265, 4.71238898)).control(cr[0])
    prog << measure(44, 0)
    print("  ACCEPTED")
except Exception as e:
    print(f"  FAILED: {type(e).__name__}: {e}")

# Approach 5: Direct measure to int
print("\n--- Approach 5: measure with int, then conditional ---")
try:
    prog = QProg()
    prog << RPHI(41, 3.14159265, 4.71238898)
    prog << measure(41, 1)  # measure to classical bit 1
    # Can we reference classical bit 1 in a conditional?
    prog << RPHI(44, 3.14159265, 4.71238898).c_if(1, 1)
    prog << measure(44, 0)
    print("  ACCEPTED")
except Exception as e:
    print(f"  FAILED: {type(e).__name__}: {e}")

# Approach 6: Check if pyqpanda3 has any conditional/feedback primitives
print("\n--- Checking available conditional primitives ---")
for name in dir():
    obj = eval(name)
    if callable(obj) and any(kw in name.lower() for kw in ['if', 'cond', 'cbit', 'classical', 'feedback']):
        print(f"  Found: {name}")

# Check pyqpanda3.core exports
print("\n--- pyqpanda3.core exports with 'if' or 'cond' ---")
import pyqpanda3.core as pqcore
for name in dir(pqcore):
    if any(kw in name.lower() for kw in ['if', 'cond', 'cbit', 'classical', 'feedback', 'measure']):
        print(f"  {name}")
