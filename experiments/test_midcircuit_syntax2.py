"""Test pyqpanda3 mid-circuit measurement with correct API."""

from pyqpanda3.core import *

print("=== Testing pyqpanda3 conditional primitives ===\n")

# Explore CBit
print("--- CBit ---")
try:
    cb = CBit(0)
    print(f"  CBit(0) = {cb}, type = {type(cb)}")
    cb2 = CBit("mybit")
    print(f"  CBit('mybit') = {cb2}, type = {type(cb2)}")
except Exception as e:
    print(f"  FAILED: {type(e).__name__}: {e}")

# Explore QIf
print("\n--- QIf ---")
try:
    print(f"  QIf = {QIf}")
    print(f"  QIfThen = {QIfThen}")
    print(f"  qif = {qif}")
    # Try to see signature
    help(QIf)
except Exception as e:
    print(f"  {type(e).__name__}: {e}")

# Try constructing a circuit with QIf
print("\n--- Approach: QIf with CBit ---")
try:
    prog = QProg()
    cb = CBit(0)
    prog << RPHI(41, 3.14159265, 4.71238898)
    prog << measure(41, cb)

    # Try QIf syntax
    if_block = QIf(cb, QProg() << RPHI(44, 3.14159265, 4.71238898))
    prog << if_block
    prog << measure(44, 0)
    print("  QIf(cb, prog) ACCEPTED")
except Exception as e:
    print(f"  FAILED: {type(e).__name__}: {e}")

print("\n--- Approach: qif function ---")
try:
    prog = QProg()
    cb = CBit(0)
    prog << RPHI(41, 3.14159265, 4.71238898)
    prog << measure(41, cb)

    # Try qif function
    then_prog = QProg() << RPHI(44, 3.14159265, 4.71238898)
    else_prog = QProg()
    if_block = qif(cb, then_prog, else_prog)
    prog << if_block
    prog << measure(44, 0)
    print("  qif(cb, then, else) ACCEPTED")
except Exception as e:
    print(f"  FAILED: {type(e).__name__}: {e}")

print("\n--- Approach: QIfThen ---")
try:
    prog = QProg()
    cb = CBit(0)
    prog << RPHI(41, 3.14159265, 4.71238898)
    prog << measure(41, cb)

    then_block = QIfThen(cb, QProg() << RPHI(44, 3.14159265, 4.71238898))
    prog << then_block
    prog << measure(44, 0)
    print("  QIfThen(cb, prog) ACCEPTED")
except Exception as e:
    print(f"  FAILED: {type(e).__name__}: {e}")

print("\n--- Approach: measure to CBit by index ---")
try:
    prog = QProg()
    prog << RPHI(41, 3.14159265, 4.71238898)
    prog << measure(41, CBit(1))  # measure to classical bit 1

    then_prog = QProg() << RPHI(44, 3.14159265, 4.71238898)
    else_prog = QProg()
    prog << qif(CBit(1), then_prog, else_prog)
    prog << measure(44, CBit(0))
    print("  measure to CBit(1) + qif ACCEPTED")
except Exception as e:
    print(f"  FAILED: {type(e).__name__}: {e}")
