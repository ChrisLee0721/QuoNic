"""Test pyqpanda3 mid-circuit measurement with correct qif syntax."""

from pyqpanda3.core import *

print("=== Testing pyqpanda3 qif with list[int] syntax ===\n")

# Approach 1: qif([int]) - correct syntax
print("--- Approach 1: qif([0]) ---")
try:
    prog = QProg()
    prog << RPHI(41, 3.14159265, 4.71238898)
    prog << measure(41, 1)  # measure q41 → classical bit 1

    then_prog = QProg() << RPHI(44, 3.14159265, 4.71238898)
    if_block = qif([1]).then(then_prog).qendif()  # if classical bit 1 == 1
    prog << if_block
    prog << measure(44, 0)
    print("  ACCEPTED!")
    print("  Circuit built successfully - mid-circuit measurement syntax is valid")
except Exception as e:
    print(f"  FAILED: {type(e).__name__}: {e}")

# Approach 2: qif with else branch
print("\n--- Approach 2: qif with else ---")
try:
    prog = QProg()
    prog << RPHI(41, 3.14159265, 4.71238898)
    prog << measure(41, 1)

    then_prog = QProg() << RPHI(44, 3.14159265, 4.71238898)
    else_prog = QProg() << H(44)
    if_block = qif([1]).then(then_prog).qelse(else_prog).qendif()
    prog << if_block
    prog << measure(44, 0)
    print("  ACCEPTED!")
except Exception as e:
    print(f"  FAILED: {type(e).__name__}: {e}")

# Approach 3: QWhile (while loop!)
print("\n--- Approach 3: QWhile ---")
try:
    prog = QProg()
    prog << RPHI(41, 3.14159265, 4.71238898)
    prog << measure(41, 1)

    body = QProg() << RPHI(44, 3.14159265, 4.71238898)
    # QWhile might exist
    from pyqpanda3.core import QWhile
    print(f"  QWhile exists: {QWhile}")
    help(QWhile)
except Exception as e:
    print(f"  {type(e).__name__}: {e}")

# Approach 4: Check qif signature more carefully
print("\n--- Approach 4: qif builder pattern ---")
try:
    # qif returns QIf, which has .then() method
    # .then() returns QIfThen, which has .qelse() and .qendif()
    qif_obj = qif([1])
    print(f"  qif([1]) type: {type(qif_obj)}")
    print(f"  methods: {[m for m in dir(qif_obj) if not m.startswith('_')]}")
except Exception as e:
    print(f"  {type(e).__name__}: {e}")

# Approach 5: Check QIfThen methods
print("\n--- Approach 5: QIfThen methods ---")
try:
    qif_obj = qif([1])
    then_obj = qif_obj.then(QProg() << H(44))
    print(f"  then() type: {type(then_obj)}")
    print(f"  methods: {[m for m in dir(then_obj) if not m.startswith('_')]}")
except Exception as e:
    print(f"  {type(e).__name__}: {e}")
