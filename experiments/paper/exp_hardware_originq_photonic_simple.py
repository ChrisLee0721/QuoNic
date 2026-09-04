"""OriginQ Photonic Hardware Validation (simplified).

Submits groverize()-compiled RUS circuits to PQPUMESH8 photonic quantum computer.
"""

import math

from quonic import qgate, qshow, reset
from quonic.gates import Ry
from quonic.qif import creg, cwhile

# RUS-Ry(2pi/3): cwhile → groverize → photonic hardware
reset()
flag = creg("flag")
with cwhile(flag, until=0):
    qgate(Ry(2 * math.pi / 3), 0)
    flag.measure(0)

qshow(backend="originq", device="PQPUMESH8", shots=1024, groverize=True)
