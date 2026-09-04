"""Hidden Subgroup Problem (HSP) — minimal demo of the general framework.

Boundary conditions:
- Abel HSP: Simon's algorithm as special case
- Non-Abel HSP: requires more complex representations
- Minimal: demonstrates the HSP framework concept

Example::

    from quonic.algorithms import hsp
    result = hsp()
"""

from __future__ import annotations

from ..result import Result
from .simon import simon


def hsp() -> Result:
    """Minimal HSP demo: Simon's algorithm as Abel HSP."""

    def period_oracle(circuit, n):
        from quonic.ir import GateOperation
        # Oracle with period s = "11"
        circuit.add(GateOperation("cx", (0, n)))
        circuit.add(GateOperation("cx", (1, n + 1)))
        circuit.add(GateOperation("cx", (0, n + 1)))
        circuit.add(GateOperation("cx", (1, n)))

    result = simon(2, period_oracle, shots=200)
    return Result.from_value(
        float(int(result.metadata.get("secret", "0"), 2)),
        secret=result.metadata.get("secret"),
        hsp_type="abel_simon",
    )
