"""Error Mitigation — PEC, CDR, symmetry verification.

Provides three error mitigation techniques beyond the existing ZNE:

- **PEC** (Probabilistic Error Cancellation) — quasi-probability decomposition
- **CDR** (Clifford Data Regression) — learn noise via near-Clifford circuits
- **Symmetry Verification** — post-select by symmetry constraints

Example::

    from quonic.mitigation import pec, cdr, symmetry_verify
"""

from __future__ import annotations

from .cdr import CDRResult, cdr
from .pec import PECResult, pec
from .symmetry import symmetry_verify, symmetry_verify_and_renormalize

__all__ = [
    "CDRResult",
    "PECResult",
    "cdr",
    "pec",
    "symmetry_verify",
    "symmetry_verify_and_renormalize",
]
