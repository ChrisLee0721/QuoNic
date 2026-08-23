"""Translator registry: every gate / classical-control operation declares its
per-backend translation in one place (one module per operation).

Adding a gate = add a module here. Adding a backend = implement the backend's
method on the affected translators plus a Backend subclass; no per-backend
if/elif dispatch table to maintain.
"""

from __future__ import annotations

from .base import Translator
from .cif import CifTranslator
from .cmeasure import CMeasureTranslator
from .controlled import CCXTranslator, CXTranslator, CZTranslator
from .cswap import CswapTranslator
from .cwhile import CwhileTranslator
from .hadamard import HadamardTranslator
from .identity import IdentityTranslator
from .mcz import MczTranslator
from .measure import MeasureTranslator
from .pauli import XTranslator, YTranslator, ZTranslator
from .rotation import CpTranslator, PTranslator, RxTranslator, RyTranslator, RzTranslator
from .swap import SwapTranslator

TRANSLATORS: dict[str, Translator] = {t.name: t for t in (
    IdentityTranslator(),
    HadamardTranslator(),
    XTranslator(),
    YTranslator(),
    ZTranslator(),
    CXTranslator(),
    CZTranslator(),
    CCXTranslator(),
    SwapTranslator(),
    CswapTranslator(),
    MczTranslator(),
    RxTranslator(),
    RyTranslator(),
    RzTranslator(),
    CpTranslator(),
    PTranslator(),
    MeasureTranslator(),
    CifTranslator(),
    CMeasureTranslator(),
    CwhileTranslator(),
)}

__all__ = ["TRANSLATORS", "Translator"]
