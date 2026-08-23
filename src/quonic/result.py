"""Unified result object.

qshow() and all algorithm templates return a Result, converging two kinds of output into a single structure:

- sampling result (kind="counts"): running a circuit / Grover search, containing a counts histogram
- scalar result (kind="value"): VQE energy / QAOA cut size, containing value + metadata

Usage:
    Result.from_counts({"00": 512, "11": 512}, shots=1024)
    Result.from_value(-2.236, params=[0.1, 0.2, ...])
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Result:
    kind: str
    counts: dict[str, int] | None = None
    shots: int = 0
    value: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_counts(cls, counts: dict[str, int], shots: int) -> Result:
        """Construct a Result from a sampling histogram."""
        return cls(
            kind="counts",
            counts={str(k): int(v) for k, v in counts.items()},
            shots=int(shots),
        )

    @classmethod
    def from_value(cls, value: float, **metadata: Any) -> Result:
        """Construct a Result from a scalar result, with extra information placed in metadata."""
        return cls(kind="value", value=float(value), metadata=dict(metadata))
