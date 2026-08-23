"""Coupling-map (connectivity) model: describes which two-qubit gates can be executed directly between qubits.

Backend-independent and touching no real hardware — it only captures the physical fact of "allowed two-qubit connections",
used by the compiler seam (compiler.py) to validate whether a circuit can fit onto the target topology, and as the
precursor abstraction for routing when later connecting real hardware / domestic engines.
"""

from __future__ import annotations

from collections.abc import Iterable

from ._i18n import tr


class CouplingMap:
    """Undirected coupling map: n qubits + a set of edges allowing direct interaction."""

    def __init__(self, n: int, edges: Iterable[tuple[int, int]] = ()) -> None:
        if n < 0:
            raise ValueError(tr("err.topology_nonneg", n=n))
        self.n: int = n
        self._edges: set = set()
        for u, v in edges:
            self._add_edge(u, v)

    def _add_edge(self, u: int, v: int) -> None:
        if u == v:
            raise ValueError(tr("err.topology_self_loop", u=u, v=v))
        if u < 0 or v < 0 or u >= self.n or v >= self.n:
            raise ValueError(tr("err.topology_out_of_range", u=u, v=v, n=self.n))
        self._edges.add((min(u, v), max(u, v)))

    @classmethod
    def fully_connected(cls, n: int) -> CouplingMap:
        """Fully connected: any two qubits can interact directly (the default for simulators)."""
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        return cls(n, edges)

    @classmethod
    def from_line(cls, n: int) -> CouplingMap:
        """One-dimensional chain: each qubit connects only to its neighbors (no edges when n=0/1)."""
        edges = [(i, i + 1) for i in range(n - 1)]
        return cls(n, edges)

    @classmethod
    def from_grid(cls, rows: int, cols: int) -> CouplingMap:
        """Two-dimensional grid: each lattice point connects to its right and down neighbors (row-major numbering)."""
        n = rows * cols
        edges = []
        for r in range(rows):
            for c in range(cols):
                idx = r * cols + c
                if c + 1 < cols:
                    edges.append((idx, idx + 1))
                if r + 1 < rows:
                    edges.append((idx, idx + cols))
        return cls(n, edges)

    def has_edge(self, u: int, v: int) -> bool:
        """Whether a direct two-qubit gate is allowed between u and v."""
        return (min(u, v), max(u, v)) in self._edges

    def edges(self) -> list[tuple[int, int]]:
        """Sorted list of edges (tuples, smaller endpoint first)."""
        return sorted(self._edges)

    def __len__(self) -> int:
        return self.n

    def __repr__(self) -> str:
        return f"CouplingMap(n={self.n}, edges={self.edges()})"
