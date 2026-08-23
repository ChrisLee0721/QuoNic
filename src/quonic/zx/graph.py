"""ZX-graph data structure: spiders, edges, and basic operations.

A ZX-graph is a directed graph where:
- Nodes are "spiders" (Z-type or X-type) with a phase angle
- Edges connect spiders (Hadamard edges are marked)
- Boundary nodes (inputs/outputs) have no phase

The key rewrite rules are:
1. Spider fusion: two adjacent same-type spiders merge
2. Spider removal: a 0-phase spider with ≤ 2 neighbors is removed
3. Hadamard edge elimination: H-edges between same-type spiders are removed
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class SpiderType(Enum):
    Z = "Z"
    X = "X"
    BOUNDARY = "boundary"


@dataclass
class ZXSpider:
    """A spider in the ZX-graph.

    Args:
        id: unique identifier
        stype: Z-type, X-type, or boundary
        phase: rotation angle (0 for boundary)
        inputs: list of input edge endpoints (for boundary spiders)
        outputs: list of output edge endpoints (for boundary spiders)
    """

    id: int
    stype: SpiderType
    phase: float = 0.0

    def __repr__(self) -> str:
        if self.stype == SpiderType.BOUNDARY:
            return f"Boundary({self.id})"
        return f"{self.stype.value}({self.id}, phase={self.phase:.3g})"


@dataclass
class ZXEdge:
    """An edge in the ZX-graph.

    Args:
        src: source spider id
        dst: destination spider id
        hadamard: if True, this is a Hadamard edge (implicit H gate)
    """

    src: int
    dst: int
    hadamard: bool = False

    def other(self, node: int) -> int:
        return self.dst if node == self.src else self.src

    def __repr__(self) -> str:
        h = "[H]" if self.hadamard else ""
        return f"Edge({self.src}->{self.dst}{h})"


class ZXGraph:
    """ZX-graph representation of a quantum circuit or computation.

    Provides methods for graph manipulation and rewrite rule application.
    """

    def __init__(self):
        self.spiders: dict[int, ZXSpider] = {}
        self.edges: list[ZXEdge] = []
        self._adj: dict[int, list[int]] = {}  # spider_id -> list of edge indices
        self._next_id: int = 0
        self._inputs: list[int] = []  # boundary spider ids for inputs
        self._outputs: list[int] = []  # boundary spider ids for outputs

    def add_spider(self, stype: SpiderType, phase: float = 0.0) -> int:
        """Add a spider and return its id."""
        sid = self._next_id
        self._next_id += 1
        self.spiders[sid] = ZXSpider(sid, stype, phase)
        self._adj[sid] = []
        return sid

    def add_edge(self, src: int, dst: int, hadamard: bool = False) -> int:
        """Add an edge and return its index."""
        idx = len(self.edges)
        self.edges.append(ZXEdge(src, dst, hadamard))
        self._adj[src].append(idx)
        self._adj[dst].append(idx)
        return idx

    def neighbors(self, sid: int) -> list[int]:
        """Get neighbor spider ids."""
        result = []
        for eidx in self._adj.get(sid, []):
            e = self.edges[eidx]
            if e.src == -1:
                continue
            other = e.other(sid)
            if other != -1:
                result.append(other)
        return result

    def edges_of(self, sid: int) -> list[int]:
        """Get edge indices connected to a spider."""
        return list(self._adj.get(sid, []))

    def set_inputs(self, sids: list[int]) -> None:
        self._inputs = list(sids)

    def set_outputs(self, sids: list[int]) -> None:
        self._outputs = list(sids)

    @property
    def inputs(self) -> list[int]:
        return self._inputs

    @property
    def outputs(self) -> list[int]:
        return self._outputs

    def remove_spider(self, sid: int) -> None:
        """Remove a spider and all its edges."""
        # Remove edges
        for eidx in self._adj.get(sid, []):
            e = self.edges[eidx]
            other = e.other(sid)
            if other in self._adj:
                self._adj[other] = [i for i in self._adj[other] if i != eidx]
        # Mark edges as removed
        for eidx in self._adj.get(sid, []):
            self.edges[eidx] = ZXEdge(-1, -1)  # tombstone
        del self.spiders[sid]
        del self._adj[sid]

    def contract_edge(self, eidx: int) -> int | None:
        """Contract an edge: merge two same-type spiders (spider fusion rule).

        If the two endpoints are the same type, merge them into one spider
        with phase = sum of phases. Returns the surviving spider id, or None
        if the types differ.
        """
        e = self.edges[eidx]
        if e.src == -1:
            return None
        s1, s2 = self.spiders.get(e.src), self.spiders.get(e.dst)
        if s1 is None or s2 is None:
            return None
        if s1.stype != s2.stype:
            return None
        if s1.stype == SpiderType.BOUNDARY:
            return None

        # Merge s2 into s1
        s1.phase += s2.phase
        # Reconnect s2's edges to s1
        for neidx in self._adj.get(e.dst, []):
            if neidx == eidx:
                continue
            ne = self.edges[neidx]
            if ne.src == e.dst:
                ne.src = e.src
            if ne.dst == e.dst:
                ne.dst = e.src
            self._adj[e.src].append(neidx)
        # Remove s2
        self.remove_spider(e.dst)
        return e.src

    def remove_id_spider(self, sid: int) -> bool:
        """Remove a 0-phase spider with ≤ 2 neighbors (identity removal).

        Connects the neighbors directly. Returns True if removed.
        """
        s = self.spiders.get(sid)
        if s is None or s.stype == SpiderType.BOUNDARY:
            return False
        if abs(s.phase) > 1e-10:
            return False
        nbs = self.neighbors(sid)
        if len(nbs) > 2:
            return False

        if len(nbs) == 0:
            self.remove_spider(sid)
            return True
        elif len(nbs) == 1:
            # Remove spider, neighbor becomes disconnected
            self.remove_spider(sid)
            return True
        else:
            # 2 neighbors: connect them directly
            n1, n2 = nbs
            # Check if edge already exists (skip tombstoned edges)
            existing = False
            for eidx in self._adj.get(n1, []):
                e = self.edges[eidx]
                if e.src == -1:
                    continue
                if e.other(n1) == n2:
                    existing = True
                    break
            if not existing:
                self.add_edge(n1, n2)
            self.remove_spider(sid)
            return True

    def copy(self) -> ZXGraph:
        """Deep copy of the graph."""
        g = ZXGraph()
        g._next_id = self._next_id
        for sid, s in self.spiders.items():
            g.spiders[sid] = ZXSpider(s.id, s.stype, s.phase)
            g._adj[sid] = list(self._adj[sid])
        g.edges = [ZXEdge(e.src, e.dst, e.hadamard) for e in self.edges]
        g._inputs = list(self._inputs)
        g._outputs = list(self._outputs)
        return g

    def __repr__(self) -> str:
        return f"ZXGraph(spiders={len(self.spiders)}, edges={len(self.edges)})"
