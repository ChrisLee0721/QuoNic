"""ZX-calculus optimization: circuit ↔ ZX-graph conversion and simplification.

Implements the core ZX-calculus rewrite rules:
1. Spider fusion: merge adjacent same-type spiders
2. Identity removal: remove 0-phase spiders with ≤ 2 neighbors
3. H-edge elimination: remove Hadamard edges between same-type spiders
4. Supplementarity: if two spiders of opposite type share all neighbors, simplify
5. Circuit extraction: convert simplified ZX-graph back to a circuit

Example::

    from quonic.zx import circuit_to_zx, optimize_zx, extract_circuit

    graph = circuit_to_zx(circuit)
    simplified = optimize_zx(graph)
    optimized = extract_circuit(simplified)
"""

from __future__ import annotations

import numpy as np

from ..ir import Circuit, GateOperation
from .graph import SpiderType, ZXEdge, ZXGraph


def circuit_to_zx(circuit: Circuit) -> ZXGraph:
    """Convert a quantum circuit to a ZX-graph.

    Each qubit becomes a "wire" of boundary spiders. Single-qubit gates become
    Z-type or X-type spiders inserted into the wire. Two-qubit gates become
    connected spiders.

    Args:
        circuit: the input circuit

    Returns:
        ZXGraph representation.
    """
    g = ZXGraph()
    n = circuit.num_qubits

    inputs = []
    outputs = []
    current = []

    for q in range(n):
        inp = g.add_spider(SpiderType.BOUNDARY)
        inputs.append(inp)
        current.append(inp)

    for op in circuit.ops:
        if not isinstance(op, GateOperation):
            continue
        name = op.name.lower()
        qubits = op.qubits

        if name == "measure":
            continue

        if len(qubits) == 1:
            q = qubits[0]
            phase = _gate_phase(name, op.params)
            stype = _gate_type(name)

            if stype is not None and phase is not None:
                s = g.add_spider(stype, phase)
                g.add_edge(current[q], s)
                current[q] = s
            elif name == "h":
                s = g.add_spider(SpiderType.Z, 0.0)
                g.add_edge(current[q], s, hadamard=True)
                current[q] = s

        elif len(qubits) == 2:
            c, t = qubits
            if name == "cx":
                s_ctrl = g.add_spider(SpiderType.Z, 0.0)
                s_tgt = g.add_spider(SpiderType.X, 0.0)
                g.add_edge(current[c], s_ctrl)
                g.add_edge(current[t], s_tgt)
                g.add_edge(s_ctrl, s_tgt)
                current[c] = s_ctrl
                current[t] = s_tgt
            elif name == "cz":
                s1 = g.add_spider(SpiderType.Z, 0.0)
                s2 = g.add_spider(SpiderType.Z, 0.0)
                g.add_edge(current[c], s1)
                g.add_edge(current[t], s2)
                g.add_edge(s1, s2, hadamard=True)
                current[c] = s1
                current[t] = s2
            elif name == "swap":
                current[c], current[t] = current[t], current[c]

    for q in range(n):
        out = g.add_spider(SpiderType.BOUNDARY)
        g.add_edge(current[q], out)
        outputs.append(out)

    g.set_inputs(inputs)
    g.set_outputs(outputs)
    return g


def optimize_zx(graph: ZXGraph, max_rounds: int = 10) -> ZXGraph:
    """Simplify a ZX-graph using rewrite rules.

    Applies spider fusion, identity removal, H-edge elimination,
    and supplementarity until no more simplifications are possible.

    Args:
        graph: input ZX-graph
        max_rounds: maximum number of simplification rounds

    Returns:
        Simplified ZX-graph.
    """
    g = graph.copy()

    for _ in range(max_rounds):
        changed = False
        changed |= _match_patterns(g)
        changed |= _phase_teleportation(g)
        changed |= _fuse_spiders(g)
        changed |= _remove_identities(g)
        changed |= _eliminate_h_edges(g)
        changed |= _supplementarity(g)
        changed |= _phase_copy(g)
        changed |= _bialgebra(g)
        if not changed:
            break

    return g


def extract_circuit(graph: ZXGraph) -> Circuit:
    """Extract a quantum circuit from a simplified ZX-graph.

    Traverses the graph from inputs to outputs, emitting gates for each
    non-boundary spider. Handles both Z-type and X-type spiders, entangling
    edges (regular and Hadamard), and multi-qubit gates.

    Args:
        graph: simplified ZX-graph

    Returns:
        Extracted Circuit.
    """
    n = len(graph.inputs)
    c = Circuit()
    c.allocate(n)

    # Map each spider to its qubit (BFS from inputs)
    spider_qubit = {}
    for q_idx, inp_id in enumerate(graph.inputs):
        spider_qubit[inp_id] = q_idx
        queue = [inp_id]
        visited = {inp_id}
        while queue:
            current = queue.pop(0)
            for nb in graph.neighbors(current):
                if nb not in visited:
                    visited.add(nb)
                    s = graph.spiders.get(nb)
                    if s is not None and (s.stype == SpiderType.BOUNDARY or nb not in spider_qubit):
                            spider_qubit[nb] = q_idx
                            queue.append(nb)

    # Emit gates: first single-qubit gates, then entangling gates
    processed = set()

    # Pass 1: single-qubit gates
    for sid, s in graph.spiders.items():
        if s.stype == SpiderType.BOUNDARY:
            continue
        q = spider_qubit.get(sid)
        if q is None:
            continue
        if sid not in processed and abs(s.phase) > 1e-10:
            processed.add(sid)
            if s.stype == SpiderType.Z:
                c.add(GateOperation("rz", (q,), (s.phase,)))
            elif s.stype == SpiderType.X:
                c.add(GateOperation("rx", (q,), (s.phase,)))

    # Pass 2: entangling gates
    for e in graph.edges:
        if e.src == -1:
            continue
        s1 = graph.spiders.get(e.src)
        s2 = graph.spiders.get(e.dst)
        if s1 is None or s2 is None:
            continue
        if s1.stype == SpiderType.BOUNDARY or s2.stype == SpiderType.BOUNDARY:
            continue

        q1 = spider_qubit.get(e.src)
        q2 = spider_qubit.get(e.dst)
        if q1 is None or q2 is None or q1 == q2:
            continue

        # Emit entangling gate
        if e.hadamard:
            c.add(GateOperation("cz", (q1, q2)))
        elif s1.stype == SpiderType.Z and s2.stype == SpiderType.X:
            c.add(GateOperation("cx", (q1, q2)))
        elif s1.stype == SpiderType.X and s2.stype == SpiderType.Z:
            c.add(GateOperation("cx", (q2, q1)))
        else:
            c.add(GateOperation("cz", (q1, q2)))

    return c


def _find_qubit_for_spider(graph: ZXGraph, sid: int, inputs: list) -> int | None:
    """Find which qubit a spider belongs to by tracing back to an input."""
    visited = {sid}
    queue = [sid]
    while queue:
        current = queue.pop(0)
        for nb in graph.neighbors(current):
            if nb in inputs:
                return inputs.index(nb)
            if nb not in visited:
                visited.add(nb)
                queue.append(nb)
    return None


def _find_edge(graph: ZXGraph, s1: int, s2: int) -> ZXEdge | None:
    """Find the edge between two spiders."""
    for e in graph.edges:
        if (e.src == s1 and e.dst == s2) or (e.src == s2 and e.dst == s1):
            return e
    return None


# ---------------------------------------------------------------------------
# Rewrite rules
# ---------------------------------------------------------------------------


def _fuse_spiders(g: ZXGraph) -> bool:
    """Merge adjacent same-type spiders."""
    changed = False
    for _ in range(len(g.edges)):
        found = False
        for eidx, e in enumerate(g.edges):
            if e.src == -1:
                continue
            s1 = g.spiders.get(e.src)
            s2 = g.spiders.get(e.dst)
            if s1 is None or s2 is None:
                continue
            if s1.stype == s2.stype and s1.stype != SpiderType.BOUNDARY:
                g.contract_edge(eidx)
                changed = True
                found = True
                break
        if not found:
            break
    return changed


def _remove_identities(g: ZXGraph) -> bool:
    """Remove 0-phase spiders with ≤ 2 neighbors."""
    changed = True
    any_changed = False
    while changed:
        changed = False
        for sid in list(g.spiders.keys()):
            if sid not in g.spiders:
                continue
            s = g.spiders[sid]
            if s.stype == SpiderType.BOUNDARY:
                continue
            if abs(s.phase) < 1e-10:
                nbs = g.neighbors(sid)
                if len(nbs) <= 2:
                    g.remove_id_spider(sid)
                    changed = True
                    any_changed = True
    return any_changed


def _eliminate_h_edges(g: ZXGraph) -> bool:
    """Eliminate Hadamard edges between same-type spiders.

    Rule: If two same-type spiders are connected by an H-edge, and one has
    phase 0, the H-edge can be removed.
    """
    changed = False
    for eidx, e in enumerate(g.edges):
        if e.src == -1 or not e.hadamard:
            continue
        s1 = g.spiders.get(e.src)
        s2 = g.spiders.get(e.dst)
        if s1 is None or s2 is None:
            continue
        if s1.stype != s2.stype:
            continue
        if s1.stype == SpiderType.BOUNDARY:
            continue

        if abs(s1.phase) < 1e-10 or abs(s2.phase) < 1e-10:
            e.hadamard = False
            changed = True

    return changed


def _phase_teleportation(g: ZXGraph) -> bool:
    """Phase teleportation: move phases from spiders toward neighbors.

    If a spider has phase α and exactly one non-boundary neighbor, teleport
    the phase to that neighbor. Only processes each spider once per call
    to prevent bouncing.
    """
    changed = False
    processed = set()

    for sid in list(g.spiders.keys()):
        if sid in processed or sid not in g.spiders:
            continue
        s = g.spiders[sid]
        if s.stype == SpiderType.BOUNDARY:
            continue
        if abs(s.phase) < 1e-10:
            continue

        nbs = g.neighbors(sid)
        non_boundary_nbs = [nb for nb in nbs if nb in g.spiders and g.spiders[nb].stype != SpiderType.BOUNDARY]

        if len(non_boundary_nbs) == 1:
            target = non_boundary_nbs[0]
            target_s = g.spiders.get(target)
            if target_s is None or target_s.stype == SpiderType.BOUNDARY:
                continue

            processed.add(sid)
            processed.add(target)

            if s.stype == target_s.stype:
                target_s.phase += s.phase
                s.phase = 0.0
                changed = True
            elif len(nbs) <= 2:
                target_s.phase -= s.phase
                s.phase = 0.0
                changed = True

    return changed


def _supplementarity(g: ZXGraph) -> bool:
    """Supplementarity rule: if a Z-spider and X-spider share all neighbors
    and have complementary phases, both can be removed.

    Rule: Z(α) and X(β) connected, sharing the same neighbor set N.
    If α + β = 0 (mod 2π), remove both spiders. The neighbors in N are
    already connected through the graph structure.
    """
    changed = False

    for eidx, e in enumerate(g.edges):
        if e.src == -1:
            continue
        s1 = g.spiders.get(e.src)
        s2 = g.spiders.get(e.dst)
        if s1 is None or s2 is None:
            continue
        if s1.stype == SpiderType.BOUNDARY or s2.stype == SpiderType.BOUNDARY:
            continue
        if s1.stype == s2.stype:
            continue

        # Get neighbors excluding each other
        nbs1 = set(g.neighbors(e.src)) - {e.dst}
        nbs2 = set(g.neighbors(e.dst)) - {e.src}

        # Same neighbors and complementary phases
        if nbs1 == nbs2 and abs(s1.phase + s2.phase) % (2 * np.pi) < 1e-10:
            # Connect each pair of shared neighbors directly
            nbs_list = list(nbs1)
            for i in range(len(nbs_list)):
                for j in range(i + 1, len(nbs_list)):
                    # Check if edge already exists
                    existing = False
                    for e2 in g.edges:
                        if e2.src == -1:
                            continue
                        if (e2.src == nbs_list[i] and e2.dst == nbs_list[j]) or \
                           (e2.src == nbs_list[j] and e2.dst == nbs_list[i]):
                            existing = True
                            break
                    if not existing:
                        g.add_edge(nbs_list[i], nbs_list[j])
            g.remove_spider(e.src)
            g.remove_spider(e.dst)
            changed = True
            break

    return changed


def _phase_copy(g: ZXGraph) -> bool:
    """Phase copy rule: a Z-spider with phase 0 connected to multiple same-type
    spiders can be absorbed, distributing its connections.

    If spider S has phase 0 and connects to multiple same-type neighbors,
    S can be removed and the neighbors connected directly.
    """
    changed = False

    for sid in list(g.spiders.keys()):
        if sid not in g.spiders:
            continue
        s = g.spiders[sid]
        if s.stype == SpiderType.BOUNDARY:
            continue
        if abs(s.phase) > 1e-10:
            continue

        nbs = g.neighbors(sid)
        same_type_nbs = [nb for nb in nbs if nb in g.spiders and g.spiders[nb].stype == s.stype]

        # If all neighbors are same type and there are ≥ 2, can distribute
        if len(same_type_nbs) >= 2 and len(same_type_nbs) == len(nbs):
            # Connect all pairs of same-type neighbors
            for i in range(len(same_type_nbs)):
                for j in range(i + 1, len(same_type_nbs)):
                    existing = False
                    for e in g.edges:
                        if e.src == -1:
                            continue
                        if (e.src == same_type_nbs[i] and e.dst == same_type_nbs[j]) or \
                           (e.src == same_type_nbs[j] and e.dst == same_type_nbs[i]):
                            existing = True
                            break
                    if not existing:
                        g.add_edge(same_type_nbs[i], same_type_nbs[j])
            g.remove_spider(sid)
            changed = True
            break

    return changed


def _bialgebra(g: ZXGraph) -> bool:
    """Bialgebra rule: Z-spider with n neighbors connected to X-spider with m
    neighbors (where the connection is the only shared edge) can be replaced
    with direct connections between all Z-neighbors and X-neighbors.

    Simplified: if a Z-spider and X-spider are connected, and each has exactly
    2 neighbors (including each other), they form a basic bialgebra pattern
    that can be simplified.
    """
    changed = False

    for eidx, e in enumerate(g.edges):
        if e.src == -1:
            continue
        s1 = g.spiders.get(e.src)
        s2 = g.spiders.get(e.dst)
        if s1 is None or s2 is None:
            continue
        if s1.stype == SpiderType.BOUNDARY or s2.stype == SpiderType.BOUNDARY:
            continue
        if s1.stype == s2.stype:
            continue

        # Both have exactly 2 neighbors (each other + one other)
        nbs1 = g.neighbors(e.src)
        nbs2 = g.neighbors(e.dst)

        if len(nbs1) == 2 and len(nbs2) == 2:
            other1 = next(nb for nb in nbs1 if nb != e.dst)
            other2 = next(nb for nb in nbs2 if nb != e.src)

            # If both other neighbors are boundaries, connect them
            sp1 = g.spiders.get(other1)
            sp2 = g.spiders.get(other2)
            if sp1 is not None and sp2 is not None and sp1.stype == SpiderType.BOUNDARY and sp2.stype == SpiderType.BOUNDARY:
                # Connect the two boundaries directly
                existing = False
                for e2 in g.edges:
                    if e2.src == -1:
                        continue
                    if (e2.src == other1 and e2.dst == other2) or \
                           (e2.src == other2 and e2.dst == other1):
                        existing = True
                        break
                if not existing:
                    g.add_edge(other1, other2)
                g.remove_spider(e.src)
                g.remove_spider(e.dst)
                changed = True
                break

    return changed


def _match_patterns(g: ZXGraph) -> bool:
    """Match and simplify common ZX-graph patterns.

    Patterns:
    1. HZH = X: Z-spider with phase π connected via H-edge → convert to X
    2. HXH = Z: X-spider with phase π connected via H-edge → convert to Z
    3. Adjacent H-edges cancel: two H-edges from same spider → remove both
    4. Phase propagation: move phases through H-edges
    """
    changed = False

    # Pattern 1 & 2: H-conjugation (any H-edge with non-boundary spider)
    for e in g.edges:
        if e.src == -1 or not e.hadamard:
            continue
        s1 = g.spiders.get(e.src)
        s2 = g.spiders.get(e.dst)
        if s1 is None or s2 is None:
            continue

        # Check each spider connected by H-edge
        for s in [s1, s2]:
            if s.stype == SpiderType.BOUNDARY:
                continue
            # HZH = X: Z-spider with phase π
            if s.stype == SpiderType.Z and abs(s.phase - np.pi) < 1e-10:
                s.stype = SpiderType.X
                s.phase = 0.0
                changed = True
            # HXH = Z: X-spider with phase π
            elif s.stype == SpiderType.X and abs(s.phase - np.pi) < 1e-10:
                s.stype = SpiderType.Z
                s.phase = 0.0
                changed = True

    # Pattern 3: Adjacent H-edges cancel
    for e1 in g.edges:
        if e1.src == -1 or not e1.hadamard:
            continue
        s1 = g.spiders.get(e1.src)
        if s1 is None or s1.stype == SpiderType.BOUNDARY:
            continue

        for e2 in g.edges:
            if e2.src == -1 or not e2.hadamard:
                continue
            if e2 is e1:
                continue
            if e2.src == e1.src or e2.dst == e1.src:
                e1.hadamard = False
                e2.hadamard = False
                changed = True
                break

    # Pattern 4: Phase propagation through H-edges
    for e in g.edges:
        if e.src == -1 or not e.hadamard:
            continue
        s1 = g.spiders.get(e.src)
        s2 = g.spiders.get(e.dst)
        if s1 is None or s2 is None:
            continue
        if s1.stype == SpiderType.BOUNDARY or s2.stype == SpiderType.BOUNDARY:
            continue

        if abs(s1.phase) > 1e-10 and abs(s2.phase) < 1e-10 and s1.stype == s2.stype:
            s2.phase = s1.phase
            s1.phase = 0.0
            changed = True

    return changed


def _gate_phase(name: str, params: tuple) -> float:
    """Extract the rotation phase from a gate."""
    if name in ("z",):
        return np.pi
    if name in ("s",):
        return np.pi / 2
    if name in ("t",):
        return np.pi / 4
    if name in ("rz", "p") and params:
        return params[0]
    if name in ("x",):
        return np.pi
    if name in ("rx",) and params:
        return params[0]
    return 0.0


def _gate_type(name: str) -> SpiderType:
    """Determine the spider type for a gate."""
    if name in ("z", "rz", "s", "t", "p"):
        return SpiderType.Z
    if name in ("x", "rx"):
        return SpiderType.X
    return None
