"""Circuit feature extraction: provide a hashable bucketing key for the scheduler.

Features depend only on the gate list (the circuit is not run) and are computed
in O(number of gates). Bucketing is intentionally coarse so that "fine-tuned
circuits" (iteration count +1, depth +/- a few) still land on the same key, thus
directly hitting the local cache and avoiding re-deciding every time.
"""

from __future__ import annotations

from typing import Any

from ..ir import Circuit
from .capabilities import CLIFFORD_GATES


def _gate_types(circuit: Circuit) -> list[str]:
    return sorted(
        {op.name for op in circuit.ops if op.name not in ("measure", "cmeasure")}
    )


def _interaction_graph(circuit: Circuit) -> set[tuple[int, int]]:
    """Qubit pairs connected by two/multi-qubit gates form undirected edges, used to estimate the entanglement structure."""
    edges: set[tuple[int, int]] = set()
    for op in circuit.ops:
        if op.name in ("measure", "cmeasure"):
            continue
        qs = op.qubits
        for i in range(len(qs)):
            for j in range(i + 1, len(qs)):
                edges.add((qs[i], qs[j]))
    return edges


def _treewidth_upper_bound(n: int, edges: set[tuple[int, int]]) -> int:
    """Upper bound on treewidth from min-degree elimination (a proxy for tensor-network complexity)."""
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    active = set(range(n))
    width = 0
    while active:
        v = min(active, key=lambda x: len(adj[x] & active))
        neigh = list(adj[v] & active)
        width = max(width, len(neigh))
        for i in range(len(neigh)):
            for j in range(i + 1, len(neigh)):
                a, b = neigh[i], neigh[j]
                adj[a].add(b)
                adj[b].add(a)
        active.remove(v)
    return width


def _bucket_key(f: dict[str, Any]) -> str:
    n = f["n"]
    n_bucket = "n<8" if n < 8 else ("n<16" if n < 16 else ("n<24" if n < 24 else "n>=24"))
    cliff = "clifford" if f["is_clifford"] else "nonclifford"
    tw = f["treewidth_ub"]
    tw_bucket = "tw0" if tw == 0 else ("tw<4" if tw < 4 else "tw>=4")
    depth_bucket = f"d{f['depth'] // 50}"
    return f"{n_bucket}|{cliff}|{tw_bucket}|{depth_bucket}"


def _entanglement_level(tw: int, n: int) -> str:
    """Entanglement level: proxy for tensor-network vs statevector choice."""
    ratio = tw / max(n, 1)
    if ratio < 0.2:
        return "low"
    if ratio < 0.5:
        return "medium"
    return "high"


def circuit_features(circuit: Circuit) -> dict[str, Any]:
    """Extract circuit features and return a dict whose features['key'] is a hashable bucketing key."""
    gate_types = _gate_types(circuit)
    edges = _interaction_graph(circuit)
    tw = _treewidth_upper_bound(circuit.num_qubits, edges)
    is_clifford = all(g in CLIFFORD_GATES for g in gate_types)
    has_ctrl = any(
        op.name in ("cif", "cmeasure", "cwhile") for op in circuit.ops
    )
    feats: dict[str, Any] = {
        "n": circuit.num_qubits,
        "depth": circuit.depth(),
        "gate_count": circuit.gate_count(),
        "gate_types": gate_types,
        "is_clifford": is_clifford,
        "treewidth_ub": tw,
        "entanglement": _entanglement_level(tw, circuit.num_qubits),
        "has_ctrl": has_ctrl,
        "requires_grad": getattr(circuit, "requires_grad", False),
    }
    feats["key"] = _bucket_key(feats)
    return feats
