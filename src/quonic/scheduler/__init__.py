"""Scheduler: route a circuit to the optimal backend via table lookup on circuit features.

Lookup chain (highest priority first):

    local cache ──► static parameter table ──► built-in rule fallback

The local cache is the user's own experience table (remembered after one run);
the static table is a general benchmark for cold start (imported from a file or
website); the rule fallback guarantees any circuit has a backend to run on.

Example:
    from quonic.scheduler import schedule, circuit_features, LocalCacheRegistry

    cache = LocalCacheRegistry(".quonic_cache.json")
    backend = schedule(circuit, cache=cache)   # cache -> rules
"""

from __future__ import annotations

from ..ir import Circuit
from .capabilities import (
    METHOD_CAPABILITIES,
    decision_class,
    eligible_methods,
)
from .features import circuit_features
from .registry import (
    BackendRegistry,
    BuiltinRegistry,
    FileRegistry,
    LocalCacheRegistry,
    MemoryRegistry,
    Recommendation,
    load_gpu_decision,
    load_measured_decision,
    load_noise_cost,
    load_performance,
    recommend_backend_autodiff,
    recommend_backend_gpu,
    recommend_method,
)

_BUILTIN = BuiltinRegistry()


def _parse(rec: str | None) -> tuple[str | None, str | None]:
    """Parse the lookup result into (backend, method); "qiskit" -> (qiskit, None),
    "qiskit:stabilizer" -> (qiskit, stabilizer), None -> (None, None)."""
    if rec is None:
        return None, None
    if ":" in rec:
        backend, method = rec.split(":", 1)
        return backend, method
    return rec, None


def schedule(
    circuit: Circuit,
    cache: BackendRegistry | None = None,
    table: BackendRegistry | None = None,
    noise: bool = False,
) -> Recommendation:
    """Return the scheduling decision (backend name + simulation method).

    Args:
        cache: local cache (LocalCacheRegistry), highest priority.
            If the cache has timing data, the backend with the best average
            time is chosen (learning scheduler).
        table: static parameter table (FileRegistry / MemoryRegistry), next.
        noise: whether to enable depolarizing noise (when enabled, method is
            always density_matrix).
        When neither finds a match, fall back to the built-in rules.

    Returns: Recommendation(backend=..., method=...). The lookup result may be
    "backend" or "backend:method"; when the method is not specified in the table,
    it is completed by "circuit analysis" (recommend_method), with no pre-stored
    data required.
    """
    feats = circuit_features(circuit)
    backend: str | None = None
    method: str | None = None

    # 1. Check cache (highest priority)
    if cache is not None:
        # Try learning-based recommendation first (timing-aware)
        if hasattr(cache, "get_best_backend"):
            best = cache.get_best_backend(feats)
            if best is not None:
                backend = best
        # Fall back to key-based lookup
        if backend is None:
            backend, method = _parse(cache.get_recommendation(feats))

    # 2. Check static table
    if backend is None and table is not None:
        backend, method = _parse(table.get_recommendation(feats))

    # 3. Fall back to built-in rules
    if backend is None:
        backend = _BUILTIN.get_recommendation(feats)
    if method is None or noise:
        method = recommend_method(feats, noise=noise)
    return Recommendation(backend=backend, method=method)


__all__ = [
    "METHOD_CAPABILITIES",
    "BackendRegistry",
    "BuiltinRegistry",
    "FileRegistry",
    "LocalCacheRegistry",
    "MemoryRegistry",
    "Recommendation",
    "circuit_features",
    "decision_class",
    "eligible_methods",
    "load_gpu_decision",
    "load_measured_decision",
    "load_noise_cost",
    "load_performance",
    "recommend_backend_autodiff",
    "recommend_backend_gpu",
    "recommend_method",
    "schedule",
]
