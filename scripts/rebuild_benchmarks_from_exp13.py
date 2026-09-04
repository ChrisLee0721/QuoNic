"""Rebuild benchmarks.json from exp13 local overnight data.

Reads exp13_local_overnight.json (697 circuits with complete all_timings for
native/qiskit/cirq/qpanda) and rebuilds the decision and performance sections
of benchmarks.json. This fixes the missing qpanda data problem.

Usage:
    python scripts/rebuild_benchmarks_from_exp13.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

EXP13_PATH = Path(__file__).parent.parent / "experiments" / "paper" / "results" / "exp13_local_overnight.json"
BENCHMARKS_PATH = Path(__file__).parent.parent / "src" / "quonic" / "scheduler" / "data" / "benchmarks.json"


def _gate_bucket(gate_count: int) -> str:
    if gate_count < 50:
        return "small"
    if gate_count < 200:
        return "medium"
    return "large"


def _depth_bucket(depth: int) -> str:
    if depth < 20:
        return "shallow"
    if depth < 50:
        return "medium"
    return "deep"


def _decision_class(features: dict) -> str:
    """Classify circuit as clifford / low_tw / general."""
    from quonic.scheduler.capabilities import CLIFFORD_GATES

    gate_types = features.get("gate_types", [])
    if not gate_types:
        # exp13 stores gate_types in meta, not features; use decision_class directly
        return features.get("decision_class", "general")
    is_clifford = all(g in CLIFFORD_GATES for g in gate_types)
    if is_clifford:
        return "clifford"
    tw = features.get("treewidth_ub", 99)
    if tw <= 4:
        return "low_tw"
    return "general"


def rebuild():
    with open(EXP13_PATH, encoding="utf-8") as f:
        data = json.load(f)

    results = data["results"]
    print(f"Loaded {len(results)} circuits from exp13")

    # Aggregate timings per decision key
    # key -> {backend/method -> [list of timings]}
    key_timings: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    # Also collect raw performance entries
    performance = []

    for r in results:
        if "all_timings" not in r:
            continue
        feats = r.get("features", {})
        n = r["n_qubits"]
        gate_count = r.get("gate_count", feats.get("gate_count", n))
        depth = r.get("depth", feats.get("depth", n))

        # Use decision_class from exp13 directly (already computed)
        cls = r.get("decision_class", "general")
        gc_bucket = _gate_bucket(gate_count)
        d_bucket = _depth_bucket(depth)
        # Family-aware key (primary) + base key (fallback)
        # Normalize family to lowercase to match detect_family() output
        _family_map = {
            "GHZ": "ghz", "LinearCluster": "linear_cluster",
            "QFT": "qft", "Grover": "grover", "QPE": "qpe",
            "VQE": "vqe", "QAOA": "qaoa",
            "RandomClifford": "random_clifford",
            "RandomNonClifford": "random_nonclifford",
        }
        exp13_family = r.get("meta", {}).get("family")
        family = _family_map.get(exp13_family, exp13_family)
        if family:
            key = f"{n}|{family}|{cls}|{gc_bucket}|{d_bucket}"
        else:
            key = f"{n}|{cls}|{gc_bucket}|{d_bucket}"

        # Collect timings (filter out zero/near-zero = errored or cached)
        base_key = f"{n}|{cls}|{gc_bucket}|{d_bucket}"
        timings_dict = {}
        for bm_key, timing_data in r["all_timings"].items():
            t = timing_data["time"] if isinstance(timing_data, dict) else timing_data
            if t > 0.001:  # filter out zero/near-zero timings (errors, cache hits)
                key_timings[key][bm_key].append(t)
                # Also populate base key for fallback
                if key != base_key:
                    key_timings[base_key][bm_key].append(t)
            timings_dict[bm_key] = t

        performance.append({
            "n": n,
            "class": cls,
            "gate_count": gate_count,
            "depth": depth,
            "timings": timings_dict,
        })

    # Build decision table: for each key, pick the fastest backend/method
    decision = {}
    for key, bm_timings in key_timings.items():
        # Take minimum timing per backend/method (consistent with gen_benchmarks.py)
        min_timings = {}
        for bm_key, times in bm_timings.items():
            if times:  # only include backends with valid timings
                min_timings[bm_key] = min(times)

        if not min_timings:
            continue

        # Pick the fastest
        fastest = min(min_timings, key=min_timings.get)
        decision[key] = {
            "timings": min_timings,
            "method": fastest,
        }

    print(f"Built decision table with {len(decision)} keys")
    print(f"Built performance table with {len(performance)} entries")

    # Load existing benchmarks.json to preserve meta, capabilities, noise, gpu sections
    with open(BENCHMARKS_PATH, encoding="utf-8") as f:
        benchmarks = json.load(f)

    # Update meta
    benchmarks["meta"]["generated_at"] = __import__("datetime").datetime.now().isoformat()
    benchmarks["meta"]["source"] = "rebuild_benchmarks_from_exp13.py"
    benchmarks["meta"]["total_jobs"] = len(results)
    benchmarks["meta"]["successful"] = len(performance)
    backends = set()
    for entry in performance:
        backends.update(entry["timings"].keys())
    benchmarks["meta"]["backends"] = sorted({b.split("/")[0] for b in backends})

    # Replace performance and decision
    benchmarks["performance"] = performance
    benchmarks["decision"] = decision

    # Write back
    with open(BENCHMARKS_PATH, "w", encoding="utf-8") as f:
        json.dump(benchmarks, f, indent=2)

    print(f"Written to {BENCHMARKS_PATH}")

    # Print summary
    fastest_counts: dict[str, int] = defaultdict(int)
    for entry in decision.values():
        backend = entry["method"].split("/")[0]
        fastest_counts[backend] += 1
    print("\nFastest backend distribution:")
    for backend, count in sorted(fastest_counts.items(), key=lambda x: -x[1]):
        print(f"  {backend}: {count} keys ({count/len(decision)*100:.1f}%)")


if __name__ == "__main__":
    rebuild()
