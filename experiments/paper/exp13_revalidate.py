"""Re-validate scheduler decisions against exp13 ground truth.

Loads exp13_local_overnight.json, uses the NEW scheduler (rebuilt benchmarks.json
+ profiles.json) to pick a backend for each circuit, and compares against the
ground truth fastest time from exp13.

No circuit rebuilding needed — the scheduler only needs features.
No backend execution needed — we compare decisions, not re-run.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
EXP13_PATH = RESULTS_DIR / "exp13_local_overnight.json"
OUTPUT_PATH = RESULTS_DIR / "exp13_revalidate.json"


def main():
    with open(EXP13_PATH, encoding="utf-8") as f:
        data = json.load(f)

    results = data["results"]
    valid = [r for r in results if "all_timings" in r]
    print(f"Loaded {len(valid)} circuits from exp13")

    from quonic.scheduler import recommend_method

    revalidate = []
    overheads = []
    old_overheads = []

    for r in valid:
        n = r["n_qubits"]
        gate_count = r.get("gate_count", n)
        depth = r.get("depth", n)
        features = r.get("features", {})
        gate_types = features.get("gate_types", [])

        # Build features dict for scheduler
        # Use decision_class from exp13 directly (exp13 doesn't store gate_types,
        # so decision_class() would misclassify Clifford as low_tw)
        dclass = r.get("decision_class", "general")
        # Map exp13 family names to detect_family() names
        _family_map = {
            "GHZ": "ghz", "LinearCluster": "linear_cluster",
            "QFT": "qft", "Grover": "grover", "QPE": "qpe",
            "VQE": "vqe", "QAOA": "qaoa",
            "RandomClifford": "random_clifford",
            "RandomNonClifford": "random_nonclifford",
        }
        exp13_family = r.get("meta", {}).get("family")
        family = _family_map.get(exp13_family)
        feats = {
            "n": n,
            "depth": depth,
            "gate_count": gate_count,
            "gate_types": gate_types,
            "is_clifford": dclass == "clifford",
            "treewidth_ub": features.get("treewidth_ub", 99),
            "family": family,
        }

        # New scheduler decision (uses benchmarks.json internally)
        rec = recommend_method(feats)
        new_key = f"{rec.backend}/{rec.method}"

        # Old scheduler decision (from exp13)
        old_rec = r.get("recommendation", {})
        old_key = f"{old_rec.get('backend', 'unknown')}/{old_rec.get('method', 'unknown')}"

        # Ground truth
        all_timings = r.get("all_timings", {})
        valid_t = {k: v["time"] if isinstance(v, dict) else v
                   for k, v in all_timings.items()
                   if (v["time"] if isinstance(v, dict) else v) > 0.001}
        if not valid_t:
            continue

        fastest_key = min(valid_t, key=valid_t.get)
        fastest_time = valid_t[fastest_key]

        new_time = valid_t.get(new_key, None)
        old_time = valid_t.get(old_key, None)

        if new_time is not None and new_time > 0.001:
            new_overhead = new_time / fastest_time
            overheads.append(new_overhead)
        else:
            new_overhead = None

        if old_time is not None and old_time > 0.001:
            old_overhead = old_time / fastest_time
            old_overheads.append(old_overhead)
        else:
            old_overhead = None

        revalidate.append({
            "circuit": r.get("circuit", ""),
            "n_qubits": n,
            "old_pick": old_key,
            "new_pick": new_key,
            "fastest": fastest_key,
            "fastest_time": round(fastest_time, 4),
            "old_overhead": round(old_overhead, 2) if old_overhead else None,
            "new_overhead": round(new_overhead, 2) if new_overhead else None,
        })

    # Summary
    def _stats(oh_list, label):
        if not oh_list:
            return
        oh_list.sort()
        print(f"\n{label} ({len(oh_list)} circuits):")
        print(f"  Mean:   {sum(oh_list)/len(oh_list):.2f}x")
        print(f"  Median: {oh_list[len(oh_list)//2]:.2f}x")
        print(f"  P90:    {oh_list[int(len(oh_list)*0.9)]:.2f}x")
        print(f"  P99:    {oh_list[int(len(oh_list)*0.99)]:.2f}x")
        print(f"  Max:    {oh_list[-1]:.2f}x")
        print(f"  >10x:   {sum(1 for o in oh_list if o > 10)} ({sum(1 for o in oh_list if o > 10)/len(oh_list)*100:.1f}%)")
        print(f"  >100x:  {sum(1 for o in oh_list if o > 100)} ({sum(1 for o in oh_list if o > 100)/len(oh_list)*100:.1f}%)")

    print(f"\n{'='*60}")
    _stats(old_overheads, "OLD scheduler overhead")
    _stats(overheads, "NEW scheduler overhead")

    # Show biggest improvements
    improvements = []
    for r in revalidate:
        if r["old_overhead"] and r["new_overhead"]:
            delta = r["old_overhead"] - r["new_overhead"]
            if delta > 1:
                improvements.append((delta, r))
    improvements.sort(key=lambda x: x[0], reverse=True)
    if improvements:
        print(f"\nBiggest improvements (top 10):")
        for delta, r in improvements[:10]:
            print(f"  {r['circuit']}: {r['old_overhead']:.1f}x -> {r['new_overhead']:.1f}x "
                  f"({r['old_pick']} -> {r['new_pick']})")

    # Save
    output = {
        "source": "exp13_revalidate.py",
        "total": len(revalidate),
        "old_stats": _stats_dict(old_overheads),
        "new_stats": _stats_dict(overheads),
        "results": revalidate,
    }
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {OUTPUT_PATH}")


def _stats_dict(oh_list):
    if not oh_list:
        return {}
    oh_list.sort()
    return {
        "mean": round(sum(oh_list)/len(oh_list), 2),
        "median": round(oh_list[len(oh_list)//2], 2),
        "p90": round(oh_list[int(len(oh_list)*0.9)], 2),
        "p99": round(oh_list[int(len(oh_list)*0.99)], 2),
        "max": round(oh_list[-1], 2),
        "gt_10x": sum(1 for o in oh_list if o > 10),
        "gt_100x": sum(1 for o in oh_list if o > 100),
    }


if __name__ == "__main__":
    main()
