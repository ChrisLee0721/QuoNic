"""Generate scheduler benchmarks.json from experiment data.

Reads exp4_native_sim.json and exp2_scheduler.json to produce
a comprehensive benchmarks.json that the scheduler can use for
data-driven decisions (no hardcoded thresholds).

Usage:
    python scripts/update_benchmarks.py
"""

from __future__ import annotations

import json
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "experiments" / "paper" / "results"
BENCHMARKS_PATH = Path(__file__).parent.parent / "src" / "quonic" / "scheduler" / "data" / "benchmarks.json"


def load_exp4() -> list[dict]:
    """Load exp4 native simulator results."""
    with open(RESULTS_DIR / "exp4_native_sim.json") as f:
        data = json.load(f)
    return data["results"]


def extract_performance_table(exp4_data: list[dict]) -> list[dict]:
    """Build performance table with (n, class, gate_count, timings)."""

    # Map circuit type to decision class (matching scheduler's classification)
    # clifford: all gates are Clifford
    # low_tw: non-Clifford but low treewidth (good for MPS)
    # general: non-Clifford, high treewidth (needs statevector)
    class_map = {
        "GHZ": "clifford",
        "Random Clifford": "clifford",
        "Random non-Clifford": "low_tw",  # linear CX chain → treewidth 1
        "QFT": "general",  # dense interactions → high treewidth
    }

    entries = []
    for circuit_data in exp4_data:
        circuit_name = circuit_data["circuit"]
        circuit_class = class_map.get(circuit_name, "general")

        for row in circuit_data["results"]:
            n = row["n"]
            gate_count = row.get("gate_count", n)

            timings = {}
            for engine in ["statevector", "stabilizer", "matrix_product_state", "density_matrix"]:
                if engine in row:
                    entry = row[engine]
                    if entry.get("error") is None and entry.get("time", 0) > 0:
                        timings[engine] = round(entry["time"], 4)

            if timings:
                entries.append({
                    "n": n,
                    "class": circuit_class,
                    "gate_count": gate_count,
                    "depth": row.get("depth", n),
                    "timings": timings,
                })

    return entries


def build_decision_table(perf_entries: list[dict]) -> dict:
    """Build decision table: for each (n, class, gate_bucket, depth_bucket), pick fastest method."""

    def gate_bucket(gc: int) -> str:
        if gc < 50:
            return "small"
        if gc < 200:
            return "medium"
        return "large"

    def depth_bucket(depth: int) -> str:
        if depth < 20:
            return "shallow"
        if depth < 50:
            return "medium"
        return "deep"

    decisions = {}
    for entry in perf_entries:
        n = entry["n"]
        cls = entry["class"]
        gc_bucket = gate_bucket(entry["gate_count"])
        d_bucket = depth_bucket(entry.get("depth", n))
        key = f"{n}|{cls}|{gc_bucket}|{d_bucket}"

        timings = entry["timings"]
        if not timings:
            continue

        fastest = min(timings, key=timings.get)
        decisions[key] = {
            "method": fastest,
            "timings": timings,
        }

    return decisions


def main():
    print("Loading experiment data...")
    exp4_data = load_exp4()

    print("Building performance table...")
    perf_entries = extract_performance_table(exp4_data)
    print(f"  {len(perf_entries)} performance entries")

    print("Building decision table...")
    decisions = build_decision_table(perf_entries)
    print(f"  {len(decisions)} decision entries")

    # Load existing benchmarks.json
    with open(BENCHMARKS_PATH) as f:
        benchmarks = json.load(f)

    # Update with new data
    benchmarks["performance"] = perf_entries
    benchmarks["decision"] = decisions
    benchmarks["meta"]["generated_at"] = "2026-08-30T00:00:00+00:00"
    benchmarks["meta"]["source"] = "exp4_native_sim + exp2_scheduler"

    # Save
    with open(BENCHMARKS_PATH, "w") as f:
        json.dump(benchmarks, f, indent=2)

    print(f"\nUpdated {BENCHMARKS_PATH}")

    # Print summary
    print("\nDecision table summary:")
    for key, val in sorted(decisions.items()):
        print(f"  {key}: {val['method']} ({val['timings']})")


if __name__ == "__main__":
    main()
