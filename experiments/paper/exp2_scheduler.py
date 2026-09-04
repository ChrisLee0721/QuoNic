"""Experiment 2: Smart Scheduler Evaluation.

Tests the scheduler's ability to pick optimal backends for different
circuit types. Compares against ALL available backend/method combinations.

Outputs: experiments/paper/results/exp2_scheduler.json
"""

from __future__ import annotations

import json
import time
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

SHOTS = 1024

# All backend/method pairs to benchmark
ALL_BACKEND_METHODS = {
    "native": ["statevector", "stabilizer", "matrix_product_state", "density_matrix"],
    "qiskit": ["statevector", "stabilizer", "matrix_product_state", "density_matrix"],
    "cupy": ["statevector", "density_matrix"],
    "cirq": ["statevector"],
    "qpanda": ["statevector", "density_matrix"],
}


def build_ghz(n: int):
    from quonic import qgate, reset
    from quonic.gates import CX, H
    from quonic.stack import current_circuit

    reset()
    qgate(H, 0)
    for i in range(n - 1):
        qgate(CX, i, i + 1)
    return current_circuit()


def build_qaoa(n: int):
    from quonic import qgate, reset
    from quonic.gates import CX, H, Rz
    from quonic.stack import current_circuit

    reset()
    for i in range(n):
        qgate(H, i)
    for _ in range(2):
        for i in range(0, n - 1, 2):
            qgate(CX, i, i + 1)
            qgate(Rz(0.5), i + 1)
            qgate(CX, i, i + 1)
        for i in range(n):
            qgate(H, i)
            qgate(Rz(0.3), i)
            qgate(H, i)
    return current_circuit()


def build_random_non_clifford(n: int, depth: int, seed: int = 42):
    import random

    from quonic import qgate, reset
    from quonic.gates import CX, H, Rx, Ry, Rz
    from quonic.stack import current_circuit

    rng = random.Random(seed)
    reset()
    for _ in range(depth):
        for q in range(n):
            gate = rng.choice([H, Rx, Ry, Rz])
            if gate in (Rx, Ry, Rz):
                qgate(gate(rng.uniform(0, 6.28)), q)
            else:
                qgate(gate, q)
        for i in range(n - 1):
            qgate(CX, i, i + 1)
    return current_circuit()


def build_noise_circuit(n: int):
    from quonic import qgate, reset
    from quonic.gates import CX, H
    from quonic.stack import current_circuit

    reset()
    qgate(H, 0)
    for i in range(n - 1):
        qgate(CX, i, i + 1)
    return current_circuit()


def run_all_backends(circuit, name: str) -> dict:
    """Run a circuit through the scheduler and all available backends."""
    from quonic.backends import get_backend
    from quonic.scheduler import schedule
    from quonic.scheduler.capabilities import decision_class, eligible_methods
    from quonic.scheduler.features import circuit_features

    features = circuit_features(circuit)
    eligible = eligible_methods(features["gate_types"], noise=False)
    dclass = decision_class(features)

    # Scheduler recommendation
    rec = schedule(circuit)

    # Run recommended
    t0 = time.perf_counter()
    result_rec = get_backend(rec.backend).run(circuit, shots=SHOTS, method=rec.method)
    time_rec = time.perf_counter() - t0

    # Run ALL backend/method combinations
    import threading
    all_timings = {}
    for backend, methods in ALL_BACKEND_METHODS.items():
        for method in methods:
            # Skip density_matrix for n > 14 (OOM)
            if method == "density_matrix" and circuit.num_qubits > 14:
                continue
            # Skip stabilizer for non-Clifford circuits
            if method == "stabilizer" and not features["is_clifford"]:
                continue
            # Skip if method not eligible for this circuit
            if method not in eligible and method != "density_matrix":
                continue
            bm_key = f"{backend}/{method}"
            result_box = {}
            def _run(b=backend, m=method):
                try:
                    r = get_backend(b).run(circuit, shots=SHOTS, method=m)
                    result_box["result"] = r
                except Exception as e:
                    result_box["error"] = str(e)[:80]
            t0 = time.perf_counter()
            thread = threading.Thread(target=_run, daemon=True)
            thread.start()
            thread.join(timeout=60)
            t = time.perf_counter() - t0
            if thread.is_alive():
                all_timings[bm_key] = {"time": 0, "outcomes": 0, "error": "Timeout (60s)"}
            elif "error" in result_box:
                all_timings[bm_key] = {"time": 0, "outcomes": 0, "error": result_box["error"]}
            else:
                all_timings[bm_key] = {"time": round(t, 4), "outcomes": len(result_box["result"].counts)}

    # Find actual fastest
    valid = {k: v for k, v in all_timings.items() if v["time"] > 0 and "error" not in v}
    fastest_key = min(valid, key=lambda k: valid[k]["time"]) if valid else "none"
    fastest_time = valid.get(fastest_key, {}).get("time", 0)

    # Is scheduler's choice the fastest?
    rec_key = f"{rec.backend}/{rec.method}"
    rec_is_fastest = rec_key == fastest_key
    rec_time = all_timings.get(rec_key, {}).get("time", time_rec)

    # Speedup of scheduler choice vs slowest valid
    slowest_time = max(v["time"] for v in valid.values()) if valid else 1

    return {
        "circuit": name,
        "n_qubits": circuit.num_qubits,
        "gate_count": circuit.gate_count(),
        "depth": circuit.depth(),
        "features": {
            "n": features["n"],
            "depth": features["depth"],
            "gate_count": features["gate_count"],
            "is_clifford": features["is_clifford"],
            "treewidth_ub": features["treewidth_ub"],
            "entanglement": features["entanglement"],
        },
        "decision_class": dclass,
        "eligible_methods": sorted(eligible),
        "recommendation": {"backend": rec.backend, "method": rec.method},
        "recommended_time": round(time_rec, 4),
        "all_timings": all_timings,
        "fastest": {"key": fastest_key, "time": fastest_time},
        "scheduler_is_fastest": rec_is_fastest,
        "speedup_vs_slowest": round(slowest_time / rec_time, 2) if rec_time > 0 else 0,
        "overhead_vs_fastest": round(rec_time / fastest_time, 2) if fastest_time > 0 else 0,
    }


def main():
    test_cases = [
        ("GHZ-20 (Clifford)", build_ghz(20)),
        ("QAOA-8 (low treewidth)", build_qaoa(8)),
        ("Random-20 (high entanglement)", build_random_non_clifford(20, 20)),
        ("Noise-ready circuit", build_noise_circuit(10)),
    ]

    all_results = []
    print(f"{'='*90}")
    print("Smart Scheduler Evaluation (All Backends)")
    print(f"{'='*90}")

    for name, circuit in test_cases:
        print(f"\n{'─'*90}")
        print(f"Circuit: {name}")
        print(f"  n={circuit.num_qubits}, gates={circuit.gate_count()}, depth={circuit.depth()}")

        result = run_all_backends(circuit, name)
        all_results.append(result)

        f = result["features"]
        print(f"  Class: {result['decision_class']}, clifford={f['is_clifford']}, "
              f"tw={f['treewidth_ub']}, entanglement={f['entanglement']}")
        print(f"  Scheduler: {result['recommendation']['backend']}/{result['recommendation']['method']} "
              f"({result['recommended_time']:.4f}s)")
        print(f"  Fastest:   {result['fastest']['key']} ({result['fastest']['time']:.4f}s)")
        print(f"  Scheduler == fastest? {result['scheduler_is_fastest']}")
        print(f"  Overhead vs fastest: {result['overhead_vs_fastest']:.2f}x")

        # Print all timings sorted
        print(f"\n  All timings:")
        valid = {k: v for k, v in result["all_timings"].items() if v["time"] > 0 and "error" not in v}
        for k in sorted(valid, key=lambda k: valid[k]["time"]):
            v = valid[k]
            marker = " <-- scheduler" if k == f"{result['recommendation']['backend']}/{result['recommendation']['method']}" else ""
            print(f"    {k:<40} {v['time']:.4f}s{marker}")

    # Summary
    print(f"\n{'='*90}")
    print("SUMMARY")
    print(f"{'='*90}")
    print(f"{'Circuit':<30} {'Scheduler':<25} {'Fastest':<25} {'Match':>6} {'Overhead':>8}")
    print("-" * 95)
    for r in all_results:
        rec = f"{r['recommendation']['backend']}/{r['recommendation']['method']}"
        fast = r["fastest"]["key"]
        match = "YES" if r["scheduler_is_fastest"] else "NO"
        print(f"{r['circuit']:<30} {rec:<25} {fast:<25} {match:>6} {r['overhead_vs_fastest']:>7.2f}x")

    n_fastest = sum(1 for r in all_results if r["scheduler_is_fastest"])
    print(f"\nScheduler picked fastest: {n_fastest}/{len(all_results)}")

    # Save
    output = {
        "experiment": "exp2_scheduler",
        "description": "Smart scheduler evaluation - all backends comparison",
        "shots": SHOTS,
        "backends_tested": list(ALL_BACKEND_METHODS.keys()),
        "results": all_results,
        "summary": {
            "n_circuits": len(all_results),
            "scheduler_picked_fastest": n_fastest,
            "avg_overhead_vs_fastest": round(
                sum(r["overhead_vs_fastest"] for r in all_results) / len(all_results), 2
            ),
        },
    }

    out_path = RESULTS_DIR / "exp2_scheduler.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
