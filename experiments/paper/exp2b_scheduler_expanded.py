"""Experiment 2b: Expanded Scheduler Evaluation.

Extends exp2 with12 diverse circuits to strengthen the1.32× overhead claim.

Outputs: experiments/paper/results/exp2b_scheduler_expanded.json
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

SHOTS = 1024

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


def build_linear_cluster(n: int):
    from quonic import qgate, reset
    from quonic.gates import CZ, H
    from quonic.stack import current_circuit
    reset()
    for i in range(n):
        qgate(H, i)
    for i in range(n - 1):
        qgate(CZ, i, i + 1)
    return current_circuit()


def build_qft(n: int):
    from quonic import qgate, reset
    from quonic.gates import CX, H, Rz
    from quonic.stack import current_circuit
    reset()
    for i in range(n):
        qgate(H, i)
        for j in range(i + 1, n):
            angle = math.pi / (2 ** (j - i))
            qgate(Rz(angle / 2), j)
            qgate(CX, j, i)
            qgate(Rz(-angle / 2), i)
            qgate(CX, j, i)
    return current_circuit()


def build_grover(n: int):
    from quonic import qgate, reset
    from quonic.gates import CX, H, X
    from quonic.stack import current_circuit
    reset()
    for i in range(n):
        qgate(H, i)
    # Oracle
    for i in range(n):
        qgate(X, i)
    qgate(H, n - 1)
    for i in range(n - 1):
        qgate(CX, i, n - 1)
    qgate(H, n - 1)
    for i in range(n):
        qgate(X, i)
    # Diffusion
    for i in range(n):
        qgate(H, i)
    for i in range(n):
        qgate(X, i)
    qgate(H, n - 1)
    for i in range(n - 1):
        qgate(CX, i, n - 1)
    qgate(H, n - 1)
    for i in range(n):
        qgate(X, i)
    for i in range(n):
        qgate(H, i)
    return current_circuit()


def build_vqe(n: int, depth: int = 3):
    import random

    from quonic import qgate, reset
    from quonic.gates import CX, Ry
    from quonic.stack import current_circuit
    rng = random.Random(123)
    reset()
    for d in range(depth):
        for i in range(n):
            qgate(Ry(rng.random() * 2 * math.pi), i)
        for i in range(n - 1):
            qgate(CX, i, i + 1)
    return current_circuit()


def build_qpe(n_estimate: int, n_target: int = 1):
    from quonic import qgate, reset
    from quonic.gates import CX, H, Rz
    from quonic.stack import current_circuit
    n_estimate + n_target
    reset()
    for i in range(n_estimate):
        qgate(H, i)
    for i in range(n_estimate):
        angle = math.pi / (2 ** i)
        qgate(Rz(angle / 2), n_estimate)
        qgate(CX, i, n_estimate)
        qgate(Rz(-angle / 2), n_estimate)
        qgate(CX, i, n_estimate)
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


def build_random_clifford(n: int, depth: int, seed: int = 42):
    import random

    from quonic import qgate, reset
    from quonic.gates import CX, CZ, H, Z
    from quonic.stack import current_circuit
    rng = random.Random(seed)
    reset()
    clifford_1q = [H, Z]
    clifford_2q = [CX, CZ]
    for _ in range(depth):
        for q in range(n):
            gate = rng.choice(clifford_1q)
            qgate(gate, q)
        for i in range(n - 1):
            gate = rng.choice(clifford_2q)
            qgate(gate, i, i + 1)
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


def run_all_backends(circuit, name: str) -> dict:
    from quonic.backends import get_backend
    from quonic.scheduler import schedule
    from quonic.scheduler.capabilities import decision_class, eligible_methods
    from quonic.scheduler.features import circuit_features

    features = circuit_features(circuit)
    eligible = eligible_methods(features["gate_types"], noise=False)
    dclass = decision_class(features)
    rec = schedule(circuit)

    t0 = time.perf_counter()
    get_backend(rec.backend).run(circuit, shots=SHOTS, method=rec.method)
    time_rec = time.perf_counter() - t0

    import threading
    all_timings = {}
    for backend, methods in ALL_BACKEND_METHODS.items():
        for method in methods:
            if method == "density_matrix" and circuit.num_qubits > 14:
                continue
            if method == "stabilizer" and not features["is_clifford"]:
                continue
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

    valid = {k: v for k, v in all_timings.items() if v["time"] > 0 and "error" not in v}
    fastest_key = min(valid, key=lambda k: valid[k]["time"]) if valid else "none"
    fastest_time = valid.get(fastest_key, {}).get("time", 0)
    rec_key = f"{rec.backend}/{rec.method}"
    rec_is_fastest = rec_key == fastest_key
    rec_time = all_timings.get(rec_key, {}).get("time", time_rec)
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
        ("GHZ-10 (Clifford)", build_ghz(10)),
        ("GHZ-20 (Clifford)", build_ghz(20)),
        ("LinearCluster-10 (Clifford)", build_linear_cluster(10)),
        ("LinearCluster-20 (Clifford)", build_linear_cluster(20)),
        ("QFT-8 (high tw)", build_qft(8)),
        ("QFT-12 (high tw)", build_qft(12)),
        ("Grover-4 (structured)", build_grover(4)),
        ("Grover-6 (structured)", build_grover(6)),
        ("VQE-6 (low tw)", build_vqe(6, depth=3)),
        ("VQE-10 (low tw)", build_vqe(10, depth=3)),
        ("QPE-4 (medium)", build_qpe(4)),
        ("QAOA-8 (low tw)", build_qaoa(8)),
        ("Random-Clifford-10", build_random_clifford(10, 10)),
        ("Random-Clifford-20", build_random_clifford(20, 10)),
        ("Random-NonClifford-10", build_random_non_clifford(10, 10)),
        ("Random-NonClifford-20", build_random_non_clifford(20, 20)),
    ]

    all_results = []
    print(f"{'='*90}")
    print("Expanded Scheduler Evaluation (16 circuits)")
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

        valid = {k: v for k, v in result["all_timings"].items() if v["time"] > 0 and "error" not in v}
        for k in sorted(valid, key=lambda k: valid[k]["time"])[:3]:
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
    overheads = [r["overhead_vs_fastest"] for r in all_results if r["overhead_vs_fastest"] > 0]
    avg_overhead = sum(overheads) / len(overheads) if overheads else 0
    max_overhead = max(overheads) if overheads else 0
    geo_mean = math.exp(sum(math.log(x) for x in overheads) / len(overheads)) if overheads else 0

    print(f"\nScheduler picked fastest: {n_fastest}/{len(all_results)}")
    print(f"Arithmetic mean overhead: {avg_overhead:.2f}x")
    print(f"Geometric mean overhead: {geo_mean:.2f}x")
    print(f"Worst-case overhead: {max_overhead:.2f}x")

    output = {
        "experiment": "exp2b_scheduler_expanded",
        "description": "Expanded scheduler evaluation -16 circuits",
        "shots": SHOTS,
        "backends_tested": list(ALL_BACKEND_METHODS.keys()),
        "results": all_results,
        "summary": {
            "n_circuits": len(all_results),
            "scheduler_picked_fastest": n_fastest,
            "avg_overhead_vs_fastest": round(avg_overhead, 2),
            "geo_mean_overhead": round(geo_mean, 2),
            "worst_case_overhead": round(max_overhead, 2),
        },
    }

    out_path = RESULTS_DIR / "exp2b_scheduler_expanded.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
