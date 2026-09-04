"""Experiment 4: Native Simulator Performance Benchmark.

Benchmarks 4 native engines (statevector, stabilizer, MPS, density_matrix)
across multiple circuit types and qubit counts. Measures time vs qubit count
and identifies crossover points.

Outputs: experiments/paper/results/exp4_native_sim.json
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

SHOTS = 1024
ENGINES = ["statevector", "stabilizer", "matrix_product_state", "density_matrix"]


def build_ghz(n: int):
    from quonic import qgate, reset
    from quonic.gates import CX, H
    from quonic.stack import current_circuit

    reset()
    qgate(H, 0)
    for i in range(n - 1):
        qgate(CX, i, i + 1)
    return current_circuit()


def build_random_clifford(n: int, depth: int, seed: int = 42):
    """Random Clifford circuit (h, x, y, z, cx, cz)."""
    import random

    from quonic import qgate, reset
    from quonic.gates import CX, CZ, H, X, Y, Z
    from quonic.stack import current_circuit

    rng = random.Random(seed)
    single = [H, X, Y, Z]
    double = [CX, CZ]

    reset()
    for _ in range(depth):
        for q in range(n):
            if rng.random() < 0.7:
                qgate(rng.choice(single), q)
        pairs = list(zip(range(0, n - 1, 2), range(1, n, 2)))
        rng.shuffle(pairs)
        for ctrl, tgt in pairs[: max(1, len(pairs) // 2)]:
            qgate(rng.choice(double), ctrl, tgt)
    return current_circuit()


def build_random_non_clifford(n: int, depth: int, seed: int = 42):
    """Random non-Clifford circuit (includes Rx, Ry, Rz)."""
    import random

    from quonic import qgate, reset
    from quonic.gates import CX, H, Rx, Ry, Rz
    from quonic.stack import current_circuit

    rng = random.Random(seed)
    single = [H, Rx, Ry, Rz]

    reset()
    for _ in range(depth):
        for q in range(n):
            if rng.random() < 0.7:
                gate = rng.choice(single)
                if gate in (Rx, Ry, Rz):
                    qgate(gate(rng.uniform(0, 6.28)), q)
                else:
                    qgate(gate, q)
        pairs = list(zip(range(0, n - 1, 2), range(1, n, 2)))
        rng.shuffle(pairs)
        for ctrl, tgt in pairs[: max(1, len(pairs) // 2)]:
            qgate(CX, ctrl, tgt)
    return current_circuit()


def build_qft(n: int):
    from quonic import qgate, reset
    from quonic.gates import CP, H
    from quonic.stack import current_circuit

    reset()
    for i in range(n):
        qgate(H, i)
        for j in range(i + 1, n):
            angle = math.pi / (2 ** (j - i))
            qgate(CP(angle), j, i)
    return current_circuit()


def run_benchmark(circuit, engine: str, timeout: float = 60.0) -> dict:
    """Run a circuit on a native engine and return timing."""
    import threading

    from quonic.backends import get_backend

    result_box = {}
    error_box = {}

    def _run():
        try:
            result = get_backend("native").run(circuit, shots=SHOTS, method=engine)
            result_box["result"] = result
        except Exception as e:
            error_box["error"] = e

    t0 = time.perf_counter()
    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    thread.join(timeout=timeout)
    elapsed = time.perf_counter() - t0

    if thread.is_alive():
        return {
            "engine": engine,
            "time": round(elapsed, 4),
            "unique_outcomes": 0,
            "error": f"Timeout ({timeout}s)",
        }

    if "error" in error_box:
        return {
            "engine": engine,
            "time": round(elapsed, 4),
            "unique_outcomes": 0,
            "error": str(error_box["error"])[:100],
        }

    result = result_box["result"]
    return {
        "engine": engine,
        "time": round(elapsed, 4),
        "unique_outcomes": len(result.counts),
        "error": None,
    }


def main():
    # Define test cases: (name, builder, qubit_range)
    test_cases = [
        ("GHZ", build_ghz, [8, 10, 12, 14, 16, 18, 20]),
        ("Random Clifford", lambda n: build_random_clifford(n, n * 2), [10, 14, 18, 22, 26, 30]),
        ("Random non-Clifford", lambda n: build_random_non_clifford(n, n), [8, 10, 12, 14, 16, 18, 20]),
        ("QFT", build_qft, [8, 10, 12, 14, 16]),
    ]

    # Skip density_matrix for large circuits (O(2^2n) memory)
    DM_MAX_QUBITS = 14

    all_results = []

    for circuit_name, builder, qubits in test_cases:
        print(f"\n{'='*60}")
        print(f"Circuit: {circuit_name}")
        print(f"{'='*60}")
        print(f"{'n':>4}  ", end="")
        for eng in ENGINES:
            print(f"{eng:>12}  ", end="")
        print()
        print("-" * 60)

        circuit_results = []
        for n in qubits:
            circuit = builder(n)
            row = {"n": n, "gate_count": circuit.gate_count(), "depth": circuit.depth()}
            print(f"{n:4d}  ", end="")

            for engine in ENGINES:
                if engine == "density_matrix" and n > DM_MAX_QUBITS:
                    r = {"engine": engine, "time": 0, "unique_outcomes": 0,
                         "error": f"Skipped (n>{DM_MAX_QUBITS})"}
                    row[engine] = r
                    print(f"{'SKIP':>12}  ", end="")
                else:
                    r = run_benchmark(circuit, engine)
                    row[engine] = r
                    if r["error"]:
                        print(f"{'ERR':>12}  ", end="")
                    else:
                        print(f"{r['time']:>10.3f}s  ", end="")
            print()
            circuit_results.append(row)

        all_results.append({
            "circuit": circuit_name,
            "qubit_range": qubits,
            "results": circuit_results,
        })

    # Find crossover points
    print(f"\n{'='*60}")
    print("CROSSOVER ANALYSIS")
    print(f"{'='*60}")
    for case in all_results:
        print(f"\n{case['circuit']}:")
        for r in case["results"]:
            times = {eng: r[eng]["time"] for eng in ENGINES if r[eng]["error"] is None}
            if times:
                fastest = min(times, key=times.get)
                print(f"  n={r['n']:3d}: fastest={fastest} ({times[fastest]:.3f}s)")

    # Save results
    output = {
        "experiment": "exp4_native_sim",
        "description": "Native simulator performance benchmark",
        "shots": SHOTS,
        "engines": ENGINES,
        "results": all_results,
    }

    out_path = RESULTS_DIR / "exp4_native_sim.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
