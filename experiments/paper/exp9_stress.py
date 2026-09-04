"""Experiment 9: Stress Test.

Finds practical limits by scaling qubit count across engines.
Tests statevector up to 30 qubits, stabilizer/Clifford up to 50.

Outputs: experiments/paper/results/exp9_stress.json
"""

from __future__ import annotations

import json
import time
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

SHOTS = 1024
TIMEOUT = 120  # seconds per run


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
    """Random Clifford circuit for stabilizer engine testing."""
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
    """Random non-Clifford circuit."""
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


def run_with_timeout(circuit, engine: str, timeout: float = TIMEOUT) -> dict:
    """Run a circuit on a native engine with timeout tracking."""
    import threading

    from quonic.backends import get_backend

    result_box = {}
    error_box = {}

    def _run():
        try:
            result = get_backend("native").run(circuit, shots=SHOTS, method=engine)
            result_box["result"] = result
        except MemoryError:
            error_box["error"] = "oom"
        except Exception as e:
            error_box["error"] = str(e)[:100]

    t0 = time.perf_counter()
    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    thread.join(timeout=timeout)
    elapsed = time.perf_counter() - t0

    if thread.is_alive():
        return {
            "engine": engine,
            "time": round(elapsed, 4),
            "status": "timeout",
            "unique_outcomes": 0,
            "error": f"Timeout ({timeout}s)",
        }

    if "error" in error_box:
        status = "oom" if error_box["error"] == "oom" else "error"
        return {
            "engine": engine,
            "time": round(elapsed, 4),
            "status": status,
            "unique_outcomes": 0,
            "error": error_box["error"],
        }

    result = result_box["result"]
    return {
        "engine": engine,
        "time": round(elapsed, 4),
        "status": "ok",
        "unique_outcomes": len(result.counts),
        "error": None,
    }


def main():
    # Test matrix: (circuit_type, builder, qubit_range, engines)
    test_matrix = [
        (
            "GHZ (Clifford)",
            build_ghz,
            [8, 12, 16, 20, 24, 28, 30, 40, 50],
            ["statevector", "stabilizer"],
        ),
        (
            "Random Clifford",
            lambda n: build_random_clifford(n, n * 2),
            [10, 14, 18, 22, 26, 30],
            ["statevector", "stabilizer"],
        ),
        (
            "Random non-Clifford",
            lambda n: build_random_non_clifford(n, n),
            [8, 10, 12, 14, 16, 18, 20],
            ["statevector", "matrix_product_state"],
        ),
    ]

    all_results = []

    for circuit_name, builder, qubits, engines in test_matrix:
        print(f"\n{'='*70}")
        print(f"Circuit: {circuit_name}")
        print(f"{'='*70}")
        print(f"{'n':>4}  ", end="")
        for eng in engines:
            print(f"{eng:>15}  ", end="")
        print()
        print("-" * (6 + 17 * len(engines)))

        circuit_results = []

        for n in qubits:
            circuit = builder(n)
            row = {
                "n": n,
                "gate_count": circuit.gate_count(),
                "depth": circuit.depth(),
                "engines": {},
            }
            print(f"{n:4d}  ", end="")

            for engine in engines:
                r = run_with_timeout(circuit, engine)
                row["engines"][engine] = r

                if r["status"] == "ok":
                    print(f"{r['time']:>13.3f}s  ", end="")
                elif r["status"] == "timeout":
                    print(f"{'TIMEOUT':>13}  ", end="")
                elif r["status"] == "oom":
                    print(f"{'OOM':>13}  ", end="")
                else:
                    print(f"{'ERR':>13}  ", end="")

            print()
            circuit_results.append(row)

        all_results.append({
            "circuit": circuit_name,
            "qubit_range": qubits,
            "engines": engines,
            "results": circuit_results,
        })

    # Find limits
    print(f"\n{'='*70}")
    print("PRACTICAL LIMITS")
    print(f"{'='*70}")
    for case in all_results:
        print(f"\n{case['circuit']}:")
        for engine in case["engines"]:
            max_ok = 0
            for r in case["results"]:
                if r["engines"][engine]["status"] == "ok":
                    max_ok = r["n"]
            print(f"  {engine}: max n={max_ok}")

    # Save
    output = {
        "experiment": "exp9_stress",
        "description": "Stress test - practical limits",
        "shots": SHOTS,
        "timeout": TIMEOUT,
        "results": all_results,
    }

    out_path = RESULTS_DIR / "exp9_stress.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
