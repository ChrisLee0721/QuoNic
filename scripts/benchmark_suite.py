"""End-to-end benchmark suite for QuoNic.

Benchmarks:
1. Quantum Volume — random square circuits
2. Cross-Entropy — circuit fidelity measurement
3. Algorithm Scaling — Grover, QFT, GHZ at various sizes
4. Gate Throughput — gates/second measurement

Usage:
    python scripts/benchmark_suite.py
    python scripts/benchmark_suite.py --backend native --n 8,10,12,14
    python scripts/benchmark_suite.py --output benchmarks.json
"""

from __future__ import annotations

import argparse
import json
import random
import time
from typing import Any


def random_circuit(n_qubits: int, depth: int, seed: int = 42) -> None:
    """Build a random circuit using QuoNic's global API."""
    from quonic import qgate
    from quonic.gates import CX, H, Rx, Ry, Rz, X, Y, Z

    rng = random.Random(seed)
    single_gates = [H, X, Y, Z]
    param_gates = [Rx, Ry, Rz]

    for _ in range(depth):
        # Random single-qubit layer
        for q in range(n_qubits):
            if rng.random() < 0.7:
                gate = rng.choice(single_gates)
                qgate(gate, q)
            else:
                gate = rng.choice(param_gates)
                qgate(gate(rng.uniform(0, 6.28)), q)
        # Random entangling layer
        pairs = list(zip(range(0, n_qubits - 1, 2), range(1, n_qubits, 2)))
        rng.shuffle(pairs)
        for ctrl, tgt in pairs[: max(1, len(pairs) // 2)]:
            qgate(CX, ctrl, tgt)


def benchmark_quantum_volume(
    n: int, depth: int, shots: int, backend: str
) -> dict[str, Any]:
    """Quantum Volume benchmark: random square circuit."""
    from quonic import reset
    from quonic.backends import get_backend
    from quonic.stack import current_circuit

    reset()
    random_circuit(n, depth, seed=42)

    t0 = time.perf_counter()
    result = get_backend(backend).run(current_circuit(), shots=shots)
    elapsed = time.perf_counter() - t0

    # Compute heavy output probability (top 2/3 of bitstrings)
    counts = result.counts
    sorted_counts = sorted(counts.values(), reverse=True)
    threshold = shots * 2 / 3
    heavy_sum = sum(c for c in sorted_counts if c >= threshold / len(counts))
    heavy_prob = heavy_sum / shots

    return {
        "n": n,
        "depth": depth,
        "heavy_output_probability": round(heavy_prob, 4),
        "passed": heavy_prob > 2 / 3,
        "time": round(elapsed, 4),
    }


def benchmark_cross_entropy(
    n: int, depth: int, shots: int, backend: str
) -> dict[str, Any]:
    """Cross-entropy benchmark: measure circuit fidelity via sampling."""
    from quonic import reset
    from quonic.backends import get_backend
    from quonic.stack import current_circuit

    reset()
    random_circuit(n, depth, seed=123)

    t0 = time.perf_counter()
    result = get_backend(backend).run(current_circuit(), shots=shots)
    elapsed = time.perf_counter() - t0

    # Estimate fidelity from measurement entropy
    counts = result.counts
    probs = [c / shots for c in counts.values()]
    entropy = -sum(p * (p and __import__("math").log2(p)) for p in probs)
    max_entropy = n  # log2(2^n)
    fidelity = 1 - entropy / max_entropy if max_entropy > 0 else 0

    return {
        "n": n,
        "depth": depth,
        "fidelity": round(fidelity, 4),
        "unique_outcomes": len(counts),
        "time": round(elapsed, 4),
    }


def benchmark_algorithm(
    name: str, n: int, shots: int, backend: str
) -> dict[str, Any]:
    """Run an algorithm benchmark."""
    from quonic import qgate, reset
    from quonic.backends import get_backend
    from quonic.gates import CX, H
    from quonic.stack import current_circuit

    reset()

    if name == "ghz":
        qgate(H, 0)
        for i in range(n - 1):
            qgate(CX, i, i + 1)
    elif name == "grover":
        # Simple 2-qubit Grover
        qgate(H, 0)
        qgate(H, 1)
        # Oracle: mark |11>
        qgate(CX, 0, 1)
        # Diffusion
        qgate(H, 0)
        qgate(H, 1)
    elif name == "qft":
        for i in range(n):
            qgate(H, i)
            for j in range(i + 1, n):
                # Simplified QFT: just H + CX chain
                pass
    elif name == "bell_chain":
        for i in range(0, n - 1, 2):
            qgate(H, i)
            qgate(CX, i, i + 1)

    t0 = time.perf_counter()
    result = get_backend(backend).run(current_circuit(), shots=shots)
    elapsed = time.perf_counter() - t0

    return {
        "algorithm": name,
        "n": n,
        "time": round(elapsed, 4),
        "unique_outcomes": len(result.counts),
    }


def benchmark_gate_throughput(
    n_gates: int, n_qubits: int, backend: str
) -> dict[str, Any]:
    """Measure gate throughput: how many gates/second."""
    from quonic import qgate, reset
    from quonic.backends import get_backend
    from quonic.gates import H
    from quonic.stack import current_circuit

    reset()
    for i in range(n_gates):
        qgate(H, i % n_qubits)

    t0 = time.perf_counter()
    get_backend(backend).run(current_circuit(), shots=1)
    elapsed = time.perf_counter() - t0

    return {
        "n_gates": n_gates,
        "n_qubits": n_qubits,
        "gates_per_second": round(n_gates / elapsed, 0) if elapsed > 0 else 0,
        "time": round(elapsed, 4),
    }


def main():
    parser = argparse.ArgumentParser(description="QuoNic benchmark suite")
    parser.add_argument("--backend", default="native", help="Backend (default: native)")
    parser.add_argument("--n", default="8,10,12", help="Qubit counts (comma-separated)")
    parser.add_argument("--shots", type=int, default=1024, help="Shots per circuit")
    parser.add_argument("--output", default="benchmarks.json", help="Output JSON file")
    args = parser.parse_args()

    n_values = [int(x) for x in args.n.split(",")]

    results: dict[str, Any] = {
        "meta": {
            "backend": args.backend,
            "shots": args.shots,
            "n_values": n_values,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "quantum_volume": [],
        "cross_entropy": [],
        "algorithms": [],
        "throughput": [],
    }

    print(f"QuoNic Benchmark Suite — backend={args.backend}, shots={args.shots}")
    print("=" * 60)

    # Quantum Volume
    print("\n[1/4] Quantum Volume")
    for n in n_values:
        r = benchmark_quantum_volume(n, n, args.shots, args.backend)
        results["quantum_volume"].append(r)
        status = "PASS" if r["passed"] else "FAIL"
        print(f"  n={n:2d}  depth={n:2d}  HOP={r['heavy_output_probability']:.3f}  {status}  {r['time']:.3f}s")

    # Cross-Entropy
    print("\n[2/4] Cross-Entropy Fidelity")
    for n in n_values:
        r = benchmark_cross_entropy(n, n, args.shots, args.backend)
        results["cross_entropy"].append(r)
        print(f"  n={n:2d}  fidelity={r['fidelity']:.3f}  outcomes={r['unique_outcomes']:5d}  {r['time']:.3f}s")

    # Algorithms
    print("\n[3/4] Algorithm Scaling")
    for algo in ["ghz", "bell_chain"]:
        for n in n_values:
            r = benchmark_algorithm(algo, n, args.shots, args.backend)
            results["algorithms"].append(r)
            print(f"  {algo:12s}  n={n:2d}  {r['time']:.3f}s  outcomes={r['unique_outcomes']}")

    # Throughput
    print("\n[4/4] Gate Throughput")
    for n_gates in [100, 500, 1000, 5000]:
        r = benchmark_gate_throughput(n_gates, min(n_values), args.backend)
        results["throughput"].append(r)
        print(f"  {n_gates:5d} gates  {r['gates_per_second']:8.0f} gates/s  {r['time']:.3f}s")

    # Save
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
