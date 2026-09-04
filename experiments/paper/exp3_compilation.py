"""Experiment 3: Compilation Optimization Comparison.

Compares QuoNic's optimization passes against Qiskit transpiler.
Measures gate count reduction, CX count reduction, and compilation time.

Outputs: experiments/paper/results/exp3_compilation.json
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

QUONIC_PASSES = [
    ("cancel",),
    ("cancel", "commute", "cancel"),
    ("cancel", "commute", "cancel", "peephole"),
    ("cancel", "commute", "cancel", "peephole", "fuse"),
    ("cancel", "commute", "cancel", "peephole", "fuse", "prune"),
]


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


def build_grover(n: int):
    """Grover search with n qubits."""
    from quonic import qgate, reset
    from quonic.gates import CX, CZ, H, X
    from quonic.stack import current_circuit

    reset()
    # Initialize
    for i in range(n):
        qgate(H, i)
    # Oracle: mark |11...1>
    for i in range(n):
        qgate(X, i)
    qgate(CZ, 0, 1)
    for i in range(n):
        qgate(X, i)
    # Diffusion
    for i in range(n):
        qgate(H, i)
    for i in range(n):
        qgate(X, i)
    qgate(CZ, 0, 1)
    for i in range(n):
        qgate(X, i)
    for i in range(n):
        qgate(H, i)
    return current_circuit()


def build_ccz_chain(n: int):
    """Chain of CCZ gates."""
    from quonic import qgate, reset
    from quonic.gates import CCX, H
    from quonic.stack import current_circuit

    reset()
    for i in range(n):
        qgate(H, i)
    for i in range(0, n - 2, 2):
        qgate(CCX, i, i + 1, i + 2)
    return current_circuit()


def build_random(n: int, depth: int, seed: int = 42):
    """Random circuit with redundant gates (for optimization testing).

    Inserts self-inverse gate pairs (H-H, CX-CX) that should cancel.
    """
    import random

    from quonic import qgate, reset
    from quonic.gates import CX, CZ, H, Rx, Ry, Rz
    from quonic.stack import current_circuit

    rng = random.Random(seed)
    reset()
    for _ in range(depth):
        for q in range(n):
            if rng.random() < 0.6:
                gate = rng.choice([H, Rx, Ry, Rz])
                if gate in (Rx, Ry, Rz):
                    qgate(gate(rng.uniform(0, 6.28)), q)
                else:
                    qgate(gate, q)
        # Insert redundant H-H pairs (30% chance per qubit)
        for q in range(n):
            if rng.random() < 0.3:
                qgate(H, q)
                qgate(H, q)
        pairs = list(zip(range(0, n - 1, 2), range(1, n, 2)))
        rng.shuffle(pairs)
        for ctrl, tgt in pairs[: max(1, len(pairs) // 2)]:
            qgate(rng.choice([CX, CZ]), ctrl, tgt)
            # Insert redundant CX-CX pairs (20% chance)
            if rng.random() < 0.2:
                qgate(CX, ctrl, tgt)
    return current_circuit()


def count_cx(circuit) -> int:
    """Count CX gates in a circuit."""
    return sum(1 for op in circuit.ops if op.name == "cx")


def quonic_optimize(circuit, passes: tuple) -> tuple[int, int, float]:
    """Optimize circuit with QuoNic passes. Returns (gate_count, cx_count, time)."""
    from quonic.compiler import optimize

    t0 = time.perf_counter()
    optimized = optimize(circuit, passes=passes)
    elapsed = time.perf_counter() - t0
    return optimized.gate_count(), count_cx(optimized), round(elapsed, 6)


def qiskit_transpile(circuit, optimization_level: int) -> tuple[int, int, float]:
    """Transpile with Qiskit. Returns (gate_count, cx_count, time)."""
    try:
        from qiskit import QuantumCircuit, transpile as qiskit_transpile_fn

        # Translate to qiskit
        from quonic.backends.translators import TRANSLATORS

        qc = QuantumCircuit(circuit.num_qubits)
        for op in circuit.ops:
            if op.name in TRANSLATORS:
                TRANSLATORS[op.name].to_qiskit(qc, op, {})

        t0 = time.perf_counter()
        transpiled = qiskit_transpile_fn(qc, optimization_level=optimization_level)
        elapsed = time.perf_counter() - t0

        gate_count = transpiled.size()
        cx_count = transpiled.count_ops().get("cx", 0)
        return gate_count, cx_count, round(elapsed, 6)
    except Exception as e:
        return -1, -1, str(e)


def main():
    test_cases = [
        ("QFT-8", lambda: build_qft(8)),
        ("QFT-12", lambda: build_qft(12)),
        ("QFT-16", lambda: build_qft(16)),
        ("Grover-4", lambda: build_grover(4)),
        ("Grover-6", lambda: build_grover(6)),
        ("CCZ-chain-8", lambda: build_ccz_chain(8)),
        ("Random-8x16", lambda: build_random(8, 16)),
        ("Random-12x24", lambda: build_random(12, 24)),
    ]

    all_results = []

    for name, builder in test_cases:
        circuit = builder()
        original_gates = circuit.gate_count()
        original_cx = count_cx(circuit)

        print(f"\n{'='*70}")
        print(f"Circuit: {name} (gates={original_gates}, cx={original_cx}, "
              f"n={circuit.num_qubits}, depth={circuit.depth()})")
        print(f"{'='*70}")
        print(f"{'Method':<35} {'Gates':>6} {'CX':>6} {'Reduction':>10} {'Time':>10}")
        print("-" * 70)

        circuit_result = {
            "circuit": name,
            "n_qubits": circuit.num_qubits,
            "original_gates": original_gates,
            "original_cx": original_cx,
            "original_depth": circuit.depth(),
            "quonic": [],
            "qiskit": [],
        }

        # QuoNic passes
        for passes in QUONIC_PASSES:
            gates, cx, t = quonic_optimize(circuit, passes)
            reduction = (1 - gates / original_gates) * 100 if original_gates > 0 else 0
            pass_name = " → ".join(passes)
            print(f"QuoNic: {pass_name:<25} {gates:>6} {cx:>6} {reduction:>9.1f}% {t:>9.4f}s")
            circuit_result["quonic"].append({
                "passes": passes,
                "gates": gates,
                "cx": cx,
                "reduction_pct": round(reduction, 1),
                "time": t,
            })

        # Qiskit transpiler
        for level in [0, 1, 2, 3]:
            gates, cx, t = qiskit_transpile(circuit, level)
            if gates >= 0:
                reduction = (1 - gates / original_gates) * 100 if original_gates > 0 else 0
                print(f"Qiskit: optimization_level={level}       "
                      f"{gates:>6} {cx:>6} {reduction:>9.1f}% {t:>9.4f}s")
                circuit_result["qiskit"].append({
                    "level": level,
                    "gates": gates,
                    "cx": cx,
                    "reduction_pct": round(reduction, 1),
                    "time": t,
                })
            else:
                print(f"Qiskit: optimization_level={level}       ERROR: {t}")

        all_results.append(circuit_result)

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Circuit':<20} {'Original':>8} {'Best QuoNic':>12} {'Best Qiskit':>12} {'Winner':>10}")
    print("-" * 65)
    for r in all_results:
        best_q = min(r["quonic"], key=lambda x: x["gates"]) if r["quonic"] else None
        best_k = min(r["qiskit"], key=lambda x: x["gates"]) if r["qiskit"] else None
        q_gates = best_q["gates"] if best_q else "N/A"
        k_gates = best_k["gates"] if best_k else "N/A"
        if isinstance(q_gates, int) and isinstance(k_gates, int):
            winner = "QuoNic" if q_gates < k_gates else "Qiskit" if k_gates < q_gates else "Tie"
        else:
            winner = "N/A"
        print(f"{r['circuit']:<20} {r['original_gates']:>8} {str(q_gates):>12} {str(k_gates):>12} {winner:>10}")

    # Save
    output = {
        "experiment": "exp3_compilation",
        "description": "Compilation optimization comparison",
        "quonic_passes": [list(p) for p in QUONIC_PASSES],
        "results": all_results,
    }

    out_path = RESULTS_DIR / "exp3_compilation.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
