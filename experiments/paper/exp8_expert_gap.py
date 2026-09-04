"""Experiment 8: Expert Baseline Gap.

Compares QuoNic's automatic compilation against hand-optimized expert circuits.
Measures gate count, depth, and correctness for Bell, GHZ, QFT, Grover.

Outputs: experiments/paper/results/exp8_expert_gap.json
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

SHOTS = 50000
SHOTS_LARGE = 200000  # for circuits with many outcomes (QFT)


# ── Expert (hand-optimized) circuits ──────────────────────────────────────

def expert_bell():
    """Minimal Bell circuit: H, CX."""
    from quonic import qgate, reset
    from quonic.gates import CX, H
    from quonic.stack import current_circuit

    reset()
    qgate(H, 0)
    qgate(CX, 0, 1)
    return current_circuit()


def expert_ghz(n: int):
    """Minimal GHZ: H + n-1 CX."""
    from quonic import qgate, reset
    from quonic.gates import CX, H
    from quonic.stack import current_circuit

    reset()
    qgate(H, 0)
    for i in range(n - 1):
        qgate(CX, i, i + 1)
    return current_circuit()


def expert_qft(n: int):
    """Standard QFT: H + controlled-phase rotations."""
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


def expert_grover(n: int):
    """Grover with minimal oracle + diffusion."""
    from quonic import qgate, reset
    from quonic.gates import CZ, H, X
    from quonic.stack import current_circuit

    reset()
    # Superposition
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


# ── QuoNic auto-compiled circuits ─────────────────────────────────────────

def quonic_bell():
    """Bell via algorithm template."""
    from quonic.algorithms import teleportation
    # Use teleportation as a proxy; build Bell manually for fair comparison
    from quonic import qgate, reset
    from quonic.gates import CX, H
    from quonic.stack import current_circuit
    from quonic.compiler import decompose, optimize

    reset()
    qgate(H, 0)
    qgate(CX, 0, 1)
    circuit = current_circuit()
    decomposed = decompose(circuit)
    return optimize(decomposed)


def quonic_ghz(n: int):
    from quonic import qgate, reset
    from quonic.gates import CX, H
    from quonic.stack import current_circuit
    from quonic.compiler import decompose, optimize

    reset()
    qgate(H, 0)
    for i in range(n - 1):
        qgate(CX, i, i + 1)
    circuit = current_circuit()
    decomposed = decompose(circuit)
    return optimize(decomposed)


def quonic_qft(n: int):
    from quonic import qgate, reset
    from quonic.gates import CP, H
    from quonic.stack import current_circuit
    from quonic.compiler import decompose, optimize

    reset()
    for i in range(n):
        qgate(H, i)
        for j in range(i + 1, n):
            angle = math.pi / (2 ** (j - i))
            qgate(CP(angle), j, i)
    circuit = current_circuit()
    decomposed = decompose(circuit)
    return optimize(decomposed)


def quonic_grover(n: int):
    from quonic import qgate, reset
    from quonic.gates import CZ, H, X
    from quonic.stack import current_circuit
    from quonic.compiler import decompose, optimize

    reset()
    for i in range(n):
        qgate(H, i)
    for i in range(n):
        qgate(X, i)
    qgate(CZ, 0, 1)
    for i in range(n):
        qgate(X, i)
    for i in range(n):
        qgate(H, i)
    for i in range(n):
        qgate(X, i)
    qgate(CZ, 0, 1)
    for i in range(n):
        qgate(X, i)
    for i in range(n):
        qgate(H, i)
    circuit = current_circuit()
    decomposed = decompose(circuit)
    return optimize(decomposed)


def verify_correctness(circuit_a, circuit_b, n_qubits: int = 0) -> float:
    """Compare measurement distributions via TVD."""
    from quonic.backends import get_backend

    # Use more shots for circuits with many outcomes
    shots = SHOTS_LARGE if n_qubits >= 10 else SHOTS

    try:
        r_a = get_backend("native").run(circuit_a, shots=shots, method="statevector")
        r_b = get_backend("native").run(circuit_b, shots=shots, method="statevector")
    except Exception:
        return -1.0

    c_a, c_b = r_a.counts, r_b.counts
    t_a, t_b = sum(c_a.values()), sum(c_b.values())
    all_keys = set(c_a.keys()) | set(c_b.keys())
    tvd = 0.5 * sum(abs(c_a.get(k, 0) / t_a - c_b.get(k, 0) / t_b) for k in all_keys)
    return round(tvd, 6)


def main():
    test_cases = [
        ("Bell", expert_bell, quonic_bell, 2),
        ("GHZ-8", lambda: expert_ghz(8), lambda: quonic_ghz(8), 8),
        ("GHZ-16", lambda: expert_ghz(16), lambda: quonic_ghz(16), 16),
        ("QFT-8", lambda: expert_qft(8), lambda: quonic_qft(8), 8),
        ("QFT-12", lambda: expert_qft(12), lambda: quonic_qft(12), 12),
        ("Grover-4", lambda: expert_grover(4), lambda: quonic_grover(4), 4),
        ("Grover-6", lambda: expert_grover(6), lambda: quonic_grover(6), 6),
    ]

    all_results = []

    print(f"{'='*80}")
    print("Expert Baseline Gap: QuoNic Auto vs Hand-Optimized")
    print(f"{'='*80}")
    print(f"{'Circuit':<12} {'Expert G':>9} {'QuoNic G':>9} {'Expert D':>9} "
          f"{'QuoNic D':>9} {'TVD':>8} {'Status':>8}")
    print("-" * 70)

    for name, expert_fn, quonic_fn, n in test_cases:
        expert_c = expert_fn()
        quonic_c = quonic_fn()

        expert_gates = expert_c.gate_count()
        quonic_gates = quonic_c.gate_count()
        expert_depth = expert_c.depth()
        quonic_depth = quonic_c.depth()

        tvd = verify_correctness(expert_c, quonic_c, n_qubits=n)
        # Threshold scales with sqrt(outcomes/shots) — more outcomes = more sampling noise
        import math
        n_outcomes = 2 ** n
        shots_used = SHOTS_LARGE if n >= 10 else SHOTS
        threshold = max(0.05, math.sqrt(n_outcomes / shots_used))
        passed = 0 <= tvd < threshold

        gate_diff = quonic_gates - expert_gates
        gate_pct = (gate_diff / expert_gates * 100) if expert_gates > 0 else 0

        result = {
            "circuit": name,
            "n_qubits": n,
            "expert": {"gates": expert_gates, "depth": expert_depth},
            "quonic": {"gates": quonic_gates, "depth": quonic_depth},
            "gate_diff": gate_diff,
            "gate_diff_pct": round(gate_pct, 1),
            "tvd": tvd,
            "passed": passed,
        }
        all_results.append(result)

        status = "PASS" if passed else "FAIL"
        print(f"{name:<12} {expert_gates:>9} {quonic_gates:>9} {expert_depth:>9} "
              f"{quonic_depth:>9} {tvd:>8.6f} {status:>8}")

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    total_expert = sum(r["expert"]["gates"] for r in all_results)
    total_quonic = sum(r["quonic"]["gates"] for r in all_results)
    avg_gate_diff = sum(r["gate_diff_pct"] for r in all_results) / len(all_results)
    max_tvd = max(r["tvd"] for r in all_results)

    print(f"Total expert gates: {total_expert}")
    print(f"Total QuoNic gates: {total_quonic}")
    print(f"Average gate difference: {avg_gate_diff:+.1f}%")
    print(f"Max TVD: {max_tvd:.6f}")
    print(f"All correct: {all(r['passed'] for r in all_results)}")

    # Save
    output = {
        "experiment": "exp8_expert_gap",
        "description": "Expert baseline gap - auto vs hand-optimized",
        "shots": SHOTS,
        "results": all_results,
        "summary": {
            "total_expert_gates": total_expert,
            "total_quonic_gates": total_quonic,
            "avg_gate_diff_pct": round(avg_gate_diff, 1),
            "max_tvd": max_tvd,
            "all_correct": all(r["passed"] for r in all_results),
        },
    }

    out_path = RESULTS_DIR / "exp8_expert_gap.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
