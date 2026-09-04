"""Experiment 12: Tuna-17 Baseline TVD (Control Experiment).

Runs a pure Clifford GHZ circuit on Tuna-17 WITHOUT groverize() to establish
baseline hardware noise. If GHZ TVD ≈ 0.1, the0.113 RUS-GHZ-2 TVD is hardware
noise. If GHZ TVD ≈ 0.01, groverize() has a problem on Tuna-17.

Outputs: experiments/paper/results/exp12_tuna_baseline.json
"""

from __future__ import annotations

import json
import time
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

SHOTS = 1024


def build_ghz(n: int):
    from quonic import qgate, reset
    from quonic.gates import CX, H
    from quonic.stack import current_circuit

    reset()
    qgate(H, 0)
    for i in range(n - 1):
        qgate(CX, i, i + 1)
    return current_circuit()


def run_simulator(circuit) -> dict:
    """Run on native statevector simulator."""
    from quonic.backends import get_backend

    t0 = time.perf_counter()
    result = get_backend("native").run(circuit, shots=SHOTS, method="statevector")
    elapsed = time.perf_counter() - t0
    return {
        "platform": "simulator",
        "backend": "native/statevector",
        "time": round(elapsed, 4),
        "counts": dict(result.counts),
    }


def run_tuna(circuit) -> dict:
    """Run on Quantum Inspire Tuna-17."""
    from quonic.backends import get_backend

    t0 = time.perf_counter()
    result = get_backend("qi", device="tuna17").run(circuit, shots=SHOTS)
    elapsed = time.perf_counter() - t0
    return {
        "platform": "tuna17",
        "backend": "qi/tuna17",
        "time": round(elapsed, 4),
        "counts": dict(result.counts),
    }


def tvd(counts_a: dict, counts_b: dict) -> float:
    """Total variation distance between two count distributions."""
    all_keys = set(counts_a) | set(counts_b)
    total_a = sum(counts_a.values())
    total_b = sum(counts_b.values())
    return 0.5 * sum(
        abs(counts_a.get(k, 0) / total_a - counts_b.get(k, 0) / total_b)
        for k in all_keys
    )


def main():
    test_cases = [
        ("GHZ-4", build_ghz(4)),
        ("GHZ-6", build_ghz(6)),
        ("GHZ-8", build_ghz(8)),
        ("GHZ-10", build_ghz(10)),
    ]

    all_results = []
    print(f"{'='*70}")
    print("Tuna-17 Baseline TVD (Control Experiment)")
    print(f"{'='*70}")
    print(f"{'Circuit':<12} {'n':>4} {'Gates':>6} {'Depth':>6} {'TVD':>8} {'Tuna(s)':>10}")
    print("-" * 70)

    for name, circuit in test_cases:
        sim = run_simulator(circuit)
        tuna = run_tuna(circuit)
        t = tvd(sim["counts"], tuna["counts"])

        print(f"{name:<12} {circuit.num_qubits:>4} {circuit.gate_count():>6} "
              f"{circuit.depth():>6} {t:>8.4f} {tuna['time']:>10.2f}")

        all_results.append({
            "circuit": name,
            "n_qubits": circuit.num_qubits,
            "gate_count": circuit.gate_count(),
            "depth": circuit.depth(),
            "simulator": sim,
            "tuna17": tuna,
            "tvd": round(t, 4),
        })

    # Compare with groverize results
    print(f"\n{'='*70}")
    print("COMPARISON WITH GROVERIZE RESULTS")
    print(f"{'='*70}")
    print(f"{'Circuit':<20} {'TVD':>8} {'Type':<20}")
    print("-" * 70)
    for r in all_results:
        print(f"{r['circuit']:<20} {r['tvd']:>8.4f} {'Clifford (no groverize)':<20}")
    print(f"{'RUS-Ry(2π/3)':<20} {'0.0527':>8} {'groverize (2q,12g)':<20}")
    print(f"{'RUS-GHZ-2':<20} {'0.1133':>8} {'groverize (3q,17g)':<20}")

    output = {
        "experiment": "exp12_tuna_baseline",
        "description": "Tuna-17 baseline TVD for Clifford circuits (control for groverize attribution)",
        "shots": SHOTS,
        "results": all_results,
        "groverize_reference": {
            "RUS-Ry": {"tvd": 0.0527, "n": 2, "gates": 12, "depth": 12},
            "RUS-GHZ-2": {"tvd": 0.1133, "n": 3, "gates": 17, "depth": 15},
        },
    }

    out_path = RESULTS_DIR / "exp12_tuna_baseline.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
