"""Quantum Inspire Hardware Validation.

Runs representative circuits on Quantum Inspire Tuna-17 hardware.
"""

from __future__ import annotations

import json
import math
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


def build_grover_4():
    from quonic import qgate, reset
    from quonic.gates import CX, H, X, CZ
    from quonic.stack import current_circuit
    reset()
    # Oracle for |1010>
    for i in range(4):
        qgate(H, i)
    # Oracle: mark |1010>
    qgate(X, 0)
    qgate(X, 2)
    qgate(CZ, 0, 1)
    qgate(CZ, 2, 3)
    qgate(X, 0)
    qgate(X, 2)
    # Diffusion
    for i in range(4):
        qgate(H, i)
        qgate(X, i)
    qgate(CZ, 0, 1)
    qgate(CZ, 2, 3)
    for i in range(4):
        qgate(X, i)
        qgate(H, i)
    return current_circuit()


def main():
    from quonic.backends import get_backend

    print("=" * 60)
    print("Quantum Inspire Hardware Validation")
    print("=" * 60)

    circuits = [
        ("GHZ-4", lambda: build_ghz(4)),
        ("GHZ-8", lambda: build_ghz(8)),
        ("QFT-4", lambda: build_qft(4)),
        ("Grover-4", lambda: build_grover_4()),
    ]

    results = []

    for name, builder in circuits:
        print(f"\n{name}:")
        circuit = builder()

        # Simulator baseline
        print("  Running on simulator...")
        sim_backend = get_backend("native")
        t0 = time.perf_counter()
        sim_result = sim_backend.run(circuit, shots=SHOTS, method="statevector")
        t_sim = time.perf_counter() - t0
        print(f"  Simulator: {sim_result.counts} ({t_sim:.2f}s)")

        # Quantum Inspire hardware
        print("  Running on Quantum Inspire Tuna-17...")
        qi_backend = get_backend("qi")
        t0 = time.perf_counter()
        qi_result = qi_backend.run(circuit, shots=SHOTS)
        t_qi = time.perf_counter() - t0
        print(f"  QI Tuna-17: {qi_result.counts} ({t_qi:.2f}s)")

        # Compute TVD
        all_keys = set(sim_result.counts) | set(qi_result.counts)
        total_sim = sum(sim_result.counts.values())
        total_qi = sum(qi_result.counts.values())
        tvd = sum(
            abs(sim_result.counts.get(k, 0) / total_sim - qi_result.counts.get(k, 0) / total_qi)
            for k in all_keys
        ) / 2
        print(f"  TVD: {tvd:.4f}")

        results.append({
            "circuit": name,
            "n_qubits": circuit.num_qubits,
            "simulator": {
                "counts": sim_result.counts,
                "time": round(t_sim, 4),
            },
            "hardware": {
                "platform": "qi_tuna17",
                "counts": qi_result.counts,
                "time": round(t_qi, 4),
            },
            "tvd": round(tvd, 4),
        })

    # Save results
    output = {
        "experiment": "exp_hardware_qi",
        "description": "Quantum Inspire Tuna-17 hardware validation",
        "shots": SHOTS,
        "results": results,
    }

    out_path = RESULTS_DIR / "exp_hardware_qi.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Circuit':<15} {'n':>4} {'TVD':>8} {'QI Time':>10}")
    print("-" * 40)
    for r in results:
        print(f"{r['circuit']:<15} {r['n_qubits']:>4} {r['tvd']:>8.4f} {r['hardware']['time']:>10.2f}s")


if __name__ == "__main__":
    main()
