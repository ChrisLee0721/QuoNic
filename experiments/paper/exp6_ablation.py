"""Experiment 6: Ablation Study.

Measures the contribution of each QuoNic component:
1. Full QuoNic (scheduler + optimize + native)
2. No scheduler (always statevector)
3. No optimization (raw circuit)
4. No decomposition (high-level gates only)
5. External backend only (qiskit, no native)

Outputs: experiments/paper/results/exp6_ablation.json
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

SHOTS = 1024


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
    from quonic import qgate, reset
    from quonic.gates import CX, CZ, H, X
    from quonic.stack import current_circuit

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
    return current_circuit()


def build_ghz(n: int):
    from quonic import qgate, reset
    from quonic.gates import CX, H
    from quonic.stack import current_circuit

    reset()
    qgate(H, 0)
    for i in range(n - 1):
        qgate(CX, i, i + 1)
    return current_circuit()


def build_random(n: int, depth: int, seed: int = 42):
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


def run_config(circuit, config_name: str) -> dict:
    """Run a circuit under a specific ablation configuration."""
    from quonic.backends import get_backend

    try:
        t0 = time.perf_counter()

        if config_name == "full_quonic":
            # Full pipeline: scheduler + optimize + native
            from quonic.compiler import optimize
            from quonic.scheduler import schedule

            rec = schedule(circuit)
            optimized = optimize(circuit)
            result = get_backend(rec.backend).run(
                optimized, shots=SHOTS, method=rec.method
            )

        elif config_name == "no_scheduler":
            # Skip scheduler, always use native/statevector
            from quonic.compiler import optimize

            optimized = optimize(circuit)
            result = get_backend("native").run(
                optimized, shots=SHOTS, method="statevector"
            )

        elif config_name == "no_optimization":
            # Skip optimization, use scheduler for backend selection
            from quonic.scheduler import schedule

            rec = schedule(circuit)
            result = get_backend(rec.backend).run(
                circuit, shots=SHOTS, method=rec.method
            )

        elif config_name == "no_decomposition":
            # Use high-level gates (CP, CCX) directly without decompose()
            # Run with native/statevector which handles high-level gates
            from quonic.compiler import optimize
            from quonic.scheduler import schedule

            rec = schedule(circuit)
            # Skip decompose, only optimize
            optimized = optimize(circuit)
            result = get_backend(rec.backend).run(
                optimized, shots=SHOTS, method=rec.method
            )

        elif config_name == "external_only":
            # Only qiskit backend, no native simulators
            from quonic.compiler import optimize

            optimized = optimize(circuit)
            result = get_backend("qiskit").run(
                optimized, shots=SHOTS, method="statevector"
            )

        else:
            return {"config": config_name, "error": f"Unknown config: {config_name}"}

        elapsed = time.perf_counter() - t0
        return {
            "config": config_name,
            "time": round(elapsed, 4),
            "unique_outcomes": len(result.counts),
            "top_outcome": max(result.counts, key=result.counts.get),
            "error": None,
        }

    except Exception as e:
        return {
            "config": config_name,
            "time": 0,
            "unique_outcomes": 0,
            "top_outcome": None,
            "error": str(e),
        }


def main():
    test_cases = [
        ("QFT-8", lambda: build_qft(8)),
        ("QFT-12", lambda: build_qft(12)),
        ("Grover-4", lambda: build_grover(4)),
        ("Grover-6", lambda: build_grover(6)),
        ("GHZ-16", lambda: build_ghz(16)),
        ("Random-8x16", lambda: build_random(8, 16)),
    ]

    configs = [
        "full_quonic",
        "no_scheduler",
        "no_optimization",
        "no_decomposition",
        "external_only",
    ]

    all_results = []

    for name, builder in test_cases:
        circuit = builder()
        print(f"\n{'='*70}")
        print(f"Circuit: {name} (n={circuit.num_qubits}, gates={circuit.gate_count()}, "
              f"depth={circuit.depth()})")
        print(f"{'='*70}")
        print(f"{'Config':<20} {'Time':>10} {'Outcomes':>10} {'Status':>8}")
        print("-" * 50)

        circuit_results = {
            "circuit": name,
            "n_qubits": circuit.num_qubits,
            "gate_count": circuit.gate_count(),
            "depth": circuit.depth(),
            "configs": {},
        }

        for config in configs:
            r = run_config(circuit, config)
            circuit_results["configs"][config] = r
            status = "OK" if r["error"] is None else f"FAIL: {r['error'][:40]}"
            print(f"{config:<20} {r['time']:>9.4f}s {r['unique_outcomes']:>10} {status:>8}")

        all_results.append(circuit_results)

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY: Speedup relative to full_quonic")
    print(f"{'='*70}")
    header = f"{'Circuit':<18}"
    for c in configs:
        header += f" {c:>14}"
    print(header)
    print("-" * (18 + 15 * len(configs)))

    for r in all_results:
        row = f"{r['circuit']:<18}"
        full_time = r["configs"]["full_quonic"]["time"]
        for c in configs:
            t = r["configs"][c]["time"]
            if t > 0 and full_time > 0:
                ratio = full_time / t
                row += f" {ratio:>13.2f}x"
            else:
                row += f" {'ERR':>13}"
        print(row)

    # Save
    output = {
        "experiment": "exp6_ablation",
        "description": "Ablation study - component contribution",
        "shots": SHOTS,
        "configs": configs,
        "results": all_results,
    }

    out_path = RESULTS_DIR / "exp6_ablation.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
