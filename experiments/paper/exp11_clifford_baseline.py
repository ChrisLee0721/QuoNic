"""Experiment 11: Clifford Baseline — Statevector vs Stabilizer Crossover.

Empirically measures the qubit count at which stabilizer becomes faster than
statevector for pure Clifford circuits (GHZ, linear cluster).

Validates the scheduler's n=24 crossover threshold and provides the empirical
constants c_sv, c_stab for Theorem 3 (Complexity-Based Scheduling).

Outputs: experiments/paper/results/exp11_clifford_baseline.json
"""

from __future__ import annotations

import json
import time
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

SHOTS = 1024
QUBIT_RANGE = [4, 8, 12, 16, 20, 24, 28, 32, 36, 40]
REPEATS = 3


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
    """Linear cluster state: H on all, then CZ chain."""
    from quonic import qgate, reset
    from quonic.gates import CZ, H
    from quonic.stack import current_circuit

    reset()
    for i in range(n):
        qgate(H, i)
    for i in range(n - 1):
        qgate(CZ, i, i + 1)
    return current_circuit()


def time_method(circuit, backend: str, method: str, repeats: int = REPEATS) -> dict:
    """Time a backend/method combination, return median time and success flag."""
    from quonic.backends import get_backend

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        try:
            get_backend(backend).run(circuit, shots=SHOTS, method=method)
            elapsed = time.perf_counter() - t0
            times.append(elapsed)
        except Exception:
            return {"time_s": None, "error": True}
    times.sort()
    return {"time_s": round(times[len(times) // 2], 6), "error": False}  # median


def main():
    circuits = {
        "GHZ": build_ghz,
        "LinearCluster": build_linear_cluster,
    }

    all_results = []
    crossover_found = None

    print(f"{'='*80}")
    print("Clifford Baseline: Statevector vs Stabilizer Crossover")
    print(f"{'='*80}")
    print(f"{'n':>4}  {'Circuit':<16} {'SV (s)':>10} {'Stab (s)':>10} {'Ratio':>8} {'Winner':<10}")
    print("-" * 80)

    for n in QUBIT_RANGE:
        for name, builder in circuits.items():
            circuit = builder(n)

            sv = time_method(circuit, "qiskit", "statevector")
            stab = time_method(circuit, "qiskit", "stabilizer")

            if sv["error"] or stab["error"]:
                print(f"{n:>4}  {name:<16} {'ERR':>10} {'ERR':>10} {'--':>8} {'--':<10}")
                all_results.append({
                    "circuit": name, "n": n,
                    "sv_time_s": None, "stab_time_s": None,
                    "ratio": None, "winner": "error",
                })
                continue

            ratio = round(sv["time_s"] / stab["time_s"], 2) if stab["time_s"] > 0 else float("inf")
            winner = "stabilizer" if stab["time_s"] < sv["time_s"] else "statevector"

            if crossover_found is None and winner == "stabilizer":
                crossover_found = n

            print(f"{n:>4}  {name:<16} {sv['time_s']:>10.4f} {stab['time_s']:>10.4f} {ratio:>8.2f}x {winner:<10}")

            all_results.append({
                "circuit": name, "n": n,
                "sv_time_s": sv["time_s"],
                "stab_time_s": stab["time_s"],
                "ratio": ratio,
                "winner": winner,
            })

    # Estimate constants c_sv, c_stab from GHZ data
    ghz_data = [r for r in all_results if r["circuit"] == "GHZ" and r["sv_time_s"] is not None]
    if len(ghz_data) >= 2:
        # T_sv = c_sv * 2^n, T_stab = c_stab * n^2
        # Use two data points to solve for constants
        d1, d2 = ghz_data[0], ghz_data[-1]
        c_sv = (d2["sv_time_s"] - d1["sv_time_s"]) / (2**d2["n"] - 2**d1["n"])
        c_stab = (d2["stab_time_s"] - d1["stab_time_s"]) / (d2["n"]**2 - d1["n"]**2)
        theoretical_crossover = None
        if c_sv > 0 and c_stab > 0:
            # c_sv * 2^n = c_stab * n^2  =>  solve numerically
            for n_test in range(4, 128):
                if c_sv * 2**n_test <= c_stab * n_test**2:
                    theoretical_crossover = n_test
                    break
    else:
        c_sv = c_stab = theoretical_crossover = None

    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Empirical crossover (first n where stabilizer wins): {crossover_found}")
    print(f"Theoretical crossover (from estimated constants): {theoretical_crossover}")
    if c_sv is not None:
        print(f"Estimated c_sv = {c_sv:.2e}, c_stab = {c_stab:.2e}")
    print("Scheduler default threshold: n=24")

    output = {
        "experiment": "exp11_clifford_baseline",
        "description": "Statevector vs stabilizer crossover for Clifford circuits",
        "shots": SHOTS,
        "repeats": REPEATS,
        "qubit_range": QUBIT_RANGE,
        "results": all_results,
        "summary": {
            "empirical_crossover_n": crossover_found,
            "theoretical_crossover_n": theoretical_crossover,
            "scheduler_threshold_n": 24,
            "c_sv": c_sv,
            "c_stab": c_stab,
        },
    }

    out_path = RESULTS_DIR / "exp11_clifford_baseline.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
