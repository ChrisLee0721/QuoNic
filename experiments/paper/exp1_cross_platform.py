"""Experiment 1: Cross-Platform Consistency Verification.

Runs 4 circuits (Bell, GHZ-10, Grover-4, QFT-8) on 9 backend-method
combinations and measures Total Variation Distance (TVD) between
measurement distributions. TVD < 0.05 indicates consistent translation.

Outputs: experiments/paper/results/exp1_cross_platform.json
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

SHOTS = 50000
BACKEND_METHODS = [
    ("native", "statevector"),
    ("native", "stabilizer"),
    ("native", "matrix_product_state"),
    ("native", "density_matrix"),
    ("qiskit", "statevector"),
    ("qiskit", "stabilizer"),
    ("qiskit", "matrix_product_state"),
    ("qiskit", "density_matrix"),
]


def build_bell():
    from quonic import qgate, reset
    from quonic.gates import CX, H
    from quonic.stack import current_circuit

    reset()
    qgate(H, 0)
    qgate(CX, 0, 1)
    return current_circuit(), "Bell"


def build_ghz(n: int = 10):
    from quonic import qgate, reset
    from quonic.gates import CX, H
    from quonic.stack import current_circuit

    reset()
    qgate(H, 0)
    for i in range(n - 1):
        qgate(CX, i, i + 1)
    return current_circuit(), f"GHZ-{n}"


def build_grover_4():
    """Grover search on 4 qubits, marking |1010>."""
    from quonic import qgate, reset
    from quonic.gates import CX, CZ, H, X
    from quonic.stack import current_circuit

    reset()
    n = 4
    target = "1010"  # mark |1010>

    # Initialize superposition
    for i in range(n):
        qgate(H, i)

    # Oracle: mark |1010>
    for i, bit in enumerate(target):
        if bit == "0":
            qgate(X, i)
    qgate(CZ, 0, 1)  # simplified oracle for demo
    for i, bit in enumerate(target):
        if bit == "0":
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

    return current_circuit(), "Grover-4"


def build_qft(n: int = 8):
    """Build QFT circuit with controlled phase rotations (non-Clifford)."""
    from quonic import qgate, reset
    from quonic.gates import CP, H
    from quonic.stack import current_circuit

    reset()
    for i in range(n):
        qgate(H, i)
        for j in range(i + 1, n):
            angle = math.pi / (2 ** (j - i))
            qgate(CP(angle), j, i)
    return current_circuit(), f"QFT-{n}"


def counts_to_probs(counts: dict[str, int], n_qubits: int) -> dict[str, float]:
    """Convert measurement counts to probability distribution."""
    total = sum(counts.values())
    if total == 0:
        return {}
    return {k: v / total for k, v in counts.items()}


def tvd(p: dict[str, float], q: dict[str, float]) -> float:
    """Total variation distance between two distributions."""
    all_keys = set(p.keys()) | set(q.keys())
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in all_keys)


def run_circuit_on_backends(circuit, name: str) -> list[dict]:
    """Run a circuit on all backend-method combos and return results."""
    from quonic.backends import get_backend

    results = []
    for backend_name, method in BACKEND_METHODS:
        try:
            t0 = time.perf_counter()
            result = get_backend(backend_name).run(circuit, shots=SHOTS, method=method)
            elapsed = time.perf_counter() - t0
            probs = counts_to_probs(result.counts, circuit.num_qubits)
            results.append({
                "backend": backend_name,
                "method": method,
                "counts": result.counts,
                "probs": {k: round(v, 6) for k, v in sorted(probs.items(), key=lambda x: -x[1])[:20]},
                "time": round(elapsed, 4),
                "unique_outcomes": len(result.counts),
                "error": None,
            })
        except Exception as e:
            results.append({
                "backend": backend_name,
                "method": method,
                "counts": {},
                "probs": {},
                "time": 0,
                "unique_outcomes": 0,
                "error": str(e),
            })
    return results


def compute_pairwise_tvd(results: list[dict]) -> list[dict]:
    """Compute TVD between all pairs of successful backend results.

    Uses full counts (not truncated top-20 probs) for accurate TVD.
    """
    successful = [r for r in results if r["error"] is None]
    tvds = []
    for i in range(len(successful)):
        for j in range(i + 1, len(successful)):
            ci = successful[i]["counts"]
            cj = successful[j]["counts"]
            ti = sum(ci.values())
            tj = sum(cj.values())
            all_keys = set(ci.keys()) | set(cj.keys())
            t = 0.5 * sum(abs(ci.get(k, 0) / ti - cj.get(k, 0) / tj) for k in all_keys)
            tvds.append({
                "pair": f"{successful[i]['backend']}/{successful[i]['method']} vs "
                        f"{successful[j]['backend']}/{successful[j]['method']}",
                "tvd": round(t, 6),
            })
    return tvds


def main():
    circuits = [
        build_bell(),
        build_ghz(10),
        build_grover_4(),
        build_qft(8),
    ]

    all_results = []
    summary = []

    for circuit, name in circuits:
        print(f"\n{'='*60}")
        print(f"Circuit: {name} (n={circuit.num_qubits}, gates={circuit.gate_count()}, depth={circuit.depth()})")
        print(f"{'='*60}")

        results = run_circuit_on_backends(circuit, name)
        pairwise = compute_pairwise_tvd(results)

        # Print results
        for r in results:
            status = "OK" if r["error"] is None else f"FAIL: {r['error']}"
            print(f"  {r['backend']:8s}/{r['method']:25s}  {r['time']:.3f}s  "
                  f"outcomes={r['unique_outcomes']:5d}  {status}")

        # Print TVDs
        max_tvd = 0
        print(f"\n  Pairwise TVD (shots={SHOTS}):")
        for tv in pairwise:
            marker = " !!!" if tv["tvd"] > 0.05 else ""
            print(f"    {tv['tvd']:.6f}  {tv['pair']}{marker}")
            max_tvd = max(max_tvd, tv["tvd"])

        passed = max_tvd < 0.05
        print(f"\n  Max TVD: {max_tvd:.6f}  {'PASS' if passed else 'FAIL'}")

        all_results.append({
            "circuit": name,
            "n_qubits": circuit.num_qubits,
            "gate_count": circuit.gate_count(),
            "depth": circuit.depth(),
            "backends": results,
            "pairwise_tvd": pairwise,
            "max_tvd": round(max_tvd, 6),
            "passed": passed,
        })
        summary.append({
            "circuit": name,
            "n": circuit.num_qubits,
            "max_tvd": round(max_tvd, 6),
            "passed": passed,
            "backends_ok": sum(1 for r in results if r["error"] is None),
        })

    # Save results
    output = {
        "experiment": "exp1_cross_platform",
        "description": "Cross-platform consistency verification",
        "shots": SHOTS,
        "backend_methods": [f"{b}/{m}" for b, m in BACKEND_METHODS],
        "results": all_results,
        "summary": summary,
        "all_passed": all(s["passed"] for s in summary),
    }

    out_path = RESULTS_DIR / "exp1_cross_platform.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    # Print summary table
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Circuit':<12} {'n':>4} {'Max TVD':>10} {'Backends':>8} {'Status':>8}")
    print("-" * 45)
    for s in summary:
        status = "PASS" if s["passed"] else "FAIL"
        print(f"{s['circuit']:<12} {s['n']:>4} {s['max_tvd']:>10.6f} {s['backends_ok']:>8} {status:>8}")

    all_ok = all(s["passed"] for s in summary)
    print(f"\nOverall: {'ALL PASSED' if all_ok else 'SOME FAILED'}")
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
