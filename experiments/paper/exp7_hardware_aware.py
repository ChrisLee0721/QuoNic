"""Experiment 7: Hardware-Aware Compilation.

Demonstrates topology mapping and hardware integration.
Applies compile(circuit, coupling_map=..., route=True) with real coupling maps
(linear, grid, heavy-hex), measures SWAP overhead, gate count growth, and
routing time vs fully-connected compilation.

Outputs: experiments/paper/results/exp7_hardware_aware.json
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# Coupling maps: list of (qubit_a, qubit_b) edges
COUPLING_MAPS = {
    "linear-16": [(i, i + 1) for i in range(15)],
    "4x4-grid": [
        (row * 4 + col, row * 4 + col + 1)
        for row in range(4)
        for col in range(3)
    ] + [
        (row * 4 + col, (row + 1) * 4 + col)
        for row in range(3)
        for col in range(4)
    ],
    "heavy-hex-16": [
        # IBM heavy-hex topology (simplified 16-qubit subset)
        (0, 1), (1, 2), (2, 3), (3, 4),
        (1, 5), (3, 6),
        (5, 6),
        (4, 7), (7, 8), (8, 9), (9, 10),
        (8, 11), (10, 12),
        (11, 12),
        (10, 13), (13, 14), (14, 15),
    ],
}


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


def build_entangled(n: int):
    """Dense entangling circuit."""
    from quonic import qgate, reset
    from quonic.gates import CX, H
    from quonic.stack import current_circuit

    reset()
    for i in range(n):
        qgate(H, i)
    for i in range(n - 1):
        qgate(CX, i, i + 1)
    for i in range(n - 1):
        qgate(CX, i + 1, i)
    return current_circuit()


def count_swaps(circuit) -> int:
    return sum(1 for op in circuit.ops if op.name == "swap")


def count_cx(circuit) -> int:
    return sum(1 for op in circuit.ops if op.name == "cx")


def run_hardware_aware(circuit, coupling_map_name: str, edges: list) -> dict:
    """Compile circuit with hardware-aware routing."""
    from quonic.compiler import compile as quonic_compile, decompose, optimize
    from quonic.topology import CouplingMap

    n_qubits = circuit.num_qubits
    max_edge_qubit = max(max(u, v) for u, v in edges) if edges else 0
    n_topo = max(n_qubits, max_edge_qubit + 1)
    coupling_map = CouplingMap(n_topo, edges)

    original_gates = circuit.gate_count()
    original_cx = count_cx(circuit)
    original_depth = circuit.depth()

    # Fully-connected compilation (no routing needed)
    t0 = time.perf_counter()
    decomposed_fc = decompose(circuit)
    optimized_fc = optimize(decomposed_fc)
    fc_time = time.perf_counter() - t0

    # Hardware-aware compilation with routing
    t0 = time.perf_counter()
    routed = quonic_compile(circuit, coupling_map=coupling_map, route=True)
    optimized_routed = optimize(routed)
    route_time = time.perf_counter() - t0

    routed_gates = optimized_routed.gate_count()
    routed_cx = count_cx(optimized_routed)
    routed_swaps = count_swaps(optimized_routed)
    routed_depth = optimized_routed.depth()

    gate_growth = ((routed_gates / original_gates) - 1) * 100 if original_gates > 0 else 0

    return {
        "coupling_map": coupling_map_name,
        "original": {
            "gates": original_gates,
            "cx": original_cx,
            "depth": original_depth,
        },
        "fully_connected": {
            "gates": optimized_fc.gate_count(),
            "cx": count_cx(optimized_fc),
            "depth": optimized_fc.depth(),
            "time": round(fc_time, 6),
        },
        "hardware_aware": {
            "gates": routed_gates,
            "cx": routed_cx,
            "swaps": routed_swaps,
            "depth": routed_depth,
            "time": round(route_time, 6),
        },
        "gate_growth_pct": round(gate_growth, 1),
        "swap_overhead": routed_swaps,
    }


def main():
    test_cases = [
        ("QFT-8", lambda: build_qft(8)),
        ("Grover-4", lambda: build_grover(4)),
        ("Entangled-8", lambda: build_entangled(8)),
    ]

    all_results = []

    for circuit_name, builder in test_cases:
        circuit = builder()
        print(f"\n{'='*70}")
        print(f"Circuit: {circuit_name} (n={circuit.num_qubits}, gates={circuit.gate_count()})")
        print(f"{'='*70}")
        print(f"{'Topology':<16} {'Orig Gates':>10} {'FC Gates':>10} {'HW Gates':>10} "
              f"{'SWAPs':>6} {'Growth':>8} {'Route Time':>10}")
        print("-" * 75)

        circuit_results = {
            "circuit": circuit_name,
            "n_qubits": circuit.num_qubits,
            "original_gates": circuit.gate_count(),
            "topologies": {},
        }

        for map_name, edges in COUPLING_MAPS.items():
            r = run_hardware_aware(circuit, map_name, edges)
            circuit_results["topologies"][map_name] = r
            print(f"{map_name:<16} {r['original']['gates']:>10} "
                  f"{r['fully_connected']['gates']:>10} "
                  f"{r['hardware_aware']['gates']:>10} "
                  f"{r['swap_overhead']:>6} "
                  f"{r['gate_growth_pct']:>7.1f}% "
                  f"{r['hardware_aware']['time']:>9.4f}s")

        all_results.append(circuit_results)

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY: SWAP overhead by topology")
    print(f"{'='*70}")
    print(f"{'Circuit':<18} {'linear-16':>12} {'4x4-grid':>12} {'heavy-hex':>12}")
    print("-" * 55)
    for r in all_results:
        row = f"{r['circuit']:<18}"
        for map_name in COUPLING_MAPS:
            swaps = r["topologies"][map_name]["swap_overhead"]
            row += f" {swaps:>11}S"
        print(row)

    # Save
    output = {
        "experiment": "exp7_hardware_aware",
        "description": "Hardware-aware compilation with topology mapping",
        "coupling_maps": {k: len(v) for k, v in COUPLING_MAPS.items()},
        "results": all_results,
    }

    out_path = RESULTS_DIR / "exp7_hardware_aware.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
