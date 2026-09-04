"""Experiment 10: Groverize Hardware Validation.

Demonstrates the cwhile → groverize → real hardware pipeline:
  1. Build a repeat-until-success (cwhile) circuit
  2. Compile it into a static Grover circuit via groverize()
  3. Run on simulator (baseline) and 3 real hardware platforms
  4. Compare success rates and TVD

Platforms:
  - Simulator: native/statevector (baseline)
  - Tuna-9: Quantum Inspire superconducting (via qi backend)
  - AWS Braket: IonQ Aria (via braket backend)
  - Wukong-180: OriginQ superconducting (via originq backend)

Outputs: experiments/paper/results/exp10_groverize_hardware.json
"""

from __future__ import annotations

import json
import time
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

SHOTS = 1024


def build_cwhile_rotation():
    """Build a cwhile circuit: Ry rotation + measure until success.

    This is the canonical repeat-until-success (RUS) pattern:
    - Apply Ry(2π/3) to qubit 0 → P(success) = sin²(π/3) = 3/4
    - Measure; if result == 0, repeat
    - Expected iterations: 1/p = 4/3 ≈ 1.33

    After groverize(), this becomes a static amplitude amplification circuit
    that can run on ANY backend without classical feedback support.
    """
    import math as _math

    from quonic import qgate, reset
    from quonic.gates import Ry
    from quonic.qif import creg, cwhile
    from quonic.stack import current_circuit

    reset()
    flag = creg("flag")
    with cwhile(flag, until=0) as loop:
        qgate(Ry(2 * _math.pi / 3), 0)
        flag.measure(0)

    return loop, current_circuit()


def build_cwhile_ghz():
    """Build a cwhile circuit: GHZ preparation + measure until all-zero.

    - Prepare GHZ state on 2 qubits
    - Measure qubit 1; if result == 1, apply X to flip and retry
    - Success when both qubits are 0

    This tests groverize on a multi-qubit entangled circuit.
    """
    from quonic import qgate, reset
    from quonic.gates import CX, H
    from quonic.qif import creg, cwhile
    from quonic.stack import current_circuit

    reset()
    flag = creg("flag")
    with cwhile(flag, until=0) as loop:
        qgate(H, 0)
        qgate(CX, 0, 1)
        flag.measure(1)

    return loop, current_circuit()


def _compile_for_hardware(circuit):
    """Compile groverized circuit to basic gates for hardware backends."""
    from quonic.compiler import compile

    return compile(circuit, route=True)


def run_simulator_baseline(circuit, name: str) -> dict:
    """Run groverized circuit on simulator as baseline."""
    from quonic.backends import get_backend

    t0 = time.perf_counter()
    result = get_backend("native").run(circuit, shots=SHOTS, method="statevector")
    t = time.perf_counter() - t0

    return {
        "platform": "simulator",
        "backend": "native/statevector",
        "time": round(t, 4),
        "counts": result.counts,
        "success_rate": _success_rate(result.counts),
    }


def run_tuna(circuit, name: str) -> dict:
    """Run on Quantum Inspire Tuna-9 superconducting hardware."""
    try:
        from quonic.backends import get_backend

        compiled = _compile_for_hardware(circuit)
        t0 = time.perf_counter()
        result = get_backend("qi", device="tuna17").run(compiled, shots=SHOTS)
        t = time.perf_counter() - t0

        return {
            "platform": "tuna17",
            "backend": "qi/tuna17",
            "time": round(t, 4),
            "counts": result.counts,
            "success_rate": _success_rate(result.counts),
        }
    except Exception as e:
        return {
            "platform": "tuna9",
            "backend": "qi/tuna9",
            "error": str(e)[:200],
        }


def run_braket(circuit, name: str) -> dict:
    """Run on AWS Braket (IonQ Aria or local simulator)."""
    try:
        from quonic.backends import get_backend

        # Try IonQ Aria first, fallback to local simulator
        devices = [
            "arn:aws:braket:us-east-1::device/qpu/ionq/Aria-1",
            "local",
        ]
        compiled = _compile_for_hardware(circuit)
        for device in devices:
            try:
                t0 = time.perf_counter()
                result = get_backend("braket", device=device).run(compiled, shots=SHOTS)
                t = time.perf_counter() - t0
                return {
                    "platform": "braket",
                    "backend": f"braket/{device.split('/')[-1]}",
                    "time": round(t, 4),
                    "counts": result.counts,
                    "success_rate": _success_rate(result.counts),
                }
            except Exception:
                continue
        return {"platform": "braket", "error": "All Braket devices failed"}
    except Exception as e:
        return {
            "platform": "braket",
            "error": str(e)[:200],
        }


def run_wukong(circuit, name: str) -> dict:
    """Run on OriginQ Wukong-180 superconducting hardware."""
    try:
        from quonic.backends import get_backend

        compiled = _compile_for_hardware(circuit)
        t0 = time.perf_counter()
        result = get_backend("originq", device="WK_C180").run(compiled, shots=SHOTS)
        t = time.perf_counter() - t0

        return {
            "platform": "wukong180",
            "backend": "originq/WK_C180",
            "time": round(t, 4),
            "counts": result.counts,
            "success_rate": _success_rate(result.counts),
        }
    except Exception as e:
        return {
            "platform": "wukong180",
            "backend": "originq/WK_C180",
            "error": str(e)[:200],
        }


def _success_rate(counts: dict[str, int]) -> float:
    """Compute success rate: fraction of shots where ancilla == 0 (success)."""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    # After groverize, ancilla is highest qubit; success = ancilla == 0
    # Bitstring convention: rightmost bit is qubit 0
    success = sum(v for k, v in counts.items() if k[-1] == "0")
    return round(success / total, 4)


def tvd(p: dict, q: dict) -> float:
    """Total Variation Distance between two count distributions."""
    all_keys = set(p) | set(q)
    total_p = sum(p.values()) or 1
    total_q = sum(q.values()) or 1
    return round(
        sum(abs(p.get(k, 0) / total_p - q.get(k, 0) / total_q) for k in all_keys) / 2,
        4,
    )


def main():
    test_cases = [
        ("RUS-Ry(2π/3)", build_cwhile_rotation),
        ("RUS-GHZ-2", build_cwhile_ghz),
    ]

    all_results = []
    print(f"{'='*80}")
    print("Experiment 10: Groverize Hardware Validation")
    print(f"{'='*80}")

    for name, builder in test_cases:
        print(f"\n{'─'*80}")
        print(f"Circuit: {name}")

        loop, _ = builder()
        groverized = loop.groverize()
        n_qubits = groverized.num_qubits
        gate_count = groverized.gate_count()
        depth = groverized.depth()
        print(f"  Groverized: {n_qubits} qubits, {gate_count} gates, depth {depth}")

        # Simulator baseline
        sim = run_simulator_baseline(groverized, name)
        print(f"  Simulator: {sim['success_rate']:.1%} success, {sim['time']:.4f}s")

        # Real hardware
        results = {"simulator": sim}
        for runner, label in [
            (run_tuna, "Tuna-9"),
            (run_braket, "AWS Braket"),
            (run_wukong, "Wukong-180"),
        ]:
            print(f"  Running on {label}...", end=" ", flush=True)
            hw = runner(groverized, name)
            if "error" in hw:
                print(f"ERROR: {hw['error'][:80]}")
            else:
                print(f"{hw['success_rate']:.1%} success, {hw['time']:.4f}s")
                hw["tvd_vs_simulator"] = tvd(sim["counts"], hw["counts"])
            results[label] = hw

        all_results.append({
            "circuit": name,
            "n_qubits": n_qubits,
            "gate_count": gate_count,
            "depth": depth,
            "results": results,
        })

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"{'Circuit':<20} {'Platform':<15} {'Success':>8} {'TVD':>8} {'Time':>8}")
    print("-" * 60)
    for r in all_results:
        for platform, data in r["results"].items():
            if "error" in data:
                print(f"{r['circuit']:<20} {platform:<15} {'ERROR':>8}")
            else:
                tvd_val = data.get("tvd_vs_simulator", "-")
                tvd_str = f"{tvd_val:.4f}" if isinstance(tvd_val, float) else str(tvd_val)
                print(f"{r['circuit']:<20} {platform:<15} {data['success_rate']:>7.1%} {tvd_str:>8} {data['time']:>7.4f}s")

    # Save
    output = {
        "experiment": "exp10_groverize_hardware",
        "description": "cwhile → groverize → real hardware validation",
        "shots": SHOTS,
        "results": all_results,
    }
    out_path = RESULTS_DIR / "exp10_groverize_hardware.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
