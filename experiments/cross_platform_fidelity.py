"""
Cross-platform F_gate extraction experiment.
Platform: Origin Quantum WK_C180 (machine1)
Qubits: 102 (control), 111 (target)
Method: Repeat CNOT k times, measure ideal outcome alternation.
        CNOT is self-inverse: odd k → |11>, even k → |10>.
"""

import os
import sys
import json
import time
import numpy as np
from datetime import datetime

sys.path.insert(0, r"C:/Users/26427/AppData/Local/Programs/Python/Python312")

import pyqpanda3 as pq
from pyqpanda3.qcloud.qcloud import QCloudService
from pyqpanda3.transpilation import Transpiler


def build_cnot_repeat_circuit(k: int):
    """Build circuit: X(q0) → CNOT(q0,q1) repeated k times → measure.

    Logical qubits 0,1. Transpiler maps to physical qubits.
    - Odd k: ideal outcome is |11> (target flipped odd times)
    - Even k: ideal outcome is |10> (target flipped back)
    """
    prog = pq.core.QProg()

    # Put control in |1>
    prog << pq.core.X(pq.core.Qubit(0))

    # Repeat CNOT k times (no RY, pure CNOT chain)
    for _ in range(k):
        prog << pq.core.CNOT(pq.core.Qubit(0), pq.core.Qubit(1))

    # Measure
    prog << pq.core.measure(pq.core.Qubit(0), 0)
    prog << pq.core.measure(pq.core.Qubit(1), 1)

    return prog


def ideal_outcome(k: int) -> str:
    """Ideal measurement outcome for k CNOTs with control=|1>."""
    # Odd k: target flipped → |11>, even k: target back → |10>
    return "11" if k % 2 == 1 else "10"


def run_experiment(depths: list[int], shots: int = 4000):
    api_key = os.environ.get("ORIGINGQ_API_KEY", "")
    if not api_key:
        print("ERROR: ORIGINGQ_API_KEY not set")
        return

    service = QCloudService(api_key)
    device = "WK_C180"

    backends = service.backends()
    if device not in backends:
        print(f"ERROR: {device} not available. Backends: {backends}")
        return

    backend = service.backend(device)
    chip_backend = backend.chip_info().get_chip_backend()
    transpiler = Transpiler()

    CONTROL_QUBIT = 102
    TARGET_QUBIT = 111

    results = {}

    for k in depths:
        print(f"\n--- Depth k={k} ({k} CNOT layers) ---")

        prog = build_cnot_repeat_circuit(k)

        # Transpile: optimization OFF, fixed qubit mapping
        prog_transpiled = transpiler.transpile(
            prog,
            chip_backend,
            init_mapping={0: CONTROL_QUBIT, 1: TARGET_QUBIT},
            optimization_level=0,
        )

        t0 = time.time()
        job = backend.run(prog_transpiled, shots)
        raw = job.result()
        elapsed = time.time() - t0

        print(f"  Status: {raw.job_status()}")
        print(f"  Time: {elapsed:.1f}s")

        if raw.error_message():
            print(f"  ERROR: {raw.error_message()}")
            continue

        probs = raw.get_probs()
        counts = {}
        for key, prob in probs.items():
            bs = str(key)
            if bs.startswith("0x"):
                bs = format(int(bs, 16), "02b")
            elif bs.startswith("0b"):
                bs = bs[2:].zfill(2)
            counts[bs] = max(1, round(prob * shots))

        print(f"  Counts: {counts}")

        ideal = ideal_outcome(k)
        p_ideal = counts.get(ideal, 0) / shots
        print(f"  Ideal: |{ideal}>, P = {p_ideal:.4f}")

        results[k] = {
            "depth": k,
            "ideal_state": ideal,
            "counts": counts,
            "p_ideal": p_ideal,
            "shots": shots,
            "elapsed": elapsed,
        }

    return results


def fit_gate_fidelity(results: dict):
    """Fit F_gate from decay curve.

    For even k: P(|10>) = 1 - 2*p_err*(1-p_err) ≈ 1 - 2*p_err
    For odd k: P(|11>) = 1 - 2*p_err*(1-p_err) ≈ 1 - 2*p_err
    where p_err = 1 - F_gate per CNOT.

    Simplified model: P(ideal) ≈ F_gate^k for moderate F_gate.
    More precisely: P(ideal) = (1 + (2F-1)^k) / 2
    """
    depths = sorted(results.keys())
    p_vals = [results[d]["p_ideal"] for d in depths]

    print("\n=== Measured Fidelities ===")
    for d, p in zip(depths, p_vals):
        print(f"  k={d:2d}: P(ideal) = {p:.4f}")

    # Fit using model: P(ideal) = (1 + (2F-1)^k) / 2
    # Let r = 2F-1, then 2*P - 1 = r^k
    # ln(2P-1) = k * ln(r)
    valid = [(d, p) for d, p in zip(depths, p_vals) if 2 * p - 1 > 0.01]
    if len(valid) < 2:
        print("Not enough data points for fitting")
        return None

    d_arr = np.array([v[0] for v in valid])
    y_arr = np.array([np.log(2 * v[1] - 1) for v in valid])

    coeffs = np.polyfit(d_arr, y_arr, 1)
    ln_r = coeffs[0]
    r = np.exp(ln_r)
    F_gate = (1 + r) / 2

    # Per-step fidelity for dynamic: F_meas * exp(-Tfb/T2) ≈ 0.83
    # Threshold = |ln(0.83)| / |ln F_gate|
    threshold = 0.1863 / (-np.log(F_gate)) if F_gate < 1 else float('inf')

    print(f"\n=== Fit Results ===")
    print(f"F_gate (per CNOT) = {F_gate:.6f}")
    print(f"r = 2F-1 = {r:.6f}")
    print(f"|ln F_gate| = {-np.log(F_gate):.6f}")
    print(f"Threshold = {threshold:.1f}")

    return {
        "F_gate": float(F_gate),
        "r": float(r),
        "threshold": float(threshold),
        "fit_points": len(valid),
    }


if __name__ == "__main__":
    DEPTHS = [3, 5, 10, 15, 20]
    SHOTS = 4000

    print("=" * 60)
    print(f"Cross-Platform F_gate Extraction")
    print(f"Platform: WK_C180 (machine1)")
    print(f"Qubits: control=102, target=111")
    print(f"Depths: {DEPTHS}")
    print(f"Shots: {SHOTS}")
    print(f"Time: {datetime.now().isoformat()}")
    print("=" * 60)

    results = run_experiment(DEPTHS, SHOTS)

    if results:
        fit = fit_gate_fidelity(results)

        output = {
            "platform": "WK_C180",
            "machine": "machine1",
            "qubits": [102, 111],
            "shots": SHOTS,
            "timestamp": datetime.now().isoformat(),
            "results": {str(k): v for k, v in results.items()},
            "fit": fit,
        }

        out_path = f"experiments/wk180_fidelity_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {out_path}")
