"""
WK_C180 CZ gate fidelity - Hadamard basis measurement.
Circuit: X(q0) → X(q1) → CZ^k → H(q0) → H(q1) → Measure
Ideal: CZ^k |11⟩ = (-1)^k |11⟩, after HH → P(|00⟩) = 1 (even k) or 0 (odd k)
With noise: P(|00⟩) decays exponentially with k.
"""

import os, sys, json, time, numpy as np
from datetime import datetime

sys.path.insert(0, r"C:/Users/26427/AppData/Local/Programs/Python/Python312")

import pyqpanda3 as pq
from pyqpanda3.qcloud.qcloud import QCloudService
from pyqpanda3.transpilation import Transpiler


def build_cz_hadamard_test(k: int):
    """X(q0) → X(q1) → CZ × k → H(q0) → H(q1) → Measure."""
    prog = pq.core.QProg()
    prog << pq.core.X(pq.core.Qubit(0))
    prog << pq.core.X(pq.core.Qubit(1))
    for _ in range(k):
        prog << pq.core.CZ(pq.core.Qubit(0), pq.core.Qubit(1))
    prog << pq.core.H(pq.core.Qubit(0))
    prog << pq.core.H(pq.core.Qubit(1))
    prog << pq.core.measure(pq.core.Qubit(0), 0)
    prog << pq.core.measure(pq.core.Qubit(1), 1)
    return prog


def ideal_p00(k: int) -> float:
    """Ideal P(|00>) after CZ^k on |11> then HH."""
    # CZ^k |11> = (-1)^k |11>
    # H⊗H |11> = (|00> - |01> - |10> + |11>)/2
    # With phase (-1)^k: (-1)^k (|00> - |01> - |10> + |11>)/2
    # P(|00>) = 1/4 for any k... that's not useful.
    # Actually: H⊗H (-1)^k |11> = (-1)^k H⊗H |11>
    # H|1> = (|0>-|1>)/√2, so H⊗H|11> = (|00>-|01>-|10>+|11>)/2
    # P(|00>) = |(-1)^k/2|² = 1/4 regardless of k
    # Hmm, that means HH doesn't help distinguish phase.

    # Better approach: measure in X basis only.
    # CZ^k|11> = (-1)^k|11>
    # H on q0 only: H⊗I (-1)^k|11> = (-1)^k (|01>+|11>)/√2 ... no
    # Actually H|1> = (|0>-|1>)/√2
    # (H⊗I)|11> = (|01>-|11>)/√2
    # With phase: (-1)^k (|01>-|11>)/√2
    # P(|01>) = P(|11>) = 1/2 regardless of k. Not useful either.

    # The correct way: CZ error detection needs a different circuit.
    # Use: |+0> → CZ → H(q0) → measure q0
    # CZ|+0> = |+0> (no effect since q1=|0>)
    # Not useful.

    # Actually the simplest: prepare |+1>, apply CZ k times, measure q0 in X basis.
    # |+1> = (|01>+|11>)/√2
    # CZ|+1> = (|01>-|11>)/√2 = |->|1> (q0 rotated to |->)
    # CZ²|+1> = CZ(|01>-|11>)/√2 = (|01>+|11>)/√2 = |+1> (back to |+>)
    # So CZ^k |+1>: even k → |+1>, odd k → |->1>
    # H(q0) then measure: even k → P(0)=1, odd k → P(0)=0 (ideal)
    # With gate error F: P(0) = (1+(2F-1)^k)/2

    return (1 + (1)**k) / 2  # placeholder, will use correct circuit


def build_fidelity_test(k: int):
    """H(q0) → X(q1) → CZ^k → H(q0) → Measure q0.
    Ideal: even k → P(0)=1, odd k → P(0)=0.
    """
    prog = pq.core.QProg()
    prog << pq.core.H(pq.core.Qubit(0))
    prog << pq.core.X(pq.core.Qubit(1))
    for _ in range(k):
        prog << pq.core.CZ(pq.core.Qubit(0), pq.core.Qubit(1))
    prog << pq.core.H(pq.core.Qubit(0))
    prog << pq.core.measure(pq.core.Qubit(0), 0)
    return prog


def main():
    DEPTHS = [1, 2, 3, 5, 7, 10, 15, 20, 30]
    SHOTS = 4000
    CONTROL, TARGET = 102, 111

    api_key = os.environ.get("ORIGINGQ_API_KEY", "")
    if not api_key:
        print("ERROR: ORIGINGQ_API_KEY not set")
        return

    service = QCloudService(api_key)
    backend = service.backend("WK_C180")
    chip_backend = backend.chip_info().get_chip_backend()
    transpiler = Transpiler()

    print("=" * 60)
    print("CZ Gate Fidelity - Hadamard Basis Test")
    print(f"Circuit: H(q0) → X(q1) → CZ^k → H(q0) → M(q0)")
    print(f"Ideal: even k → P(0)=1, odd k → P(0)=0")
    print(f"Qubits: {CONTROL}, {TARGET}")
    print(f"Depths: {DEPTHS}, Shots: {SHOTS}")
    print("=" * 60)

    results = {}
    for k in DEPTHS:
        prog = build_fidelity_test(k)
        prog_t = transpiler.transpile(
            prog, chip_backend,
            init_mapping={0: CONTROL, 1: TARGET},
            optimization_level=0,
        )

        t0 = time.time()
        job = backend.run(prog_t, SHOTS)
        raw = job.result()
        elapsed = time.time() - t0

        if raw.error_message():
            print(f"k={k}: ERROR {raw.error_message()}")
            continue

        probs = raw.get_probs()
        counts = {}
        for key, prob in probs.items():
            bs = str(key)
            if bs.startswith("0x"):
                bs = format(int(bs, 16), "02b")
            elif bs.startswith("0b"):
                bs = bs[2:].zfill(2)
            counts[bs] = max(1, round(prob * SHOTS))

        # P(q0=0) - single qubit measurement, key is '0' or '1'
        p_q0_0 = counts.get("0", 0) / SHOTS
        # Ideal: even k → 1, odd k → 0
        ideal = 1.0 if k % 2 == 0 else 0.0

        print(f"k={k:2d}: P(q0=0)={p_q0_0:.4f} (ideal={ideal}) | {counts}")

        results[k] = {
            "p_q0_0": p_q0_0,
            "ideal": ideal,
            "counts": counts,
            "shots": SHOTS,
            "elapsed": elapsed,
        }

    # Fit: for even k, P(0) = (1 + r^k)/2, r = 2F-1
    # For odd k, P(0) = (1 - r^k)/2
    # Combined: P(0) = (1 + (-1)^k * r^k) / 2
    # → (-1)^k * (2*P(0)-1) = r^k
    # → ln|2P(0)-1| = k * ln(r)  (use even k for simplicity)
    print("\n=== Fit ===")
    even_pts = [(k, results[k]["p_q0_0"]) for k in results if k % 2 == 0 and results[k]["p_q0_0"] > 0.5]
    odd_pts = [(k, results[k]["p_q0_0"]) for k in results if k % 2 == 1 and results[k]["p_q0_0"] < 0.5]

    if len(even_pts) >= 2:
        d_arr = np.array([p[0] for p in even_pts])
        y_arr = np.array([np.log(2 * p[1] - 1) for p in even_pts])
        coeffs = np.polyfit(d_arr, y_arr, 1)
        ln_r = coeffs[0]
        r = np.exp(ln_r)
        F_cz = (1 + r) / 2
        threshold = 0.1863 / (-np.log(F_cz)) if F_cz < 1 else float('inf')
        print(f"F_CZ = {F_cz:.6f}")
        print(f"|ln F_CZ| = {-np.log(F_cz):.6f}")
        print(f"Threshold (vs measurement tax) = {threshold:.1f}")
    else:
        print("Not enough even-k data points for fitting")
        F_cz = None

    output = {
        "platform": "WK_C180",
        "qubits": [CONTROL, TARGET],
        "gate": "CZ",
        "shots": SHOTS,
        "timestamp": datetime.now().isoformat(),
        "F_CZ": F_cz,
        "results": {str(k): v for k, v in results.items()},
    }
    out_path = f"experiments/wk180_cz_fidelity_v2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
