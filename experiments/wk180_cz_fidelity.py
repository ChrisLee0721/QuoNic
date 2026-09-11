"""
WK_C180 CZ gate fidelity extraction.
Uses native CZ gate directly. CNOT decomposition is buggy in transpiler.

Method: Prepare |+1⟩ (H on q0, X on q1), apply CZ k times with RY(q0, small_angle) to break periodicity.
Ideal: each CZ entangles, fidelity decays with k.
"""

import os, sys, json, time, numpy as np
from datetime import datetime

sys.path.insert(0, r"C:/Users/26427/AppData/Local/Programs/Python/Python312")

import pyqpanda3 as pq
from pyqpanda3.qcloud.qcloud import QCloudService
from pyqpanda3.transpilation import Transpiler


def build_cz_chain(k: int):
    """Build circuit: H(q0) → X(q1) → [CZ → RY(q0, π/3)] × k → Measure.

    RY breaks CZ self-inverse periodicity so fidelity decays monotonically.
    """
    prog = pq.core.QProg()
    prog << pq.core.H(pq.core.Qubit(0))
    prog << pq.core.X(pq.core.Qubit(1))

    for _ in range(k):
        prog << pq.core.CZ(pq.core.Qubit(0), pq.core.Qubit(1))
        prog << pq.core.RY(pq.core.Qubit(0), np.pi / 3)

    prog << pq.core.measure(pq.core.Qubit(0), 0)
    prog << pq.core.measure(pq.core.Qubit(1), 1)
    return prog


def run_job(prog, shots, backend, chip_backend, transpiler, control, target):
    prog_t = transpiler.transpile(
        prog, chip_backend,
        init_mapping={0: control, 1: target},
        optimization_level=0,
    )
    t0 = time.time()
    job = backend.run(prog_t, shots)
    raw = job.result()
    elapsed = time.time() - t0

    if raw.error_message():
        print(f"  ERROR: {raw.error_message()}")
        return None, elapsed

    probs = raw.get_probs()
    counts = {}
    for key, prob in probs.items():
        bs = str(key)
        if bs.startswith("0x"):
            bs = format(int(bs, 16), "02b")
        elif bs.startswith("0b"):
            bs = bs[2:].zfill(2)
        counts[bs] = max(1, round(prob * shots))
    return counts, elapsed


def noiseless_simulation(k: int):
    """Simulate 2-qubit circuit using numpy statevector.

    Circuit: H(q0) → X(q1) → [CZ → RY(q0, π/3)] × k → Measure
    State vector: |q0 q1⟩, index = q0*2 + q1
    """
    # H gate: (1/√2)[[1,1],[1,-1]]
    H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    # X gate
    X = np.array([[0, 1], [1, 0]])
    # CZ = diag(1, 1, 1, -1)
    CZ = np.diag([1, 1, 1, -1])
    # RY(θ) = [[cos(θ/2), -sin(θ/2)], [sin(θ/2), cos(θ/2)]]
    theta = np.pi / 3
    RY = np.array([[np.cos(theta/2), -np.sin(theta/2)],
                    [np.sin(theta/2), np.cos(theta/2)]])

    # Start: |00⟩
    state = np.array([1, 0, 0, 0], dtype=complex)

    # H on q0: I⊗H (q0 is high bit in |q0 q1⟩)
    gate = np.kron(H, np.eye(2))
    state = gate @ state

    # X on q1
    gate = np.kron(np.eye(2), X)
    state = gate @ state

    for _ in range(k):
        # CZ
        state = CZ @ state
        # RY on q0
        gate = np.kron(RY, np.eye(2))
        state = gate @ state

    probs = np.abs(state) ** 2
    # Map: index 0→|00⟩, 1→|01⟩, 2→|10⟩, 3→|11⟩
    return {
        "00": float(probs[0]),
        "01": float(probs[1]),
        "10": float(probs[2]),
        "11": float(probs[3]),
    }


def main():
    DEPTHS = [1, 2, 3, 5, 7, 10, 15, 20]
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
    print("CZ Gate Fidelity Extraction - WK_C180")
    print(f"Qubits: {CONTROL}(ctrl), {TARGET}(tgt)")
    print(f"Depths: {DEPTHS}, Shots: {SHOTS}")
    print(f"Time: {datetime.now().isoformat()}")
    print("=" * 60)

    # First: noiseless simulation for ideal probabilities
    print("\n=== Noiseless Simulation ===")
    ideal_probs = {}
    for k in DEPTHS:
        probs = noiseless_simulation(k)
        ideal_probs[k] = probs
        top = sorted(probs.items(), key=lambda x: -x[1])[:4]
        print(f"  k={k:2d}: {dict(top)}")

    # Hardware runs
    print("\n=== Hardware Results ===")
    results = {}
    for k in DEPTHS:
        print(f"\n--- k={k} ---")
        prog = build_cz_chain(k)
        counts, elapsed = run_job(prog, SHOTS, backend, chip_backend, transpiler, CONTROL, TARGET)
        if counts is None:
            continue

        # Calculate TVD from ideal
        # Hardware bit order is q111 q102 (high-low), simulation is q0 q1
        # Flip hardware bits to match simulation convention
        ideal = ideal_probs[k]
        flipped_counts = {}
        for bs, c in counts.items():
            flipped_counts[bs[::-1]] = c  # reverse bitstring

        tvd = 0
        all_keys = set(flipped_counts.keys()) | set(ideal.keys())
        for bs in all_keys:
            p_hw = flipped_counts.get(bs, 0) / SHOTS
            p_ideal = ideal.get(bs, 0)
            tvd += abs(p_hw - p_ideal)
        tvd /= 2

        # Show both orderings
        top_hw = sorted(counts.items(), key=lambda x: -x[1])[:4]
        top_flip = sorted(flipped_counts.items(), key=lambda x: -x[1])[:4]

        print(f"  Raw (q111 q102): {dict(top_hw)}")
        print(f"  Flipped (q0 q1): {dict(top_flip)}")
        print(f"  TVD from ideal: {tvd:.4f}")
        print(f"  Time: {elapsed:.1f}s")

        results[k] = {
            "depth": k,
            "counts": counts,
            "ideal_probs": ideal,
            "tvd": tvd,
            "shots": SHOTS,
            "elapsed": elapsed,
        }

    # Save
    output = {
        "platform": "WK_C180",
        "qubits": [CONTROL, TARGET],
        "gate": "CZ",
        "shots": SHOTS,
        "timestamp": datetime.now().isoformat(),
        "results": {str(k): v for k, v in results.items()},
    }
    out_path = f"experiments/wk180_cz_fidelity_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
