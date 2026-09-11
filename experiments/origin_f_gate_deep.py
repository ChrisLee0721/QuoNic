"""
F_gate deep extraction on Origin Quantum WK_C180.
Odd k values: 51, 101, 201, 301, 401, 501, 801
"""

import os
os.environ['ORIGINGQ_API_KEY'] = 'c5e8b82fef2a8ae457191c9bb88c0a03884cb9be4bf96d0f90ad1691b7406652d1a811063fec7dd8750192907010cad769477863464d33346366764364717168'

import numpy as np
import json
import time
from datetime import datetime
import pyqpanda3 as pq
from pyqpanda3.qcloud.qcloud import QCloudService


def build_circuit(k):
    """H(q0) → [CZ(q0,q1) → H(q0) → H(q1)] × k, measure q0."""
    q0 = pq.core.Qubit(0)
    q1 = pq.core.Qubit(1)
    c0 = pq.core.CBit(0)

    circ = pq.core.QCircuit()
    circ << pq.core.H(q0)
    for _ in range(k):
        circ << pq.core.CZ(q0, q1)
        circ << pq.core.H(q0)
        circ << pq.core.H(q1)

    prog = pq.core.QProg()
    prog << circ
    prog << pq.core.measure(q0, c0)
    return prog


def main():
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_file = f"F:/PyQQQ/experiments/origin_f_gate_deep_{ts}.json"

    print("=== F_gate Deep Extraction: Origin Quantum WK_C180 ===")
    print(f"Time: {datetime.now().isoformat()}")

    service = QCloudService(os.environ['ORIGINGQ_API_KEY'])
    backend = service.backend('WK_C180')

    # Deep odd k values
    depths = [51, 101, 201, 301, 401, 501, 801]
    shots = 4000

    results = []
    for k in depths:
        prog = build_circuit(k)
        print(f"k={k}: submitting...", end=" ", flush=True)

        try:
            job = backend.run(prog, shots)
            result = job.result()
            probs = result.get_probs()

            # Parse probs
            p0 = 0
            for key, prob in probs.items():
                bs = str(key)
                if bs.startswith('0x'):
                    bs = format(int(bs, 16), '02b')
                elif bs.startswith('0b'):
                    bs = bs[2:]
                if bs[0] == '0':
                    p0 += prob

            results.append({'depth': k, 'p0': p0, 'ideal': 1.0})
            print(f"P(0)={p0:.4f}", flush=True)

        except Exception as e:
            print(f"ERROR: {e}", flush=True)
            results.append({'depth': k, 'p0': None, 'error': str(e)[:200]})

        time.sleep(3)

    # Fit F_gate
    print("\n=== F_gate Fit ===")
    valid = [(r['depth'], r['p0']) for r in results if r.get('p0') is not None]
    if len(valid) >= 2:
        k_arr = np.array([d for d, _ in valid])
        p_arr = np.array([p for _, p in valid])

        # Fit: P(0) = 0.5 + 0.5 * F^(2k)
        # So P(0) - 0.5 = 0.5 * F^(2k)
        # ln(P(0) - 0.5) = ln(0.5) + 2k * ln(F)
        dev = p_arr - 0.5
        if np.all(dev > 0):
            slope, intercept = np.polyfit(k_arr, np.log(dev), 1)
            F = np.exp(slope / 2)
            print(f"F_per_step = {F:.6f}")
            print(f"Error per step = {(1-F)*100:.4f}%")

            # R²
            predicted = intercept + slope * k_arr
            r_squared = 1 - np.sum((np.log(dev) - predicted)**2) / np.sum((np.log(dev) - np.mean(np.log(dev)))**2)
            print(f"R² = {r_squared:.4f}")
        else:
            print("Cannot fit: some P(0) <= 0.5")
            F = None
    else:
        print("Not enough data")
        F = None

    # Save
    output = {
        'platform': 'Origin Quantum WK_C180',
        'results': results,
        'F_per_step': F,
        'timestamp': ts,
    }
    with open(out_file, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_file}")


if __name__ == '__main__':
    main()
