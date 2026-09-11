"""
F_gate extraction on Origin Quantum WK_C180.
CZ + H alternating circuit, odd k values.
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
    out_file = f"experiments/origin_f_gate_{ts}.json"

    print("=== F_gate Extraction: Origin Quantum WK_C180 ===")
    print(f"Time: {datetime.now().isoformat()}")

    service = QCloudService(os.environ['ORIGINGQ_API_KEY'])
    backend = service.backend('WK_C180')
    print(f"Backend: WK_C180")

    # Odd k values for F_gate fitting
    depths = [1, 2, 3, 5, 10, 20, 50, 100]
    shots = 1000

    results = []
    for k in depths:
        prog = build_circuit(k)
        ideal = 1.0 if k % 2 == 1 else 0.5
        print(f"k={k}: submitting...", end=" ", flush=True)

        try:
            job = backend.run(prog, shots)
            result = job.result()
            probs = result.get_probs()

            # Parse probs - key might be hex or binary
            p0 = 0
            for key, prob in probs.items():
                bs = str(key)
                if bs.startswith('0x'):
                    bs = format(int(bs, 16), '02b')
                elif bs.startswith('0b'):
                    bs = bs[2:]
                # First bit is q0
                if bs[0] == '0':
                    p0 += prob

            results.append({'depth': k, 'p0': p0, 'ideal': ideal})
            print(f"P(0)={p0:.4f} (ideal={ideal})", flush=True)

        except Exception as e:
            print(f"ERROR: {e}", flush=True)
            results.append({'depth': k, 'p0': None, 'error': str(e)[:200]})

        time.sleep(2)

    # Fit F_gate from odd k data
    print("\n=== F_gate Extraction ===")
    odd = [(r['depth'], r['p0']) for r in results if r.get('p0') is not None and r['depth'] % 2 == 1]
    F = None
    if len(odd) >= 2:
        k_arr = np.array([d for d, _ in odd])
        p_arr = np.array([p for _, p in odd])
        valid = p_arr > 0.5
        if valid.sum() >= 2:
            dev = p_arr[valid] - 0.5
            slope, _ = np.polyfit(k_arr[valid], np.log(dev), 1)
            F = np.exp(slope / 2)
            print(f"F_per_step = {F:.6f}")
            print(f"Error per step = {(1-F)*100:.4f}%")

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
