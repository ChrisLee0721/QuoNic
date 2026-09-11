"""
F_gate extraction on AWS Braket: IQM Garnet and Emerald.
"""

import os
os.environ['AWS_ACCESS_KEY_ID'] = os.environ.get('AWS_ACCESS_KEY_ID', 'YOUR_KEY')
os.environ['AWS_SECRET_ACCESS_KEY'] = os.environ.get('AWS_SECRET_ACCESS_KEY', 'YOUR_SECRET')

import numpy as np
import json
import time
from datetime import datetime
from braket.circuits import Circuit
from braket.aws import AwsDevice


def build_circuit(k):
    """H(q0) → [CZ(q0,q1) → H(q0) → H(q1)] × k, measure q0."""
    qc = Circuit()
    qc.h(0)
    for _ in range(k):
        qc.cz(0, 1)
        qc.h(0)
        qc.h(1)
    qc.measure(0)
    return qc


def run_device(device_arn, device_name, depths):
    device = AwsDevice(device_arn)
    print(f"\n=== {device_name} ===")
    print(f"Status: {device.status}")
    print(f"Queue depth: {device.queue_depth()}")

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_file = f"experiments/braket_{device_name}_{ts}.json"

    results = []
    for k in depths:
        qc = build_circuit(k)
        ideal = 1.0 if k % 2 == 1 else 0.5
        print(f"k={k}: submitting...", end=" ", flush=True)
        try:
            task = device.run(qc, shots=1000)
            result = task.result()
            counts = result.measurement_counts
            # counts keys are like '0' or '00' depending on qubit count
            # We measured only qubit 0, so key is '0'
            p0 = sum(v for k2, v in counts.items() if k2[-1] == '0') / sum(counts.values())
            results.append({'depth': k, 'p0': p0, 'ideal': ideal, 'counts': dict(counts)})
            print(f"P(0)={p0:.4f} (ideal={ideal})")
        except Exception as e:
            print(f"ERROR: {e}")
            results.append({'depth': k, 'p0': None, 'error': str(e)[:200]})
        time.sleep(2)

    # Fit F_gate from odd k data
    print(f"\n--- {device_name} F_gate ---")
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
    with open(out_file, 'w') as f:
        json.dump({
            'platform': device_name,
            'device_arn': device_arn,
            'results': results,
            'F_per_step': F,
            'timestamp': ts,
        }, f, indent=2)
    print(f"Saved: {out_file}")
    return F


def main():
    print(f"=== F_gate Extraction: AWS Braket ===")
    print(f"Time: {datetime.now().isoformat()}")

    depths = [1, 2, 3, 5, 10, 20, 50, 100]

    # IQM Garnet (32 qubits)
    garnet_arn = 'arn:aws:braket:eu-north-1::device/qpu/iqm/Garnet'
    F_garnet = run_device(garnet_arn, 'IQM-Garnet', depths)

    # IQM Emerald (24 qubits)
    emerald_arn = 'arn:aws:braket:eu-north-1::device/qpu/iqm/Emerald'
    F_emerald = run_device(emerald_arn, 'IQM-Emerald', depths)

    print("\n=== Summary ===")
    print(f"IQM Garnet:   F={F_garnet:.6f} ({(1-F_garnet)*100:.4f}%/step)")
    print(f"IQM Emerald:  F={F_emerald:.6f} ({(1-F_emerald)*100:.4f}%/step)")


if __name__ == '__main__':
    main()
