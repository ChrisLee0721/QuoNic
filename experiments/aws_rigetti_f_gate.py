"""
F_gate extraction on AWS Braket - Rigetti Cepheus-1-108Q.
Circuit: H(q0) → [CZ(q0,q1) → H(q0) → H(q1)] × k → Measure q0
Odd k → ideal P(0)=1.0, Even k → ideal P(0)=0.5
"""

import json
import time
from datetime import datetime
import numpy as np
from braket.aws import AwsDevice
from braket.circuits import Circuit

DEVICE_ARN = "arn:aws:braket:us-west-1::device/qpu/rigetti/Cepheus-1-108Q"
SHOTS = 1000
DEPTHS = [1, 3, 5, 10, 20, 50, 100]
Q0, Q1 = 0, 1  # Will be updated after checking connectivity


def build_circuit(k, q0, q1):
    circ = Circuit()
    circ.h(q0)
    for _ in range(k):
        circ.cz(q0, q1)
        circ.h(q0)
        circ.h(q1)
    circ.measure(q0)
    return circ


def main():
    print(f"=== AWS Braket Rigetti Cepheus F_gate Extraction ===")
    print(f"Time: {datetime.now().isoformat()}")

    device = AwsDevice(DEVICE_ARN)
    print(f"Device: {device.name}, Status: {device.status}")
    print(f"Qubits: {device.properties.paradigm.qubitCount}")

    # Check supported gates
    for action_type, action_spec in device.properties.action.items():
        if hasattr(action_spec, 'supportedOperations'):
            print(f"Gates: {action_spec.supportedOperations}")
            break

    # Find connected qubit pair
    connectivity = device.properties.paradigm.connectivity
    if hasattr(connectivity, 'connectivityGraph'):
        graph = connectivity.connectivityGraph
        q0 = int(list(graph.keys())[0])
        neighbors = graph[q0]
        q1 = int(list(neighbors.keys())[0]) if isinstance(neighbors, dict) else int(neighbors[0])
        print(f"Using qubit pair: ({q0}, {q1})")
    else:
        q0, q1 = Q0, Q1
        print(f"Using default qubit pair: ({q0}, {q1})")

    results = []
    for k in DEPTHS:
        circ = build_circuit(k, q0, q1)
        print(f"k={k}: submitting...", flush=True)
        try:
            task = device.run(circ, shots=SHOTS)
            print(f"  Task: {task.id}")
            result = task.result()
            counts = result.measurement_counts
            total = sum(counts.values())
            p0 = counts.get('0', 0) / total if total > 0 else 0
            ideal = 1.0 if k % 2 == 1 else 0.5
            results.append({'depth': k, 'p0': p0, 'ideal': ideal, 'counts': dict(counts), 'task': task.id})
            print(f"  P(0)={p0:.4f} (ideal={ideal})")
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({'depth': k, 'p0': None, 'error': str(e)})

    # Fit
    print("\n=== F_gate Extraction ===")
    odd = [(r['depth'], r['p0']) for r in results if r['p0'] is not None and r['depth'] % 2 == 1]
    if len(odd) >= 2:
        kv = np.array([d for d, p in odd])
        pv = np.array([p for d, p in odd])
        valid = pv > 0.5
        if valid.sum() >= 2:
            slope, _ = np.polyfit(kv[valid], np.log(pv[valid] - 0.5), 1)
            F = np.exp(slope / 2)
            print(f"F_per_step = {F:.6f}")
            print(f"Error per step = {(1-F)*100:.4f}%")

    fname = f"experiments/aws_rigetti_f_gate_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(fname, 'w') as f:
        json.dump({'platform': 'Rigetti Cepheus-1-108Q', 'qubits': [q0, q1], 'shots': SHOTS, 'results': results}, f, indent=2, default=str)
    print(f"\nSaved: {fname}")


if __name__ == '__main__':
    main()
