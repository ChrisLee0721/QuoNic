"""
Tuna-17: Deep F_gate extraction for cross-platform validation.
Measures P(0) for odd depths k=51-501, 5 repetitions each.
"""

import numpy as np
import json
import time
from datetime import datetime
from qiskit import QuantumCircuit
from qiskit.compiler import transpile
from qiskit_quantuminspire.qi_provider import QIProvider

OUTPUT_FILE = "tuna17_deep_fgate.json"

# Odd depths to match IBM data range
DEPTHS = [51, 101, 201, 301, 501]
REPS = 5
SHOTS = 1000

def build_cz_h_circuit(k):
    """CZ+H alternating circuit. For odd k, ideal P(0) = 1.0."""
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.h(1)
    for _ in range(k):
        qc.cz(0, 1)
        qc.h(0)
        qc.h(1)
    qc.measure([0, 1], [0, 1])
    return qc

def run_circuit(backend, qc, shots=SHOTS, retries=3):
    """Run circuit with retries."""
    for attempt in range(retries):
        try:
            qc_t = transpile(qc, backend, optimization_level=0)
            job = backend.run(qc_t, shots=shots)
            return job.result().get_counts()
        except Exception as e:
            print(f"    Attempt {attempt+1} failed: {e}")
            if '429' in str(e) or 'rate' in str(e).lower():
                time.sleep(30 * (attempt + 1))
            else:
                time.sleep(10)
    return None

def main():
    results = {
        'timestamp_start': datetime.now().isoformat(),
        'backend': 'Tuna-17',
        'depths': DEPTHS,
        'reps': REPS,
        'shots': SHOTS,
        'note': 'Deep F_gate extraction for cross-platform validation',
        'data': {}
    }

    print(f"=== Tuna-17 Deep F_gate Extraction ===")
    print(f"Depths: {DEPTHS}")
    print(f"Reps: {REPS}, Shots: {SHOTS}")
    print(f"Start: {datetime.now().isoformat()}\n")

    provider = QIProvider()
    backend = provider.get_backend('Tuna-17')
    print(f"Backend: {backend.name}\n")

    for k in DEPTHS:
        results['data'][k] = []
        print(f"Depth k={k}:")

        for rep in range(REPS):
            print(f"  Rep {rep+1}/{REPS}...", end=" ", flush=True)

            qc = build_cz_h_circuit(k)
            counts = run_circuit(backend, qc)

            if counts:
                total = sum(counts.values())
                p0 = counts.get('00', 0) / total
                results['data'][k].append({
                    'p0': p0,
                    'counts': counts,
                    'total': total
                })
                print(f"P(0) = {p0:.4f}")
            else:
                results['data'][k].append({'p0': None, 'error': 'failed'})
                print("FAILED")

            time.sleep(5)

        # Save after each depth
        with open(OUTPUT_FILE, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"  Saved.\n")

    # Summary
    print("\n=== Summary ===")
    print("Depth\tMean P(0)\tStd")
    print("-" * 40)

    for k in DEPTHS:
        p0_vals = [r['p0'] for r in results['data'][k] if r.get('p0') is not None]
        if p0_vals:
            mean_p0 = np.mean(p0_vals)
            std_p0 = np.std(p0_vals)
            print(f"{k}\t{mean_p0:.4f}\t{std_p0:.4f}")

    results['timestamp_end'] = datetime.now().isoformat()
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {OUTPUT_FILE}")

if __name__ == '__main__':
    main()
