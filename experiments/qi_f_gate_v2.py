"""
F_gate extraction on Quantum Inspire Tuna-17 with retries.
"""

import numpy as np
import json
import time
from datetime import datetime
from qiskit import QuantumCircuit
from qiskit.compiler import transpile
from qiskit_quantuminspire.qi_provider import QIProvider


def build_circuit(k):
    qc = QuantumCircuit(2, 1)
    qc.h(0)
    for _ in range(k):
        qc.cz(0, 1)
        qc.h(0)
        qc.h(1)
    qc.measure(0, 0)
    return qc


def run_with_retry(backend, qc_t, shots=1000, max_retries=5):
    for attempt in range(max_retries):
        try:
            job = backend.run(qc_t, shots=shots)
            result = job.result()
            return result.get_counts()
        except Exception as e:
            err = str(e)
            wait = 30 * (attempt + 1)
            print(f"  attempt {attempt+1} failed: {err[:80]}... waiting {wait}s", flush=True)
            time.sleep(wait)
    return None


def main():
    print(f"=== QI Tuna-17 F_gate Extraction ===")
    print(f"Time: {datetime.now().isoformat()}", flush=True)

    provider = QIProvider()
    backend = provider.get_backend('Tuna-17')
    print(f"Backend: {backend.name} ({backend.num_qubits} qubits)", flush=True)

    depths = [1, 2, 3, 5, 10, 20, 50, 100]
    results = []

    for k in depths:
        qc = build_circuit(k)
        qc_t = transpile(qc, backend, optimization_level=0, initial_layout=[0, 1])
        print(f"k={k}: submitting...", flush=True)
        counts = run_with_retry(backend, qc_t, shots=1000)
        if counts:
            p0 = counts.get('0', 0) / sum(counts.values())
            ideal = 1.0 if k % 2 == 1 else 0.5
            results.append({'depth': k, 'p0': p0, 'ideal': ideal, 'counts': counts})
            print(f"  P(0)={p0:.4f} (ideal={ideal})", flush=True)
        else:
            print(f"  FAILED after retries", flush=True)
            results.append({'depth': k, 'p0': None})
        time.sleep(15)  # rate limit spacing

    # Fit
    print("\n=== F_gate Extraction ===")
    odd_data = [(r['depth'], r['p0']) for r in results if r['p0'] is not None and r['depth'] % 2 == 1]
    if len(odd_data) >= 2:
        k_vals = np.array([d for d, p in odd_data])
        p0_vals = np.array([p for d, p in odd_data])
        valid = p0_vals > 0.5
        if valid.sum() >= 2:
            dev = p0_vals[valid] - 0.5
            slope, _ = np.polyfit(k_vals[valid], np.log(dev), 1)
            F = np.exp(slope / 2)
            print(f"F_per_step = {F:.6f}")
            print(f"Error per step = {(1-F)*100:.4f}%")

    fname = f"experiments/qi_f_gate_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(fname, 'w') as f:
        json.dump({'platform': 'QI Tuna-17', 'qubit_pair': [0,1], 'results': results}, f, indent=2)
    print(f"\nSaved: {fname}")


if __name__ == '__main__':
    main()
