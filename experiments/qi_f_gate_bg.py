"""
F_gate extraction on QI Tuna-17 — single-shot background runner.
Submit all circuits in one batch, wait patiently for results.
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


def main():
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_file = f"experiments/qi_f_gate_{ts}.json"

    print(f"=== F_gate Extraction: QI Tuna-17 ===", flush=True)
    print(f"Time: {datetime.now().isoformat()}", flush=True)

    provider = QIProvider()
    backend = provider.get_backend('Tuna-17')

    depths = [1, 2, 3, 5, 10, 20, 50, 100]
    jobs = []

    # Submit all, retry on 429 with long waits
    for k in depths:
        qc = build_circuit(k)
        qc_t = transpile(qc, backend, optimization_level=0, initial_layout=[0, 1])
        while True:
            try:
                job = backend.run(qc_t, shots=1000)
                print(f"k={k}: submitted {job.job_id()}", flush=True)
                jobs.append((k, job))
                break
            except Exception as e:
                if '429' in str(e):
                    print(f"k={k}: queue full, waiting 120s...", flush=True)
                    time.sleep(120)
                else:
                    print(f"k={k}: error {e}", flush=True)
                    jobs.append((k, None))
                    break

    # Wait for all results
    print(f"\nAll submitted. Waiting for {len(jobs)} results...", flush=True)
    results = []
    for k, job in jobs:
        if job is None:
            results.append({'depth': k, 'p0': None, 'error': 'submit_failed'})
            continue
        try:
            result = job.result(timeout=3600)  # 1 hour per job
            counts = result.get_counts()
            p0 = counts.get('0', 0) / sum(counts.values())
            ideal = 1.0 if k % 2 == 1 else 0.5
            results.append({'depth': k, 'p0': p0, 'ideal': ideal, 'counts': counts})
            print(f"k={k}: P(0)={p0:.4f} (ideal={ideal})", flush=True)
        except Exception as e:
            print(f"k={k}: result error {e}", flush=True)
            results.append({'depth': k, 'p0': None, 'error': str(e)[:200]})

        # Save after each
        with open(out_file, 'w') as f:
            json.dump({'platform': 'QI Tuna-17', 'qubit_pair': [0, 1], 'results': results}, f, indent=2)

    # Fit
    print("\n=== F_gate Extraction ===")
    odd = [(r['depth'], r['p0']) for r in results if r.get('p0') is not None and r['depth'] % 2 == 1]
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

    print(f"\nSaved: {out_file}")


if __name__ == '__main__':
    main()
