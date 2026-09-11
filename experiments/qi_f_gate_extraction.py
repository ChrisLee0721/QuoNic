"""
F_gate extraction on Quantum Inspire Tuna-17.
Submit one circuit at a time, wait for completion before submitting next.
Background-friendly: saves results after each circuit.
"""

import numpy as np
import json
import time
import sys
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


def wait_for_job(job, max_wait=600):
    """Poll job status until done or timeout."""
    for i in range(max_wait // 10):
        time.sleep(10)
        try:
            status = job.status()
            s = str(status)
            if 'DONE' in s or 'COMPLETED' in s:
                return True
            if 'ERROR' in s or 'CANCELLED' in s:
                print(f"  Job failed: {status}", flush=True)
                return False
            if i % 6 == 0:  # Print every 60s
                print(f"  [{i*10}s] {status}", flush=True)
        except Exception as e:
            print(f"  [{i*10}s] Poll error: {e}", flush=True)
    return False


def main():
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_file = f"experiments/qi_f_gate_{ts}.json"

    print(f"=== F_gate Extraction: QI Tuna-17 ===", flush=True)
    print(f"Time: {datetime.now().isoformat()}", flush=True)
    print(f"Output: {out_file}", flush=True)

    provider = QIProvider()
    backend = provider.get_backend('Tuna-17')

    # All odd depths + some even for comparison
    depths = [1, 2, 3, 5, 10, 20, 50, 100]

    results = []
    for k in depths:
        qc = build_circuit(k)
        qc_t = transpile(qc, backend, optimization_level=0, initial_layout=[0, 1])
        ideal = 1.0 if k % 2 == 1 else 0.5

        print(f"\nk={k} (ideal={ideal}): submitting...", flush=True)
        try:
            job = backend.run(qc_t, shots=1000)
            print(f"  Job submitted, waiting...", flush=True)

            if wait_for_job(job, max_wait=600):
                result = job.result(timeout=60)
                counts = result.get_counts()
                p0 = counts.get('0', 0) / sum(counts.values())
                entry = {'depth': k, 'p0': p0, 'ideal': ideal, 'counts': counts}
                results.append(entry)
                print(f"  P(0) = {p0:.4f}", flush=True)
            else:
                print(f"  TIMEOUT", flush=True)
                results.append({'depth': k, 'p0': None, 'error': 'timeout'})

        except Exception as e:
            err = str(e)
            print(f"  ERROR: {err[:100]}", flush=True)
            results.append({'depth': k, 'p0': None, 'error': err[:200]})
            if '429' in err or 'offline' in err.lower():
                print("  Platform issue, waiting 60s...", flush=True)
                time.sleep(60)

        # Save after each circuit
        with open(out_file, 'w') as f:
            json.dump({
                'platform': 'Quantum Inspire Tuna-17',
                'qubit_pair': [0, 1],
                'results': results,
                'timestamp': ts,
            }, f, indent=2)

    # Fit F_gate
    print("\n=== F_gate Extraction ===")
    odd_data = [(r['depth'], r['p0']) for r in results if r.get('p0') is not None and r['depth'] % 2 == 1]
    if len(odd_data) >= 2:
        k_vals = np.array([d for d, p in odd_data])
        p0_vals = np.array([p for d, p in odd_data])
        valid = p0_vals > 0.5
        if valid.sum() >= 2:
            dev = p0_vals[valid] - 0.5
            slope, intercept = np.polyfit(k_vals[valid], np.log(dev), 1)
            F = np.exp(slope / 2)
            print(f"F_per_step = {F:.6f}")
            print(f"Error per step = {(1-F)*100:.4f}%")
        else:
            print("Not enough valid data")
    else:
        print("Not enough odd-k data")

    print(f"\nResults saved: {out_file}")


if __name__ == '__main__':
    main()
