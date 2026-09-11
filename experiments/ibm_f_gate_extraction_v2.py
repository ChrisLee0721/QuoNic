"""
Experiment 3: Extract F_gate from IBM ibm_marrakesh.
Batch mode - all circuits in one job to minimize queue time.
"""

import json
import os
import numpy as np
from datetime import datetime
from qiskit import QuantumCircuit
from qiskit.compiler import transpile
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler


def build_circuit(k):
    """Build k-step CZ toggle circuit on 2 logical qubits (0,1)."""
    qc = QuantumCircuit(2, 1)
    qc.h(0)
    for _ in range(k):
        qc.cz(0, 1)
        qc.h(0)
        qc.h(1)
    qc.measure(0, 0)
    return qc


def main():
    print("=== Experiment 3: IBM F_gate Extraction (Batch) ===")
    print(f"Time: {datetime.now().isoformat()}", flush=True)

    service = QiskitRuntimeService(
        channel='ibm_quantum_platform',
        token='Pga6CcTge-z9ALVI-YV4XZa65Qs5HJvG8k_DVoGCITNN'
    )
    backend = service.backend('ibm_marrakesh')
    print(f"Backend: {backend.name}", flush=True)

    pairs = [(0, 1), (1, 2), (2, 3), (3, 4), (14, 15)]
    depths = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20]

    # Build all circuits
    print("Building and transpiling circuits...", flush=True)
    all_circuits = []
    circuit_info = []
    for q1, q2 in pairs:
        for k in depths:
            qc = build_circuit(k)
            qc_t = transpile(qc, backend, optimization_level=0, initial_layout=[q1, q2])
            all_circuits.append(qc_t)
            circuit_info.append({'qubits': (q1, q2), 'depth': k})

    print(f"Total circuits: {len(all_circuits)}", flush=True)

    # Submit all as one batch job
    print("Submitting batch job...", flush=True)
    sampler = Sampler(backend)
    sampler.options.default_shots = 4000
    job = sampler.run(all_circuits)
    print(f"Job ID: {job.job_id()}, waiting for results...", flush=True)

    result = job.result()
    print("Got results!", flush=True)

    # Parse results
    results = []
    for i, info in enumerate(circuit_info):
        counts = result[i].data.c.get_counts()
        p0 = counts.get('0', 0) / sum(counts.values())
        results.append({
            'qubits': info['qubits'],
            'depth': info['depth'],
            'p0': p0,
            'counts': counts,
        })

    # Print table
    print("\n=== Results ===")
    print(f"{'Qubits':>10} {'k':>4} {'p0':>8} {'ideal':>8}")
    for r in results:
        ideal = 1.0 if r['depth'] % 2 == 0 else 0.0
        print(f"{str(r['qubits']):>10} {r['depth']:>4} {r['p0']:>8.4f} {ideal:>8.1f}")

    # Fit F_gate
    print("\n=== F_gate Extraction ===")
    for q1, q2 in pairs:
        pair_data = [r for r in results if r['qubits'] == (q1, q2)]
        even_data = [(r['depth'], r['p0']) for r in pair_data if r['depth'] % 2 == 0]
        if len(even_data) >= 2:
            depths_even = np.array([d for d, p in even_data])
            p0_even = np.array([p for d, p in even_data])
            valid = p0_even > 0.5
            if valid.sum() >= 2:
                log_decay = np.log(p0_even[valid] - 0.5)
                k_valid = depths_even[valid]
                slope, intercept = np.polyfit(k_valid, log_decay, 1)
                a_fit = np.exp(slope)
                F_cz_fit = np.sqrt(a_fit)
                print(f"  ({q1},{q2}): F_cz(fitted) = {F_cz_fit:.4f}")

    # Save
    outpath = os.path.join(os.path.dirname(__file__),
                           f"ibm_f_gate_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(outpath, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved: {outpath}")


if __name__ == '__main__':
    main()
