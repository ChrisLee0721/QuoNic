"""
Experiment 3c: Deep F_gate extraction (odd k) from IBM ibm_marrakesh.

Odd k → ideal P(0)=1.0, decay = F_gate signal.
Qubit pair (3,4) with T2=138us.
"""

import json
import os
import numpy as np
from datetime import datetime
from qiskit import QuantumCircuit
from qiskit.compiler import transpile
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler


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
    print("=== Experiment 3c: Deep F_gate (odd k) ===")
    print(f"Time: {datetime.now().isoformat()}", flush=True)

    service = QiskitRuntimeService(
        channel='ibm_quantum_platform',
        token='Pga6CcTge-z9ALVI-YV4XZa65Qs5HJvG8k_DVoGCITNN'
    )
    backend = service.backend('ibm_marrakesh')
    print(f"Backend: {backend.name}", flush=True)

    q1, q2 = 3, 4
    depths = [51, 101, 201, 501, 801, 1001]

    print(f"Qubit pair: ({q1}, {q2})", flush=True)
    print(f"Depths (odd): {depths}", flush=True)

    print("Transpiling...", flush=True)
    all_circuits = []
    for k in depths:
        qc = build_circuit(k)
        qc_t = transpile(qc, backend, optimization_level=0, initial_layout=[q1, q2])
        all_circuits.append(qc_t)

    print("Submitting...", flush=True)
    sampler = Sampler(backend)
    sampler.options.default_shots = 4000
    job = sampler.run(all_circuits)
    print(f"Job ID: {job.job_id()}, waiting...", flush=True)

    result = job.result()
    print("Done!", flush=True)

    results = []
    print(f"\n{'k':>6} {'P(0)':>8} {'ideal':>8} {'decay':>8}")
    for i, k in enumerate(depths):
        counts = result[i].data.c.get_counts()
        p0 = counts.get('0', 0) / sum(counts.values())
        decay = 1.0 - p0
        results.append({
            'qubits': (q1, q2),
            'depth': k,
            'p0': p0,
            'ideal': 1.0,
            'decay': decay,
            'counts': counts,
        })
        print(f"{k:>6} {p0:>8.4f} {1.0:>8.1f} {decay:>8.4f}")

    # Fit F_gate
    print("\n=== F_gate Extraction ===")
    k_vals = np.array([r['depth'] for r in results])
    p0_vals = np.array([r['p0'] for r in results])

    # P(0) = 0.5 + 0.5 * F^(2k)
    # decay = 1 - P(0) = 0.5 - 0.5 * F^(2k) = 0.5 * (1 - F^(2k))
    # For small decay: decay ≈ 0.5 * 2k * (1-F) = k * (1-F)
    # So slope of decay vs k ≈ (1-F)

    valid = p0_vals > 0.5
    if valid.sum() >= 2:
        dev = p0_vals[valid] - 0.5  # = 0.5 * F^(2k)
        log_dev = np.log(dev)
        k_valid = k_vals[valid]
        slope, intercept = np.polyfit(k_valid, log_dev, 1)
        # slope = 2 * ln(F)
        F_per_step = np.exp(slope / 2)
        print(f"  F_per_step = {F_per_step:.6f}")
        print(f"  Error per step = {(1-F_per_step)*100:.4f}%")
        print(f"  At k=1000, expected P(0) = {0.5 + 0.5 * F_per_step**(2*1000):.4f}")

    # Also try linear fit for small k
    print("\n=== Linear fit (small k) ===")
    small_k = [r for r in results if r['depth'] <= 201]
    if len(small_k) >= 2:
        sk = np.array([r['depth'] for r in small_k])
        sp = np.array([r['p0'] for r in small_k])
        sv = sp > 0.5
        if sv.sum() >= 2:
            sd = sp[sv] - 0.5
            slog = np.log(sd)
            skv = sk[sv]
            sl, si = np.polyfit(skv, slog, 1)
            Fsmall = np.exp(sl / 2)
            print(f"  F_per_step (k<=201) = {Fsmall:.6f}")
            print(f"  Error per step = {(1-Fsmall)*100:.4f}%")

    outpath = os.path.join(os.path.dirname(__file__),
                           f"ibm_f_gate_deep_odd_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(outpath, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved: {outpath}")


if __name__ == '__main__':
    main()
