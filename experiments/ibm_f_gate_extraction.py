"""
Experiment 3: Extract F_gate from IBM ibm_marrakesh (superconducting).

Circuit: H(q0) → [CZ(q0,q1) → H(q0) → H(q1)] × k → Measure
At even k: qubit0 should be |0⟩ (CZ toggles back)
At odd k: qubit0 should be |1⟩

Decay from ideal → F_gate^k

Target: 2-qubit CZ chain, depths k=1..20, on best qubit pairs.
"""

import json
import os
import numpy as np
from datetime import datetime
from qiskit import QuantumCircuit
from qiskit.compiler import transpile
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler


def build_circuit(k, q0, q1):
    """Build k-step CZ toggle circuit.

    Step = CZ(q0,q1) + H(q0) + H(q1)
    After even steps: q0 should be |0⟩
    After odd steps: q0 should be |1⟩
    """
    qc = QuantumCircuit(max(q0, q1) + 1, 1)
    qc.h(q0)

    for _ in range(k):
        qc.cz(q0, q1)
        qc.h(q0)
        qc.h(q1)

    qc.measure(q0, 0)
    return qc


def find_best_pairs(props, n_pairs=5):
    """Find the n_pairs best fidelity CZ pairs on the chip."""
    cz_pairs = []
    for gate in props.gates:
        if gate.gate == 'cz' and len(gate.qubits) == 2:
            q1, q2 = gate.qubits
            fid = 1 - gate.parameters[0].value
            cz_pairs.append((q1, q2, fid))

    cz_pairs.sort(key=lambda x: -x[2])
    # Deduplicate (q1,q2) and (q2,q1)
    seen = set()
    unique = []
    for q1, q2, fid in cz_pairs:
        key = (min(q1, q2), max(q1, q2))
        if key not in seen:
            seen.add(key)
            unique.append((q1, q2, fid))
    return unique[:n_pairs]


def main():
    print("=== Experiment 3: IBM F_gate Extraction ===")
    print(f"Time: {datetime.now().isoformat()}")
    print()

    service = QiskitRuntimeService(
        channel='ibm_quantum_platform',
        token='Pga6CcTge-z9ALVI-YV4XZa65Qs5HJvG8k_DVoGCITNN'
    )
    backend = service.backend('ibm_marrakesh')
    props = backend.properties()

    # Find best qubit pairs
    best_pairs = find_best_pairs(props, n_pairs=3)
    print("Best CZ pairs:")
    for q1, q2, fid in best_pairs:
        print(f"  CZ({q1},{q2}): {fid:.4f}")

    # Depths to test
    depths = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20]

    results = []
    sampler = Sampler(backend)
    sampler.options.default_shots = 4000

    for q1, q2, cz_fid in best_pairs:
        print(f"\n--- Qubit pair ({q1}, {q2}), CZ fidelity: {cz_fid:.4f} ---")

        for k in depths:
            qc = build_circuit(k, q1, q2)
            qc_t = transpile(qc, backend, optimization_level=0)
            try:
                job = sampler.run([qc_t])
                result = job.result()
                counts = result[0].data.c.get_counts()
                p0 = counts.get('0', 0) / sum(counts.values())
                results.append({
                    'qubits': (q1, q2),
                    'depth': k,
                    'cz_fid_cal': cz_fid,
                    'p0': p0,
                    'counts': counts,
                })
                ideal = 1.0 if k % 2 == 0 else 0.0
                print(f"  k={k:2d}: p0={p0:.4f} (ideal={ideal:.1f})")
            except Exception as e:
                print(f"  k={k:2d}: ERROR - {e}")
                results.append({
                    'qubits': (q1, q2),
                    'depth': k,
                    'cz_fid_cal': cz_fid,
                    'p0': None,
                    'error': str(e),
                })

    # Fit F_gate from decay
    print("\n=== F_gate Extraction ===")
    for q1, q2, cz_fid in best_pairs:
        pair_data = [r for r in results if r['qubits'] == (q1, q2) and r['p0'] is not None]
        if len(pair_data) < 3:
            print(f"  ({q1},{q2}): insufficient data")
            continue

        # At even k, ideal p0 = 1.0. Decay: p0 = 0.5 + 0.5 * F^(2k)
        # Each step = CZ + 2 single-qubit gates. Single qubit fid ~ 0.9999
        # p0_even ≈ 0.5 + 0.5 * (F_cz)^(2k)
        even_data = [(r['depth'], r['p0']) for r in pair_data if r['depth'] % 2 == 0]
        if len(even_data) >= 2:
            depths_even = np.array([d for d, p in even_data])
            p0_even = np.array([p for d, p in even_data])

            # Fit: p0 = 0.5 + 0.5 * a^k, where a = F_cz^2
            # ln(p0 - 0.5) = ln(0.5) + k * ln(a)
            valid = p0_even > 0.5
            if valid.sum() >= 2:
                log_decay = np.log(p0_even[valid] - 0.5)
                k_valid = depths_even[valid]
                slope, intercept = np.polyfit(k_valid, log_decay, 1)
                a_fit = np.exp(slope)
                F_cz_fit = np.sqrt(a_fit)
                print(f"  ({q1},{q2}): F_cz(fitted) = {F_cz_fit:.4f}, F_cz(calibration) = {cz_fid:.4f}")

    # Save
    outpath = os.path.join(os.path.dirname(__file__),
                           f"ibm_f_gate_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(outpath, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved: {outpath}")


if __name__ == '__main__':
    main()
