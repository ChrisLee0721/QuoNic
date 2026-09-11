"""
Tuna-17 overnight experiment: Complete superconducting platform characterization.
Measures physical parameters for the decision boundary framework.

Run this in Python 3.13 environment with qiskit-quantuminspire.
"""

import numpy as np
import json
import time
from datetime import datetime
from qiskit import QuantumCircuit
from qiskit.compiler import transpile
from qiskit_quantuminspire.qi_provider import QIProvider

OUTPUT_FILE = "tuna17_physical_params.json"

def run_circuit(backend, qc, shots=1000, max_retries=3):
    """Run circuit with retry logic."""
    for attempt in range(max_retries):
        try:
            qc_t = transpile(qc, backend, optimization_level=0, initial_layout=[0, 1])
            job = backend.run(qc_t, shots=shots)
            result = job.result()
            counts = result.get_counts()
            return counts
        except Exception as e:
            if '429' in str(e) or 'queue' in str(e).lower():
                wait = 30 * (attempt + 1)
                print(f"    Rate limited, waiting {wait}s...", flush=True)
                time.sleep(wait)
            else:
                print(f"    Error: {e}", flush=True)
                return None
    return None

def main():
    results = {
        'timestamp_start': datetime.now().isoformat(),
        'backend': 'Tuna-17',
        'readout_fidelity': None,
        'single_qubit_rb': None,
        'cz_fidelity': None,
        't1_t2': None,
        'depth_scan': {},
    }

    print(f"=== Tuna-17 Physical Parameter Measurement ===")
    print(f"Start: {datetime.now().isoformat()}")
    print(f"Output: {OUTPUT_FILE}\n")

    # Connect to backend
    provider = QIProvider()
    backend = provider.get_backend('Tuna-17')
    print(f"Backend: {backend.name} ({backend.num_qubits} qubits)")
    print(f"Operations: {backend.operation_names}\n")

    # === Part 1: Readout Fidelity ===
    print("=" * 60)
    print("PART 1: Readout Fidelity")
    print("=" * 60)

    # Prepare |0>, measure
    qc0 = QuantumCircuit(1, 1)
    qc0.measure(0, 0)
    print("  |0> state...", end=" ", flush=True)
    counts0 = run_circuit(backend, qc0, shots=4000)
    if counts0:
        p0_given_0 = counts0.get('0', 0) / sum(counts0.values())
        print(f"P(0|0) = {p0_given_0:.6f}")

    time.sleep(5)

    # Prepare |1>, measure
    qc1 = QuantumCircuit(1, 1)
    qc1.x(0)
    qc1.measure(0, 0)
    print("  |1> state...", end=" ", flush=True)
    counts1 = run_circuit(backend, qc1, shots=4000)
    if counts1:
        p1_given_1 = counts1.get('1', 0) / sum(counts1.values())
        print(f"P(1|1) = {p1_given_1:.6f}")

    if counts0 and counts1:
        readout_fid = (p0_given_0 + p1_given_1) / 2
        results['readout_fidelity'] = {
            'p0_given_0': p0_given_0,
            'p1_given_1': p1_given_1,
            'average': readout_fid,
        }
        print(f"  Readout fidelity = {readout_fid:.6f}")

    # Save
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=2)

    time.sleep(10)

    # === Part 2: Single-Qubit Gate Fidelity ===
    print("\n" + "=" * 60)
    print("PART 2: Single-Qubit Gate Fidelity (H^2n sequences)")
    print("=" * 60)

    single_qubit_results = {}
    for n in [1, 2, 5, 10, 20, 50]:
        qc = QuantumCircuit(1, 1)
        for _ in range(2 * n):
            qc.h(0)
        qc.measure(0, 0)

        print(f"  H^{2*n}...", end=" ", flush=True)
        counts = run_circuit(backend, qc, shots=4000)
        if counts:
            p0 = counts.get('0', 0) / sum(counts.values())
            single_qubit_results[f'H_{2*n}'] = p0
            print(f"P(0) = {p0:.4f}")
        else:
            print("FAILED")

        time.sleep(5)

    results['single_qubit_rb'] = single_qubit_results

    # Save
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=2)

    time.sleep(10)

    # === Part 3: CZ Gate Fidelity ===
    print("\n" + "=" * 60)
    print("PART 3: CZ Gate Test")
    print("=" * 60)

    # Bell state test: H(0) - CZ(0,1) - H(0)
    qc_bell = QuantumCircuit(2, 2)
    qc_bell.h(0)
    qc_bell.cz(0, 1)
    qc_bell.measure([0, 1], [0, 1])

    print("  Bell state (H-CZ)...", end=" ", flush=True)
    counts_bell = run_circuit(backend, qc_bell, shots=4000)
    if counts_bell:
        total = sum(counts_bell.values())
        p00 = counts_bell.get('00', 0) / total
        p11 = counts_bell.get('11', 0) / total
        print(f"P(00)={p00:.4f}, P(11)={p11:.4f}")
        results['cz_fidelity'] = {
            'p00': p00,
            'p11': p11,
            'bell_sum': p00 + p11,
        }

    # Save
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=2)

    time.sleep(10)

    # === Part 4: T1 Measurement ===
    print("\n" + "=" * 60)
    print("PART 4: T1 Relaxation")
    print("=" * 60)

    t1_results = {}
    delays_us = [0, 10, 20, 50, 100, 200, 500]

    for delay in delays_us:
        qc = QuantumCircuit(1, 1)
        qc.x(0)  # Prepare |1>
        if delay > 0:
            qc.delay(delay, 0, unit='us')
        qc.measure(0, 0)

        print(f"  Delay {delay}μs...", end=" ", flush=True)
        counts = run_circuit(backend, qc, shots=2000)
        if counts:
            p1 = counts.get('1', 0) / sum(counts.values())
            t1_results[f'{delay}us'] = p1
            print(f"P(1) = {p1:.4f}")
        else:
            print("FAILED")

        time.sleep(5)

    results['t1_t2'] = t1_results

    # Save
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=2)

    time.sleep(10)

    # === Part 5: F_gate Depth Scan ===
    print("\n" + "=" * 60)
    print("PART 5: F_gate Depth Scan (CZ+H circuit)")
    print("=" * 60)

    depths = [1, 2, 3, 5, 10, 20, 50, 100]
    R = 3

    for k in depths:
        results['depth_scan'][k] = []

        for r in range(R):
            print(f"  k={k}, run {r+1}/{R}...", end=" ", flush=True)

            qc = QuantumCircuit(2, 1)
            qc.h(0)
            for _ in range(k):
                qc.cz(0, 1)
                qc.h(0)
                qc.h(1)
            qc.measure(0, 0)

            counts = run_circuit(backend, qc, shots=2000)
            if counts:
                p0 = counts.get('0', 0) / sum(counts.values())
                results['depth_scan'][k].append({
                    'p0': p0,
                    'counts': counts,
                    'timestamp': time.time(),
                })
                print(f"P(0)={p0:.4f}")
            else:
                results['depth_scan'][k].append({'error': 'failed'})
                print("FAILED")

            time.sleep(5)

        # Save after each depth
        with open(OUTPUT_FILE, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"  k={k} complete. Saved.\n")

    # === Summary ===
    results['timestamp_end'] = datetime.now().isoformat()

    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)
    print(f"End: {datetime.now().isoformat()}")
    print(f"Results saved to: {OUTPUT_FILE}")

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == '__main__':
    main()
